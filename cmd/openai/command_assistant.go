package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/ebitengine/oto/v3"
	"github.com/hajimehoshi/go-mp3"
	"github.com/openai/openai-go"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

func uploadLocalFileDirectoryForAssistants(ctx context.Context, prefix, fullPath string) ([]*openai.FileObject, error) {
	uploadResponses := []*openai.FileObject{}

	// Get the full path.
	fullPath, err := filepath.Abs(fullPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path: %w", err)
	}

	// Check if the file exists, and is a directory or not.
	fi, err := os.Stat(fullPath)
	if err != nil {
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	if !fi.IsDir() {
		return nil, fmt.Errorf("path is not a directory: %s", fullPath)
	}

	// Recursively walk the directory.
	err = filepath.Walk(fullPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("failed to walk path %q: %w", path, err)
		}

		// Skip directories (we can't upload directories).
		if info.IsDir() {
			return nil
		}

		// Skip hidden files.
		if strings.HasPrefix(info.Name(), ".") {
			return nil
		}

		// Remove the root directory from the path, and add the prefix if it exists.
		name := strings.TrimPrefix(strings.TrimPrefix(path, fullPath), "/")

		if prefix != "" {
			name = filepath.Join(prefix, name)
		}

		// Upload the file.
		resp, err := uploadLocalFileForAssistants(ctx, name, path)
		if err != nil {
			return fmt.Errorf("failed to upload file %q: %w", path, err)
		}

		uploadResponses = append(uploadResponses, resp)

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to walk directory %q: %w", fullPath, err)
	}

	return uploadResponses, nil
}

var assistantFileDirectoryUploadCommand = &cobra.Command{
	Use:   "directory",
	Short: "Upload a directory of files for an assistant",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		path := args[0]

		prefix := cmd.Flag("prefix").Value.String()

		uploadResps, err := uploadLocalFileDirectoryForAssistants(ctx, prefix, path)
		if err != nil {
			return fmt.Errorf("failed to upload directory %q: %w", path, err)
		}

		for _, resp := range uploadResps {
			fmt.Println(resp.ID)
		}

		if cmd.Flag("assistants").Changed {
			assistants, err := cmd.Flags().GetStringSlice("assistants")
			if err != nil {
				return fmt.Errorf("failed to get assistants flag: %w", err)
			}

			for _, assistantID := range assistants {
				_, err := client.Beta.Assistants.Update(ctx, assistantID, openai.BetaAssistantUpdateParams{
					ToolResources: openai.F(openai.BetaAssistantUpdateParamsToolResources{
						FileSearch: openai.F(openai.BetaAssistantUpdateParamsToolResourcesFileSearch{
							VectorStoreIDs: openai.F(
								func() []string {
									var vectorStoreIDs []string
									for _, resp := range uploadResps {
										vectorStoreIDs = append(vectorStoreIDs, resp.ID)
									}
									return vectorStoreIDs
								}(),
							),
						}),
					}),
				})
				if err != nil {
					return fmt.Errorf("failed to update assistant %q: %w", assistantID, err)
				}
			}
		}

		return nil
	},
}

func init() {
	assistantFileDirectoryUploadCommand.Flags().String("prefix", "", "the prefix to add to all file names (in front of root directory name)")
	assistantFileDirectoryUploadCommand.Flags().StringSliceP("assistants", "a", nil, "the assistant IDs to update to use the uploaded file")
}

func uploadLocalFileForAssistants(ctx context.Context, name, path string) (*openai.FileObject, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %q: %w", path, err)
	}
	defer fh.Close()

	// Determine the content type mime type of the file.
	b := make([]byte, 512)
	_, err = fh.Read(b)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %q: %w", path, err)
	}
	fh.Seek(0, io.SeekStart)
	contentType := http.DetectContentType(b)

	uploadResp, err := client.Files.New(ctx, openai.FileNewParams{
		File:    openai.FileParam(fh, name, contentType),
		Purpose: openai.F(openai.FilePurposeAssistants),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to upload file %q: %w", path, err)
	}

	return uploadResp, nil
}

func uploadHTTPFileForAssistants(ctx context.Context, httpClient *http.Client, url string) (*openai.FileObject, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	uploadResp, err := client.Files.New(ctx, openai.FileNewParams{
		File:    openai.F[io.Reader](resp.Body),
		Purpose: openai.F(openai.FilePurposeAssistants),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to upload file: %w", err)
	}

	return uploadResp, nil
}

var assistantFileUploadCommand = &cobra.Command{
	Use:   "upload",
	Short: "Upload a file for an assistant",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		path := args[0]

		name := cmd.Flag("name").Value.String()
		if name == "" {
			name = filepath.Base(path)
		}

		if strings.HasPrefix(path, "https://") || strings.HasPrefix(path, "http://") {
			uploadResp, err := uploadHTTPFileForAssistants(ctx, http.DefaultClient, path)
			if err != nil {
				return fmt.Errorf("failed to upload file: %w", err)
			}

			fmt.Println(uploadResp.ID)
			return nil
		}

		uploadResp, err := uploadLocalFileForAssistants(ctx, name, path)
		if err != nil {
			return fmt.Errorf("failed to upload file: %w", err)
		}

		fmt.Println(uploadResp.ID)

		return nil
	},
}

func init() {
	assistantFileUploadCommand.Flags().String("name", "", "the name of the file")

	assistantFileUploadCommand.AddCommand(
		assistantFileDirectoryUploadCommand,
	)
}

var assistantFileDeleteAllCommand = &cobra.Command{
	Use:   "all",
	Short: "Delete all assistant files",
	Args:  cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		listResp, err := client.Files.List(ctx, openai.FileListParams{
			Limit:   openai.Int(10_000),
			Purpose: openai.String(string(openai.FileObjectPurposeAssistants)),
		})
		if err != nil {
			return fmt.Errorf("failed to list files: %w", err)
		}

		deleteFile := func(fileID string) error {
			resp, err := client.Files.Delete(ctx, fileID)
			if err != nil {
				return fmt.Errorf("failed to delete file %s: %w", fileID, err)
			}

			if resp.Deleted {
				fmt.Printf("file %s deleted\n", fileID)
			}

			return nil
		}

		for _, file := range listResp.Data {
			err := deleteFile(file.ID)
			if err != nil {
				return fmt.Errorf("failed to delete file %s: %w", file.ID, err)
			}
		}

		for {
			listResp, err = listResp.GetNextPage()
			if err != nil {
				return fmt.Errorf("failed to get next page of files: %w", err)
			}
			if listResp == nil {
				break
			}

			for _, file := range listResp.Data {
				err := deleteFile(file.ID)
				if err != nil {
					return fmt.Errorf("failed to delete file %s: %w", file.ID, err)
				}
			}
		}

		return nil
	},
}

var assistantFileDeleteCommand = &cobra.Command{
	Use:   "delete <file-id>",
	Short: "Delete an assistant file", // Note: this could be used to delete any file, not just assistant files.
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		fileID := args[0]

		resp, err := client.Files.Delete(ctx, fileID)
		if err != nil {
			return fmt.Errorf("failed to delete file: %w", err)
		}

		if resp.Deleted {
			fmt.Printf("deleted file: %s\n", resp.ID)
			return nil
		}

		return fmt.Errorf("failed to delete file: %s", resp.ID)
	},
}

func init() {
	assistantFileDeleteCommand.AddCommand(assistantFileDeleteAllCommand)
}

var assistantFileInfoCommand = &cobra.Command{
	Use:   "info <file-id>",
	Short: "Get information about an assistant file",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		fileID := args[0]

		resp, err := client.Files.Get(ctx, fileID)
		if err != nil {
			return fmt.Errorf("failed to get file info: %w", err)
		}

		b, err := json.Marshal(resp)
		if err != nil {
			return fmt.Errorf("failed to marshal file info: %w", err)
		}

		fmt.Println(string(b))

		return nil
	},
}

var assistantFileListCommand = &cobra.Command{
	Use:   "list",
	Short: "List assistant files",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		// TODO handle pagination

		listResp, err := client.Files.List(ctx, openai.FileListParams{
			Limit:   openai.Int(10_000),
			Purpose: openai.String(string(openai.FileObjectPurposeAssistants)),
		})
		if err != nil {
			return fmt.Errorf("failed to list files: %w", err)
		}

		for _, file := range listResp.Data {
			fmt.Println(file.ID, file.Filename, file.Bytes)
		}

		return nil
	},
}

var assistantFileCommand = &cobra.Command{
	Use:   "file",
	Short: "Manage assistant files",
}

func init() {
	assistantFileCommand.AddCommand(
		assistantFileListCommand,
		assistantFileUploadCommand,
		assistantFileInfoCommand,
		assistantFileDeleteCommand,
		// NOTE: Not allowed to download files of purpose: assistants
	)
}

var assistantInfoCommand = &cobra.Command{
	Use:   "info <assistant-id>",
	Short: "Get information about an assistant",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		readResp, err := client.Beta.Assistants.Get(ctx, args[0])
		if err != nil {
			return fmt.Errorf("failed to read assistant: %w", err)
		}

		// Print JSON.
		b, err := json.Marshal(readResp)
		if err != nil {
			return fmt.Errorf("failed to marshal assistant info: %w", err)
		}

		fmt.Println(string(b))

		return nil
	},
}

var assistantDeleteCommand = &cobra.Command{
	Use:   "delete <assistant-id>",
	Short: "Delete an assistant",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		resp, err := client.Beta.Assistants.Delete(ctx, args[0])
		if err != nil {
			return fmt.Errorf("failed to delete assistant: %w", err)
		}

		if !resp.Deleted {
			return fmt.Errorf("failed to delete assistant: %s", resp.ID)
		}

		return nil
	},
}

var assistantListCommand = &cobra.Command{
	Use:   "list",
	Short: "List assistants",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		// TODO handle pagination

		listResp, err := client.Beta.Assistants.List(ctx, openai.BetaAssistantListParams{
			Limit: openai.Int(100),
		})
		if err != nil {
			return fmt.Errorf("failed to list assistants: %w", err)
		}

		for _, assistant := range listResp.Data {
			fmt.Println(assistant.ID, assistant.Name, assistant.Description)
		}

		return nil
	},
}

var assistantCreateCommand = &cobra.Command{
	Use:   "create",
	Short: "Create an assistant",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		var (
			model        = cmd.Flag("model").Value.String()
			instructions = cmd.Flag("instructions").Value.String()
			name         = cmd.Flag("name").Value.String()
			description  = cmd.Flag("description").Value.String()

			tools []openai.AssistantToolUnionParam
		)

		// Handle the tools.
		{
			if cmd.Flag("code-interpreter").Value.String() == "true" {
				tools = append(tools, openai.AssistantToolParam{
					Type: openai.F(openai.AssistantToolTypeCodeInterpreter),
				})
			}

			if cmd.Flag("retrieval").Value.String() == "true" {
				tools = append(tools, openai.FileSearchToolParam{
					Type: openai.F(openai.FileSearchToolTypeFileSearch),
				})
			}
		}

		assistant, err := client.Beta.Assistants.New(ctx, openai.BetaAssistantNewParams{
			Model:        openai.F(model),
			Instructions: openai.F(instructions),
			Name:         openai.F(name),
			Description:  openai.F(description),
			Tools:        openai.F(tools),
		})
		if err != nil {
			return fmt.Errorf("failed to create assistant: %w", err)
		}

		fmt.Println(assistant.ID)

		return nil
	},
}

func init() {
	assistantCreateCommand.Flags().String("model", openai.ChatModelGPT4o, "the model to use for the assistant")
	assistantCreateCommand.Flags().String("instructions", "", "the instructions for the assistant")
	assistantCreateCommand.Flags().String("name", "", "the name of the assistant")
	assistantCreateCommand.Flags().String("description", "", "the description of the assistant")
	assistantCreateCommand.Flags().Bool("code-interpreter", true, "enable the code interpreter tool")
	assistantCreateCommand.Flags().Bool("retrieval", true, "enable the retrieval tool")
	assistantCreateCommand.Flags().StringSliceP("files", "f", nil, "the file IDs to use for the assistant")
}

var assistantChatThreadCommand = &cobra.Command{
	Use: "thread",
}

var assistantChatThreadDeleteCommand = &cobra.Command{
	Use:   "delete <thread-id>",
	Short: "Delete an assistant chat thread",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		resp, err := client.Beta.Threads.Delete(ctx, args[0])
		if err != nil {
			return fmt.Errorf("failed to delete assistant chat thread: %w", err)
		}

		if !resp.Deleted {
			return fmt.Errorf("failed to delete assistant chat thread: %s", resp.ID)
		}

		return nil
	},
}

var assistantChatThreadInfoCommand = &cobra.Command{
	Use:   "info <thread-id>",
	Short: "Get information about an assistant chat thread",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		thread, err := client.Beta.Threads.Get(ctx, args[0])
		if err != nil {
			return fmt.Errorf("failed to get assistant chat thread: %w", err)
		}

		b, err := json.Marshal(thread)
		if err != nil {
			return fmt.Errorf("failed to marshal thread: %w", err)
		}
		b = append(b, '\n')

		_, err = cmd.OutOrStdout().Write(b)
		if err != nil {
			return fmt.Errorf("failed to write thread: %w", err)
		}

		return nil
	},
}

var assistantChatCommand = &cobra.Command{
	Use:   "chat [assistant-id]",
	Short: "Start an interactive assistant chat session",
	Args:  cobra.MaximumNArgs(1),
	Long: `Start an interactive ephemeral chat session with the OpenAI API.

This is similar to the basic "openai assistant" command, but enables
interaction with existing assistants.

This is a REPL-like command-line program that allows you use the new assistant API.

The assistant API is a powerful new feature that allows you to create a custom chatbot that can
answer questions, perform tasks, and even generate code.
`,
	RunE: func(cmd *cobra.Command, args []string) error {
		if len(args) == 1 {
			assistantID := args[0]

			var threadID string
			if cmd.Flag("thread").Value.String() != "" {
				threadID = cmd.Flag("thread").Value.String()
			}

			return startAssistantChat(client, model, assistantID, threadID)
		}

		return startAssistantChat(client, model, "", "")
	},
}

func init() {
	assistantChatCommand.Flags().String("model", model, "the model to use for the assistant")
	assistantChatCommand.Flags().StringP("thread", "t", "", "the thread ID to use for the assistant")

	assistantChatCommand.AddCommand(assistantChatThreadCommand)
	assistantChatThreadCommand.AddCommand(assistantChatThreadDeleteCommand)
	assistantChatThreadCommand.AddCommand(assistantChatThreadInfoCommand)
}

var assistantCommand = &cobra.Command{
	Use:   "assistant",
	Short: "Start an interactive assistant chat session",
	Long: `Interact with the OpenAI API using the assistant API.
	
This can be used to both quickly create a temporary assistant and  manage long-lived assistants. 
	`,
	Example: `  $ openai assistant      # create a temporary assistant and start chatting
  $ openai assistant chat # same as above
  $ openai assistant create --name "Example" --model "gpt-4-turbo-preview" --description "..." --instructions "..." --code-interpreter --retrieval
  $ openai assistant list
  $ openai assistant info <assistant-id>
  $ openai assistant chat <assistant-id>
  $ openai assistant delete <assistant-id>
	`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return startAssistantChat(client, model, "", "")
	},
}

func init() {
	assistantCommand.AddCommand(
		assistantChatCommand,
		assistantCreateCommand,
		assistantInfoCommand,
		assistantDeleteCommand,
		assistantListCommand,
		assistantFileCommand,
	)

	rootCmd.AddCommand(
		assistantCommand,
	)
}

// startAssistantChat starts an interactive chat session with the OpenAI API, this is a REPL-like
// command-line program that allows you use the new assistant API (in beta).
func startAssistantChat(client *openai.Client, model, assistantID, threadID string) error {
	ctx := context.Background()

	var speak bool

	// I don't totally understand why this configuration works, but it does.
	op := &oto.NewContextOptions{
		// Usually 44100 or 48000. Other values might cause distortions in Oto
		SampleRate: 48000 / 2,

		// Use default buffer size.
		BufferSize: 0,

		// Stereo sound. Mono is also ok.
		ChannelCount: 2,

		// Format of the source. go-mp3's format is signed 16bit integers.
		Format: oto.FormatSignedInt16LE,
	}

	// Remember that you should **not** create more than one context
	otoCtx, readyChan, err := oto.NewContext(op)
	if err != nil {
		return fmt.Errorf("failed to create oto context: %w", err)
	}

	tools := []openai.AssistantToolUnionParam{
		openai.AssistantToolParam{
			Type: openai.F(openai.AssistantToolTypeCodeInterpreter),
		},
		openai.FileSearchToolParam{
			Type: openai.F(openai.FileSearchToolTypeFileSearch),
		},
		// openai.FunctionToolParam{
		// 	Type: openai.F(openai.FunctionToolTypeFunction),
		// },
	}

	var ephemeralAssistant bool
	if assistantID == "" {
		assistant, err := client.Beta.Assistants.New(ctx, openai.BetaAssistantNewParams{
			Model:        openai.F(model),
			Instructions: openai.F("You are a helpful assistant for all kinds of tasks. Answer as concisely as possible."),
			Name:         openai.F("openai-cli-assistant"),
			Description:  openai.F("A helpful assistant for all kinds of tasks."),
			Tools:        openai.F(tools),
		})
		if err != nil {
			return fmt.Errorf("failed to create assistant: %w", err)
		}

		defer client.Beta.Assistants.Delete(ctx, assistant.ID)

		assistantID = assistant.ID

		ephemeralAssistant = true
	}

	if threadID == "" {
		thread, err := client.Beta.Threads.New(ctx, openai.BetaThreadNewParams{})
		if err != nil {
			return fmt.Errorf("failed to create thread: %w", err)
		}
		if ephemeralAssistant {
			defer client.Beta.Threads.Delete(ctx, thread.ID)
		}
		threadID = thread.ID
	} else {
		_, err = client.Beta.Threads.Get(ctx, threadID)
		if err != nil {
			return fmt.Errorf("failed to get thread: %w", err)
		}
	}

	// Set the terminal to raw mode.
	fd := int(os.Stdin.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return fmt.Errorf("failed to set terminal to raw mode: %w", err)
	}
	defer term.Restore(fd, oldState)

	termWidth, termHeight, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		return fmt.Errorf("failed to get terminal size: %w", err)
	}

	termReadWriter := struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stdout}

	t := term.NewTerminal(termReadWriter, "") // Will set the prompt later.

	t.SetSize(termWidth, termHeight)

	// Use buffered output so we can write to the terminal without
	// having to wait for a newline, and so we can clear the screen
	// and move the cursor around without having to worry about
	// overwriting the prompt.
	bt := bufio.NewWriter(t)

	clearScreen := func() {
		// Clear the screen.
		bt.WriteString("\033[2J")

		// Move to the top left.
		bt.WriteString("\033[H")

		// Flush the buffer to the terminal.
		bt.Flush()
	}

	clearScreen()

	// Autocomplete for commands.
	t.AutoCompleteCallback = func(line string, pos int, key rune) (newLine string, newPos int, ok bool) {
		// If the user presses tab, then autocomplete the command.
		if key == '\t' {
			for _, cmd := range []string{"exit", "clear", "ls", "copy", "upload", "system:", "<clipboard>"} {
				if strings.HasPrefix(cmd, line) {
					// Autocomplete the command.
					// t.Write([]byte(cmd[len(line):]))

					// Return the new line and position, which must come after the
					// command.
					return cmd, len(cmd), true
				}
			}
		}

		// If the user hit backspace on the example system message, then we'll
		// just delete the whole line, re-add the "system:" prefix, and return
		// the new line and position.

		// Otherwise, we'll just return the line.
		return line, pos, false
	}

	// Print welcome message.
	bt.WriteString(styleBold.Render("Welcome to the OpenAI API CLI assistant mode!\n\n"))

	if ephemeralAssistant {
		// Move to left edge.
		bt.WriteString("\033[0G")

		// Print warning message.
		bt.WriteString(styleWarning.Render("WARNING") + styleFaint.Render(": Messages and files disappear after exiting.\n\n"))
		bt.Flush()
	} else {
		// Move to left edge.
		bt.WriteString("\033[0G")

		// Print assistant ID and thread ID.

		bt.WriteString(fmt.Sprintf("%s Assistant %s chat thread %s session started.\n\n", styleSuccess.Render("▪"), styleBold.Render(assistantID), styleBold.Render(threadID)))
		bt.Flush()
	}

	if !ephemeralAssistant {
		defer func() {
			bt.WriteString("\033[0G")
			bt.WriteString(fmt.Sprintf("\n%s Assistant %s chat thread %s session ended.\n\n", styleInfo.Render("▪"), styleBold.Render(assistantID), styleBold.Render(threadID)))
			bt.Flush()
		}()
	}

	for {
		// Move to left edge.
		bt.WriteString("\033[0G")

		// Print the prompt.
		bt.WriteString("‣ ")

		// Flush the buffer to the terminal.
		bt.Flush()

		// Read up to line from STDIN.
		input, err := t.ReadLine()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("failed to read line: %w", err)
		}

		switch strings.TrimSpace(input) {
		case "exit":
			return nil
		case "speak!":
			speak = true

			bt.WriteString(styleFaint.Render("speaking mode enabled\n"))

			// It might take a bit for the hardware audio devices to be ready, so we wait on the channel.
			<-readyChan

			continue
		case "speak?":
			if speak {
				bt.WriteString(styleFaint.Render("speaking mode enabled\n"))
			} else {
				bt.WriteString(styleFaint.Render("speaking mode disabled\n"))
			}

			continue
		case "quiet!":
			speak = false

			bt.WriteString(styleFaint.Render("speaking mode disabled\n"))

			continue
		case "clear":
			// Clear the screen.
			clearScreen()
			continue
		case "ls":
			listFilesResp, err := client.Files.List(ctx, openai.FileListParams{
				Limit: openai.Int(10_000),
			})
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to list files: %s\n", err))
				continue
			}

			sort.SliceStable(listFilesResp.Data, func(i, j int) bool {
				return listFilesResp.Data[i].CreatedAt > listFilesResp.Data[j].CreatedAt
			})

			if len(listFilesResp.Data) == 0 {
				bt.WriteString("no files\n")
				continue
			}

			for _, f := range listFilesResp.Data {
				if f.Purpose != "assistants" {
					continue
				}
				bt.WriteString(fmt.Sprintf("%s: %s (%d)\n", f.ID, f.Filename, f.Bytes))
			}

			continue
		case "copy":
			// Copy last message to clipboard.
			listResp, err := client.Beta.Threads.Messages.List(ctx, threadID, openai.BetaThreadMessageListParams{
				Limit: openai.Int(100),
			})
			if err != nil {
				return fmt.Errorf("failed to list messages: %w", err)
			}

			lastMsg := listResp.Data[0].Content[0].Text.Value

			// Write the last message to the clipboard.
			err = writeClipboard(lastMsg)
			if err != nil {
				return fmt.Errorf("failed to write to clipboard: %w", err)
			}
		}

		// Check if the message has any <clipboard> tags.
		if strings.Contains(input, "<clipboard>") {
			// Get the clipboard contents.
			str, err := readClipboard()
			if err != nil {
				return fmt.Errorf("failed to read clipboard: %w", err)
			}

			// Replace the <clipboard> tag with the clipboard contents.
			input = strings.Replace(input, "<clipboard>", str, -1)
		}

		if fields := strings.Fields(input); len(fields) > 0 && fields[0] == "upload" {
			if len(fields) != 2 {
				bt.WriteString("usage: upload <file>\n")
				continue
			}

			// Check if we're uploading content from a URL.
			if strings.HasPrefix(fields[1], "https://") || strings.HasPrefix(fields[1], "http://") {
				// Download the file from the URL to upload it.
				req, err := http.NewRequestWithContext(ctx, http.MethodGet, fields[1], nil)
				if err != nil {
					bt.WriteString(fmt.Sprintf("failed to create request: %s\n", err))
					continue
				}

				resp, err := http.DefaultClient.Do(req)
				if err != nil {
					bt.WriteString(fmt.Sprintf("failed to download file: %s\n", err))
					continue
				}

				uploadResp, err := client.Files.New(ctx, openai.FileNewParams{
					File:    openai.FileParam(resp.Body, fields[1], resp.Header.Get("Content-Type")),
					Purpose: openai.F(openai.FilePurposeAssistants),
				})

				resp.Body.Close()

				if err != nil {
					bt.WriteString(fmt.Sprintf("failed to upload file: %s\n", err))
					continue
				}

				if ephemeralAssistant {
					defer client.Files.Delete(ctx, uploadResp.ID)
				}

				bt.WriteString(fmt.Sprintf("uploaded URL content: %s\n", uploadResp.ID))

				continue
			}

			// Check if the file exists, and is a directory or not.
			fi, err := os.Stat(fields[1])
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to stat file: %s\n", err))
				continue
			}

			if fi.IsDir() {
				// TODO: upload all files in directory
				bt.WriteString("uploading directories is not supported yet\n")
				continue
			}

			fh, err := os.Open(fields[1])
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to open file: %s\n", err))
				continue
			}

			// Detect content type of the file by reading the first 512 bytes.
			buf := make([]byte, 512)
			_, err = fh.Read(buf)
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to read file: %s\n", err))
				continue
			}

			// Reset the file pointer to the beginning of the file.
			_, err = fh.Seek(0, io.SeekStart)
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to seek file: %s\n", err))
				continue
			}

			uploadResp, err := client.Files.New(ctx, openai.FileNewParams{
				File:    openai.FileParam(fh, fields[1], http.DetectContentType(buf)),
				Purpose: openai.F(openai.FilePurposeAssistants),
			})

			fh.Close()

			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to upload file: %s\n", err))
				continue
			}

			if ephemeralAssistant {
				defer client.Files.Delete(ctx, uploadResp.ID)
			}

			bt.WriteString(fmt.Sprintf("uploaded file: %s\n", uploadResp.ID))

			continue
		}

		// Check if the user is setting a "system:" message, which we use
		// as the instructions for the assistant.
		if strings.HasPrefix(input, "system:") {
			// Update the assistant instructions.
			_, err := client.Beta.Assistants.Update(ctx, assistantID, openai.BetaAssistantUpdateParams{
				Instructions: openai.F(strings.TrimPrefix(input, "system:")),
			})
			if err != nil {
				return fmt.Errorf("failed to update assistant: %w", err)
			}

			// Continue to the next message.
			continue
		}

		_, err = client.Beta.Threads.Messages.New(ctx, threadID, openai.BetaThreadMessageNewParams{
			Role: openai.F(openai.BetaThreadMessageNewParamsRoleUser),
			Content: openai.F([]openai.MessageContentPartParamUnion{
				openai.TextContentBlockParam{
					Type: openai.F(openai.TextContentBlockParamTypeText),
					Text: openai.F(input),
				},
			}),
		})
		if err != nil {
			return fmt.Errorf("failed to create message: %w", err)
		}

		runResp, err := client.Beta.Threads.Runs.New(ctx, threadID, openai.BetaThreadRunNewParams{
			AssistantID: openai.F(assistantID),
		})
		if err != nil {
			return fmt.Errorf("failed to create run: %w", err)
		}

		run, err := client.Beta.Threads.Runs.PollStatus(ctx, threadID, runResp.ID, int((700 * time.Millisecond).Milliseconds()))
		if err != nil {
			return fmt.Errorf("failed to wait for run: %w", err)
		}
		if run.Status == "failed" {
			return fmt.Errorf("run failed: %s: %v: %s", run.ID, run.LastError.Code, run.LastError.Message)
		}

		listResp, err := client.Beta.Threads.Messages.List(ctx, threadID, openai.BetaThreadMessageListParams{
			Limit: openai.Int(1),
		})
		if err != nil {
			return fmt.Errorf("failed to last messages: %w", err)
		}

		nextMsg := listResp.Data[0].Content[0].Text.Value

		// render 60% of terminal width
		nextMsgMd, err := renderMarkdown(nextMsg, int(float64(termWidth)*0.6))
		if err != nil {
			return fmt.Errorf("failed to render markdown: %w", err)
		}

		bt.WriteString(nextMsgMd)

		if speak {
			bt.Flush()

			// Check the message is less than or equal to 4096 characters.
			if len(nextMsg) > 4096 {
				nextMsg = nextMsg[:4096]
			}

			resp, err := client.Audio.Speech.New(ctx, openai.AudioSpeechNewParams{
				Model:          openai.F(openai.SpeechModelTTS1HD),
				Voice:          openai.F(openai.AudioSpeechNewParamsVoiceFable),
				Input:          openai.F(nextMsg),
				ResponseFormat: openai.F(openai.AudioSpeechNewParamsResponseFormatMP3),
			})
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to create speech: %s\n", err))
				continue
			}
			defer resp.Body.Close()

			decodedMP3, err := mp3.NewDecoder(resp.Body)
			if err != nil {
				return fmt.Errorf("failed to decode mp3: %w", err)
			}

			// Create a new 'player' that will handle our sound. Paused by default.
			player := otoCtx.NewPlayer(decodedMP3)

			// Play starts playing the sound and returns without waiting for it (Play() is async).
			player.Play()

			// Print a message to the terminal to indicate that we're playing audio.
			bt.WriteString(styleFaint.Render("playing audio..."))

			bt.Flush()

			// We can wait for the sound to finish playing using something like this
			for player.IsPlaying() {
				time.Sleep(time.Millisecond)
			}

			// Clear the playing audio message.
			bt.WriteString("\033[2K")

			err = player.Close()
			if err != nil {
				return fmt.Errorf("failed to close player: %w", err)
			}
		}
	}
}
