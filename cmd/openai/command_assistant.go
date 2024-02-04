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
	"github.com/picatz/openai"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

func uploadLocalFileDirectoryForAssistants(ctx context.Context, prefix, fullPath string) ([]*openai.UploadFileResponse, error) {
	uploadResponses := []*openai.UploadFileResponse{}

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
				_, err := client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
					ID: assistantID,
					FileIDs: func() []string {
						var fileIDs []string
						for _, resp := range uploadResps {
							fileIDs = append(fileIDs, resp.ID)
						}
						return fileIDs
					}(),
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

func uploadLocalFileForAssistants(ctx context.Context, name, path string) (*openai.UploadFileResponse, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %q: %w", path, err)
	}
	defer fh.Close()

	if name == "" {
		name = filepath.Base(path)
	}

	uploadResp, err := client.UploadFile(ctx, &openai.UploadFileRequest{
		Name:    name,
		Purpose: "assistants",
		Body:    fh,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to upload file %q: %w", path, err)
	}

	return uploadResp, nil
}

func uploadHTTPFileForAssistants(ctx context.Context, httpClient *http.Client, name, url string) (*openai.UploadFileResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	uploadResp, err := client.UploadFile(ctx, &openai.UploadFileRequest{
		Name:    name,
		Purpose: "assistants",
		Body:    resp.Body,
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

		if strings.HasPrefix(path, "https://") || strings.HasPrefix(path, "http://") {
			uploadResp, err := uploadHTTPFileForAssistants(ctx, http.DefaultClient, name, path)
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

		if cmd.Flag("assistants").Changed {
			assistants, err := cmd.Flags().GetStringSlice("assistants")
			if err != nil {
				return fmt.Errorf("failed to get assistants flag: %w", err)
			}

			for _, assistantID := range assistants {
				_, err := client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
					ID: assistantID,
					FileIDs: []string{
						uploadResp.ID,
					},
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
	assistantFileUploadCommand.Flags().String("name", "", "the name of the file")
	assistantFileUploadCommand.Flags().StringSliceP("assistants", "a", nil, "the assistant IDs to update to use the uploaded file")

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

		listResp, err := client.ListFiles(ctx, &openai.ListFilesRequest{
			Purpose: "assistants",
		})
		if err != nil {
			return fmt.Errorf("failed to list files: %w", err)
		}

		for _, file := range listResp.Data {
			resp, err := client.DeleteFile(ctx, &openai.DeleteFileRequest{
				ID: file.ID,
			})
			if err != nil {
				return fmt.Errorf("failed to delete file: %w", err)
			}

			if resp.Deleted {
				fmt.Printf("deleted file: %s\n", resp.ID)
				continue
			}

			return fmt.Errorf("failed to delete file: %s", resp.ID)
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

		resp, err := client.DeleteFile(ctx, &openai.DeleteFileRequest{
			ID: args[0],
		})
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

		resp, err := client.GetFileInfo(ctx, &openai.GetFileInfoRequest{
			ID: args[0],
		})
		if err != nil {
			return fmt.Errorf("failed to read file: %w", err)
		}

		b, err := json.Marshal(resp)
		if err != nil {
			return fmt.Errorf("failed to marshal file info: %w", err)
		}

		fmt.Println(string(b))

		return nil
	},
}

var assistantFileReadCommand = &cobra.Command{
	Use:   "read <file-id>",
	Short: "Read an assistant file",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		resp, err := client.GetFileContent(ctx, &openai.GetFileContentRequest{
			ID: args[0],
		})
		if err != nil {
			return fmt.Errorf("failed to read file: %w", err)
		}
		defer resp.Body.Close()

		_, err = io.Copy(os.Stdout, resp.Body)
		if err != nil {
			return fmt.Errorf("failed to read file: %w", err)
		}

		return nil
	},
}

var assistantFileListCommand = &cobra.Command{
	Use:   "list",
	Short: "List assistant files",
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		// TODO handle pagination

		listResp, err := client.ListFiles(ctx, &openai.ListFilesRequest{
			Purpose: "assistants",
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
		assistantFileReadCommand,
		assistantFileDeleteCommand,
	)
}

var assistantInfoCommand = &cobra.Command{
	Use:   "info <assistant-id>",
	Short: "Get information about an assistant",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		readResp, err := client.GetAssistant(ctx, &openai.GetAssistantRequest{
			ID: args[0],
		})
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

var assistantUpdateCommand = &cobra.Command{
	Use:   "update <assistant-id>",
	Short: "Update an assistant",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		// Handle the tools.
		var tools []map[string]any
		{
			if cmd.Flag("code-interpreter").Value.String() == "true" {
				tools = append(tools, map[string]any{
					"type": "code_interpreter",
				})
			}

			if cmd.Flag("retrieval").Value.String() == "true" {
				tools = append(tools, map[string]any{
					"type": "retrieval",
				})
			}
		}

		// Handle the files.
		var fileIDs []string
		{
			filesFlag, err := cmd.Flags().GetStringSlice("files")
			if err != nil {
				return fmt.Errorf("failed to get files flag: %w", err)
			}

			fileIDs = filesFlag
		}

		_, err := client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
			ID:           args[0],
			Instructions: cmd.Flag("instructions").Value.String(),
			Name:         cmd.Flag("name").Value.String(),
			Description:  cmd.Flag("description").Value.String(),
			Tools:        tools,
			FileIDs:      fileIDs,
		})
		if err != nil {
			return fmt.Errorf("failed to update assistant: %w", err)
		}

		return nil
	},
}

func init() {
	assistantUpdateCommand.Flags().String("instructions", "", "the instructions for the assistant")
	assistantUpdateCommand.Flags().String("name", "", "the name of the assistant")
	assistantUpdateCommand.Flags().String("description", "", "the description of the assistant")
	assistantUpdateCommand.Flags().BoolP("code-interpreter", "c", true, "enable the code interpreter tool")
	assistantUpdateCommand.Flags().BoolP("retrieval", "r", true, "enable the retrieval tool")
	assistantUpdateCommand.Flags().StringSliceP("files", "f", nil, "the file IDs to use for the assistant")
}

var assistantDeleteCommand = &cobra.Command{
	Use:   "delete <assistant-id>",
	Short: "Delete an assistant",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx := cmd.Context()

		err := client.DeleteAssistant(ctx, &openai.DeleteAssistantRequest{
			ID: args[0],
		})
		if err != nil {
			return fmt.Errorf("failed to delete assistant: %w", err)
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

		listResp, err := client.ListAssistants(ctx, &openai.ListAssistantsRequest{})
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

			tools   []map[string]any
			fileIDs []string
		)

		// Handle the tools.
		{
			if cmd.Flag("code-interpreter").Value.String() == "true" {
				tools = append(tools, map[string]any{
					"type": "code_interpreter",
				})
			}

			if cmd.Flag("retrieval").Value.String() == "true" {
				tools = append(tools, map[string]any{
					"type": "retrieval",
				})
			}
		}

		// Handle the files.
		{
			filesFlag, err := cmd.Flags().GetStringSlice("files")
			if err != nil {
				return fmt.Errorf("failed to get files flag: %w", err)
			}

			fileIDs = filesFlag
		}

		assistant, err := client.CreateAssistant(ctx, &openai.CreateAssistantRequest{
			Model:        model,
			Instructions: instructions,
			Name:         name,
			Description:  description,
			Tools:        tools,
			FileIDs:      fileIDs,
		})
		if err != nil {
			return fmt.Errorf("failed to create assistant: %w", err)
		}

		fmt.Println(assistant.ID)

		return nil
	},
}

func init() {
	assistantCreateCommand.Flags().String("model", openai.ModelGPT4TurboPreview, "the model to use for the assistant")
	assistantCreateCommand.Flags().String("instructions", "", "the instructions for the assistant")
	assistantCreateCommand.Flags().String("name", "", "the name of the assistant")
	assistantCreateCommand.Flags().String("description", "", "the description of the assistant")
	assistantCreateCommand.Flags().Bool("code-interpreter", true, "enable the code interpreter tool")
	assistantCreateCommand.Flags().Bool("retrieval", true, "enable the retrieval tool")
	assistantCreateCommand.Flags().StringSliceP("files", "f", nil, "the file IDs to use for the assistant")
}

var assistantChatCommand = &cobra.Command{
	Use:   "chat [assistant-id]",
	Short: "Start an interactive assistant chat session",
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

			return startAssistantChat(client, model, assistantID)
		}

		return startAssistantChat(client, model, "")
	},
}

var assistantCommand = &cobra.Command{
	Use:   "assistant",
	Short: "Start an interactive assistant chat session",
	Long: `Interact with the OpenAI API using the assistant API.
	
This can be used to create a temporary assistant, or interact with an existing assistant.
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
		return startAssistantChat(client, model, "")
	},
}

func init() {
	assistantCommand.AddCommand(
		assistantChatCommand,
		assistantCreateCommand,
		assistantInfoCommand,
		assistantUpdateCommand,
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
func startAssistantChat(client *openai.Client, model, assistantID string) error {
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

	var ephemeralAssistant bool
	if assistantID == "" {
		assistant, err := client.CreateAssistant(ctx, &openai.CreateAssistantRequest{
			Model:        model,
			Instructions: "You are a helpful assistant for all kinds of tasks. Answer as concisely as possible.",
			Name:         "openai-cli-assistant",
			Description:  "A helpful assistant for all kinds of tasks.",
			Tools: []map[string]any{
				{
					"type": "code_interpreter",
				},
				{
					"type": "retrieval",
				},
				// {
				// 	"type": "function",
				//  ...
				// },
			},
		})
		if err != nil {
			return fmt.Errorf("failed to create assistant: %w", err)
		}

		defer client.DeleteAssistant(ctx, &openai.DeleteAssistantRequest{
			ID: assistant.ID,
		})

		assistantID = assistant.ID

		ephemeralAssistant = true
	}

	thread, err := client.CreateThread(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to create thread: %w", err)
	}

	defer client.DeleteThread(ctx, &openai.DeleteThreadRequest{
		ID: thread.ID,
	})

	// Set the terminal to raw mode.
	oldState, err := term.MakeRaw(0)
	if err != nil {
		return fmt.Errorf("failed to set terminal to raw mode: %w", err)
	}
	defer term.Restore(0, oldState)

	termWidth, termHeight, err := term.GetSize(0)
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

		// Print assistant ID.
		bt.WriteString("Assistant ID: " + styleBold.Render(assistantID) + "\n\n")
		bt.Flush()
	}

	for {
		// Move to left edge.
		bt.WriteString("\033[0G")

		// Print the prompt.
		bt.WriteString("> ")

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
			listFilesResp, err := client.ListFiles(ctx, &openai.ListFilesRequest{})
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
			listResp, err := client.ListMessages(ctx, &openai.ListMessagesRequest{
				ThreadID: thread.ID,
				Limit:    1,
			})
			if err != nil {
				return fmt.Errorf("failed to list messages: %w", err)
			}

			lastMsg := listResp.Data[0].Content[0].Text()

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

				uploadResp, err := client.UploadFile(ctx, &openai.UploadFileRequest{
					Name:    fields[1],
					Purpose: "assistants",
					Body:    resp.Body,
				})

				if ephemeralAssistant {
					defer client.DeleteFile(ctx, &openai.DeleteFileRequest{
						ID: uploadResp.ID,
					})
				}

				resp.Body.Close()

				if err != nil {
					bt.WriteString(fmt.Sprintf("failed to upload file: %s\n", err))
					continue
				}

				bt.WriteString(fmt.Sprintf("uploaded URL content: %s\n", uploadResp.ID))

				_, err = client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
					ID: assistantID,
					FileIDs: []string{
						uploadResp.ID,
					},
				})
				if err != nil {
					bt.WriteString(fmt.Sprintf("failed to update assistant: %s\n", err))
				}

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

			uploadResp, err := client.UploadFile(ctx, &openai.UploadFileRequest{
				Name:    fields[1],
				Purpose: "assistants",
				Body:    fh,
			})

			fh.Close()

			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to upload file: %s\n", err))
				continue
			}

			if ephemeralAssistant {
				defer client.DeleteFile(ctx, &openai.DeleteFileRequest{
					ID: uploadResp.ID,
				})
			}

			bt.WriteString(fmt.Sprintf("uploaded file: %s\n", uploadResp.ID))

			_, err = client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
				ID: assistantID,
				FileIDs: []string{
					uploadResp.ID,
				},
			})
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to update assistant: %s\n", err))
			}

			continue
		}

		// Check if the user is setting a "system:" message, which we use
		// as the instructions for the assistant.
		if strings.HasPrefix(input, "system:") {
			// Update the assistant instructions.
			_, err := client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
				ID:           assistantID,
				Instructions: strings.TrimPrefix(input, "system:"),
			})
			if err != nil {
				return fmt.Errorf("failed to update assistant: %w", err)
			}

			// Continue to the next message.
			continue
		}

		_, err = client.CreateMessage(ctx, &openai.CreateMessageRequest{
			ThreadID: thread.ID,
			Role:     openai.ChatRoleUser,
			Content:  input,
		})
		if err != nil {
			return fmt.Errorf("failed to create message: %w", err)
		}

		runResp, err := client.CreateRun(ctx, &openai.CreateRunRequest{
			ThreadID:    thread.ID,
			AssistantID: assistantID,
		})
		if err != nil {
			return fmt.Errorf("failed to create run: %w", err)
		}

		err = openai.WaitForRun(ctx, client, thread.ID, runResp.ID, 700*time.Millisecond)
		if err != nil {
			return fmt.Errorf("failed to wait for run: %w", err)
		}

		listResp, err := client.ListMessages(ctx, &openai.ListMessagesRequest{
			ThreadID: thread.ID,
			Limit:    1,
		})
		if err != nil {
			return fmt.Errorf("failed to list messages: %w", err)
		}

		nextMsg := listResp.Data[0].Content[0].Text()

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

			audioStream, err := client.CreateSpeech(ctx, &openai.CreateSpeechRequest{
				Model:          openai.ModelTTS1HD1106,
				Voice:          "fable",
				Input:          nextMsg,
				ResponseFormat: "mp3",
			})
			if err != nil {
				bt.WriteString(fmt.Sprintf("failed to create speech: %s\n", err))
				continue
			}

			decodedMP3, err := mp3.NewDecoder(audioStream)
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
