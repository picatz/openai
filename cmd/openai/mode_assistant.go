package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/ebitengine/oto/v3"
	"github.com/hajimehoshi/go-mp3"
	"github.com/picatz/openai"
	"golang.org/x/term"
)

// startAssistantChat starts an interactive chat session with the OpenAI API, this is a REPL-like
// command-line program that allows you use the new assistant API (in beta).
func startAssistantChat(client *openai.Client, model string) error {
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

	cls := func() {
		// Clear the screen.
		bt.WriteString("\033[2J")

		// Move to the top left.
		bt.WriteString("\033[H")

		// Flush the buffer to the terminal.
		bt.Flush()
	}

	cls()

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

	// Move to left edge.
	bt.WriteString("\033[0G")

	bt.WriteString(styleWarning.Render("WARNING") + styleFaint.Render(": Messages and files disappear after exiting.\n\n"))
	bt.Flush()

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
			cls()
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
				defer client.DeleteFile(ctx, &openai.DeleteFileRequest{
					ID: uploadResp.ID,
				})

				resp.Body.Close()

				if err != nil {
					bt.WriteString(fmt.Sprintf("failed to upload file: %s\n", err))
					continue
				}

				bt.WriteString(fmt.Sprintf("uploaded URL content: %s\n", uploadResp.ID))

				_, err = client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
					ID: assistant.ID,
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

			defer client.DeleteFile(ctx, &openai.DeleteFileRequest{
				ID: uploadResp.ID,
			})

			bt.WriteString(fmt.Sprintf("uploaded file: %s\n", uploadResp.ID))

			_, err = client.UpdateAssistant(ctx, &openai.UpdateAssistantRequest{
				ID: assistant.ID,
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
				ID:           assistant.ID,
				Instructions: input,
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
			AssistantID: assistant.ID,
		})
		if err != nil {
			return fmt.Errorf("failed to create run: %w", err)
		}

		// Wait for the run to complete.
		// Wait for the run to finish
		var ranResp *openai.Run
		for {
			// bt.WriteString(fmt.Sprintf("waiting for run to complete: %s\n", runResp.ID))
			time.Sleep(700 * time.Millisecond)

			ranResp, err = client.GetRun(ctx, &openai.GetRunRequest{
				ThreadID: thread.ID,
				RunID:    runResp.ID,
			})
			if err != nil {
				return fmt.Errorf("failed to get run: %w", err)
			}

			var done bool

			switch ranResp.Status {
			case openai.RunStatusCompleted:
				done = true
			case openai.RunStatusQueued, openai.RunStatusInProgress:
				continue
			default:
				return fmt.Errorf("unexpected run status: %s", ranResp.Status)
			}

			if done {
				break
			}
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

			decodedMp3, err := mp3.NewDecoder(audioStream)
			if err != nil {
				return fmt.Errorf("failed to decode mp3: %w", err)
			}

			// Create a new 'player' that will handle our sound. Paused by default.
			player := otoCtx.NewPlayer(decodedMp3)

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
