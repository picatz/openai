package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/picatz/openai/internal/responses"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

const (
	keyAltLeft  = 0xd800 + 5
	keyAltRight = 0xd800 + 6
)

func init() {
	responsesCommand.AddCommand(
		responsesChatCommand,
		responsesGetCommand,
		responsesDeleteCommand,
	)

	rootCmd.AddCommand(
		responsesCommand,
	)
}

func getAPIKeyEnvVariable() (string, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("OPENAI_API_KEY environment variable is not set")
	}
	return apiKey, nil
}

var responsesCommand = &cobra.Command{
	Use:   "responses",
	Short: "Manage the OpenAI Responses API",
	RunE: func(cmd *cobra.Command, args []string) error {
		apiKey, err := getAPIKeyEnvVariable()
		if err != nil {
			return fmt.Errorf("failed to get API key: %w", err)
		}

		client := responses.NewClient(apiKey, http.DefaultClient)

		startResponsesChat(cmd.Context(), client, chatModel)

		return nil
	},
}

var responsesChatCommand = &cobra.Command{
	Use:   "chat",
	Short: "Chat with the OpenAI Responses API",
	RunE: func(cmd *cobra.Command, args []string) error {
		apiKey, err := getAPIKeyEnvVariable()
		if err != nil {
			return fmt.Errorf("failed to get API key: %w", err)
		}

		client := responses.NewClient(apiKey, http.DefaultClient)

		startResponsesChat(cmd.Context(), client, chatModel)

		return nil
	},
}

var responsesGetCommand = &cobra.Command{
	Use:   "get",
	Short: "Get a single response",
	RunE: func(cmd *cobra.Command, args []string) error {
		apiKey, err := getAPIKeyEnvVariable()
		if err != nil {
			return fmt.Errorf("failed to get API key: %w", err)
		}

		client := responses.NewClient(apiKey, http.DefaultClient)

		resp, err := client.Create(cmd.Context(), responses.Request{
			Model:      chatModel,
			Input:      responses.Text(strings.Join(args, " ")),
			ToolChoice: responses.RequestToolChoiceAuto,
			Tools: responses.RequestTools{
				responses.RequestToolWebSearchPreview{},
			},
			Store: false,
		})
		if err != nil {
			return fmt.Errorf("failed to create response: %w", err)
		}

		for _, output := range resp.Output {
			if output.Type != "message" {
				continue
			}
			cmd.OutOrStdout().Write([]byte(output.Content[0].Text + "\n"))
		}

		return nil
	},
}

var responsesDeleteCommand = &cobra.Command{
	Use:   "delete",
	Short: "Delete a single response",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		apiKey, err := getAPIKeyEnvVariable()
		if err != nil {
			return fmt.Errorf("failed to get API key: %w", err)
		}

		client := responses.NewClient(apiKey, http.DefaultClient)

		respID := args[0]
		if err := client.Delete(cmd.Context(), respID); err != nil {
			return fmt.Errorf("failed to delete response %q: %w", respID, err)
		}

		fmt.Fprintf(cmd.OutOrStdout(), "Deleted response %q\n", respID)
		return nil
	},
}

func startResponsesChat(ctx context.Context, client *responses.Client, model string) error {
	// Set the terminal to raw mode.
	fd := int(os.Stdout.Fd())
	oldState, err := term.MakeRaw(fd)
	if err != nil {
		return fmt.Errorf("failed to set terminal to raw mode: %w", err)
	}
	defer term.Restore(0, oldState)

	termWidth, termHeight, err := term.GetSize(fd)
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

	// Track file path completions so repeated tab presses cycle through
	// matches. We allow cycling forward with <Tab> or Alt+Right and cycling
	// backward with Alt+Left.
	var fileComplete struct {
		prefix  string
		matches []string
		index   int
	}

	// Autocomplete for commands and special tokens. Tab and Alt+Right cycle
	// forward while Alt+Left cycles backward through file path matches.
	t.AutoCompleteCallback = func(line string, pos int, key rune) (newLine string, newPos int, ok bool) {
		switch key {
		case '\t', keyAltRight, keyAltLeft:
		default:
			return line, pos, false
		}

		for _, cmd := range []string{"exit", "clear", "delete", "copy", "tokens", "help"} {
			if strings.HasPrefix(cmd, line) {
				return cmd, len(cmd), true
			}
		}

		// Autocomplete the last "word" if it looks like a special token.
		parts := strings.Fields(line)
		if len(parts) == 0 {
			return line, pos, false
		}
		last := parts[len(parts)-1]

		if strings.HasPrefix(last, "<clip") {
			token := "<clipboard>"
			if strings.HasPrefix(token, last) {
				parts[len(parts)-1] = token
				newLine = strings.Join(parts, " ")
				return newLine, len(newLine), true
			}
		}

		if strings.HasPrefix(last, "#file:") {
			prefix := strings.TrimPrefix(last, "#file:")
			if prefix != fileComplete.prefix {
				fileComplete.prefix = prefix
				fileComplete.index = 0
				fileComplete.matches, _ = filepath.Glob(prefix + "*")
			}
			if len(fileComplete.matches) > 0 {
				if key == keyAltLeft {
					fileComplete.index--
					if fileComplete.index < 0 {
						fileComplete.index = len(fileComplete.matches) - 1
					}
				} else {
					fileComplete.index = (fileComplete.index + 1) % len(fileComplete.matches)
				}
				suggestion := "#file:" + fileComplete.matches[fileComplete.index]
				parts[len(parts)-1] = suggestion
				newLine = strings.Join(parts, " ")
				return newLine, len(newLine), true
			}
		}

		if strings.HasPrefix(last, "#url:") {
			token := "#url:"
			if last == token {
				return line, pos, true
			}
		}

		return line, pos, false
	}

	// Print welcome message.
	bt.WriteString(styleBold.Render("Welcome to the OpenAI API Responses CLI chat mode!"))
	bt.WriteString("\n\n")
	bt.WriteString(styleWarning.Render("WARNING") + styleFaint.Render(": All responses are stored and deleted after exiting.\n\n"))
	bt.WriteString("\033[0G")
	printResponsesChatHelp(bt)

	var allRespIDs []string
	defer func() {
		total := len(allRespIDs)
		if total == 0 {
			return
		}
		bt.WriteString("\n")
		bt.Flush()

		for i, respID := range allRespIDs {
			if err := client.Delete(ctx, respID); err != nil {
				bt.WriteString(respID + ":" + err.Error() + "\n")
				bt.Flush()
				return
			}
			var (
				progress      = i + 1
				percent       = float64(progress) / float64(total)
				barWidth      = 20
				completedBars = int(percent * float64(barWidth))
				remainingBars = barWidth - completedBars
				progressBar   = strings.Repeat("█", completedBars) + strings.Repeat("_", remainingBars)
			)
			bt.WriteString(styleFaint.Render("\033[0G" + fmt.Sprintf("Deleting responses %s (%d/%d)", progressBar, progress, total)))
			bt.Flush()
		}
		bt.WriteString("\n")
		bt.WriteString("\n")
		bt.Flush()
	}()

	var (
		prevRespID  string
		lastMessage string
		totalTokens int
	)

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
			bt.WriteString(err.Error())
			bt.Flush()
			return fmt.Errorf("failed to read line: %w", err)
		}

		// Check if the user wants to exit.
		if strings.TrimSpace(input) == "exit" {
			break
		}

		// Check if user wants to clear the screen.
		if strings.TrimSpace(input) == "clear" {
			// Clear the screen.
			cls()
			continue
		}

		if strings.TrimSpace(input) == "help" {
			cls()
			printResponsesChatHelp(bt)
			continue
		}

		if strings.TrimSpace(input) == "tokens" {
			bt.WriteString(fmt.Sprintf("Tokens used: %d\n", totalTokens))
			bt.Flush()
			continue
		}

		if strings.TrimSpace(input) == "copy" {
			if err := writeClipboard(lastMessage); err != nil {
				bt.WriteString("Clipboard error: " + err.Error() + "\n")
			}
			bt.Flush()
			continue
		}

		fields := strings.Fields(input)

		switch fields[0] {
		case "delete":
			if len(fields) == 1 {
				bt.WriteString("Usage: delete <number>\n")
				bt.Flush()
				continue
			}
			if len(fields) == 2 {
				num, err := strconv.Atoi(fields[1])
				if err != nil {
					bt.WriteString("Usage: delete <number>\n")
					bt.Flush()
					continue
				}
				var deletedN int
				for i := 0; i < num && len(allRespIDs) > 0; i++ {
					respID := allRespIDs[len(allRespIDs)-1]     // Get the last response ID.
					allRespIDs = allRespIDs[:len(allRespIDs)-1] // Remove it from the list.

					if err := client.Delete(ctx, respID); err != nil {
						bt.WriteString(respID + ":" + err.Error() + "\n")
						bt.Flush()
						return fmt.Errorf("failed to delete response %q: %w", respID, err)
					}

					if prevRespID == respID {
						prevRespID = ""
					}

					deletedN++
				}
				fmt.Fprintf(bt, "Deleted %d responses.\n", deletedN)
				bt.Flush()
				continue
			}
		}

		// Replace special tokens before sending to the API.
		processors := []struct {
			token   string
			process func(context.Context, string) (string, error)
		}{
			{"#file:", addFilesToInput},
			{"#url:", addURLsToInput},
			{"<clipboard>", addClipboardToInput},
		}

		var procErr error
		for _, p := range processors {
			if strings.Contains(input, p.token) {
				input, procErr = p.process(ctx, input)
				if procErr != nil {
					bt.WriteString(fmt.Sprintf("Error processing %s: %s\n", p.token, procErr))
					bt.Flush()
					break
				}
			}
		}
		if procErr != nil {
			continue
		}

		resp, err := client.Create(ctx, responses.Request{
			Model:              model,
			PreviousResponseID: prevRespID,
			Input:              responses.Text(input),
			ToolChoice:         responses.RequestToolChoiceAuto,
			Tools: responses.RequestTools{
				responses.RequestToolWebSearchPreview{},
			},
		})
		if err != nil {
			bt.WriteString(err.Error())
			bt.Flush()
			return fmt.Errorf("failed to create response: %w", err)
		}

		for _, output := range resp.Output {
			if output.Type != "message" {
				continue
			}

			// Get 3/4 of the terminal width.
			threeQuarterWidth := termWidth * 3 / 4

			// Print the output using markdown-friendly terminal rendering.
			s, err := renderMarkdown(strings.TrimRight(output.Content[0].Text, "\n"), threeQuarterWidth)
			if err != nil {
				bt.WriteString(err.Error())
				bt.Flush()
				return fmt.Errorf("failed to render markdown: %w", err)
			}

			bt.WriteString(s)
		}

		allRespIDs = append(allRespIDs, resp.ID)
		prevRespID = resp.ID
		lastMessage = strings.TrimRight(resp.Output[0].Content[0].Text, "\n")
		totalTokens += resp.Usage.TotalTokens
	}

	// Flush the buffer to the terminal.
	if err := bt.Flush(); err != nil {
		return fmt.Errorf("failed to flush buffer: %w", err)
	}

	return nil
}

// addFilesToInput looks for "#file:" tokens in the provided input and
// replaces them with the referenced file's contents. The modified input is
// returned. If a file cannot be opened the original input and an error are
// returned.
// The command's context is passed so file reading can be canceled if needed.
func addFilesToInput(ctx context.Context, input string) (string, error) {
	fields := strings.Fields(input)
	for _, field := range fields {
		if strings.HasPrefix(field, "#file:") {
			filePath := strings.TrimPrefix(field, "#file:")
			data, err := os.ReadFile(filePath)
			if err != nil {
				return input, fmt.Errorf("failed to open file %q: %w", filePath, err)
			}
			input = strings.Replace(input, field, string(data), 1)
		}
	}
	return input, nil
}

// addURLsToInput looks for "#url:" tokens in the provided input, fetches the
// contents of the referenced URLs, and replaces the tokens with the fetched
// data. The modified input is returned. If a URL cannot be fetched the
// original input and an error are returned.
// A context parameter allows request cancellation.
func addURLsToInput(ctx context.Context, input string) (string, error) {
	fields := strings.Fields(input)
	for _, field := range fields {
		if strings.HasPrefix(field, "#url:") {
			url := strings.TrimPrefix(field, "#url:")
			if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
				url = "https://" + url
			}
			if strings.HasPrefix(url, "http://") {
				url = strings.Replace(url, "http://", "https://", 1)
			}
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			if err != nil {
				return input, fmt.Errorf("failed to create request for URL %q: %w", url, err)
			}
			resp, err := http.DefaultClient.Do(req)
			if err != nil {
				return input, fmt.Errorf("failed to fetch URL %q: %w", url, err)
			}
			defer resp.Body.Close()
			if resp.StatusCode < 200 || resp.StatusCode >= 300 {
				io.Copy(io.Discard, resp.Body)
				return input, fmt.Errorf("failed to fetch URL %q: %s", url, resp.Status)
			}

			body, err := io.ReadAll(resp.Body)
			if err != nil {
				return input, fmt.Errorf("error reading response body from URL %q: %w", url, err)
			}
			input = strings.Replace(input, field, string(body), 1)
		}
	}
	return input, nil
}

// addClipboardToInput looks for the "<clipboard>" token in the provided input
// and replaces it with the system clipboard contents. The context can be used
// to abort clipboard operations.
func addClipboardToInput(ctx context.Context, input string) (string, error) {
	clip, err := readClipboard()
	if err != nil {
		return input, err
	}
	return strings.ReplaceAll(input, "<clipboard>", clip), nil
}

// printResponsesChatHelp prints available commands and token usage instructions
// to the provided buffer.
func printResponsesChatHelp(bt *bufio.Writer) {
	bt.WriteString(styleBold.Render("Commands") + " " + styleFaint.Render("(tab complete)") + "\n\n")
	bt.WriteString("- " + styleFaint.Render("clear") + " to clear screen.\n")
	bt.WriteString("- " + styleFaint.Render("delete") + " to delete previous response (up to given number).\n")
	bt.WriteString("- " + styleFaint.Render("copy") + " to copy last response to the clipboard.\n")
	bt.WriteString("- " + styleFaint.Render("tokens") + " to show token usage.\n")
	bt.WriteString("- " + styleFaint.Render("help") + " to show this help.\n")
	bt.WriteString("- " + styleFaint.Render("exit") + " to quit.\n\n")
	bt.WriteString("Use " + styleInfo.Render("<clipboard>") + " to include clipboard content in a message.\n")
	bt.WriteString("Use " + styleInfo.Render("#file:") + stylePath.Render("path") + " to include file content in a message.\n")
	bt.WriteString("Use \t to cycle file paths forward and \u2190/\u2192 (Alt+Left/Right) to cycle backward or forward.\n")
	bt.WriteString("Use " + styleInfo.Render("#url:") + stylePath.Render("path") + " to include URL content in a message.\n")
	bt.WriteString("\n")
	bt.Flush()
}
