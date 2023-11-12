package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	"github.com/picatz/openai"
	"golang.org/x/term"
)

var (
	styleBold   = lipgloss.NewStyle().Bold(true)
	styleFaint  = lipgloss.NewStyle().Faint(true)
	numberColor = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("69"))
)

// TODO: make cross platform, macos only for now
func readClipboard() (string, error) {
	cmd := exec.Command("pbpaste")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func writeClipboard(s string) error {
	cmd := exec.Command("pbcopy")
	cmd.Stdin = strings.NewReader(s)
	return cmd.Run()
}

// Small command-line utility to use the OpenAI API. It requires
// an API key to be set in the OPENAI_API_KEY environment variable.
//
// It reads information from a given STDIN pipe, if provided, to put it in "edit mode".
// Arguments, are usesd as the prompt to customize the behavior of the edit. Otherwise,
// if there's no STDIN pipe, then arguments are the prompt for "complete mode".
//
// # Edit Mode
//
// 	$ echo "This is a test" | openai change the word 'test' to 'example'
// 	This is an example
//
// # Complete Mode
//
// 	$ opeani "Once upon a time"
// 	Once upon a time, there was a beautiful princess who lived in a castle in a far-off kingdom.
// 	She was beloved by all the people in her kingdom, especially her parents, the king and queen.
// 	One day, a mysterious stranger appeared at the castle gates, claiming to have a special message
// 	for the princess. The princess, curious about the stranger's message, invited him inside. The
// 	stranger told her that she was the chosen one, destined to save her kingdom from a powerful dark
// 	force that threatened to consume it. Despite her fear, the brave princess accepted her fate and set
// 	off on a quest to save her kingdom. Along the way, she encountered many obstacles and trials which
// 	she overcame with determination and courage. After a long and difficult journey, the princess
// 	eventually succeeded in her mission and saved her kingdom. She returned home to great celebration
// 	and was forever remembered as a hero.
//
// # Chat Mode
//
//  This mode will start an interactive chat session with the OpenAI API on the command line. It can
//  be used similar to ChatGPT's web interface, but on the command line.
//
// 	$ openai chat
//  ...
//  > system: You are a X, only do Y and Z. Not A, B, C.
//  ...
//  > Whatever you want to say to the AI
//  ...
//
// # Assistant Mode
//
//  This mode will start an interactive chat session with the OpenAI API on the command line. It can
//  be used similar to ChatGPT's web interface, but on the command line. It is an advanced version of
//  chat mode with more features.

// Mode of operation.
type Mode string

// Modes of operation.
const (
	ModeEdit      Mode = "edit"
	ModeComplete  Mode = "complete"
	ModeChat      Mode = "chat"
	ModeAssistant Mode = "assistant"
)

func main() {
	// Check if we have an API key.
	apiKey := os.Getenv("OPENAI_API_KEY")
	model := os.Getenv("OPENAI_MODEL")

	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY environment variable is not set")
		os.Exit(1)
	}

	if model == "" {
		// TODO: maybe pre-flight check to see if GPT4 is available when not OPENAI_MODEL isn't set?
		//       Then, in the future, make this GPT-4 by default when it's more widely available.
		model = openai.ModelGPT35Turbo
	}

	// Check if we have any arguments.
	args := os.Args[1:]

	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "No arguments provided")
		os.Exit(1)
	}

	// Check if STDIN is provided.
	fi, err := os.Stdin.Stat()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s", err)
		os.Exit(1)
	}

	// Get OpenAI client.
	client := openai.NewClient(apiKey)

	// Identify mode.
	var mode Mode

	// Check if the user wants to start a chat session.
	if len(args) == 1 && args[0] == "chat" {
		mode = ModeChat
	} else if len(args) == 1 && args[0] == "assistant" {
		mode = ModeAssistant
	} else if fi.Mode()&os.ModeCharDevice == 0 {
		mode = ModeEdit
	} else {
		mode = ModeComplete
	}

	// Wait maxiumum of 120 seconds for a response, which provides
	// a lot of time for the API to respond, but it should a matter
	// of seconds, not minutes for most models / requests.
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Handle requests based on the mode.
	switch mode {
	case ModeChat:
		startChat(client, model)
	case ModeAssistant:
		startAssistantChat(client, model)
	case ModeEdit:
		// Read up to 4096 characters from STDIN.
		b := make([]byte, 4096)
		n, err := io.ReadAtLeast(os.Stdin, b, 1)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s", err)
			os.Exit(1)
		}

		// Create edit request.
		resp, err := client.CreateEdit(ctx, &openai.CreateEditRequest{
			Model:       openai.ModelTextDavinciEdit001, // This is the only model that supports edit at the moment.
			Instruction: strings.Join(args, " "),
			Input:       string(b[:n]),
			Temperature: 1,
			N:           1,
		})

		if err != nil {
			fmt.Fprintf(os.Stderr, "%s", err)
			os.Exit(1)
		}

		// Print the output.
		fmt.Println(strings.Trim(resp.Choices[0].Text, "\n"))
	case ModeComplete:
		// Create completion request.
		resp, err := client.CreateCompletion(ctx, &openai.CreateCompletionRequest{
			Model:       openai.ModelTextDavinciEdit003, // openai.ModelCodeDavinci002 ?
			Prompt:      []string{strings.Join(args, " ")},
			MaxTokens:   2048,
			Temperature: 0.9, // allow for more creativity than edit mode
			N:           1,
		})

		if err != nil {
			fmt.Fprintf(os.Stderr, "%s", err)
			os.Exit(1)
		}

		// Print the output.
		fmt.Println(strings.Trim(resp.Choices[0].Text, "\n"))
	}
}

var cacheFilePath = os.Getenv("HOME") + "/.openai-cli-chat-cache"

// startChat starts an interactive chat session with the OpenAI API, this is a REPL-like
// command-line program that allows you to chat with the API.
func startChat(client *openai.Client, model string) {

	// Keep track of the chat messages, both from the user and the API.
	messages := []openai.ChatMessage{}

	// Open cache file from users home directory.
	f, err := os.OpenFile(cacheFilePath, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s", err)
		os.Exit(1)
	}

	if fi, err := f.Stat(); err == nil && fi.Size() > 0 {
		// Read and parse the cache file into messages.
		err = json.NewDecoder(f).Decode(&messages)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s", err)
			os.Exit(1)
		}
	}

	// Close the cache file.
	err = f.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s", err)
		os.Exit(1)
	}

	var systemMessage openai.ChatMessage

	// Set the terminal to raw mode.
	oldState, err := term.MakeRaw(0)
	if err != nil {
		panic(err)
	}
	defer term.Restore(0, oldState)

	termWidth, termHeight, err := term.GetSize(0)
	if err != nil {
		panic(err)
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

	tokens := 0

	// Print a welcome message to explain how to use the chat mode.
	// t.Write([]byte("Welcome to the OpenAI API CLI chat mode. Type '\033[2mdelete\033[0m' to forget last message. Type '\033[2mexit\033[0m' to quit.\n\n"))

	// Autocomplete for commands.
	t.AutoCompleteCallback = func(line string, pos int, key rune) (newLine string, newPos int, ok bool) {
		// If the user presses tab, then autocomplete the command.
		if key == '\t' {
			for _, cmd := range []string{"exit", "clear", "delete", "copy", "erase", "system:", "<clipboard>"} {
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
	bt.WriteString(styleBold.Render("Welcome to the OpenAI API CLI chat mode!"))
	bt.WriteString("\n\n")
	bt.WriteString(styleBold.Render("Commands") + " " + styleFaint.Render("(tab complete)") + "\n\n")
	bt.WriteString("- " + styleFaint.Render("delete") + " to forget last message.\n")
	bt.WriteString("- " + styleFaint.Render("erase") + " to forget all messages.\n")
	bt.WriteString("- " + styleFaint.Render("clear") + " to clear screen.\n")
	bt.WriteString("- " + styleFaint.Render("copy") + " to copy last message to clipboard.\n")
	bt.WriteString("- " + styleFaint.Render("exit") + " to quit.\n\n")
	bt.WriteString("Use '" + styleFaint.Render("<clipboard>") + "' to include clipboard content in a message.\n\n")
	bt.Flush()

	for {
		// Move to left edge.
		bt.WriteString("\033[0G")

		// Print the prompt.
		bt.WriteString(styleBold.Render(styleBold.Render("(") +
			"messages: " + numberColor.Render(fmt.Sprintf("%d", len(messages))) + " " +
			"tokens: " + numberColor.Render(fmt.Sprintf("%d", tokens)) +
			styleBold.Render(")") +
			" > "))

		// Flush the buffer to the terminal.
		bt.Flush()

		// Read up to line from STDIN.
		input, err := t.ReadLine()
		if err != nil {
			bt.WriteString(err.Error())
			bt.Flush()
			return
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

		// Check if the user wants to erase the whole chat.
		if strings.TrimSpace(input) == "erase" {
			// Reset the messages.
			messages = []openai.ChatMessage{}
			tokens = 0
			continue
		}

		// Check if the user wants to erase the last message.
		if strings.TrimSpace(input) == "delete" {
			// Remove the last message.
			if len(messages) > 0 {
				messages = messages[:len(messages)-1]
			}
			continue
		}

		if strings.TrimSpace(input) == "copy" {
			// Copy the last message to the clipboard.
			if len(messages) > 0 {
				// Write the last message to the clipboard.
				err := writeClipboard(messages[len(messages)-1].Content)
				if err != nil {
					bt.WriteString(err.Error())
					bt.Flush()
					return
				}
			}
			continue
		}

		// Check if the message has any <clipboard> tags.
		if strings.Contains(input, "<clipboard>") {
			// Get the clipboard contents.
			str, err := readClipboard()
			if err != nil {
				bt.WriteString(err.Error())
				bt.Flush()
				return
			}

			// Replace the <clipboard> tag with the clipboard contents.
			input = strings.Replace(input, "<clipboard>", str, -1)
		}

		// Add the system message to the messages if the input starts with "system:".
		//
		// Note, you wouldn't want to do this if didn't trust the user. But, it's a nice
		// way to add some context to the chat session for a CLI program.
		if strings.HasPrefix(input, "system:") {
			// Add the system message to the messages.
			systemMessage = openai.ChatMessage{
				Role:    openai.ChatRoleSystem,
				Content: input,
			}
			messages = append(messages, systemMessage)

			bt.WriteString(styleFaint.Render("Confiured system.\n"))

			continue
		}

		// Add the user input to the messages.
		messages = append(messages, openai.ChatMessage{
			Role:    openai.ChatRoleUser,
			Content: input,
		})

		resp, err := chatRequest(client, model, messages)

		if err != nil {
			bt.WriteString(err.Error())
			bt.Flush()
			return
		}

		// Print the output using markdown-friendly terminal rendering.
		s, err := renderMarkdown(strings.TrimRight(resp.Choices[0].Message.Content, "\n"), termWidth)
		if err != nil {
			bt.WriteString(err.Error())
			bt.Flush()
			return
		}

		bt.WriteString(s)

		// Add the bot response to the messages.
		messages = append(messages, resp.Choices[0].Message)

		// Keep track of tokens used for the session to avoid going over the limit
		// which is generally 4096 for gpt3 and 8000 for gpt4.
		tokens += resp.Usage.TotalTokens

		if tokens >= 8000 {
			summary, summaryTokens := summarizeMessages(client, model, messages, 0)

			// Print the generated summary so the user can see what the bot is
			// thinking the conversation is about to inject any additional context
			// that was forgotten or missed.
			bt.WriteString(
				lipgloss.NewStyle().
					Width(80).
					Background(lipgloss.Color("69")).
					Foreground(lipgloss.Color("15")).
					Padding(1, 2).
					Render(summary),
			)
			bt.WriteString("\n")

			// Reset the messages to the summary.
			messages = []openai.ChatMessage{}

			// Add the summary to the messages.
			messages = append(messages, openai.ChatMessage{
				Role:    openai.ChatRoleSystem,
				Content: "Summary of previous messages for context: " + summary,
			})

			// Reset the token count.
			tokens = summaryTokens
		}
	}

	// Save the messages to the cache file.
	f, err = os.OpenFile(cacheFilePath, os.O_TRUNC|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		bt.WriteString(err.Error())
		bt.Flush()
		return
	}

	// Write the messages to the cache file.
	err = json.NewEncoder(f).Encode(messages)
	if err != nil {
		bt.WriteString(err.Error())
		bt.Flush()
		return
	}

	// Close the cache file.
	err = f.Close()
	if err != nil {
		bt.WriteString(err.Error())
		bt.Flush()
		return
	}
}

func renderMarkdown(s string, width int) (string, error) {
	r, err := glamour.NewTermRenderer(
		glamour.WithStylePath("dark"),
		glamour.WithWordWrap(width),
		glamour.WithPreservedNewLines(),
	)
	if err != nil {
		return fmt.Errorf("failed to create markdown renderer: %w", err).Error(), nil
	}

	out, err := r.Render(s)
	if err != nil {
		return fmt.Errorf("failed to render markdown: %w", err).Error(), nil
	}

	return out, nil
}

func chatRequest(client *openai.Client, model string, messages []openai.ChatMessage) (*openai.CreateChatResponse, error) {
	// Wait maxiumum of 5 minutes for a response, which provides
	// a lot of time for the API to respond, but it should a matter
	// of seconds, not minutes.
	ctx, cancel := reqCtx(5 * time.Minute)
	defer cancel()

	// Create completion request.
	resp, err := client.CreateChat(ctx, &openai.CreateChatRequest{
		Model:     model,
		Messages:  messages,
		MaxTokens: 2048,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create chat: %w", err)
	}

	return resp, nil
}

func reqCtx(timeout time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), timeout)
}

// summarizeMessages summarizes the messages using the OpenAI API.
func summarizeMessages(client *openai.Client, model string, messages []openai.ChatMessage, attempts int) (string, int) {
	// Create a context with a timeout of 5 minutes.
	ctx, cancel := reqCtx(5 * time.Minute)
	defer cancel()

	summaryMsgs := []openai.ChatMessage{
		{
			Role: openai.ChatRoleSystem,
			Content: strings.Join(
				[]string{
					"You are an expert at summarizing conversations.",
					"Write a detailed recap of the given conversation, including all the important information produced by the bot to continue having the conversation.",
					"Ensure to inclue all the people, places, and things said by the bot.",
					"The summary must be at least 100 characters long.",
					"Your summary should be no more than 2048 characters long.",
					"Ignore any irelevant information from the conversation that doesn't seem to fit.",
				}, " ",
			),
		},
	}

	b := strings.Builder{}

	for _, m := range messages {
		if m.Role == openai.ChatRoleSystem {
			continue
		}
		if m.Role == openai.ChatRoleUser {
			b.WriteString("User: ")
		} else {
			b.WriteString("Bot: ")
		}
		b.WriteString(m.Content)
		b.WriteString("\n\n")
	}

	summaryMsgs = append(summaryMsgs, openai.ChatMessage{
		Role:    openai.ChatRoleUser,
		Content: b.String(),
	})

	// Track the number of attempts for retries.
	attempts++

	// Create completion request.
	resp, err := client.CreateChat(ctx, &openai.CreateChatRequest{
		Model:     model,
		Messages:  summaryMsgs,
		MaxTokens: 1024,
	})
	if err != nil {
		// TODO: This is a hack to handle the 429 error. We should handle this better.
		if attempts < 5 && strings.Contains(err.Error(), " unexpected status code: 429: Too Many Request") {
			// If we get a 429 error, it means we've exceeded the API rate limit.
			// In this case, we'll just wait 5 seconds and try again.
			time.Sleep(5 * time.Second)

			// Try again.
			return summarizeMessages(client, model, messages, attempts)
		}

		panic(err)
	}

	return resp.Choices[0].Message.Content, resp.Usage.TotalTokens
}

func startAssistantChat(client *openai.Client, model string) {
	ctx := context.Background()

	assistant, err := client.CreateAssistant(ctx, &openai.CreateAssistantRequest{
		Model:        openai.ModelGPT41106Previw, // TODO: use given model
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
			// },
		},
	})
	if err != nil {
		panic(err)
	}

	defer func() {
		err := client.DeleteAssistant(ctx, &openai.DeleteAssistantRequest{
			ID: assistant.ID,
		})
		if err != nil {
			panic(err)
		}
	}()

	thread, err := client.CreateThread(ctx, nil)
	if err != nil {
		panic(err)
	}

	defer func() {
		err := client.DeleteThread(ctx, &openai.DeleteThreadRequest{
			ID: thread.ID,
		})
		if err != nil {
			panic(err)
		}
	}()

	// Set the terminal to raw mode.
	oldState, err := term.MakeRaw(0)
	if err != nil {
		panic(err)
	}
	defer term.Restore(0, oldState)

	termWidth, termHeight, err := term.GetSize(0)
	if err != nil {
		panic(err)
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

	// Print a welcome message to explain how to use the chat mode.
	// t.Write([]byte("Welcome to the OpenAI API CLI chat mode. Type '\033[2mdelete\033[0m' to forget last message. Type '\033[2mexit\033[0m' to quit.\n\n"))

	// Autocomplete for commands.
	t.AutoCompleteCallback = func(line string, pos int, key rune) (newLine string, newPos int, ok bool) {
		// If the user presses tab, then autocomplete the command.
		if key == '\t' {
			for _, cmd := range []string{"exit", "clear", "delete", "copy", "erase", "system:", "<clipboard>"} {
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
	bt.WriteString(styleBold.Render("Welcome to the OpenAI API CLI assistant mode!\n"))
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
			bt.WriteString(err.Error())
			bt.Flush()
			return
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

		_, err = client.CreateMessage(ctx, &openai.CreateMessageRequest{
			ThreadID: thread.ID,
			Role:     openai.ChatRoleUser,
			Content:  input,
		})
		if err != nil {
			panic(err)
		}

		runResp, err := client.CreateRun(ctx, &openai.CreateRunRequest{
			ThreadID:    thread.ID,
			AssistantID: assistant.ID,
		})
		if err != nil {
			panic(err)
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
				panic(err)
			}

			var done bool

			switch ranResp.Status {
			case openai.RunStatusCompleted:
				done = true
			case openai.RunStatusQueued, openai.RunStatusInProgress:
				continue
			default:
				panic(fmt.Errorf("unexpected run status: %s", ranResp.Status))
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
			panic(err)
		}

		textMap := listResp.Data[0].Content[0]["text"].(map[string]any)

		nextMsg := fmt.Sprintf("%s", textMap["value"])

		nextMsgMd, err := renderMarkdown(nextMsg, termWidth)
		if err != nil {
			panic(err)
		}

		bt.WriteString(nextMsgMd)
	}
}
