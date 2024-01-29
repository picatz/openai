package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/picatz/openai"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

var chatCommand = &cobra.Command{
	Use:   "chat",
	Short: "Chat with the OpenAI API",
	RunE: func(cmd *cobra.Command, args []string) error {
		startChat(client, model)

		return nil
	},
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
