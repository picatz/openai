package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/picatz/openai"
)

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

// Mode of operation.
type Mode string

// Modes of operation.
const (
	ModeEdit     Mode = "edit"
	ModeComplete Mode = "complete"
	ModeChat     Mode = "chat"
)

func main() {
	// Check if we have an API key.
	apiKey := os.Getenv("OPENAI_API_KEY")

	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY environment variable is not set")
		os.Exit(1)
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
		startChat(client)
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

// startChat starts an interactive chat session with the OpenAI API, this is a REPL-like
// command-line program that allows you to chat with the API.
func startChat(client *openai.Client) {
	// Print a welcome message to explain how to use the chat mode.
	fmt.Print("Welcome to the OpenAI API CLI chat mode. Type '\033[2mdelete\033[0m' to forget last message. Type '\033[2mexit\033[0m' to quit.\n\n")

	// Keep track of the chat messages, both from the user and the API.
	messages := []openai.ChatMessage{}

	var systemMessage openai.ChatMessage

	tokens := 0

	for {
		// Print a prompt to the user using bold ANSI escape codes.
		fmt.Printf("\033[1m(messages: %d: tokens: %d)> \033[0m", len(messages), tokens)

		// Read up to 4096 characters from STDIN.
		b := make([]byte, 4096)
		n, err := io.ReadAtLeast(os.Stdin, b, 1)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s", err)
			os.Exit(1)
		}

		// Get the user input.
		input := strings.Trim(string(b[:n]), "\r")

		// Check if the user wants to exit.
		if strings.TrimSpace(input) == "exit" {
			break
		}

		// Check if the user wants to erase.
		if strings.TrimSpace(input) == "delete" {
			// Remove the last message.
			if len(messages) > 0 {
				messages = messages[:len(messages)-1]
			}
			continue
		}

		// Add the system message to the messages if the input starts with "system:".
		//
		// Note, you wouldn't want to do this if didn't trust the user. But, it's a nice
		// way to add some context to the chat session.
		if strings.HasPrefix(input, "system:") {
			// Add the system message to the messages.
			systemMessage = openai.ChatMessage{
				Role:    openai.ChatRoleSystem,
				Content: input,
			}
			messages = append(messages, systemMessage)
			fmt.Print("\n\033[2m")
			fmt.Println("Configured system.")
			fmt.Print("\033[0m\n")
			continue
		}

		// Add the user input to the messages.
		messages = append(messages, openai.ChatMessage{
			Role:    openai.ChatRoleUser,
			Content: input,
		})

		resp, err := chatRequest(client, messages)

		if err != nil {
			fmt.Fprintf(os.Stderr, "%s", err)
			os.Exit(1)
		}

		// Print the output using ASNI dim escape codes.
		fmt.Print("\n\033[2m")
		fmt.Println(strings.TrimSpace(resp.Choices[0].Message.Content))
		fmt.Print("\033[0m\n")

		// Add the bot response to the messages.
		messages = append(messages, resp.Choices[0].Message)

		// Keep track of tokens used for the session to avoid going over the limit
		// which is 4096
		tokens += resp.Usage.TotalTokens

		if tokens > 4000 {
			summary, summaryTokens := summarizeMessages(client, messages, 0)

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
}

func chatRequest(client *openai.Client, messages []openai.ChatMessage) (*openai.CreateChatResponse, error) {
	// Wait maxiumum of 5 minutes for a response, which provides
	// a lot of time for the API to respond, but it should a matter
	// of seconds, not minutes.
	ctx, cancel := reqCtx(5 * time.Minute)
	defer cancel()

	// Create completion request.
	resp, err := client.CreateChat(ctx, &openai.CreateChatRequest{
		Model:     openai.ModelGPT35Turbo, // openai.ModelCodeDavinci002 ?
		Messages:  messages,
		MaxTokens: 2048,
	})
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func reqCtx(timeout time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), timeout)
}

// summarizeMessages summarizes the messages using the OpenAI API.
func summarizeMessages(client *openai.Client, messages []openai.ChatMessage, attempts int) (string, int) {
	// Create a prompt from the messages.
	prompt := []string{}
	prompt = append(prompt, "Generate a summary of previous chat messages for context:")
	for _, m := range messages {
		prompt = append(prompt, fmt.Sprintf("%s: %s", m.Role, m.Content))
	}

	// Create a context with a timeout of 120 seconds.
	ctx, cancel := reqCtx()
	defer cancel()

	// Track the number of attempts for retries.
	attempts++

	// Create completion request.
	resp, err := client.CreateCompletion(ctx, &openai.CreateCompletionRequest{
		Model:     openai.ModelGPT35Turbo,
		Prompt:    prompt,
		MaxTokens: 1024,
	})
	if err != nil {
		// TODO: This is a hack to handle the 429 error. We should handle this better.
		if attempts < 5 && strings.Contains(err.Error(), " unexpected status code: 429: Too Many Request") {
			// If we get a 429 error, it means we've exceeded the API rate limit.
			// In this case, we'll just wait 5 seconds and try again.
			time.Sleep(5 * time.Second)

			// Try again.
			return summarizeMessages(client, messages, attempts)
		}
		fmt.Fprintf(os.Stderr, "failed to summarize previous chat messages after %d attempts: %s", attempts, err)
		os.Exit(1)
	}

	return resp.Choices[0].Text, resp.Usage.TotalTokens
}
