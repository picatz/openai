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

// Mode of operation.
type Mode string

// Modes of operation.
const (
	ModeEdit     Mode = "edit"
	ModeComplete Mode = "complete"
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

	// If STDIN is provided, we are in "edit mode". Otherwise,
	// we are in "complete mode".
	if fi.Mode()&os.ModeCharDevice == 0 {
		mode = ModeEdit
	} else {
		mode = ModeComplete
	}

	// Wait maxiumum of 30 seconds for a response.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// Handle requests based on the mode.
	switch mode {
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
			Model:       openai.ModelTextDavinciEdit001,
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
