package main

import (
	"fmt"
	"os"

	"github.com/charmbracelet/glamour"
	"github.com/picatz/openai"
)

func main() {
	var (
		apiKey = os.Getenv("OPENAI_API_KEY")
		model  = os.Getenv("OPENAI_MODEL")
	)

	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY environment variable is not set")
		os.Exit(1)
	}

	if model == "" {
		model = openai.ModelGPT4TurboPreview
	}

	// Check if we have any arguments.
	args := os.Args[1:]

	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "No arguments provided")
		os.Exit(1)
	}

	// Get OpenAI client.
	client := openai.NewClient(apiKey)

	// Identify mode.
	var mode Mode
	if len(args) == 1 && args[0] == "chat" {
		mode = ModeChat
	} else if len(args) == 1 && args[0] == "assistant" {
		mode = ModeAssistant
	}

	switch mode {
	case ModeChat:
		startChat(client, model)
	case ModeAssistant:
		err := startAssistantChat(client, model)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s", err)
			os.Exit(1)
		}
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
