package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/picatz/openai"
)

func startCompletionMode(ctx context.Context, client *openai.Client, args []string) {
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
