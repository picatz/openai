package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/picatz/openai"
)

func startEditMode(ctx context.Context, client *openai.Client, args []string) {
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
}
