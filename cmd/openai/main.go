package main

import (
	"fmt"
	"os"

	"github.com/picatz/openai"
)

var (
	apiKey = os.Getenv("OPENAI_API_KEY")
	model  = os.Getenv("OPENAI_MODEL")

	client *openai.Client
)

func init() {
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY environment variable is not set")
		os.Exit(1)
	}

	if model == "" {
		model = openai.ModelGPT4TurboPreview
	}

	client = openai.NewClient(apiKey)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
