package main

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

var (
	apiKey    = os.Getenv("OPENAI_API_KEY")
	chatModel = cmp.Or(os.Getenv("OPENAI_MODEL"), openai.ChatModel("gpt-4.1"))
	baseURL   = os.Getenv("OPENAI_API_URL")

	client *openai.Client
)

func init() {
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "OPENAI_API_KEY environment variable is not set")
		os.Exit(1)
	}

	clientOptions := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if baseURL != "" {
		clientOptions = append(clientOptions, option.WithBaseURL(baseURL))
	}

	client = openai.NewClient(clientOptions...)
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	if err := rootCmd.ExecuteContext(ctx); err != nil {
		os.Exit(1)
	}
}
