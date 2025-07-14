package main

import (
	"cmp"
	"context"
	"os"
	"os/signal"

	"github.com/charmbracelet/fang"
	"github.com/openai/openai-go"
)

var (
	chatModel = cmp.Or(os.Getenv("OPENAI_MODEL"), openai.ChatModel("gpt-4o"))

	client *openai.Client
)

func ptr[T any](v T) *T {
	return &v
}

func init() {
	client = ptr(openai.NewClient())
}

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, os.Kill)
	defer cancel()

	err := fang.Execute(ctx, rootCmd)
	if err != nil {
		os.Exit(1)
	}
}
