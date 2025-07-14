package main

import (
	"net/http"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "openai",
	Short: "OpenAI CLI",
	RunE: func(cmd *cobra.Command, args []string) error {
		c := openai.NewClient(
			option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
			option.WithHTTPClient(http.DefaultClient),
		)
		return startResponsesChat(cmd.Context(), &c, chatModel)
	},
}
