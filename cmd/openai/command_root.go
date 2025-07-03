package main

import (
	"net/http"
	"os"

	"github.com/picatz/openai/internal/responses"
	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "openai",
	Short: "OpenAI CLI",
	RunE: func(cmd *cobra.Command, args []string) error {
		client := responses.NewClient(os.Getenv("OPENAI_API_KEY"), http.DefaultClient)
		return startResponsesChat(cmd.Context(), client, chatModel)
	},
}
