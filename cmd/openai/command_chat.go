package main

import (
	"fmt"

	"github.com/picatz/openai/internal/chat"
	"github.com/spf13/cobra"
)

var chatCommand = &cobra.Command{
	Use:   "chat",
	Short: "Chat with the OpenAI API",
	RunE: func(cmd *cobra.Command, args []string) error {
		chatSession, restore, err := chat.NewSession(cmd.Context(), client, model, cmd.InOrStdin(), cmd.OutOrStdout())
		if err != nil {
			return fmt.Errorf("failed to create chat session: %w", err)
		}
		defer restore()

		chatSession.Run(cmd.Context())

		return nil
	},
}

func init() {
	rootCmd.AddCommand(
		chatCommand,
	)
}
