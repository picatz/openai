package main

import "github.com/spf13/cobra"

var rootCmd = &cobra.Command{
	Use:   "openai",
	Short: "OpenAI CLI",
	RunE: func(cmd *cobra.Command, args []string) error {
		return startAssistantChat(client, model, "", "")
	},
}
