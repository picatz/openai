package main

import "github.com/spf13/cobra"

func newAssistantCommand() *cobra.Command {
	return &cobra.Command{Use: "assistant", Deprecated: "use openai responses chat instead"}
}
