package main

import (
	"github.com/spf13/cobra"
)

var assistantCommand = &cobra.Command{
	Use:        "assistant",
	Deprecated: "use the responses API instead!\n\n\t$ openai responses chat\n",
}

func init() {
	assistantCommand.AddCommand()

	rootCmd.AddCommand(
		assistantCommand,
	)
}
