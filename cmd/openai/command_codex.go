package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/picatz/openai/codex"
	"github.com/spf13/cobra"
)

func newCodexCommand() *cobra.Command {
	var model, output string
	var timeout time.Duration
	unsupportedFlags := []string{"base-url", "stream", "web-search", "legacy", "temporary", "history-dir", "session"}
	cmd := &cobra.Command{
		Use:   "codex",
		Short: "Run the separately installed Codex CLI",
		Long: `Run Codex turns using the installed codex executable.
Use exec for one non-interactive turn or tui for an interactive fresh-thread workspace.
Codex configuration, authentication, and permission defaults remain in effect.
This command cannot answer interactive approval requests. Runs may incur charges.`,
		Args: cobra.NoArgs,
		// Codex owns its configuration and authentication. Do not initialize the
		// HTTP SDK or inherit the Responses model and request-timeout defaults.
		PersistentPreRunE: func(cmd *cobra.Command, _ []string) error {
			for _, name := range unsupportedFlags {
				if cmd.Flags().Changed(name) || cmd.Root().PersistentFlags().Changed(name) {
					return fmt.Errorf("--%s is not supported by openai codex; use Codex configuration or the native codex CLI for additional options", name)
				}
			}
			if cmd.Flags().Changed("model") && strings.TrimSpace(model) == "" {
				return fmt.Errorf("model must not be empty")
			}
			if timeout < 0 {
				return fmt.Errorf("timeout must not be negative")
			}
			if output != "text" && output != "json" {
				return fmt.Errorf("output must be text or json")
			}
			return nil
		},
		RunE: func(cmd *cobra.Command, _ []string) error { return cmd.Help() },
	}
	cmd.PersistentFlags().StringVar(&model, "model", "", "Codex model (default: installed Codex configuration)")
	cmd.PersistentFlags().DurationVar(&timeout, "timeout", 0, "Whole-run deadline, including reading stdin (0 disables)")
	cmd.PersistentFlags().StringVarP(&output, "output", "o", "text", "Completed result format: text or json")
	// Shadow unsupported root flags so help does not advertise API/session
	// options. Keep parsing them solely to return the actionable error above.
	for _, name := range unsupportedFlags {
		switch name {
		case "base-url", "history-dir", "session":
			cmd.PersistentFlags().String(name, "", "Unsupported with Codex")
		case "temporary":
			cmd.PersistentFlags().BoolP(name, "t", false, "Unsupported with Codex")
		default:
			cmd.PersistentFlags().Bool(name, false, "Unsupported with Codex")
		}
		_ = cmd.PersistentFlags().MarkHidden(name)
	}

	var directory, resumeID string
	execCmd := &cobra.Command{
		Use:   "exec [prompt...]",
		Short: "Run one foreground Codex turn",
		Long: `Run one non-interactive turn and print the final response after success.
Pass prompt arguments, or pipe a prompt on stdin (an explicit - also reads stdin).
Prompt arguments are joined with spaces; when present, stdin is not read.
Use -- before a prompt that begins with a dash. No shell expansion is performed.

The model is selected by Codex unless --model is provided. --timeout is a
whole-run deadline, not an HTTP request timeout. Other inherited API, streaming,
and local-session flags are unsupported and are rejected when explicitly set.`,
		Example: `  openai codex exec 'Summarize this repository'
  cat question.txt | openai codex exec -
  openai codex exec --cd ./project --timeout 5m 'Explain the test layout'
  openai codex exec --output json 'Summarize this repository'
  openai codex exec --resume THREAD_ID 'Explain the next step'`,
		Args: cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if cmd.Flags().Changed("cd") && strings.TrimSpace(directory) == "" {
				return fmt.Errorf("cd must not be empty")
			}
			if cmd.Flags().Changed("resume") && strings.TrimSpace(resumeID) == "" {
				return fmt.Errorf("resume requires an explicit Codex thread ID")
			}
			ctx := cmd.Context()
			if timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, timeout)
				defer cancel()
			}
			prompt, err := readPrompt(ctx, cmd.InOrStdin(), args)
			if err != nil {
				return err
			}
			options := codex.ThreadOptions{Model: model, WorkingDirectory: directory}
			var thread *codex.Thread
			if resumeID != "" {
				thread, err = codex.ResumeThread(resumeID, codex.Options{}, options)
			} else {
				thread, err = codex.NewThread(codex.Options{}, options)
			}
			if err != nil {
				return fmt.Errorf("Codex CLI unavailable; install it from https://github.com/openai/codex and ensure codex is on PATH: %w", err)
			}
			turn, err := thread.RunText(ctx, prompt, nil)
			if err != nil {
				return fmt.Errorf("run Codex turn: %w", err)
			}
			if output == "json" {
				return json.NewEncoder(cmd.OutOrStdout()).Encode(struct {
					ThreadID      string       `json:"thread_id"`
					FinalResponse string       `json:"final_response"`
					Usage         *codex.Usage `json:"usage"`
				}{ThreadID: thread.ID(), FinalResponse: turn.FinalResponse, Usage: turn.Usage})
			}
			_, err = fmt.Fprintln(cmd.OutOrStdout(), turn.FinalResponse)
			return err
		},
	}
	execCmd.Flags().StringVarP(&directory, "cd", "C", "", "Working directory for Codex (default: current directory)")
	execCmd.Flags().StringVar(&resumeID, "resume", "", "Continue an explicit Codex thread ID; not an openai local session ID")
	cmd.AddCommand(execCmd, newCodexTUICommand(&model, &timeout))
	return cmd
}
