package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/picatz/openai/internal/codexsession"
	"github.com/picatz/openai/internal/codexworkspace"
	"github.com/spf13/cobra"
)

func newCodexTUICommand(model *string, timeout *time.Duration) *cobra.Command {
	var directory string
	var noColor, reducedMotion bool
	cmd := &cobra.Command{
		Use:   "tui",
		Short: "Open an interactive Codex workspace on a fresh thread",
		Long: `Open a fresh Codex app-server thread using the separately installed codex executable.
Enter submits a text prompt. Ctrl+C requests interruption of the known active
turn; Ctrl+Q closes this workspace and its direct child process. Multiple text
turns share the fresh thread. The timeline shows the current turn only.

Installed Codex configuration, authentication, permissions, and persistence
remain in effect. This client does not override sandbox or approval settings.
Unsupported server requests, including approvals and input, are rejected and
reported. Use the native Codex client for those workflows. Turns may incur
charges. No existing history is read, and there is no automatic reconnect.

Requires terminal input and output. Install Codex separately from
https://github.com/openai/codex and make codex available on PATH.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			if cmd.Flags().Changed("output") || cmd.Root().PersistentFlags().Changed("output") {
				return fmt.Errorf("--output is not supported by openai codex tui; use codex exec for completed text or JSON")
			}
			if cmd.Flags().Changed("cd") && strings.TrimSpace(directory) == "" {
				return fmt.Errorf("cd must not be empty")
			}
			in, out, tty := terminalFiles(cmd)
			if !tty {
				return fmt.Errorf("openai codex tui requires terminal input and output; use openai codex exec for piped prompts")
			}
			ctx := cmd.Context()
			if *timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, *timeout)
				defer cancel()
			}
			session, err := codexsession.Open(ctx, codexsession.Options{Directory: directory, Model: *model})
			if err != nil {
				return fmt.Errorf("open Codex workspace: %w", err)
			}
			return codexworkspace.Run(codexworkspace.Config{Context: ctx, Backend: session, NoColor: noColor || os.Getenv("NO_COLOR") != "", ReducedMotion: reducedMotion}, tea.WithInput(in), tea.WithOutput(out))
		},
	}
	cmd.Flags().StringVarP(&directory, "cd", "C", "", "Working directory for the new Codex thread (default: current directory)")
	cmd.Flags().BoolVar(&noColor, "no-color", false, "Disable workspace colors (also respects nonempty NO_COLOR)")
	cmd.Flags().BoolVar(&reducedMotion, "reduced-motion", false, "Disable cosmetic activity animation")
	return cmd
}
