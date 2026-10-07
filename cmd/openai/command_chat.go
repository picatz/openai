package main

import (
	"context"
	"fmt"
	"os"

	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/picatz/openai/internal/chat"
	"github.com/picatz/openai/internal/chat/storage"
	pebbleStorage "github.com/picatz/openai/internal/chat/storage/pebble"
	"github.com/picatz/openai/internal/conversation"
	"github.com/spf13/cobra"
)

type stderrLoggerAndTracer struct{}

func (l *stderrLoggerAndTracer) Infof(format string, args ...interface{}) {}
func (l *stderrLoggerAndTracer) Fatalf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format, args...)
	os.Exit(1)
}

func (l *stderrLoggerAndTracer) Eventf(ctx context.Context, format string, args ...interface{}) {}
func (l *stderrLoggerAndTracer) IsTracingEnabled(ctx context.Context) bool {
	return false
}

func newChatCommand(app *application) *cobra.Command {
	chatCommand := &cobra.Command{
		Use:   "chat [prompt]",
		Args:  cobra.ArbitraryArgs,
		Short: "Chat with the OpenAI API",
		RunE: func(cmd *cobra.Command, args []string) error {
			if !app.legacy {
				if _, _, tty := terminalFiles(cmd); tty && len(args) == 0 && app.output == "text" {
					return app.runTUI(cmd, conversation.Chat)
				}
				return app.createChat(cmd, args)
			}
			if len(args) != 0 {
				return fmt.Errorf("legacy chat does not accept a prompt argument")
			}
			if app.sessionID != "" {
				return fmt.Errorf("--session is not supported by the legacy chat")
			}

			codec := &storage.JSONCodec[string, chat.ReqRespPair]{}

			var opts = &pebble.Options{
				LoggerAndTracer:    &stderrLoggerAndTracer{},
				FormatMajorVersion: pebble.FormatVirtualSSTables,
			}

			if useTemp, _ := cmd.Flags().GetBool("temporary"); useTemp {
				opts.FS = vfs.NewMem()
			}

			storageBackend, err := pebbleStorage.NewBackend(chat.DefaultCachePath, opts, codec)
			if err != nil {
				return fmt.Errorf("failed to create pebble backend: %w", err)
			}
			defer storageBackend.Close(cmd.Context())

			chatSession, restore, err := chat.NewSession(cmd.Context(), &app.client, app.model, cmd.InOrStdin(), cmd.OutOrStdout(), storageBackend)
			if err != nil {
				return fmt.Errorf("failed to create chat session: %w", err)
			}
			defer restore()

			return chatSession.Run(cmd.Context())
		},
	}

	return chatCommand
}
