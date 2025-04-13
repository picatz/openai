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

var chatCommand = &cobra.Command{
	Use:   "chat",
	Short: "Chat with the OpenAI API",
	RunE: func(cmd *cobra.Command, args []string) error {
		codec := &storage.JSONCodec[string, chat.ReqRespPair]{}

		var opts = &pebble.Options{
			LoggerAndTracer: &stderrLoggerAndTracer{},
		}

		if useTemp, _ := cmd.Flags().GetBool("temporary"); useTemp {
			opts.FS = vfs.NewMem()
		}

		storageBackend, err := pebbleStorage.NewBackend(chat.DefaultCachePath, opts, codec)
		if err != nil {
			return fmt.Errorf("failed to create pebble backend: %w", err)
		}
		defer storageBackend.Close(cmd.Context())

		chatSession, restore, err := chat.NewSession(cmd.Context(), client, model, cmd.InOrStdin(), cmd.OutOrStdout(), storageBackend)
		if err != nil {
			return fmt.Errorf("failed to create chat session: %w", err)
		}
		defer restore()

		chatSession.Run(cmd.Context())

		return nil
	},
}

func init() {
	chatCommand.Flags().BoolP("temporary", "t", false, "Use a temporary in-memory chat storage backend")

	rootCmd.AddCommand(
		chatCommand,
	)
}
