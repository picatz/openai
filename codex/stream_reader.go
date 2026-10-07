package codex

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
)

// EventStream decodes one event per nonblank JSONL line. Malformed input is a
// terminal error, not a retryable record. Lines are not limited to 64 KiB.
// The iterator owns stream: completion, a decode error, cancellation (including
// a context different from Exec.Run's), or an early break stops and reaps it.
func EventStream(ctx context.Context, stream *ExecStream) iter.Seq2[*ThreadEvent, error] {
	return func(yield func(*ThreadEvent, error) bool) {
		defer stream.Close()
		stop := context.AfterFunc(ctx, func() { _ = stream.Close() })
		defer stop()
		reader := bufio.NewReader(stream.Stdout())
		for {
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}
			line, readErr := reader.ReadBytes('\n')
			if err := ctx.Err(); err != nil {
				yield(nil, err)
				return
			}
			if line = bytes.TrimSpace(line); len(line) != 0 {
				var event ThreadEvent
				if err := json.Unmarshal(line, &event); err != nil {
					yield(nil, fmt.Errorf("parse codex event: %w", err))
					return
				}
				if !yield(&event, nil) {
					return
				}
			}
			if readErr != nil {
				if readErr != io.EOF {
					yield(nil, fmt.Errorf("read codex output: %w", readErr))
					return
				}
				break
			}
		}
		if err := stream.Wait(); err != nil {
			yield(nil, err)
		}
	}
}
