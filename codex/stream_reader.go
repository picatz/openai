package codex

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
)

// EventStream returns an iterator that yields ThreadEvents decoded from the provided ExecStream.
// The iterator yields (*ThreadEvent, nil) for each event, and (nil, error) if an error occurs.
//
// This is a convenience function for consuming events from a Codex execution stream. It
// decodes JSON-encoded events from the stream's stdout until EOF or an error occurs.
// The context can be used to cancel the iteration early, and the stream's Wait method is called
// at the end to ensure proper cleanup. The stream is also closed when iteration ends.
func EventStream(ctx context.Context, stream *ExecStream) iter.Seq2[*ThreadEvent, error] {
	decoder := json.NewDecoder(stream.Stdout())

	return func(yield func(*ThreadEvent, error) bool) {
		defer stream.Close()

		for ctx.Err() == nil {
			var event ThreadEvent
			if err := decoder.Decode(&event); err != nil {
				if err == io.EOF {
					break
				}
				if !yield(nil, fmt.Errorf("failed to decode codex event from stream: %w", err)) {
					return
				}
				continue
			}
			if !yield(&event, nil) {
				return
			}
		}

		if err := stream.Wait(); err != nil {
			yield(nil, err)
		}
	}
}
