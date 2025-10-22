package codex

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
)

// Thread represents a conversation with an agent. A thread can span multiple turns.
type Thread struct {
	exec          *Exec
	options       Options
	threadOptions ThreadOptions

	mu sync.RWMutex
	id string
}

// ID returns the identifier of the thread once assigned by the codex backend.
func (t *Thread) ID() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.id
}

func (t *Thread) setID(id string) {
	if id == "" {
		return
	}
	t.mu.Lock()
	t.id = id
	t.mu.Unlock()
}

func (t *Thread) currentID() string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.id
}

// Turn contains the result of a completed agent turn.
type Turn struct {
	// Items are the completed thread items emitted during the turn.
	Items []ThreadItem
	// FinalResponse is the assistant's last agent_message, when present.
	FinalResponse string
	// Usage reports token consumption for the turn. A nil value indicates the CLI
	// did not emit usage information.
	Usage *Usage
}

// RunResult aliases Turn for parity with the TypeScript SDK.
type RunResult = Turn

// StreamedTurn streams thread events as they are produced during a run.
type StreamedTurn struct {
	// Events yields parsed events in the order emitted by the CLI.
	Events   <-chan ThreadEvent
	waitFn   func() error
	waitOnce sync.Once
	waitErr  error
}

// Wait blocks until the underlying run completes and returns the terminal error, if any.
func (s *StreamedTurn) Wait() error {
	s.waitOnce.Do(func() {
		if s.waitFn != nil {
			s.waitErr = s.waitFn()
		}
	})
	return s.waitErr
}

// RunStreamedResult aliases StreamedTurn for parity with the TypeScript SDK.
type RunStreamedResult = StreamedTurn

// Run executes a complete agent turn with the provided input and returns its result.
// The call blocks until the CLI exits or the context is cancelled.
func (t *Thread) Run(ctx context.Context, input Input, turnOptions *TurnOptions) (Turn, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	streamed, err := t.runStreamedInternal(ctx, input, turnOptions)
	if err != nil {
		return Turn{}, err
	}

	var (
		items         []ThreadItem
		finalResponse string
		usage         *Usage
		turnFailure   *ThreadError
	)

loop:
	for event := range streamed.Events {
		switch event.Type {
		case EventTypeItemCompleted:
			if event.Item != nil {
				if msg, ok := event.Item.(*AgentMessageItem); ok {
					finalResponse = msg.Text
				}
				items = append(items, event.Item)
			}
		case EventTypeTurnCompleted:
			usage = event.Usage
		case EventTypeTurnFailed:
			if event.Error != nil {
				turnFailure = event.Error
			} else {
				turnFailure = &ThreadError{Message: "turn failed"}
			}
			cancel()
			break loop
		}
	}

	waitErr := streamed.Wait()

	if turnFailure != nil {
		if waitErr != nil && !errors.Is(waitErr, context.Canceled) {
			return Turn{}, waitErr
		}
		return Turn{}, errors.New(turnFailure.Message)
	}

	if waitErr != nil {
		return Turn{}, waitErr
	}

	return Turn{Items: items, FinalResponse: finalResponse, Usage: usage}, nil
}

// RunText is a convenience wrapper for Run with a simple text prompt.
func (t *Thread) RunText(ctx context.Context, prompt string, turnOptions *TurnOptions) (Turn, error) {
	return t.Run(ctx, TextInput(prompt), turnOptions)
}

// RunStreamed streams events for a single agent turn. Callers should drain Events
// and then invoke Wait to retrieve any terminal error from the CLI.
func (t *Thread) RunStreamed(ctx context.Context, input Input, turnOptions *TurnOptions) (*StreamedTurn, error) {
	return t.runStreamedInternal(ctx, input, turnOptions)
}

// RunStreamedText is a convenience wrapper for RunStreamed with a text prompt.
func (t *Thread) RunStreamedText(ctx context.Context, prompt string, turnOptions *TurnOptions) (*StreamedTurn, error) {
	return t.RunStreamed(ctx, TextInput(prompt), turnOptions)
}

func (t *Thread) runStreamedInternal(ctx context.Context, input Input, turnOptions *TurnOptions) (*StreamedTurn, error) {
	if turnOptions == nil {
		turnOptions = &TurnOptions{}
	}

	schemaFile, err := createOutputSchemaFile(turnOptions.OutputSchema)
	if err != nil {
		return nil, err
	}

	prompt, images, err := normalizeInput(input)
	if err != nil {
		_ = schemaFile.Cleanup()
		return nil, err
	}

	stream, err := t.exec.Run(ctx, Args{
		Input:            prompt,
		BaseURL:          t.options.BaseURL,
		APIKey:           t.options.APIKey,
		ThreadID:         t.currentID(),
		Images:           images,
		Model:            t.threadOptions.Model,
		SandboxMode:      t.threadOptions.SandboxMode,
		WorkingDirectory: t.threadOptions.WorkingDirectory,
		SkipGitRepoCheck: t.threadOptions.SkipGitRepoCheck,
		OutputSchemaFile: schemaFile.Path(),
	})
	if err != nil {
		_ = schemaFile.Cleanup()
		return nil, err
	}

	events := make(chan ThreadEvent)
	errCh := make(chan error, 1)

	go func() {
		defer close(events)
		stdout := stream.Stdout()
		defer stdout.Close()
		defer func() {
			_ = schemaFile.Cleanup()
		}()

		reader := bufio.NewReader(stdout)
		var runErr error

		for {
			if ctxErr := ctx.Err(); ctxErr != nil {
				runErr = ctxErr
				break
			}

			line, readErr := reader.ReadBytes('\n')
			trimmed := bytes.TrimSpace(line)
			if len(trimmed) > 0 {
				var event ThreadEvent
				if err := json.Unmarshal(trimmed, &event); err != nil {
					runErr = fmt.Errorf("parse codex event: %w", err)
					break
				}

				if event.Type == EventTypeThreadStarted && event.ThreadID != "" {
					t.setID(event.ThreadID)
				}

				select {
				case events <- event:
				case <-ctx.Done():
					runErr = ctx.Err()
					break
				}
			}

			if readErr != nil {
				if errors.Is(readErr, io.EOF) {
					break
				}
				if runErr == nil {
					runErr = fmt.Errorf("read codex output: %w", readErr)
				}
				break
			}

			if runErr != nil {
				break
			}
		}

		waitErr := stream.Wait()
		if runErr == nil {
			runErr = waitErr
		} else if waitErr != nil && !errors.Is(runErr, waitErr) {
			runErr = fmt.Errorf("%w; wait error: %v", runErr, waitErr)
		}

		errCh <- runErr
	}()

	return &StreamedTurn{
		Events: events,
		waitFn: func() error {
			if err := <-errCh; err != nil {
				return err
			}
			return nil
		},
	}, nil
}
