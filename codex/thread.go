package codex

import (
	"context"
	"errors"
	"fmt"
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

// NewThread prepares a conversation; it does not start a Codex process until Run.
// Use ResumeThread to continue a persisted thread by ID.
func NewThread(options Options, threadOptions ThreadOptions) (*Thread, error) {
	executor, err := NewExec(options.CodexPathOverride)
	if err != nil {
		return nil, err
	}
	threadOptions.ConfigOverrides = append([]string(nil), threadOptions.ConfigOverrides...)
	threadOptions.AdditionalDirectories = append([]string(nil), threadOptions.AdditionalDirectories...)
	if threadOptions.NetworkAccessEnabled != nil {
		value := *threadOptions.NetworkAccessEnabled
		threadOptions.NetworkAccessEnabled = &value
	}
	return &Thread{exec: executor, options: options, threadOptions: threadOptions}, nil
}

// ResumeThread prepares a conversation using a previously returned ID. The CLI
// validates that the thread exists when the next turn starts.
func ResumeThread(id string, options Options, threadOptions ThreadOptions) (*Thread, error) {
	if id == "" {
		return nil, errors.New("thread ID must not be empty")
	}
	thread, err := NewThread(options, threadOptions)
	if err != nil {
		return nil, err
	}
	thread.id = id
	return thread, nil
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
	cancel   context.CancelFunc
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

// Close cancels the turn and waits for cleanup, including its output-schema file.
// Call it if you stop consuming Events before the channel closes.
func (s *StreamedTurn) Close() error {
	if s.cancel != nil {
		s.cancel()
	}
	err := s.Wait()
	if errors.Is(err, context.Canceled) {
		return nil
	}
	return err
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
		completed     bool
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
			completed = true
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
			return Turn{}, errors.Join(errors.New(turnFailure.Message), waitErr)
		}
		return Turn{}, errors.New(turnFailure.Message)
	}

	if waitErr != nil {
		return Turn{}, waitErr
	}

	if !completed {
		return Turn{}, fmt.Errorf("codex stream ended before turn.completed")
	}

	return Turn{Items: items, FinalResponse: finalResponse, Usage: usage}, nil
}

// RunText is a convenience wrapper for Run with a simple text prompt.
func (t *Thread) RunText(ctx context.Context, prompt string, turnOptions *TurnOptions) (Turn, error) {
	return t.Run(ctx, TextInput(prompt), turnOptions)
}

// RunStreamed streams events for a single agent turn. Callers should drain Events
// and then invoke Wait to retrieve any terminal error from the CLI. If they stop
// reading early, they must call Close or cancel ctx before Wait. Run consecutive
// turns serially; concurrent turns on the same conversation are not supported.
func (t *Thread) RunStreamed(ctx context.Context, input Input, turnOptions *TurnOptions) (*StreamedTurn, error) {
	return t.runStreamedInternal(ctx, input, turnOptions)
}

// RunStreamedText is a convenience wrapper for RunStreamed with a text prompt.
func (t *Thread) RunStreamedText(ctx context.Context, prompt string, turnOptions *TurnOptions) (*StreamedTurn, error) {
	return t.RunStreamed(ctx, TextInput(prompt), turnOptions)
}

func (t *Thread) runStreamedInternal(ctx context.Context, input Input, turnOptions *TurnOptions) (*StreamedTurn, error) {
	if t.exec == nil {
		return nil, errors.New("uninitialized thread: use NewThread or ResumeThread")
	}
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

	runCtx, cancel := context.WithCancel(ctx)
	stream, err := t.exec.Run(runCtx, Args{
		Input:                 prompt,
		BaseURL:               t.options.BaseURL,
		APIKey:                t.options.APIKey,
		ThreadID:              t.currentID(),
		Images:                images,
		Model:                 t.threadOptions.Model,
		SandboxMode:           t.threadOptions.SandboxMode,
		WorkingDirectory:      t.threadOptions.WorkingDirectory,
		SkipGitRepoCheck:      t.threadOptions.SkipGitRepoCheck,
		OutputSchemaFile:      schemaFile.Path(),
		ApprovalPolicy:        t.threadOptions.ApprovalPolicy,
		ModelReasoningEffort:  t.threadOptions.ModelReasoningEffort,
		AdditionalDirectories: t.threadOptions.AdditionalDirectories,
		NetworkAccessEnabled:  t.threadOptions.NetworkAccessEnabled,
		WebSearchMode:         t.threadOptions.WebSearchMode,
		ConfigOverrides:       t.threadOptions.ConfigOverrides,
	})
	if err != nil {
		cancel()
		_ = schemaFile.Cleanup()
		return nil, err
	}

	events := make(chan ThreadEvent)
	done := make(chan struct{})
	var runErr error
	go func() {
		defer close(done)
		defer close(events)
		defer cancel()
		defer schemaFile.Cleanup()
		for event, err := range EventStream(runCtx, stream) {
			if err != nil {
				runErr = err
				return
			}
			if event.Type == EventTypeThreadStarted {
				t.setID(event.ThreadID)
			}
			select {
			case events <- *event:
			case <-runCtx.Done():
				runErr = runCtx.Err()
				return
			}
		}
	}()
	return &StreamedTurn{
		Events: events,
		cancel: cancel,
		waitFn: func() error { <-done; return runErr },
	}, nil
}
