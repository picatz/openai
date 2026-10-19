package codexworkspace

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
)

type failedInput struct{ err error }

func (f failedInput) Read([]byte) (int, error) { return 0, f.err }

func TestRunOwnsClose(t *testing.T) {
	for _, mode := range []string{"quit", "pre-canceled", "input error", "override context"} {
		t.Run(mode, func(t *testing.T) {
			b := newFake()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			cfg := Config{Context: ctx, Backend: b}
			options := []tea.ProgramOption{tea.WithInput(strings.NewReader("\x11")), tea.WithOutput(io.Discard), tea.WithoutRenderer(), tea.WithoutSignalHandler()}
			var want error
			switch mode {
			case "pre-canceled":
				cancel()
				want = context.Canceled
			case "input error":
				want = errors.New("synthetic input failure")
				options = append(options, tea.WithInput(failedInput{want}))
			case "override context":
				canceled, stop := context.WithCancel(t.Context())
				stop()
				options = append(options, tea.WithContext(canceled))
			}
			done := make(chan error, 1)
			go func() { done <- Run(cfg, options...) }()
			select {
			case err := <-done:
				if !errors.Is(err, want) {
					t.Fatalf("error = %v; want %v", err, want)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("Run blocked")
			}
			b.mu.Lock()
			defer b.mu.Unlock()
			if b.closes != 1 {
				t.Fatalf("Close called %d times", b.closes)
			}
		})
	}
}
