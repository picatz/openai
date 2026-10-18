package tui

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

func (r failedInput) Read([]byte) (int, error) { return 0, r.err }

func TestRunCanceledBeforeStart(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	cancel()
	cfg := testModel().cfg
	cfg.Context = ctx
	if err := Run(cfg); !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v", err)
	}
}

func TestRunStopsWatcherOnInputError(t *testing.T) {
	sentinel := errors.New("synthetic input failure")
	cfg := testModel().cfg
	done := make(chan error, 1)
	go func() {
		done <- Run(cfg, tea.WithInput(failedInput{sentinel}), tea.WithOutput(io.Discard), tea.WithoutRenderer(), tea.WithoutSignalHandler())
	}()
	select {
	case err := <-done:
		if !errors.Is(err, sentinel) {
			t.Fatalf("error = %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("input failure left cancellation watcher blocked")
	}
}

func TestRunWithoutContextCanQuit(t *testing.T) {
	cfg := testModel().cfg
	cfg.Context = nil
	done := make(chan error, 1)
	go func() {
		done <- Run(cfg, tea.WithInput(strings.NewReader("\x11")), tea.WithOutput(io.Discard), tea.WithoutRenderer(), tea.WithoutSignalHandler())
	}()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("normal quit left cancellation watcher blocked")
	}
}
