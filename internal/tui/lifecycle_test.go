package tui

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/picatz/openai/internal/conversation"
)

// Run the streaming command, rather than manufacturing streamMsg values, so
// these tests cover cancellation of the producer and its bounded event queue.
func beginStream(t *testing.T, m Model) (Model, tea.Msg) {
	t.Helper()
	m.input.SetValue("original prompt")
	m, cmd := update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	if cmd == nil {
		t.Fatal("submission did not schedule streaming")
	}
	batch, ok := cmd().(tea.BatchMsg)
	if !ok || len(batch) != 2 {
		t.Fatal("expected spinner and stream commands")
	}
	first := make(chan tea.Msg, 1)
	go func() { first <- batch[1]() }()
	select {
	case msg := <-first:
		return m, msg
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not produce an event")
		return m, nil
	}
}

func TestCancellationReleasesBackpressuredStream(t *testing.T) {
	for _, key := range []rune{'c', 'q'} {
		t.Run(string(key), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			blocked, exited := make(chan struct{}), make(chan error, 1)
			m := testModel()
			m.cfg.Context = ctx
			m.cfg.Backend = conversation.BackendFunc(func(ctx context.Context, req conversation.Request, emit func(string) error) (conversation.Result, error) {
				for i := 0; ; i++ {
					if i == 17 {
						close(blocked)
					}
					if err := emit("chunk"); err != nil {
						exited <- err
						return conversation.Result{}, err
					}
				}
			})
			m, first := beginStream(t, m)
			generation := m.generation
			m, _ = update(m, first)
			select {
			case <-blocked:
			case <-time.After(3 * time.Second):
				t.Fatal("producer did not fill the event buffer")
			}
			m.input.SetValue("next draft")
			m, cmd := update(m, ctrl(key))
			if key == 'q' {
				if cmd == nil {
					t.Fatal("quit not scheduled")
				}
				if _, ok := cmd().(tea.QuitMsg); !ok {
					t.Fatal("expected quit")
				}
			} else if cmd != nil {
				t.Fatal("cancel unexpectedly quit")
			}
			select {
			case err := <-exited:
				if !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("canceled producer remained blocked")
			}
			if m.busy || m.cancel != nil || m.input.Value() != "next draft" || len(m.session.Messages) != 0 {
				t.Fatal("cancel lost the new draft or committed history")
			}
			// Drain events already queued before cancellation; none may resurrect the request.
			drained := false
			for !drained {
				select {
				case event, ok := <-m.events:
					if !ok {
						drained = true
						break
					}
					m, _ = update(m, event)
				case <-time.After(3 * time.Second):
					t.Fatal("canceled stream did not close its event queue")
				}
			}
			if m.generation == generation || m.busy || len(m.session.Messages) != 0 {
				t.Fatal("queued events revived the canceled request")
			}
		})
	}
}

func TestStreamFailureCanStartFreshSession(t *testing.T) {
	m := testModel()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	m.cfg.Context = ctx
	m.cfg.Backend = conversation.BackendFunc(func(ctx context.Context, req conversation.Request, emit func(string) error) (conversation.Result, error) {
		if err := emit("unfinished"); err != nil {
			return conversation.Result{}, err
		}
		return conversation.Result{}, errors.New("synthetic stream failure")
	})
	m, first := beginStream(t, m)
	oldGeneration, oldID := m.generation, m.session.ID
	m, next := update(m, first)
	if next == nil {
		t.Fatal("stream did not schedule next event")
	}
	event := make(chan tea.Msg, 1)
	go func() { event <- next() }()
	select {
	case msg := <-event:
		m, _ = update(m, msg)
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not report its error")
	}
	if m.busy || m.cancel != nil || len(m.session.Messages) != 0 || m.input.Value() != "original prompt" || !strings.Contains(m.status, "synthetic stream failure") {
		t.Fatal("stream failure did not restore the draft")
	}
	m, _ = update(m, ctrl('n'))
	if m.session.ID == oldID || m.partial != "" || m.pending != "" || m.input.Value() != "" {
		t.Fatal("new session retained failed request state")
	}
	received := make(chan conversation.Request, 1)
	m.cfg.Backend = conversation.BackendFunc(func(ctx context.Context, req conversation.Request, emit func(string) error) (conversation.Result, error) {
		received <- req
		return conversation.Result{Text: "fresh reply"}, nil
	})
	m, done := beginStream(t, m)
	m, _ = update(m, streamMsg{generation: oldGeneration, done: true, result: conversation.Result{Text: "stale reply"}})
	if !m.busy || len(m.session.Messages) != 0 {
		t.Fatal("old request modified the fresh session")
	}
	m, _ = update(m, done)
	var req conversation.Request
	select {
	case req = <-received:
	case <-time.After(3 * time.Second):
		t.Fatal("new session did not reach backend")
	}
	if len(req.Messages) != 1 || len(m.session.Messages) != 2 || m.session.Messages[1].Content != "fresh reply" {
		t.Fatal("fresh session replayed failed history")
	}
}

func TestEmptyInputAndResizeDoNotSubmit(t *testing.T) {
	m := testModel()
	m.cfg.Backend = conversation.BackendFunc(func(context.Context, conversation.Request, func(string) error) (conversation.Result, error) {
		t.Error("empty input called backend")
		return conversation.Result{}, nil
	})
	for _, text := range []string{"", " \t\n "} {
		m.input.SetValue(text)
		draft := m.input.Value()
		for _, size := range [][2]int{{0, 0}, {1, 1}, {80, 24}, {160, 60}, {30, 12}} {
			m, _ = update(m, tea.WindowSizeMsg{Width: size[0], Height: size[1]})
			_ = m.View()
			var cmd tea.Cmd
			m, cmd = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
			if cmd != nil || m.busy || m.generation != 0 || m.input.Value() != draft {
				t.Fatal("empty submit or resize changed request state")
			}
		}
	}
}
