//go:build !windows

package tui

import (
	"bytes"
	"context"
	"errors"
	"io"
	"reflect"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/creack/pty"
	"github.com/picatz/openai/internal/conversation"
	"golang.org/x/term"
)

func TestActiveProgramRestoresTerminal(t *testing.T) {
	for _, action := range []string{"quit", "context cancellation"} {
		t.Run(action, func(t *testing.T) {
			master, slave, err := pty.Open()
			if err != nil {
				t.Fatal(err)
			}
			defer master.Close()
			defer slave.Close()
			if err := pty.Setsize(master, &pty.Winsize{Rows: 24, Cols: 80}); err != nil {
				t.Fatal(err)
			}
			before, err := term.GetState(int(slave.Fd()))
			if err != nil {
				t.Fatal(err)
			}
			var output bytes.Buffer
			drained := make(chan struct{})
			go func() { defer close(drained); _, _ = io.Copy(&output, master) }()
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			started, stopped := make(chan struct{}), make(chan struct{})
			m := testModel()
			m.cfg.Context = ctx

			m.cfg.Backend = conversation.BackendFunc(func(ctx context.Context, req conversation.Request, emit func(string) error) (conversation.Result, error) {
				defer close(stopped)
				if err := emit("partial reply"); err != nil {
					return conversation.Result{}, err
				}
				close(started)
				<-ctx.Done()
				return conversation.Result{}, ctx.Err()
			})
			done := make(chan error, 1)
			go func() {
				done <- Run(m.cfg, tea.WithInput(slave), tea.WithOutput(slave), tea.WithEnvironment([]string{"TERM=xterm-256color"}), tea.WithoutSignalHandler())
			}()
			// Wait for raw mode before typing, so the PTY does not echo or buffer input.
			for {
				state, err := term.GetState(int(slave.Fd()))
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before, state) {
					break
				}
				select {
				case err := <-done:
					t.Fatalf("program exited before raw mode: %v", err)
				case <-ctx.Done():
					t.Fatal("raw mode timeout")
				case <-time.After(time.Millisecond):
				}
			}
			if _, err := master.Write([]byte("synthetic prompt\r")); err != nil {
				t.Fatal(err)
			}
			select {
			case <-started:
			case <-ctx.Done():
				t.Fatal("backend did not start")
			}
			during, err := term.GetState(int(slave.Fd()))
			if err != nil {
				t.Fatal(err)
			}
			if reflect.DeepEqual(before, during) {
				t.Fatal("program never entered raw mode")
			}
			if action == "quit" {
				if _, err := master.Write([]byte{17}); err != nil {
					t.Fatal(err)
				}
			} else {
				cancel()
				cancel()
			}
			select {
			case err := <-done:
				if action == "quit" && err != nil {
					t.Fatal(err)
				}
				if action != "quit" && !errors.Is(err, context.Canceled) {
					t.Fatalf("cancellation error = %v", err)
				}
			case <-time.After(3 * time.Second):
				t.Fatal("program did not exit")
			}
			select {
			case <-stopped:
			case <-time.After(3 * time.Second):
				t.Fatal("backend survived program shutdown")
			}
			after, err := term.GetState(int(slave.Fd()))
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before, after) {
				t.Fatal("terminal state was not restored after active stream")
			}
			// Closing the slave ends the master read. Only inspect output after
			// joining its writer, so the capture itself is race-free.
			if err := slave.Close(); err != nil {
				t.Fatal(err)
			}
			select {
			case <-drained:
			case <-time.After(3 * time.Second):
				t.Fatal("terminal output did not drain")
			}
			for _, sequence := range []string{"\x1b[?1049h", "\x1b[?1049l", "\x1b[?25h"} {
				if !bytes.Contains(output.Bytes(), []byte(sequence)) {
					t.Errorf("terminal output missing restoration sequence %q", sequence)
				}
			}
		})
	}
}
