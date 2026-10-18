//go:build !windows

package codexworkspace

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/creack/pty"
	"golang.org/x/term"
)

type ptyBackend struct {
	*fakeBackend
	promptsSent   chan string
	interruptSent chan struct{}
	blockStart    bool
	startStopped  chan struct{}
}

func (b *ptyBackend) StartTurn(ctx context.Context, prompt string) error {
	b.promptsSent <- prompt
	if b.blockStart {
		defer close(b.startStopped)
		<-ctx.Done()
		return ctx.Err()
	}
	return b.fakeBackend.StartTurn(ctx, prompt)
}
func (b *ptyBackend) Interrupt(ctx context.Context) error {
	b.interruptSent <- struct{}{}
	return b.fakeBackend.Interrupt(ctx)
}

type lockedOutput struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (o *lockedOutput) Write(p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.Write(p)
}
func (o *lockedOutput) text() string { o.mu.Lock(); defer o.mu.Unlock(); return o.buf.String() }

type terminalHarness struct {
	t             *testing.T
	master, slave *os.File
	before        *term.State
	output        *lockedOutput
	backend       *ptyBackend
	ctx           context.Context
	cancel        context.CancelFunc
	done          chan error
	drained       chan struct{}
}

func openTerminal(t *testing.T, blockStart bool) *terminalHarness {
	t.Helper()
	master, slave, err := pty.Open()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { master.Close(); slave.Close() })
	if err := pty.Setsize(master, &pty.Winsize{Rows: 24, Cols: 80}); err != nil {
		t.Fatal(err)
	}
	before, err := term.GetState(int(slave.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 8*time.Second)
	t.Cleanup(cancel)
	h := &terminalHarness{t: t, master: master, slave: slave, before: before, ctx: ctx, cancel: cancel, output: &lockedOutput{}, done: make(chan error, 1), drained: make(chan struct{})}
	h.backend = &ptyBackend{fakeBackend: newFake(), promptsSent: make(chan string, 8), interruptSent: make(chan struct{}, 8), blockStart: blockStart, startStopped: make(chan struct{})}
	go func() { defer close(h.drained); _, _ = io.Copy(h.output, master) }()
	go func() {
		h.done <- Run(Config{Context: ctx, Backend: h.backend, NoColor: true, ReducedMotion: true}, tea.WithInput(slave), tea.WithOutput(slave), tea.WithEnvironment([]string{"TERM=xterm-256color"}), tea.WithoutSignalHandler())
	}()
	h.wait("raw mode", func() bool {
		state, err := term.GetState(int(slave.Fd()))
		return err == nil && !reflect.DeepEqual(before, state)
	})
	h.waitNext(1)
	return h
}
func (h *terminalHarness) wait(label string, predicate func() bool) {
	h.t.Helper()
	for !predicate() {
		select {
		case <-h.ctx.Done():
			h.t.Fatalf("timed out waiting for %s; output: %q", label, h.output.text())
		case err := <-h.done:
			h.t.Fatalf("program exited waiting for %s: %v", label, err)
		case <-time.After(time.Millisecond):
		}
	}
}
func (h *terminalHarness) waitNext(n int) {
	h.t.Helper()
	h.wait("ordered event consumer", func() bool { h.backend.mu.Lock(); defer h.backend.mu.Unlock(); return h.backend.nextCalls >= n })
}
func (h *terminalHarness) write(s string) {
	h.t.Helper()
	if _, err := h.master.Write([]byte(s)); err != nil {
		h.t.Fatal(err)
	}
}
func (h *terminalHarness) waitOutput(s string) {
	h.t.Helper()
	h.wait(s, func() bool { return strings.Contains(ansi.Strip(h.output.text()), s) })
}
func (h *terminalHarness) prompt(want string) {
	h.t.Helper()
	select {
	case got := <-h.backend.promptsSent:
		if got != want {
			h.t.Fatalf("prompt = %q; want %q", got, want)
		}
	case <-h.ctx.Done():
		h.t.Fatal("prompt not submitted")
	}
}
func (h *terminalHarness) noPrompt() {
	h.t.Helper()
	select {
	case prompt := <-h.backend.promptsSent:
		h.t.Fatalf("unexpected submitted prompt %q", prompt)
	default:
	}
}
func (h *terminalHarness) active() {
	h.t.Helper()
	h.write("synthetic prompt\r")
	h.prompt("synthetic prompt")
	h.backend.events <- nextMsg{event: event("begin", "turn", 1, 1)}
	h.waitNext(2)
	h.waitOutput("running")
}
func (h *terminalHarness) stopped(want error) {
	h.t.Helper()
	select {
	case err := <-h.done:
		if !errors.Is(err, want) {
			h.t.Fatalf("Run error = %v; want %v", err, want)
		}
	case <-time.After(3 * time.Second):
		h.t.Fatal("program did not exit")
	}
	after, err := term.GetState(int(h.slave.Fd()))
	if err != nil {
		h.t.Fatal(err)
	}
	if !reflect.DeepEqual(h.before, after) {
		h.t.Fatal("terminal state not restored")
	}
	h.backend.mu.Lock()
	closes := h.backend.closes
	h.backend.mu.Unlock()
	if closes != 1 {
		h.t.Fatalf("backend Close calls = %d", closes)
	}
	if err := h.slave.Close(); err != nil {
		h.t.Fatal(err)
	}
	select {
	case <-h.drained:
	case <-time.After(3 * time.Second):
		h.t.Fatal("output did not drain")
	}
	output := h.output.text()
	for _, seq := range []string{"\x1b[?1049h", "\x1b[?1049l", "\x1b[?25h", "\x1b[?2004h", "\x1b[?2004l"} {
		if !strings.Contains(output, seq) {
			h.t.Errorf("missing terminal restoration sequence %q", seq)
		}
	}
}

func TestPTYActiveShutdownAndTerminalRestore(t *testing.T) {
	for _, action := range []string{"quit", "context", "transport EOF"} {
		t.Run(action, func(t *testing.T) {
			h := openTerminal(t, false)
			h.active()
			var want error
			switch action {
			case "quit":
				h.write("\x11")
			case "context":
				want = context.Canceled
				h.cancel()
				h.cancel()
			case "transport EOF":
				want = io.ErrUnexpectedEOF
				h.backend.events <- nextMsg{err: io.EOF}
				h.waitOutput("outcome unknown")
				h.write("\x11")
			}
			h.stopped(want)
		})
	}
}

func TestPTYPasteRepeatedTurnsAndInterrupt(t *testing.T) {
	h := openTerminal(t, false)
	h.write("\x1b[200~first\nsecond\x1b[201~\nthird")
	h.waitOutput("third")
	h.noPrompt()
	h.write("\r")
	h.prompt("first\nsecond\nthird")
	h.backend.events <- nextMsg{event: event("begin", "first", 1, 1)}
	h.waitNext(2)
	h.waitOutput("running")
	h.write("\x03\x03\x03")
	select {
	case <-h.backend.interruptSent:
	case <-h.ctx.Done():
		t.Fatal("Ctrl+C did not interrupt")
	}
	h.waitOutput("awaiting final turn state")
	h.write("next\r")
	h.waitOutput("next")
	h.noPrompt()
	select {
	case <-h.backend.interruptSent:
		t.Fatal("repeated Ctrl+C sent duplicate interrupt")
	default:
	}
	e := event("end", "first", 1, 2)
	e.Status = "interrupted"
	h.backend.events <- nextMsg{event: e}
	h.waitNext(3)
	h.waitOutput("Turn interrupted")
	h.write("\r")
	h.prompt("next")
	h.backend.events <- nextMsg{event: event("begin", "second", 2, 1)}
	e = event("end", "second", 2, 2)
	e.Status = "completed"
	h.backend.events <- nextMsg{event: e}
	h.waitNext(5)
	h.waitOutput("completed")
	h.write("\x11")
	h.stopped(nil)
}

func TestPTYPendingStartQuitDoesNotInterrupt(t *testing.T) {
	h := openTerminal(t, true)
	h.write("pending\r")
	h.prompt("pending")
	h.write("\x03\x03")
	h.waitOutput("outcome uncertain")
	select {
	case <-h.backend.interruptSent:
		t.Fatal("pending start was remotely interrupted")
	default:
	}
	h.write("\x11")
	h.stopped(nil)
	select {
	case <-h.backend.startStopped:
	case <-time.After(3 * time.Second):
		t.Fatal("pending start survived local shutdown")
	}
}
