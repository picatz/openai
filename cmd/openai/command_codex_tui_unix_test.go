//go:build !windows

package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"
)

type codexTUIOutput struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (o *codexTUIOutput) Write(p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.Write(p)
}
func (o *codexTUIOutput) String() string { o.mu.Lock(); defer o.mu.Unlock(); return o.buf.String() }

func TestCodexTUICommandSyntheticEndToEnd(t *testing.T) {
	installSyntheticCodex(t)
	t.Setenv("OPENAI_CODEX_TUI_FAKE", "1")
	capture := filepath.Join(t.TempDir(), "requests.jsonl")
	t.Setenv("OPENAI_CODEX_TUI_CAPTURE", capture)
	t.Setenv("TERM", "xterm-256color")
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
	ctx, cancel := context.WithTimeout(t.Context(), 8*time.Second)
	defer cancel()
	output := &codexTUIOutput{}
	drained := make(chan struct{})
	go func() { defer close(drained); _, _ = io.Copy(output, master) }()
	directory := t.TempDir()
	done := make(chan int, 1)
	go func() {
		done <- run(ctx, []string{"codex", "tui", "--cd", directory, "--model", "fake-model", "--no-color"}, slave, slave, slave)
	}()
	wait := func(label string, ready func() bool) {
		t.Helper()
		for !ready() {
			select {
			case code := <-done:
				t.Fatalf("%s: exited %d; %s", label, code, output.String())
			case <-ctx.Done():
				t.Fatalf("%s timed out; %s", label, output.String())
			case <-time.After(time.Millisecond):
			}
		}
	}
	wait("raw mode", func() bool {
		state, e := term.GetState(int(slave.Fd()))
		return e == nil && !reflect.DeepEqual(before, state)
	})
	if _, err := master.Write([]byte("fake typed prompt\r")); err != nil {
		t.Fatal(err)
	}
	// A raw PTY stream is a sequence of cursor-positioned updates, not the
	// final screen. The fake final snapshot uses wholly distinct bytes from
	// its streaming delta so this marker cannot be rendered as a reused prefix.
	wait("actual app-server result in workspace", func() bool { return strings.Contains(output.String(), "FINAL_RESULT_8427") })
	if _, err := master.Write([]byte{17}); err != nil {
		t.Fatal(err)
	}
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("exit=%d; %s", code, output.String())
		}
	case <-ctx.Done():
		t.Fatal("quit timed out")
	}
	after, err := term.GetState(int(slave.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatal("terminal not restored")
	}
	if err := slave.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-drained:
	case <-ctx.Done():
		t.Fatal("output not joined")
	}
	requests, err := os.ReadFile(capture)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(requests), "fake typed prompt") || !strings.Contains(string(requests), "fake-model") {
		t.Fatal(string(requests))
	}
	for _, bad := range []string{"approvalPolicy", "sandbox", "experimentalApi", "thread/list", "thread/resume"} {
		if strings.Contains(string(requests), bad) {
			t.Fatalf("unexpected transmitted override/history request %q", bad)
		}
	}
}
