//go:build !windows

package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"reflect"
	"syscall"
	"testing"
	"time"

	"github.com/creack/pty"
	"golang.org/x/term"
)

func TestIdleTerminalCancellationRestoresState(t *testing.T) {
	for _, mode := range []string{"responses", "chat"} {
		for _, action := range []string{"SIGTERM", "Ctrl-C", "Ctrl-D"} {
			t.Run(mode+"/"+action, func(t *testing.T) {
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
				executable, err := os.Executable()
				if err != nil {
					t.Fatal(err)
				}
				ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
				defer cancel()
				cmd := exec.CommandContext(ctx, executable)
				// No credentials, no prompt, no real API. Keep any cache in a test directory.
				cmd.Env = []string{"OPENAI_CLI_PTY_HELPER=1", "OPENAI_CLI_PTY_MODE=" + mode, "TERM=xterm-256color", "HOME=" + t.TempDir()}
				cmd.Stdin = slave
				cmd.Stdout = slave
				cmd.Stderr = slave
				if err := cmd.Start(); err != nil {
					t.Fatal(err)
				}
				done := make(chan error, 1)
				go func() { done <- cmd.Wait() }()
				defer func() { cmd.Process.Kill() }()
				deadline := time.Now().Add(5 * time.Second)
				for {
					state, err := term.GetState(int(slave.Fd()))
					if err != nil {
						t.Fatal(err)
					}
					if !reflect.DeepEqual(state, before) {
						break
					}
					if time.Now().After(deadline) {
						t.Fatal("terminal never entered raw mode")
					}
					time.Sleep(5 * time.Millisecond)
				}
				wantCode := 130
				switch action {
				case "SIGTERM":
					err = cmd.Process.Signal(syscall.SIGTERM)
				case "Ctrl-C":
					_, err = master.Write([]byte{3})
				case "Ctrl-D":
					_, err = master.Write([]byte{4})
					wantCode = 0
				}
				if err != nil {
					t.Fatal(err)
				}
				select {
				case err := <-done:
					code := 0
					if err != nil {
						var exit *exec.ExitError
						if !errors.As(err, &exit) {
							t.Fatal(err)
						}
						code = exit.ExitCode()
					}
					if code != wantCode {
						t.Errorf("exit=%d want=%d", code, wantCode)
					}
				case <-ctx.Done():
					t.Fatal("idle terminal did not respond to cancellation")
				}
				after, err := term.GetState(int(slave.Fd()))
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(before, after) {
					t.Error("terminal state was not restored")
				}
			})
		}
	}
}
