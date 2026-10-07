package codex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

// The helper is this test binary, never a real Codex installation.
func TestMain(m *testing.M) {
	if os.Getenv("CODEX_TEST_HELPER") == "1" && len(os.Args) > 1 && os.Args[1] == "exec" {
		if os.Getenv("CODEX_TEST_MODE") == "blocked-stdin" {
			fmt.Println(`{"type":"turn.started"}`)
			time.Sleep(time.Hour)
		}
		input, _ := io.ReadAll(os.Stdin)
		if file := os.Getenv("CODEX_TEST_CAPTURE"); file != "" {
			capture := struct {
				Args   []string
				Input  string
				Schema json.RawMessage
			}{Args: os.Args[1:], Input: string(input)}
			for i, arg := range os.Args {
				if arg == "--output-schema" && i+1 < len(os.Args) {
					capture.Schema, _ = os.ReadFile(os.Args[i+1])
				}
			}
			data, _ := json.Marshal(capture)
			if err := os.WriteFile(file, data, 0600); err != nil {
				os.Exit(9)
			}
		}
		switch os.Getenv("CODEX_TEST_MODE") {
		case "failure":
			fmt.Fprintln(os.Stderr, "synthetic failure")
			os.Exit(7)
		case "wait":
			fmt.Println(`{"type":"turn.started"}`)
			time.Sleep(time.Hour)
		case "malformed":
			fmt.Println(`{"type":`)
			time.Sleep(time.Hour)
		case "failed-turn":
			fmt.Println(`{"type":"turn.failed","error":{"message":"synthetic turn failure"}}`)
			time.Sleep(time.Hour)
		case "truncated":
			fmt.Println(`{"type":"turn.started"}`)
			os.Exit(0)
		case "stderr-flood":
			fmt.Fprint(os.Stderr, strings.Repeat("x", 200000), "tail-marker")
			os.Exit(7)
		case "large":
			data, _ := json.Marshal(ThreadEvent{Type: EventTypeItemCompleted, Item: &AgentMessageItem{Type: ItemTypeAgentMessage, Text: strings.Repeat("a", 100000)}})
			fmt.Println(string(data))
			fmt.Print(`{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}`)
			os.Exit(0)
		}
		if string(input) != "test prompt" {
			fmt.Fprintln(os.Stderr, "unexpected stdin")
			os.Exit(8)
		}
		fmt.Println(`{"type":"thread.started","thread_id":"thread_test"}`)
		fmt.Println(`{"type":"item.completed","item":{"id":"item_test","type":"agent_message","text":"synthetic reply"}}`)
		fmt.Println(`{"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":3}}`)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func fakeExec(t *testing.T, mode string) *Exec {
	t.Helper()
	for _, key := range []string{"OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_BASE_URL", "OPENAI_API_URL", "CODEX_API_KEY", "CODEX_HOME"} {
		t.Setenv(key, "")
	}
	t.Setenv("CODEX_TEST_HELPER", "1")
	t.Setenv("CODEX_TEST_MODE", mode)
	path, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	e, err := NewExec(path)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestExecSyntheticStream(t *testing.T) {
	e := fakeExec(t, "success")
	stream, err := e.Run(t.Context(), Args{Input: "test prompt", Model: "fake", SandboxMode: SandboxModeReadOnly})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var types []EventType
	for event, err := range EventStream(t.Context(), stream) {
		if err != nil {
			t.Fatal(err)
		}
		if event != nil {
			types = append(types, event.Type)
		}
	}
	if len(types) != 3 || types[0] != EventTypeThreadStarted || types[2] != EventTypeTurnCompleted {
		t.Fatalf("events: %v", types)
	}
	if err := stream.Wait(); err != nil {
		t.Fatal(err)
	}
}

func TestExecSyntheticFailure(t *testing.T) {
	e := fakeExec(t, "failure")
	stream, err := e.Run(t.Context(), Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	io.Copy(io.Discard, stream.Stdout())
	if err := stream.Wait(); err == nil || !strings.Contains(err.Error(), "synthetic failure") {
		t.Fatalf("error = %v", err)
	}
}

func TestExecSyntheticCancellation(t *testing.T) {
	e := fakeExec(t, "wait")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	stream, err := e.Run(ctx, Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	cancel()
	if err := stream.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestExecCloseStopsAndReaps(t *testing.T) {
	for _, mode := range []string{"wait", "blocked-stdin"} {
		t.Run(mode, func(t *testing.T) {
			e := fakeExec(t, mode)
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			stream, err := e.Run(ctx, Args{Input: strings.Repeat("x", 2*1024*1024)})
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			decoder := json.NewDecoder(stream.Stdout())
			var event ThreadEvent
			if err := decoder.Decode(&event); err != nil {
				t.Fatal(err)
			}
			done := make(chan error, 1)
			go func() { done <- stream.Close() }()
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
			case <-time.After(2 * time.Second):
				cancel()
				t.Fatal("Close did not stop and reap the helper")
			}
			if err := stream.Wait(); !errors.Is(err, context.Canceled) {
				t.Fatalf("Wait = %v", err)
			}
			if err := stream.Close(); err != nil {
				t.Fatalf("second Close = %v", err)
			}
		})
	}
}

func TestExecConcurrentCloseAndWait(t *testing.T) {
	e := fakeExec(t, "wait")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	stream, err := e.Run(ctx, Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() { _ = stream.Wait() })
		wg.Go(func() { _ = stream.Close() })
	}
	wg.Wait()
	if err := stream.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait = %v", err)
	}
}

func TestExecExitErrorAndBoundedStderr(t *testing.T) {
	e := fakeExec(t, "stderr-flood")
	stream, err := e.Run(t.Context(), Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	_, _ = io.Copy(io.Discard, stream.Stdout())
	err = stream.Wait()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 7 {
		t.Fatalf("exit error = %v", err)
	}
	if !strings.HasSuffix(err.Error(), "tail-marker") || len(err.Error()) > 66*1024 {
		t.Fatalf("stderr error length = %d", len(err.Error()))
	}
}

func TestExecStartFailure(t *testing.T) {
	e, err := NewExec(filepath.Join(t.TempDir(), "missing-codex"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.Run(t.Context(), Args{Input: "test prompt"}); err == nil {
		t.Fatal("missing executable succeeded")
	}
}

func TestExecArgumentDelivery(t *testing.T) {
	e := fakeExec(t, "success")
	capture := filepath.Join(t.TempDir(), "capture.json")
	t.Setenv("CODEX_TEST_CAPTURE", capture)
	args := Args{Input: "test prompt", ThreadID: "thread-test", Images: []string{"image with spaces.png"}, ApprovalPolicy: ApprovalModeNever, SandboxMode: SandboxModeReadOnly}
	stream, err := e.Run(t.Context(), args)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	for _, err := range EventStream(t.Context(), stream) {
		if err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(capture)
	if err != nil {
		t.Fatal(err)
	}
	var got struct {
		Args  []string
		Input string
	}
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	expected, _ := args.commandArgs()
	if strings.Join(got.Args, "\x00") != strings.Join(expected, "\x00") || got.Input != args.Input {
		t.Fatalf("capture = %+v", got)
	}
}

func TestExecDeadlineRemainsInspectable(t *testing.T) {
	e := fakeExec(t, "wait")
	ctx, cancel := context.WithTimeout(t.Context(), 100*time.Millisecond)
	defer cancel()
	stream, err := e.Run(ctx, Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	for _, err = range EventStream(ctx, stream) {
		if err != nil {
			break
		}
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("stream deadline = %v", err)
	}
	if err := stream.Wait(); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Wait deadline = %v", err)
	}
}
