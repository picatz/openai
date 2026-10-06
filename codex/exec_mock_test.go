package codex

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

// The helper is this test binary, never a real Codex installation.
func TestMain(m *testing.M) {
	if os.Getenv("CODEX_TEST_HELPER") == "1" && len(os.Args) > 1 && os.Args[1] == "exec" {
		input, _ := io.ReadAll(os.Stdin)
		switch os.Getenv("CODEX_TEST_MODE") {
		case "failure":
			fmt.Fprintln(os.Stderr, "synthetic failure")
			os.Exit(7)
		case "wait":
			time.Sleep(time.Hour)
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
	if err := stream.Wait(); err == nil {
		t.Fatal("expected cancellation failure")
	}
}
