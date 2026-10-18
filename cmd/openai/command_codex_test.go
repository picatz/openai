package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

type codexExecCapture struct {
	Args  []string `json:"args"`
	Input string   `json:"input"`
}

// TestMain dispatches here only when this test executable is used as the fake
// codex process. No installed Codex, configuration, credentials, or API is used.
func runCodexExecTestHelper() {
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		os.Exit(9)
	}
	capture, _ := json.Marshal(codexExecCapture{Args: os.Args[1:], Input: string(input)})
	if err := os.WriteFile(os.Getenv("OPENAI_CODEX_EXEC_CAPTURE"), capture, 0600); err != nil {
		os.Exit(9)
	}
	switch os.Getenv("OPENAI_CODEX_EXEC_MODE") {
	case "exit-error":
		fmt.Fprintln(os.Stderr, "synthetic login required: run codex login")
		os.Exit(7)
	case "truncated":
		fmt.Println(`{"type":"item.completed","item":{"id":"partial","type":"agent_message","text":"incomplete reply"}}`)
		return
	case "malformed":
		fmt.Println(`{"type":`)
		time.Sleep(time.Hour)
	case "turn-failed":
		fmt.Println(`{"type":"turn.failed","error":{"message":"synthetic turn failed"}}`)
		time.Sleep(time.Hour)
	case "wait":
		fmt.Println(`{"type":"turn.started"}`)
		time.Sleep(time.Hour)
	}
	fmt.Println(`{"type":"thread.started","thread_id":"thread_synthetic"}`)
	fmt.Println(`{"type":"item.completed","item":{"id":"commentary","type":"agent_message","text":"working"}}`)
	fmt.Println(`{"type":"item.completed","item":{"id":"final","type":"agent_message","text":"synthetic final reply"}}`)
	fmt.Println(`{"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":3}}`)
	if os.Getenv("OPENAI_CODEX_EXEC_MODE") == "completed-exit-error" {
		fmt.Fprintln(os.Stderr, "synthetic exit failure after completion")
		os.Exit(7)
	}
}

func installSyntheticCodex(t *testing.T) {
	t.Helper()
	directory := t.TempDir()
	name := "codex"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	in, err := os.Open(self)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.OpenFile(filepath.Join(directory, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
	if err != nil {
		t.Fatal(err)
	}
	_, copyErr := io.Copy(out, in)
	if err := errors.Join(copyErr, out.Close()); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	t.Setenv("OPENAI_CODEX_EXEC_HELPER", "1")
	t.Setenv("CODEX_HOME", t.TempDir())
	for _, key := range []string{"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_URL", "OPENAI_MODEL"} {
		t.Setenv(key, "")
	}
}

func readCodexExecCapture(t *testing.T, path string) codexExecCapture {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var capture codexExecCapture
	if err := json.Unmarshal(data, &capture); err != nil {
		t.Fatal(err)
	}
	return capture
}

func TestCodexExec(t *testing.T) {
	installSyntheticCodex(t)
	for _, tc := range []struct {
		name   string
		args   []string
		stdin  string
		prompt string
		flags  []string
	}{
		{name: "arguments", args: []string{"codex", "exec", "hello", "there"}, stdin: "unused input", prompt: "hello there"},
		{name: "pipe", args: []string{"codex", "exec"}, stdin: "line one\nline two\n", prompt: "line one\nline two\n"},
		{name: "stdin marker", args: []string{"codex", "exec", "-"}, stdin: "piped prompt", prompt: "piped prompt"},
		{name: "literal flags", args: []string{"codex", "exec", "--", "--full-auto", "$(touch file)"}, prompt: "--full-auto $(touch file)"},
		{name: "directory and model", args: []string{"codex", "exec", "-C", "directory with spaces", "--model", "synthetic-model", "hello"}, prompt: "hello", flags: []string{"--model", "synthetic-model", "--cd", "directory with spaces"}},
		{name: "root flag position", args: []string{"--model", "synthetic-model", "--timeout", "30s", "codex", "exec", "hello"}, prompt: "hello", flags: []string{"--model", "synthetic-model"}},
		{name: "parent flag position", args: []string{"codex", "--model", "synthetic-model", "--timeout", "30s", "exec", "hello"}, prompt: "hello", flags: []string{"--model", "synthetic-model"}},
		{name: "resume", args: []string{"codex", "exec", "--resume", "thread_existing", "hello"}, prompt: "hello", flags: []string{"resume"}},
		{name: "literal resume ID", args: []string{"codex", "exec", "--resume=--full-auto", "hello"}, prompt: "hello", flags: []string{"resume"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			capturePath := filepath.Join(t.TempDir(), "capture.json")
			t.Setenv("OPENAI_CODEX_EXEC_CAPTURE", capturePath)
			t.Setenv("OPENAI_CODEX_EXEC_MODE", "success")
			// Neither root API environment defaults nor their validation apply.
			t.Setenv("OPENAI_MODEL", "responses-only-model")
			t.Setenv("OPENAI_API_URL", "invalid legacy API URL")
			var out, errOut bytes.Buffer
			code := run(t.Context(), tc.args, strings.NewReader(tc.stdin), &out, &errOut)
			if code != 0 || out.String() != "synthetic final reply\n" || errOut.Len() != 0 {
				t.Fatalf("code=%d out=%q stderr=%q", code, &out, &errOut)
			}
			capture := readCodexExecCapture(t, capturePath)
			wantArgs := append([]string{"exec", "--json"}, tc.flags...)
			wantArgs = append(wantArgs, "--")
			if tc.name == "resume" {
				wantArgs = append(wantArgs, "thread_existing")
			}
			if tc.name == "literal resume ID" {
				wantArgs = append(wantArgs, "--full-auto")
			}
			wantArgs = append(wantArgs, "-")
			// An exact argv match also proves no implicit sandbox, approval,
			// network, model, or other configuration override was introduced.
			if capture.Input != tc.prompt || !reflect.DeepEqual(capture.Args, wantArgs) {
				t.Fatalf("capture=%+v; want args=%q prompt=%q", capture, wantArgs, tc.prompt)
			}
		})
	}

	for _, args := range [][]string{
		{"codex", "exec", "--output", "json", "hello"},
		{"--output", "json", "codex", "exec", "hello"},
		{"codex", "-o", "json", "exec", "hello"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			t.Setenv("OPENAI_CODEX_EXEC_CAPTURE", filepath.Join(t.TempDir(), "capture.json"))
			var out, errOut bytes.Buffer
			if code := run(t.Context(), args, strings.NewReader(""), &out, &errOut); code != 0 || errOut.Len() != 0 {
				t.Fatalf("code=%d stderr=%q", code, &errOut)
			}
			var result struct {
				ThreadID      string         `json:"thread_id"`
				FinalResponse string         `json:"final_response"`
				Usage         map[string]int `json:"usage"`
			}
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatalf("one JSON result required: %v: %q", err, &out)
			}
			if result.ThreadID != "thread_synthetic" || result.FinalResponse != "synthetic final reply" || result.Usage["input_tokens"] != 2 || result.Usage["output_tokens"] != 3 {
				t.Fatalf("result=%+v", result)
			}
		})
	}

	for _, tc := range []struct{ mode, message string }{
		{"exit-error", "synthetic login required: run codex login"},
		{"truncated", "before turn.completed"},
		{"malformed", "parse codex event"},
		{"turn-failed", "synthetic turn failed"},
		{"completed-exit-error", "synthetic exit failure after completion"},
	} {
		for _, format := range []string{"text", "json"} {
			t.Run(tc.mode+"/"+format, func(t *testing.T) {
				t.Setenv("OPENAI_CODEX_EXEC_MODE", tc.mode)
				t.Setenv("OPENAI_CODEX_EXEC_CAPTURE", filepath.Join(t.TempDir(), "capture.json"))
				ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
				defer cancel()
				var out, errOut bytes.Buffer
				code := run(ctx, []string{"codex", "exec", "--output", format, "hello"}, strings.NewReader(""), &out, &errOut)
				if code != 1 || out.Len() != 0 || !strings.Contains(errOut.String(), tc.message) {
					t.Fatalf("code=%d out=%q stderr=%q", code, &out, &errOut)
				}
			})
		}
	}

	t.Run("cancel running process", func(t *testing.T) {
		t.Setenv("OPENAI_CODEX_EXEC_MODE", "wait")
		capturePath := filepath.Join(t.TempDir(), "capture.json")
		t.Setenv("OPENAI_CODEX_EXEC_CAPTURE", capturePath)
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		var out, errOut bytes.Buffer
		done := make(chan int, 1)
		go func() { done <- run(ctx, []string{"codex", "exec", "hello"}, strings.NewReader(""), &out, &errOut) }()
		deadline := time.After(10 * time.Second)
		for {
			if _, err := os.Stat(capturePath); err == nil {
				break
			}
			select {
			case <-deadline:
				t.Fatal("synthetic process did not start")
			case <-time.After(5 * time.Millisecond):
			}
		}
		cancel()
		select {
		case code := <-done:
			if code != 130 || out.Len() != 0 || !strings.Contains(errOut.String(), "context canceled") {
				t.Fatalf("code=%d out=%q stderr=%q", code, &out, &errOut)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("canceled Codex process did not finish")
		}
	})

	t.Run("whole-run deadline", func(t *testing.T) {
		t.Setenv("OPENAI_CODEX_EXEC_MODE", "wait")
		t.Setenv("OPENAI_CODEX_EXEC_CAPTURE", filepath.Join(t.TempDir(), "capture.json"))
		var out, errOut bytes.Buffer
		code := run(t.Context(), []string{"--timeout", "200ms", "codex", "exec", "hello"}, strings.NewReader(""), &out, &errOut)
		if code != 1 || out.Len() != 0 || !strings.Contains(errOut.String(), "context deadline exceeded") {
			t.Fatalf("code=%d out=%q stderr=%q", code, &out, &errOut)
		}
	})

	t.Run("output error", func(t *testing.T) {
		t.Setenv("OPENAI_CODEX_EXEC_CAPTURE", filepath.Join(t.TempDir(), "capture.json"))
		cmd := newRootCommand()
		cmd.SetArgs([]string{"codex", "exec", "hello"})
		cmd.SetOut(codexErrorWriter{})
		if err := cmd.ExecuteContext(t.Context()); !errors.Is(err, io.ErrClosedPipe) {
			t.Fatalf("error=%v", err)
		}
	})
}

type codexErrorWriter struct{}

func (codexErrorWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestCodexValidationBeforeProcess(t *testing.T) {
	// An empty PATH makes unintended process startup observable as the wrong error.
	t.Setenv("PATH", t.TempDir())
	for _, tc := range []struct {
		args    []string
		input   string
		message string
	}{
		{args: []string{"codex", "exec"}, message: "prompt must not be empty"},
		{args: []string{"codex", "exec", "  "}, message: "prompt must not be empty"},
		{args: []string{"codex", "exec", "--model=", "hello"}, message: "model must not be empty"},
		{args: []string{"--model=", "codex", "exec", "hello"}, message: "model must not be empty"},
		{args: []string{"codex", "exec", "--cd=", "hello"}, message: "cd must not be empty"},
		{args: []string{"codex", "exec", "--resume= ", "hello"}, message: "resume requires an explicit Codex thread ID"},
		{args: []string{"codex", "exec", "--timeout=-1s", "hello"}, message: "timeout must not be negative"},
		{args: []string{"codex", "exec", "--output=xml", "hello"}, message: "output must be text or json"},
		{args: []string{"codex", "exec", "--unknown", "hello"}, message: "unknown flag"},
		{args: []string{"codex", "unknown"}, message: "unknown command"},
		{args: []string{"codex", "exec", "hello"}, message: "Codex CLI unavailable; install it from https://github.com/openai/codex and ensure codex is on PATH"},
		{args: []string{"codex", "exec", "-"}, input: strings.Repeat("x", maxPromptBytes+1), message: "prompt exceeds 16 MiB"},
	} {
		t.Run(strings.Join(tc.args, " "), func(t *testing.T) {
			var out, errOut bytes.Buffer
			code := run(t.Context(), tc.args, strings.NewReader(tc.input), &out, &errOut)
			if code != 1 || out.Len() != 0 || !strings.Contains(errOut.String(), tc.message) {
				t.Fatalf("code=%d out=%q stderr=%q", code, &out, &errOut)
			}
		})
	}
	for _, flag := range []string{"--base-url=https://example.invalid", "--stream", "--stream=false", "--web-search", "--legacy", "--temporary", "-t", "--history-dir=unused", "--session=unused"} {
		for _, args := range [][]string{{flag, "codex", "exec", "hello"}, {"codex", flag, "exec", "hello"}, {"codex", "exec", flag, "hello"}} {
			t.Run(strings.Join(args, " "), func(t *testing.T) {
				var out, errOut bytes.Buffer
				code := run(t.Context(), args, strings.NewReader(""), &out, &errOut)
				if code != 1 || out.Len() != 0 || !strings.Contains(errOut.String(), "is not supported by openai codex") {
					t.Fatalf("code=%d out=%q stderr=%q", code, &out, &errOut)
				}
			})
		}
	}
}

func TestCodexDeadlineIncludesStdin(t *testing.T) {
	in, writer := io.Pipe()
	defer in.Close()
	defer writer.Close()
	cmd := newRootCommand()
	cmd.SetArgs([]string{"codex", "exec", "--timeout=20ms", "-"})
	cmd.SetIn(in)
	if err := cmd.ExecuteContext(t.Context()); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error=%v", err)
	}
}

func TestCodexHelpOffline(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("OPENAI_MODEL", "responses-only-model")
	t.Setenv("OPENAI_BASE_URL", "invalid API URL")
	for _, args := range [][]string{{"--help"}, {"codex"}, {"codex", "--help"}, {"codex", "exec", "--help"}} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			var out, errOut bytes.Buffer
			if code := run(t.Context(), args, strings.NewReader(""), &out, &errOut); code != 0 || errOut.Len() != 0 || !strings.Contains(out.String(), "codex") {
				t.Fatalf("out=%q stderr=%q", &out, &errOut)
			}
			if args[0] == "codex" && (strings.Contains(out.String(), "responses-only-model") || strings.Contains(out.String(), "default 2m0s")) {
				t.Fatalf("Codex help leaked API defaults: %s", &out)
			}
			if args[0] == "codex" {
				for _, flag := range []string{"--base-url", "--history-dir", "--session", "--legacy", "--stream", "--temporary", "--web-search"} {
					if strings.Contains(out.String(), flag) {
						t.Fatalf("Codex help advertised unsupported flag %s: %s", flag, &out)
					}
				}
			}
		})
	}
}

func TestCodexHelpDoesNotChangeOtherCommands(t *testing.T) {
	cmd := newRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	help := func(target ...string) string {
		t.Helper()
		out.Reset()
		cmd.SetArgs(append([]string{"help"}, target...))
		if err := cmd.ExecuteContext(t.Context()); err != nil {
			t.Fatal(err)
		}
		return out.String()
	}
	before := map[string]string{}
	for _, name := range []string{"", "chat", "live"} {
		var target []string
		if name != "" {
			target = []string{name}
		}
		before[name] = help(target...)
		for _, flag := range []string{"--base-url", "--history-dir", "--temporary", "--session", "--stream", "--legacy", "--web-search"} {
			if !strings.Contains(before[name], flag) {
				t.Fatalf("%q help lost root flag %s", name, flag)
			}
		}
	}
	help("codex")
	help("codex", "exec")
	for _, name := range []string{"", "chat", "live"} {
		var target []string
		if name != "" {
			target = []string{name}
		}
		if after := help(target...); after != before[name] {
			t.Fatalf("%q help changed after Codex help:\nbefore:\n%s\nafter:\n%s", name, before[name], after)
		}
	}
}
