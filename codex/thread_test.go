package codex

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func fakeThread(t *testing.T, mode string) *Thread {
	t.Helper()
	e := fakeExec(t, mode)
	thread, err := NewThread(Options{CodexPathOverride: e.path}, ThreadOptions{ApprovalPolicy: ApprovalModeNever, SandboxMode: SandboxModeReadOnly})
	if err != nil {
		t.Fatal(err)
	}
	return thread
}

func schemaTempDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, key := range []string{"TMPDIR", "TMP", "TEMP"} {
		t.Setenv(key, dir)
	}
	return dir
}

func assertSchemaCleaned(t *testing.T, dir string) {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "codex-output-schema-*"))
	if err != nil || len(files) != 0 {
		t.Fatalf("schema cleanup: %v, %v", files, err)
	}
}

func TestThreadRunResumeAndStructuredOutput(t *testing.T) {
	dir := schemaTempDir(t)
	thread := fakeThread(t, "success")
	capture := filepath.Join(t.TempDir(), "capture.json")
	t.Setenv("CODEX_TEST_CAPTURE", capture)
	schema := map[string]any{"type": "object", "properties": map[string]any{"answer": map[string]any{"type": "string"}}}
	for turn := range 2 {
		result, err := thread.RunText(t.Context(), "test prompt", &TurnOptions{OutputSchema: schema})
		if err != nil {
			t.Fatal(err)
		}
		if result.FinalResponse != "synthetic reply" || len(result.Items) != 1 || result.Usage.OutputTokens != 3 || thread.ID() != "thread_test" {
			t.Fatalf("turn = %+v, id = %q", result, thread.ID())
		}
		raw, err := os.ReadFile(capture)
		if err != nil {
			t.Fatal(err)
		}
		var got struct {
			Args   []string
			Schema json.RawMessage
		}
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		expected, _ := json.Marshal(schema)
		if string(got.Schema) != string(expected) {
			t.Fatalf("schema = %s", got.Schema)
		}
		resumed := strings.Contains(strings.Join(got.Args, " "), "resume -- thread_test -")
		if resumed != (turn == 1) {
			t.Fatalf("resume args = %v", got.Args)
		}
		assertSchemaCleaned(t, dir)
	}
	resumed, err := ResumeThread("saved-thread", Options{CodexPathOverride: thread.exec.path}, ThreadOptions{})
	if err != nil || resumed.ID() != "saved-thread" {
		t.Fatalf("resume = %v, %v", resumed, err)
	}
	if _, err := ResumeThread("", Options{}, ThreadOptions{}); err == nil {
		t.Fatal("accepted empty ID")
	}
}

func TestThreadFailureCancellationAndSchemaCleanup(t *testing.T) {
	for _, tc := range []struct{ mode, want string }{
		{"malformed", "parse codex event"}, {"failed-turn", "synthetic turn failure"}, {"failure", "synthetic failure"}, {"truncated", "before turn.completed"},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			dir := schemaTempDir(t)
			thread := fakeThread(t, tc.mode)
			ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
			defer cancel()
			_, err := thread.RunText(ctx, "test prompt", &TurnOptions{OutputSchema: map[string]any{"type": "object"}})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want %s", err, tc.want)
			}
			assertSchemaCleaned(t, dir)
		})
	}
}

func TestStreamedTurnCloseWithoutDraining(t *testing.T) {
	dir := schemaTempDir(t)
	thread := fakeThread(t, "wait")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	result, err := thread.RunStreamedText(ctx, "test prompt", &TurnOptions{OutputSchema: map[string]any{"type": "object"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := result.Close(); err != nil {
		t.Fatal(err)
	}
	if err := result.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait = %v", err)
	}
	if _, ok := <-result.Events; ok {
		t.Fatal("events channel still open")
	}
	assertSchemaCleaned(t, dir)
}

func TestThreadSchemaCleanupOnStartAndInputFailure(t *testing.T) {
	dir := schemaTempDir(t)
	thread, err := NewThread(Options{CodexPathOverride: filepath.Join(t.TempDir(), "missing")}, ThreadOptions{})
	if err != nil {
		t.Fatal(err)
	}
	opts := &TurnOptions{OutputSchema: map[string]any{"type": "object"}}
	if _, err := thread.RunText(t.Context(), "test prompt", opts); err == nil {
		t.Fatal("start succeeded")
	}
	assertSchemaCleaned(t, dir)
	if _, err := thread.Run(t.Context(), ComposeInput(LocalImagePart("")), opts); err == nil {
		t.Fatal("invalid input succeeded")
	}
	assertSchemaCleaned(t, dir)
	var zero Thread
	if _, err := zero.RunText(t.Context(), "test prompt", nil); err == nil {
		t.Fatal("uninitialized thread succeeded")
	}
}

func TestThreadOptionsAreCopied(t *testing.T) {
	e := fakeExec(t, "success")
	enabled := true
	options := ThreadOptions{ConfigOverrides: []string{"value=true"}, AdditionalDirectories: []string{"original"}, NetworkAccessEnabled: &enabled}
	thread, err := NewThread(Options{CodexPathOverride: e.path}, options)
	if err != nil {
		t.Fatal(err)
	}
	options.ConfigOverrides[0] = "other=true"
	options.AdditionalDirectories[0] = "changed"
	enabled = false
	if thread.threadOptions.ConfigOverrides[0] != "value=true" || thread.threadOptions.AdditionalDirectories[0] != "original" || !*thread.threadOptions.NetworkAccessEnabled {
		t.Fatal("thread aliases caller-owned options")
	}
}

func TestResumeThreadPassesUntrustedIDsAsPositionals(t *testing.T) {
	for _, id := range []string{"--last", "--dangerously-bypass-approvals-and-sandbox", "--config=approval_policy=never"} {
		t.Run(id, func(t *testing.T) {
			e := fakeExec(t, "success")
			capture := filepath.Join(t.TempDir(), "capture.json")
			t.Setenv("CODEX_TEST_CAPTURE", capture)
			thread, err := ResumeThread(id, Options{CodexPathOverride: e.path}, ThreadOptions{SandboxMode: SandboxModeReadOnly, ApprovalPolicy: ApprovalModeOnRequest})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := thread.Run(t.Context(), ComposeInput(TextPart("test prompt"), LocalImagePart("image with spaces.png")), nil); err != nil {
				t.Fatal(err)
			}
			raw, err := os.ReadFile(capture)
			if err != nil {
				t.Fatal(err)
			}
			var got struct{ Args []string }
			if err := json.Unmarshal(raw, &got); err != nil {
				t.Fatal(err)
			}
			want := []string{"exec", "--json", "--config", `approval_policy="on-request"`, "--sandbox", "read-only", "resume", "--image", "image with spaces.png", "--", id, "-"}
			if !reflect.DeepEqual(got.Args, want) {
				t.Fatalf("args = %#v, want %#v", got.Args, want)
			}
		})
	}
}
