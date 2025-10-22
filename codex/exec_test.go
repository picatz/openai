package codex

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/go-git/go-git/v5"
)

func TestBuildEnvironment(t *testing.T) {
	t.Setenv(internalOriginatorEnv, "")
	t.Setenv("OPENAI_BASE_URL", "existing")
	t.Setenv("CODEX_API_KEY", "existing")

	env := buildEnvironment("https://example.com", "test-key")

	assertEnvContains := func(key, expected string) {
		t.Helper()
		for _, entry := range env {
			if value, ok := strings.CutPrefix(entry, key+"="); ok {
				if value != expected {
					t.Fatalf("expected %s to be %q, got %q", key, expected, value)
				}
				return
			}
		}
		t.Fatalf("expected environment to contain %s", key)
	}

	assertEnvContains(internalOriginatorEnv, goSDKOriginator)
	assertEnvContains("OPENAI_BASE_URL", "https://example.com")
	assertEnvContains("CODEX_API_KEY", "test-key")
}

func TestExec_Run(t *testing.T) {
	dir := t.TempDir()

	repo, err := git.PlainCloneContext(t.Context(), dir, false, &git.CloneOptions{
		URL: "https://github.com/hashicorp/go-getter.git",
	})
	if err != nil {
		t.Fatalf("failed to clone repo: %v", err)
	}

	pathToCodex, err := findCodexPath()
	if err != nil {
		t.Fatalf("failed to find codex path: %v", err)
	}

	exec, err := NewExec(pathToCodex)
	if err != nil {
		t.Fatalf("failed to create exec: %v", err)
	}

	firstStream, err := exec.Run(t.Context(), Args{
		Input:       "What is the main purpose of this repository?",
		Model:       "gpt-5-codex",
		SandboxMode: SandboxModeWorkspaceWrite,
		WorkingDirectory: func() string {
			w, err := repo.Worktree()
			if err != nil {
				t.Fatalf("failed to get worktree: %v", err)
			}
			return w.Filesystem.Root()
		}(),
	})
	if err != nil {
		t.Fatalf("failed to run codex: %v", err)
	}
	defer firstStream.Close()

	decoder := json.NewDecoder(firstStream.Stdout())

	var threadId string
	for {
		var event ThreadEvent
		if err := decoder.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("failed to decode event: %v", err)
		}
		if event.Type == EventTypeThreadStarted {
			threadId = event.ThreadID
		}

		t.Logf("Received event: %s: %s", threadId, event.String())
	}

	err = firstStream.Wait()
	if err != nil {
		t.Fatalf("codex execution failed: %v", err)
	}

	secondStream, err := exec.Run(t.Context(), Args{
		Input:    "What language is it written in?",
		ThreadID: threadId,
	})
	if err != nil {
		t.Fatalf("failed to run codex second time: %v", err)
	}
	defer secondStream.Close()

	decoder = json.NewDecoder(secondStream.Stdout())

	for {
		var event ThreadEvent
		if err := decoder.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("failed to decode event: %v", err)
		}

		t.Logf("Received event: %s: %s", threadId, event.String())
	}

	err = secondStream.Wait()
	if err != nil {
		t.Fatalf("codex execution failed: %v", err)
	}

	thirdStream, err := exec.Run(t.Context(), Args{
		Input:    "Who is the main author of the project?",
		ThreadID: threadId,
	})
	if err != nil {
		t.Fatalf("failed to run codex third time: %v", err)
	}
	defer thirdStream.Close()

	decoder = json.NewDecoder(thirdStream.Stdout())

	for {
		var event ThreadEvent
		if err := decoder.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("failed to decode event: %v", err)
		}

		t.Logf("Received event: %s: %s", threadId, event.String())
	}

	err = thirdStream.Wait()
	if err != nil {
		t.Fatalf("codex execution failed: %v", err)
	}
}
