package codexworkspace

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/picatz/openai/internal/codexsession"
)

func workspaceFixture(t *testing.T, plain bool) Model {
	t.Helper()
	m, start := begin(t, New(Config{Context: t.Context(), Backend: newFake(), NoColor: plain, ReducedMotion: true}), "synthetic-turn", 1)
	m, _ = update(m, start())
	items := []codexsession.Item{
		{ID: "plan", Kind: "plan", Title: "Keep cancellation honest", Text: "1. Separate acknowledgements from completion\n2. Preserve unfinished output\n3. Exercise terminal lifecycle", Status: "completed"},
		{ID: "command", Kind: "command", Title: "go test ./internal/codexworkspace", Text: "ok  internal/codexworkspace\nok  internal/codextui", Status: "completed"},
		{ID: "files", Kind: "files", Title: "Tighten session shutdown", Files: []codexsession.FileChange{{Path: "internal/codexworkspace/run.go", Kind: "update"}}, Diff: "@@ -1,3 +1,4 @@\n func shutdown() {\n-    forceKill()\n+    requestQuit()\n+    closeSession()\n }", Status: "completed"},
		{ID: "message", Kind: "message", Title: "Summary", Text: "The composer keeps your next draft while a turn runs. Interruption remains pending until the server confirms a terminal state.", Status: "running"},
	}
	for i, item := range items {
		e := event("item", "synthetic-turn", 1, uint64(i+2))
		e.Item = item
		m = emit(m, e)
	}
	m, _ = update(m, key(tea.KeyEscape, 0))
	m, _ = update(m, key(tea.KeyDown, 0))
	m, _ = update(m, key(tea.KeyDown, 0))
	m, _ = update(m, key(tea.KeyF6, 0))
	m.input.SetValue("Explain the remaining trade-offs.\nKeep the terminal cleanup bounded.")
	m = emit(m, codexsession.Event{Kind: "notice", Notice: "Unsupported approval request rejected; no permission was granted."})
	return m
}

func TestFixedWorkspaceFrames(t *testing.T) {
	for _, tc := range []struct {
		name          string
		width, height int
		plain         bool
	}{{"wide", 120, 32, false}, {"standard", 80, 24, false}, {"compact", 40, 20, false}, {"plain", 80, 24, true}} {
		t.Run(tc.name, func(t *testing.T) {
			m := workspaceFixture(t, tc.plain)
			m, _ = update(m, tea.WindowSizeMsg{Width: tc.width, Height: tc.height})
			got := m.View().Content + "\n"
			path := filepath.Join("testdata", tc.name+".golden")
			if os.Getenv("UPDATE_CODEXWORKSPACE_GOLDENS") == "1" {
				if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(got), 0644); err != nil {
					t.Fatal(err)
				}
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if got != strings.ReplaceAll(string(want), "\r\n", "\n") {
				t.Fatalf("workspace frame changed: inspect %s", path)
			}
			if !strings.Contains(got, "F1 help") || !strings.Contains(got, "Ctrl+Q") {
				t.Fatal("help or quit hint clipped")
			}
		})
	}
}

func TestInputShorteningIsVisible(t *testing.T) {
	for _, size := range [][2]int{{40, 20}, {80, 24}} {
		m := New(Config{Context: t.Context(), Backend: newFake(), NoColor: true})
		m, _ = update(m, tea.WindowSizeMsg{Width: size[0], Height: size[1]})
		m, _ = update(m, tea.PasteMsg{Content: strings.Repeat("x", maxPromptBytes+100)})
		if !m.inputShortened || !strings.Contains(m.View().Content, "shortened") {
			t.Fatal("shortened paste was silent")
		}
		if len(m.input.Value()) > maxPromptBytes {
			t.Fatal("paste exceeded byte cap")
		}
	}
}
