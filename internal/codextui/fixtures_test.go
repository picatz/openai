package codextui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

func TestFixedTerminalFixtures(t *testing.T) {
	for _, tc := range []struct {
		name          string
		width, height int
		plain         bool
	}{
		{"wide", 120, 32, false}, {"standard", 80, 24, false}, {"compact", 40, 16, false}, {"plain", 80, 24, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := fixture()
			m.cfg.NoColor = tc.plain
			m = step(m, tea.WindowSizeMsg{Width: tc.width, Height: tc.height})
			got := m.View().Content + "\n"
			path := filepath.Join("testdata", tc.name+".golden")
			if os.Getenv("UPDATE_CODEXTUI_GOLDENS") == "1" {
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
			if got != normalizeFixtureLineEndings(string(want)) {
				t.Fatalf("fixture changed: %s; inspect before intentionally regenerating", path)
			}
		})
	}
}

// Git may check text fixtures out as CRLF on Windows. Compare terminal frames,
// not checkout line endings, without ignoring any other whitespace or styles.
func normalizeFixtureLineEndings(s string) string { return strings.ReplaceAll(s, "\r\n", "\n") }

func TestFixtureLineEndings(t *testing.T) {
	want := "\x1b[35mframe\x1b[m  \nnext\n"
	for _, s := range []string{want, strings.ReplaceAll(want, "\n", "\r\n")} {
		if normalizeFixtureLineEndings(s) != want {
			t.Fatal("fixture normalization changed frame content")
		}
	}
	if normalizeFixtureLineEndings("single\rcarriage") != "single\rcarriage" {
		t.Fatal("normalization hid a non-CRLF difference")
	}
}

func BenchmarkWorkspaceView(b *testing.B) {
	m := fixture()
	m = step(m, tea.WindowSizeMsg{Width: 120, Height: 32})
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = m.View()
	}
}

func FuzzDisplayBounds(f *testing.F) {
	f.Add("hello 日本語🙂", uint8(80), uint8(24))
	f.Add("\x1b]52;c;attack\a\u009b2J\x1b[2J", uint8(30), uint8(10))
	f.Fuzz(func(t *testing.T, text string, width, height uint8) {
		m := New(Config{NoColor: true})
		m = step(m, BeginTurnMsg{scope(0)})
		m = step(m, ItemMsg{scope(1), Item{ID: "a", Kind: Message, Title: text, Text: text, Status: Running}})
		m = step(m, tea.WindowSizeMsg{Width: int(width), Height: int(height)})
		assertFrame(t, m.View().Content, int(width), int(height))
	})
}
