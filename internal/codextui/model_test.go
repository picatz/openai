package codextui

import (
	"fmt"
	"strings"
	"testing"
	"unicode/utf8"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func step(m Model, msg tea.Msg) Model { next, _ := m.Update(msg); return next.(Model) }
func scope(seq uint64) Scope {
	return Scope{Generation: 1, ThreadID: "thread-fixture", TurnID: "turn-fixture", Sequence: seq}
}
func key(code rune) tea.KeyPressMsg { return tea.KeyPressMsg{Code: code} }
func fixture() Model {
	m := New(Config{Workspace: "~/work/openai", Model: "fixture model", Capabilities: Capabilities{ToolOutput: true, Diffs: true, Plans: true}, ReducedMotion: true})
	m = step(m, BeginTurnMsg{scope(0)})
	items := []Item{
		{ID: "message", Kind: Message, Title: "A focused plan for this change", Text: "I’ll inspect the command flow, add the smallest useful workspace view, and verify keyboard behavior with deterministic tests.\n\nThe existing safety and authentication defaults remain in effect.", Status: Completed},
		{ID: "plan", Kind: Plan, Title: "Build the workspace view", Text: "[x] Inspect the current interface\n[x] Model message and tool states\n[ ] Verify narrow terminal layouts", Status: Running},
		{ID: "command", Kind: Command, Title: "go test ./internal/codextui", Text: "ok  github.com/picatz/openai/internal/codextui\n\nSynthetic fixture: this command was not executed by the preview.", Status: Completed},
		{ID: "files", Kind: Files, Title: "Workspace presentation", Files: []FileChange{{Path: "internal/codextui/view.go", Kind: "update"}, {Path: "internal/codextui/model_test.go", Kind: "add"}}, Diff: "--- a/internal/codextui/view.go\n+++ b/internal/codextui/view.go\n@@ -1,3 +1,4 @@\n package codextui\n-// Text-only transcript.\n+// Responsive workspace view.\n+// Keyboard-first, with honest capabilities.", Status: Completed},
		{ID: "tool", Kind: Tool, Title: "repository · inspect_tree", Text: "Found the command, model, and terminal lifecycle tests.\nNo files were read by this preview.", Status: Completed},
	}
	for i, item := range items {
		m = step(m, ItemMsg{Scope: scope(uint64(i + 1)), Item: item})
	}
	m.selected = 3
	m.detail.GotoTop()
	m.refresh(false)
	return m
}

func TestSnapshotReplacesDeltasAndTerminalDoesNotRegress(t *testing.T) {
	m := New(Config{})
	m = step(m, BeginTurnMsg{scope(0)})
	m = step(m, ItemMsg{scope(1), Item{ID: "a", Kind: Message, Text: "Hello", Status: Running}})
	m = step(m, TextDeltaMsg{scope(2), "a", " there"})
	if m.items[0].Text != "Hello there" {
		t.Fatal(m.items)
	}
	m = step(m, ItemMsg{scope(3), Item{ID: "a", Kind: Message, Text: "Authoritative final", Status: Completed}})
	m = step(m, TextDeltaMsg{scope(4), "a", " late"})
	m = step(m, ItemMsg{scope(5), Item{ID: "a", Kind: Message, Text: "stale running", Status: Running}})
	if len(m.items) != 1 || m.items[0].Text != "Authoritative final" || m.items[0].Status != Completed {
		t.Fatal(m.items)
	}
}

func TestStaleDuplicateForeignAndMissingItemEventsAreIgnored(t *testing.T) {
	m := New(Config{})
	m = step(m, BeginTurnMsg{scope(0)})
	m = step(m, ItemMsg{scope(1), Item{ID: "a", Text: "kept", Status: Running}})
	for _, s := range []Scope{scope(1), {Generation: 0, ThreadID: "thread-fixture", TurnID: "turn-fixture", Sequence: 10}, {Generation: 1, ThreadID: "other", TurnID: "turn-fixture", Sequence: 10}, {Generation: 1, ThreadID: "thread-fixture", TurnID: "old", Sequence: 10}} {
		m = step(m, ItemMsg{s, Item{ID: "a", Text: "bad", Status: Running}})
	}
	m = step(m, TextDeltaMsg{scope(2), "missing", "bad"})
	if len(m.items) != 1 || m.items[0].Text != "kept" {
		t.Fatal(m.items)
	}
	m = step(m, EndTurnMsg{scope(3), Interrupted})
	m = step(m, ItemMsg{scope(4), Item{ID: "late", Text: "bad"}})
	if m.active || len(m.items) != 1 || m.items[0].Status != Interrupted {
		t.Fatal(m.items)
	}
	m = step(m, BeginTurnMsg{Scope{Generation: 2, ThreadID: "next", TurnID: "next"}})
	m = step(m, ItemMsg{scope(5), Item{ID: "late", Text: "bad"}})
	if len(m.items) != 0 || !m.active {
		t.Fatal("old generation revived")
	}
}

func TestBoundsOwnershipAndOpaqueIDs(t *testing.T) {
	m := New(Config{})
	m = step(m, BeginTurnMsg{scope(0)})
	files := []FileChange{{Path: "original", Kind: "add"}}
	m = step(m, ItemMsg{scope(1), Item{ID: "\x1ba", Files: files, Status: Running}})
	files[0].Path = "mutated"
	m = step(m, ItemMsg{scope(2), Item{ID: "a", Status: Running}})
	if len(m.items) != 2 || m.items[0].Files[0].Path != "original" {
		t.Fatal("alias or ID collision")
	}
	m = step(m, ItemMsg{scope(3), Item{ID: strings.Repeat("x", 513)}})
	if len(m.items) != 2 {
		t.Fatal("oversized ID retained")
	}
	for i := 0; i < maxItems+5; i++ {
		m = step(m, ItemMsg{scope(uint64(i + 4)), Item{ID: fmt.Sprint(i), Title: "bounded", Text: strings.Repeat("🙂", maxTextBytes), Diff: strings.Repeat("x", maxTextBytes*2), Status: Running}})
	}
	if len(m.items) != maxItems || m.dropped != 7 {
		t.Fatalf("retention: %d dropped %d", len(m.items), m.dropped)
	}
	for _, item := range m.items {
		if len(item.Text) > maxTextBytes+32 || len(item.Diff) > maxTextBytes+32 {
			t.Fatal("unbounded text")
		}
	}
}

func TestScrollAndSelectionRemainStableDuringUpdates(t *testing.T) {
	m := New(Config{Capabilities: Capabilities{ToolOutput: true}})
	m = step(m, BeginTurnMsg{scope(0)})
	m = step(m, ItemMsg{scope(1), Item{ID: "a", Text: strings.Repeat("line\n", 100), Status: Running}})
	m.detail.GotoTop()
	m = step(m, TextDeltaMsg{scope(2), "a", "latest\n"})
	if m.detail.YOffset() != 0 {
		t.Fatal("stream moved scrollback")
	}
	m = step(m, ItemMsg{scope(3), Item{ID: "b", Title: "next"}})
	if m.selected != 0 {
		t.Fatal("stream stole selection")
	}
	m = step(m, key(tea.KeyDown))
	if m.selected != 1 || m.detail.YOffset() != 0 {
		t.Fatal("selection failed")
	}
	m = step(m, key(tea.KeyHome))
	m = step(m, tea.KeyPressMsg{Code: tea.KeyEnd, Mod: tea.ModCtrl})
	m = step(m, TextDeltaMsg{scope(4), "a", "following\n"})
	if !m.detail.AtBottom() {
		t.Fatal("explicit follow lost")
	}
	m = step(m, key(tea.KeyTab))
	old := m.selected
	m = step(m, key(tea.KeyUp))
	if m.selected != old {
		t.Fatal("detail focus changed selection")
	}
}

func TestHelpAndQuitNeverSubmitOrApprove(t *testing.T) {
	m := fixture()
	m = step(m, key('?'))
	if !m.help || !strings.Contains(m.View().Content, "KEYBOARD") {
		t.Fatal("help missing")
	}
	for _, k := range []rune{tea.KeyEnter, 'y', 'n'} {
		next, cmd := m.Update(key(k))
		m = next.(Model)
		if cmd != nil {
			t.Fatal("display sent action")
		}
	}
	m = step(m, key(tea.KeyEsc))
	if m.help {
		t.Fatal("help did not close")
	}
	_, cmd := m.Update(tea.KeyPressMsg{Code: 'q', Mod: tea.ModCtrl})
	if cmd == nil {
		t.Fatal("quit missing")
	}
	if _, ok := cmd().(tea.QuitMsg); !ok {
		t.Fatal("not quit")
	}
}

func TestLayoutColorMotionAndHostileText(t *testing.T) {
	for _, plain := range []bool{false, true} {
		for _, size := range [][2]int{{0, 0}, {1, 1}, {29, 9}, {30, 10}, {40, 16}, {80, 24}, {100, 24}, {120, 32}, {160, 60}} {
			m := fixture()
			m.cfg.NoColor = plain
			m = step(m, tea.WindowSizeMsg{Width: size[0], Height: size[1]})
			m = step(m, ItemMsg{scope(6), Item{ID: "files", Kind: Files, Title: "\x1b]52;c;ATTACK\a日本語🙂", Text: "\x1b[2Jbad\x00\x7f", Diff: "+日本語🙂\n-\u202eBAD", Status: Completed}})
			content := m.View().Content
			if size[0] == 0 || size[1] == 0 {
				if content != "" {
					t.Fatal("zero layout")
				}
				continue
			}
			if len(strings.Split(content, "\n")) > size[1] {
				t.Fatalf("height %v", size)
			}
			for _, line := range strings.Split(content, "\n") {
				if ansi.StringWidth(line) > size[0] {
					t.Fatalf("width %v %q", size, line)
				}
			}
			for _, bad := range []string{"ATTACK", "\x1b[2J", "\x00", "\x7f", "\u202e"} {
				if strings.Contains(content, bad) {
					t.Fatalf("unsafe %q", bad)
				}
			}
			if plain && strings.Contains(content, "\x1b") {
				t.Fatal("NO_COLOR emitted escape")
			}
		}
	}
	for _, cfg := range []Config{{NoColor: true}, {ReducedMotion: true}} {
		m := New(cfg)
		m = step(m, BeginTurnMsg{scope(0)})
		before := m.View().Content
		m = step(m, FrameMsg{})
		if m.View().Content != before {
			t.Fatal("motion in reduced mode")
		}
	}
	m := New(Config{})
	m = step(m, BeginTurnMsg{scope(0)})
	before := m.View().Content
	m = step(m, FrameMsg{})
	if m.View().Content == before {
		t.Fatal("cosmetic frame did not change")
	}
}

func TestUnsupportedCapabilitiesDoNotInventOutput(t *testing.T) {
	m := fixture()
	m.cfg.Capabilities = Capabilities{}
	for i := range m.items {
		m.selected = i
		m.refresh(false)
		text := ansi.Strip(m.View().Content)
		switch m.items[i].Kind {
		case Files:
			if !strings.Contains(text, "Diff unavailable") || strings.Contains(text, "+//") {
				t.Fatal(text)
			}
		case Command, Tool:
			if !strings.Contains(text, "Output unavailable") {
				t.Fatal(text)
			}
		case Plan:
			if !strings.Contains(text, "Plan detail unavailable") {
				t.Fatal(text)
			}
		}
	}
}

func assertFrame(t *testing.T, text string, width, height int) {
	t.Helper()
	if width == 0 || height == 0 {
		if text != "" {
			t.Fatal("nonempty zero-sized frame")
		}
		return
	}
	lines := strings.Split(text, "\n")
	if len(lines) > height {
		t.Fatal("height overflow")
	}
	for _, line := range lines {
		if ansi.StringWidth(line) > width {
			t.Fatal("width overflow")
		}
	}
	if strings.ContainsRune(text, '\x1b') {
		t.Fatal("control sequence in plain frame")
	}
}

func TestTerminalTurnDoesNotLeaveRunningItemsOrAnimate(t *testing.T) {
	for _, status := range []Status{Completed, Failed, Interrupted} {
		m := New(Config{})
		m = step(m, BeginTurnMsg{scope(0)})
		m = step(m, ItemMsg{scope(1), Item{ID: "open", Status: Running}})
		m = step(m, EndTurnMsg{scope(2), status})
		if m.items[0].Status == Running {
			t.Fatal("terminal turn left running item")
		}
		if status == Completed && m.items[0].Status == Completed {
			t.Fatal("invented item success")
		}
		before := m.View().Content
		m = step(m, FrameMsg{})
		if m.View().Content != before {
			t.Fatal("terminal turn animated")
		}
	}
}

func TestNormalizedTextBoundsIncludeMalformedUTF8AndMarker(t *testing.T) {
	for _, raw := range []string{
		strings.Repeat("\xffx", maxTextBytes/2),
		strings.Repeat("\xffx", maxTextBytes/2-1),
		strings.Repeat("🙂", maxTextBytes),
	} {
		m := New(Config{})
		m = step(m, BeginTurnMsg{scope(0)})
		m = step(m, ItemMsg{scope(1), Item{ID: "bounded", Text: raw, Diff: raw, Status: Running}})
		for _, text := range []string{m.items[0].Text, m.items[0].Diff} {
			if len(text) > maxTextBytes || !utf8.ValidString(text) || !strings.HasSuffix(text, "[display truncated]") {
				t.Fatalf("invalid normalized bound: %d bytes", len(text))
			}
		}
		m = step(m, TextDeltaMsg{scope(2), "bounded", raw})
		if len(m.items[0].Text) > maxTextBytes || !utf8.ValidString(m.items[0].Text) {
			t.Fatal("delta expanded normalized bound")
		}
	}
}

func TestBidiControlsAreRemovedWithoutBreakingEmoji(t *testing.T) {
	for _, control := range []rune{'\u061c', '\u200e', '\u200f', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e', '\u2066', '\u2067', '\u2068', '\u2069'} {
		if got := safeText("before" + string(control) + "after"); got != "beforeafter" {
			t.Fatalf("bidi control retained: U+%04X", control)
		}
	}
	if got := safeText("👩‍💻 e\u0301 العربية"); got != "👩‍💻 e\u0301 العربية" {
		t.Fatal("valid text was damaged")
	}
}

func TestEndTurnRejectsNonterminalAndUnknownStatus(t *testing.T) {
	for _, status := range []Status{"", Running, "\x1b[31mrunning", "future-status"} {
		m := New(Config{})
		m = step(m, BeginTurnMsg{scope(0)})
		m = step(m, EndTurnMsg{scope(1), status})
		if !m.active || m.status != Running || m.scope.Sequence != 0 {
			t.Fatalf("invalid end status accepted: %q", status)
		}
	}
}
