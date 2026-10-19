package codexworkspace

import (
	"context"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/picatz/openai/codex/appserver"
	"github.com/picatz/openai/internal/codexsession"
)

type fakeBackend struct {
	mu                            sync.Mutex
	prompts                       []string
	interrupts, closes, nextCalls int
	events                        chan nextMsg
	startErr, interruptErr        error
}

func newFake() *fakeBackend { return &fakeBackend{events: make(chan nextMsg, 16)} }
func (b *fakeBackend) Info() codexsession.Info {
	return codexsession.Info{ThreadID: "thread", CWD: "/synthetic/workspace", Model: "test-model", ApprovalPolicy: "on-request", Sandbox: "workspace-write", ApprovalsReviewer: "user"}
}
func (b *fakeBackend) StartTurn(_ context.Context, prompt string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.prompts = append(b.prompts, prompt)
	return b.startErr
}
func (b *fakeBackend) Interrupt(context.Context) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.interrupts++
	return b.interruptErr
}
func (b *fakeBackend) Close() error { b.mu.Lock(); defer b.mu.Unlock(); b.closes++; return nil }
func (b *fakeBackend) Next(ctx context.Context) (codexsession.Event, error) {
	b.mu.Lock()
	b.nextCalls++
	b.mu.Unlock()
	select {
	case e := <-b.events:
		return e.event, e.err
	case <-ctx.Done():
		return codexsession.Event{}, ctx.Err()
	}
}
func update(m Model, msg tea.Msg) (Model, tea.Cmd) {
	model, cmd := m.Update(msg)
	return model.(Model), cmd
}
func key(code rune, mod tea.KeyMod) tea.KeyPressMsg {
	msg := tea.KeyPressMsg{Code: code, Mod: mod}
	if mod == 0 && code > 31 && code < 127 {
		msg.Text = string(code)
	}
	return msg
}
func emit(m Model, e codexsession.Event) Model { m, _ = update(m, nextMsg{event: e}); return m }
func event(kind, turn string, generation, sequence uint64) codexsession.Event {
	return codexsession.Event{Kind: kind, ThreadID: "thread", TurnID: turn, Generation: generation, Sequence: sequence}
}
func begin(t *testing.T, m Model, turn string, generation uint64) (Model, tea.Cmd) {
	t.Helper()
	m.input.SetValue("synthetic prompt")
	m, cmd := update(m, key(tea.KeyEnter, 0))
	if cmd == nil || !m.busy || !m.starting {
		t.Fatal("submission did not enter pending state")
	}
	return emit(m, event("begin", turn, generation, 1)), cmd
}
func end(m Model, turn string, generation, sequence uint64, status string) Model {
	e := event("end", turn, generation, sequence)
	e.Status = status
	return emit(m, e)
}

func TestComposerPasteNewlineAndFocus(t *testing.T) {
	b := newFake()
	m := New(Config{Context: t.Context(), Backend: b, NoColor: true})
	for _, msg := range []tea.Msg{key('j', 0), key('k', 0), key('?', 0), key('j', tea.ModCtrl), key(tea.KeyEnter, tea.ModAlt), tea.PasteMsg{Content: "one\ntwo\nthree\nfour\r\nfive"}} {
		m, _ = update(m, msg)
	}
	if m.help || !m.inputFocus || m.busy || len(b.prompts) != 0 {
		t.Fatal("composer text/paste was intercepted")
	}
	want := "jk?\n\none\ntwo\nthree\nfour\nfive"
	if m.input.Value() != want {
		t.Fatalf("input = %q; want %q", m.input.Value(), want)
	}
	m, cmd := update(m, key(tea.KeyEnter, 0))
	if cmd == nil {
		t.Fatal("Enter did not submit")
	}
	m, _ = update(m, cmd())
	if len(b.prompts) != 1 || b.prompts[0] != want {
		t.Fatalf("prompts = %q", b.prompts)
	}
	m, _ = update(m, key(tea.KeyEscape, 0))
	if m.inputFocus {
		t.Fatal("Esc did not focus workspace")
	}
	m, _ = update(m, key(tea.KeyTab, 0))
	m, _ = update(m, key(tea.KeyF6, 0))
	if !m.inputFocus {
		t.Fatal("F6 did not return composer")
	}
	m, _ = update(m, key(tea.KeyF1, 0))
	if !m.help || !strings.Contains(m.View().Content, "KEYBOARD") {
		t.Fatal("F1 help missing")
	}
}

func TestEarlyBeginEndAndRepeatedTurns(t *testing.T) {
	b := newFake()
	m := New(Config{Context: t.Context(), Backend: b})
	m, start := begin(t, m, "first", 1)
	m, interrupt := update(m, key('c', tea.ModCtrl))
	if interrupt != nil || m.interruptRequested || !strings.Contains(m.status, "Start pending") {
		t.Fatal("pending start allowed interruption")
	}
	m = end(m, "first", 1, 2, "completed")
	if m.busy || m.active || !m.starting {
		t.Fatal("early completion was not retained")
	}
	m.input.SetValue("second")
	m, cmd := update(m, key(tea.KeyEnter, 0))
	if cmd != nil {
		t.Fatal("new start raced pending start RPC")
	}
	m, _ = update(m, start())
	if m.status != "Turn completed" || m.busy || m.starting {
		t.Fatalf("late start ack resurrected turn: %+v", m)
	}
	m, cmd = update(m, key(tea.KeyEnter, 0))
	if cmd == nil {
		t.Fatal("matching completion did not allow second turn")
	}
	m, _ = update(m, cmd())
	m = emit(m, event("begin", "second", 2, 1))
	m = end(m, "first", 1, 99, "completed")
	if !m.active {
		t.Fatal("foreign terminal event ended current turn")
	}
	m = end(m, "second", 2, 2, "completed")
	if m.busy || len(b.prompts) != 2 {
		t.Fatal("repeated same-thread turn failed")
	}
}

func TestInterruptAcknowledgementNeverCompletes(t *testing.T) {
	b := newFake()
	m, start := begin(t, New(Config{Context: t.Context(), Backend: b}), "turn", 1)
	m, _ = update(m, start())
	m, interrupt := update(m, key('c', tea.ModCtrl))
	if interrupt == nil {
		t.Fatal("active turn did not interrupt")
	}
	for i := 0; i < 5; i++ {
		var cmd tea.Cmd
		m, cmd = update(m, key('c', tea.ModCtrl))
		if cmd != nil {
			t.Fatal("duplicate interrupt")
		}
	}
	m, _ = update(m, interrupt())
	if !m.active || !m.busy || b.interrupts != 1 || !strings.Contains(m.status, "awaiting final") {
		t.Fatal("RPC ack was treated as completion")
	}
	m = end(m, "turn", 1, 2, "interrupted")
	if m.active || m.busy || m.status != "Turn interrupted" {
		t.Fatal("matching terminal state missing")
	}
	m, _ = update(m, interruptedMsg{request: 1, scope: scopeOf(event("begin", "turn", 1, 1)), err: errors.New("late ack")})
	if m.status != "Turn interrupted" {
		t.Fatal("late interrupt changed completed turn")
	}
}

func TestFailureNoticeAndTransportState(t *testing.T) {
	b := newFake()
	m, start := begin(t, New(Config{Context: t.Context(), Backend: b, NoColor: true}), "turn", 1)
	m, _ = update(m, start())
	e := event("item", "turn", 1, 2)
	e.Item = codexsession.Item{ID: "message", Kind: "message", Title: "Response", Text: "partial\x1b[31m output\u202e", Status: "running"}
	m = emit(m, e)
	m = emit(m, codexsession.Event{Kind: "notice", Notice: "Unsupported approval request; no permission was granted\x1b]0;bad\a"})
	m, cmd := update(m, nextMsg{err: io.EOF})
	frame := m.View().Content
	if cmd != nil || !m.disconnected || m.active || !errors.Is(m.transportErr, io.ErrUnexpectedEOF) {
		t.Fatal("EOF did not stop single consumer")
	}
	for _, want := range []string{"incomplete", "outcome unknown", "Unsupported approval", "partial output"} {
		if !strings.Contains(frame, want) {
			t.Errorf("frame missing %q:\n%s", want, frame)
		}
	}
	if strings.ContainsAny(frame, "\x1b\u202e") {
		t.Fatal("untrusted controls survived")
	}
	m.input.SetValue("cannot send")
	m, cmd = update(m, key(tea.KeyEnter, 0))
	if cmd != nil {
		t.Fatal("disconnected session accepted prompt")
	}
}

func TestNoticeSurvivesCompletionAndLateStartError(t *testing.T) {
	m, _ := begin(t, New(Config{Context: t.Context(), Backend: newFake()}), "turn", 1)
	m = emit(m, codexsession.Event{Kind: "notice", Notice: "Unsupported approval request"})
	m = end(m, "turn", 1, 2, "failed")
	if !strings.Contains(m.View().Content, "Unsupported approval request") {
		t.Fatal("completion swallowed unsupported request notice")
	}
	m, _ = update(m, startedMsg{request: m.request, err: errors.New("ack lost")})
	if !strings.Contains(m.notice, "Unsupported approval request") {
		t.Fatal("late start error swallowed unsupported notice")
	}
	if m.busy || m.status != "Turn failed" {
		t.Fatal("late start error revived terminal turn")
	}
}

func TestDefiniteStartRejectionAndUncertainStart(t *testing.T) {
	for _, uncertain := range []bool{false, true} {
		b := newFake()
		b.startErr = &appserver.RPCError{Code: -32000, Message: "synthetic rejection"}
		if uncertain {
			b.startErr = errors.New("reply lost")
		}
		m := New(Config{Context: t.Context(), Backend: b})
		m.input.SetValue("retry this")
		m, cmd := update(m, key(tea.KeyEnter, 0))
		m, _ = update(m, cmd())
		if m.busy != uncertain {
			t.Fatalf("busy = %v for uncertain=%v", m.busy, uncertain)
		}
		if !uncertain && m.input.Value() != "retry this" {
			t.Fatal("definite rejection lost draft")
		}
	}
}

func TestOneOrderedNextChain(t *testing.T) {
	b := newFake()
	m := New(Config{Context: t.Context(), Backend: b})
	cmd := m.Init()
	for i := 0; i < 3; i++ {
		b.events <- nextMsg{event: codexsession.Event{Kind: "notice", Notice: "synthetic"}}
		m, cmd = update(m, cmd())
		if cmd == nil || b.nextCalls != i+1 {
			t.Fatal("reader chain did not schedule exactly once")
		}
		m.input.SetValue("draft")
		if i == 0 {
			var start tea.Cmd
			m, start = update(m, key(tea.KeyEnter, 0))
			m, _ = update(m, start())
		}
		if b.nextCalls != i+1 {
			t.Fatal("non-event update started a second reader")
		}
	}
	b.events <- nextMsg{err: io.EOF}
	m, cmd = update(m, cmd())
	if cmd != nil {
		t.Fatal("EOF rescheduled reader")
	}
}

func TestBoundedFramesAndPrompt(t *testing.T) {
	for _, plain := range []bool{true, false} {
		m := New(Config{Context: t.Context(), Backend: newFake(), NoColor: plain, ReducedMotion: true})
		m, _ = update(m, tea.PasteMsg{Content: strings.Repeat("界\u202e\x1b[32m", maxPromptBytes)})
		if s := m.input.Value(); len(s) > maxPromptBytes || !utf8.ValidString(s) || strings.ContainsAny(s, "\x1b\u202e") {
			t.Fatal("prompt not bounded or sanitized")
		}
		for _, size := range [][2]int{{0, 0}, {1, 1}, {20, 5}, {30, 17}, {40, 20}, {80, 24}, {120, 32}, {200, 60}} {
			m, _ = update(m, tea.WindowSizeMsg{Width: size[0], Height: size[1]})
			frame := m.View().Content
			if frame == "" {
				continue
			}
			rows := strings.Split(frame, "\n")
			if len(rows) > size[1] {
				t.Fatalf("height overflow %v", size)
			}
			for _, row := range rows {
				if ansi.StringWidth(row) > size[0] {
					t.Fatalf("width overflow %v", size)
				}
			}
			if plain && strings.Contains(frame, "\x1b") {
				t.Fatal("NoColor emitted escape")
			}
		}
	}
}

func TestWorkspaceScrollPreservesSelectedDetail(t *testing.T) {
	m, start := begin(t, New(Config{Context: t.Context(), Backend: newFake(), NoColor: true}), "turn", 1)
	m, _ = update(m, start())
	e := event("item", "turn", 1, 2)
	e.Item = codexsession.Item{ID: "message", Kind: "message", Title: "Long response", Text: strings.Repeat("body line\n", 50) + "BOTTOM", Status: "running"}
	m = emit(m, e)
	m, _ = update(m, key(tea.KeyEscape, 0))
	m, _ = update(m, key(tea.KeyHome, tea.ModCtrl))
	if strings.Contains(m.View().Content, "BOTTOM") {
		t.Fatal("could not scroll to top")
	}
	e = event("delta", "turn", 1, 3)
	e.ItemID, e.Delta = "message", " NEW"
	m = emit(m, e)
	if strings.Contains(m.View().Content, "BOTTOM") {
		t.Fatal("stream stole scrollback")
	}
	m, _ = update(m, key(tea.KeyEnd, tea.ModCtrl))
	if !strings.Contains(m.View().Content, "BOTTOM NEW") {
		t.Fatal("follow bottom failed")
	}
}

func TestPlainComposerCursor(t *testing.T) {
	m := New(Config{Context: t.Context(), Backend: newFake(), NoColor: true})
	view := m.View()
	if view.Cursor == nil || view.Cursor.Color != nil || view.Cursor.Blink || view.Cursor.Y < 18 || view.Cursor.Y > 20 {
		t.Fatal("plain composer lost its visible static cursor")
	}
	m, _ = update(m, key(tea.KeyF6, 0))
	if m.View().Cursor != nil {
		t.Fatal("workspace focus retained input cursor")
	}
	m, _ = update(m, key(tea.KeyF6, 0))
	m, _ = update(m, key(tea.KeyF1, 0))
	if m.View().Cursor != nil {
		t.Fatal("help retained input cursor")
	}
}

func TestEOFAfterEarlyCompletionKeepsKnownOutcome(t *testing.T) {
	m, _ := begin(t, New(Config{Context: t.Context(), Backend: newFake(), NoColor: true}), "turn", 1)
	m = end(m, "turn", 1, 2, "completed")
	m, _ = update(m, nextMsg{err: io.EOF})
	if !errors.Is(m.transportErr, io.EOF) || strings.Contains(m.status, "unknown") || !strings.Contains(m.View().Content, "completed") {
		t.Fatal("transport close discarded a confirmed terminal outcome")
	}
}

func TestEOFBeforeBeginIsUncertainEvenAfterStartAck(t *testing.T) {
	for _, acknowledged := range []bool{false, true} {
		m := New(Config{Context: t.Context(), Backend: newFake(), NoColor: true})
		m.input.SetValue("prompt")
		m, cmd := update(m, key(tea.KeyEnter, 0))
		if acknowledged {
			m, _ = update(m, cmd())
		}
		m, _ = update(m, nextMsg{err: io.EOF})
		if !errors.Is(m.transportErr, io.ErrUnexpectedEOF) || !strings.Contains(m.status, "unknown") {
			t.Fatal("unconfirmed turn outcome was lost")
		}
	}
}

func TestEOFAfterAdapterTerminalRetainsOutcomeCertainty(t *testing.T) {
	for _, status := range []string{"incomplete", "completed", "interrupted", "failed", "declined"} {
		t.Run(status, func(t *testing.T) {
			m, start := begin(t, New(Config{Context: t.Context(), Backend: newFake(), NoColor: true}), "turn", 1)
			m, _ = update(m, start())
			// Session.stopLocked emits an incomplete end before Next returns EOF.
			m = end(m, "turn", 1, 2, status)
			unknown := status == "incomplete"
			if m.busy || m.active || m.outcomeUnknown != unknown {
				t.Fatal("terminal event did not preserve outcome certainty")
			}
			m, _ = update(m, nextMsg{err: io.EOF})
			want := io.EOF
			if unknown {
				want = io.ErrUnexpectedEOF
			}
			if !errors.Is(m.transportErr, want) || strings.Contains(m.status, "unknown") != unknown {
				t.Fatalf("status=%q error=%v after %s end", m.status, m.transportErr, status)
			}
			if !strings.Contains(m.View().Content, status) {
				t.Fatal("transport closure discarded the terminal display")
			}
		})
	}
}

func TestNewTurnClearsPreviousIncompleteOutcome(t *testing.T) {
	m, start := begin(t, New(Config{Context: t.Context(), Backend: newFake(), NoColor: true}), "first", 1)
	m, _ = update(m, start())
	m = end(m, "first", 1, 2, "incomplete")
	m.input.SetValue("next prompt")
	m, start = update(m, key(tea.KeyEnter, 0))
	if start == nil || m.outcomeUnknown {
		t.Fatal("new explicit submission retained the previous outcome flag")
	}
	m, _ = update(m, start())
	m = emit(m, event("begin", "second", 2, 1))
	m = end(m, "second", 2, 2, "completed")
	m, _ = update(m, nextMsg{err: io.EOF})
	if !errors.Is(m.transportErr, io.EOF) || strings.Contains(m.status, "unknown") {
		t.Fatal("prior incomplete turn contaminated the newly confirmed outcome")
	}
}
