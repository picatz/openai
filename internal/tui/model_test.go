package tui

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/picatz/openai/internal/conversation"
)

func testModel() Model {
	return New(Config{Context: context.Background(), Session: conversation.New(conversation.Responses, "https://example.invalid/v1/", "test-model"), Backend: conversation.BackendFunc(func(context.Context, conversation.Request, func(string) error) (conversation.Result, error) {
		return conversation.Result{}, nil
	})})
}
func update(m Model, msg tea.Msg) (Model, tea.Cmd) {
	next, cmd := m.Update(msg)
	return next.(Model), cmd
}
func ctrl(r rune) tea.KeyPressMsg { return tea.KeyPressMsg{Code: r, Mod: tea.ModCtrl} }

func TestLayoutBounds(t *testing.T) {
	for _, size := range [][2]int{{1, 1}, {29, 11}, {30, 12}, {40, 16}, {80, 24}, {160, 60}} {
		m := testModel()
		m.session.Messages = []conversation.Message{{Role: "user", Content: strings.Repeat("very long word 日本語🙂 ", 100)}, {Role: "assistant", Content: "\x1b]52;c;evil\a\x1b[2Jresponse\x00\x7f"}}
		m, _ = update(m, tea.WindowSizeMsg{Width: size[0], Height: size[1]})
		view := m.View()
		lines := strings.Split(view.Content, "\n")
		if len(lines) > size[1] {
			t.Errorf("%v: height %d", size, len(lines))
		}
		for _, line := range lines {
			if ansi.StringWidth(line) > size[0] {
				t.Errorf("%v: overflow %q", size, line)
			}
		}
		if strings.Contains(view.Content, "evil") || strings.Contains(view.Content, "\x1b[2J") {
			t.Error("server terminal controls escaped sanitization")
		}
		if !view.AltScreen {
			t.Error("alternate screen not enabled")
		}
	}
}
func TestMultilineAndSingleSubmission(t *testing.T) {
	m := testModel()
	m.input.SetValue("first")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter, Mod: tea.ModAlt})
	m.input.InsertString("second")
	if m.input.Value() != "first\nsecond" {
		t.Fatalf("input=%q", m.input.Value())
	}
	m, cmd := update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	if cmd == nil || !m.busy || m.pending != "first\nsecond" {
		t.Fatalf("did not submit: %+v", m)
	}
	generation := m.generation
	m, again := update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	if again != nil || m.generation != generation {
		t.Error("duplicate submission")
	}
	m.cancelRequest()
}
func TestCancellationIgnoresStaleEvents(t *testing.T) {
	m := testModel()
	m.input.SetValue("prompt")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	generation := m.generation
	m, _ = update(m, streamMsg{generation: generation, delta: "partial"})
	m, _ = update(m, ctrl('c'))
	if m.busy || m.input.Value() != "prompt" || len(m.session.Messages) != 0 {
		t.Fatal("cancel changed durable history or lost draft")
	}
	m, _ = update(m, streamMsg{generation: generation, done: true, result: conversation.Result{Text: "late reply"}})
	if len(m.session.Messages) != 0 || m.partial != "partial" {
		t.Fatal("stale request modified session")
	}
}
func TestFailureRollsBackHistory(t *testing.T) {
	m := testModel()
	m.input.SetValue("prompt")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	m, _ = update(m, streamMsg{generation: m.generation, done: true, err: errors.New("synthetic error")})
	if len(m.session.Messages) != 0 || !strings.Contains(m.status, "synthetic error") || m.input.Value() != "prompt" {
		t.Fatal("failure did not preserve history and draft")
	}
}
func TestCompleteAndSaveFailure(t *testing.T) {
	m := testModel()
	store := conversation.Store{Dir: t.TempDir()}
	m.cfg.Store = &store
	m.input.SetValue("prompt")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	m.started = time.Now()
	m, save := update(m, streamMsg{generation: m.generation, done: true, result: conversation.Result{Text: "reply", ResponseID: "resp_test", Usage: conversation.Usage{Total: 7}}})
	if len(m.session.Messages) != 2 || m.session.Usage.Total != 7 || !m.dirty || save == nil {
		t.Fatal("response not committed in memory")
	}
	generation := m.generation
	m, _ = update(m, saveMsg{generation: generation, err: errors.New("disk full")})
	m.input.SetValue("next")
	m, cmd := update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	if m.busy || cmd != nil {
		t.Fatal("sent with unsaved history")
	}
	m, cmd = update(m, ctrl('s'))
	if cmd == nil {
		t.Fatal("save retry not available")
	}
	m, _ = update(m, cmd())
	if m.dirty || m.session.Revision != 1 {
		t.Fatalf("save failed: %s", m.status)
	}
	loaded, err := store.Load(m.session.ID)
	if err != nil || len(loaded.Messages) != 2 {
		t.Fatalf("load=%+v err=%v", loaded, err)
	}
}
func TestSessionSelectionRespectsEndpointAndAPI(t *testing.T) {
	m := testModel()
	m.cfg.Store = &conversation.Store{Dir: t.TempDir()}
	m, _ = update(m, ctrl('o'))
	compatible := conversation.New(conversation.Responses, m.session.Endpoint, "other-model")
	compatible.Messages = []conversation.Message{{Role: "user", Content: "saved"}}
	otherAPI := conversation.New(conversation.Chat, m.session.Endpoint, "test")
	otherEndpoint := conversation.New(conversation.Responses, "https://elsewhere.invalid/v1/", "test")
	m, _ = update(m, sessionsMsg{generation: m.pickerGeneration, sessions: []conversation.Session{compatible, otherAPI, otherEndpoint}})
	if len(m.sessions) != 1 {
		t.Fatal("incompatible sessions exposed for replay")
	}
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	if m.session.ID != compatible.ID || m.choosing {
		t.Fatal("session not selected")
	}
}
func TestScrollbackSurvivesStreaming(t *testing.T) {
	m := testModel()
	m.session.Messages = []conversation.Message{{Role: "assistant", Content: strings.Repeat("line\n", 100)}}
	m.refresh(true)
	m.input.SetValue("next")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	m.viewport.GotoTop()
	before := m.viewport.YOffset()
	m, _ = update(m, streamMsg{generation: m.generation, delta: "new text"})
	if m.viewport.YOffset() != before {
		t.Error("stream yanked scrollback to bottom")
	}
	m.cancelRequest()
}

func TestInitialFocusAcceptsTypingAndPaste(t *testing.T) {
	m := testModel()
	m.Init()
	m, _ = update(m, tea.KeyPressMsg{Code: 'h', Text: "h"})
	m, _ = update(m, tea.KeyPressMsg{Code: 'i', Text: "i"})
	m, _ = update(m, tea.PasteMsg{Content: " pasted\nline"})
	if m.input.Value() != "hi pasted\nline" {
		t.Fatalf("input ignored after initialization: %q", m.input.Value())
	}
}

type completionObserver struct {
	Model
	completed chan struct{}
}

func (m completionObserver) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	next, cmd := m.Model.Update(msg)
	m.Model = next.(Model)
	if event, ok := msg.(streamMsg); ok && event.done && event.err == nil {
		close(m.completed)
		return m, tea.Quit
	}
	return m, cmd
}
func TestProgramAcceptsRealInputEvents(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	received := make(chan conversation.Request, 1)
	m := testModel()
	m.cfg.Context = ctx
	m.cfg.Backend = conversation.BackendFunc(func(ctx context.Context, req conversation.Request, emit func(string) error) (conversation.Result, error) {
		received <- req
		emit("synthetic reply")
		return conversation.Result{Text: "synthetic reply"}, nil
	})
	completed := make(chan struct{})
	program := tea.NewProgram(completionObserver{Model: m, completed: completed}, tea.WithContext(ctx), tea.WithInput(strings.NewReader("hello\r")), tea.WithOutput(io.Discard), tea.WithoutRenderer(), tea.WithoutSignalHandler())
	final, err := program.Run()
	if err != nil {
		t.Fatal(err)
	}
	select {
	case req := <-received:
		if req.Messages[len(req.Messages)-1].Content != "hello" {
			t.Fatalf("request=%+v", req)
		}
	default:
		t.Fatal("typed input never reached backend")
	}
	if len(final.(completionObserver).session.Messages) != 2 {
		t.Fatal("completed reply not retained")
	}
}

func TestCompletionPreservesScrollback(t *testing.T) {
	m := testModel()
	m.session.Messages = []conversation.Message{{Role: "assistant", Content: strings.Repeat("line\n", 100)}}
	m.refresh(true)
	m.input.SetValue("next")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	m.viewport.GotoTop()
	m, _ = update(m, streamMsg{generation: m.generation, done: true, result: conversation.Result{Text: "complete"}})
	if m.viewport.YOffset() != 0 {
		t.Fatal("completion moved scrollback to bottom")
	}
}
func TestPickerTitlesAndStatusAreOneLine(t *testing.T) {
	m := testModel()
	m.choosing = true
	first := conversation.New(conversation.Responses, m.session.Endpoint, "test")
	first.Messages = []conversation.Message{{Role: "user", Content: strings.Repeat("a\n", 20)}}
	second := conversation.New(conversation.Responses, m.session.Endpoint, "test")
	second.Messages = []conversation.Message{{Role: "user", Content: "visible second"}}
	m.sessions = []conversation.Session{first, second}
	m.selected = 1
	m.status = "Ready\nmalicious status\nline"
	view := ansi.Strip(m.View().Content)
	if !strings.Contains(view, "visible second") || !strings.Contains(view, "Enter send") || !strings.Contains(view, "Ready malicious status line") {
		t.Fatalf("picker overflow:\n%s", view)
	}
}

func TestSessionPickerIgnoresStaleLoads(t *testing.T) {
	m := testModel()
	m.cfg.Store = &conversation.Store{Dir: t.TempDir()}
	m, _ = update(m, ctrl('o'))
	first := m.pickerGeneration
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEscape})
	m, _ = update(m, ctrl('o'))
	second := m.pickerGeneration
	newest := conversation.New(m.session.API, m.session.Endpoint, m.session.Model)
	m, _ = update(m, sessionsMsg{generation: second, sessions: []conversation.Session{newest}})
	m, _ = update(m, sessionsMsg{generation: first, sessions: nil, err: errors.New("old error")})
	if len(m.sessions) != 1 || m.sessions[0].ID != newest.ID || strings.Contains(m.status, "old error") {
		t.Fatal("old picker request overwrote current results")
	}
}

func TestQuitInvalidatesQueuedCompletion(t *testing.T) {
	m := testModel()
	m.cfg.Store = &conversation.Store{Dir: t.TempDir()}
	m.input.SetValue("prompt")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	generation := m.generation
	m, quit := update(m, ctrl('q'))
	if quit == nil {
		t.Fatal("quit was not scheduled")
	}
	m, save := update(m, streamMsg{generation: generation, done: true, result: conversation.Result{Text: "too late"}})
	if len(m.session.Messages) != 0 || m.busy || save != nil {
		t.Fatal("queued completion survived quitting")
	}
}
func TestCanceledProgramDiscardsQueuedCompletion(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	m := testModel()
	m.cfg.Context = ctx
	m.input.SetValue("prompt")
	m, _ = update(m, tea.KeyPressMsg{Code: tea.KeyEnter})
	generation := m.generation
	cancel()
	m, _ = update(m, streamMsg{generation: generation, done: true, result: conversation.Result{Text: "too late"}})
	if len(m.session.Messages) != 0 || m.busy {
		t.Fatal("completion survived process cancellation")
	}
}
