// Package codexworkspace hosts a Codex app-server session and prompt composer.
// The presentation model is separate from transport and has no approval controls.
package codexworkspace

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"

	"charm.land/bubbles/v2/textarea"
	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/picatz/openai/codex/appserver"
	"github.com/picatz/openai/internal/codexsession"
	"github.com/picatz/openai/internal/codextui"
)

const maxPromptBytes = 16 << 10

// Backend is an owned session. Calls must unblock on context cancellation or
// Close. Next is called by exactly one ordered consumer for the model lifetime.
type Backend interface {
	Info() codexsession.Info
	StartTurn(context.Context, string) error
	Next(context.Context) (codexsession.Event, error)
	Interrupt(context.Context) error
	Close() error
}

type Config struct {
	Context                context.Context
	Backend                Backend
	NoColor, ReducedMotion bool
}

// Model retains only the current-turn presentation and a bounded draft/notice.
// Start and interrupt acknowledgements never manufacture terminal turn state.
type Model struct {
	cfg                                                      Config
	info                                                     codexsession.Info
	workspace                                                codextui.Model
	input                                                    textarea.Model
	width, height                                            int
	inputFocus, help, inputShortened                         bool
	scope                                                    codextui.Scope
	request                                                  uint64
	busy, starting, active, interruptRequested, disconnected bool
	status, notice, pendingPrompt                            string
	transportErr                                             error
	outcomeUnknown                                           bool
}

type nextMsg struct {
	event codexsession.Event
	err   error
}
type startedMsg struct {
	request uint64
	err     error
}
type interruptedMsg struct {
	request uint64
	scope   codextui.Scope
	err     error
}

func New(cfg Config) Model {
	if cfg.Context == nil {
		cfg.Context = context.Background()
	}
	m := Model{cfg: cfg, width: 80, height: 24, inputFocus: true, status: "Ready"}
	if cfg.Backend != nil {
		m.info = cfg.Backend.Info()
	} else {
		m.disconnected = true
		m.status = "No app-server session connected"
	}
	m.workspace = codextui.New(codextui.Config{
		Runtime: true, Workspace: m.info.CWD, Model: m.info.Model,
		Capabilities: codextui.Capabilities{ToolOutput: true, Diffs: true, Plans: true},
		NoColor:      cfg.NoColor, ReducedMotion: cfg.ReducedMotion,
	})
	m.input = textarea.New()
	m.input.Placeholder = "Ask Codex…"
	m.input.Prompt = "› "
	m.input.ShowLineNumbers = false
	m.input.CharLimit = maxPromptBytes
	m.input.MaxHeight = 3
	m.input.MaxContentHeight = maxPromptBytes
	m.input.SetHeight(3)
	m.input.SetVirtualCursor(!cfg.NoColor)
	m.input.KeyMap.Paste.SetEnabled(false) // Never read the system clipboard implicitly.
	m.input.KeyMap.CopySelection.SetEnabled(false)
	styles := m.input.Styles()
	styles.Cursor.Blink = false
	m.input.SetStyles(styles)
	m.input.Focus()
	m.resize()
	return m
}

func (m Model) Init() tea.Cmd { return m.next() }

func (m Model) next() tea.Cmd {
	if m.disconnected || m.cfg.Backend == nil {
		return nil
	}
	return func() tea.Msg {
		event, err := m.cfg.Backend.Next(m.cfg.Context)
		return nextMsg{event: event, err: err}
	}
}

func (m *Model) display(msg tea.Msg) {
	model, _ := m.workspace.Update(msg)
	m.workspace = model.(codextui.Model)
}

func (m *Model) resize() {
	m.input.SetWidth(max(1, m.width-4))
	m.display(tea.WindowSizeMsg{Width: m.width, Height: max(0, m.height-7)})
}

func validID(id string) bool { return id != "" && len(id) <= 512 }
func terminal(s string) bool {
	switch codextui.Status(s) {
	case codextui.Completed, codextui.Failed, codextui.Declined, codextui.Interrupted, codextui.Incomplete:
		return true
	}
	return false
}
func scopeOf(e codexsession.Event) codextui.Scope {
	return codextui.Scope{Generation: e.Generation, Sequence: e.Sequence, ThreadID: e.ThreadID, TurnID: e.TurnID}
}
func (m Model) accepts(s codextui.Scope) bool {
	return m.active && s.Generation == m.scope.Generation && s.ThreadID == m.scope.ThreadID && s.TurnID == m.scope.TurnID && s.Sequence > m.scope.Sequence
}

func (m *Model) event(e codexsession.Event) {
	if e.Kind == "notice" {
		// Notices are independent of a terminal turn notification. An unsupported
		// server request must remain visible even when completion follows it.
		m.notice = bounded(e.Notice, 2048)
		return
	}
	scope := scopeOf(e)
	if e.Kind == "begin" {
		if !m.busy || m.active || scope.Generation <= m.scope.Generation || !validID(scope.ThreadID) || !validID(scope.TurnID) || scope.ThreadID != m.info.ThreadID {
			return
		}
		m.scope, m.active, m.outcomeUnknown = scope, true, false
		m.display(codextui.BeginTurnMsg{Scope: scope})
		if !m.starting {
			m.status = "Running"
		}
		return
	}
	if !m.accepts(scope) {
		return
	}
	switch e.Kind {
	case "item":
		item := codextui.Item{ID: e.Item.ID, Kind: codextui.Kind(e.Item.Kind), Title: e.Item.Title, Text: e.Item.Text, Diff: e.Item.Diff, Status: codextui.Status(e.Item.Status)}
		for _, f := range e.Item.Files[:min(len(e.Item.Files), 64)] {
			item.Files = append(item.Files, codextui.FileChange{Path: f.Path, Kind: f.Kind})
		}
		m.display(codextui.ItemMsg{Scope: scope, Item: item})
	case "delta":
		m.display(codextui.TextDeltaMsg{Scope: scope, ItemID: e.ItemID, Delta: e.Delta})
	case "end":
		if !terminal(e.Status) {
			return
		}
		m.display(codextui.EndTurnMsg{Scope: scope, Status: codextui.Status(e.Status)})
		m.active, m.busy, m.interruptRequested = false, false, false
		m.status = "Turn " + e.Status
		m.outcomeUnknown = e.Status == string(codextui.Incomplete)
		if m.outcomeUnknown {
			m.status = "Turn incomplete; outcome unknown"
		}
		m.pendingPrompt = ""
	default:
		return
	}
	m.scope = scope
}

func (m *Model) start() tea.Cmd {
	if m.busy || m.starting || m.disconnected || m.cfg.Backend == nil {
		return nil
	}
	prompt := bounded(m.input.Value(), maxPromptBytes)
	if len(m.input.Value()) > maxPromptBytes {
		m.input.SetValue(prompt)
		m.inputShortened = true
		return nil // The shortened draft needs a separate explicit submission.
	}
	if strings.TrimSpace(prompt) == "" {
		return nil
	}
	m.request++
	request := m.request
	m.busy, m.starting, m.interruptRequested, m.outcomeUnknown = true, true, false, false
	m.status = "Starting turn; waiting for acknowledgement"
	m.pendingPrompt = prompt
	m.input.Reset()
	m.inputShortened = false
	return func() tea.Msg {
		return startedMsg{request: request, err: m.cfg.Backend.StartTurn(m.cfg.Context, prompt)}
	}
}

func (m *Model) interrupt() tea.Cmd {
	if m.starting {
		if !m.busy {
			return nil
		}
		m.status = "Start pending; turn outcome uncertain. Wait or Ctrl+Q to close locally"
		return nil
	}
	if !m.active || m.interruptRequested || m.disconnected {
		return nil
	}
	m.interruptRequested = true
	m.status = "Interrupt requested; awaiting final turn state"
	request, scope := m.request, m.scope
	return func() tea.Msg {
		return interruptedMsg{request: request, scope: scope, err: m.cfg.Backend.Interrupt(m.cfg.Context)}
	}
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = max(0, msg.Width), max(0, msg.Height)
		m.resize()
		return m, nil
	case nextMsg:
		if m.disconnected {
			return m, nil
		}
		if msg.err != nil {
			unfinished := m.busy || m.active || m.outcomeUnknown
			m.disconnected, m.busy = true, false
			m.transportErr = msg.err
			m.status = "Connection closed; no further prompts can be sent"
			if unfinished {
				m.status = "Connection lost; outcome unknown (incomplete)"
				if errors.Is(msg.err, io.EOF) {
					m.transportErr = io.ErrUnexpectedEOF
				}
			}
			if m.active {
				// This is a local incomplete display, never server-confirmed success
				// or interruption. Keep partial work visible after transport loss.
				m.scope.Sequence++
				m.display(codextui.EndTurnMsg{Scope: m.scope, Status: codextui.Incomplete})
			}
			m.active = false
			return m, nil
		}
		m.event(msg.event)
		return m, m.next()
	case startedMsg:
		if msg.request != m.request || !m.starting {
			return m, nil
		}
		m.starting = false
		if m.disconnected {
			return m, nil
		}
		if msg.err != nil {
			m.addNotice("Start acknowledgement failed: " + msg.err.Error())
			if m.busy {
				m.status = "Start outcome uncertain; awaiting turn state or Ctrl+Q to close"
				var rpc *appserver.RPCError
				if !m.active && errors.As(msg.err, &rpc) {
					m.busy = false
					m.status = "Turn rejected; edit the draft and resubmit"
					if m.input.Value() == "" {
						m.input.SetValue(m.pendingPrompt)
					}
				}
			}
		} else if m.busy {
			m.status = "Waiting for turn to begin"
			if m.active {
				m.status = "Running"
			}
		}
		return m, nil
	case interruptedMsg:
		if msg.request != m.request || !m.active || msg.scope.Generation != m.scope.Generation || msg.scope.TurnID != m.scope.TurnID {
			return m, nil
		}
		if msg.err != nil {
			m.addNotice("Interrupt acknowledgement failed: " + msg.err.Error())
			m.status = "Interrupt outcome uncertain; awaiting final turn state"
		}
		return m, nil
	case tea.KeyPressMsg:
		switch msg.String() {
		case "ctrl+q":
			return m, tea.Quit
		case "ctrl+c":
			cmd := m.interrupt()
			return m, cmd
		case "f1":
			m.help = !m.help
			return m, nil
		case "esc":
			m.help, m.inputFocus = false, false
			m.input.Blur()
			m.display(msg)
			return m, nil
		case "f6":
			m.help = false
			m.inputFocus = !m.inputFocus
			if m.inputFocus {
				m.input.Focus()
			} else {
				m.input.Blur()
			}
			return m, nil
		}
		if m.help {
			return m, nil
		}
		if !m.inputFocus {
			// F1 is host-owned, and '?' remains ordinary composer text.
			if msg.String() != "?" {
				m.display(msg)
			}
			return m, nil
		}
		switch msg.String() {
		case "enter":
			cmd := m.start()
			return m, cmd
		case "alt+enter", "ctrl+j":
			before := len(m.input.Value())
			m.input.InsertString("\n")
			if len(m.input.Value()) <= before {
				m.inputShortened = true
			}
			m.boundInput()
			return m, nil
		}
		// Clamp even synthetic multi-character key messages before textarea.
		if len(msg.Text) > maxPromptBytes || len(m.input.Value())-len(m.input.SelectedText())+len(msg.Text) > maxPromptBytes {
			m.inputShortened = true
		}
		msg.Text = bounded(msg.Text, maxPromptBytes)
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		m.boundInput()
		return m, cmd
	case tea.PasteMsg:
		if !m.inputFocus || m.help {
			return m, nil
		}
		if len(msg.Content) > maxPromptBytes {
			m.inputShortened = true
		}
		msg.Content = bounded(strings.ReplaceAll(msg.Content, "\r\n", "\n"), maxPromptBytes)
		expected := len(m.input.Value()) - len(m.input.SelectedText()) + len(msg.Content)
		var cmd tea.Cmd
		m.input, cmd = m.input.Update(msg)
		if len(m.input.Value()) < expected {
			m.inputShortened = true
		}
		m.boundInput()
		return m, cmd
	case tea.MouseWheelMsg:
		if !m.inputFocus && !m.help {
			m.display(msg)
		}
	}
	return m, nil
}

func (m *Model) addNotice(s string) {
	if m.notice != "" {
		s = m.notice + " · " + s
	}
	m.notice = bounded(s, 2048)
}

func (m *Model) boundInput() {
	if s := m.input.Value(); len(s) > maxPromptBytes {
		m.input.SetValue(bounded(s, maxPromptBytes))
		m.inputShortened = true
	}
}

// Bound raw bytes before stripping sequences; never retain remote control codes
// or bidi controls. Prompts are capped without adding synthetic prompt content.
func bounded(s string, n int) string {
	if len(s) > n {
		s = s[:n]
	}
	s = ansi.Strip(s)
	s = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) && r != '\n' && r != '\t' || unicode.Is(unicode.Bidi_Control, r) {
			return -1
		}
		return r
	}, strings.ToValidUTF8(s, "�"))
	if len(s) > n {
		s = s[:n]
	}
	for !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}

func line(s string, width int) string {
	return ansi.Truncate(strings.Join(strings.Fields(s), " "), max(0, width), "…")
}

func (m Model) View() tea.View {
	if m.width <= 0 || m.height <= 0 {
		return tea.NewView("")
	}
	if m.width < 30 || m.height < 17 {
		v := tea.NewView(line("Resize to 30 × 17 · Ctrl+Q quits", m.width))
		v.AltScreen = true
		return v
	}
	body := m.workspace.View().Content
	if m.help {
		help := []string{"CODEX WORKSPACE · KEYBOARD", "", "Enter         Send prompt", "Alt+Enter / Ctrl+J  Insert newline", "Bracketed paste    Insert without sending", "F6            Composer / workspace focus", "Esc           Focus workspace", "Tab           Item / detail focus (workspace)", "↑/↓ or j/k    Select item (workspace)", "PgUp/PgDn     Scroll detail (workspace)", "Ctrl+Home/End Top / follow detail (workspace)", "Ctrl+C        Interrupt known active turn", "Ctrl+Q        Close local session and quit", "F1            Toggle help", "", "Approvals and input requests are unsupported."}
		body = strings.Join(help[:min(len(help), m.height-7)], "\n")
	}
	rows := strings.Split(body, "\n")
	for len(rows) < m.height-7 {
		rows = append(rows, "")
	}
	rows = rows[:m.height-7]
	focus := "COMPOSER"
	if !m.inputFocus {
		focus = "COMPOSER · F6 to edit"
	}
	statusLine := focus + " · " + m.status
	if m.width < 70 {
		statusLine = m.status
	}
	rows = append(rows, "  "+statusLine)
	input := strings.Split(m.input.View(), "\n")
	for i := 0; i < 3; i++ {
		s := ""
		if i < len(input) {
			s = input[i]
		}
		rows = append(rows, "  "+s)
	}
	notice := m.notice
	if notice == "" {
		notice = fmt.Sprintf("Policy: %s · Reviewer: %s · Sandbox: %s", bounded(m.info.ApprovalPolicy, 128), bounded(m.info.ApprovalsReviewer, 128), bounded(m.info.Sandbox, 128))
	}
	hints := "Enter send · Alt+Enter/Ctrl+J newline · F6 workspace · F1 help"
	quitHint := "Ctrl+C interrupt · Ctrl+Q close locally · draft is not saved"
	if m.width < 70 {
		hints = "Enter send · F1 help · F6 focus"
		quitHint = "Ctrl+C interrupt · Ctrl+Q quit"
		if m.width < 40 {
			quitHint = "Ctrl+C · Ctrl+Q quit"
		}
	}
	if m.inputShortened {
		hints = "Input shortened to 16 KiB or line limit; review draft before sending"
		if m.width < 70 {
			hints = "Draft shortened · review before Enter"
		}
	}
	rows = append(rows, "  "+line(notice, m.width-4), "  "+hints, "  "+quitHint)
	for i, row := range rows {
		if m.cfg.NoColor {
			row = ansi.Strip(row)
		}
		rows[i] = ansi.Truncate(row, m.width, "…")
	}
	v := tea.NewView(strings.Join(rows, "\n"))
	v.AltScreen = true
	if m.cfg.NoColor && m.inputFocus && !m.help {
		// Stripping styles also removes the virtual cursor. Keep editing usable
		// in plain mode with a terminal-owned, uncolored static cursor instead.
		if cursor := m.input.Cursor(); cursor != nil {
			cursor.X += 2
			cursor.Y += m.height - 6
			cursor.Color = nil
			cursor.Blink = false
			v.Cursor = cursor
		}
	}
	return v
}
