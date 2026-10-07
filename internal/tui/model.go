// Package tui provides the Bubble Tea v2 conversation UI. All network and disk
// operations run as commands; Update owns state and rejects stale request events.
package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/picatz/openai/internal/conversation"
)

type Config struct {
	Context   context.Context
	Backend   conversation.Backend
	Session   conversation.Session
	Store     *conversation.Store
	WebSearch bool
}

type Model struct {
	pickerGeneration                           int
	initCmd                                    tea.Cmd
	cfg                                        Config
	session                                    conversation.Session
	input                                      textarea.Model
	viewport                                   viewport.Model
	spinner                                    spinner.Model
	width, height                              int
	generation                                 int
	busy, saving, dirty, choosing, confirmQuit bool
	cancel                                     context.CancelFunc
	events                                     <-chan streamMsg
	pending, partial, status                   string
	started                                    time.Time
	sessions                                   []conversation.Session
	selected                                   int
}

type streamMsg struct {
	generation int
	delta      string
	result     conversation.Result
	err        error
	done       bool
}
type saveMsg struct {
	generation int
	session    conversation.Session
	err        error
}
type sessionsMsg struct {
	generation int
	sessions   []conversation.Session
	err        error
}

func New(cfg Config) Model {
	if cfg.Context == nil {
		cfg.Context = context.Background()
	}
	input := textarea.New()
	input.Placeholder = "Ask anything…"
	input.Prompt = "› "
	input.ShowLineNumbers = false
	input.CharLimit = 1 << 20
	input.MaxHeight = 6
	input.SetHeight(4)
	input.SetVirtualCursor(true)
	input.KeyMap.Paste.SetEnabled(false) // Bracketed terminal paste works; never read the clipboard implicitly.
	focus := input.Focus()
	m := Model{initCmd: focus, cfg: cfg, session: cfg.Session.Clone(), input: input, viewport: viewport.New(), spinner: spinner.New(spinner.WithSpinner(spinner.Dot)), width: 80, height: 24, status: "Ready"}
	m.resize()
	m.refresh(true)
	return m
}
func (m Model) Init() tea.Cmd { return m.initCmd }
func waitEvent(ch <-chan streamMsg) tea.Cmd {
	return func() tea.Msg {
		event, ok := <-ch
		if !ok {
			return nil
		}
		return event
	}
}
func (m *Model) start() tea.Cmd {
	if m.busy || m.saving {
		return nil
	}
	if m.dirty {
		m.status = "History not saved. Ctrl+S retries saving before another request"
		return nil
	}
	prompt := m.input.Value()
	if strings.TrimSpace(prompt) == "" {
		return nil
	}
	if m.cfg.Backend == nil {
		m.status = "No API backend configured"
		return nil
	}
	m.generation++
	generation := m.generation
	ctx, cancel := context.WithCancel(m.cfg.Context)
	m.cancel = cancel
	ch := make(chan streamMsg, 16)
	m.events = ch
	request := conversation.Request{API: m.session.API, Model: m.session.Model, Messages: append(append([]conversation.Message(nil), m.session.Messages...), conversation.Message{Role: "user", Content: prompt}), ResponsesItems: conversation.CloneItems(m.session.ResponsesItems), WebSearch: m.cfg.WebSearch}
	m.busy = true
	m.pending = prompt
	m.partial = ""
	m.status = "Receiving response"
	m.started = time.Now()
	m.input.SetValue("")
	m.refresh(true)
	backend := m.cfg.Backend
	return tea.Batch(m.spinner.Tick, func() tea.Msg {
		go func() {
			defer close(ch)
			result, err := backend.Generate(ctx, request, func(delta string) error {
				select {
				case ch <- streamMsg{generation: generation, delta: delta}:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			})
			select {
			case ch <- streamMsg{generation: generation, result: result, err: err, done: true}:
			case <-ctx.Done():
			}
		}()
		return waitEvent(ch)()
	})
}
func (m *Model) save() tea.Cmd {
	if m.cfg.Store == nil {
		m.dirty = false
		return nil
	}
	m.saving = true
	m.status = "Saving local history"
	session := m.session.Clone()
	store := *m.cfg.Store
	generation := m.generation
	return func() tea.Msg {
		err := store.Save(&session)
		return saveMsg{generation: generation, session: session, err: err}
	}
}
func (m *Model) cancelRequest() {
	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}
	m.generation++
	m.busy = false
	if m.input.Value() == "" {
		m.input.SetValue(m.pending)
	}
	m.status = "Canceled. Partial output is not saved; edit the draft to retry"
	m.refresh(false)
}
func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.resize()
		m.refresh(false)
		return m, nil
	case streamMsg:
		if m.cfg.Context.Err() != nil {
			m.cancelRequest()
			return m, tea.Quit
		}
		if msg.generation != m.generation || !m.busy {
			return m, nil
		}
		if msg.delta != "" {
			m.partial += msg.delta
			m.refresh(m.viewport.AtBottom())
		}
		if !msg.done {
			return m, waitEvent(m.events)
		}
		m.busy = false
		if m.cancel != nil {
			m.cancel()
			m.cancel = nil
		}
		if msg.err != nil {
			m.status = "Request failed: " + msg.err.Error()
			if errors.Is(msg.err, context.Canceled) {
				m.status = "Request canceled"
			}
			if m.input.Value() == "" {
				m.input.SetValue(m.pending)
			}
			m.refresh(false)
			return m, nil
		}
		keepAtBottom := m.viewport.AtBottom()
		m.session.Messages = append(m.session.Messages, conversation.Message{Role: "user", Content: m.pending}, conversation.Message{Role: "assistant", Content: msg.result.Text})
		m.session.ResponsesItems = conversation.CloneItems(msg.result.ResponsesItems)
		m.session.LastResponseID = msg.result.ResponseID
		m.session.Usage.Input += msg.result.Usage.Input
		m.session.Usage.Output += msg.result.Usage.Output
		m.session.Usage.Total += msg.result.Usage.Total
		m.pending = ""
		m.partial = ""
		m.dirty = m.cfg.Store != nil
		m.status = fmt.Sprintf("Done · %s · %d tokens this request", time.Since(m.started).Round(time.Millisecond), msg.result.Usage.Total)
		m.refresh(keepAtBottom)
		if m.dirty {
			return m, m.save()
		}
		return m, nil
	case saveMsg:
		if msg.generation != m.generation {
			return m, nil
		}
		m.saving = false
		if msg.err != nil {
			m.status = "Reply received, but history was not saved: " + msg.err.Error() + " · Ctrl+S retries"
			return m, nil
		}
		m.session = msg.session
		m.dirty = false
		m.status = "Saved locally · " + shortID(m.session.LastResponseID)
		return m, nil
	case sessionsMsg:
		if !m.choosing || msg.generation != m.pickerGeneration {
			return m, nil
		}
		if msg.err != nil {
			m.status = "Read history: " + msg.err.Error()
			return m, nil
		}
		m.sessions = nil
		for _, s := range msg.sessions {
			if s.API == m.session.API && s.Endpoint == m.session.Endpoint {
				m.sessions = append(m.sessions, s)
			}
		}
		m.selected = 0
		m.status = "Choose a session · Enter opens · Esc returns"
		return m, nil
	case tea.KeyPressMsg:
		key := msg.String()
		if key != "ctrl+q" {
			m.confirmQuit = false
		}
		switch key {
		case "ctrl+q":
			if m.saving {
				m.status = "Waiting for history to finish saving"
				return m, nil
			}
			if m.dirty && !m.confirmQuit {
				m.confirmQuit = true
				m.status = "Unsaved reply. Ctrl+S retries; Ctrl+Q again discards it and quits"
				return m, nil
			}
			if m.busy {
				m.cancelRequest()
			}
			return m, tea.Quit
		case "ctrl+c":
			if m.busy {
				m.cancelRequest()
				return m, nil
			}
			if m.choosing {
				m.choosing = false
				return m, nil
			}
			if m.input.Value() != "" {
				m.input.SetValue("")
				m.status = "Draft cleared"
				return m, nil
			}
			if m.dirty || m.saving {
				m.status = "History has not been saved. Ctrl+S retries; Ctrl+Q quits"
				return m, nil
			}
			return m, tea.Quit
		case "ctrl+s":
			if m.dirty && !m.saving {
				return m, m.save()
			}
			return m, nil
		}
		if m.choosing {
			switch key {
			case "esc", "ctrl+o":
				m.choosing = false
				m.status = "Ready"
			case "up", "k":
				if m.selected > 0 {
					m.selected--
				}
			case "down", "j":
				if m.selected < len(m.sessions)-1 {
					m.selected++
				}
			case "enter":
				if len(m.sessions) > 0 {
					m.session = m.sessions[m.selected].Clone()
					m.choosing = false
					m.pending = ""
					m.partial = ""
					m.input.SetValue("")
					m.status = "Opened local session"
					m.refresh(true)
				}
			}
			return m, nil
		}
		switch key {
		case "enter":
			return m, m.start()
		case "alt+enter", "shift+enter", "ctrl+j":
			m.input.InsertString("\n")
			return m, nil
		case "ctrl+n":
			if m.busy || m.dirty || m.saving {
				m.status = "Finish or cancel the current request and save before starting a new session"
				return m, nil
			}
			m.session = conversation.New(m.session.API, m.session.Endpoint, m.session.Model)
			m.pending = ""
			m.partial = ""
			m.input.SetValue("")
			m.status = "New conversation"
			m.refresh(true)
			return m, nil
		case "ctrl+o":
			if m.cfg.Store == nil {
				m.status = "Temporary mode: local session history is disabled"
				return m, nil
			}
			if m.busy || m.dirty || m.saving {
				m.status = "Finish or cancel the current request and save before opening another session"
				return m, nil
			}
			m.choosing = true
			m.pickerGeneration++
			pickerGeneration := m.pickerGeneration
			m.sessions = nil
			m.status = "Loading local sessions"
			store := *m.cfg.Store
			return m, func() tea.Msg {
				sessions, err := store.List()
				return sessionsMsg{generation: pickerGeneration, sessions: sessions, err: err}
			}
		case "pgup", "pgdown", "ctrl+home", "ctrl+end":
			if key == "ctrl+home" {
				m.viewport.GotoTop()
				return m, nil
			}
			if key == "ctrl+end" {
				m.viewport.GotoBottom()
				return m, nil
			}
			var cmd tea.Cmd
			m.viewport, cmd = m.viewport.Update(msg)
			return m, cmd
		}
	case tea.MouseWheelMsg:
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		return m, cmd
	}
	var cmds []tea.Cmd
	if m.busy {
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		cmds = append(cmds, cmd)
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	cmds = append(cmds, cmd)
	return m, tea.Batch(cmds...)
}
func (m *Model) resize() {
	width := max(1, m.width-4)
	m.input.SetWidth(width)
	m.input.SetHeight(min(4, max(1, m.height/5)))
	m.viewport.SetWidth(width)
	m.viewport.SetHeight(max(1, m.height-m.input.Height()-7))
	m.viewport.SoftWrap = true
}
func safeText(s string) string {
	return strings.Map(func(r rune) rune {
		if r < 32 && r != '\n' && r != '\t' || r == 127 {
			return -1
		}
		return r
	}, ansi.Strip(s))
}
func shortID(id string) string {
	if id == "" {
		return "new session"
	}
	r := []rune(id)
	if len(r) > 14 {
		return string(r[:14]) + "…"
	}
	return id
}
func (m *Model) refresh(bottom bool) {
	var text strings.Builder
	if len(m.session.Messages) == 0 && m.pending == "" {
		text.WriteString("A fresh conversation.\n\nEnter sends · Alt+Enter adds a line\nCtrl+O opens local sessions · Ctrl+N starts a new one\n\nYour API and model stay visible below.\n")
	}
	for _, message := range m.session.Messages {
		label := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#A6DA95"))
		if message.Role == "assistant" {
			label = label.Foreground(lipgloss.Color("#C6A0F6"))
		}
		text.WriteString(label.Render(strings.ToUpper(message.Role)) + "\n" + safeText(message.Content) + "\n\n")
	}
	if m.pending != "" {
		text.WriteString("YOU\n" + safeText(m.pending) + "\n\nASSISTANT\n" + safeText(m.partial) + "\n")
	}
	m.viewport.SetContent(text.String())
	if bottom {
		m.viewport.GotoBottom()
	}
}
func (m Model) View() tea.View {
	width := max(1, m.width)
	if m.width < 30 || m.height < 12 {
		v := tea.NewView(ansi.Truncate("Resize terminal to at least 30 × 12 · Ctrl+Q quits", width, ""))
		v.AltScreen = true
		return v
	}
	dim := lipgloss.NewStyle().Foreground(lipgloss.Color("#8A93A6"))
	header := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#C6A0F6")).Render("openai") + "  " + string(m.session.API) + " · " + singleLine(m.session.Model)
	mode := "local history"
	if m.cfg.Store == nil {
		mode = "temporary"
	}
	meta := fmt.Sprintf("%s · %s · %d tokens", shortID(m.session.ID), mode, m.session.Usage.Total)
	body := m.viewport.View()
	if m.choosing {
		lines := []string{"LOCAL SESSIONS · same API and endpoint", ""}
		if len(m.sessions) == 0 {
			lines = append(lines, "No saved sessions for this API and endpoint")
		}
		height := max(1, m.viewport.Height()-3)
		start := max(0, m.selected-height+1)
		for i := start; i < len(m.sessions) && i < start+height; i++ {
			prefix := "  "
			if i == m.selected {
				prefix = "› "
			}
			lines = append(lines, ansi.Truncate(prefix+singleLine(m.sessions[i].Title())+" · "+shortID(m.sessions[i].ID), m.width-4, "…"))
		}
		body = lipgloss.NewStyle().Height(m.viewport.Height()).Render(strings.Join(lines, "\n"))
	}
	status := singleLine(m.status)
	if m.busy {
		status = m.spinner.View() + " " + status + " · " + time.Since(m.started).Round(time.Second).String()
	}
	parts := []string{ansi.Truncate(header, m.width-4, "…"), dim.Render(ansi.Truncate(meta, m.width-4, "…")), "", body, dim.Render(strings.Repeat("─", m.width-4)), m.input.View(), ansi.Truncate(status, m.width-4, "…"), dim.Render(ansi.Truncate("Enter send · Alt+Enter newline · PgUp scroll · Ctrl+C cancel · Ctrl+Q quit", m.width-4, "…"))}
	content := lipgloss.NewStyle().Padding(0, 2).Render(strings.Join(parts, "\n"))
	// Constrain every frame, including narrow sizes and hostile/long server text.
	lines := strings.Split(content, "\n")
	if len(lines) > m.height {
		lines = lines[:m.height]
	}
	for i, line := range lines {
		lines[i] = ansi.Truncate(line, width, "")
	}
	v := tea.NewView(strings.Join(lines, "\n"))
	v.AltScreen = true
	v.MouseMode = tea.MouseModeCellMotion
	return v
}

func singleLine(s string) string { return strings.Join(strings.Fields(safeText(s)), " ") }
