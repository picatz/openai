package codextui

import (
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
)

func fit(s string, width int) string {
	if width <= 0 {
		return ""
	}
	s = ansi.Truncate(s, width, "…")
	return s + strings.Repeat(" ", max(0, width-ansi.StringWidth(s)))
}

func (m Model) renderDetail() string {
	if len(m.items) == 0 {
		if m.cfg.Runtime {
			return "A clear view of the work.\n\nSubmit a prompt below. Messages, commands, tools, plans, and changed files appear here as events arrive.\n\nApproval and input requests are unsupported in this workspace."
		}
		return "A clear view of the work.\n\nMessages, commands, tools, and changed files appear here as display events arrive.\n\nThis preview has no agent connection. It cannot submit prompts, execute commands, or answer approvals."
	}
	item := m.items[m.selected]
	var parts []string
	parts = append(parts, m.accent(strings.ToUpper(string(item.Kind)))+"  "+m.statusLabel(item.Status), item.Title, "")
	if item.Kind == Files {
		for _, file := range item.Files {
			parts = append(parts, singleLine(file.Kind)+"  "+file.Path)
		}
		if len(item.Files) == maxFiles {
			parts = append(parts, "[showing at most 64 paths]")
		}
		if item.Diff != "" && m.cfg.Capabilities.Diffs {
			parts = append(parts, "", "DIFF · supplied snapshot")
			for _, line := range strings.Split(item.Diff, "\n") {
				switch {
				case strings.HasPrefix(line, "+"):
					line = m.paint(line, "2", false)
				case strings.HasPrefix(line, "-"):
					line = m.paint(line, "1", false)
				case strings.HasPrefix(line, "@@"):
					line = m.accent(line)
				}
				parts = append(parts, line)
			}
		} else {
			parts = append(parts, "", "Diff unavailable in this display source")
		}
	} else if (item.Kind == Command || item.Kind == Tool) && !m.cfg.Capabilities.ToolOutput {
		parts = append(parts, "Output unavailable in this display source")
	} else if item.Kind == Plan && !m.cfg.Capabilities.Plans {
		parts = append(parts, "Plan detail unavailable in this display source")
	} else if item.Text != "" {
		parts = append(parts, item.Text)
	} else {
		parts = append(parts, "No output received")
	}
	return strings.Join(parts, "\n")
}

func (m Model) timeline(width, height int) []string {
	lines := make([]string, 0, height)
	if len(m.items) == 0 {
		return []string{"No items yet"}
	}
	rows := max(1, height/2)
	start := max(0, m.selected-rows+1)
	for i := start; i < len(m.items) && len(lines) < height; i++ {
		item := m.items[i]
		marker := "  "
		if i == m.selected {
			marker = "> "
		}
		title := marker + singleLine(item.Title)
		if i == m.selected {
			title = m.accent(title)
		}
		lines = append(lines, fit(title, width))
		if len(lines) < height {
			lines = append(lines, fit("  "+string(item.Kind)+" "+m.statusLabel(item.Status), width))
		}
	}
	return lines
}

func (m Model) View() tea.View {
	if m.width <= 0 || m.height <= 0 {
		return tea.NewView("")
	}
	if m.width < 30 || m.height < 10 {
		v := tea.NewView(ansi.Truncate("Resize to 30 × 10 · Ctrl+Q quits", m.width, ""))
		v.AltScreen = true
		return v
	}
	w := m.width - 4
	state := "Ready"
	if m.status != "" {
		state = string(m.status)
	}
	if m.active && !m.cfg.ReducedMotion && !m.cfg.NoColor {
		state = string("|/-\\"[m.frame]) + " " + state
	}
	model := singleLine(m.cfg.Model)
	if model == "" {
		model = "model not connected"
	}
	workspace := singleLine(m.cfg.Workspace)
	if workspace == "" {
		workspace = "workspace not connected"
	}
	label := "WORKSPACE PREVIEW"
	if m.cfg.Runtime {
		label = "WORKSPACE"
		model = "Configured model: " + model
	}
	lines := []string{m.accent("openai codex") + "  /  " + label, workspace + "  ·  " + model, strings.Repeat("─", w)}
	contentHeight := m.height - 7
	if m.help {
		help := []string{"KEYBOARD", "", "↑/↓ or j/k  Select an item", "Tab         Switch item / detail focus", "PgUp/PgDn   Scroll detail", "Ctrl+Home   Top of detail", "Ctrl+End    Follow latest detail", "Home/End    First / last item (item focus)", "? or F1     Toggle this help", "Esc         Close help / focus items", "Ctrl+Q      Quit preview", "", "Display only. No prompts or approvals are sent."}
		for i := 0; i < contentHeight; i++ {
			if i < len(help) {
				lines = append(lines, help[i])
			} else {
				lines = append(lines, "")
			}
		}
	} else if m.width >= 100 {
		left := m.width / 3
		right := m.width - left - 7
		tl := m.timeline(left, contentHeight)
		detail := strings.Split(m.detail.View(), "\n")
		for i := 0; i < contentHeight; i++ {
			a, b := "", ""
			if i < len(tl) {
				a = tl[i]
			}
			if i < len(detail) {
				b = detail[i]
			}
			lines = append(lines, fit(a, left)+" │ "+fit(b, right))
		}
	} else {
		detail := strings.Split(m.detail.View(), "\n")
		for i := 0; i < contentHeight; i++ {
			if i < len(detail) {
				lines = append(lines, detail[i])
			} else {
				lines = append(lines, "")
			}
		}
	}
	focus := "items"
	if m.focusDetail {
		focus = "detail"
	}
	selection := "No items"
	if len(m.items) > 0 {
		selection = fmt.Sprintf("Item %d/%d", m.selected+1, len(m.items))
	}
	if m.dropped > 0 {
		selection += fmt.Sprintf(" · %d older items omitted", m.dropped)
	}
	hints := "↑/↓ items · Tab focus · PgUp/PgDn scroll · ? help · Ctrl+Q quit"
	if w < 70 {
		hints = "↑/↓ · ? help · Ctrl+Q quit"
	}
	source := "DISPLAY ONLY · no agent connection"
	if m.cfg.Runtime {
		source = "APP-SERVER · current turn · approvals/input unsupported"
		hints = "Tab focus · PgUp/PgDn scroll · F6 composer · F1 help"
	}
	lines = append(lines, strings.Repeat("─", w), selection+" · Focus: "+focus+" · "+state, source, hints)
	// No untrusted terminal sequences survive normalization. Bound every frame
	// by display-cell width, including wide Unicode and a tiny terminal.
	for i, line := range lines {
		lines[i] = "  " + fit(line, w) + "  "
	}
	if len(lines) > m.height {
		lines = lines[:m.height]
	}
	v := tea.NewView(strings.Join(lines, "\n"))
	v.AltScreen = true
	return v
}
