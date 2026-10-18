package codextui

import "charm.land/lipgloss/v2"

// Terminal-owned colors preserve the user's palette. Labels and selection
// markers carry the same information when styling is disabled.
func (m Model) accent(s string) string { return m.paint(s, "5", true) }
func (m Model) paint(s, color string, bold bool) string {
	if m.cfg.NoColor {
		return s
	}
	style := lipgloss.NewStyle().Foreground(lipgloss.Color(color)).Bold(bold)
	return style.Render(s)
}
func (m Model) statusLabel(status Status) string {
	label := "[" + singleLine(string(status)) + "]"
	switch status {
	case Completed:
		return m.paint(label, "2", false)
	case Failed, Declined:
		return m.paint(label, "1", true)
	case Running:
		return m.accent(label)
	default:
		return label
	}
}
