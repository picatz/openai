package main

import "github.com/charmbracelet/lipgloss"

var (
	styleBold    = lipgloss.NewStyle().Bold(true)
	styleFaint   = lipgloss.NewStyle().Faint(true)
	styleWarning = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
	styleInfo    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#57C7FF")) // Blue
	stylePath    = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFD75F"))
	styleAI      = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#6BCB77"))
	styleKBD     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#FF9CAC"))
)
