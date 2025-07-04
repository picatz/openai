package main

import "github.com/charmbracelet/lipgloss"

var (
	styleBold    = lipgloss.NewStyle().Bold(true)
	styleFaint   = lipgloss.NewStyle().Faint(true)
	styleWarning = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
	numberColor  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("69"))
	styleInfo    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#57C7FF")) // Blue
	styleSuccess = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#5AFFA"))  // Green
	styleError   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#FF5A5F")) // Red
	stylePath    = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFD75F"))
)
