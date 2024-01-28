package main

import "github.com/charmbracelet/lipgloss"

var (
	styleBold    = lipgloss.NewStyle().Bold(true)
	styleFaint   = lipgloss.NewStyle().Faint(true)
	styleWarning = lipgloss.NewStyle().Foreground(lipgloss.Color("9"))
	numberColor  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("69"))
)
