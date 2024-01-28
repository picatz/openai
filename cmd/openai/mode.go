package main

// Mode of operation.
type Mode string

// Modes of operation.
const (
	ModeEdit      Mode = "edit"
	ModeComplete  Mode = "complete"
	ModeChat      Mode = "chat"
	ModeAssistant Mode = "assistant"
)
