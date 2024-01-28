package main

// Mode of operation.
type Mode string

// Modes of operation for the CLI.
const (
	// Deprecated: this was shut down by OpenAI.
	ModeEdit Mode = "edit"

	// Deprecated: this was shut down by OpenAI.
	// https://platform.openai.com/docs/deprecations/2023-07-06-gpt-and-embeddings
	ModeComplete Mode = "complete"

	ModeChat      Mode = "chat"
	ModeAssistant Mode = "assistant"
)
