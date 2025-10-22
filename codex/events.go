package codex

import (
	"encoding/json"
	"fmt"
)

// EventType enumerates the JSON events emitted by `codex exec`.
type EventType string

const (
	EventTypeThreadStarted EventType = "thread.started"
	EventTypeTurnStarted   EventType = "turn.started"
	EventTypeTurnCompleted EventType = "turn.completed"
	EventTypeTurnFailed    EventType = "turn.failed"
	EventTypeItemStarted   EventType = "item.started"
	EventTypeItemUpdated   EventType = "item.updated"
	EventTypeItemCompleted EventType = "item.completed"
	EventTypeError         EventType = "error"
)

// Usage reports token usage for a turn.
type Usage struct {
	InputTokens       int `json:"input_tokens"`
	CachedInputTokens int `json:"cached_input_tokens"`
	OutputTokens      int `json:"output_tokens"`
}

// ThreadError describes a fatal error emitted by a turn.
type ThreadError struct {
	Message string `json:"message"`
}

// ThreadEvent represents a single line event emitted by codex exec.
type ThreadEvent struct {
	// Type identifies the event kind.
	Type EventType `json:"type"`
	// ThreadID is populated on thread.started events with the server-issued identifier.
	ThreadID string `json:"thread_id,omitempty"`
	// Usage is populated on turn.completed events with token usage.
	Usage *Usage `json:"usage,omitempty"`
	// Error is populated on turn.failed events with the failure message.
	Error *ThreadError `json:"error,omitempty"`
	// Item contains the thread item payload for item.* events. It is nil for other event types.
	Item ThreadItem `json:"item,omitempty"`
	// Message is populated on top-level error events.
	Message string `json:"message,omitempty"`
}

// String renders a human-readable description of the event for debugging and tests.
func (e ThreadEvent) String() string {
	switch e.Type {
	case EventTypeThreadStarted:
		if e.ThreadID != "" {
			return fmt.Sprintf("thread.started id=%s", e.ThreadID)
		}
		return "thread.started"
	case EventTypeTurnStarted:
		return "turn.started"
	case EventTypeTurnCompleted:
		if e.Usage != nil {
			return fmt.Sprintf("turn.completed usage=%+v", *e.Usage)
		}
		return "turn.completed"
	case EventTypeTurnFailed:
		if e.Error != nil {
			return fmt.Sprintf("turn.failed error=%s", e.Error.Message)
		}
		return "turn.failed"
	case EventTypeItemStarted, EventTypeItemUpdated, EventTypeItemCompleted:
		if e.Item != nil {
			return fmt.Sprintf("%s item=%s", e.Type, itemSummary(e.Item))
		}
		return string(e.Type)
	case EventTypeError:
		if e.Message != "" {
			return fmt.Sprintf("error message=%s", e.Message)
		}
		return "error"
	default:
		return string(e.Type)
	}
}

func itemSummary(item ThreadItem) string {
	switch v := item.(type) {
	case *AgentMessageItem:
		return fmt.Sprintf("agent_message text=%q", v.Text)
	case *ReasoningItem:
		return fmt.Sprintf("reasoning text=%q", v.Text)
	case *CommandExecutionItem:
		return fmt.Sprintf("command_execution command=%q status=%s", v.Command, v.Status)
	case *FileChangeItem:
		return fmt.Sprintf("file_change changes=%d status=%s", len(v.Changes), v.Status)
	case *McpToolCallItem:
		return fmt.Sprintf("mcp_tool_call server=%q tool=%q status=%s", v.Server, v.Tool, v.Status)
	case *WebSearchItem:
		return fmt.Sprintf("web_search query=%q", v.Query)
	case *TodoListItem:
		return fmt.Sprintf("todo_list items=%d", len(v.Items))
	case *ErrorItem:
		return fmt.Sprintf("error message=%q", v.Message)
	case *UnknownThreadItem:
		return fmt.Sprintf("unknown type=%s", v.Type)
	default:
		return fmt.Sprintf("%T", item)
	}
}

// UnmarshalJSON customizes decoding to handle the polymorphic item payload.
func (e *ThreadEvent) UnmarshalJSON(data []byte) error {
	var aux struct {
		Type     EventType       `json:"type"`
		ThreadID string          `json:"thread_id,omitempty"`
		Usage    *Usage          `json:"usage,omitempty"`
		Error    *ThreadError    `json:"error,omitempty"`
		Item     json.RawMessage `json:"item,omitempty"`
		Message  string          `json:"message,omitempty"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	e.Type = aux.Type
	e.ThreadID = aux.ThreadID
	e.Usage = aux.Usage
	e.Error = aux.Error
	e.Message = aux.Message

	if len(aux.Item) > 0 {
		item, err := UnmarshalThreadItem(aux.Item)
		if err != nil {
			return fmt.Errorf("decode thread item: %w", err)
		}
		e.Item = item
	} else {
		e.Item = nil
	}

	return nil
}
