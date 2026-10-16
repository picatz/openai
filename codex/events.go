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
	InputTokens           int `json:"input_tokens"`
	CachedInputTokens     int `json:"cached_input_tokens"`
	OutputTokens          int `json:"output_tokens"`
	CacheWriteInputTokens int `json:"cache_write_input_tokens"`
	ReasoningOutputTokens int `json:"reasoning_output_tokens"`
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
	// Raw is an owned copy of the complete original event, including unknown fields
	// and event kinds. It is excluded from marshaling; decode it for future extensions.
	Raw json.RawMessage `json:"-"`
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
	var discriminator struct {
		Type EventType `json:"type"`
	}
	if err := json.Unmarshal(data, &discriminator); err != nil {
		return err
	}
	if discriminator.Type == "" {
		return fmt.Errorf("thread event missing type discriminator")
	}
	switch discriminator.Type {
	case EventTypeThreadStarted, EventTypeTurnStarted, EventTypeTurnCompleted,
		EventTypeTurnFailed, EventTypeItemStarted, EventTypeItemUpdated,
		EventTypeItemCompleted, EventTypeError:
	default:
		// A future event need not follow today's item or usage schemas.
		*e = ThreadEvent{Type: discriminator.Type, Raw: append(json.RawMessage(nil), data...)}
		return nil
	}
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

	if aux.Type == "" {
		return fmt.Errorf("thread event missing type discriminator")
	}
	decoded := ThreadEvent{Type: aux.Type, ThreadID: aux.ThreadID, Usage: aux.Usage,
		Error: aux.Error, Message: aux.Message, Raw: append(json.RawMessage(nil), data...)}
	switch aux.Type {
	case EventTypeItemStarted, EventTypeItemUpdated, EventTypeItemCompleted:
		if len(aux.Item) == 0 || string(aux.Item) == "null" {
			return fmt.Errorf("%s missing item", aux.Type)
		}
	case EventTypeThreadStarted:
		if aux.ThreadID == "" {
			return fmt.Errorf("thread.started missing thread_id")
		}
	case EventTypeTurnCompleted:
		if aux.Usage == nil {
			return fmt.Errorf("turn.completed missing usage")
		}
	case EventTypeTurnFailed:
		if aux.Error == nil {
			return fmt.Errorf("turn.failed missing error")
		}
	}
	switch aux.Type {
	case EventTypeItemStarted, EventTypeItemUpdated, EventTypeItemCompleted:
		item, err := UnmarshalThreadItem(aux.Item)
		if err != nil {
			return fmt.Errorf("decode thread item: %w", err)
		}
		decoded.Item = item
	}
	*e = decoded

	return nil
}
