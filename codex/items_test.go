package codex

import (
	"encoding/json"
	"testing"
)

func TestUnmarshalThreadItemAgentMessage(t *testing.T) {
	raw := []byte(`{"id":"1","type":"agent_message","text":"hello"}`)
	item, err := UnmarshalThreadItem(raw)
	if err != nil {
		t.Fatalf("UnmarshalThreadItem returned error: %v", err)
	}

	message, ok := item.(*AgentMessageItem)
	if !ok {
		t.Fatalf("expected AgentMessageItem, got %T", item)
	}
	if message.Text != "hello" {
		t.Fatalf("expected text to be %q, got %q", "hello", message.Text)
	}
}

func TestUnmarshalThreadItemUnknown(t *testing.T) {
	raw := []byte(`{"type":"new_item","value":42}`)
	item, err := UnmarshalThreadItem(raw)
	if err != nil {
		t.Fatalf("UnmarshalThreadItem returned error: %v", err)
	}
	unknown, ok := item.(*UnknownThreadItem)
	if !ok {
		t.Fatalf("expected UnknownThreadItem, got %T", item)
	}
	if unknown.Type != "new_item" {
		t.Fatalf("unexpected item type %q", unknown.Type)
	}
	if len(unknown.Raw) == 0 {
		t.Fatal("expected raw payload to be retained")
	}
}

func TestThreadEventUnmarshal(t *testing.T) {
	raw := []byte(`{"type":"item.completed","item":{"id":"2","type":"agent_message","text":"done"}}`)
	var event ThreadEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		t.Fatalf("json.Unmarshal returned error: %v", err)
	}
	if event.Type != EventTypeItemCompleted {
		t.Fatalf("expected event type %q, got %q", EventTypeItemCompleted, event.Type)
	}
	if event.Item == nil {
		t.Fatal("expected event to contain an item")
	}
	if _, ok := event.Item.(*AgentMessageItem); !ok {
		t.Fatalf("expected AgentMessageItem, got %T", event.Item)
	}
}
