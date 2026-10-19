package codex

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestCollabToolCallEvent(t *testing.T) {
	for _, eventType := range []EventType{EventTypeItemStarted, EventTypeItemUpdated, EventTypeItemCompleted} {
		t.Run(string(eventType), func(t *testing.T) {
			raw := []byte(fmt.Sprintf(`{"type":%q,"future_event":true,"item":{"id":"call-1","type":"collab_tool_call","tool":"wait","sender_thread_id":"parent","receiver_thread_ids":["agent-1","agent-2"],"prompt":"private prompt","agents_states":{"agent-1":{"status":"completed","message":"private result","future_state":9007199254740993},"agent-2":{"status":"errored","message":"private error"}},"status":"completed","future_item":{"value":9007199254740993}}}`, eventType))
			var event ThreadEvent
			if err := json.Unmarshal(raw, &event); err != nil {
				t.Fatal(err)
			}
			item, ok := event.Item.(*CollabToolCallItem)
			if !ok {
				t.Fatalf("item = %T, want *CollabToolCallItem", event.Item)
			}
			if event.Type != eventType || item.ItemType() != ItemTypeCollabToolCall || item.Type != ItemTypeCollabToolCall || item.ID != "call-1" || item.Tool != CollabToolWait || item.SenderThreadID != "parent" || item.Status != CollabToolCallStatusCompleted {
				t.Fatalf("event/item = %+v/%+v", event, item)
			}
			if !reflect.DeepEqual(item.ReceiverThreadIDs, []string{"agent-1", "agent-2"}) || item.Prompt == nil || *item.Prompt != "private prompt" || len(item.AgentsStates) != 2 {
				t.Fatalf("collab payload = %+v", item)
			}
			for id, want := range map[string]struct {
				status  CollabAgentStatus
				message string
			}{"agent-1": {CollabAgentStatusCompleted, "private result"}, "agent-2": {CollabAgentStatusErrored, "private error"}} {
				state := item.AgentsStates[id]
				if state.Status != want.status || state.Message == nil || *state.Message != want.message {
					t.Fatalf("agent %s = %+v", id, state)
				}
			}
			wantSummary := string(eventType) + ` item=collab_tool_call tool="wait" receivers=2 status=completed`
			if got := event.String(); got != wantSummary {
				t.Fatalf("summary = %q, want %q", got, wantSummary)
			}
			// Raw includes future fields at every level, with exact numeric bytes.
			before := string(raw)
			raw[0] = 'x'
			if string(event.Raw) != before {
				t.Fatal("complete original event was not retained in owned storage")
			}
			encoded, err := json.Marshal(item)
			if err != nil {
				t.Fatal(err)
			}
			roundTrip, err := UnmarshalThreadItem(encoded)
			if err != nil || !reflect.DeepEqual(roundTrip, item) {
				t.Fatalf("typed round trip = %+v, %v", roundTrip, err)
			}
		})
	}
}

func TestCollabToolCallOptionalFields(t *testing.T) {
	for _, tc := range []struct {
		name, fields string
		present      bool
	}{
		{"omitted", `"agents_states":{"agent":{"status":"running"}}`, false},
		{"null", `"prompt":null,"agents_states":{"agent":{"status":"running","message":null}}`, false},
		{"empty", `"prompt":"","agents_states":{"agent":{"status":"running","message":""}}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			item, err := UnmarshalThreadItem([]byte(`{"type":"collab_tool_call",` + tc.fields + `}`))
			if err != nil {
				t.Fatal(err)
			}
			call := item.(*CollabToolCallItem)
			message := call.AgentsStates["agent"].Message
			if (call.Prompt != nil) != tc.present || (message != nil) != tc.present {
				t.Fatalf("prompt/message = %v/%v, present = %v", call.Prompt, message, tc.present)
			}
			if tc.present && (*call.Prompt != "" || *message != "") {
				t.Fatal("empty strings changed")
			}
		})
	}
	for _, fields := range []string{``, `,"receiver_thread_ids":null,"agents_states":null`, `,"receiver_thread_ids":[],"agents_states":{}`} {
		item, err := UnmarshalThreadItem([]byte(`{"type":"collab_tool_call"` + fields + `}`))
		if err != nil {
			t.Fatal(err)
		}
		call := item.(*CollabToolCallItem)
		if len(call.ReceiverThreadIDs) != 0 || len(call.AgentsStates) != 0 {
			t.Fatalf("empty collections = %+v", call)
		}
	}
}

func TestCollabToolCallFutureValues(t *testing.T) {
	raw := []byte(`{"type":"item.completed","item":{"type":"collab_tool_call","tool":"future_tool","status":"future_call_status","agents_states":{"agent":{"status":"future_agent_status","message":null}}}}`)
	var event ThreadEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		t.Fatal(err)
	}
	item, ok := event.Item.(*CollabToolCallItem)
	if !ok || item.Tool != "future_tool" || item.Status != "future_call_status" || item.AgentsStates["agent"].Status != "future_agent_status" {
		t.Fatalf("future values = %+v", event.Item)
	}
	if !bytes.Equal(event.Raw, raw) || !strings.Contains(event.String(), "status=future_call_status") {
		t.Fatal("future values not retained in raw event/summary")
	}
}

func TestCollabToolCallMalformedFields(t *testing.T) {
	for _, field := range []string{
		`"id":4`, `"tool":false`, `"sender_thread_id":[]`,
		`"receiver_thread_ids":"agent"`, `"receiver_thread_ids":[4]`,
		`"prompt":{}`, `"status":4`, `"agents_states":[]`,
		`"agents_states":{"agent":false}`,
		`"agents_states":{"agent":{"status":4}}`,
		`"agents_states":{"agent":{"message":[]}}`,
	} {
		t.Run(field, func(t *testing.T) {
			raw := []byte(`{"type":"collab_tool_call",` + field + `}`)
			if _, err := UnmarshalThreadItem(raw); err == nil {
				t.Fatal("accepted malformed known item field")
			}
			// An invalid new item must not partially replace a reused event.
			original := ThreadEvent{Type: EventTypeTurnStarted, Raw: json.RawMessage(`{"type":"turn.started"}`)}
			event := original
			if err := json.Unmarshal(append(append([]byte(`{"type":"item.completed","item":`), raw...), '}'), &event); err == nil {
				t.Fatal("accepted malformed known item in event")
			}
			if !reflect.DeepEqual(event, original) {
				t.Fatal("failed decode mutated the event")
			}
		})
	}
}

// Synthetic exec JSONL follows rust-v0.163.0's schema; no tools are executed.
func TestRustV01630CollabFixture(t *testing.T) {
	file, err := os.Open("testdata/rust-v0.163.0-collab.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		event  EventType
		tool   CollabTool
		status CollabToolCallStatus
	}{
		{EventTypeItemStarted, CollabToolSpawnAgent, CollabToolCallStatusInProgress},
		{EventTypeItemCompleted, CollabToolSpawnAgent, CollabToolCallStatusCompleted},
		{EventTypeItemStarted, CollabToolSendInput, CollabToolCallStatusInProgress},
		{EventTypeItemCompleted, CollabToolSendInput, CollabToolCallStatusCompleted},
		{EventTypeItemStarted, CollabToolWait, CollabToolCallStatusInProgress},
		{EventTypeItemCompleted, CollabToolWait, CollabToolCallStatusCompleted},
		{EventTypeItemCompleted, CollabToolCloseAgent, CollabToolCallStatusCompleted},
		{EventTypeItemCompleted, CollabToolWait, CollabToolCallStatusFailed},
	}
	states := make(map[CollabAgentStatus]bool)
	count := 0
	for event, err := range EventStream(t.Context(), &ExecStream{stdout: file}) {
		if err != nil {
			t.Fatal(err)
		}
		item, ok := event.Item.(*CollabToolCallItem)
		if !ok || count >= len(want) {
			t.Fatalf("unexpected event %d: %+v", count, event)
		}
		w := want[count]
		if event.Type != w.event || item.Tool != w.tool || item.Status != w.status {
			t.Fatalf("event %d = %s, want %+v", count, event, w)
		}
		for _, state := range item.AgentsStates {
			states[state.Status] = true
		}
		count++
	}
	if count != len(want) {
		t.Fatalf("events = %d, want %d", count, len(want))
	}
	for _, status := range []CollabAgentStatus{CollabAgentStatusPendingInit, CollabAgentStatusRunning, CollabAgentStatusInterrupted, CollabAgentStatusCompleted, CollabAgentStatusErrored, CollabAgentStatusShutdown, CollabAgentStatusNotFound} {
		if !states[status] {
			t.Errorf("fixture did not cover %q", status)
		}
	}
}

func FuzzCollabToolCallEvent(f *testing.F) {
	for _, seed := range []string{
		`{"type":"collab_tool_call","tool":"spawn_agent","status":"in_progress","prompt":null,"agents_states":{}}`,
		`{"type":"collab_tool_call","tool":"future_tool","status":"future_status","receiver_thread_ids":["a","b"],"agents_states":{"a":{"status":"future_status","message":""}}}`,
		`{"type":"collab_tool_call","agents_states":{"a":{"status":4}}}`,
		`{"type":"future_item","value":9007199254740993}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, itemData []byte) {
		raw := append(append([]byte(`{"type":"item.completed","item":`), itemData...), '}')
		var event ThreadEvent
		if err := json.Unmarshal(raw, &event); err != nil {
			return
		}
		if !bytes.Equal(event.Raw, raw) {
			t.Fatal("event lost raw bytes")
		}
		_ = event.String()
		if _, err := json.Marshal(event); err != nil {
			t.Fatalf("decoded event cannot be marshaled: %v", err)
		}
		before := string(event.Raw)
		raw[0] = 'x'
		if string(event.Raw) != before {
			t.Fatal("raw event aliases input")
		}
	})
}
