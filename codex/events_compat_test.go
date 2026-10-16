package codex

import (
	"bytes"
	"encoding/json"
	"os"
	"testing"
)

func TestCurrentEventPayloads(t *testing.T) {
	raw := []byte(`{"type":"item.completed","item":{"id":"m1","type":"mcp_tool_call","server":"fixture","tool":"synthetic","arguments":{"id":9007199254740993},"result":{"content":[{"type":"text","text":"ok"}],"structured_content":{"answer":42},"_meta":{"tag":"kept"}},"status":"completed"},"future_field":true}`)
	var event ThreadEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		t.Fatal(err)
	}
	item := event.Item.(*McpToolCallItem)
	if string(item.Arguments) != `{"id":9007199254740993}` || len(item.Result.Content) != 1 || string(item.Result.Meta) != `{"tag":"kept"}` {
		t.Fatalf("MCP = %+v", item)
	}
	if !bytes.Equal(event.Raw, raw) {
		t.Fatal("complete event not retained")
	}
	if err := json.Unmarshal([]byte(`{"type":"turn.completed","usage":{"input_tokens":1,"cached_input_tokens":2,"output_tokens":3,"cache_write_input_tokens":4,"reasoning_output_tokens":5}}`), &event); err != nil {
		t.Fatal(err)
	}
	if event.Usage.CacheWriteInputTokens != 4 || event.Usage.ReasoningOutputTokens != 5 || event.Item != nil {
		t.Fatalf("usage/event = %+v", event)
	}
}

func TestUnknownEventAndItemPreserveOwnedBytes(t *testing.T) {
	raw := []byte(`{"type":"future.event","item":false,"usage":[1],"extra":9007199254740993}`)
	var event ThreadEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		t.Fatal(err)
	}
	before := string(raw)
	raw[0] = 'x'
	if string(event.Raw) != before || event.Type != "future.event" || event.Item != nil {
		t.Fatalf("unknown event = %+v", event)
	}
	raw = []byte(`{"type":"collab_tool_call","tool":"spawn_agent","receiver_thread_ids":["agent-1"],"agents_states":{},"status":"completed"}`)
	item, err := UnmarshalThreadItem(raw)
	if err != nil {
		t.Fatal(err)
	}
	before = string(raw)
	raw[0] = 'x'
	unknown := item.(*UnknownThreadItem)
	if string(unknown.Raw) != before {
		t.Fatal("unknown item aliases caller buffer")
	}
	got, err := json.Marshal(unknown)
	if err != nil || string(got) != before {
		t.Fatalf("unknown marshal = %s, %v", got, err)
	}
}

func TestMalformedKnownEventsFail(t *testing.T) {
	for _, raw := range []string{`null`, `{}`, `{"type":4}`, `{"type":"item.completed"}`, `{"type":"item.started","item":null}`, `{"type":"thread.started"}`, `{"type":"turn.completed"}`, `{"type":"turn.failed"}`, `{"type":"item.completed","item":{"type":"agent_message","text":false}}`} {
		var event ThreadEvent
		if err := json.Unmarshal([]byte(raw), &event); err == nil {
			t.Fatalf("accepted malformed event: %s", raw)
		}
	}
}

func TestCurrentWebSearchAndStatusPayloads(t *testing.T) {
	item, err := UnmarshalThreadItem([]byte(`{"type":"web_search","id":"w1","query":"fixture","action":{"type":"search","queries":["fixture"]},"results":[{"url":"https://example.invalid"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	search := item.(*WebSearchItem)
	if len(search.Action) == 0 || len(search.Results) != 1 {
		t.Fatalf("search = %+v", search)
	}
	item, err = UnmarshalThreadItem([]byte(`{"type":"command_execution","status":"declined","exit_code":null}`))
	if err != nil || item.(*CommandExecutionItem).Status != CommandExecutionStatusDeclined {
		t.Fatalf("command = %+v, %v", item, err)
	}
}

// Synthetic fixture follows the tagged Rust schema; no listed tools are executed.
func TestRustV01601Fixture(t *testing.T) {
	file, err := os.Open("testdata/rust-v0.160.1.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	stream := &ExecStream{stdout: file}
	count, unknown := 0, 0
	for event, err := range EventStream(t.Context(), stream) {
		if err != nil {
			t.Fatal(err)
		}
		count++
		if _, ok := event.Item.(*UnknownThreadItem); ok {
			unknown++
		}
		if event.Type == EventTypeTurnCompleted && (event.Usage.CacheWriteInputTokens != 2 || event.Usage.ReasoningOutputTokens != 1) {
			t.Fatalf("usage = %+v", event.Usage)
		}
	}
	if count != 9 || unknown != 1 {
		t.Fatalf("events/unknown items = %d/%d", count, unknown)
	}
}
