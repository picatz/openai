package main

import (
	"context"
	"encoding/json"
	"errors"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/picatz/openai/internal/conversation"
	"io"
	"net/http"
	"strings"
	"testing"
)

func backendTest(t *testing.T, body string, check func(*http.Request)) sdkBackend {
	client := openai.NewClient(option.WithAPIKey("synthetic-key"), option.WithMaxRetries(0), option.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if check != nil {
			check(r)
		}
		return fakeResponse(r, 200, "text/event-stream", body), nil
	})}))
	return sdkBackend{client: &client, preserveReasoning: true}
}
func TestChatStreamContracts(t *testing.T) {
	delta := `data: {"id":"chat_test","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}` + "\n\n"
	stop := `data: {"id":"chat_test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
	usage := `data: {"id":"chat_test","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}` + "\n\n"
	for _, tc := range []struct {
		name, body string
		fail       bool
	}{
		{"normal", delta + stop + usage + "data: [DONE]\n\n", false},
		{"terminal with usage", delta + `data: {"id":"chat_test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"total_tokens":5}}` + "\n\n", false},
		{"truncated", delta, true},
		{"HTTP200 error", `data: {"error":{"message":"synthetic"},"choices":[]}` + "\n\n", true},
		{"length", delta + `data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}` + "\n\n", true},
		{"tool", `data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"tool","function":{"name":"fn"}}]}}]}` + "\n\n", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var text strings.Builder
			result, err := backendTest(t, tc.body, func(r *http.Request) {
				if r.URL.Path != "/v1/chat/completions" {
					t.Error(r.URL)
				}
			}).Generate(t.Context(), conversation.Request{API: conversation.Chat, Model: "test", Messages: []conversation.Message{{Role: "user", Content: "prompt"}}}, func(s string) error { text.WriteString(s); return nil })
			if (err != nil) != tc.fail {
				t.Fatalf("result=%+v err=%v", result, err)
			}
			if !tc.fail && (result.Text != "hello" || result.Usage.Total != 5) {
				t.Fatalf("result=%+v", result)
			}
		})
	}
}
func TestResponsesNativeReplayPreservesItems(t *testing.T) {
	history := []json.RawMessage{json.RawMessage(`{"type":"reasoning","id":"rs_test","encrypted_content":"synthetic","future_field":true}`)}
	fixture := strings.Replace(responseFixture, `"output":[`, `"output":[{"type":"reasoning","id":"rs_next","encrypted_content":"synthetic-next","future":123},`, 1)
	body := "data: " + `{"type":"response.completed","response":` + fixture + "}\n\n"
	backend := backendTest(t, body, func(r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var p map[string]json.RawMessage
		if err := json.Unmarshal(data, &p); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(p["input"]), `"future_field":true`) || !strings.Contains(string(p["include"]), "reasoning.encrypted_content") || string(p["store"]) != "false" {
			t.Errorf("body=%s", data)
		}
		if _, ok := p["previous_response_id"]; ok {
			t.Error("unexpected server-side state")
		}
	})
	result, err := backend.Generate(t.Context(), conversation.Request{API: conversation.Responses, Model: "test", Messages: []conversation.Message{{Role: "user", Content: "next"}}, ResponsesItems: history}, func(string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ResponsesItems) != 4 || !strings.Contains(string(result.ResponsesItems[2]), `"future":123`) {
		t.Fatalf("replay lost native items: %s", result.ResponsesItems)
	}
}
func TestBackendEmitFailure(t *testing.T) {
	backend := backendTest(t, "data: "+`{"type":"response.output_text.delta","delta":"text"}`+"\n\n", nil)
	want := errors.New("output closed")
	_, err := backend.Generate(context.Background(), conversation.Request{API: conversation.Responses, Model: "test"}, func(string) error { return want })
	if !errors.Is(err, want) {
		t.Fatalf("err=%v", err)
	}
}

func TestResponsesToolCallIsNotOrdinaryReply(t *testing.T) {
	fixture := strings.Replace(responseFixture, `"output":[`, `"output":[{"type":"function_call","id":"tool_test","call_id":"call_test","name":"unconfigured_tool","arguments":"{}"},`, 1)
	backend := backendTest(t, "data: "+`{"type":"response.completed","response":`+fixture+"}\n\n", nil)
	result, err := backend.Generate(t.Context(), conversation.Request{API: conversation.Responses, Model: "test", Messages: []conversation.Message{{Role: "user", Content: "hello"}}}, func(string) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "tool-aware") || len(result.ResponsesItems) != 0 {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}
