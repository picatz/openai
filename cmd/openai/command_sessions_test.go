package main

import (
	"encoding/json"
	"github.com/picatz/openai/internal/conversation"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
)

func TestSessionPromptAndEndpointAffinity(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sessions")
	calls := 0
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		return fakeResponse(r, 200, "text/event-stream", "data: "+`{"type":"response.completed","response":`+responseFixture+"}\n\n"), nil
	})
	out, errOut, err := executeTest(t, []string{"responses", "create", "hello", "--session", "new", "--history-dir", dir}, "", transport)
	if err != nil || out != "hello world\n" || !strings.HasPrefix(errOut, "Session: ") {
		t.Fatalf("out=%q stderr=%q err=%v", out, errOut, err)
	}
	id := strings.TrimSpace(strings.TrimPrefix(errOut, "Session: "))
	saved, err := (conversation.Store{Dir: dir}).Load(id)
	if err != nil || len(saved.ResponsesItems) != 2 {
		t.Fatalf("saved=%+v err=%v", saved, err)
	}
	out, _, err = executeTest(t, []string{"responses", "create", "next", "--session", id, "--history-dir", dir, "--output", "json"}, "", transport)
	if err != nil || !json.Valid([]byte(out)) || !strings.Contains(out, `"saved":true`) {
		t.Fatalf("out=%q err=%v", out, err)
	}
	beforeCalls := calls
	_, _, err = executeTest(t, []string{"responses", "create", "next", "--session", id, "--history-dir", dir, "--base-url", "https://elsewhere.invalid/v1/"}, "", transport)
	if err == nil || calls != beforeCalls {
		t.Fatal("replayed private context to another endpoint")
	}
}
func TestFailedSessionRequestPreservesDisk(t *testing.T) {
	store := conversation.Store{Dir: t.TempDir()}
	session := conversation.New(conversation.Responses, "https://api.openai.com/v1/", "gpt-4o")
	session.Messages = []conversation.Message{{Role: "user", Content: "previous"}, {Role: "assistant", Content: "saved"}}
	if err := store.Save(&session); err != nil {
		t.Fatal(err)
	}
	_, _, err := executeTest(t, []string{"responses", "create", "new", "--session", session.ID, "--history-dir", store.Dir}, "", func(r *http.Request) (*http.Response, error) {
		return fakeResponse(r, 200, "text/event-stream", `data: {"type":"response.output_text.delta","delta":"partial"}`+"\n\n"), nil
	})
	if err == nil {
		t.Fatal("truncated response succeeded")
	}
	after, err := store.Load(session.ID)
	if err != nil || after.Revision != 1 || len(after.Messages) != 2 {
		t.Fatalf("saved history changed: %+v %v", after, err)
	}
}
func TestChatOneShotUsesChatEndpoint(t *testing.T) {
	for _, format := range []string{"text", "json"} {
		out, _, err := executeTest(t, []string{"chat", "hello", "--output", format}, "", func(r *http.Request) (*http.Response, error) {
			if r.URL.Path != "/v1/chat/completions" {
				t.Error(r.URL)
			}
			return fakeResponse(r, 200, "application/json", `{"id":"chat_test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"unknown_field":123}`), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if format == "text" && out != "hello\n" || format == "json" && (!json.Valid([]byte(out)) || !strings.Contains(out, `"unknown_field":123`)) {
			t.Fatalf("out=%q", out)
		}
	}
}
func TestSessionsCommandsNeedNoAPI(t *testing.T) {
	store := conversation.Store{Dir: t.TempDir()}
	s := conversation.New(conversation.Chat, "local", "test")
	s.Messages = []conversation.Message{{Role: "user", Content: "hello"}}
	if err := store.Save(&s); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"sessions", "list"}, {"sessions", "show", s.ID}} {
		args = append(args, "--history-dir", store.Dir, "--output", "json")
		out, _, err := executeTest(t, args, "", func(r *http.Request) (*http.Response, error) { t.Fatal("unexpected API call"); return nil, nil })
		if err != nil || !json.Valid([]byte(out)) {
			t.Fatalf("out=%q err=%v", out, err)
		}
	}
}
