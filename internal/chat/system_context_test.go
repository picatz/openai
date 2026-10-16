package chat

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/picatz/openai/internal/chat/storage/memory"
)

type contractTransport func(*http.Request) (*http.Response, error)

func (f contractTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func syntheticCompletion(r *http.Request, text string) *http.Response {
	body, _ := json.Marshal(map[string]any{"id": "synthetic", "choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": text}, "finish_reason": "stop"}}, "usage": map[string]any{"total_tokens": 5}})
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(bytes.NewReader(body)), Request: r}
}
func TestSystemUpdateReplacesFirstInstruction(t *testing.T) {
	backend := memory.NewBackend[string, ReqRespPair]()
	pair := ReqRespPair{Req: openai.ChatCompletionMessage{Role: "user", Content: "saved question"}, Resp: openai.ChatCompletionMessage{Role: "assistant", Content: "saved answer"}}
	backend.Set(t.Context(), "old", pair)
	s := Session{Commands: builtinCommands, StorageBackend: backend, Messages: []openai.ChatCompletionMessage{{Role: "system", Content: "system: Talk like a pirate."}, {Role: "user", Content: "Hello!"}, {Role: "assistant", Content: "Ahoy!"}}, OutWriter: bufio.NewWriter(io.Discard)}
	s.processInput(t.Context(), ptr("system: Talk like a dog."))
	if len(s.Messages) != 3 || s.Messages[0].Role != "system" || s.Messages[0].Content != "Talk like a dog." {
		t.Fatalf("messages=%+v", s.Messages)
	}
	if s.Messages[1].Content != "Hello!" || s.Messages[2].Content != "Ahoy!" {
		t.Fatal("conversation data was removed")
	}
	saved, ok, err := backend.Get(t.Context(), "old")
	if err != nil || !ok || saved.Req.Content != pair.Req.Content || saved.Resp.Content != pair.Resp.Content {
		t.Fatal("stored history was rewritten")
	}
	s.setSystemContext("Speak concisely.")
	count := 0
	for _, m := range s.Messages {
		if m.Role == "system" {
			count++
		}
	}
	if count != 1 {
		t.Fatal("system instructions accumulated")
	}
	s.processInput(t.Context(), ptr("system:"))
	for _, m := range s.Messages {
		if m.Role == "system" {
			t.Fatal("empty system command did not clear the instruction")
		}
	}
}
func TestSystemUpdatePreservesLegacySummaryAsData(t *testing.T) {
	s := Session{Messages: []openai.ChatCompletionMessage{{Role: "system", Content: summaryPrefix + "Earlier facts"}, {Role: "user", Content: "Next question"}}}
	s.setSystemContext("Current instruction")
	if len(s.Messages) != 3 || s.Messages[1].Role != "assistant" || !strings.Contains(s.Messages[1].Content, "Earlier facts") {
		t.Fatalf("summary lost/elevated: %+v", s.Messages)
	}
}
func TestFirstSystemOnlyProviderReceivesLatestInstruction(t *testing.T) {
	client := openai.NewClient(option.WithAPIKey("synthetic"), option.WithBaseURL("https://synthetic.invalid/v1/"), option.WithMaxRetries(0), option.WithHTTPClient(&http.Client{Transport: contractTransport(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatal(r.URL)
		}
		var body struct {
			Messages []struct{ Role, Content string }
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if len(body.Messages) < 2 || body.Messages[0].Role != "system" || body.Messages[0].Content != "Talk like a dog." {
			t.Fatalf("first-system template sees wrong instruction: %+v", body.Messages)
		}
		count := 0
		for _, m := range body.Messages {
			if m.Role == "system" {
				count++
			}
		}
		if count != 1 {
			t.Fatal("contradictory system messages remain")
		}
		return syntheticCompletion(r, "synthetic reply"), nil
	})}))
	s := Session{Client: &client, ChatModel: "test", OutWriter: bufio.NewWriter(io.Discard), TermWidth: 80, StorageBackend: memory.NewBackend[string, ReqRespPair]()}
	s.setSystemContext("Talk like a pirate.")
	s.Messages = append(s.Messages, openai.ChatCompletionMessage{Role: "user", Content: "hello"}, openai.ChatCompletionMessage{Role: "assistant", Content: "ahoy"})
	s.setSystemContext("Talk like a dog.")
	if err := s.chatRequest(t.Context(), openai.ChatCompletionMessage{Role: "user", Content: "hello again"}); err != nil {
		t.Fatal(err)
	}
}
func TestSummarizationKeepsInstructionAndDoesNotPromoteData(t *testing.T) {
	client := openai.NewClient(option.WithAPIKey("synthetic"), option.WithBaseURL("https://synthetic.invalid/v1/"), option.WithMaxRetries(0), option.WithHTTPClient(&http.Client{Transport: contractTransport(func(r *http.Request) (*http.Response, error) {
		var body struct {
			Messages []struct{ Role, Content string }
		}
		json.NewDecoder(r.Body).Decode(&body)
		if !strings.Contains(body.Messages[1].Content, "Earlier facts") {
			t.Fatal("legacy summary omitted from recap input")
		}
		return syntheticCompletion(r, "Summarized facts"), nil
	})}))
	s := Session{Client: &client, ChatModel: "test", OutWriter: bufio.NewWriter(io.Discard), StorageBackend: memory.NewBackend[string, ReqRespPair](), CurrentTokensUsed: 10, SummarizeContextWindowSize: 1, Messages: []openai.ChatCompletionMessage{{Role: "system", Content: "Talk like a dog."}, {Role: "system", Content: summaryPrefix + "Earlier facts"}, {Role: "user", Content: "New fact"}}}
	if err := s.maybeSummarize(t.Context()); err != nil {
		t.Fatal(err)
	}
	if len(s.Messages) != 2 || s.Messages[0].Role != "system" || s.Messages[0].Content != "Talk like a dog." || s.Messages[1].Role != "assistant" || !strings.Contains(s.Messages[1].Content, "Summarized facts") {
		t.Fatalf("messages=%+v", s.Messages)
	}
}

func TestExplicitSummaryLookingInstructionRemainsAnInstruction(t *testing.T) {
	prompt := summaryPrefix + "Reply in French"
	client := openai.NewClient(option.WithAPIKey("synthetic"), option.WithBaseURL("https://synthetic.invalid/v1/"), option.WithMaxRetries(0), option.WithHTTPClient(&http.Client{Transport: contractTransport(func(r *http.Request) (*http.Response, error) {
		return syntheticCompletion(r, "Conversation facts"), nil
	})}))
	s := Session{Client: &client, ChatModel: "test", OutWriter: bufio.NewWriter(io.Discard), StorageBackend: memory.NewBackend[string, ReqRespPair](), CurrentTokensUsed: 10, SummarizeContextWindowSize: 1}
	s.setSystemContext(prompt)
	s.Messages = append(s.Messages, openai.ChatCompletionMessage{Role: "user", Content: "hello"})
	if err := s.maybeSummarize(t.Context()); err != nil {
		t.Fatal(err)
	}
	if len(s.Messages) != 2 || s.Messages[0].Role != "system" || s.Messages[0].Content != prompt {
		t.Fatalf("explicit instruction was reclassified: %+v", s.Messages)
	}
	s.setSystemContext("")
	for _, m := range s.Messages {
		if m.Content == prompt {
			t.Fatal("cleared instruction was retained as summary data")
		}
	}
	s.setSystemContext(prompt)
	s.setSystemContext("Replacement")
	for _, m := range s.Messages {
		if m.Content == prompt {
			t.Fatal("replaced instruction was retained as summary data")
		}
	}
}
