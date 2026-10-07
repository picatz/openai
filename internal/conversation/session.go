// Package conversation holds provider-neutral local chat state. API modes remain
// explicit: the UI does not translate unsupported provider capabilities.
package conversation

import (
	"context"
	"encoding/json"
	"time"

	"github.com/segmentio/ksuid"
)

type API string

const (
	Chat      API = "chat"
	Responses API = "responses"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type Usage struct {
	Input  int64 `json:"input"`
	Output int64 `json:"output"`
	Total  int64 `json:"total"`
}
type Session struct {
	Version        int               `json:"version"`
	ID             string            `json:"id"`
	Revision       int               `json:"revision"`
	API            API               `json:"api"`
	Endpoint       string            `json:"endpoint"`
	Model          string            `json:"model"`
	Created        time.Time         `json:"created"`
	Updated        time.Time         `json:"updated"`
	ResponsesItems []json.RawMessage `json:"responses_items,omitempty"`
	Messages       []Message         `json:"messages"`
	Usage          Usage             `json:"usage"`
	LastResponseID string            `json:"last_response_id,omitempty"`
}

func New(api API, endpoint, model string) Session {
	now := time.Now().UTC()
	return Session{Version: 1, ID: ksuid.New().String(), API: api, Endpoint: endpoint, Model: model, Created: now, Updated: now}
}
func (s Session) Clone() Session {
	s.Messages = append([]Message(nil), s.Messages...)
	s.ResponsesItems = CloneItems(s.ResponsesItems)
	return s
}
func (s Session) Title() string {
	for _, m := range s.Messages {
		if m.Role == "user" {
			r := []rune(m.Content)
			if len(r) > 48 {
				return string(r[:48]) + "…"
			}
			return m.Content
		}
	}
	return "New conversation"
}

type Request struct {
	ResponsesItems []json.RawMessage
	API            API
	Model          string
	Messages       []Message
	WebSearch      bool
}
type Result struct {
	ResponsesItems []json.RawMessage
	Text           string
	ResponseID     string
	Usage          Usage
}

// Backend returns a complete result or an error. emit applies backpressure and
// may abort generation by returning an error. Implementations must honor ctx.
type Backend interface {
	Generate(context.Context, Request, func(string) error) (Result, error)
}
type BackendFunc func(context.Context, Request, func(string) error) (Result, error)

func (f BackendFunc) Generate(ctx context.Context, r Request, emit func(string) error) (Result, error) {
	return f(ctx, r, emit)
}

func CloneItems(items []json.RawMessage) []json.RawMessage {
	out := make([]json.RawMessage, len(items))
	for i, item := range items {
		out[i] = append(json.RawMessage(nil), item...)
	}
	return out
}
