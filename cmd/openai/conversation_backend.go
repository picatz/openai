package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/picatz/openai/internal/conversation"
)

type sdkBackend struct {
	client            *openai.Client
	preserveReasoning bool
}

func (b sdkBackend) Generate(ctx context.Context, req conversation.Request, emit func(string) error) (conversation.Result, error) {
	if req.API == conversation.Chat {
		return b.chat(ctx, req, emit)
	}
	if req.API != conversation.Responses {
		return conversation.Result{}, fmt.Errorf("unsupported API %q", req.API)
	}
	input := make([]responses.ResponseInputItemUnionParam, 0, len(req.ResponsesItems)+len(req.Messages))
	replay := conversation.CloneItems(req.ResponsesItems)
	if len(replay) > 0 {
		if len(req.Messages) == 0 || req.Messages[len(req.Messages)-1].Role != "user" {
			return conversation.Result{}, fmt.Errorf("a new user message is required")
		}
		for _, item := range replay {
			input = append(input, param.Override[responses.ResponseInputItemUnionParam](item))
		}
		last := req.Messages[len(req.Messages)-1]
		message := responses.ResponseInputItemParamOfMessage(last.Content, responses.EasyInputMessageRole(last.Role))
		raw, err := json.Marshal(message)
		if err != nil {
			return conversation.Result{}, err
		}
		replay = append(replay, raw)
		input = append(input, message)
	} else {
		for _, m := range req.Messages {
			message := responses.ResponseInputItemParamOfMessage(m.Content, responses.EasyInputMessageRole(m.Role))
			raw, err := json.Marshal(message)
			if err != nil {
				return conversation.Result{}, err
			}
			replay = append(replay, raw)
			input = append(input, message)
		}
	}
	params := responses.ResponseNewParams{Model: req.Model, Store: openai.Bool(false), Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input}}
	if b.preserveReasoning {
		params.Include = []responses.ResponseIncludable{responses.ResponseIncludableReasoningEncryptedContent}
	}

	if req.WebSearch {
		params.Tools = []responses.ToolUnionParam{responses.ToolParamOfWebSearchPreview(responses.WebSearchPreviewToolTypeWebSearchPreview)}
	}
	stream := b.client.Responses.NewStreaming(ctx, params)
	defer stream.Close()
	var streamed strings.Builder
	for stream.Next() {
		e := stream.Current()
		switch e.Type {
		case "response.output_text.delta", "response.refusal.delta":
			streamed.WriteString(e.Delta)
			if streamed.Len() > 8<<20 {
				return conversation.Result{}, fmt.Errorf("response exceeds the 8 MiB interactive limit")
			}
			if err := emit(e.Delta); err != nil {
				return conversation.Result{}, err
			}
		case "response.completed":
			if err := responseError(&e.Response); err != nil {
				return conversation.Result{}, err
			}
			for _, item := range e.Response.Output {
				switch item.Type {
				case "message", "reasoning", "web_search_call":
				default:
					return conversation.Result{}, fmt.Errorf("response contains unsupported output %q; use a tool-aware client (saved history is unchanged)", item.Type)
				}
			}
			// The completed payload is canonical; providers may coalesce text events.
			result := conversation.Result{Text: responseText(&e.Response), ResponseID: e.Response.ID, Usage: conversation.Usage{Input: e.Response.Usage.InputTokens, Output: e.Response.Usage.OutputTokens, Total: e.Response.Usage.TotalTokens}}
			if !strings.HasPrefix(result.Text, streamed.String()) {
				return result, fmt.Errorf("completed response text did not match the streamed text")
			}
			if suffix := strings.TrimPrefix(result.Text, streamed.String()); suffix != "" {
				if err := emit(suffix); err != nil {
					return result, err
				}
			}
			if result.Text == "" {
				return result, fmt.Errorf("response completed without text; use --output json to inspect non-text output")
			}
			for _, item := range e.Response.Output {
				replay = append(replay, json.RawMessage(item.RawJSON()))
			}
			result.ResponsesItems = replay
			return result, nil
		case "response.failed", "response.incomplete":
			return conversation.Result{}, responseError(&e.Response)
		case "error":
			return conversation.Result{}, fmt.Errorf("response stream: %s", e.Message)
		}
	}
	if err := stream.Err(); err != nil {
		return conversation.Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return conversation.Result{}, err
	}
	return conversation.Result{}, fmt.Errorf("response stream ended before completion")
}
func (b sdkBackend) chat(ctx context.Context, req conversation.Request, emit func(string) error) (conversation.Result, error) {
	if req.WebSearch {
		return conversation.Result{}, fmt.Errorf("built-in web search is not available through Chat Completions")
	}
	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(req.Messages))
	for _, m := range req.Messages {
		switch m.Role {
		case "system":
			messages = append(messages, openai.SystemMessage(m.Content))
		case "user":
			messages = append(messages, openai.UserMessage(m.Content))
		case "assistant":
			messages = append(messages, openai.AssistantMessage(m.Content))
		default:
			return conversation.Result{}, fmt.Errorf("unsupported message role %q", m.Role)
		}
	}
	stream := b.client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{Model: req.Model, Messages: messages, StreamOptions: openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)}})
	defer stream.Close()
	var result conversation.Result
	var text strings.Builder
	finished := false
	for stream.Next() {
		chunk := stream.Current()
		// Some compatible providers report stream errors in HTTP 200 frames.
		var envelope struct {
			Error json.RawMessage `json:"error"`
		}
		if json.Unmarshal([]byte(chunk.RawJSON()), &envelope) == nil && len(envelope.Error) > 0 && string(envelope.Error) != "null" {
			return result, fmt.Errorf("Chat Completions stream returned an error frame")
		}
		if chunk.ID != "" {
			result.ResponseID = chunk.ID
		}
		if chunk.Usage.TotalTokens > 0 {
			result.Usage = conversation.Usage{Input: chunk.Usage.PromptTokens, Output: chunk.Usage.CompletionTokens, Total: chunk.Usage.TotalTokens}
		}
		for _, choice := range chunk.Choices {
			if choice.Index != 0 {
				return result, fmt.Errorf("multiple completion choices are not supported in interactive chat")
			}
			if len(choice.Delta.ToolCalls) > 0 || choice.Delta.FunctionCall.Name != "" {
				return result, fmt.Errorf("tool calls require a tool-aware client; this chat accepts text only")
			}
			delta := choice.Delta.Content + choice.Delta.Refusal
			text.WriteString(delta)
			if text.Len() > 8<<20 {
				return result, fmt.Errorf("response exceeds the 8 MiB interactive limit")
			}
			if delta != "" {
				if err := emit(delta); err != nil {
					return result, err
				}
			}
			switch choice.FinishReason {
			case "":
			case "stop":
				finished = true
			default:
				return result, fmt.Errorf("chat completion did not finish normally: %s", choice.FinishReason)
			}
		}
	}
	if err := stream.Err(); err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if !finished {
		return result, fmt.Errorf("chat completion stream ended before completion")
	}
	result.Text = text.String()
	if result.Text == "" {
		return result, fmt.Errorf("chat completion returned no text")
	}
	return result, nil
}
