// Package decisions is a narrow adapter for the public-beta Decisions endpoint.
// The pinned Go SDK has no generated Decisions service; Client.Post preserves its
// authentication, base URL, retries, error handling and cancellation semantics.
// Reference: https://developers.openai.com/api/docs/guides/decisions
package decisions

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/picatz/openai/internal/httpguard"
)

type Choice struct {
	Value       string `json:"value"`
	Description string `json:"description"`
}
type Level struct {
	Label       string `json:"label"`
	Description string `json:"description"`
}
type Question struct {
	Type         string   `json:"type"`
	Name         string   `json:"name"`
	Instructions string   `json:"instructions"`
	Choices      []Choice `json:"choices,omitempty"`
	Levels       []Level  `json:"levels,omitempty"`
}
type Request struct {
	Model     string     `json:"model"`
	Input     string     `json:"input"`
	Questions []Question `json:"questions"`
}
type Answer struct {
	Type        string   `json:"type"`
	Name        string   `json:"name"`
	Probability *float64 `json:"probability,omitempty"`
	Choice      string   `json:"choice,omitempty"`
	Score       *float64 `json:"score,omitempty"`
	Confidence  *float64 `json:"confidence,omitempty"`
}
type Result struct {
	Answers []Answer        `json:"answers"`
	Raw     json.RawMessage `json:"-"`
}

func (r Request) Validate() error {
	if strings.TrimSpace(r.Model) == "" || strings.TrimSpace(r.Input) == "" || len(r.Questions) == 0 {
		return fmt.Errorf("model, input, and at least one question are required")
	}
	names := map[string]bool{}
	for _, q := range r.Questions {
		if strings.TrimSpace(q.Name) == "" || strings.TrimSpace(q.Instructions) == "" {
			return fmt.Errorf("question name and instructions are required")
		}
		if names[q.Name] {
			return fmt.Errorf("duplicate question name %q", q.Name)
		}
		names[q.Name] = true
		switch q.Type {
		case "predicate":
			if len(q.Choices) > 0 || len(q.Levels) > 0 {
				return fmt.Errorf("predicate questions do not take choices or levels")
			}
		case "choice":
			if len(q.Choices) < 2 || len(q.Levels) > 0 {
				return fmt.Errorf("choice questions require at least two choices and no levels")
			}
			values := map[string]bool{}
			for _, c := range q.Choices {
				if strings.TrimSpace(c.Value) == "" || strings.TrimSpace(c.Description) == "" || values[c.Value] {
					return fmt.Errorf("choices need unique nonempty values and descriptions")
				}
				values[c.Value] = true
			}
		case "score":
			if len(q.Levels) < 2 || len(q.Choices) > 0 {
				return fmt.Errorf("score questions require at least two ordered levels and no choices")
			}
			labels := map[string]bool{}
			for _, l := range q.Levels {
				if strings.TrimSpace(l.Label) == "" || strings.TrimSpace(l.Description) == "" || labels[l.Label] {
					return fmt.Errorf("levels need unique nonempty labels and descriptions")
				}
				labels[l.Label] = true
			}
		default:
			return fmt.Errorf("unsupported question type %q", q.Type)
		}
	}
	return nil
}
func Create(ctx context.Context, client *openai.Client, req Request) (Result, error) {
	var result Result
	if err := req.Validate(); err != nil {
		return result, err
	}
	var raw json.RawMessage
	if err := client.Post(ctx, "decisions", req, &raw, httpguard.LimitJSONResponse(8<<20, 1<<20)); err != nil {
		return result, fmt.Errorf("create decision: %w", err)
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return result, fmt.Errorf("decode decision: %w", err)
	}
	result.Raw = append(json.RawMessage(nil), raw...)
	if len(result.Answers) != len(req.Questions) {
		return result, fmt.Errorf("decision returned %d answers for %d questions", len(result.Answers), len(req.Questions))
	}
	questions := map[string]Question{}
	for _, q := range req.Questions {
		questions[q.Name] = q
	}
	for _, answer := range result.Answers {
		q, ok := questions[answer.Name]
		if !ok || (answer.Type != q.Type && answer.Type != "refusal") {
			return result, fmt.Errorf("unexpected decision answer %q", answer.Name)
		}
		delete(questions, answer.Name)
		if answer.Confidence != nil && (*answer.Confidence < 0 || *answer.Confidence > 1) {
			return result, fmt.Errorf("invalid confidence for %q", answer.Name)
		}
		switch answer.Type {
		case "predicate":
			if answer.Probability == nil || *answer.Probability < 0 || *answer.Probability > 1 {
				return result, fmt.Errorf("invalid probability for %q", answer.Name)
			}
		case "choice":
			found := false
			for _, c := range q.Choices {
				if c.Value == answer.Choice {
					found = true
				}
			}
			if !found {
				return result, fmt.Errorf("unknown choice for %q", answer.Name)
			}
		case "score":
			if answer.Score == nil || *answer.Score < 0 || *answer.Score > float64(len(q.Levels)-1) {
				return result, fmt.Errorf("invalid score for %q", answer.Name)
			}
		}
	}
	return result, nil
}
