package decisions

import (
	"strings"
	"testing"
)

func TestRequestValidate(t *testing.T) {
	predicate := Question{Type: "predicate", Name: "q1", Instructions: "is it true"}
	choice := Question{Type: "choice", Name: "q2", Instructions: "pick one", Choices: []Choice{
		{Value: "a", Description: "first"},
		{Value: "b", Description: "second"},
	}}
	score := Question{Type: "score", Name: "q3", Instructions: "rate it", Levels: []Level{
		{Label: "low", Description: "minor"},
		{Label: "high", Description: "major"},
	}}

	for _, tc := range []struct {
		name    string
		req     Request
		wantErr string
	}{
		{"valid predicate", Request{Model: "m", Input: "in", Questions: []Question{predicate}}, ""},
		{"valid choice", Request{Model: "m", Input: "in", Questions: []Question{choice}}, ""},
		{"valid score", Request{Model: "m", Input: "in", Questions: []Question{score}}, ""},
		{"valid mixed", Request{Model: "m", Input: "in", Questions: []Question{predicate, choice, score}}, ""},
		{"missing model", Request{Model: "", Input: "in", Questions: []Question{predicate}}, "model, input, and at least one question are required"},
		{"missing input", Request{Model: "m", Input: "", Questions: []Question{predicate}}, "model, input, and at least one question are required"},
		{"whitespace model", Request{Model: "  ", Input: "in", Questions: []Question{predicate}}, "model, input, and at least one question are required"},
		{"no questions", Request{Model: "m", Input: "in", Questions: nil}, "model, input, and at least one question are required"},
		{"missing question name", Request{Model: "m", Input: "in", Questions: []Question{{Type: "predicate", Name: "", Instructions: "x"}}}, "question name and instructions are required"},
		{"missing question instructions", Request{Model: "m", Input: "in", Questions: []Question{{Type: "predicate", Name: "q", Instructions: ""}}}, "question name and instructions are required"},
		{"duplicate question name", Request{Model: "m", Input: "in", Questions: []Question{predicate, predicate}}, "duplicate question name"},
		{"predicate with choices", Request{Model: "m", Input: "in", Questions: []Question{{Type: "predicate", Name: "q", Instructions: "x", Choices: []Choice{{Value: "a", Description: "b"}}}}}, "predicate questions do not take choices or levels"},
		{"predicate with levels", Request{Model: "m", Input: "in", Questions: []Question{{Type: "predicate", Name: "q", Instructions: "x", Levels: score.Levels}}}, "predicate questions do not take choices or levels"},
		{"choice too few", Request{Model: "m", Input: "in", Questions: []Question{{Type: "choice", Name: "q", Instructions: "x", Choices: []Choice{{Value: "a", Description: "b"}}}}}, "choice questions require at least two choices"},
		{"choice with levels", Request{Model: "m", Input: "in", Questions: []Question{{Type: "choice", Name: "q", Instructions: "x", Choices: choice.Choices, Levels: score.Levels}}}, "choice questions require at least two choices"},
		{"choice duplicate value", Request{Model: "m", Input: "in", Questions: []Question{{Type: "choice", Name: "q", Instructions: "x", Choices: []Choice{
			{Value: "a", Description: "first"},
			{Value: "a", Description: "second"},
		}}}}, "choices need unique nonempty values"},
		{"choice empty value", Request{Model: "m", Input: "in", Questions: []Question{{Type: "choice", Name: "q", Instructions: "x", Choices: []Choice{
			{Value: "", Description: "first"},
			{Value: "b", Description: "second"},
		}}}}, "choices need unique nonempty values"},
		{"choice empty description", Request{Model: "m", Input: "in", Questions: []Question{{Type: "choice", Name: "q", Instructions: "x", Choices: []Choice{
			{Value: "a", Description: ""},
			{Value: "b", Description: "second"},
		}}}}, "choices need unique nonempty values"},
		{"score too few", Request{Model: "m", Input: "in", Questions: []Question{{Type: "score", Name: "q", Instructions: "x", Levels: []Level{{Label: "low", Description: "minor"}}}}}, "score questions require at least two ordered levels"},
		{"score with choices", Request{Model: "m", Input: "in", Questions: []Question{{Type: "score", Name: "q", Instructions: "x", Levels: score.Levels, Choices: choice.Choices}}}, "score questions require at least two ordered levels"},
		{"score duplicate label", Request{Model: "m", Input: "in", Questions: []Question{{Type: "score", Name: "q", Instructions: "x", Levels: []Level{
			{Label: "low", Description: "minor"},
			{Label: "low", Description: "major"},
		}}}}, "levels need unique nonempty labels"},
		{"score empty label", Request{Model: "m", Input: "in", Questions: []Question{{Type: "score", Name: "q", Instructions: "x", Levels: []Level{
			{Label: "", Description: "minor"},
			{Label: "high", Description: "major"},
		}}}}, "levels need unique nonempty labels"},
		{"score empty description", Request{Model: "m", Input: "in", Questions: []Question{{Type: "score", Name: "q", Instructions: "x", Levels: []Level{
			{Label: "low", Description: ""},
			{Label: "high", Description: "major"},
		}}}}, "levels need unique nonempty labels"},
		{"unsupported type", Request{Model: "m", Input: "in", Questions: []Question{{Type: "essay", Name: "q", Instructions: "x"}}}, `unsupported question type "essay"`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.req.Validate()
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("unexpected error: %v", err)
			case tc.wantErr != "" && err == nil:
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			case tc.wantErr != "" && !strings.Contains(err.Error(), tc.wantErr):
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantErr)
			}
		})
	}
}
