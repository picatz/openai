package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestDecisionQuestions(t *testing.T) {
	for _, tc := range []struct {
		kind         string
		options      []string
		answer, want string
	}{
		{"predicate", nil, `{"name":"result","type":"predicate","probability":0.75}`, "probability=0.750000"},
		{"choice", []string{"--choice", "a=first", "--choice", "b=second"}, `{"name":"result","type":"choice","choice":"b","confidence":0.8,"probabilities":[{"value":"b","probability":0.8}]}`, "b\tconfidence=0.800000"},
		{"score", []string{"--level", "low=minor", "--level", "high=major"}, `{"name":"result","type":"score","score":0.25,"confidence":0.8}`, "score=0.250000"},
	} {
		t.Run(tc.kind, func(t *testing.T) {
			for _, format := range []string{"text", "json"} {
				args := append([]string{"decisions", tc.kind, "input", "--question", "evaluate this", "--output", format}, tc.options...)
				out, _, err := executeTest(t, args, "", func(r *http.Request) (*http.Response, error) {
					if r.Method != "POST" || r.URL.Path != "/v1/decisions" {
						t.Error(r.Method, r.URL)
					}
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Fatal(err)
					}
					if body["model"] != "gpt-6-luna" || body["input"] != "input" {
						t.Errorf("body=%v", body)
					}
					return fakeResponse(r, 200, "application/json", `{"answers":[`+tc.answer+`],"future":"preserved"}`), nil
				})
				if err != nil {
					t.Fatal(err)
				}
				if format == "text" && !strings.Contains(out, tc.want) || format == "json" && (!json.Valid([]byte(out)) || !strings.Contains(out, `"future":"preserved"`)) {
					t.Fatalf("out=%q", out)
				}
			}
		})
	}
}
func TestDecisionValidationBeforeRequest(t *testing.T) {
	for _, args := range [][]string{
		{"decisions", "predicate", "input"},
		{"decisions", "predicate", "input", "--question", ""},
		{"decisions", "choice", "input", "--question", "q", "--choice", "a=one"},
		{"decisions", "choice", "input", "--question", "q", "--choice", "a=one", "--choice", "a=duplicate"},
		{"decisions", "score", "input", "--question", "q", "--level", "broken"},
		{"decisions", "predicate", "input", "--question", "q", "--stream"},
	} {
		_, _, err := executeTest(t, args, "", func(r *http.Request) (*http.Response, error) {
			t.Fatal("invalid input made a request")
			return nil, nil
		})
		if err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}
func TestDecisionResponseValidation(t *testing.T) {
	for _, response := range []string{`{}`, `{"answers":[]}`, `{"answers":[{"type":"predicate","name":"wrong","probability":0.2}]}`, `{"answers":[{"type":"predicate","name":"result","probability":2}]}`} {
		_, _, err := executeTest(t, []string{"decisions", "predicate", "input", "--question", "q"}, "", func(r *http.Request) (*http.Response, error) {
			return fakeResponse(r, 200, "application/json", response), nil
		})
		if err == nil {
			t.Fatalf("accepted %s", response)
		}
	}
}
func TestDecisionDoesNotFallBackOnUnsupportedProvider(t *testing.T) {
	calls := 0
	_, _, err := executeTest(t, []string{"decisions", "predicate", "input", "--question", "q", "--base-url", "https://compatible.invalid/v1"}, "", func(r *http.Request) (*http.Response, error) {
		calls++
		if r.URL.Path != "/v1/decisions" {
			t.Error("translated endpoint")
		}
		return fakeResponse(r, 404, "application/json", `{"error":{"message":"unsupported"}}`), nil
	})
	if err == nil || calls != 1 {
		t.Fatalf("calls=%d err=%v", calls, err)
	}
}

func TestDecisionRefusalIsAValidOutcome(t *testing.T) {
	for _, format := range []string{"text", "json"} {
		out, _, err := executeTest(t, []string{"decisions", "predicate", "input", "--question", "q", "--output", format}, "", func(r *http.Request) (*http.Response, error) {
			return fakeResponse(r, 200, "application/json", `{"answers":[{"type":"refusal","name":"result"}],"future":true}`), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if format == "text" && out != "result\trefusal\n" || format == "json" && (!json.Valid([]byte(out)) || !strings.Contains(out, `"type":"refusal"`)) {
			t.Fatalf("out=%q", out)
		}
	}
}
