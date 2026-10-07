package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func TestMain(m *testing.M) {
	if os.Getenv("OPENAI_CLI_PTY_HELPER") == "1" {
		os.Args = []string{"openai", "--legacy"}
		if os.Getenv("OPENAI_CLI_PTY_MODE") == "tui" {
			os.Args = []string{"openai", "--temporary"}
		}
		if os.Getenv("OPENAI_CLI_PTY_MODE") == "chat" {
			os.Args = append(os.Args, "chat", "--temporary")
		}
		main()
	}

	for _, key := range []string{"OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_BASE_URL", "OPENAI_API_URL", "OPENAI_MODEL"} {
		os.Unsetenv(key)
	}
	os.Exit(m.Run())
}

const responseFixture = `{"id":"resp_test","object":"response","status":"completed","model":"test-model","output":[{"type":"message","id":"msg_test","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello world","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5},"future_field":"preserved"}`

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func fakeResponse(r *http.Request, status int, contentType, body string) *http.Response {
	return &http.Response{StatusCode: status, Header: http.Header{"Content-Type": []string{contentType}}, Body: io.NopCloser(strings.NewReader(body)), Request: r}
}
func executeTest(t *testing.T, args []string, input string, transport roundTripFunc) (string, string, error) {
	t.Helper()
	cmd := newRootCommand(option.WithAPIKey("synthetic-key"), option.WithHTTPClient(&http.Client{Transport: transport}), option.WithMaxRetries(0))
	var out, errOut bytes.Buffer
	cmd.SetArgs(args)
	cmd.SetIn(strings.NewReader(input))
	cmd.SetOut(&out)
	cmd.SetErr(&errOut)
	err := cmd.ExecuteContext(t.Context())
	return out.String(), errOut.String(), err
}

func TestCreateResponseCommands(t *testing.T) {
	for _, tc := range []struct {
		name          string
		args          []string
		input, prompt string
	}{
		{"root argument", []string{"hello", "there"}, "unused", "hello there"},
		{"root pipe", nil, "piped\ninput", "piped\ninput"},
		{"responses argument", []string{"responses", "hello"}, "", "hello"},
		{"create pipe", []string{"responses", "create"}, "input", "input"},
		{"stdin sentinel", []string{"responses", "create", "-"}, "input", "input"},
		{"chat pipe", []string{"responses", "chat"}, "input", "input"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			out, errOut, err := executeTest(t, tc.args, tc.input, func(r *http.Request) (*http.Response, error) {
				calls++
				if r.Method != "POST" || r.URL.Path != "/v1/responses" {
					t.Errorf("request = %s %s", r.Method, r.URL)
				}
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Fatal(err)
				}
				if body["input"] != tc.prompt || body["store"] != false {
					t.Errorf("body = %v", body)
				}
				if _, ok := body["tools"]; ok {
					t.Error("unexpected paid web search tool")
				}
				return fakeResponse(r, 200, "application/json", responseFixture), nil
			})
			if err != nil || out != "hello world\n" || errOut != "" || calls != 1 {
				t.Fatalf("out=%q stderr=%q calls=%d err=%v", out, errOut, calls, err)
			}
		})
	}
}

func TestResponseGetRetrieves(t *testing.T) {
	out, _, err := executeTest(t, []string{"responses", "get", "resp_test", "--output", "json"}, "", func(r *http.Request) (*http.Response, error) {
		if r.Method != "GET" || r.URL.Path != "/v1/responses/resp_test" {
			t.Errorf("request = %s %s", r.Method, r.URL)
		}
		return fakeResponse(r, 200, "application/json", responseFixture), nil
	})
	if err != nil || !json.Valid([]byte(out)) || !strings.Contains(out, `"future_field":"preserved"`) {
		t.Fatalf("out=%q err=%v", out, err)
	}
}

func TestInvalidInputNeverRequests(t *testing.T) {
	for _, args := range [][]string{
		{}, {"responses", "get"}, {"responses", "get", "a", "b"}, {"responses", "delete"},
		{"--output", "yaml", "hello"}, {"--timeout", "-1s", "hello"}, {"--base-url", "file:///tmp", "hello"},
		{"--base-url", "https://user:pass@example.com/v1", "hello"}, {"--model", "", "hello"},
	} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			out, _, err := executeTest(t, args, " \n", func(r *http.Request) (*http.Response, error) {
				t.Fatal("unexpected request")
				return nil, errors.New("unexpected")
			})
			if err == nil || out != "" {
				t.Fatalf("out=%q err=%v", out, err)
			}
		})
	}
}

func TestAPIErrorPropagates(t *testing.T) {
	out, _, err := executeTest(t, []string{"responses", "chat"}, "hello", func(r *http.Request) (*http.Response, error) {
		return fakeResponse(r, 429, "application/json", `{"error":{"message":"synthetic rate limit","type":"rate_limit_error"}}`), nil
	})
	if err == nil || !strings.Contains(err.Error(), "429") || out != "" {
		t.Fatalf("out=%q err=%v", out, err)
	}
}

func TestConfigurationPrecedence(t *testing.T) {
	t.Setenv("OPENAI_API_URL", "https://legacy.invalid/api/")
	t.Setenv("OPENAI_BASE_URL", "https://current.invalid/v1/")
	t.Setenv("OPENAI_MODEL", "configured-model")
	for _, tc := range []struct {
		args        []string
		host, model string
	}{
		{[]string{"hello"}, "current.invalid", "configured-model"},
		{[]string{"--base-url", "https://flag.invalid/v1/", "--model", "flag-model", "hello"}, "flag.invalid", "flag-model"},
	} {
		_, _, err := executeTest(t, tc.args, "", func(r *http.Request) (*http.Response, error) {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			if r.URL.Host != tc.host || body["model"] != tc.model {
				t.Errorf("url=%s body=%v", r.URL, body)
			}
			return fakeResponse(r, 200, "application/json", responseFixture), nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("OPENAI_BASE_URL", "")
	_, _, err := executeTest(t, []string{"hello"}, "", func(r *http.Request) (*http.Response, error) {
		if r.URL.Host != "legacy.invalid" {
			t.Errorf("url=%s", r.URL)
		}
		return fakeResponse(r, 200, "application/json", responseFixture), nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestResponseStreaming(t *testing.T) {
	event := func(body string) string { return "data: " + body + "\n\n" }
	delta := event(`{"type":"response.output_text.delta","delta":"hello "}`) + event(`{"type":"response.output_text.delta","delta":"world"}`)
	completed := event(`{"type":"response.completed","response":` + responseFixture + `}`)
	for _, tc := range []struct{ name, body, format, want, errorText string }{
		{"text", delta + completed, "text", "hello world\n", ""},
		{"json", delta + completed, "json", responseFixture + "\n", ""},
		{"truncated", delta, "text", "hello world", "before completion"},
		{"malformed", event(`{bad json`), "text", "", "response stream"},
		{"failed", event(`{"type":"response.failed","response":{"status":"failed","error":{"message":"synthetic failure"}}}`), "text", "", "synthetic failure"},
		{"incomplete", event(`{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`), "text", "", "max_output_tokens"},
		{"error", event(`{"type":"error","message":"synthetic error"}`), "text", "", "synthetic error"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, _, err := executeTest(t, []string{"hello", "--stream", "--output", tc.format}, "", func(r *http.Request) (*http.Response, error) {
				return fakeResponse(r, 200, "text/event-stream", tc.body), nil
			})
			if out != tc.want {
				t.Errorf("out=%q want=%q", out, tc.want)
			}
			if tc.errorText == "" && err != nil || tc.errorText != "" && (err == nil || !strings.Contains(err.Error(), tc.errorText)) {
				t.Errorf("err=%v", err)
			}
		})
	}
}

type brokenWriter struct{}

func (brokenWriter) Write(p []byte) (int, error) { return 0, io.ErrClosedPipe }
func TestOutputFailurePropagates(t *testing.T) {
	cmd := newRootCommand(option.WithAPIKey("synthetic-key"), option.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return fakeResponse(r, 200, "application/json", responseFixture), nil
	})}))
	cmd.SetArgs([]string{"hello"})
	cmd.SetOut(brokenWriter{})
	if err := cmd.ExecuteContext(t.Context()); !errors.Is(err, io.ErrClosedPipe) {
		t.Fatalf("err=%v", err)
	}
}

func TestRequestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	cmd := newRootCommand(option.WithAPIKey("synthetic-key"), option.WithMaxRetries(0), option.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		cancel()
		<-r.Context().Done()
		return nil, r.Context().Err()
	})}))
	cmd.SetArgs([]string{"hello"})
	cmd.SetOut(io.Discard)
	if err := cmd.ExecuteContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v", err)
	}
}

func TestRunExitCodes(t *testing.T) {
	for _, tc := range []struct {
		args   []string
		cancel bool
		code   int
	}{
		{[]string{"--help"}, false, 0}, {[]string{"--invalid"}, false, 1}, {[]string{"hello"}, true, 130},
	} {
		ctx, cancel := context.WithCancel(t.Context())
		if tc.cancel {
			cancel()
		}
		var out, errOut bytes.Buffer
		code := run(ctx, tc.args, strings.NewReader(""), &out, &errOut)
		cancel()
		if code != tc.code {
			t.Errorf("args=%v code=%d stderr=%s", tc.args, code, errOut.String())
		}
	}
}

func TestProcessExitCodes(t *testing.T) {
	if os.Getenv("OPENAI_CLI_TEST_HELPER") == "1" {
		args := strings.Split(os.Getenv("OPENAI_CLI_TEST_ARGS"), " ")
		os.Exit(run(context.Background(), args, strings.NewReader(""), os.Stdout, os.Stderr))
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		args string
		code int
	}{{"--help", 0}, {"--invalid", 1}, {"responses get", 1}} {
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		cmd := exec.CommandContext(ctx, executable, "-test.run=^TestProcessExitCodes$")
		// Explicit environment: no inherited API keys, credentials, or SDK config.
		cmd.Env = []string{"OPENAI_CLI_TEST_HELPER=1", "OPENAI_CLI_TEST_ARGS=" + tc.args}
		var out, errOut bytes.Buffer
		cmd.Stdout = &out
		cmd.Stderr = &errOut
		err := cmd.Run()
		cancel()
		code := 0
		if err != nil {
			var ee *exec.ExitError
			if !errors.As(err, &ee) {
				t.Fatal(err)
			}
			code = ee.ExitCode()
		}
		if code != tc.code {
			t.Errorf("%s: code=%d stderr=%s", tc.args, code, errOut.String())
		}
		if code != 0 && (out.Len() != 0 || !strings.HasPrefix(errOut.String(), "openai:")) {
			t.Errorf("stdout=%q stderr=%q", out.String(), errOut.String())
		}
	}
}

func TestPromptLimitAndCanceledRead(t *testing.T) {
	_, err := readPrompt(t.Context(), strings.NewReader(strings.Repeat("x", maxPromptBytes+1)), nil)
	if err == nil {
		t.Fatal("expected oversized prompt error")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := readPrompt(ctx, strings.NewReader("input"), nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func Example() {
	fmt.Println("openai responses create 'Hello' --output json")
	// Output: openai responses create 'Hello' --output json
}

func TestRefusalText(t *testing.T) {
	fixture := `{"id":"resp_refusal","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"Synthetic refusal."}]}]}`
	for _, args := range [][]string{{"responses", "create", "hello"}, {"responses", "get", "resp_refusal"}} {
		out, _, err := executeTest(t, args, "", func(r *http.Request) (*http.Response, error) {
			return fakeResponse(r, 200, "application/json", fixture), nil
		})
		if err != nil || out != "Synthetic refusal.\n" {
			t.Fatalf("out=%q err=%v", out, err)
		}
	}
}

func TestCleanupAfterCancellation(t *testing.T) {
	var paths []string
	client := openai.NewClient(option.WithAPIKey("synthetic-key"), option.WithHTTPClient(&http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Context().Err() != nil {
			t.Error("cleanup inherited cancellation")
		}
		if r.Method != "DELETE" {
			t.Errorf("method=%s", r.Method)
		}
		paths = append(paths, r.URL.Path)
		if strings.HasSuffix(r.URL.Path, "first") {
			return fakeResponse(r, 500, "application/json", `{"error":{"message":"synthetic failure"}}`), nil
		}
		return fakeResponse(r, 204, "application/json", ""), nil
	})}))
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	err := cleanupResponses(ctx, &client, []string{"first", "second"})
	if err == nil || !strings.Contains(err.Error(), "first") || len(paths) != 2 {
		t.Fatalf("paths=%v err=%v", paths, err)
	}
}
