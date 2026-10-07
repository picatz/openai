package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/picatz/openai/internal/provider"
)

func fullCapabilities() provider.Capabilities {
	capabilities := make(provider.Capabilities)
	for _, endpoint := range []provider.Endpoint{provider.ChatCompletions, provider.Responses} {
		capabilities[endpoint] = provider.EndpointCapabilities{
			Endpoint: provider.Supported, Streaming: provider.Supported, Tools: provider.Supported, ToolChoice: provider.Supported,
			StructuredOutput: provider.Supported, State: provider.Supported,
		}
	}
	return capabilities
}

func newHandler(t *testing.T, config Config) *Handler {
	t.Helper()
	handler, err := New(config)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func request(t *testing.T, handler http.Handler, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(recorder, req)
	return recorder
}

func TestExactRoutesCredentialsAndPayload(t *testing.T) {
	for _, endpoint := range []provider.Endpoint{provider.ChatCompletions, provider.Responses} {
		t.Run(string(endpoint), func(t *testing.T) {
			// Unknown item variants, usage fields and whitespace must survive.
			body := " {\n\"model\":\"local-a\",\"store\":false,\"input\":[{\"type\":\"future_item\",\"opaque\":[1,2]}],\"future\": { \"a\": true }}\n"
			response := " {\"id\":\"synthetic-id\",\"output\":[{\"type\":\"reasoning\",\"opaque\":7},{\"type\":\"function_call\",\"call_id\":\"call-1\"},{\"type\":\"unknown_item\"}],\"usage\":{\"future\":9}}\n"
			var calls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/nested/v1/"+endpoint.Path() || r.URL.RawQuery != "" {
					t.Errorf("method/path = %s %s", r.Method, r.URL)
				}
				if r.Header.Get("Authorization") != "Bearer fake-a" || r.Header.Get("OpenAI-Organization") != "fake-org-a" || r.Header.Get("OpenAI-Project") != "fake-project-a" {
					t.Error("route-specific credentials were not selected")
				}
				for _, name := range []string{"Cookie", "X-Api-Key", "Forwarded", "X-Forwarded-Host", "X-Forwarded-For", "Idempotency-Key", "X-Secret"} {
					if r.Header.Get(name) != "" || r.Trailer.Get(name) != "" {
						t.Errorf("caller header or trailer forwarded: %s", name)
					}
				}
				if r.Header.Get("Accept") != "application/json" {
					t.Error("Accept was lost")
				}
				got, _ := io.ReadAll(r.Body)
				if string(got) != body {
					t.Errorf("request bytes changed: %q", got)
				}
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("X-Request-Id", "synthetic-request")
				w.Header().Set("Connection", "X-Upstream-Hop")
				w.Header().Set("X-Upstream-Hop", "must-be-removed")
				w.WriteHeader(http.StatusCreated)
				io.WriteString(w, response)
			}))
			defer upstream.Close()
			handler := newHandler(t, Config{Routes: []Route{{Model: "local-a", BaseURL: upstream.URL + "/nested/v1/", APIKey: "fake-a", Organization: "fake-org-a", Project: "fake-project-a", Capabilities: fullCapabilities()}}})
			recorder := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/"+endpoint.Path(), strings.NewReader(body))
			req.Header = http.Header{
				"Content-Type": {"application/json; charset=utf-8"}, "Accept": {"application/json"},
				"Authorization": {"Bearer caller-secret"}, "Cookie": {"session=caller-secret"}, "X-Api-Key": {"caller-secret"},
				"Openai-Organization": {"caller-org"}, "Openai-Project": {"caller-project"}, "Forwarded": {"for=bad"},
				"X-Forwarded-For": {"bad"}, "X-Forwarded-Host": {"bad"}, "Idempotency-Key": {"no-retry"},
				"Connection": {"X-Secret"}, "X-Secret": {"caller-secret"},
			}
			req.Trailer = http.Header{"Authorization": {"trailer-secret"}, "X-Api-Key": {"trailer-secret"}}
			handler.ServeHTTP(recorder, req)
			if recorder.Code != http.StatusCreated || recorder.Body.String() != response || recorder.Header().Get("X-Request-Id") != "synthetic-request" || recorder.Header().Get("X-Upstream-Hop") != "" || calls.Load() != 1 {
				t.Fatalf("code=%d body=%q headers=%v calls=%d", recorder.Code, recorder.Body, recorder.Header(), calls.Load())
			}
		})
	}
}

func TestModelRoutingIsFixedAndCopied(t *testing.T) {
	var a, b atomic.Int32
	server := func(counter *atomic.Int32, expectedKey string) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			counter.Add(1)
			if r.Header.Get("Authorization") != expectedKey {
				t.Error("credential crossed model routes")
			}
			io.WriteString(w, `{"ok":true}`)
		}))
	}
	first, second := server(&a, "Bearer fake-first"), server(&b, "")
	defer first.Close()
	defer second.Close()
	capabilities := fullCapabilities()
	config := Config{Routes: []Route{
		{Model: "first", BaseURL: first.URL + "/v1", APIKey: "fake-first", Capabilities: capabilities},
		{Model: "second", BaseURL: second.URL + "/v1", Capabilities: capabilities},
	}}
	handler := newHandler(t, config)
	config.Routes[0].APIKey = "must-not-change"
	config.Routes[0].BaseURL = second.URL
	delete(capabilities, provider.ChatCompletions)
	for _, model := range []string{"first", "second", "first"} {
		if got := request(t, handler, "/v1/chat/completions", `{"model":"`+model+`"}`); got.Code != 200 {
			t.Fatalf("%s: %d %s", model, got.Code, got.Body)
		}
	}
	if a.Load() != 2 || b.Load() != 1 {
		t.Fatalf("first=%d second=%d", a.Load(), b.Load())
	}
}

func TestInvalidRequestsNeverReachUpstream(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1) }))
	defer upstream.Close()
	handler := newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL + "/v1", Capabilities: fullCapabilities()}}, MaxBodyBytes: 256})
	for _, tc := range []struct {
		name, method, path, contentType, encoding, body string
		status                                          int
	}{
		{"unknown endpoint", "POST", "/v1/models", "application/json", "", `{"model":"local"}`, 404},
		{"retrieval excluded", "GET", "/v1/responses/resp_fake", "application/json", "", "", 404},
		{"query excluded", "POST", "/v1/responses?upstream=http://evil.invalid", "application/json", "", `{"model":"local"}`, 404},
		{"encoded path excluded", "POST", "/v1/%72esponses", "application/json", "", `{"model":"local"}`, 404},
		{"wrong method", "DELETE", "/v1/responses", "application/json", "", `{"model":"local"}`, 405},
		{"missing type", "POST", "/v1/responses", "", "", `{"model":"local"}`, 415},
		{"gzip excluded", "POST", "/v1/responses", "application/json", "gzip", `{"model":"local"}`, 415},
		{"not object", "POST", "/v1/responses", "application/json", "", `[]`, 400},
		{"invalid json", "POST", "/v1/responses", "application/json", "", `{`, 400},
		{"multiple objects", "POST", "/v1/responses", "application/json", "", `{"model":"local"}{}`, 400},
		{"duplicate model", "POST", "/v1/responses", "application/json", "", `{"model":"local","model":"other"}`, 400},
		{"escaped duplicate model", "POST", "/v1/responses", "application/json", "", `{"model":"local","\u006dodel":"other"}`, 400},
		{"case alias model", "POST", "/v1/responses", "application/json", "", `{"model":"local","Model":"other"}`, 400},
		{"case alias stream", "POST", "/v1/responses", "application/json", "", `{"model":"local","Stream":true}`, 400},
		{"missing model", "POST", "/v1/responses", "application/json", "", `{}`, 400},
		{"model object", "POST", "/v1/responses", "application/json", "", `{"model":{}}`, 400},
		{"model null", "POST", "/v1/responses", "application/json", "", `{"model":null}`, 400},
		{"url as model", "POST", "/v1/responses", "application/json", "", `{"model":"https://evil.invalid/v1"}`, 400},
		{"unknown model", "POST", "/v1/responses", "application/json", "", `{"model":"other"}`, 400},
		{"oversize", "POST", "/v1/responses", "application/json", "", `{"model":"local","input":"` + strings.Repeat("a", 256) + `"}`, 413},
		{"invalid stream", "POST", "/v1/responses", "application/json", "", `{"model":"local","stream":"true"}`, 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
			req.Header.Set("Content-Type", tc.contentType)
			req.Header.Set("Content-Encoding", tc.encoding)
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, req)
			if recorder.Code != tc.status || !json.Valid(recorder.Body.Bytes()) {
				t.Fatalf("status=%d body=%q", recorder.Code, recorder.Body)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("upstream calls=%d", calls.Load())
	}
}

func TestCapabilityGatesAreEndpointSpecific(t *testing.T) {
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		io.WriteString(w, "{}")
	}))
	defer upstream.Close()
	capabilities := provider.Capabilities{
		provider.ChatCompletions: {Endpoint: provider.Supported, Streaming: provider.Unsupported},
		provider.Responses:       {Endpoint: provider.Supported, Streaming: provider.Supported, State: provider.Unsupported},
	}
	handler := newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL, Capabilities: capabilities}}})
	for _, tc := range []struct{ path, fields, want string }{
		{"chat/completions", `,"stream":true`, "unsupported"},
		{"chat/completions", `,"tools":[{"type":"function"}]`, "unknown"},
		{"chat/completions", `,"tool_choice":"required"`, "unknown"},
		{"chat/completions", `,"functions":[{}]`, "unknown"},
		{"chat/completions", `,"response_format":{"type":"json_schema"}`, "unknown"},
		{"chat/completions", `,"store":true`, "unknown"},
		{"responses", `,"text":{"format":{"type":"json_schema"}}`, "unknown"},
		{"responses", `,"previous_response_id":"resp_fake","store":false`, "unsupported"},
		{"responses", `,"conversation":"conv_fake","store":false`, "unsupported"},
		{"responses", `,"background":true,"store":false`, "unsupported"},
		{"responses", `,"store":true`, "unsupported"},
		{"responses", ``, "store:false"},
		{"responses", `,"store":null`, "store:false"},
		{"decisions", ``, "unsupported"},
	} {
		t.Run(tc.path+tc.fields, func(t *testing.T) {
			got := request(t, handler, "/v1/"+tc.path, `{"model":"local"`+tc.fields+`}`)
			if got.Code != 501 || !strings.Contains(got.Body.String(), tc.want) {
				t.Fatalf("status=%d body=%s", got.Code, got.Body)
			}
		})
	}
	if calls.Load() != 0 {
		t.Fatalf("unsupported requests reached upstream: %d", calls.Load())
	}
	for _, tc := range []struct{ path, fields string }{
		{"chat/completions", `,"tools":[],"response_format":{"type":"text"}`},
		{"responses", `,"stream":true,"store":false,"text":{"format":{"type":"text"}}`},
	} {
		got := request(t, handler, "/v1/"+tc.path, `{"model":"local"`+tc.fields+`}`)
		if got.Code != 200 {
			t.Fatalf("stateless request failed: %d %s", got.Code, got.Body)
		}
	}
}

func TestDecisionsRequiresItsOwnExplicitDeclaration(t *testing.T) {
	capabilities := fullCapabilities()
	capabilities[provider.Decisions] = provider.EndpointCapabilities{Endpoint: provider.Supported}
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/custom/decisions" {
			t.Errorf("Decisions was translated: %s", r.URL.Path)
		}
		io.WriteString(w, `{"future_decision_contract":true}`)
	}))
	defer upstream.Close()
	handler := newHandler(t, Config{Routes: []Route{{Model: "explicit", BaseURL: upstream.URL + "/custom", Capabilities: capabilities}}})
	got := request(t, handler, "/v1/decisions", `{"model":"explicit","future_request":true}`)
	if got.Code != 200 || got.Body.String() != `{"future_decision_contract":true}` {
		t.Fatalf("%d %s", got.Code, got.Body)
	}
}

func TestStatusErrorAndRedirectArePassedWithoutRetry(t *testing.T) {
	for _, status := range []int{400, 401, 429, 500, 503, 307} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls, redirected atomic.Int32
			target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { redirected.Add(1) }))
			defer target.Close()
			body := `{"error":{"message":"synthetic upstream failure","unknown_detail":123}}`
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("Retry-After", "9")
				w.Header().Set("Location", target.URL)
				w.WriteHeader(status)
				io.WriteString(w, body)
			}))
			defer upstream.Close()
			handler := newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL, APIKey: "fake-route-key", Capabilities: fullCapabilities()}}})
			got := request(t, handler, "/v1/responses", `{"model":"local","store":false}`)
			if got.Code != status || got.Body.String() != body || got.Header().Get("Retry-After") != "9" || calls.Load() != 1 || redirected.Load() != 0 {
				t.Fatalf("status=%d body=%s calls=%d redirected=%d", got.Code, got.Body, calls.Load(), redirected.Load())
			}
		})
	}
}

func TestSSEBytesFlushAndTrailers(t *testing.T) {
	for _, path := range []string{"chat/completions", "responses"} {
		t.Run(path, func(t *testing.T) {
			first := ": keepalive\r\nevent: future.event\r\ndata: {\"delta\":\"hello\"}\r\n\r\n"
			last := "data: {\"choices\":[],\"usage\":{\"total_tokens\":7},\"future\":true}\n\ndata: [DONE]\n\n"
			release := make(chan struct{})
			var once sync.Once
			defer once.Do(func() { close(release) })
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Trailer", "X-Final-Usage")
				io.WriteString(w, first)
				w.(http.Flusher).Flush()
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
				io.WriteString(w, last)
				w.Header().Set("X-Final-Usage", "synthetic-7")
			}))
			defer upstream.Close()
			server := httptest.NewServer(newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL + "/v1", Capabilities: fullCapabilities()}}}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
			defer cancel()
			req, _ := http.NewRequestWithContext(ctx, "POST", server.URL+"/v1/"+path, strings.NewReader(`{"model":"local","stream":true,"store":false}`))
			req.Header.Set("Content-Type", "application/json")
			response, err := server.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			gotFirst := make([]byte, len(first))
			if _, err := io.ReadFull(response.Body, gotFirst); err != nil || string(gotFirst) != first {
				t.Fatalf("first frame failed before completion: %q %v", gotFirst, err)
			}
			once.Do(func() { close(release) })
			gotLast, err := io.ReadAll(response.Body)
			if err != nil || string(gotLast) != last || response.Trailer.Get("X-Final-Usage") != "synthetic-7" {
				t.Fatalf("last=%q trailers=%v err=%v", gotLast, response.Trailer, err)
			}
		})
	}
}

func TestClientCancellationReachesUpstream(t *testing.T) {
	canceled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(canceled)
	}))
	defer upstream.Close()
	server := httptest.NewServer(newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL, Capabilities: fullCapabilities()}}}))
	defer server.Close()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", server.URL+"/v1/responses", strings.NewReader(`{"model":"local","stream":true}`))
	req.Header.Set("Content-Type", "application/json")
	response, err := server.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if _, err := bufio.NewReader(response.Body).ReadString('\n'); err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("downstream cancellation did not reach upstream")
	}
}

func TestUpstreamTimeoutBeforeHeaders(t *testing.T) {
	canceled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		<-r.Context().Done()
		close(canceled)
	}))
	defer upstream.Close()
	server := httptest.NewServer(newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL, Capabilities: fullCapabilities()}}, RequestTimeout: 50 * time.Millisecond}))
	defer server.Close()
	response, err := server.Client().Post(server.URL+"/v1/responses", "application/json", strings.NewReader(`{"model":"local"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(response.Body)
	if response.StatusCode != 504 || !bytes.Contains(body, []byte("upstream_timeout")) {
		t.Fatalf("status=%d body=%s", response.StatusCode, body)
	}
	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("deadline did not cancel upstream")
	}
}

func TestTruncatedOutputAbortsWithoutJSONSuffixOrRetry(t *testing.T) {
	var calls atomic.Int32
	prefix := "data: {\"delta\":\"partial\"}\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Content-Length", "10000")
		io.WriteString(w, prefix)
		w.(http.Flusher).Flush()
	}))
	defer upstream.Close()
	server := httptest.NewServer(newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL, Capabilities: fullCapabilities()}}}))
	defer server.Close()
	response, err := server.Client().Post(server.URL+"/v1/responses", "application/json", strings.NewReader(`{"model":"local","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	got, err := io.ReadAll(response.Body)
	if err == nil || string(got) != prefix || calls.Load() != 1 {
		t.Fatalf("body=%q err=%v calls=%d", got, err, calls.Load())
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestTransportFailureDoesNotExposeDetails(t *testing.T) {
	var calls int
	handler := newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: "https://fixed.invalid/v1", APIKey: "fake-route-secret", Capabilities: fullCapabilities()}}, Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		return nil, errors.New("fake-route-secret https://private.invalid sensitive-prompt")
	})})
	got := request(t, handler, "/v1/responses", `{"model":"local"}`)
	if got.Code != 502 || calls != 1 || strings.Contains(got.Body.String(), "secret") || strings.Contains(got.Body.String(), "private.invalid") || strings.Contains(got.Body.String(), "sensitive-prompt") {
		t.Fatalf("status=%d body=%s calls=%d", got.Code, got.Body, calls)
	}
}

func TestConfigurationValidation(t *testing.T) {
	for _, config := range []Config{
		{}, {MaxBodyBytes: -1}, {RequestTimeout: -1},
		{Routes: []Route{{Model: "", BaseURL: "https://fixed.invalid/v1"}}},
		{Routes: []Route{{Model: "x", BaseURL: "https://fixed.invalid/v1"}, {Model: "x", BaseURL: "https://other.invalid/v1"}}},
		{Routes: []Route{{Model: "x", BaseURL: "https://fixed.invalid/v1", APIKey: "fake\r\nBad: header"}}},
		{Routes: []Route{{Model: "x", BaseURL: "https://fixed.invalid/v1", Capabilities: provider.Capabilities{"unknown": {Endpoint: provider.Supported}}}}},
		{Routes: []Route{{Model: "x", BaseURL: "https://fixed.invalid/v1", Capabilities: provider.Capabilities{provider.Responses: {Streaming: 100}}}}},
	} {
		if _, err := New(config); err == nil {
			t.Fatal("invalid config was accepted")
		}
	}
	for _, base := range []string{"", "/v1", "file:///tmp", "https://fake-user:fake-password@fixed.invalid/v1", "https://fixed.invalid/v1?token=fake", "https://fixed.invalid/v1#frag", "https://fixed.invalid/v1?", "https://fixed.invalid/%76%31", "https://fixed.invalid/v1/../private"} {
		_, err := New(Config{Routes: []Route{{Model: "x", BaseURL: base}}})
		if err == nil || strings.Contains(err.Error(), "fake-password") || strings.Contains(err.Error(), "token=fake") {
			t.Fatalf("invalid URL accepted or exposed: %v", err)
		}
	}
}

type countedBody struct {
	reads  atomic.Int32
	closed atomic.Bool
}

func (b *countedBody) Read(p []byte) (int, error) {
	b.reads.Add(1)
	for i := range p {
		p[i] = 'x'
	}
	return len(p), nil
}

func (b *countedBody) Close() error {
	b.closed.Store(true)
	return nil
}

type blockedWriter struct {
	header  http.Header
	entered chan int
	release chan struct{}
}

func (w *blockedWriter) Header() http.Header { return w.header }
func (w *blockedWriter) WriteHeader(int)     {}
func (w *blockedWriter) Flush()              {}
func (w *blockedWriter) Write(p []byte) (int, error) {
	w.entered <- len(p)
	<-w.release
	return 0, io.ErrClosedPipe
}

func TestBackpressureDoesNotBufferUpstreamAndWriteFailureStopsReading(t *testing.T) {
	body := &countedBody{}
	handler := newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: "https://fixed.invalid/v1", Capabilities: fullCapabilities()}}, Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: body, ContentLength: -1, Request: r}, nil
	})})
	writer := &blockedWriter{header: make(http.Header), entered: make(chan int, 1), release: make(chan struct{})}
	var once sync.Once
	defer once.Do(func() { close(writer.release) })
	done := make(chan struct{})
	go func() {
		defer close(done)
		req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"model":"local","stream":true}`))
		req.Header.Set("Content-Type", "application/json")
		handler.ServeHTTP(writer, req)
	}()
	select {
	case size := <-writer.entered:
		if size > 32*1024 || body.reads.Load() != 1 {
			t.Fatalf("read-ahead exceeded one copy buffer: size=%d reads=%d", size, body.reads.Load())
		}
	case <-time.After(3 * time.Second):
		t.Fatal("proxy did not begin writing")
	}
	// Since Write cannot finish until release, the synchronous copy must not
	// consume another upstream buffer while the downstream is blocked.
	if body.reads.Load() != 1 {
		t.Fatal("proxy read ahead of the blocked downstream writer")
	}
	once.Do(func() { close(writer.release) })
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("write failure did not stop forwarding")
	}
	if body.reads.Load() != 1 || !body.closed.Load() {
		t.Fatalf("reads=%d body closed=%t", body.reads.Load(), body.closed.Load())
	}
}

func TestBodyLimitExactBoundaryAndUnknownLength(t *testing.T) {
	body := `{"model":"local"}`
	var calls atomic.Int32
	handler := newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: "https://fixed.invalid/v1", Capabilities: fullCapabilities()}}, MaxBodyBytes: int64(len(body)), Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("{}")), Request: r}, nil
	})})
	for _, extra := range []string{"", " "} {
		req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body+extra))
		req.ContentLength = -1
		req.Header.Set("Content-Type", "application/json")
		got := httptest.NewRecorder()
		handler.ServeHTTP(got, req)
		want := 200
		if extra != "" {
			want = 413
		}
		if got.Code != want {
			t.Fatalf("extra=%q status=%d body=%s", extra, got.Code, got.Body)
		}
	}
	if calls.Load() != 1 {
		t.Fatalf("upstream calls=%d", calls.Load())
	}
}

func TestTimeoutAfterOutputAbortsStreamWithoutRerouting(t *testing.T) {
	var calls atomic.Int32
	prefix := "data: partial\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, prefix)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer upstream.Close()
	server := httptest.NewServer(newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: upstream.URL, Capabilities: fullCapabilities()}}, RequestTimeout: 100 * time.Millisecond}))
	defer server.Close()
	response, err := server.Client().Post(server.URL+"/v1/responses", "application/json", strings.NewReader(`{"model":"local","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	got, err := io.ReadAll(response.Body)
	if response.StatusCode != 200 || string(got) != prefix || err == nil || calls.Load() != 1 {
		t.Fatalf("status=%d body=%q err=%v calls=%d", response.StatusCode, got, err, calls.Load())
	}
}
