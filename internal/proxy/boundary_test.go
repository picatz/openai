package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"unicode"

	"github.com/picatz/openai/internal/provider"
)

func TestReservedTopLevelParametersRejectEveryUnicodeFoldAlias(t *testing.T) {
	reserved := []string{"model", "stream", "tools", "tool_choice", "functions", "function_call", "response_format", "text", "store", "background", "previous_response_id", "conversation"}
	for _, key := range reserved {
		for offset, original := range key {
			for folded := unicode.SimpleFold(original); folded != original; folded = unicode.SimpleFold(folded) {
				alias := key[:offset] + string(folded) + key[offset+len(string(original)):]
				encoded, _ := json.Marshal(alias)
				body := []byte(fmt.Sprintf(`{"%s":false,%s:true}`, key, encoded))
				if _, err := requestObject(body); err == nil {
					t.Errorf("accepted ambiguous alias %q for %q", alias, key)
				}
			}
		}
	}
}

func TestCapabilityBoundaryAmbiguitiesNeverReachEitherAPI(t *testing.T) {
	calls := 0
	handler := newHandler(t, Config{Routes: []Route{{
		Model: "local", BaseURL: "https://fixed.invalid/v1", Capabilities: provider.Capabilities{
			provider.ChatCompletions: {Endpoint: provider.Supported, StructuredOutput: provider.Unsupported, State: provider.Unsupported},
			provider.Responses:       {Endpoint: provider.Supported, StructuredOutput: provider.Unsupported, State: provider.Unsupported},
		},
	}}, Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		calls++
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("{}")), Request: r}, nil
	})})

	for _, path := range []string{"chat/completions", "responses"} {
		for _, fields := range []string{
			`"ſtore":true,"ſtream":true`,
			`"\u017ftore":true,"\u017ftream":true`,
			`"ſTORE":true`,
			`"Store":true`,
			`"Stream":true`,
			`"TOOLſ":[{}]`,
			`"bacKground":true`,
			`"previous_response_iD":"synthetic"`,
			`"Model":"unconfigured"`,
			`"model":"unconfigured"`,
			`"\u006dodel":"unconfigured"`,
		} {
			t.Run(path+"/"+fields, func(t *testing.T) {
				got := request(t, handler, "/v1/"+path, `{"model":"local","store":false,`+fields+`}`)
				if got.Code != 400 {
					t.Fatalf("ambiguous envelope accepted: status=%d body=%s", got.Code, got.Body)
				}
			})
		}
		for _, format := range []string{
			`{"type":"json_schema","TYPE":"text"}`,
			`{"TYPE":"text","type":"json_schema"}`,
			`{"type":"text","TYPE":"json_schema"}`,
			`{"Type":"text"}`,
			`{"type":"json_schema","type":"text"}`,
			`{"type":"text","type":"json_schema"}`,
			`{"type":"json_schema","\u0074ype":"text"}`,
		} {
			t.Run(path+"/format/"+format, func(t *testing.T) {
				field := `"response_format":` + format
				if path == "responses" {
					field = `"text":{"format":` + format + `}`
				}
				got := request(t, handler, "/v1/"+path, `{"model":"local","store":false,`+field+`}`)
				if got.Code != 400 {
					t.Fatalf("ambiguous format accepted: status=%d body=%s", got.Code, got.Body)
				}
			})
		}
	}
	for _, text := range []string{
		`{"format":{"type":"json_schema"},"FORMAT":{"type":"text"}}`,
		`{"FORMAT":{"type":"text"},"format":{"type":"json_schema"}}`,
		`{"Format":{"type":"text"}}`,
		`{"format":{"type":"json_schema"},"format":{"type":"text"}}`,
		`{"format":{"type":"text"},"format":{"type":"json_schema"}}`,
		`{"format":{"type":"json_schema"},"\u0066ormat":{"type":"text"}}`,
	} {
		t.Run("responses/text/"+text, func(t *testing.T) {
			got := request(t, handler, "/v1/responses", `{"model":"local","store":false,"text":`+text+`}`)
			if got.Code != 400 {
				t.Fatalf("ambiguous text format accepted: status=%d body=%s", got.Code, got.Body)
			}
		})
	}
	if calls != 0 {
		t.Fatalf("ambiguous requests reached upstream: %d", calls)
	}
}

func TestStructuredFormatsUseExactTypeAndPreserveUnknownPayloads(t *testing.T) {
	for _, endpoint := range []provider.Endpoint{provider.ChatCompletions, provider.Responses} {
		for _, support := range []provider.Support{provider.Unknown, provider.Unsupported, provider.Supported} {
			for _, kind := range []string{"text", "json_schema", "json_object", "future_format"} {
				t.Run(fmt.Sprintf("%s/%s/%s", endpoint, support, kind), func(t *testing.T) {
					// Unknown nested fields (even duplicate or case-varying ones)
					// remain opaque. Only the relevant envelope keys are parsed.
					format := fmt.Sprintf(`{ "type":%q,"future":1,"future":2,"Future":3,"schema":{"type":"object","TYPE":"string","type":"array"}}`, kind)
					field := `"response_format":` + format
					if endpoint == provider.Responses {
						field = `"text":{ "format":` + format + `,"future":1,"future":2,"Future":3}`
					}
					body := " {\n\"model\":\"local\",\"store\":false,\"future\": {\"TYPE\":\"unknown\",\"type\":\"opaque\"},\"Future\":1," + field + " }\n"
					calls := 0
					handler := newHandler(t, Config{Routes: []Route{{Model: "local", BaseURL: "https://fixed.invalid/v1", Capabilities: provider.Capabilities{
						endpoint: {Endpoint: provider.Supported, StructuredOutput: support, State: provider.Unsupported},
					}}}, Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
						calls++
						got, _ := io.ReadAll(r.Body)
						if string(got) != body {
							t.Error("unknown fields or original request bytes changed")
						}
						return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("{}")), Request: r}, nil
					})})
					got := request(t, handler, "/v1/"+endpoint.Path(), body)
					if kind != "text" && support != provider.Supported {
						if got.Code != 501 || calls != 0 || !strings.Contains(got.Body.String(), support.String()) {
							t.Fatalf("structured format bypassed gate: status=%d calls=%d body=%s", got.Code, calls, got.Body)
						}
					} else if got.Code != 200 || calls != 1 {
						t.Fatalf("unambiguous request failed: status=%d calls=%d body=%s", got.Code, calls, got.Body)
					}
				})
			}
		}
	}
}
