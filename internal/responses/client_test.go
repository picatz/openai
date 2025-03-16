package responses_test

import (
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/picatz/openai/internal/responses"
	"github.com/shoenig/test/must"
)

// retryTransport wraps an [net/http.RoundTripper] and retries requests on failures.
type retryTransport struct {
	base       http.RoundTripper
	maxRetries int
	backoff    time.Duration
}

// RoundTrip implements the [net/http.RoundTripper] interface that
// executes a single HTTP transaction, but with potentially multiple
// attempts if the request fails or is retried due to certain conditions.
// It retries the request on network errors and certain HTTP status codes
// (5xx and 429).
func (rt *retryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var (
		resp *http.Response
		err  error
	)

	for attempt := 0; attempt <= rt.maxRetries; attempt++ {
		r := req.Clone(req.Context())

		resp, err = rt.base.RoundTrip(r)
		switch {
		case err != nil:
			// Network error, retry
		case resp.StatusCode == http.StatusTooManyRequests:
			// Too Many Requests, retry
		case resp.StatusCode >= 500 && resp.StatusCode < 600:
			// Server error, retry
		default:
			// Success or client error, do not retry
			return resp, nil
		}

		if resp != nil {
			resp.Body.Close()
		}

		time.Sleep(rt.backoff)
	}

	return resp, err
}

// testHTTPClient creates a new [net/http.Client] with a retry transport and a timeout.
// It is used for testing purposes to ensure that requests are retried
func testHTTPClient(t *testing.T) *http.Client {
	t.Helper()

	client := &http.Client{
		Transport: &retryTransport{
			base:       http.DefaultTransport,
			maxRetries: 3,
			backoff:    2 * time.Second,
		},
		Timeout: 30 * time.Second,
	}

	return client
}

// testClient creates a new [responses.Client] for testing purposes with
// a retry transport and a timeout. It is used to test the OpenAI Responses
// API client. The client is configured to use a retry transport with a
// maximum of 3 retries and a backoff duration of 2 seconds, and a timeout
// of 30 seconds for HTTP requests.
//
// It retrieves the API key from the environment variable, and if not set,
// it skips the test.
func testClient(t *testing.T) *responses.Client {
	t.Helper()

	if os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("Skipping test because OPENAI_API_KEY is not set")
	}

	client := responses.NewClient(os.Getenv("OPENAI_API_KEY"), testHTTPClient(t))
	if client == nil {
		t.Fatal("Failed to create client")
	}
	return client
}

// TestSimpleInputText tests the creation of a response using a simple text input.
func TestSimpleInputText(t *testing.T) {
	ctx := t.Context()

	client := testClient(t)

	request := responses.Request{
		Model: "gpt-4o",
		Input: responses.Text("Hey there!"),
	}

	resp, err := client.CreateResponse(ctx, request)
	must.NoError(t, err)
	must.StrHasPrefix(t, "gpt-4o", resp.Model) // it won't be the exact model name you requested
	must.Eq(t, 1, len(resp.Output))
	must.Eq(t, 1, len(resp.Output[0].Content))

	t.Logf("Response: %q", resp.Output[0].Content[0].Text)
}

// TestSimpleReasoningInputText tests the creation of a response using a simple text input
// with reasoning enabled using a high effort level.
func TestSimpleReasoningInputText(t *testing.T) {
	ctx := t.Context()

	client := testClient(t)

	request := responses.Request{
		Model: "o3-mini",
		Input: responses.Text("How much wood would a woodchuck chuck?"),
		Reasoning: &responses.RequestResoning{
			Effort: "high",
		},
	}

	resp, err := client.CreateResponse(ctx, request)
	must.NoError(t, err)
	must.StrHasPrefix(t, "o3-mini", resp.Model) // it won't be the exact model name you requested

	for _, output := range resp.Output {
		if output.Type == "message" {
			t.Logf("Output: %q", output.Content[0].Text)
		}
	}
}

// TestSimpleFunctionCallInputText tests the creation of a response using a simple text input
// with a function call. It includes a tool choice and a function definition.
func TestSimpleFunctionCallInputText(t *testing.T) {
	ctx := t.Context()

	client := testClient(t)

	request := responses.Request{
		Model:      "gpt-4o",
		Input:      responses.Text("What is the weather like in Ann Arbor today?"),
		ToolChoice: responses.RequestToolChoiceAuto,
		Tools: responses.RequestTools{
			responses.RequestToolFunction{
				Name:        "get_current_weather",
				Description: "Get the current weather in a given location",
				Parameters: &responses.JSONSchema{
					Type: "object",
					Properties: map[string]*responses.JSONSchema{
						"location": {
							Type:        "string",
							Description: "The city and state, e.g. San Francisco, CA",
						},
						"unit": {
							Type: "string",
							Enum: []string{"celsius", "fahrenheit"},
						},
					},
				},
			},
		},
	}

	resp, err := client.CreateResponse(ctx, request)
	must.NoError(t, err)

	for _, output := range resp.Output {
		t.Logf("%#+v", output)
	}
}

// TestSimpleWebSearchPreview tests the creation of a response using a simple text input
// with a web search preview tool. It includes a tool choice for web search preview.
func TestSimpleWebSearchPreview(t *testing.T) {
	ctx := t.Context()

	client := testClient(t)

	request := responses.Request{
		Model:      "gpt-4o",
		Input:      responses.Text("What is the weather like in Ann Arbor today?"),
		ToolChoice: responses.RequestToolChoiceAuto,
		Tools: responses.RequestTools{
			responses.RequestToolWebSearchPreview{},
		},
	}

	resp, err := client.CreateResponse(ctx, request)
	must.NoError(t, err)

	for _, output := range resp.Output {
		t.Logf("%#+v", output.Content)
	}
}

// TestSimpleChatFlow_manual_history tests a simple chat flow with manual history management.
func TestSimpleChatFlow_manual_history(t *testing.T) {
	ctx := t.Context()

	client := testClient(t)

	history := responses.InputItemList{}

	history = append(history, responses.Message{
		Role:    responses.RoleUser,
		Content: responses.Text("Hello!"),
	})

	resp, err := client.CreateResponse(ctx, responses.Request{
		Model: "gpt-4o",
		Input: history,
	})
	must.NoError(t, err)
	must.StrHasPrefix(t, "gpt-4o", resp.Model) // it won't be the exact model name you requested

	for _, output := range resp.Output {
		must.Eq(t, "message", output.Type)
		history = append(history, responses.Message{
			Role:    responses.Role(output.Role),
			Content: responses.Text(output.Content[0].Text),
		})
	}

	history = append(history, responses.Message{
		Role:    responses.RoleUser,
		Content: responses.Text("what is a good name for a cat?"),
	})

	resp, err = client.CreateResponse(ctx, responses.Request{
		Model: "gpt-4o",
		Input: history,
	})
	must.NoError(t, err)

	for _, output := range resp.Output {
		must.Eq(t, "message", output.Type)

		history = append(history, responses.Message{
			Role:    responses.Role(output.Role),
			Content: responses.Text(output.Content[0].Text),
		})
	}

	history = append(history, responses.Message{
		Role:    responses.RoleUser,
		Content: responses.Text("pick one of the names you suggested, provide ONLY the name and nothing else in the response"),
	})

	resp, err = client.CreateResponse(ctx, responses.Request{
		Model: "gpt-4o",
		Input: history,
	})
	must.NoError(t, err)

	for _, output := range resp.Output {
		must.Eq(t, "message", output.Type)

		must.True(t, slices.ContainsFunc(history, func(item responses.InputItem) bool {
			itemMessage, ok := item.(responses.Message)
			must.True(t, ok)

			itemMessageContentText, ok := itemMessage.Content.(responses.Text)
			must.True(t, ok)

			return strings.Contains(string(itemMessageContentText), output.Content[0].Text)
		}))

		history = append(history, responses.Message{
			Role:    responses.Role(output.Role),
			Content: responses.Text(output.Content[0].Text),
		})
	}

	for i, item := range history {
		itemMessage, ok := item.(responses.Message)
		must.True(t, ok)

		itemMessageContentText, ok := itemMessage.Content.(responses.Text)
		must.True(t, ok)

		t.Logf("Step %d: %q", i, itemMessageContentText)
	}
}

// TestSimpleChatFlow_automatic_history tests a simple chat flow with automatic history management.
func TestSimpleChatFlow_automatic_history(t *testing.T) {
	ctx := t.Context()

	client := testClient(t)

	firstResp, err := client.CreateResponse(ctx, responses.Request{
		Model: "gpt-4o",
		Input: responses.Text("what is a good name for a cat?"),
	})
	must.NoError(t, err)

	var potentialCatNames string

	for _, output := range firstResp.Output {
		must.Eq(t, "message", output.Type)
		potentialCatNames += output.Content[0].Text
	}

	secondResp, err := client.CreateResponse(ctx, responses.Request{
		Model:              "gpt-4o",
		Input:              responses.Text("pick one of the names you suggested, provide ONLY the name and nothing else in the response"),
		PreviousResponseID: firstResp.ID,
	})
	must.NoError(t, err)

	for _, output := range secondResp.Output {
		must.Eq(t, "message", output.Type)
		must.True(t, strings.Contains(potentialCatNames, output.Content[0].Text))
	}
}
