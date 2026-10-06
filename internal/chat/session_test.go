package chat_test

import (
	"bytes"
	"github.com/openai/openai-go/v3/option"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/openai/openai-go/v3"
	"github.com/picatz/openai/internal/chat"
	"github.com/picatz/openai/internal/chat/storage"
	pebbleStorage "github.com/picatz/openai/internal/chat/storage/pebble"
	"github.com/shoenig/test/must"
)

func TestChatSession(t *testing.T) {
	var (
		client = openai.NewClient(option.WithAPIKey("test-key"), option.WithBaseURL("https://synthetic.invalid/v1/"), option.WithHTTPClient(&http.Client{Transport: fakeChatTransport{t}}), option.WithMaxRetries(0))
		input  = bytes.NewBuffer(nil)
		output = bytes.NewBuffer(nil)
	)

	typeInTerminal := func(s string) {
		for line := range strings.Lines(s) {
			n, err := input.WriteString(line + "\r\n")
			must.NoError(t, err)
			must.Eq(t, len(line)+2, n)
		}
	}

	pebbleOptions := &pebble.Options{
		FS: vfs.NewMem(),
	}

	codec := &storage.JSONCodec[string, chat.ReqRespPair]{}

	memBackend, err := pebbleStorage.NewBackend("", pebbleOptions, codec)
	must.NoError(t, err)
	must.NotNil(t, memBackend)
	t.Cleanup(func() {
		must.NoError(t, memBackend.Close(t.Context()))
	})

	chatSession, restore, err := chat.NewSession(t.Context(), &client, openai.ChatModelGPT4o, input, output, memBackend)
	must.NoError(t, err)
	t.Cleanup(restore)
	must.NotNil(t, chatSession)

	typeInTerminal("hello")

	done, err := chatSession.RunOnce(t.Context())
	must.NoError(t, err)
	must.False(t, done)

	t.Log(output.String())
}

func TestChunkString(t *testing.T) {
	var (
		input     = "This is a test string that is longer than the chunk size."
		chunkSize = int64(8)
	)

	chunks, err := chat.ChunkString(input, chunkSize)
	must.NoError(t, err)

	expectedChunks := []string{
		"This is a test string",
		"that is longer than",
		"the chunk size.",
	}

	must.Eq(t, expectedChunks, chunks)
}

type fakeChatTransport struct{ t *testing.T }

func (f fakeChatTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	f.t.Helper()
	if r.URL.Path != "/v1/chat/completions" {
		f.t.Fatalf("unexpected request: %s", r.URL.Path)
	}
	body := `{"id":"chatcmpl_test","object":"chat.completion","model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"Hello from a fake transport"},"finish_reason":"stop"}],"usage":{"total_tokens":8}}`
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(body)), Request: r}, nil
}
