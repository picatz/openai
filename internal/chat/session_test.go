package chat_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/cockroachdb/pebble"
	"github.com/cockroachdb/pebble/vfs"
	"github.com/openai/openai-go"
	"github.com/picatz/openai/internal/chat"
	"github.com/picatz/openai/internal/chat/storage"
	pebbleStorage "github.com/picatz/openai/internal/chat/storage/pebble"
	"github.com/shoenig/test/must"
)

func TestChatSession(t *testing.T) {
	var (
		client = openai.NewClient()
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

	chatSession, restore, err := chat.NewSession(t.Context(), client, "gpt-4o", input, output, memBackend)
	must.NoError(t, err)
	t.Cleanup(restore)
	must.NotNil(t, chatSession)

	typeInTerminal("hello")

	done, err := chatSession.RunOnce(t.Context())
	must.NoError(t, err)
	must.False(t, done)

	t.Log(output.String())
}
