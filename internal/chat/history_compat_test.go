package chat_test

import (
	"testing"

	"github.com/picatz/openai/internal/chat"
	"github.com/picatz/openai/internal/chat/storage"
)

// This is the on-disk JSON shape written before the SDK v3 migration. Reading it
// must not require a cache reset or change role/content/usage values.
func TestExistingHistoryJSON(t *testing.T) {
	fixture := []byte(`{"model":"gpt-4o","req":{"role":"user","content":"old question"},"req_tokens":3,"resp":{"role":"assistant","content":"old answer"},"resp_tokens":4}`)
	codec := storage.JSONCodec[string, chat.ReqRespPair]{}
	pair, err := codec.DecodeValue(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if pair.Model != "gpt-4o" || pair.Req.Content != "old question" || pair.Resp.Content != "old answer" || pair.ReqTokens != 3 || pair.RespTokens != 4 {
		t.Fatalf("lost historical data: %+v", pair)
	}
	encoded, err := codec.EncodeValue(pair)
	if err != nil {
		t.Fatal(err)
	}
	again, err := codec.DecodeValue(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if again.Req.Role != "user" || again.Resp.Role != "assistant" || again.Resp.Content != pair.Resp.Content {
		t.Fatalf("roundtrip: %+v", again)
	}
}
