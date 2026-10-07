package liveclient

import (
	"context"
	"github.com/coder/websocket"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestWebSocketHandshakeContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/live/sessions" || r.URL.RawQuery != "" || r.Header.Get("Authorization") != "Bearer synthetic" {
			t.Error("unexpected handshake", r.URL)
		}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Error(err)
			return
		}
		defer c.CloseNow()
		kind, data, err := c.Read(r.Context())
		if err != nil {
			t.Error(err)
			return
		}
		c.Write(r.Context(), kind, data)
	}))
	defer server.Close()
	socket, err := Dial(t.Context(), ConnectionConfig{BaseURL: server.URL + "/v1/", APIKey: "synthetic"})
	if err != nil {
		t.Fatal(err)
	}
	defer socket.Close()
	if err := socket.Write(t.Context(), []byte(`{"type":"synthetic"}`)); err != nil {
		t.Fatal(err)
	}
	data, err := socket.Read(t.Context())
	if err != nil || string(data) != `{"type":"synthetic"}` {
		t.Fatalf("data=%s err=%v", data, err)
	}
}
func TestWebSocketNeverFollowsRedirect(t *testing.T) {
	var leaked atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { leaked.Add(1) }))
	defer target.Close()
	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, target.URL, 307) }))
	defer redirect.Close()
	_, err := Dial(context.Background(), ConnectionConfig{BaseURL: redirect.URL + "/v1", APIKey: "synthetic"})
	if err == nil || leaked.Load() != 0 {
		t.Fatalf("err=%v leaked=%d", err, leaked.Load())
	}
}
func TestLiveEndpointValidation(t *testing.T) {
	for _, base := range []string{"http://example.com/v1/", "https://user:pass@example.com/v1/", "https://example.com/v1/?key=secret", "file:///tmp", "https://example.com/#fragment"} {
		if _, err := Endpoint(base); err == nil {
			t.Errorf("accepted %q", base)
		}
	}
	endpoint, err := Endpoint("")
	if err != nil || endpoint != "wss://api.openai.com/v1/live/sessions" || strings.Contains(endpoint, "?") {
		t.Fatalf("endpoint=%q err=%v", endpoint, err)
	}
}
