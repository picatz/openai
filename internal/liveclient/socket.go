package liveclient

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/picatz/websocket"
)

// ConnectionConfig configures the Live WebSocket connection. The handshake
// is performed by the WebSocket package directly, so it never follows
// redirects and never consults HTTP proxy environment variables.
type ConnectionConfig struct {
	BaseURL, APIKey, Organization, Project string
}

func Endpoint(base string) (string, error) {
	if base == "" {
		base = "https://api.openai.com/v1/"
	}
	u, err := url.Parse(base)
	if err != nil || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", fmt.Errorf("invalid Live base URL")
	}
	switch u.Scheme {
	case "https":
		u.Scheme = "wss"
	case "http":
		host := u.Hostname()
		ip := net.ParseIP(host)
		if host != "localhost" && (ip == nil || !ip.IsLoopback()) {
			return "", fmt.Errorf("unencrypted Live transport is allowed only on loopback")
		}
		u.Scheme = "ws"
	default:
		return "", fmt.Errorf("Live base URL must use HTTP(S)")
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/live/sessions"
	return u.String(), nil
}
func Dial(ctx context.Context, cfg ConnectionConfig) (Socket, error) {
	endpoint, err := Endpoint(cfg.BaseURL)
	if err != nil {
		return nil, err
	}
	header := http.Header{}
	if cfg.APIKey != "" {
		header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	if cfg.Organization != "" {
		header.Set("OpenAI-Organization", cfg.Organization)
	}
	if cfg.Project != "" {
		header.Set("OpenAI-Project", cfg.Project)
	}
	dialCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(dialCtx, endpoint, websocket.WithHeader(header), websocket.WithMaxMessageSize(1<<20))
	if err != nil {
		return nil, fmt.Errorf("connect Live: %w", err)
	}
	return &socket{conn}, nil
}

type socket struct{ conn *websocket.Conn }

// expired is a deadline in the past, used to interrupt blocked I/O when a
// context ends. The WebSocket package does not take contexts after Dial, and
// timeouts are terminal for the connection, which callers close afterwards.
var expired = time.Unix(1, 0)

func (s *socket) Read(ctx context.Context) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	stop := context.AfterFunc(ctx, func() { s.conn.SetDeadline(expired) })
	kind, data, err := s.conn.ReadMessage()
	stop()
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, err
	}
	if kind != websocket.TextMessage {
		return nil, fmt.Errorf("Live events must use JSON text frames")
	}
	return data, nil
}
func (s *socket) Write(ctx context.Context, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	deadline, _ := ctx.Deadline() // zero when unset, which clears the deadline
	if err := s.conn.SetWriteDeadline(deadline); err != nil {
		return err
	}
	stop := context.AfterFunc(ctx, func() { s.conn.SetWriteDeadline(expired) })
	err := s.conn.WriteMessage(websocket.TextMessage, data)
	stop()
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
	}
	return err
}
func (s *socket) Close() error { return s.conn.Close() }
