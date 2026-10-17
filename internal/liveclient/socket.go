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

// ConnectionConfig configures the Live WebSocket connection. Dial honors
// HTTP_PROXY, HTTPS_PROXY and NO_PROXY, tunnelling through the proxy so the
// WebSocket handshake and TLS stay end to end, and never follows redirects.
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
	dialer := &websocket.Dialer{
		Header:         header,
		MaxMessageSize: 1 << 20,
		Proxy:          http.ProxyFromEnvironment,
	}
	conn, _, err := dialer.Dial(dialCtx, endpoint)
	if err != nil {
		return nil, fmt.Errorf("connect Live: %w", err)
	}
	return &socket{conn}, nil
}

type socket struct{ conn *websocket.Conn }

// Read and Write abort the connection if ctx ends mid-call; Run closes the
// socket on cancellation anyway, so this matches its lifecycle.
func (s *socket) Read(ctx context.Context) ([]byte, error) {
	kind, data, err := s.conn.ReadMessageContext(ctx)
	if err != nil {
		return nil, err
	}
	if kind != websocket.TextMessage {
		return nil, fmt.Errorf("Live events must use JSON text frames")
	}
	return data, nil
}
func (s *socket) Write(ctx context.Context, data []byte) error {
	return s.conn.WriteMessageContext(ctx, websocket.TextMessage, data)
}
func (s *socket) Close() error { return s.conn.Close() }
