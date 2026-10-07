package liveclient

import (
	"context"
	"fmt"
	"github.com/coder/websocket"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type ConnectionConfig struct {
	BaseURL, APIKey, Organization, Project string
	HTTPClient                             *http.Client
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
	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	safeClient := *client
	safeClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
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
	conn, _, err := websocket.Dial(dialCtx, endpoint, &websocket.DialOptions{HTTPClient: &safeClient, HTTPHeader: header})
	if err != nil {
		return nil, fmt.Errorf("connect Live: %w", err)
	}
	conn.SetReadLimit(1 << 20)
	return &socket{conn}, nil
}

type socket struct{ conn *websocket.Conn }

func (s *socket) Read(ctx context.Context) ([]byte, error) {
	kind, data, err := s.conn.Read(ctx)
	if err != nil {
		return nil, err
	}
	if kind != websocket.MessageText {
		return nil, fmt.Errorf("Live events must use JSON text frames")
	}
	return data, nil
}
func (s *socket) Write(ctx context.Context, data []byte) error {
	return s.conn.Write(ctx, websocket.MessageText, data)
}
func (s *socket) Close() error { return s.conn.CloseNow() }
