package appserver

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"time"
	"unicode/utf8"
)

// ClientInfo identifies this client to Codex. Name and Version are required.
type ClientInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title,omitempty"`
	Version string `json:"version"`
}

// InitializeResult is server metadata, not a capability or version negotiation.
// CodexHome is informational and is never opened by this package.
type InitializeResult struct {
	UserAgent      string `json:"userAgent"`
	CodexHome      string `json:"codexHome"`
	PlatformFamily string `json:"platformFamily"`
	PlatformOS     string `json:"platformOs"`
}

// Limits bounds retained transport state. Zero fields use the documented
// defaults. Negative values are invalid. Bytes count encoded JSON payloads;
// ordinary Go object and allocator overhead is additional.
type Limits struct {
	FrameBytes          int // Default 16 MiB, excluding CRLF.
	EventCount          int // Default 256.
	EventBytes          int // Default 32 MiB; raw frame, params, method, and handle ID/key bytes.
	WriteCount          int // Default 128, including an active write.
	WriteBytes          int // Default 32 MiB, including active frame and LF.
	PendingCalls        int // Default 128, including abandoned calls.
	AbandonedCalls      int // Default 128.
	PendingRequests     int // Default 128.
	PendingRequestBytes int // Default 32 MiB; retained encoded IDs and keys.
}

// Options is copied by Connect. No experimental or optional capability is
// advertised unless explicitly selected here. Registered methods do not cause
// capability advertisement, run a handler, or grant execution permission.
type Options struct {
	ClientInfo                ClientInfo
	ExperimentalAPI           bool
	OptOutNotificationMethods []string
	ServerRequestMethods      []string
	InitializeTimeout         time.Duration // Zero means 10 seconds; negative is invalid.
	Limits                    Limits
}

func (o *Options) defaults() error {
	if o.ClientInfo.Name == "" || o.ClientInfo.Version == "" || !utf8.ValidString(o.ClientInfo.Name) || !utf8.ValidString(o.ClientInfo.Title) || !utf8.ValidString(o.ClientInfo.Version) {
		return errors.New("appserver: valid client name and version required")
	}
	if o.InitializeTimeout < 0 {
		return errors.New("appserver: negative initialization timeout")
	}
	if o.InitializeTimeout == 0 {
		o.InitializeTimeout = 10 * time.Second
	}
	vals := []*int{&o.Limits.FrameBytes, &o.Limits.EventCount, &o.Limits.EventBytes, &o.Limits.WriteCount, &o.Limits.WriteBytes, &o.Limits.PendingCalls, &o.Limits.AbandonedCalls, &o.Limits.PendingRequests, &o.Limits.PendingRequestBytes}
	defs := []int{16 << 20, 256, 32 << 20, 128, 32 << 20, 128, 128, 128, 32 << 20}
	for i, v := range vals {
		if *v < 0 || *v > math.MaxInt-2 {
			return errors.New("appserver: invalid limit")
		}
		if *v == 0 {
			*v = defs[i]
		}
	}
	for _, methods := range [][]string{o.ServerRequestMethods, o.OptOutNotificationMethods} {
		for _, s := range methods {
			if s == "" || !utf8.ValidString(s) || len(s) > o.Limits.FrameBytes {
				return errors.New("appserver: invalid method")
			}
		}
	}
	return nil
}

// Connect owns transport immediately, including on failure. ctx governs the
// connection lifetime; it should not be a short-lived per-call context.
// Initialize sends initialize then initialized, and returns only after both
// writes and the initialize response. No normal Call can precede this sequence.
func Connect(ctx context.Context, transport io.ReadWriteCloser, options Options) (*Client, error) {
	if transport == nil {
		return nil, errors.New("appserver: nil transport")
	}
	if ctx == nil {
		_ = transport.Close()
		return nil, errors.New("appserver: nil context")
	}
	if err := options.defaults(); err != nil {
		_ = transport.Close()
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		_ = transport.Close()
		return nil, err
	}
	params := struct {
		ClientInfo   ClientInfo `json:"clientInfo"`
		Capabilities any        `json:"capabilities,omitempty"`
	}{ClientInfo: options.ClientInfo}
	if options.ExperimentalAPI || len(options.OptOutNotificationMethods) > 0 {
		params.Capabilities = struct {
			ExperimentalAPI bool     `json:"experimentalApi,omitempty"`
			OptOut          []string `json:"optOutNotificationMethods,omitempty"`
		}{options.ExperimentalAPI, options.OptOutNotificationMethods}
	}
	raw, err := json.Marshal(params)
	if err != nil || len(raw) > options.Limits.FrameBytes {
		_ = transport.Close()
		return nil, ErrFrameLimit
	}
	c := newClient(transport, options)
	c.start(ctx)
	initCtx, cancel := context.WithTimeout(ctx, options.InitializeTimeout)
	defer cancel()
	result, err := c.call(initCtx, "initialize", raw)
	if err == nil {
		var f map[string]json.RawMessage
		if json.Unmarshal(result, &f) != nil || f == nil {
			err = ErrProtocol
		} else {
			fields := []struct {
				name   string
				target *string
			}{
				{"userAgent", &c.initialization.UserAgent},
				{"codexHome", &c.initialization.CodexHome},
				{"platformFamily", &c.initialization.PlatformFamily},
				{"platformOs", &c.initialization.PlatformOS},
			}
			for _, field := range fields {
				v, ok := f[field.name]
				if !ok || len(v) == 0 || v[0] != '"' || json.Unmarshal(v, field.target) != nil {
					err = ErrProtocol
					break
				}
			}
		}
	}
	if err == nil {
		var frame []byte
		frame, err = encodeFrame(struct {
			Method string `json:"method"`
		}{"initialized"}, c.limits.FrameBytes)
		if err == nil {
			err = c.sendInitialized(initCtx, frame)
		}
	}
	if err != nil {
		c.shutdown(err)
		_ = c.Wait()
		return nil, err
	}
	return c, nil
}
func (c *Client) Initialization() InitializeResult { return c.initialization }
