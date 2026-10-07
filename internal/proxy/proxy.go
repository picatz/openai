// Package proxy provides a bounded, fixed-route HTTP forwarding handler. It does
// not listen on a network address, translate APIs, retry, or choose fallback models.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"github.com/picatz/openai/internal/provider"
)

const (
	DefaultMaxBodyBytes = 1 << 20
	DefaultTimeout      = 2 * time.Minute
)

// Route maps one exact request model to an upstream API base URL, e.g.
// http://127.0.0.1:11434/v1. The model and request bytes are never rewritten.
// Credentials belong to this route; downstream credentials are never forwarded.
// Configuration is trusted application input, never request-controlled input.
type Route struct {
	Model        string
	BaseURL      string
	APIKey       string
	Organization string
	Project      string
	Capabilities provider.Capabilities
}

// Config is copied by New. Zero limits select conservative defaults. Negative
// limits are rejected; there is no unbounded mode. Transport is optional and must
// implement context cancellation; the default has no environment HTTP proxy and
// disables automatic decompression. A supplied transport must not retry requests.
type Config struct {
	Routes         []Route
	MaxBodyBytes   int64
	RequestTimeout time.Duration
	Transport      http.RoundTripper
}

type route struct {
	capabilities provider.Capabilities
	proxy        *httputil.ReverseProxy
}

// Handler accepts only POST /v1/chat/completions, /v1/responses and /v1/decisions.
// No public listener or server CLI is provided. Any future listener must default
// to loopback, authenticate callers before exposing it, and set HTTP server limits.
type Handler struct {
	routes       map[string]route
	maxBodyBytes int64
	timeout      time.Duration
}

// New validates all routes without making network requests. No credentials,
// upstream URLs, request bodies or transport errors are logged or echoed.
func New(config Config) (*Handler, error) {
	if config.MaxBodyBytes < 0 || config.RequestTimeout < 0 {
		return nil, errors.New("proxy limits must not be negative")
	}
	if config.MaxBodyBytes == 0 {
		config.MaxBodyBytes = DefaultMaxBodyBytes
	}
	if config.RequestTimeout == 0 {
		config.RequestTimeout = DefaultTimeout
	}
	if len(config.Routes) == 0 {
		return nil, errors.New("proxy requires at least one explicit model route")
	}
	transport := config.Transport
	if transport == nil {
		transport = &http.Transport{
			Proxy:                 nil,
			DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          32,
			MaxIdleConnsPerHost:   4,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 30 * time.Second,
			ExpectContinueTimeout: time.Second,
			DisableCompression:    true,
		}
	}
	h := &Handler{routes: make(map[string]route), maxBodyBytes: config.MaxBodyBytes, timeout: config.RequestTimeout}
	for i, configured := range config.Routes {
		if strings.TrimSpace(configured.Model) == "" || strings.TrimSpace(configured.Model) != configured.Model {
			return nil, fmt.Errorf("route %d requires a nonempty exact model name", i)
		}
		if _, exists := h.routes[configured.Model]; exists {
			return nil, fmt.Errorf("route %d duplicates a model", i)
		}
		base, err := validateBaseURL(configured.BaseURL)
		if err != nil {
			return nil, fmt.Errorf("route %d: %w", i, err)
		}
		for _, header := range []string{configured.APIKey, configured.Organization, configured.Project} {
			if strings.ContainsAny(header, "\r\n") {
				return nil, fmt.Errorf("route %d has an invalid credential header", i)
			}
		}
		capabilities := make(provider.Capabilities, len(configured.Capabilities))
		for endpoint, capability := range configured.Capabilities {
			if endpoint.Path() == "" {
				return nil, fmt.Errorf("route %d declares an unknown endpoint", i)
			}
			for _, support := range []provider.Support{capability.Endpoint, capability.Streaming, capability.Tools, capability.ToolChoice, capability.StructuredOutput, capability.State} {
				if support > provider.Unsupported {
					return nil, fmt.Errorf("route %d has an invalid capability", i)
				}
			}
			capabilities[endpoint] = capability
		}
		upstream := &httputil.ReverseProxy{
			Transport: transport,
			// Immediate writes keep SSE frames, comments and final usage frames
			// byte-for-byte. ReverseProxy streams with bounded copy buffers.
			FlushInterval: -1,
			ErrorLog:      log.New(io.Discard, "", 0),
			Rewrite: func(request *httputil.ProxyRequest) {
				endpoint := endpointForPath(request.In.URL.Path)
				target := *base
				target.Path = strings.TrimSuffix(base.Path, "/") + "/" + endpoint.Path()
				request.Out.URL = &target
				request.Out.Host = target.Host
				request.Out.Header = make(http.Header)
				request.Out.Header.Set("Content-Type", "application/json")
				if accept := request.In.Header.Get("Accept"); accept != "" {
					request.Out.Header.Set("Accept", accept)
				}
				if configured.APIKey != "" {
					request.Out.Header.Set("Authorization", "Bearer "+configured.APIKey)
				}
				if configured.Organization != "" {
					request.Out.Header.Set("OpenAI-Organization", configured.Organization)
				}
				if configured.Project != "" {
					request.Out.Header.Set("OpenAI-Project", configured.Project)
				}
				// A caller's trailers or idempotency key must not introduce
				// credentials or enable net/http's transparent POST retries.
				request.Out.Trailer = nil
				request.Out.GetBody = nil
			},
			ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
				if errors.Is(err, context.DeadlineExceeded) || errors.Is(r.Context().Err(), context.DeadlineExceeded) {
					writeError(w, http.StatusGatewayTimeout, "upstream_timeout", "upstream request timed out")
					return
				}
				writeError(w, http.StatusBadGateway, "upstream_error", "upstream request failed")
			},
		}
		h.routes[configured.Model] = route{capabilities: capabilities, proxy: upstream}
	}
	return h, nil
}

func validateBaseURL(raw string) (*url.URL, error) {
	base, err := url.Parse(raw)
	if err != nil || base.Host == "" || base.Hostname() == "" || (base.Scheme != "https" && base.Scheme != "http") || base.User != nil || base.RawQuery != "" || base.ForceQuery || base.Fragment != "" || base.RawPath != "" || base.Opaque != "" {
		return nil, errors.New("base URL must be an absolute HTTP(S) API URL without credentials, query or fragment")
	}
	for _, part := range strings.Split(base.Path, "/") {
		if part == "." || part == ".." {
			return nil, errors.New("base URL path must not contain dot segments")
		}
	}
	return base, nil
}

func endpointForPath(path string) provider.Endpoint {
	switch path {
	case "/v1/chat/completions":
		return provider.ChatCompletions
	case "/v1/responses":
		return provider.Responses
	case "/v1/decisions":
		return provider.Decisions
	default:
		return ""
	}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	endpoint := endpointForPath(r.URL.Path)
	if endpoint == "" || r.URL.RawPath != "" || r.URL.RawQuery != "" || r.URL.ForceQuery {
		writeError(w, http.StatusNotFound, "unknown_endpoint", "endpoint is not configured")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "only POST is supported")
		return
	}
	contentType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || contentType != "application/json" || (r.Header.Get("Content-Encoding") != "" && r.Header.Get("Content-Encoding") != "identity") {
		writeError(w, http.StatusUnsupportedMediaType, "unsupported_media_type", "an uncompressed application/json body is required")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), h.timeout)
	defer cancel()
	controller := http.NewResponseController(w)
	deadline, _ := ctx.Deadline()
	// Allow one second of bounded write grace to send a timeout error.
	// Real net/http connections support these deadlines. Embedders wrapping
	// ResponseWriter must expose Unwrap or configure equivalent server deadlines.
	_ = controller.SetReadDeadline(deadline)
	_ = controller.SetWriteDeadline(deadline.Add(time.Second))
	defer controller.SetReadDeadline(time.Time{})
	defer controller.SetWriteDeadline(time.Time{})
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, h.maxBodyBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeError(w, http.StatusRequestEntityTooLarge, "body_too_large", "request body exceeds the configured limit")
		} else {
			writeError(w, http.StatusBadRequest, "invalid_body", "could not read request body")
		}
		return
	}
	object, err := requestObject(body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_json", "request must be one JSON object with unambiguous, case-sensitive top-level parameters")
		return
	}
	var model string
	if json.Unmarshal(object["model"], &model) != nil || model == "" {
		writeError(w, http.StatusBadRequest, "invalid_model", "a nonempty model string is required")
		return
	}
	configured, ok := h.routes[model]
	if !ok {
		writeError(w, http.StatusBadRequest, "unknown_model", "model has no configured route")
		return
	}
	if err := checkCapabilities(endpoint, configured.capabilities.For(endpoint), object); err != nil {
		if errors.Is(err, errInvalidParameter) {
			writeError(w, http.StatusBadRequest, "invalid_parameter", err.Error())
		} else {
			writeError(w, http.StatusNotImplemented, "unsupported_capability", err.Error())
		}
		return
	}
	forward := r.Clone(ctx)
	forward.Body = io.NopCloser(bytes.NewReader(body))
	forward.ContentLength = int64(len(body))
	forward.TransferEncoding = nil
	configured.proxy.ServeHTTP(w, forward)
}

// requestObject reads only the top-level envelope, retaining the exact original
// bytes for forwarding. Duplicate keys are rejected to prevent parser-dependent
// model routing. Unknown fields and nested raw items are not reconstructed.
func requestObject(body []byte) (map[string]json.RawMessage, error) {
	return envelopeObject(body, true, "model", "stream", "tools", "tool_choice", "functions", "function_call", "response_format", "text", "store", "background", "previous_response_id", "conversation")
}

// envelopeObject uses exact keys for capability decisions, rejecting aliases
// that a case-insensitive upstream could interpret differently. EqualFold also
// covers Unicode aliases such as long s; ToLower does not. At nested boundaries,
// only relevant keys are reserved, leaving provider-specific fields opaque.
func envelopeObject(body []byte, rejectAllDuplicates bool, reserved ...string) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, errors.New("not an object")
	}
	object := make(map[string]json.RawMessage)
	for decoder.More() {
		token, err = decoder.Token()
		if err != nil {
			return nil, err
		}
		key, ok := token.(string)
		if !ok {
			return nil, errors.New("invalid key")
		}
		isReserved := false
		for _, canonical := range reserved {
			if strings.EqualFold(key, canonical) {
				if key != canonical {
					return nil, errors.New("ambiguous parameter case")
				}
				isReserved = true
				break
			}
		}
		if _, exists := object[key]; exists && (rejectAllDuplicates || isReserved) {
			return nil, errors.New("duplicate key")
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, err
		}
		object[key] = value
	}
	if _, err := decoder.Token(); err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != io.EOF {
		return nil, errors.New("trailing content")
	}
	return object, nil
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"type": "proxy_error", "code": code, "message": message}})
}
