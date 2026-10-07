// Package httpguard bounds SDK response bodies before decoding them.
package httpguard

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/openai/openai-go/v3/option"
	"io"
	"net/http"
	"sync"
)

func LimitResponse(successLimit, errorLimit int64) option.RequestOption {
	return option.WithMiddleware(func(r *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		resp, err := next(r)
		if err != nil || resp == nil || resp.Body == nil {
			return resp, err
		}
		limit := successLimit
		if resp.StatusCode >= 400 {
			limit = errorLimit
		}
		resp.Body = &boundedBody{ReadCloser: resp.Body, remaining: limit, limit: limit}
		return resp, nil
	})
}

type boundedBody struct {
	io.ReadCloser
	remaining, limit int64
}

func (b *boundedBody) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if b.remaining == 0 {
		var extra [1]byte
		n, err := b.ReadCloser.Read(extra[:])
		if n > 0 {
			return 0, fmt.Errorf("response body exceeds %d bytes", b.limit)
		}
		return 0, err
	}
	if int64(len(p)) > b.remaining {
		p = p[:b.remaining]
	}
	n, err := b.ReadCloser.Read(p)
	b.remaining -= int64(n)
	return n, err
}

// LimitJSONResponse validates the entire bounded success body before the SDK's
// permissive decoder sees it. Validation happens during Body.Read, so a malformed
// successful body is not retried as a transport failure.
func LimitJSONResponse(successLimit, errorLimit int64) option.RequestOption {
	return option.WithMiddleware(func(r *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		resp, err := next(r)
		if err != nil || resp == nil || resp.Body == nil {
			return resp, err
		}
		limit := successLimit
		if resp.StatusCode >= 400 {
			limit = errorLimit
		}
		body := &boundedBody{ReadCloser: resp.Body, remaining: limit, limit: limit}
		resp.Body = body
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			resp.Body = &jsonBody{source: body}
		}
		return resp, nil
	})
}

type jsonBody struct {
	source io.ReadCloser
	once   sync.Once
	reader *bytes.Reader
	err    error
}

func (b *jsonBody) Read(p []byte) (int, error) {
	b.once.Do(func() {
		data, err := io.ReadAll(b.source)
		if err != nil {
			b.err = err
			return
		}
		if !json.Valid(data) {
			b.err = fmt.Errorf("response is not one complete JSON document")
			return
		}
		b.reader = bytes.NewReader(data)
	})
	if b.err != nil {
		return 0, b.err
	}
	return b.reader.Read(p)
}
func (b *jsonBody) Close() error { return b.source.Close() }
