package appserver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"unicode/utf8"
)

// ServerRequest is an unforgeable, connection-bound, single-use reply handle.
// Do not copy it. Done closes when a reply is attempted, the server resolves it,
// or the connection closes. A reply racing server resolution may already be
// writing; success only means the frame was written, not that the server acted.
type ServerRequest struct {
	client *Client
	id     requestID
	digest [32]byte
	done   chan struct{}
	used   bool
	job    *writeJob
}

func (r *ServerRequest) ID() json.RawMessage   { return bytes.Clone(r.id.raw) }
func (r *ServerRequest) Done() <-chan struct{} { return r.done }
func (r *ServerRequest) invalidateLocked() {
	if !r.used {
		r.used = true
		close(r.done)
	}
}
func (c *Client) removeRequestLocked(r *ServerRequest) {
	if c.requests[r.id.key] == r {
		delete(c.requests, r.id.key)
		c.requestBytes -= len(r.id.raw) + len(r.id.key)
	}
}

// Respond attempts a result once. Nil result sends JSON null. Invalid local
// payloads and already-canceled contexts do not consume the handle.
func (r *ServerRequest) Respond(ctx context.Context, result json.RawMessage) error {
	if result == nil {
		result = json.RawMessage("null")
	}
	if r == nil || r.client == nil {
		return ErrRequestDone
	}
	if err := validPayload(result, r.client.limits.FrameBytes); err != nil {
		return err
	}
	frame, err := encodeFrame(struct {
		ID     json.RawMessage `json:"id"`
		Result json.RawMessage `json:"result"`
	}{r.id.raw, result}, r.client.limits.FrameBytes)
	if err != nil {
		return err
	}
	return r.reply(ctx, frame)
}

// Reject attempts one method-level error, without executing the request.
func (r *ServerRequest) Reject(ctx context.Context, rpcError RPCError) error {
	if r == nil || r.client == nil {
		return ErrRequestDone
	}
	if !utf8.ValidString(rpcError.Message) {
		return ErrProtocol
	}
	if len(rpcError.Message) > r.client.limits.FrameBytes {
		return ErrFrameLimit
	}
	if err := validPayload(rpcError.Data, r.client.limits.FrameBytes); err != nil {
		return err
	}
	frame, err := encodeFrame(struct {
		ID    json.RawMessage `json:"id"`
		Error RPCError        `json:"error"`
	}{r.id.raw, rpcError}, r.client.limits.FrameBytes)
	if err != nil {
		return err
	}
	return r.reply(ctx, frame)
}
func (r *ServerRequest) reply(ctx context.Context, frame []byte) error {
	if ctx == nil {
		return errors.New("appserver: nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	c := r.client
	c.mu.Lock()
	if r.used || c.requests[r.id.key] != r || c.terminal != nil {
		c.mu.Unlock()
		return ErrRequestDone
	}
	r.invalidateLocked()
	j, err := c.enqueueLocked(frame, r)
	r.job = j
	if err != nil {
		c.removeRequestLocked(r)
	}
	c.mu.Unlock()
	if err != nil {
		if errors.Is(err, ErrWriteLimit) {
			c.shutdown(err)
		}
		return err
	}
	return c.awaitWrite(ctx, j)
}
