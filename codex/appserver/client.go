package appserver

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"io"
	"strconv"
	"sync"
	"unicode/utf8"
)

// Event is an owned notification or server request in wire order. Request is
// non-nil only for explicitly registered methods. Rejected marks a request
// queued for rejection with -32601; it carries no reply authority. Delivery
// may still fail with the connection. Raw includes extensions.
type Event struct {
	Method      string
	Params, Raw json.RawMessage
	Request     *ServerRequest
	Rejected    bool
}
type queuedEvent struct {
	event Event
	bytes int
}
type response struct {
	result json.RawMessage
	err    error
}
type pendingCall struct {
	response  chan response
	job       *writeJob
	abandoned bool
}
type writeJob struct {
	frame   []byte
	done    chan error
	state   int
	request *ServerRequest
}

// Client is one connection. Do not copy it. Call, Close, Wait, and reply handles
// are concurrency-safe. Use one Next consumer to retain application ordering.
type Client struct {
	transport              io.ReadWriteCloser
	limits                 Limits
	registered             map[string]bool
	initialization         InitializeResult
	readReady              chan struct{}
	mu                     sync.Mutex
	terminal               error
	stopping, done         chan struct{}
	eventReady, writeReady chan struct{}
	events                 []queuedEvent
	eventBytes             int
	writes                 []*writeJob
	writeCount, writeBytes int
	pending                map[string]*pendingCall
	nextID                 uint64
	abandoned              int
	requests               map[string]*ServerRequest
	requestBytes           int
	workers                sync.WaitGroup
}

func newClient(t io.ReadWriteCloser, o Options) *Client {
	c := &Client{readReady: make(chan struct{}), transport: t, limits: o.Limits, registered: map[string]bool{}, stopping: make(chan struct{}), done: make(chan struct{}), eventReady: make(chan struct{}, 1), writeReady: make(chan struct{}, 1), pending: map[string]*pendingCall{}, requests: map[string]*ServerRequest{}}
	for _, method := range o.ServerRequestMethods {
		c.registered[method] = true
	}
	return c
}
func (c *Client) start(ctx context.Context) {
	c.workers.Add(3)
	go func() {
		defer c.workers.Done()
		select {
		case <-c.readReady:
			c.readLoop()
		case <-c.stopping:
		}
	}()
	go func() { defer c.workers.Done(); c.writeLoop() }()
	go func() {
		defer c.workers.Done()
		select {
		case <-ctx.Done():
			c.shutdown(ctx.Err())
		case <-c.stopping:
		}
	}()
	go func() { c.workers.Wait(); close(c.done) }()
}
func signal(ch chan struct{}) {
	select {
	case ch <- struct{}{}:
	default:
	}
}
func (c *Client) shutdown(err error) {
	c.mu.Lock()
	if c.terminal != nil {
		c.mu.Unlock()
		return
	}
	c.terminal = err
	close(c.stopping)
	for _, r := range c.requests {
		r.invalidateLocked()
	}
	c.requests = nil
	c.requestBytes = 0
	for _, w := range c.writes {
		c.writeCount--
		c.writeBytes -= len(w.frame)
		w.frame = nil
		w.state = 2
		w.done <- err
	}
	c.writes = nil
	c.pending = nil
	c.abandoned = 0
	c.mu.Unlock()
	// Close is outside the state lock and must interrupt both I/O workers.
	_ = c.transport.Close()
}

// Done closes only after all I/O and cancellation workers have stopped.
func (c *Client) Done() <-chan struct{} { return c.done }

// Wait joins workers without requiring Next to drain queued events. It returns
// the first terminal cause, including ErrClosed for intentional closure.
func (c *Client) Wait() error { <-c.done; c.mu.Lock(); defer c.mu.Unlock(); return c.terminal }

// Close is idempotent, unblocks I/O, and joins workers. Inspect Wait for the
// original terminal cause, which Close never overwrites.
func (c *Client) Close() error { c.shutdown(ErrClosed); <-c.done; return nil }

// Call waits for one method result. Nil params means omitted; JSON null is a
// distinct valid payload. A canceled queued write is removed. Once writing has
// begun cancellation only abandons the local wait, and the remote outcome is
// unknown. Such calls occupy bounded correlation slots until answered or closed.
func (c *Client) Call(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error) {
	if method == "initialize" || method == "initialized" {
		return nil, errors.New("appserver: initialization is managed by Connect")
	}
	return c.call(ctx, method, params)
}
func (c *Client) call(ctx context.Context, method string, params json.RawMessage) (json.RawMessage, error) {
	if ctx == nil {
		return nil, errors.New("appserver: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if method == "" || !utf8.ValidString(method) {
		return nil, ErrProtocol
	}
	if len(method) > c.limits.FrameBytes {
		return nil, ErrFrameLimit
	}
	if err := validPayload(params, c.limits.FrameBytes); err != nil {
		return nil, err
	}
	c.mu.Lock()
	if c.terminal != nil {
		err := c.terminal
		c.mu.Unlock()
		return nil, err
	}
	if len(c.pending) >= c.limits.PendingCalls {
		c.mu.Unlock()
		return nil, ErrPendingLimit
	}
	if c.nextID == ^uint64(0) {
		c.mu.Unlock()
		c.shutdown(ErrPendingLimit)
		return nil, ErrPendingLimit
	}
	c.nextID++
	id := "go:" + strconv.FormatUint(c.nextID, 10)
	p := &pendingCall{response: make(chan response, 1)}
	c.pending["s:"+id] = p
	c.mu.Unlock()
	frame, err := encodeFrame(struct {
		ID     string          `json:"id"`
		Method string          `json:"method"`
		Params json.RawMessage `json:"params,omitempty"`
	}{id, method, params}, c.limits.FrameBytes)
	if err == nil {
		c.mu.Lock()
		if err = ctx.Err(); err == nil {
			p.job, err = c.enqueueLocked(frame, nil)
			if err == nil && method == "initialize" {
				close(c.readReady)
			}
		}
		c.mu.Unlock()
	}
	if err != nil {
		c.mu.Lock()
		delete(c.pending, "s:"+id)
		c.mu.Unlock()
		if errors.Is(err, ErrWriteLimit) {
			c.shutdown(err)
		}
		return nil, err
	}
	select {
	case r := <-p.response:
		if err := c.awaitWrite(ctx, p.job); err != nil {
			return nil, err
		}
		return r.result, r.err
	case <-ctx.Done():
		c.mu.Lock()
		current, ok := c.pending["s:"+id]
		overflow := false
		if ok && current == p {
			if c.cancelWriteLocked(p.job, ctx.Err()) {
				delete(c.pending, "s:"+id)
			} else {
				p.abandoned = true
				c.abandoned++
				overflow = c.abandoned > c.limits.AbandonedCalls
			}
		}
		c.mu.Unlock()
		if overflow {
			c.shutdown(ErrAbandonedLimit)
		}
		return nil, ctx.Err()
	case <-c.stopping:
		select {
		case r := <-p.response:
			if err := c.awaitWrite(ctx, p.job); err != nil {
				return nil, err
			}
			return r.result, r.err
		default:
		}
		c.mu.Lock()
		err := c.terminal
		c.mu.Unlock()
		return nil, err
	}
}

// Next drains accepted events before reporting terminal failure. Its context
// cancels only this wait. Event payload slices are independent owned data.
func (c *Client) Next(ctx context.Context) (Event, error) {
	if ctx == nil {
		return Event{}, errors.New("appserver: nil context")
	}
	for {
		if err := ctx.Err(); err != nil {
			return Event{}, err
		}
		c.mu.Lock()
		if len(c.events) > 0 {
			q := c.events[0]
			c.events[0] = queuedEvent{}
			c.events = c.events[1:]
			c.eventBytes -= q.bytes
			c.mu.Unlock()
			return q.event, nil
		}
		if c.terminal != nil {
			err := c.terminal
			c.mu.Unlock()
			return Event{}, err
		}
		c.mu.Unlock()
		select {
		case <-ctx.Done():
			return Event{}, ctx.Err()
		case <-c.stopping:
		case <-c.eventReady:
		}
	}
}
func (c *Client) queueEvent(e Event) error {
	size := len(e.Raw) + len(e.Params) + len(e.Method)
	if e.Request != nil {
		size += len(e.Request.id.raw) + len(e.Request.id.key)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.terminal != nil {
		return c.terminal
	}
	if len(c.events) >= c.limits.EventCount || size > c.limits.EventBytes-c.eventBytes {
		return ErrEventLimit
	}
	c.events = append(c.events, queuedEvent{e, size})
	c.eventBytes += size
	signal(c.eventReady)
	return nil
}
func (c *Client) enqueueLocked(frame []byte, r *ServerRequest) (*writeJob, error) {
	if c.terminal != nil {
		return nil, c.terminal
	}
	if c.writeCount >= c.limits.WriteCount || len(frame) > c.limits.WriteBytes-c.writeBytes {
		return nil, ErrWriteLimit
	}
	j := &writeJob{frame: frame, done: make(chan error, 1), request: r}
	c.writes = append(c.writes, j)
	c.writeCount++
	c.writeBytes += len(frame)
	signal(c.writeReady)
	return j, nil
}
func (c *Client) cancelWriteLocked(j *writeJob, err error) bool {
	if j == nil || j.state != 0 {
		return false
	}
	for i, w := range c.writes {
		if w == j {
			copy(c.writes[i:], c.writes[i+1:])
			c.writes[len(c.writes)-1] = nil
			c.writes = c.writes[:len(c.writes)-1]
			c.writeCount--
			c.writeBytes -= len(j.frame)
			j.frame = nil
			j.state = 2
			j.done <- err
			return true
		}
	}
	return false
}
func (c *Client) sendInitialized(ctx context.Context, frame []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.mu.Lock()
	j, err := c.enqueueLocked(frame, nil)
	c.mu.Unlock()
	if err != nil {
		if errors.Is(err, ErrWriteLimit) {
			c.shutdown(err)
		}
		return err
	}
	return c.awaitWrite(ctx, j)
}
func (c *Client) awaitWrite(ctx context.Context, j *writeJob) error {
	// Once both response and write are complete, a later EOF must not erase
	// the accepted result. Check again on terminal/cancellation selection too.
	select {
	case err := <-j.done:
		return err
	default:
	}
	select {
	case err := <-j.done:
		return err
	case <-ctx.Done():
		select {
		case err := <-j.done:
			return err
		default:
		}
		c.mu.Lock()
		c.cancelWriteLocked(j, ctx.Err())
		if j.request != nil {
			c.removeRequestLocked(j.request)
		}
		c.mu.Unlock()
		return ctx.Err()
	case <-c.stopping:
		select {
		case err := <-j.done:
			return err
		default:
		}
		c.mu.Lock()
		err := c.terminal
		c.mu.Unlock()
		return err
	}
}

func (c *Client) writeLoop() {
	for {
		c.mu.Lock()
		if c.terminal != nil {
			c.mu.Unlock()
			return
		}
		if len(c.writes) == 0 {
			c.mu.Unlock()
			select {
			case <-c.stopping:
				return
			case <-c.writeReady:
			}
			continue
		}
		j := c.writes[0]
		c.writes[0] = nil
		c.writes = c.writes[1:]
		j.state = 1
		if j.request != nil {
			c.removeRequestLocked(j.request)
		}
		frame := j.frame
		c.mu.Unlock()
		n, err := c.transport.Write(frame)
		if n != len(frame) && err == nil {
			err = io.ErrShortWrite
		}
		if n < 0 || n > len(frame) {
			err = io.ErrShortWrite
		}

		c.mu.Lock()
		c.writeCount--
		c.writeBytes -= len(frame)
		j.frame = nil
		j.state = 2
		j.done <- err
		c.mu.Unlock()
		if err != nil {
			c.shutdown(err)
			return
		}
	}
}
func (c *Client) readLoop() {
	r := newFrameReader(c.transport, c.limits.FrameBytes)
	for {
		raw, err := r.next()
		if err == nil {
			var e envelope
			e, err = parseEnvelope(raw)
			if err == nil {
				err = c.dispatch(e)
			}
		}
		if err != nil {
			c.shutdown(err)
			return
		}
	}
}
func (c *Client) dispatch(e envelope) error {
	if e.method == "" {
		c.mu.Lock()
		defer c.mu.Unlock()
		p, ok := c.pending[e.id.key]
		if !ok || p.job == nil || p.job.state == 0 {
			return ErrProtocol
		}
		delete(c.pending, e.id.key)
		if p.abandoned {
			c.abandoned--
			return nil
		}
		var err error
		if e.rpcError != nil {
			err = e.rpcError
		}
		p.response <- response{e.result, err}
		return nil
	}
	event := Event{Method: e.method, Params: e.params, Raw: e.raw}
	if e.hasID {
		if !c.registered[e.method] {
			frame, err := encodeFrame(struct {
				ID    json.RawMessage `json:"id"`
				Error RPCError        `json:"error"`
			}{e.id.raw, RPCError{Code: -32601, Message: "Unsupported server request"}}, c.limits.FrameBytes)
			if err != nil {
				return err
			}
			c.mu.Lock()
			_, err = c.enqueueLocked(frame, nil)
			c.mu.Unlock()
			if err != nil {
				return err
			}
			event.Rejected = true
		} else {
			digest := sha256.Sum256(e.raw)
			c.mu.Lock()
			if c.terminal != nil {
				err := c.terminal
				c.mu.Unlock()
				return err
			}
			if old, ok := c.requests[e.id.key]; ok {
				same := old.digest == digest
				c.mu.Unlock()
				if !same {
					return ErrProtocol
				}
				return nil
			}
			size := len(e.id.key) + len(e.id.raw)
			if len(c.requests) >= c.limits.PendingRequests || size > c.limits.PendingRequestBytes-c.requestBytes {
				c.mu.Unlock()
				return ErrRequestLimit
			}
			req := &ServerRequest{client: c, id: e.id, digest: digest, done: make(chan struct{})}
			c.requests[e.id.key] = req
			c.requestBytes += size
			event.Request = req
			c.mu.Unlock()
		}
	} else if e.method == "serverRequest/resolved" {
		var params map[string]json.RawMessage
		if json.Unmarshal(e.params, &params) != nil || params == nil {
			return ErrProtocol
		}
		id, err := parseID(params["requestId"])
		if err != nil {
			return err
		}
		var thread string
		raw, ok := params["threadId"]
		if !ok || len(raw) == 0 || raw[0] != '"' || json.Unmarshal(raw, &thread) != nil {
			return ErrProtocol
		}
		c.mu.Lock()
		if req := c.requests[id.key]; req != nil {
			req.invalidateLocked()
			if req.job != nil {
				c.cancelWriteLocked(req.job, ErrRequestDone)
			}
			c.removeRequestLocked(req)
		}
		c.mu.Unlock()
	}
	return c.queueEvent(event)
}
