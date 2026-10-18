package appserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

func options() Options {
	return Options{ClientInfo: ClientInfo{Name: "synthetic", Version: "test"}, InitializeTimeout: 2 * time.Second}
}

type peer struct {
	net.Conn
	r *frameReader
}

func (p *peer) read(t *testing.T) envelope {
	t.Helper()
	b, e := p.r.next()
	if e != nil {
		t.Fatal(e)
	}
	v, e := parseEnvelope(b)
	if e != nil {
		t.Fatal(e)
	}
	return v
}
func (p *peer) write(t *testing.T, s string) {
	t.Helper()
	if _, e := io.WriteString(p.Conn, s+"\n"); e != nil {
		t.Fatal(e)
	}
}
func handshake(p net.Conn) error {
	r := newFrameReader(p, 1<<20)
	b, err := r.next()
	if err != nil {
		return err
	}
	e, err := parseEnvelope(b)
	if err != nil || e.method != "initialize" {
		return fmt.Errorf("initialize: %v", err)
	}
	if strings.Contains(string(b), "jsonrpc") || strings.Contains(string(b), "capabilities") {
		return errors.New("unexpected default capability or jsonrpc")
	}
	_, err = fmt.Fprintf(p, `{"id":%s,"result":{"userAgent":"synthetic","codexHome":"/never/open","platformFamily":"unix","platformOs":"linux"}}`+"\n", e.id.raw)
	if err != nil {
		return err
	}
	b, err = r.next()
	if err != nil {
		return err
	}
	if string(b) != `{"method":"initialized"}` {
		return fmt.Errorf("unexpected initialized: %q", b)
	}
	return nil
}
func connectPeer(t *testing.T, o Options) (*Client, *peer) {
	t.Helper()
	a, b := net.Pipe()
	_ = b.SetDeadline(time.Now().Add(5 * time.Second))
	ch := make(chan error, 1)
	go func() { ch <- handshake(b) }()
	c, err := Connect(t.Context(), a, o)
	if err != nil {
		_ = b.Close()
		t.Fatal(err)
	}
	if err := <-ch; err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close(); _ = b.Close() })
	return c, &peer{b, newFrameReader(b, 1<<20)}
}
func await[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case x := <-ch:
		return x
	case <-time.After(3 * time.Second):
		t.Fatal("worker did not terminate")
		var z T
		return z
	}
}
func waitCause(t *testing.T, c *Client, cause error) {
	t.Helper()
	ch := make(chan error, 1)
	go func() { ch <- c.Wait() }()
	if e := await(t, ch); !errors.Is(e, cause) {
		t.Fatalf("Wait = %v, want %v", e, cause)
	}
}

func TestHandshakeAndConcurrentCalls(t *testing.T) {
	c, p := connectPeer(t, options())
	if c.Initialization().CodexHome != "/never/open" {
		t.Fatal(c.Initialization())
	}
	if _, err := c.Call(t.Context(), "initialize", nil); err == nil {
		t.Fatal("duplicate initialization allowed")
	}
	const n = 12
	out := make(chan response, n)
	for i := 0; i < n; i++ {
		go func(i int) {
			b, e := c.Call(t.Context(), "echo", json.RawMessage(fmt.Sprint(i)))
			out <- response{b, e}
		}(i)
	}
	calls := make([]envelope, n)
	for i := range calls {
		calls[i] = p.read(t)
	}
	p.write(t, `{"method":"future","params":{"x":1},"emittedAtMs":123}`)
	for i := n - 1; i >= 0; i-- {
		p.write(t, fmt.Sprintf(`{"id":%s,"result":%s}`, calls[i].id.raw, calls[i].params))
	}
	seen := map[string]bool{}
	for range n {
		r := await(t, out)
		if r.err != nil {
			t.Fatal(r.err)
		}
		seen[string(r.result)] = true
	}
	if len(seen) != n {
		t.Fatal("correlation lost", seen)
	}
	e, err := c.Next(t.Context())
	if err != nil || e.Method != "future" || !strings.Contains(string(e.Raw), "emittedAtMs") {
		t.Fatal(e, err)
	}
	result := make(chan response, 1)
	go func() { b, e := c.Call(t.Context(), "fail", nil); result <- response{b, e} }()
	call := p.read(t)
	p.write(t, fmt.Sprintf(`{"id":%s,"error":{"code":-32000,"message":"private detail","data":{"future":1}}}`, call.id.raw))
	r := await(t, result)
	var rpc *RPCError
	if !errors.As(r.err, &rpc) || rpc.Code != -32000 || string(rpc.Data) != `{"future":1}` || strings.Contains(r.err.Error(), "private") {
		t.Fatal(r.err)
	}
	result = make(chan response, 1)
	go func() { b, e := c.Call(t.Context(), "null", json.RawMessage("null")); result <- response{b, e} }()
	call = p.read(t)
	p.write(t, fmt.Sprintf(`{"id":%s,"result":null}`, call.id.raw))
	r = await(t, result)
	if r.err != nil || string(r.result) != "null" {
		t.Fatal(r)
	}
}
func TestDefaultRequestRejection(t *testing.T) {
	c, p := connectPeer(t, options())
	methods := []string{"item/commandExecution/requestApproval", "item/fileChange/requestApproval", "item/tool/requestUserInput", "mcpServer/elicitation/request", "item/tool/call", "account/chatgptAuthTokens/refresh", "attestation/generate", "future/request"}
	for i, m := range methods {
		p.write(t, fmt.Sprintf(`{"id":%d,"method":%q,"params":{}}`, i, m))
		reply := p.read(t)
		if reply.rpcError == nil || reply.rpcError.Code != -32601 || reply.id.key != fmt.Sprintf("i:%d", i) {
			t.Fatal(reply)
		}
		e, err := c.Next(t.Context())
		if err != nil || !e.Rejected || e.Request != nil || e.Method != m {
			t.Fatal(e, err)
		}
	}
}
func TestRequestsReplayResolutionAndSingleUse(t *testing.T) {
	o := options()
	o.ServerRequestMethods = []string{"synthetic/request"}
	c, p := connectPeer(t, o)
	req := `{"id":9223372036854775807,"method":"synthetic/request","params":{"x":1}}`
	p.write(t, req)
	p.write(t, req)
	p.write(t, `{"method":"marker"}`)
	e, err := c.Next(t.Context())
	if err != nil || e.Request == nil {
		t.Fatal(e, err)
	}
	r := e.Request
	copyID := r.ID()
	copyID[0] = '0'
	if string(r.ID()) != "9223372036854775807" {
		t.Fatal("ID aliases")
	}
	marker, err := c.Next(t.Context())
	if err != nil || marker.Method != "marker" {
		t.Fatal("replay was re-emitted", marker, err)
	}
	replyDone := make(chan error, 1)
	go func() { replyDone <- r.Respond(t.Context(), json.RawMessage(`{"decision":"synthetic"}`)) }()
	wire := p.read(t)
	if wire.id.key != "i:9223372036854775807" || string(wire.result) != `{"decision":"synthetic"}` {
		t.Fatal(wire)
	}
	if err := await(t, replyDone); err != nil {
		t.Fatal(err)
	}
	select {
	case <-r.Done():
	default:
		t.Fatal("reply did not invalidate handle")
	}
	if err := r.Respond(t.Context(), nil); !errors.Is(err, ErrRequestDone) {
		t.Fatal(err)
	}
	// A later reuse creates a fresh handle; no cached approval is replayed.
	p.write(t, req)
	e, err = c.Next(t.Context())
	if err != nil || e.Request == r || e.Request == nil {
		t.Fatal(e, err)
	}
	p.write(t, `{"method":"serverRequest/resolved","params":{"threadId":"synthetic","requestId":9223372036854775807}}`)
	resolved, err := c.Next(t.Context())
	if err != nil || resolved.Method != "serverRequest/resolved" {
		t.Fatal(resolved, err)
	}
	select {
	case <-e.Request.Done():
	default:
		t.Fatal("resolution did not invalidate")
	}
	if err := e.Request.Reject(t.Context(), RPCError{Code: -1, Message: "no"}); !errors.Is(err, ErrRequestDone) {
		t.Fatal(err)
	}
	p.write(t, `{"method":"serverRequest/resolved","params":{"threadId":"synthetic","requestId":"unknown"}}`)
	if _, err := c.Next(t.Context()); err != nil {
		t.Fatal(err)
	}
	p.write(t, req)
	fresh, _ := c.Next(t.Context())
	_ = c.Close()
	if err := fresh.Request.Respond(t.Context(), nil); !errors.Is(err, ErrRequestDone) {
		t.Fatal(err)
	}
}
func TestConflictingReplayTerminates(t *testing.T) {
	o := options()
	o.ServerRequestMethods = []string{"x"}
	c, p := connectPeer(t, o)
	p.write(t, `{"id":"same","method":"x","params":1}`)
	p.write(t, `{"id":"same","method":"x","params":2}`)
	waitCause(t, c, ErrProtocol)
	e, err := c.Next(t.Context())
	if err != nil || e.Request == nil {
		t.Fatal(e, err)
	}
	select {
	case <-e.Request.Done():
	default:
		t.Fatal("stale request")
	}
}
func TestOverflowDrainsAcceptedEventsAndUnblocksCalls(t *testing.T) {
	for _, kind := range []string{"count", "bytes"} {
		t.Run(kind, func(t *testing.T) {
			o := options()
			if kind == "count" {
				o.Limits.EventCount = 2
			} else {
				o.Limits.EventBytes = 2 * (len(`{"method":"tick"}`) + len("tick"))
			}
			c, p := connectPeer(t, o)
			out := make(chan error, 1)
			go func() { _, e := c.Call(t.Context(), "wait", nil); out <- e }()
			_ = p.read(t)
			for range 3 {
				p.write(t, `{"method":"tick"}`)
			}
			waitCause(t, c, ErrEventLimit)
			if e := await(t, out); !errors.Is(e, ErrEventLimit) {
				t.Fatal(e)
			}
			for range 2 {
				e, err := c.Next(t.Context())
				if err != nil || e.Method != "tick" {
					t.Fatal(e, err)
				}
			}
			if _, err := c.Next(t.Context()); !errors.Is(err, ErrEventLimit) {
				t.Fatal(err)
			}
		})
	}
}
func TestPendingAndAbandonedCalls(t *testing.T) {
	o := options()
	o.Limits.PendingCalls = 1
	c, p := connectPeer(t, o)
	ctx, cancel := context.WithCancel(t.Context())
	out := make(chan error, 1)
	go func() { _, e := c.Call(ctx, "wait", nil); out <- e }()
	call := p.read(t)
	if _, e := c.Call(t.Context(), "over", nil); !errors.Is(e, ErrPendingLimit) {
		t.Fatal(e)
	}
	cancel()
	if e := await(t, out); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if _, e := c.Call(t.Context(), "still-full", nil); !errors.Is(e, ErrPendingLimit) {
		t.Fatal(e)
	}
	p.write(t, fmt.Sprintf(`{"id":%s,"result":null}`, call.id.raw))
	p.write(t, `{"method":"barrier"}`)
	_, _ = c.Next(t.Context())
	ctx, cancel = context.WithCancel(t.Context())
	cancel()
	if _, e := c.Call(ctx, "never", nil); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	c.mu.Lock()
	n := len(c.pending)
	c.mu.Unlock()
	if n != 0 {
		t.Fatal("abandoned response did not release slot")
	}
}
func TestAbandonedLimit(t *testing.T) {
	o := options()
	o.Limits.AbandonedCalls = 1
	c, p := connectPeer(t, o)
	for range 2 {
		ctx, cancel := context.WithCancel(t.Context())
		ch := make(chan error, 1)
		go func() { _, e := c.Call(ctx, "wait", nil); ch <- e }()
		_ = p.read(t)
		cancel()
		if e := await(t, ch); !errors.Is(e, context.Canceled) {
			t.Fatal(e)
		}
	}
	waitCause(t, c, ErrAbandonedLimit)
}
func TestUnexpectedAndDuplicateResponses(t *testing.T) {
	for _, s := range []string{`{"id":"unknown","result":null}`, `{"id":null,"result":null}`, `{"method":"x","method":"y"}`, `{"method":"serverRequest/resolved","params":{}}`} {
		t.Run(s, func(t *testing.T) { c, p := connectPeer(t, options()); p.write(t, s); waitCause(t, c, ErrProtocol) })
	}
	c, p := connectPeer(t, options())
	ch := make(chan error, 1)
	go func() { _, e := c.Call(t.Context(), "one", nil); ch <- e }()
	req := p.read(t)
	wire := fmt.Sprintf(`{"id":%s,"result":null}`, req.id.raw)
	p.write(t, wire)
	if e := await(t, ch); e != nil {
		t.Fatal(e)
	}
	p.write(t, wire)
	waitCause(t, c, ErrProtocol)
}
func TestConcurrentCloseJoinsWorkers(t *testing.T) {
	c, p := connectPeer(t, options())
	waiting := make(chan error, 1)
	go func() { _, e := c.Next(t.Context()); waiting <- e }()
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() { defer wg.Done(); _ = c.Close(); _ = c.Wait() }()
	}
	finished := make(chan bool, 1)
	go func() { wg.Wait(); finished <- true }()
	await(t, finished)
	if e := await(t, waiting); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	select {
	case <-c.Done():
	default:
		t.Fatal("workers remain")
	}
	_ = p.Close()
}
func TestConnectionCancellationAndEOF(t *testing.T) {
	for _, cancelConnection := range []bool{true, false} {
		t.Run(fmt.Sprint(cancelConnection), func(t *testing.T) {
			a, b := net.Pipe()
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			go func() { _ = handshake(b) }()
			c, err := Connect(ctx, a, options())
			if err != nil {
				t.Fatal(err)
			}
			if cancelConnection {
				cancel()
				waitCause(t, c, context.Canceled)
			} else {
				_ = b.Close()
				waitCause(t, c, io.EOF)
			}
			_ = b.Close()
			_ = c.Close()
		})
	}
}

func TestFullEventQueueDoesNotBlockResponse(t *testing.T) {
	o := options()
	o.Limits.EventCount = 1
	c, p := connectPeer(t, o)
	ch := make(chan response, 1)
	go func() { r, e := c.Call(t.Context(), "synthetic", nil); ch <- response{r, e} }()
	req := p.read(t)
	p.write(t, `{"method":"queued"}`)
	p.write(t, fmt.Sprintf(`{"id":%s,"result":true}`, req.id.raw))
	r := await(t, ch)
	if r.err != nil || string(r.result) != "true" {
		t.Fatal(r)
	}
	event, err := c.Next(t.Context())
	if err != nil || event.Method != "queued" {
		t.Fatal(event, err)
	}
}
func TestInboundAndOutboundIDNamespacesAreSeparate(t *testing.T) {
	o := options()
	o.ServerRequestMethods = []string{"synthetic"}
	c, p := connectPeer(t, o)
	ch := make(chan response, 1)
	go func() { r, e := c.Call(t.Context(), "outbound", nil); ch <- response{r, e} }()
	req := p.read(t)
	p.write(t, fmt.Sprintf(`{"id":%s,"method":"synthetic"}`, req.id.raw))
	p.write(t, fmt.Sprintf(`{"id":%s,"result":"outbound-result"}`, req.id.raw))
	r := await(t, ch)
	if r.err != nil || string(r.result) != `"outbound-result"` {
		t.Fatal(r)
	}
	event, err := c.Next(t.Context())
	if err != nil || event.Request == nil {
		t.Fatal(event, err)
	}
	replied := make(chan error, 1)
	go func() { replied <- event.Request.Respond(t.Context(), json.RawMessage(`"inbound-result"`)) }()
	wire := p.read(t)
	if wire.id.key != req.id.key || string(wire.result) != `"inbound-result"` {
		t.Fatal(wire)
	}
	if err := await(t, replied); err != nil {
		t.Fatal(err)
	}
}

func TestEventBudgetRetainsResolvedHandleIDCharge(t *testing.T) {
	o := options()
	o.ServerRequestMethods = []string{"x"}
	if err := o.defaults(); err != nil {
		t.Fatal(err)
	}
	a, b := net.Pipe()
	defer a.Close()
	defer b.Close()
	c := newClient(a, o)
	raw := []byte(`{"id":"synthetic","method":"x","params":null}`)
	e, err := parseEnvelope(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err = c.dispatch(e); err != nil {
		t.Fatal(err)
	}
	want := len(raw) + len("null") + len("x") + len(`"synthetic"`) + len("s:synthetic")
	c.mu.Lock()
	got := c.eventBytes
	c.mu.Unlock()
	if got != want {
		t.Fatalf("event bytes %d want %d", got, want)
	}
	resolved, _ := parseEnvelope([]byte(`{"method":"serverRequest/resolved","params":{"threadId":"fixture","requestId":"synthetic"}}`))
	if err = c.dispatch(resolved); err != nil {
		t.Fatal(err)
	}
	c.mu.Lock()
	if c.requestBytes != 0 || c.eventBytes < want {
		t.Fatal("resolved ID lost queue budget")
	}
	c.mu.Unlock()
}
