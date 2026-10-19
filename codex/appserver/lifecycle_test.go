package appserver

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type observedTransport struct {
	net.Conn
	writes  atomic.Int32
	started chan struct{}
	fault   string
}

func (o *observedTransport) Write(p []byte) (int, error) {
	if o.writes.Add(1) == 3 {
		close(o.started)
		switch o.fault {
		case "short":
			return len(p) / 2, nil
		case "zero":
			return 0, nil
		case "failure":
			return len(p) / 2, io.ErrUnexpectedEOF
		case "invalid":
			return len(p) + 1, nil
		}
	}
	return o.Conn.Write(p)
}
func connectObserved(t *testing.T, o Options, fault string) (*Client, *peer, *observedTransport) {
	t.Helper()
	a, b := net.Pipe()
	_ = b.SetDeadline(time.Now().Add(5 * time.Second))
	transport := &observedTransport{Conn: a, started: make(chan struct{}), fault: fault}
	ch := make(chan error, 1)
	go func() { ch <- handshake(b) }()
	c, err := Connect(t.Context(), transport, o)
	if err != nil {
		t.Fatal(err)
	}
	if err := <-ch; err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close(); _ = b.Close() })
	return c, &peer{b, newFrameReader(b, 1<<20)}, transport
}
func TestShortWriteIsTerminalWithoutRetry(t *testing.T) {
	for _, fault := range []string{"short", "zero", "failure", "invalid"} {
		t.Run(fault, func(t *testing.T) {
			c, _, tr := connectObserved(t, options(), fault)
			_, err := c.Call(t.Context(), "synthetic", nil)
			want := io.ErrShortWrite
			if fault == "failure" {
				want = io.ErrUnexpectedEOF
			}
			if !errors.Is(err, want) {
				t.Fatal(err)
			}
			waitCause(t, c, want)
			if tr.writes.Load() != 3 {
				t.Fatal("retried damaged frame", tr.writes.Load())
			}
		})
	}
}
func TestCancelQueuedWriteDoesNotTransmit(t *testing.T) {
	c, p, tr := connectObserved(t, options(), "")
	first := make(chan error, 1)
	go func() { _, e := c.Call(t.Context(), "first", nil); first <- e }()
	await(t, tr.started)
	ctx, cancel := context.WithCancel(t.Context())
	second := make(chan error, 1)
	go func() { _, e := c.Call(ctx, "must-not-send", nil); second <- e }()
	// Synchronize on admission while the first net.Pipe write is blocked.
	deadline := time.After(3 * time.Second)
	for {
		c.mu.Lock()
		queued := len(c.writes)
		c.mu.Unlock()
		if queued == 1 {
			break
		}
		select {
		case <-deadline:
			t.Fatal("second call not queued")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	if e := await(t, second); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	req := p.read(t)
	if req.method != "first" {
		t.Fatal(req.method)
	}
	p.write(t, fmt.Sprintf(`{"id":%s,"result":null}`, req.id.raw))
	if e := await(t, first); e != nil {
		t.Fatal(e)
	}
	final := make(chan error, 1)
	go func() { _, e := c.Call(t.Context(), "final", nil); final <- e }()
	req = p.read(t)
	if req.method != "final" {
		t.Fatal("canceled queued call transmitted", req.method)
	}
	p.write(t, fmt.Sprintf(`{"id":%s,"result":null}`, req.id.raw))
	if e := await(t, final); e != nil {
		t.Fatal(e)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.pending) != 0 || c.abandoned != 0 || c.writeCount != 0 || c.writeBytes != 0 {
		t.Fatal("retained canceled state")
	}
}
func TestWriteQueueBoundsIncludeActiveWrite(t *testing.T) {
	for _, limit := range []string{"count", "bytes"} {
		t.Run(limit, func(t *testing.T) {
			o := options()
			if limit == "count" {
				o.Limits.WriteCount = 1
			} else {
				o.Limits.WriteBytes = 160
			}
			c, _, tr := connectObserved(t, o, "")
			first := make(chan error, 1)
			go func() { _, e := c.Call(t.Context(), strings.Repeat("x", 80), nil); first <- e }()
			await(t, tr.started)
			_, err := c.Call(t.Context(), strings.Repeat("y", 80), nil)
			if !errors.Is(err, ErrWriteLimit) {
				t.Fatal(err)
			}
			waitCause(t, c, ErrWriteLimit)
			if e := await(t, first); !errors.Is(e, ErrWriteLimit) {
				t.Fatal(e)
			}
		})
	}
}
func TestCloseUnblocksActiveWrite(t *testing.T) {
	c, _, tr := connectObserved(t, options(), "")
	out := make(chan error, 1)
	go func() { _, e := c.Call(t.Context(), "blocked", nil); out <- e }()
	await(t, tr.started)
	closed := make(chan error, 1)
	go func() { closed <- c.Close() }()
	if e := await(t, closed); e != nil {
		t.Fatal(e)
	}
	if e := await(t, out); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	waitCause(t, c, ErrClosed)
}
func TestIncomingPendingBounds(t *testing.T) {
	for _, kind := range []string{"count", "bytes"} {
		t.Run(kind, func(t *testing.T) {
			o := options()
			o.ServerRequestMethods = []string{"x"}
			if kind == "count" {
				o.Limits.PendingRequests = 1
			} else {
				o.Limits.PendingRequestBytes = 7
			}
			c, p := connectPeer(t, o)
			p.write(t, `{"id":"a","method":"x"}`)
			first, err := c.Next(t.Context())
			if err != nil {
				t.Fatal(err)
			}
			p.write(t, `{"id":"b","method":"x"}`)
			waitCause(t, c, ErrRequestLimit)
			select {
			case <-first.Request.Done():
			default:
				t.Fatal("request remains live")
			}
		})
	}
}
func TestResolvedRequestCancelsQueuedReply(t *testing.T) {
	o := options()
	o.ServerRequestMethods = []string{"synthetic"}
	c, p, tr := connectObserved(t, o, "")
	first := make(chan error, 1)
	go func() { _, e := c.Call(t.Context(), "blocked", nil); first <- e }()
	await(t, tr.started)
	p.write(t, `{"id":"approval","method":"synthetic"}`)
	event, err := c.Next(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	replied := make(chan error, 1)
	go func() { replied <- event.Request.Respond(t.Context(), nil) }()
	for {
		c.mu.Lock()
		queued := len(c.writes)
		c.mu.Unlock()
		if queued == 1 {
			break
		}
		time.Sleep(time.Millisecond)
	}
	p.write(t, `{"method":"serverRequest/resolved","params":{"threadId":"fixture","requestId":"approval"}}`)
	if _, e := c.Next(t.Context()); e != nil {
		t.Fatal(e)
	}
	if e := await(t, replied); !errors.Is(e, ErrRequestDone) {
		t.Fatal(e)
	}
	_ = c.Close()
	if e := await(t, first); !errors.Is(e, ErrClosed) {
		t.Fatal(e)
	}
	if tr.writes.Load() != 3 {
		t.Fatal("resolved reply reached writer")
	}
}
func TestResolutionVersusReplyRace(t *testing.T) {
	for range 50 {
		o := options()
		o.ServerRequestMethods = []string{"x"}
		c, p := connectPeer(t, o)
		p.write(t, `{"id":1,"method":"x"}`)
		event, err := c.Next(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		reply := make(chan error, 1)
		readDone := make(chan struct{})
		go func() { defer close(readDone); _, _ = p.r.next() }()
		go func() { reply <- event.Request.Respond(t.Context(), nil) }()
		p.write(t, `{"method":"serverRequest/resolved","params":{"threadId":"fixture","requestId":1}}`)
		_, err = c.Next(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		err = await(t, reply)
		if err != nil && !errors.Is(err, ErrRequestDone) {
			t.Fatal(err)
		}
		_ = c.Close()
		await(t, readDone)
	}
}
func TestRequestHandleCannotAffectOtherConnection(t *testing.T) {
	o := options()
	o.ServerRequestMethods = []string{"x"}
	c1, p1 := connectPeer(t, o)
	c2, p2 := connectPeer(t, o)
	p1.write(t, `{"id":1,"method":"x"}`)
	one, _ := c1.Next(t.Context())
	_ = c1.Close()
	p2.write(t, `{"id":1,"method":"x"}`)
	two, _ := c2.Next(t.Context())
	if e := one.Request.Respond(t.Context(), nil); !errors.Is(e, ErrRequestDone) {
		t.Fatal(e)
	}
	ch := make(chan error, 1)
	go func() { ch <- two.Request.Reject(t.Context(), RPCError{Code: -1, Message: "synthetic"}) }()
	wire := p2.read(t)
	if wire.rpcError == nil || wire.rpcError.Code != -1 {
		t.Fatal(wire)
	}
	if err := await(t, ch); err != nil {
		t.Fatal(err)
	}
}
func TestHandshakeFailuresCloseTransport(t *testing.T) {
	for _, kind := range []string{"timeout", "malformed", "rpc-error", "wrong-id", "missing-field", "case-alias", "pre-ready-overflow"} {
		t.Run(kind, func(t *testing.T) {
			a, b := net.Pipe()
			defer b.Close()
			_ = b.SetDeadline(time.Now().Add(3 * time.Second))
			o := options()
			if kind == "timeout" {
				o.InitializeTimeout = 20 * time.Millisecond
			}
			if kind == "pre-ready-overflow" {
				o.Limits.EventCount = 1
			}
			server := make(chan error, 1)
			go func() {
				r := newFrameReader(b, 1<<20)
				raw, e := r.next()
				if e != nil {
					server <- e
					return
				}
				req, e := parseEnvelope(raw)
				if e != nil {
					server <- e
					return
				}
				switch kind {
				case "timeout":
				case "malformed":
					_, _ = io.WriteString(b, "not json\n")
				case "rpc-error":
					_, _ = fmt.Fprintf(b, `{"id":%s,"error":{"code":-1,"message":"denied"}}`+"\n", req.id.raw)
				case "wrong-id":
					_, _ = io.WriteString(b, `{"id":"wrong","result":null}`+"\n")
				case "missing-field":
					_, _ = fmt.Fprintf(b, `{"id":%s,"result":{"userAgent":"synthetic"}}`+"\n", req.id.raw)
				case "case-alias":
					_, _ = fmt.Fprintf(b, `{"id":%s,"result":{"userAgent":"expected","UserAgent":"wrong","codexHome":"/fixture","platformFamily":"unix","platformOs":"linux"}}`+"\n", req.id.raw)
					_, _ = r.next()
				case "pre-ready-overflow":
					_, _ = io.WriteString(b, "{\"method\":\"tick\"}\n{\"method\":\"tick\"}\n")
				}
				_, e = r.next()
				server <- e
			}()
			c, err := Connect(t.Context(), a, o)
			if kind == "case-alias" {
				if err != nil || c.Initialization().UserAgent != "expected" {
					t.Fatal(c, err)
				}
				_ = c.Close()
			} else if err == nil {
				_ = c.Close()
				t.Fatal("handshake accepted")
			}
			if e := await(t, server); !errors.Is(e, io.EOF) && !errors.Is(e, io.ErrClosedPipe) {
				t.Fatal("transport not closed", e)
			}
		})
	}
}
func TestInvalidOptionsAndCanceledConnectCloseTransport(t *testing.T) {
	for _, kind := range []string{"identity", "limit", "timeout", "canceled", "nil-context"} {
		t.Run(kind, func(t *testing.T) {
			a, b := net.Pipe()
			defer b.Close()
			o := options()
			ctx := t.Context()
			switch kind {
			case "identity":
				o.ClientInfo.Name = ""
			case "limit":
				o.Limits.FrameBytes = -1
			case "timeout":
				o.InitializeTimeout = -1
			case "canceled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "nil-context":
				ctx = nil
			}
			if c, err := Connect(ctx, a, o); err == nil || c != nil {
				t.Fatal(c, err)
			}
			_ = b.SetReadDeadline(time.Now().Add(time.Second))
			if _, err := b.Read(make([]byte, 1)); err != io.EOF {
				t.Fatal(err)
			}
		})
	}
}
func TestCallerCancellationDoesNotCancelConnection(t *testing.T) {
	c, p := connectPeer(t, options())
	ctx, cancel := context.WithCancel(t.Context())
	ch := make(chan error, 1)
	go func() { _, e := c.Next(ctx); ch <- e }()
	cancel()
	if e := await(t, ch); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	p.write(t, `{"method":"still-alive"}`)
	e, err := c.Next(t.Context())
	if err != nil || e.Method != "still-alive" {
		t.Fatal(e, err)
	}
}
func TestCallsAndCloseRace(t *testing.T) {
	for range 20 {
		c, p := connectPeer(t, options())
		var wg sync.WaitGroup
		for range 8 {
			wg.Add(1)
			go func() { defer wg.Done(); _, _ = c.Call(t.Context(), "race", nil) }()
		}
		go func() {
			for {
				e, err := p.r.next()
				if err != nil {
					return
				}
				req, err := parseEnvelope(e)
				if err != nil {
					return
				}
				_, err = fmt.Fprintf(p, `{"id":%s,"result":null}`+"\n", req.id.raw)
				if err != nil {
					return
				}
			}
		}()
		_ = c.Close()
		done := make(chan bool, 1)
		go func() { wg.Wait(); done <- true }()
		await(t, done)
	}
}

func TestEarlyIncomingRequestsPreserveInitializeFirst(t *testing.T) {
	for _, registered := range []bool{false, true} {
		t.Run(fmt.Sprint(registered), func(t *testing.T) {
			a, b := net.Pipe()
			defer b.Close()
			_ = b.SetDeadline(time.Now().Add(3 * time.Second))
			o := options()
			if registered {
				o.ServerRequestMethods = []string{"synthetic"}
			}
			server := make(chan error, 1)
			go func() {
				r := newFrameReader(b, 1<<20)
				// Deliberately send before reading initialize, probing startup order.
				if _, err := io.WriteString(b, `{"id":1,"method":"synthetic"}`+"\n"); err != nil {
					server <- err
					return
				}
				raw, err := r.next()
				if err != nil {
					server <- err
					return
				}
				req, err := parseEnvelope(raw)
				if err != nil || req.method != "initialize" {
					server <- fmt.Errorf("first outbound was not initialize: %s", raw)
					return
				}
				if !registered {
					raw, err = r.next()
					if err != nil {
						server <- err
						return
					}
					rejection, e := parseEnvelope(raw)
					if e != nil || rejection.rpcError == nil || rejection.rpcError.Code != -32601 {
						server <- fmt.Errorf("missing rejection: %s", raw)
						return
					}
				}
				_, err = fmt.Fprintf(b, `{"id":%s,"result":{"userAgent":"fixture","codexHome":"/fixture","platformFamily":"unix","platformOs":"linux"}}`+"\n", req.id.raw)
				if err != nil {
					server <- err
					return
				}
				raw, err = r.next()
				if err == nil && string(raw) != `{"method":"initialized"}` {
					err = fmt.Errorf("unexpected ready frame: %s", raw)
				}
				server <- err
			}()
			c, err := Connect(t.Context(), a, o)
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if err := await(t, server); err != nil {
				t.Fatal(err)
			}
			e, err := c.Next(t.Context())
			if err != nil || e.Method != "synthetic" || e.Rejected == registered || (e.Request != nil) != registered {
				t.Fatal(e, err)
			}
		})
	}
}

func TestExplicitInitializationCapabilities(t *testing.T) {
	a, b := net.Pipe()
	defer b.Close()
	_ = b.SetDeadline(time.Now().Add(3 * time.Second))
	o := options()
	o.ExperimentalAPI = true
	o.OptOutNotificationMethods = []string{"synthetic/tick"}
	received := make(chan string, 1)
	go func() {
		r := newFrameReader(b, 1<<20)
		raw, err := r.next()
		if err != nil {
			received <- err.Error()
			return
		}
		received <- string(raw)
		req, _ := parseEnvelope(raw)
		_, _ = fmt.Fprintf(b, `{"id":%s,"result":{"userAgent":"fixture","codexHome":"/fixture","platformFamily":"unix","platformOs":"linux"}}`+"\n", req.id.raw)
		_, _ = r.next()
	}()
	c, err := Connect(t.Context(), a, o)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	raw := await(t, received)
	if !strings.Contains(raw, `"experimentalApi":true`) || !strings.Contains(raw, `"optOutNotificationMethods":["synthetic/tick"]`) || strings.Contains(raw, "attestation") || strings.Contains(raw, "protocolVersion") {
		t.Fatal(raw)
	}
}
func TestQueueNotificationsBeforeReady(t *testing.T) {
	a, b := net.Pipe()
	defer b.Close()
	_ = b.SetDeadline(time.Now().Add(3 * time.Second))
	server := make(chan error, 1)
	go func() {
		r := newFrameReader(b, 1<<20)
		raw, err := r.next()
		if err != nil {
			server <- err
			return
		}
		req, _ := parseEnvelope(raw)
		_, _ = io.WriteString(b, `{"method":"early","params":null}`+"\n")
		_, _ = fmt.Fprintf(b, `{"id":%s,"result":{"userAgent":"fixture","codexHome":"/fixture","platformFamily":"unix","platformOs":"linux"}}`+"\n", req.id.raw)
		_, err = r.next()
		server <- err
	}()
	c, err := Connect(t.Context(), a, options())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if err := await(t, server); err != nil {
		t.Fatal(err)
	}
	e, err := c.Next(t.Context())
	if err != nil || e.Method != "early" {
		t.Fatal(e, err)
	}
}
func TestFrameAndPayloadBounds(t *testing.T) {
	o := options()
	o.Limits.FrameBytes = 512
	c, p := connectPeer(t, o)
	if _, err := c.Call(t.Context(), "x", []byte(`{"x":1,"x":2}`)); !errors.Is(err, ErrProtocol) {
		t.Fatal(err)
	}
	if _, err := c.Call(t.Context(), strings.Repeat("x", 513), nil); !errors.Is(err, ErrFrameLimit) {
		t.Fatal(err)
	}
	if _, err := c.Call(t.Context(), "x", []byte(`"`+strings.Repeat("x", 513)+`"`)); !errors.Is(err, ErrFrameLimit) {
		t.Fatal(err)
	}
	// Use a separate connection to exercise actual read framing limits.
	_ = p.Close()
	_ = c.Close()
	c, p = connectPeer(t, o)
	_, _ = io.WriteString(p, `{"method":"x","params":"`+strings.Repeat("x", 513)+`"}`+"\n")
	waitCause(t, c, ErrFrameLimit)
}

func TestCompletedResponseSurvivesLaterShutdown(t *testing.T) {
	for range 100 {
		o := options()
		if err := o.defaults(); err != nil {
			t.Fatal(err)
		}
		a, b := net.Pipe()
		c := newClient(a, o)
		out := make(chan response, 1)
		go func() { r, e := c.Call(t.Context(), "last", nil); out <- response{r, e} }()
		deadline := time.After(3 * time.Second)
		for {
			c.mu.Lock()
			ready := len(c.writes) == 1
			c.mu.Unlock()
			if ready {
				break
			}
			select {
			case <-deadline:
				t.Fatal("call not queued")
			default:
				time.Sleep(time.Millisecond)
			}
		}
		c.mu.Lock()
		j := c.writes[0]
		c.writes = nil
		c.writeCount = 0
		c.writeBytes = 0
		j.state = 2
		j.frame = nil
		j.done <- nil
		c.mu.Unlock()
		e, err := parseEnvelope([]byte(`{"id":"go:1","result":true}`))
		if err != nil {
			t.Fatal(err)
		}
		if err = c.dispatch(e); err != nil {
			t.Fatal(err)
		}
		c.shutdown(io.EOF)
		r := await(t, out)
		if r.err != nil || string(r.result) != "true" {
			t.Fatal("completed response lost", r)
		}
		_ = b.Close()
	}
}

type pipeDuplex struct {
	read  *io.PipeReader
	write *io.PipeWriter
	once  sync.Once
}

func (p *pipeDuplex) Read(b []byte) (int, error)  { return p.read.Read(b) }
func (p *pipeDuplex) Write(b []byte) (int, error) { return p.write.Write(b) }
func (p *pipeDuplex) Close() error {
	p.once.Do(func() { _ = p.read.Close(); _ = p.write.Close() })
	return nil
}
func TestPairedIOPipes(t *testing.T) {
	ar, bw := io.Pipe()
	br, aw := io.Pipe()
	a, b := &pipeDuplex{read: ar, write: aw}, &pipeDuplex{read: br, write: bw}
	defer b.Close()
	server := make(chan error, 1)
	go func() {
		r := newFrameReader(b, 1<<20)
		raw, err := r.next()
		if err != nil {
			server <- err
			return
		}
		req, err := parseEnvelope(raw)
		if err != nil {
			server <- err
			return
		}
		_, err = fmt.Fprintf(b, `{"id":%s,"result":{"userAgent":"fixture","codexHome":"/fixture","platformFamily":"unix","platformOs":"linux"}}`+"\n", req.id.raw)
		if err != nil {
			server <- err
			return
		}
		_, err = r.next()
		server <- err
	}()
	c, err := Connect(t.Context(), a, options())
	if err != nil {
		t.Fatal(err)
	}
	if err := await(t, server); err != nil {
		t.Fatal(err)
	}
	_ = c.Close()
	waitCause(t, c, ErrClosed)
}

func TestInvalidLocalReplyLeavesHandleUsable(t *testing.T) {
	o := options()
	o.ServerRequestMethods = []string{"synthetic"}
	c, p := connectPeer(t, o)
	p.write(t, `{"id":"request","method":"synthetic"}`)
	event, err := c.Next(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	r := event.Request
	if err := r.Reject(t.Context(), RPCError{Code: -1, Message: string([]byte{0xff})}); !errors.Is(err, ErrProtocol) {
		t.Fatal(err)
	}
	if err := r.Reject(t.Context(), RPCError{Code: -1, Message: "x", Data: []byte(`{"x":1,"x":2}`)}); !errors.Is(err, ErrProtocol) {
		t.Fatal(err)
	}
	if err := r.Respond(t.Context(), []byte(`"\ud800"`)); !errors.Is(err, ErrProtocol) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := r.Respond(ctx, nil); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	select {
	case <-r.Done():
		t.Fatal("invalid local input consumed request")
	default:
	}
	ch := make(chan error, 1)
	go func() {
		ch <- r.Reject(t.Context(), RPCError{Code: -1, Message: "synthetic rejection", Data: []byte("null")})
	}()
	reply := p.read(t)
	if reply.rpcError == nil || reply.rpcError.Message != "synthetic rejection" {
		t.Fatal(reply)
	}
	if err := await(t, ch); err != nil {
		t.Fatal(err)
	}
}

func TestPrematureResponseNeverDispatchesQueuedCall(t *testing.T) {
	c, p, tr := connectObserved(t, options(), "")
	first := make(chan error, 1)
	go func() { _, e := c.Call(t.Context(), "blocked", nil); first <- e }()
	await(t, tr.started)
	second := make(chan error, 1)
	go func() { _, e := c.Call(t.Context(), "never-dispatch", nil); second <- e }()
	var id string
	deadline := time.After(3 * time.Second)
	for {
		c.mu.Lock()
		if len(c.writes) == 1 {
			frame := c.writes[0].frame
			req, err := parseEnvelope(frame[:len(frame)-1])
			if err != nil {
				c.mu.Unlock()
				t.Fatal(err)
			}
			id = string(req.id.raw)
		}
		c.mu.Unlock()
		if id != "" {
			break
		}
		select {
		case <-deadline:
			t.Fatal("second call not queued")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	p.write(t, fmt.Sprintf(`{"id":%s,"result":true}`, id))
	waitCause(t, c, ErrProtocol)
	for _, ch := range []chan error{first, second} {
		if err := await(t, ch); !errors.Is(err, ErrProtocol) {
			t.Fatal(err)
		}
	}
	if tr.writes.Load() != 3 {
		t.Fatal("queued side effect was written after premature response")
	}
}
