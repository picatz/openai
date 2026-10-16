package liveclient

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

type fakeSocket struct {
	incoming chan []byte
	writes   chan []byte
	done     chan struct{}
	once     sync.Once
}

func newFake() *fakeSocket {
	return &fakeSocket{incoming: make(chan []byte, 32), writes: make(chan []byte, 32), done: make(chan struct{})}
}
func (f *fakeSocket) Read(ctx context.Context) ([]byte, error) {
	select {
	case data := <-f.incoming:
		return data, nil
	case <-f.done:
		return nil, io.EOF
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
func (f *fakeSocket) Write(ctx context.Context, data []byte) error {
	select {
	case f.writes <- append([]byte(nil), data...):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-f.done:
		return io.ErrClosedPipe
	}
}
func (f *fakeSocket) Close() error      { f.once.Do(func() { close(f.done) }); return nil }
func (f *fakeSocket) event(data string) { f.incoming <- []byte(data) }
func readWrite(t *testing.T, f *fakeSocket, want string) map[string]any {
	t.Helper()
	for {
		select {
		case data := <-f.writes:
			var e map[string]any
			if err := json.Unmarshal(data, &e); err != nil {
				t.Fatal(err)
			}
			if e["type"] == want {
				return e
			}
		case <-time.After(time.Second):
			t.Fatalf("did not send %s", want)
		}
	}
}
func options(input []byte) Options {
	return Options{Session: DefaultSession(), Input: io.NopCloser(bytes.NewReader(input)), Output: nopWriteCloser{io.Discard}, MaxDuration: time.Second, ListenAfterEOF: 20 * time.Millisecond, StartupTimeout: time.Second, CloseTimeout: 100 * time.Millisecond, FrameDuration: time.Millisecond}
}

type outcome struct {
	result Result
	err    error
}

func start(t *testing.T, ctx context.Context, s *fakeSocket, opts Options) <-chan outcome {
	t.Helper()
	done := make(chan outcome, 1)
	go func() { r, e := Run(ctx, s, opts); done <- outcome{r, e} }()
	readWrite(t, s, "session.start")
	return done
}
func finish(t *testing.T, done <-chan outcome) outcome {
	t.Helper()
	select {
	case r := <-done:
		return r
	case <-time.After(2 * time.Second):
		t.Fatal("Live run did not end")
		return outcome{}
	}
}
func TestLiveProtocolAndFinalUsage(t *testing.T) {
	socket := newFake()
	var output bytes.Buffer
	opts := options([]byte{1, 2, 3, 4})
	opts.Output = nopWriteCloser{&output}
	done := start(t, t.Context(), socket, opts)
	select {
	case data := <-socket.writes:
		t.Fatalf("sent before started: %s", data)
	case <-time.After(5 * time.Millisecond):
	}
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	audio := readWrite(t, socket, "session.input_audio.append")
	decoded, err := base64.StdEncoding.DecodeString(audio["audio"].(string))
	if err != nil || !bytes.Equal(decoded[:4], []byte{1, 2, 3, 4}) || len(decoded)%2 != 0 {
		t.Fatalf("audio=%v err=%v", decoded, err)
	}
	socket.event(`{"type":"session.usage.updated","usage":{"seconds":2}}`)
	socket.event(`{"type":"session.usage.updated","usage":{"seconds":3}}`)
	socket.event(`{"type":"session.output_audio.delta","delta":"AQIDBA=="}`)
	socket.event(`{"type":"response.event","event":{"type":"response.completed","response":{"usage":{"total_tokens":7}}}}`)
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":4},"reason":"close_requested"}`)
	result := finish(t, done)
	if result.err != nil || !result.result.Finalized || result.result.VoiceSeconds != 4 || len(result.result.BackendUsage) != 1 || !bytes.Equal(output.Bytes(), []byte{1, 2, 3, 4}) {
		t.Fatalf("result=%+v err=%v output=%v", result.result, result.err, output.Bytes())
	}
}
func TestLiveCancellationStillFinalizes(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	socket := newFake()
	opts := options(make([]byte, 48000))
	done := start(t, ctx, socket, opts)
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	readWrite(t, socket, "session.input_audio.append")
	cancel()
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":1},"reason":"close_requested"}`)
	result := finish(t, done)
	if !errors.Is(result.err, context.Canceled) || !result.result.Finalized {
		t.Fatalf("result=%+v err=%v", result.result, result.err)
	}
}
func TestLiveMissingFinalizationIsError(t *testing.T) {
	for _, kind := range []string{"disconnect", "timeout", "missing_usage"} {
		t.Run(kind, func(t *testing.T) {
			socket := newFake()
			done := start(t, t.Context(), socket, options(nil))
			socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
			readWrite(t, socket, "session.close")
			switch kind {
			case "disconnect":
				socket.Close()
			case "missing_usage":
				socket.event(`{"type":"session.closed","usage":{}}`)
			}
			result := finish(t, done)
			if result.err == nil || result.result.Finalized {
				t.Fatalf("result=%+v err=%v", result.result, result.err)
			}
		})
	}
}
func TestLiveMuteNeedsMatchingAcknowledgment(t *testing.T) {
	socket := newFake()
	commands := make(chan Command, 4)
	opts := options(bytes.Repeat([]byte{1, 2}, 24000))
	opts.Controls = commands
	opts.ListenAfterEOF = time.Second
	events := make(chan Event, 8)
	opts.Events = events
	done := start(t, t.Context(), socket, opts)
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	<-events
	commands <- Mute
	mute := readWrite(t, socket, "session.input_audio.mute")
	socket.event(`{"type":"session.input_audio.muted","client_event_id":"wrong"}`)
	<-events
	socket.event(`{"type":"session.input_audio.muted","client_event_id":"` + mute["event_id"].(string) + `"}`)
	<-events
	audio := readWrite(t, socket, "session.input_audio.append")
	data, _ := base64.StdEncoding.DecodeString(audio["audio"].(string))
	for _, b := range data {
		if b != 0 {
			t.Fatal("muted audio was transmitted")
		}
	}
	commands <- Unmute
	unmute := readWrite(t, socket, "session.input_audio.unmute")
	socket.event(`{"type":"session.input_audio.unmuted","client_event_id":"` + unmute["event_id"].(string) + `"}`)
	<-events
	commands <- Close
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":1}}`)
	result := finish(t, done)
	if result.err != nil || result.result.InputMuted {
		t.Fatalf("result=%+v err=%v", result.result, result.err)
	}
}
func TestLiveRejectsOddPCMAndErrors(t *testing.T) {
	socket := newFake()
	done := start(t, t.Context(), socket, options([]byte{1}))
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":0.01}}`)
	result := finish(t, done)
	if result.err == nil || !strings.Contains(result.err.Error(), "incomplete sample") || !result.result.Finalized {
		t.Fatalf("result=%+v err=%v", result.result, result.err)
	}
}
func TestLiveStartupError(t *testing.T) {
	socket := newFake()
	done := start(t, t.Context(), socket, options(nil))
	socket.event(`{"type":"error","error":{"message":"synthetic failure"}}`)
	result := finish(t, done)
	if result.err == nil || !strings.Contains(result.err.Error(), "synthetic failure") {
		t.Fatal(result.err)
	}
}

type nopWriteCloser struct{ io.Writer }

func (nopWriteCloser) Close() error { return nil }

type blockingOutput struct {
	started chan struct{}
	closed  chan struct{}
	once    sync.Once
}

func (b *blockingOutput) Write(p []byte) (int, error) {
	close(b.started)
	<-b.closed
	return 0, io.ErrClosedPipe
}
func (b *blockingOutput) Close() error { b.once.Do(func() { close(b.closed) }); return nil }
func TestCancelUnblocksAudioBackpressureAndFinalizes(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	socket := newFake()
	output := &blockingOutput{started: make(chan struct{}), closed: make(chan struct{})}
	opts := options(make([]byte, 48000))
	opts.Output = output
	done := start(t, ctx, socket, opts)
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	socket.event(`{"type":"session.output_audio.delta","delta":"AQI="}`)
	select {
	case <-output.started:
	case <-time.After(time.Second):
		t.Fatal("audio did not reach output")
	}
	cancel()
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":0.1}}`)
	result := finish(t, done)
	if !errors.Is(result.err, context.Canceled) || !result.result.Finalized {
		t.Fatalf("result=%+v err=%v", result.result, result.err)
	}
}
func TestMuteUnmuteCommandsDoNotOvertakeAcknowledgments(t *testing.T) {
	socket := newFake()
	commands := make(chan Command, 4)
	opts := options(make([]byte, 48000))
	opts.Controls = commands
	done := start(t, t.Context(), socket, opts)
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	readWrite(t, socket, "session.input_audio.append")
	commands <- Mute
	mute := readWrite(t, socket, "session.input_audio.mute")
	commands <- Unmute
	socket.event(`{"type":"session.input_audio.muted","client_event_id":"` + mute["event_id"].(string) + `"}`)
	unmute := readWrite(t, socket, "session.input_audio.unmute")
	socket.event(`{"type":"session.input_audio.unmuted","client_event_id":"` + unmute["event_id"].(string) + `"}`)
	commands <- Close
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":0.1}}`)
	result := finish(t, done)
	if result.err != nil || result.result.InputMuted {
		t.Fatalf("result=%+v err=%v", result.result, result.err)
	}
}

func TestLiveRejectsMalformedUsageUpdates(t *testing.T) {
	for _, usage := range []string{`{}`, `{"seconds":null}`, `{"seconds":-1}`, `null`} {
		socket := newFake()
		done := start(t, t.Context(), socket, options(nil))
		socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
		socket.event(`{"type":"session.usage.updated","usage":` + usage + `}`)
		result := finish(t, done)
		if result.err == nil || result.result.Finalized {
			t.Fatalf("usage=%s result=%+v err=%v", usage, result.result, result.err)
		}
	}
}

func TestSlowEventConsumerCannotHoldSessionOpen(t *testing.T) {
	socket := newFake()
	opts := options(nil)
	opts.Events = make(chan Event)
	opts.EventDeliveryTimeout = 5 * time.Millisecond
	done := start(t, t.Context(), socket, opts)
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":0.1}}`)
	result := finish(t, done)
	if result.err == nil || !strings.Contains(result.err.Error(), "event consumer") || !result.result.Finalized {
		t.Fatalf("result=%+v err=%v", result.result, result.err)
	}
}
func TestCancelDuringFinalizationDoesNotBecomeSuccess(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	socket := newFake()
	done := start(t, ctx, socket, options(nil))
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	readWrite(t, socket, "session.close")
	cancel()
	socket.event(`{"type":"session.closed","usage":{"seconds":0.1}}`)
	result := finish(t, done)
	if !errors.Is(result.err, context.Canceled) || !result.result.Finalized {
		t.Fatalf("result=%+v err=%v", result.result, result.err)
	}
}
func TestQueuedMuteStopsInputBeforeEarlierUnmuteAck(t *testing.T) {
	socket := newFake()
	commands := make(chan Command, 4)
	opts := options(bytes.Repeat([]byte{1, 2}, 24000))
	opts.Controls = commands
	done := start(t, t.Context(), socket, opts)
	socket.event(`{"type":"session.started","session":{"id":"live_test"}}`)
	readWrite(t, socket, "session.input_audio.append")
	commands <- Unmute
	unmute := readWrite(t, socket, "session.input_audio.unmute")
	commands <- Mute
	// Allow the command loop to observe mute, then drain any earlier queued frame.
	time.Sleep(5 * time.Millisecond)
	for len(socket.writes) > 0 {
		<-socket.writes
	}
	audio := readWrite(t, socket, "session.input_audio.append")
	data, _ := base64.StdEncoding.DecodeString(audio["audio"].(string))
	for _, b := range data {
		if b != 0 {
			t.Fatal("queued mute leaked PCM while waiting for prior ack")
		}
	}
	socket.event(`{"type":"session.input_audio.unmuted","client_event_id":"` + unmute["event_id"].(string) + `"}`)
	mute := readWrite(t, socket, "session.input_audio.mute")
	socket.event(`{"type":"session.input_audio.muted","client_event_id":"` + mute["event_id"].(string) + `"}`)
	commands <- Close
	readWrite(t, socket, "session.close")
	socket.event(`{"type":"session.closed","usage":{"seconds":0.1}}`)
	result := finish(t, done)
	if result.err != nil {
		t.Fatal(result.err)
	}
}
