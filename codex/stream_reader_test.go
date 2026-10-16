package codex

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

func TestEventStreamEarlyBreakReaps(t *testing.T) {
	e := fakeExec(t, "wait")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	stream, err := e.Run(ctx, Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	for _, err := range EventStream(ctx, stream) {
		if err != nil {
			t.Fatal(err)
		}
		break
	}
	if err := stream.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("Wait after break = %v", err)
	}
}

func TestEventStreamMalformedStopsOnce(t *testing.T) {
	e := fakeExec(t, "malformed")
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	stream, err := e.Run(ctx, Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for event, err := range EventStream(ctx, stream) {
		count++
		if event != nil || err == nil || !strings.Contains(err.Error(), "parse codex event") {
			t.Fatalf("event/error = %v/%v", event, err)
		}
		if count > 1 {
			t.Fatal("decode error was retried")
		}
	}
	if err := stream.Wait(); !errors.Is(err, context.Canceled) {
		t.Fatalf("helper was not canceled: %v", err)
	}
}

func TestEventStreamSeparateContextCancellation(t *testing.T) {
	e := fakeExec(t, "wait")
	processCtx, stop := context.WithTimeout(t.Context(), 5*time.Second)
	defer stop()
	stream, err := e.Run(processCtx, Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	count := 0
	for event, err := range EventStream(ctx, stream) {
		if event != nil {
			cancel()
			continue
		}
		count++
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error = %v", err)
		}
	}
	if count != 1 {
		t.Fatalf("errors = %d", count)
	}
}

func TestEventStreamLargeAndFinalUnterminatedLine(t *testing.T) {
	e := fakeExec(t, "large")
	stream, err := e.Run(t.Context(), Args{Input: "test prompt"})
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for event, err := range EventStream(t.Context(), stream) {
		if err != nil {
			t.Fatal(err)
		}
		count++
		if count == 1 && len(event.Item.(*AgentMessageItem).Text) != 100000 {
			t.Fatal("long event truncated")
		}
	}
	if count != 2 {
		t.Fatalf("events = %d", count)
	}
}

func TestEventStreamBlankLinesAndStrictJSONL(t *testing.T) {
	for _, tc := range []struct {
		input         string
		count, errors int
	}{
		{"\n  \r\n{\"type\":\"turn.started\"}\r\n", 1, 0},
		{`{"type":"turn.started"}{"type":"turn.started"}`, 0, 1},
		{`null`, 0, 1},
	} {
		waited := 0
		stream := &ExecStream{stdout: io.NopCloser(strings.NewReader(tc.input)), waitFn: func() error { waited++; return nil }}
		events, errs := 0, 0
		for event, err := range EventStream(t.Context(), stream) {
			if event != nil {
				events++
			}
			if err != nil {
				errs++
			}
		}
		if events != tc.count || errs != tc.errors || waited != 1 {
			t.Fatalf("input %q: events/errors/waits = %d/%d/%d", tc.input, events, errs, waited)
		}
	}
}
