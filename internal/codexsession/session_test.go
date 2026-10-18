package codexsession

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/picatz/openai/codex/appserver"
)

func fixtureFrames(t *testing.T) ([]map[string]json.RawMessage, []string) {
	t.Helper()
	data, err := os.ReadFile("testdata/conversation.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	var frames []map[string]json.RawMessage
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var f map[string]json.RawMessage
		if err := json.Unmarshal([]byte(line), &f); err != nil {
			t.Fatal(err)
		}
		frames = append(frames, f)
	}
	index, err := os.ReadFile("testdata/conversation.index.json")
	if err != nil {
		t.Fatal(err)
	}
	var entries []struct {
		Direction string `json:"direction"`
	}
	if err := json.Unmarshal(index, &entries); err != nil {
		t.Fatal(err)
	}
	var directions []string
	for _, e := range entries {
		directions = append(directions, e.Direction)
	}
	return frames, directions
}
func TestPinnedConversation(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	frames, directions := fixtureFrames(t)
	client, server := net.Pipe()
	defer server.Close()
	peerDone := make(chan error, 1)
	go func() {
		r := bufio.NewReader(server)
		for i, f := range frames {
			if directions[i] == "server" {
				b, _ := json.Marshal(f)
				if _, err := server.Write(append(b, '\n')); err != nil {
					peerDone <- err
					return
				}
				continue
			}
			line, err := r.ReadBytes('\n')
			if err != nil {
				peerDone <- err
				return
			}
			var got map[string]json.RawMessage
			if err := json.Unmarshal(line, &got); err != nil {
				peerDone <- err
				return
			}
			if string(got["method"]) != string(f["method"]) || string(got["id"]) != string(f["id"]) {
				peerDone <- fmt.Errorf("frame %d mismatch: %s", i, line)
				return
			}
			if i == 0 { // Identity title is presentation only; no capabilities advertised.
				var p map[string]json.RawMessage
				_ = json.Unmarshal(got["params"], &p)
				if len(p) != 1 || p["clientInfo"] == nil {
					peerDone <- errors.New("unexpected initialize capability")
					return
				}
			} else if want, ok := f["params"]; ok {
				var a, b any
				_ = json.Unmarshal(want, &a)
				_ = json.Unmarshal(got["params"], &b)
				if !reflect.DeepEqual(a, b) {
					peerDone <- fmt.Errorf("frame %d params differ", i)
					return
				}
			}
		}
		peerDone <- nil
	}()
	s, err := connect(ctx, client, Options{Directory: "/synthetic/workspace", Model: "synthetic-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if s.Info().Model != "synthetic-model" || s.Info().Sandbox != "workspaceWrite" {
		t.Fatal(s.Info())
	}
	if err := s.StartTurn(ctx, "Create a greeting file and check its contents."); err != nil {
		t.Fatal(err)
	}
	first := collectTurn(t, ctx, s)
	if first[len(first)-1].Status != "completed" {
		t.Fatal(first)
	}
	var command, files, assistant bool
	for _, e := range first {
		if e.Kind == "item" {
			switch e.Item.Kind {
			case "command":
				command = true
			case "files":
				files = true
			case "message":
				assistant = true
			}
		}
	}
	if !command || !files || !assistant {
		t.Fatalf("missing actual item mappings: command=%t files=%t assistant=%t", command, files, assistant)
	}
	var secondPrompt string
	var request struct {
		Input []textInput `json:"input"`
	}
	_ = json.Unmarshal(frames[24]["params"], &request)
	secondPrompt = request.Input[0].Text
	if err := s.StartTurn(ctx, secondPrompt); err != nil {
		t.Fatal(err)
	}
	if err := s.Interrupt(ctx); err != nil {
		t.Fatal(err)
	}
	second := collectTurn(t, ctx, s)
	if second[len(second)-1].Status != "interrupted" {
		t.Fatal(second)
	}
	if first[0].Generation >= second[0].Generation {
		t.Fatal("generation did not increase")
	}
	select {
	case err := <-peerDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}
func collectTurn(t *testing.T, ctx context.Context, s *Session) []Event {
	t.Helper()
	var out []Event
	for {
		e, err := s.Next(ctx)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, e)
		if e.Kind == "end" {
			return out
		}
	}
}

func reducerSession() *Session {
	return &Session{info: Info{ThreadID: "thread"}, active: true, turnID: "turn", generation: 1, items: map[string]itemState{}, events: make(chan Event, maxEvents)}
}
func notification(method string, p any) appserver.Event {
	b, _ := json.Marshal(p)
	return appserver.Event{Method: method, Params: b}
}
func itemEvent(method, id, kind, text, status string) appserver.Event {
	return notification(method, map[string]any{"threadId": "thread", "turnId": "turn", "item": map[string]any{"id": id, "type": kind, "text": text, "status": status}})
}
func deltaEvent(id, delta string) appserver.Event {
	return notification("item/agentMessage/delta", map[string]any{"threadId": "thread", "turnId": "turn", "itemId": id, "delta": delta})
}
func TestReducerMonotonicAndFinalSnapshot(t *testing.T) {
	s := reducerSession()
	for _, e := range []appserver.Event{itemEvent("item/started", "a", "agentMessage", "", ""), deltaEvent("a", "partial"), itemEvent("item/completed", "a", "agentMessage", "final", ""), deltaEvent("a", "stale"), itemEvent("item/started", "a", "agentMessage", "resurrect", "")} {
		if err := s.reduceLocked(e); err != nil {
			t.Fatal(err)
		}
	}
	if len(s.events) != 3 {
		t.Fatalf("events %d", len(s.events))
	}
	<-s.events
	<-s.events
	if e := <-s.events; e.Item.Text != "final" || e.Item.Status != "completed" {
		t.Fatal(e)
	}
	if err := s.reduceLocked(notification("turn/completed", map[string]any{"threadId": "thread", "turn": map[string]any{"id": "turn", "status": "completed", "items": []any{}}})); err != nil {
		t.Fatal(err)
	}
	if e := <-s.events; e.Kind != "end" || e.Status != "completed" {
		t.Fatal(e)
	}
	if err := s.reduceLocked(deltaEvent("a", "late")); err != nil || len(s.events) != 0 {
		t.Fatal("accepted post-terminal data", err)
	}
}
func TestReducerForeignUnknownAndRejected(t *testing.T) {
	s := reducerSession()
	for _, e := range []appserver.Event{notification("future/method", map[string]any{"secret": "do not display"}), notification("item/agentMessage/delta", map[string]any{"threadId": "foreign", "turnId": "turn", "itemId": "a", "delta": "foreign"})} {
		if err := s.reduceLocked(e); err != nil {
			t.Fatal(err)
		}
	}
	if len(s.events) != 0 {
		t.Fatal("foreign or unknown event emitted")
	}
	if err := s.reduceLocked(appserver.Event{Method: "item/commandExecution/requestApproval", Rejected: true, Params: json.RawMessage(`{"command":"sensitive command"}`)}); err != nil {
		t.Fatal(err)
	}
	e := <-s.events
	if e.Kind != "notice" || !strings.Contains(e.Notice, "rejected") || strings.Contains(e.Notice, "sensitive") {
		t.Fatal(e)
	}
}
func TestReducerBounds(t *testing.T) {
	s := reducerSession()
	if err := s.reduceLocked(itemEvent("item/started", "a", "agentMessage", "", "")); err != nil {
		t.Fatal(err)
	}
	<-s.events
	if err := s.reduceLocked(deltaEvent("a", strings.Repeat("x", 2*maxTextBytes))); err != nil {
		t.Fatal(err)
	}
	e := <-s.events
	if len(e.Delta) > maxTextBytes || !strings.Contains(e.Delta, "truncated") {
		t.Fatal("delta is unbounded or unlabeled")
	}
	for range 100 {
		if err := s.reduceLocked(deltaEvent("a", "more")); err != nil {
			t.Fatal(err)
		}
	}
	if len(s.events) != 0 {
		t.Fatal("continued accumulating truncated text")
	}
	for i := 0; i < maxItems; i++ {
		s.items[fmt.Sprint(i)] = itemState{terminal: true}
	}
	if err := s.reduceLocked(itemEvent("item/started", "new", "agentMessage", "", "")); !errors.Is(err, ErrLimit) {
		t.Fatal(err)
	}
	s = reducerSession()
	for range maxEvents {
		if err := s.emitLocked(Event{Kind: "notice"}); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.emitLocked(Event{Kind: "notice"}); !errors.Is(err, ErrLimit) {
		t.Fatal(err)
	}
}
func TestMalformedRuntimePayloads(t *testing.T) {
	for _, e := range []appserver.Event{
		notification("turn/completed", map[string]any{"threadId": "thread", "turn": map[string]any{"id": "turn", "status": "inProgress", "items": []any{}}}),
		notification("turn/completed", map[string]any{"threadId": "thread", "turn": map[string]any{"id": "turn", "status": "completed"}}),
		deltaEvent("missing", "text"),
		itemEvent("item/completed", "cmd", "commandExecution", "", "inProgress"),
	} {
		s := reducerSession()
		if err := s.reduceLocked(e); !errors.Is(err, ErrProtocol) {
			t.Fatalf("%s: %v", e.Method, err)
		}
	}
}

// A real pipe transport is used for uncertainty tests. The peer only performs
// the initialize/thread handshake, then invokes the caller's scripted behavior.
func connectedPeer(t *testing.T, body func(net.Conn, *bufio.Reader)) (*Session, context.Context) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	t.Cleanup(cancel)
	client, server := net.Pipe()
	t.Cleanup(func() { server.Close() })
	frames, _ := fixtureFrames(t)
	go func() {
		defer server.Close()
		r := bufio.NewReader(server)
		for _, i := range []int{0, 2, 3} {
			if _, err := r.ReadBytes('\n'); err != nil {
				return
			}
			if i == 0 || i == 3 {
				index := 1
				if i == 3 {
					index = 4
				}
				b, _ := json.Marshal(frames[index])
				if _, err := server.Write(append(b, '\n')); err != nil {
					return
				}
			}
		}
		body(server, r)
	}()
	s, err := connect(ctx, client, Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s, ctx
}
func replyTurn(server net.Conn, id string) {
	fmt.Fprintf(server, "{\"id\":%q,\"result\":{\"turn\":{\"id\":\"turn\",\"status\":\"inProgress\",\"items\":[]}}}\n", id)
}
func TestStartCancellationUncertain(t *testing.T) {
	seen := make(chan struct{})
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) {
		_, _ = r.ReadBytes('\n')
		close(seen)
		_, _ = io.Copy(io.Discard, r)
	})
	callCtx, cancel := context.WithCancel(ctx)
	result := make(chan error, 1)
	go func() { result <- s.StartTurn(callCtx, "hello") }()
	<-seen
	cancel()
	if err := <-result; !errors.Is(err, ErrStartUncertain) {
		t.Fatal(err)
	}
	if err := s.StartTurn(ctx, "do not retry"); !errors.Is(err, ErrStartUncertain) {
		t.Fatal(err)
	}
	if _, err := s.Next(ctx); !errors.Is(err, ErrStartUncertain) {
		t.Fatal(err)
	}
}
func TestEOFMakesTurnIncomplete(t *testing.T) {
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) { _, _ = r.ReadBytes('\n'); replyTurn(c, "go:3") })
	if err := s.StartTurn(ctx, "hello"); err != nil {
		t.Fatal(err)
	}
	events := collectTurn(t, ctx, s)
	if events[len(events)-1].Status != "incomplete" {
		t.Fatal(events)
	}
	if _, err := s.Next(ctx); !errors.Is(err, io.EOF) {
		t.Fatal(err)
	}
}
func TestEarlyCompletionWaitsForStartIdentity(t *testing.T) {
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) {
		_, _ = r.ReadBytes('\n')
		fmt.Fprintln(c, `{"method":"turn/completed","params":{"threadId":"0199a000-0000-7000-8000-000000000001","turn":{"id":"turn","status":"completed","items":[]}}}`)
		replyTurn(c, "go:3")
		_, _ = io.Copy(io.Discard, r)
	})
	if err := s.StartTurn(ctx, "hello"); err != nil {
		t.Fatal(err)
	}
	events := collectTurn(t, ctx, s)
	if len(events) != 2 || events[0].Kind != "begin" || events[1].Status != "completed" {
		t.Fatal(events)
	}
}
func TestStartRejectAllowsNewExplicitSubmission(t *testing.T) {
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) {
		_, _ = r.ReadBytes('\n')
		fmt.Fprintln(c, `{"id":"go:3","error":{"code":-32000,"message":"private error"}}`)
		_, _ = r.ReadBytes('\n')
		replyTurn(c, "go:4")
		_, _ = io.Copy(io.Discard, r)
	})
	if err := s.StartTurn(ctx, "first"); err == nil || strings.Contains(err.Error(), "private") {
		t.Fatal(err)
	}
	if err := s.StartTurn(ctx, "second"); err != nil {
		t.Fatal(err)
	}
}
func TestPromptValidationDoesNotSend(t *testing.T) {
	s := reducerSession()
	for _, p := range []string{"", " \n", strings.Repeat("x", MaxPromptBytes+1), string([]byte{0xff})} {
		if err := s.StartTurn(context.Background(), p); err == nil {
			t.Fatal("accepted bad prompt")
		}
	}
}

func TestMain(m *testing.M) {
	if os.Getenv("OPENAI_CODEX_SESSION_FAKE") == "1" && len(os.Args) == 2 && os.Args[1] == "app-server" {
		fakeAppServer()
		os.Exit(0)
	}
	os.Exit(m.Run())
}
func fakeAppServer() {
	capture, err := os.Create(os.Getenv("OPENAI_CODEX_SESSION_CAPTURE"))
	if err != nil {
		os.Exit(8)
	}
	defer capture.Close()
	scan := bufio.NewScanner(os.Stdin)
	for scan.Scan() {
		line := append([]byte(nil), scan.Bytes()...)
		_, _ = capture.Write(append(line, '\n'))
		var call struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if json.Unmarshal(line, &call) != nil {
			os.Exit(9)
		}
		switch call.Method {
		case "initialize":
			fmt.Printf(`{"id":%s,"result":{"userAgent":"synthetic","codexHome":"/never/read","platformFamily":"unix","platformOs":"linux"}}`+"\n", call.ID)
		case "thread/start":
			fmt.Printf(`{"id":%s,"result":{"thread":{"id":"fake-thread"},"model":"fake-model","cwd":"/synthetic","approvalPolicy":"on-request","approvalsReviewer":"user","sandbox":{"type":"readOnly"}}}`+"\n", call.ID)
		case "turn/start":
			fmt.Printf(`{"id":%s,"result":{"turn":{"id":"fake-turn","status":"inProgress","items":[]}}}`+"\n", call.ID)
			fmt.Println(`{"method":"item/started","params":{"threadId":"fake-thread","turnId":"fake-turn","startedAtMs":0,"item":{"id":"fake-item","type":"agentMessage","text":""}}}`)
			fmt.Println(`{"method":"item/agentMessage/delta","params":{"threadId":"fake-thread","turnId":"fake-turn","itemId":"fake-item","delta":"synthetic streamed answer"}}`)
			fmt.Println(`{"method":"item/completed","params":{"threadId":"fake-thread","turnId":"fake-turn","completedAtMs":1,"item":{"id":"fake-item","type":"agentMessage","text":"synthetic final answer"}}}`)
			fmt.Println(`{"method":"turn/completed","params":{"threadId":"fake-thread","turn":{"id":"fake-turn","status":"completed","items":[]}}}`)
		}
	}
}
func TestOpenUsesOnlySyntheticChildAndPreservesConfiguration(t *testing.T) {
	dir := t.TempDir()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	name := "codex"
	if strings.HasSuffix(self, ".exe") {
		name += ".exe"
	}
	in, err := os.Open(self)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.OpenFile(filepath.Join(dir, name), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.Copy(out, in)
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	capture := filepath.Join(t.TempDir(), "requests.jsonl")
	t.Setenv("PATH", dir)
	t.Setenv("OPENAI_CODEX_SESSION_FAKE", "1")
	t.Setenv("OPENAI_CODEX_SESSION_CAPTURE", capture)
	t.Setenv("CODEX_HOME", t.TempDir())
	for _, name := range []string{"OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_URL"} {
		t.Setenv(name, "")
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	s, err := Open(ctx, Options{Directory: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.StartTurn(ctx, "synthetic prompt"); err != nil {
		t.Fatal(err)
	}
	events := collectTurn(t, ctx, s)
	if events[len(events)-1].Status != "completed" {
		t.Fatal(events)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	requests, err := os.ReadFile(capture)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(requests), "sandbox") || strings.Contains(string(requests), "approvalPolicy") || strings.Contains(string(requests), "experimentalApi") || strings.Contains(string(requests), "config") {
		t.Fatalf("unexpected setting override: %s", requests)
	}
	if !strings.Contains(string(requests), "synthetic prompt") || !strings.Contains(string(requests), "text_elements") {
		t.Fatal(string(requests))
	}
}

func TestUnsupportedRequestRejectedAndVisible(t *testing.T) {
	peerErr := make(chan error, 1)
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) {
		_, _ = r.ReadBytes('\n')
		replyTurn(c, "go:3")
		fmt.Fprintln(c, `{"id":"approval:1","method":"item/commandExecution/requestApproval","params":{"command":"private synthetic command"}}`)
		line, err := r.ReadBytes('\n')
		if err != nil {
			peerErr <- err
			return
		}
		var result struct {
			ID    string `json:"id"`
			Error struct {
				Code int `json:"code"`
			} `json:"error"`
		}
		if json.Unmarshal(line, &result) != nil || result.ID != "approval:1" || result.Error.Code != -32601 {
			peerErr <- fmt.Errorf("bad rejection: %s", line)
			return
		}
		peerErr <- nil
		fmt.Fprintln(c, `{"method":"turn/completed","params":{"threadId":"0199a000-0000-7000-8000-000000000001","turn":{"id":"turn","status":"failed","items":[]}}}`)
		_, _ = io.Copy(io.Discard, r)
	})
	if err := s.StartTurn(ctx, "synthetic"); err != nil {
		t.Fatal(err)
	}
	events := collectTurn(t, ctx, s)
	found := false
	for _, e := range events {
		if e.Kind == "notice" {
			found = true
			if strings.Contains(e.Notice, "private synthetic") {
				t.Fatal("raw request exposed")
			}
		}
	}
	if !found {
		t.Fatal("rejection was invisible")
	}
	if err := <-peerErr; err != nil {
		t.Fatal(err)
	}
}
func TestInterruptAcknowledgementDoesNotSetTerminalOutcome(t *testing.T) {
	ack := make(chan struct{})
	finish := make(chan struct{})
	peerErr := make(chan error, 1)
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) {
		_, _ = r.ReadBytes('\n')
		replyTurn(c, "go:3")
		line, err := r.ReadBytes('\n')
		if err != nil {
			peerErr <- err
			return
		}
		var req struct {
			Method string                            `json:"method"`
			Params struct{ ThreadID, TurnID string } `json:"params"`
		}
		if json.Unmarshal(line, &req) != nil || req.Method != "turn/interrupt" || req.Params.TurnID != "turn" {
			peerErr <- fmt.Errorf("bad interrupt: %s", line)
			return
		}
		fmt.Fprintln(c, `{"id":"go:4","result":{}}`)
		close(ack)
		<-finish
		fmt.Fprintln(c, `{"method":"turn/completed","params":{"threadId":"0199a000-0000-7000-8000-000000000001","turn":{"id":"turn","status":"completed","items":[]}}}`)
		peerErr <- nil
		_, _ = io.Copy(io.Discard, r)
	})
	if err := s.StartTurn(ctx, "synthetic"); err != nil {
		t.Fatal(err)
	}
	if err := s.Interrupt(ctx); err != nil {
		t.Fatal(err)
	}
	<-ack
	if err := s.Interrupt(ctx); err != nil {
		t.Fatal(err)
	}
	s.mu.Lock()
	active := s.active
	s.mu.Unlock()
	if !active {
		t.Fatal("ack manufactured completion")
	}
	close(finish)
	events := collectTurn(t, ctx, s)
	if events[len(events)-1].Status != "completed" {
		t.Fatal("natural completion lost interrupt race", events)
	}
	if err := <-peerErr; err != nil {
		t.Fatal(err)
	}
}
func TestInterruptCanceledDoesNotCancelTurnOrRetry(t *testing.T) {
	seen := make(chan struct{})
	finish := make(chan struct{})
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) {
		_, _ = r.ReadBytes('\n')
		replyTurn(c, "go:3")
		_, _ = r.ReadBytes('\n')
		close(seen)
		<-finish
		fmt.Fprintln(c, `{"id":"go:4","result":{}}`)
		fmt.Fprintln(c, `{"method":"turn/completed","params":{"threadId":"0199a000-0000-7000-8000-000000000001","turn":{"id":"turn","status":"interrupted","items":[]}}}`)
		_, _ = io.Copy(io.Discard, r)
	})
	if err := s.StartTurn(ctx, "synthetic"); err != nil {
		t.Fatal(err)
	}
	callCtx, cancel := context.WithCancel(ctx)
	result := make(chan error, 1)
	go func() { result <- s.Interrupt(callCtx) }()
	<-seen
	cancel()
	if err := <-result; !errors.Is(err, ErrInterruptUncertain) {
		t.Fatal(err)
	}
	if err := s.Interrupt(ctx); !errors.Is(err, ErrInterruptUncertain) {
		t.Fatal("repeated interrupt forgot uncertainty", err)
	}
	close(finish)
	events := collectTurn(t, ctx, s)
	if events[len(events)-1].Status != "interrupted" {
		t.Fatal(events)
	}
}
func TestAdapterQueueOverflowIsTerminal(t *testing.T) {
	s, ctx := connectedPeer(t, func(c net.Conn, r *bufio.Reader) {
		_, _ = r.ReadBytes('\n')
		replyTurn(c, "go:3")
		for range maxEvents + 10 {
			if _, err := fmt.Fprintln(c, `{"method":"error","params":{"message":"not rendered"}}`); err != nil {
				return
			}
		}
	})
	if err := s.StartTurn(ctx, "synthetic"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-s.done:
	case <-ctx.Done():
		t.Fatal("queue never terminated")
	}
	count := 0
	for {
		_, err := s.Next(ctx)
		if err != nil {
			if !errors.Is(err, ErrLimit) {
				t.Fatal(err)
			}
			break
		}
		count++
	}
	if count > maxEvents {
		t.Fatal("unbounded adapter queue", count)
	}
}

func TestUnsupportedVariantsDoNotDecodeOtherVariantFields(t *testing.T) {
	for _, raw := range []string{
		`{"id":"reason","type":"reasoning","summary":["summary"],"content":["opaque reasoning"]}`,
		`{"id":"future","type":"futureKind","content":{"arbitrary":true},"status":42,"command":[]}`,
		`{"id":"known","type":"agentMessage","text":"hello","content":["future extension"],"status":42}`,
	} {
		s := reducerSession()
		if err := s.itemLocked(json.RawMessage(raw), true); err != nil {
			t.Fatal(err)
		}
		e := <-s.events
		if e.Kind != "item" || e.Item.Status != "completed" {
			t.Fatal(e)
		}
	}
}

func TestFilePatchUpdatedAndMoveDestination(t *testing.T) {
	s := reducerSession()
	if err := s.itemLocked(json.RawMessage(`{"id":"file","type":"fileChange","status":"inProgress","changes":[]}`), false); err != nil {
		t.Fatal(err)
	}
	<-s.events
	patch := notification("item/fileChange/patchUpdated", map[string]any{"threadId": "thread", "turnId": "turn", "itemId": "file", "changes": []any{map[string]any{"path": "old.txt", "kind": map[string]any{"type": "update", "move_path": "new.txt"}, "diff": "+updated"}}})
	if err := s.reduceLocked(patch); err != nil {
		t.Fatal(err)
	}
	e := <-s.events
	if len(e.Item.Files) != 1 || !strings.Contains(e.Item.Files[0].Path, "new.txt") || e.Item.Diff != "+updated" {
		t.Fatal(e)
	}
	if err := s.itemLocked(json.RawMessage(`{"id":"file","type":"fileChange","status":"completed","changes":[]}`), true); err != nil {
		t.Fatal(err)
	}
	<-s.events
	if err := s.reduceLocked(patch); err != nil || len(s.events) != 0 {
		t.Fatal("terminal file resurrected", err)
	}
}

func TestCountLimitedSnapshotsAreLabeled(t *testing.T) {
	s := reducerSession()
	plan := make([]map[string]string, 65)
	for i := range plan {
		plan[i] = map[string]string{"step": "short", "status": "pending"}
	}
	if err := s.reduceLocked(notification("turn/plan/updated", map[string]any{"threadId": "thread", "turnId": "turn", "plan": plan})); err != nil {
		t.Fatal(err)
	}
	if e := <-s.events; !strings.Contains(e.Item.Text, "omitted") {
		t.Fatal(e)
	}
	changes := make([]wireFile, 65)
	if err := s.projectItemLocked(wireItem{ID: "files", Type: "fileChange", Changes: changes, Status: "completed"}, true); err != nil {
		t.Fatal(err)
	}
	if e := <-s.events; len(e.Item.Files) != 64 || !strings.Contains(e.Item.Title, "64") {
		t.Fatal(e)
	}
}
