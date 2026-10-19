package codexsession

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/picatz/openai/codex/appserver"
)

type Session struct {
	client               *appserver.Client
	pipe                 io.ReadWriteCloser
	closeErr             error
	ctx                  context.Context
	cancel               context.CancelFunc
	info                 Info
	mu                   sync.Mutex
	active, interrupted  bool
	turnID               string
	generation, sequence uint64
	items                map[string]itemState
	seenTurns            map[string]bool
	interruptErr         error
	terminal             error
	events               chan Event
	done                 chan struct{}
	closeOnce            sync.Once
}
type itemState struct {
	kind      string
	terminal  bool
	bytes     int
	truncated bool
}

// Open launches the installed Codex app-server. It never installs a binary or
// changes configuration. The caller owns Close even after a completed turn.
func Open(ctx context.Context, options Options) (*Session, error) {
	if ctx == nil {
		return nil, errors.New("Codex workspace requires a context")
	}
	dir, err := filepath.Abs(options.Directory)
	if err != nil {
		return nil, errors.New("cannot resolve Codex working directory")
	}
	options.Directory = dir
	pipe, err := startProcess(ctx, dir)
	if err != nil {
		return nil, err
	}
	return connect(ctx, pipe, options)
}

// connect owns pipe on every path. Kept private so production cannot supply
// arbitrary transports or enable unreviewed request handlers through this API.
func connect(ctx context.Context, pipe io.ReadWriteCloser, options Options) (*Session, error) {
	if ctx == nil {
		_ = pipe.Close()
		return nil, errors.New("Codex workspace requires a context")
	}
	ctx, cancel := context.WithCancel(ctx)
	client, err := appserver.Connect(ctx, pipe, appserver.Options{ClientInfo: appserver.ClientInfo{Name: "picatz_openai", Title: "OpenAI Codex workspace", Version: "0.1.0"}})
	if err != nil {
		cancel()
		return nil, fmt.Errorf("initialize Codex app-server: %w", err)
	}
	fail := func(err error) (*Session, error) { cancel(); _ = client.Close(); return nil, err }
	params, _ := json.Marshal(struct {
		CWD   string `json:"cwd,omitempty"`
		Model string `json:"model,omitempty"`
	}{options.Directory, options.Model})
	callCtx, stop := context.WithTimeout(ctx, 10*time.Second)
	raw, err := client.Call(callCtx, "thread/start", params)
	stop()
	if err != nil {
		return fail(fmt.Errorf("start fresh Codex thread: %w", err))
	}
	info, err := parseInfo(raw)
	if err != nil {
		return fail(err)
	}
	s := &Session{client: client, pipe: pipe, ctx: ctx, cancel: cancel, info: info, events: make(chan Event, maxEvents), done: make(chan struct{})}
	go s.pump()
	return s, nil
}

func (s *Session) Info() Info { return s.info }

// StartTurn sends exactly one text input. A failed dispatched start has an
// uncertain outcome: the session is closed rather than automatically retried.
// Holding the reducer lock across Call deliberately buffers early notifications
// in the bounded transport until its authoritative turn ID is known. Response
// correlation never depends on the event consumer, so this cannot deadlock it.
func (s *Session) StartTurn(ctx context.Context, prompt string) error {
	if ctx == nil {
		return errors.New("Codex turn requires a context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if strings.TrimSpace(prompt) == "" || len(prompt) > MaxPromptBytes || !utf8.ValidString(prompt) {
		return errors.New("Codex prompt must be valid text between 1 byte and 64 KiB")
	}
	s.mu.Lock()
	if s.terminal != nil {
		err := s.terminal
		s.mu.Unlock()
		return err
	}
	if s.active {
		s.mu.Unlock()
		return ErrBusy
	}
	params, _ := json.Marshal(struct {
		ThreadID string      `json:"threadId"`
		Input    []textInput `json:"input"`
	}{s.info.ThreadID, []textInput{{Type: "text", Text: prompt, TextElements: []any{}}}})
	raw, err := s.client.Call(ctx, "turn/start", params)
	if err != nil {
		// A JSON-RPC error is a definite rejected operation; it can be retried
		// only by a new user submission. Cancellation/transport loss is uncertain.
		var rpc *appserver.RPCError
		if errors.As(err, &rpc) {
			s.mu.Unlock()
			return fmt.Errorf("Codex rejected turn start: %w", err)
		}
		s.stopLocked(ErrStartUncertain)
		s.mu.Unlock()
		s.cancel()
		return ErrStartUncertain
	}
	var response struct {
		Turn wireTurn `json:"turn"`
	}
	if json.Unmarshal(raw, &response) != nil || !validID(response.Turn.ID) || !validTurnStatus(response.Turn.Status) || response.Turn.Items == nil {
		s.stopLocked(ErrProtocol)
		s.mu.Unlock()
		s.cancel()
		return ErrProtocol
	}
	s.generation++
	if s.seenTurns == nil {
		s.seenTurns = make(map[string]bool)
	}
	if s.seenTurns[response.Turn.ID] {
		s.stopLocked(ErrProtocol)
		s.mu.Unlock()
		s.cancel()
		return ErrProtocol
	}
	if len(s.seenTurns) >= 4096 {
		s.stopLocked(ErrLimit)
		s.mu.Unlock()
		s.cancel()
		return ErrLimit
	}
	s.seenTurns[response.Turn.ID] = true
	if s.generation == 0 {
		s.stopLocked(ErrLimit)
		s.mu.Unlock()
		s.cancel()
		return ErrLimit
	}
	s.sequence = 0
	s.turnID = response.Turn.ID
	s.active = true
	s.interrupted = false
	s.interruptErr = nil
	s.items = make(map[string]itemState)
	err = s.emitLocked(Event{Kind: "begin"})
	// A response is an admission result, never turn completion. Only a matching
	// turn/completed notification can establish a terminal outcome.
	if err != nil {
		s.stopLocked(err)
	}
	s.mu.Unlock()
	if err != nil {
		s.cancel()
	}
	return err
}

type textInput struct {
	Type         string `json:"type"`
	Text         string `json:"text"`
	TextElements []any  `json:"text_elements"`
}

// Interrupt sends turn/interrupt for the current known turn only. Success is
// an acknowledgement, not proof of interruption; wait for turn/completed.
func (s *Session) Interrupt(ctx context.Context) error {
	if ctx == nil {
		return errors.New("Codex interruption requires a context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.terminal != nil {
		return s.terminal
	}
	if !s.active {
		return ErrNoTurn
	}
	if s.interrupted {
		return s.interruptErr
	}
	s.interrupted = true
	params, _ := json.Marshal(struct {
		ThreadID string `json:"threadId"`
		TurnID   string `json:"turnId"`
	}{s.info.ThreadID, s.turnID})
	raw, err := s.client.Call(ctx, "turn/interrupt", params)
	if err != nil {
		s.interruptErr = ErrInterruptUncertain
		return s.interruptErr
	}
	var result map[string]json.RawMessage
	if json.Unmarshal(raw, &result) != nil || result == nil {
		s.interruptErr = ErrInterruptUncertain
		return s.interruptErr
	}
	s.interrupted = true
	return nil
}

// Next must have a single consumer. Accepted events drain before the terminal
// error. Queue exhaustion is explicit, never silently dropped.
func (s *Session) Next(ctx context.Context) (Event, error) {
	if ctx == nil {
		return Event{}, errors.New("Codex event read requires a context")
	}
	select {
	case e := <-s.events:
		return e, nil
	default:
	}
	select {
	case e := <-s.events:
		return e, nil
	case <-s.done:
		select {
		case e := <-s.events:
			return e, nil
		default:
		}
		s.mu.Lock()
		err := s.terminal
		s.mu.Unlock()
		return Event{}, err
	case <-ctx.Done():
		return Event{}, ctx.Err()
	}
}

// Close closes the transport and joins its reader and owned child. It does not
// send a remote interrupt or establish that a model turn completed/canceled.
func (s *Session) Close() error {
	s.closeOnce.Do(func() {
		s.cancel()
		s.closeErr = s.client.Close()
		<-s.done
		s.closeErr = errors.Join(s.closeErr, s.pipe.Close())
	})
	return s.closeErr
}

func (s *Session) pump() {
	defer close(s.done)
	for {
		e, err := s.client.Next(s.ctx)
		if err != nil {
			s.mu.Lock()
			if s.terminal == nil {
				if errors.Is(err, context.Canceled) {
					err = ErrClosed
				}
				s.stopLocked(err)
			}
			s.mu.Unlock()
			s.cancel()
			_ = s.client.Close()
			return
		}
		s.mu.Lock()
		if s.terminal != nil {
			s.mu.Unlock()
			s.cancel()
			_ = s.client.Close()
			return
		}
		err = s.reduceLocked(e)
		if err != nil {
			s.stopLocked(err)
		}
		s.mu.Unlock()
		if err != nil {
			s.cancel()
			_ = s.client.Close()
			return
		}
	}
}

func (s *Session) stopLocked(err error) {
	if s.terminal != nil {
		return
	}
	s.terminal = err
	if s.active {
		_ = s.emitLocked(Event{Kind: "end", Status: "incomplete"})
		s.active = false
	}
}
func (s *Session) emitLocked(e Event) error {
	s.sequence++
	e.ThreadID = s.info.ThreadID
	e.TurnID = s.turnID
	e.Generation = s.generation
	e.Sequence = s.sequence
	select {
	case s.events <- e:
		return nil
	default:
		return ErrLimit
	}
}
