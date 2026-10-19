// Package taskrun defines a deterministic, in-memory task lifecycle for local
// experiments. It performs no work and is not connected to a CLI or backend.
//
// States are immutable values. A caller must explicitly start each attempt,
// supply events, and authorize any retry. Revision checks reject stale commands
// against the state passed in; they are not a store or a concurrency lock.
package taskrun

import (
	"errors"
	"strings"
)

// Status describes the task, rather than an external worker or process.
type Status string

const (
	Queued      Status = "queued"
	Running     Status = "running"
	Succeeded   Status = "succeeded"
	Failed      Status = "failed"
	Cancelled   Status = "cancelled"
	Interrupted Status = "interrupted"
)

// Budget bounds logical transitions. It does not bound tokens, money, wall time,
// or effects in any external system. Zero MaxSteps permits terminal events only.
type Budget struct {
	MaxAttempts uint32
	MaxSteps    uint32 // Maximum progress events in each attempt.
}

// AttemptID identifies an attempt within one task's lifecycle. Task IDs must be
// unique within the caller's scope; reusing an ID for a new lifecycle is unsafe.
type AttemptID struct {
	TaskID string
	Number uint32
}

type EventKind string

const (
	EventProgress    EventKind = "progress"
	EventSucceeded   EventKind = "succeeded"
	EventFailed      EventKind = "failed"
	EventInterrupted EventKind = "interrupted"
)

// Event is a synthetic observation, not a request to execute anything. Sequence
// starts at one in each attempt and must advance without gaps.
type Event struct {
	Attempt  AttemptID
	Sequence uint64
	Kind     EventKind
}

// Snapshot is a read-only copy of a State's public data. When queued for a retry,
// Attempt, EventSequence, and Steps still describe the previous attempt. Start
// advances the attempt number and resets its counters.
type Snapshot struct {
	TaskID        string
	Status        Status
	Revision      uint64
	Attempt       AttemptID
	EventSequence uint64
	Steps         uint32
	Budget        Budget
}

// State is a constant-size immutable value. Its zero value cannot be used; call
// New. No clock, event history, persistence, runner, or external handle is kept.
type State struct {
	snapshot Snapshot
	last     Event
}

var (
	ErrInvalidTask       = errors.New("taskrun: task ID must not be blank")
	ErrInvalidBudget     = errors.New("taskrun: max attempts must be positive")
	ErrUninitialized     = errors.New("taskrun: state is not initialized")
	ErrRevision          = errors.New("taskrun: stale revision")
	ErrRevisionExhausted = errors.New("taskrun: revision exhausted")
	ErrTransition        = errors.New("taskrun: transition is not allowed")
	ErrAttempt           = errors.New("taskrun: event belongs to a different attempt")
	ErrEvent             = errors.New("taskrun: invalid event")
	ErrSequence          = errors.New("taskrun: stale or out-of-order event sequence")
	ErrConflictingEvent  = errors.New("taskrun: event conflicts with the latest receipt")
	ErrBudget            = errors.New("taskrun: logical budget exhausted")
)

// New creates an idle task. Creating or copying a state never starts work.
func New(taskID string, budget Budget) (State, error) {
	if strings.TrimSpace(taskID) == "" {
		return State{}, ErrInvalidTask
	}
	if budget.MaxAttempts == 0 {
		return State{}, ErrInvalidBudget
	}
	return State{snapshot: Snapshot{
		TaskID: taskID, Status: Queued, Revision: 1, Budget: budget,
	}}, nil
}

func (s State) Snapshot() Snapshot { return s.snapshot }

// Start marks exactly one explicitly queued attempt as running. It does not
// invoke a runner. Failed or interrupted tasks must first be explicitly retried.
func (s State) Start(expectedRevision uint64) (State, error) {
	if err := s.checkRevision(expectedRevision); err != nil {
		return s, err
	}
	if s.snapshot.Status != Queued {
		return s, ErrTransition
	}
	if s.snapshot.Attempt.Number >= s.snapshot.Budget.MaxAttempts {
		return s, ErrBudget
	}
	next := s
	next.snapshot.Status = Running
	next.snapshot.Revision++
	next.snapshot.Attempt = AttemptID{s.snapshot.TaskID, s.snapshot.Attempt.Number + 1}
	next.snapshot.EventSequence = 0
	next.snapshot.Steps = 0
	next.last = Event{}
	return next, nil
}

// Retry queues a stopped attempt only after an explicit caller decision. It does
// not start work or prove that repeating external effects would be safe. An
// interrupted attempt has an unknown outcome and is never retried automatically.
func (s State) Retry(expectedRevision uint64) (State, error) {
	if err := s.checkRevision(expectedRevision); err != nil {
		return s, err
	}
	if s.snapshot.Status != Failed && s.snapshot.Status != Interrupted {
		return s, ErrTransition
	}
	if s.snapshot.Attempt.Number >= s.snapshot.Budget.MaxAttempts {
		return s, ErrBudget
	}
	next := s
	next.snapshot.Status = Queued
	next.snapshot.Revision++
	return next, nil
}

// Cancel permanently stops this lifecycle. It can also close a failed or
// interrupted task to prevent retry. An accepted cancellation cannot be undone
// by a late event; it does not cancel or roll back an external operation.
// Repeating cancellation is an unchanged success, even with an old revision.
func (s State) Cancel(expectedRevision uint64) (State, error) {
	if s.snapshot.Status == Cancelled {
		return s, nil
	}
	if err := s.checkRevision(expectedRevision); err != nil {
		return s, err
	}
	if s.snapshot.Status == Succeeded {
		return s, ErrTransition
	}
	next := s
	next.snapshot.Status = Cancelled
	next.snapshot.Revision++
	return next, nil
}

// Apply accepts one ordered event for the current running attempt. Terminal
// events remain available when the progress budget is exhausted.
//
// An exact replay of the latest accepted event is an unchanged success, even
// after cancellation/retry or with an old revision. A different event at the
// same sequence conflicts; older events and events from previous attempts are
// rejected without mutation. Only the latest receipt is retained.
func (s State) Apply(expectedRevision uint64, event Event) (State, error) {
	if s.snapshot.Revision == 0 {
		return s, ErrUninitialized
	}
	if event.Sequence == 0 || !event.Kind.valid() {
		return s, ErrEvent
	}
	if event == s.last {
		return s, nil
	}
	if event.Attempt != s.snapshot.Attempt || event.Attempt.Number == 0 {
		return s, ErrAttempt
	}
	if event.Sequence == s.snapshot.EventSequence {
		return s, ErrConflictingEvent
	}
	if event.Sequence != s.snapshot.EventSequence+1 {
		return s, ErrSequence
	}
	if err := s.checkRevision(expectedRevision); err != nil {
		return s, err
	}
	if s.snapshot.Status != Running {
		return s, ErrTransition
	}
	next := s
	switch event.Kind {
	case EventProgress:
		if s.snapshot.Steps >= s.snapshot.Budget.MaxSteps {
			return s, ErrBudget
		}
		next.snapshot.Steps++
	case EventSucceeded:
		next.snapshot.Status = Succeeded
	case EventFailed:
		next.snapshot.Status = Failed
	case EventInterrupted:
		next.snapshot.Status = Interrupted
	}
	next.snapshot.Revision++
	next.snapshot.EventSequence = event.Sequence
	next.last = event
	return next, nil
}

func (s State) checkRevision(expected uint64) error {
	if s.snapshot.Revision == 0 {
		return ErrUninitialized
	}
	if expected != s.snapshot.Revision {
		return ErrRevision
	}
	if s.snapshot.Revision == ^uint64(0) {
		return ErrRevisionExhausted
	}
	return nil
}

func (k EventKind) valid() bool {
	switch k {
	case EventProgress, EventSucceeded, EventFailed, EventInterrupted:
		return true
	default:
		return false
	}
}
