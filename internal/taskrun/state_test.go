package taskrun

import (
	"errors"
	"testing"
)

func newTask(t *testing.T, attempts, steps uint32) State {
	t.Helper()
	s, err := New("task-1", Budget{MaxAttempts: attempts, MaxSteps: steps})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func started(t *testing.T, attempts, steps uint32) State {
	t.Helper()
	s := newTask(t, attempts, steps)
	next, err := s.Start(s.Snapshot().Revision)
	if err != nil {
		t.Fatal(err)
	}
	return next
}

func eventFor(s State, kind EventKind) Event {
	v := s.Snapshot()
	return Event{Attempt: v.Attempt, Sequence: v.EventSequence + 1, Kind: kind}
}

func apply(t *testing.T, s State, kind EventKind) State {
	t.Helper()
	next, err := s.Apply(s.Snapshot().Revision, eventFor(s, kind))
	if err != nil {
		t.Fatal(err)
	}
	return next
}

func requireUnchanged(t *testing.T, before, after State, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
	if after != before {
		t.Fatalf("rejected/no-op transition changed state: before=%+v after=%+v", before, after)
	}
}

func TestNew(t *testing.T) {
	for _, tc := range []struct {
		name   string
		id     string
		budget Budget
		want   error
	}{
		{"empty ID", "", Budget{1, 1}, ErrInvalidTask},
		{"blank ID", " \n\t", Budget{1, 1}, ErrInvalidTask},
		{"no attempts", "task", Budget{}, ErrInvalidBudget},
		{"terminal only", "task", Budget{1, 0}, nil},
		{"bounded progress", "task", Budget{2, 3}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, err := New(tc.id, tc.budget)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
			if err != nil {
				if s != (State{}) {
					t.Fatal("invalid creation returned usable state")
				}
				return
			}
			want := Snapshot{TaskID: tc.id, Status: Queued, Revision: 1, Budget: tc.budget}
			if s.Snapshot() != want {
				t.Fatalf("snapshot = %+v, want %+v", s.Snapshot(), want)
			}
		})
	}
}

func TestZeroStateRejected(t *testing.T) {
	var s State
	next, err := s.Start(0)
	requireUnchanged(t, s, next, err, ErrUninitialized)
	next, err = s.Retry(0)
	requireUnchanged(t, s, next, err, ErrUninitialized)
	next, err = s.Cancel(0)
	requireUnchanged(t, s, next, err, ErrUninitialized)
	next, err = s.Apply(0, Event{})
	requireUnchanged(t, s, next, err, ErrUninitialized)
}

func TestExplicitLifecycle(t *testing.T) {
	queued := newTask(t, 2, 3)
	running, err := queued.Start(queued.Snapshot().Revision)
	if err != nil {
		t.Fatal(err)
	}
	if queued.Snapshot().Status != Queued {
		t.Fatal("Start mutated the old snapshot")
	}
	v := running.Snapshot()
	if v.Status != Running || v.Revision != 2 || v.Attempt != (AttemptID{"task-1", 1}) {
		t.Fatalf("unexpected start: %+v", v)
	}
	progress := apply(t, running, EventProgress)
	if progress.Snapshot().Steps != 1 || progress.Snapshot().EventSequence != 1 {
		t.Fatalf("unexpected progress: %+v", progress.Snapshot())
	}
	finished := apply(t, progress, EventSucceeded)
	v = finished.Snapshot()
	if v.Status != Succeeded || v.Revision != 4 || v.EventSequence != 2 || v.Steps != 1 {
		t.Fatalf("unexpected completion: %+v", v)
	}
	copy := finished.Snapshot()
	copy.Status = Running
	copy.Budget.MaxAttempts++
	if finished.Snapshot() == copy {
		t.Fatal("mutating the public snapshot mutated state")
	}
}

func TestCommandTransitionTable(t *testing.T) {
	running := started(t, 3, 2)
	failed := apply(t, running, EventFailed)
	interrupted := apply(t, running, EventInterrupted)
	succeeded := apply(t, running, EventSucceeded)
	cancelled, err := running.Cancel(running.Snapshot().Revision)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		state                State
		start, retry, cancel error
	}{
		{newTask(t, 3, 2), nil, ErrTransition, nil},
		{running, ErrTransition, ErrTransition, nil},
		{failed, ErrTransition, nil, nil},
		{interrupted, ErrTransition, nil, nil},
		{succeeded, ErrTransition, ErrTransition, ErrTransition},
		{cancelled, ErrTransition, ErrTransition, nil},
	} {
		t.Run(string(tc.state.Snapshot().Status), func(t *testing.T) {
			for _, cmd := range []struct {
				name string
				fn   func(uint64) (State, error)
				want error
			}{
				{"start", tc.state.Start, tc.start},
				{"retry", tc.state.Retry, tc.retry},
				{"cancel", tc.state.Cancel, tc.cancel},
			} {
				t.Run(cmd.name, func(t *testing.T) {
					next, err := cmd.fn(tc.state.Snapshot().Revision)
					if !errors.Is(err, cmd.want) {
						t.Fatalf("error = %v, want %v", err, cmd.want)
					}
					if err != nil && next != tc.state {
						t.Fatal("rejected command mutated state")
					}
				})
			}
		})
	}
}

func TestExplicitRetryAndAttemptIsolation(t *testing.T) {
	for _, outcome := range []EventKind{EventFailed, EventInterrupted} {
		t.Run(string(outcome), func(t *testing.T) {
			running := started(t, 2, 2)
			oldEvent := eventFor(running, outcome)
			stopped := apply(t, running, outcome)
			next, err := stopped.Start(stopped.Snapshot().Revision)
			requireUnchanged(t, stopped, next, err, ErrTransition)
			queued, err := stopped.Retry(stopped.Snapshot().Revision)
			if err != nil || queued.Snapshot().Status != Queued {
				t.Fatalf("retry = %+v, %v", queued, err)
			}
			// A duplicate failure cannot undo the explicit queue decision.
			next, err = queued.Apply(running.Snapshot().Revision, oldEvent)
			requireUnchanged(t, queued, next, err, nil)
			running2, err := queued.Start(queued.Snapshot().Revision)
			if err != nil {
				t.Fatal(err)
			}
			v := running2.Snapshot()
			if v.Attempt.Number != 2 || v.Steps != 0 || v.EventSequence != 0 {
				t.Fatalf("retry counters = %+v", v)
			}
			next, err = running2.Apply(v.Revision, oldEvent)
			requireUnchanged(t, running2, next, err, ErrAttempt)
			oldEvent.Kind = EventSucceeded
			next, err = running2.Apply(v.Revision, oldEvent)
			requireUnchanged(t, running2, next, err, ErrAttempt)
			failed2 := apply(t, running2, EventFailed)
			next, err = failed2.Retry(failed2.Snapshot().Revision)
			requireUnchanged(t, failed2, next, err, ErrBudget)
		})
	}
}

func TestEventOrderingAndReplay(t *testing.T) {
	running := started(t, 2, 4)
	first := eventFor(running, EventProgress)
	progress := apply(t, running, EventProgress)
	next, err := progress.Apply(running.Snapshot().Revision, first)
	requireUnchanged(t, progress, next, err, nil)
	conflict := first
	conflict.Kind = EventSucceeded
	next, err = progress.Apply(progress.Snapshot().Revision, conflict)
	requireUnchanged(t, progress, next, err, ErrConflictingEvent)
	second := apply(t, progress, EventProgress)
	next, err = second.Apply(second.Snapshot().Revision, first)
	requireUnchanged(t, second, next, err, ErrSequence)
	for _, tc := range []struct {
		name  string
		event Event
		want  error
	}{
		{"sequence gap", Event{second.Snapshot().Attempt, 4, EventSucceeded}, ErrSequence},
		{"zero sequence", Event{second.Snapshot().Attempt, 0, EventSucceeded}, ErrEvent},
		{"unknown kind", Event{second.Snapshot().Attempt, 3, "unknown"}, ErrEvent},
		{"different task", Event{AttemptID{"other-task", 1}, 3, EventSucceeded}, ErrAttempt},
		{"future attempt", Event{AttemptID{"task-1", 2}, 3, EventSucceeded}, ErrAttempt},
		{"zero attempt", Event{AttemptID{"task-1", 0}, 3, EventSucceeded}, ErrAttempt},
	} {
		t.Run(tc.name, func(t *testing.T) {
			next, err := second.Apply(second.Snapshot().Revision, tc.event)
			requireUnchanged(t, second, next, err, tc.want)
		})
	}
	finished := apply(t, second, EventSucceeded)
	next, err = finished.Apply(second.Snapshot().Revision, eventFor(second, EventSucceeded))
	requireUnchanged(t, finished, next, err, nil)
	next, err = finished.Apply(finished.Snapshot().Revision, eventFor(finished, EventFailed))
	requireUnchanged(t, finished, next, err, ErrTransition)
}

func TestCancellationOrdering(t *testing.T) {
	for _, kind := range []EventKind{EventProgress, EventSucceeded, EventFailed, EventInterrupted} {
		t.Run("cancel before "+string(kind), func(t *testing.T) {
			running := started(t, 3, 2)
			late := eventFor(running, kind)
			cancelled, err := running.Cancel(running.Snapshot().Revision)
			if err != nil {
				t.Fatal(err)
			}
			next, err := cancelled.Apply(running.Snapshot().Revision, late)
			requireUnchanged(t, cancelled, next, err, ErrRevision)
			next, err = cancelled.Apply(cancelled.Snapshot().Revision, late)
			requireUnchanged(t, cancelled, next, err, ErrTransition)
			next, err = cancelled.Cancel(0)
			requireUnchanged(t, cancelled, next, err, nil)
		})
	}
	running := started(t, 2, 2)
	progress := apply(t, running, EventProgress)
	cancelled, err := progress.Cancel(progress.Snapshot().Revision)
	if err != nil {
		t.Fatal(err)
	}
	next, err := cancelled.Apply(running.Snapshot().Revision, eventFor(running, EventProgress))
	requireUnchanged(t, cancelled, next, err, nil)
	// Cancellation does not retroactively undo an already accepted success.
	succeeded := apply(t, running, EventSucceeded)
	next, err = succeeded.Cancel(succeeded.Snapshot().Revision)
	requireUnchanged(t, succeeded, next, err, ErrTransition)
}

func TestBudgets(t *testing.T) {
	for _, limit := range []uint32{0, 1, 4} {
		for _, terminal := range []EventKind{EventSucceeded, EventFailed, EventInterrupted} {
			s := started(t, 1, limit)
			for range limit {
				s = apply(t, s, EventProgress)
			}
			next, err := s.Apply(s.Snapshot().Revision, eventFor(s, EventProgress))
			requireUnchanged(t, s, next, err, ErrBudget)
			finished := apply(t, s, terminal)
			if finished.Snapshot().Steps != limit || finished.Snapshot().EventSequence != uint64(limit)+1 {
				t.Fatalf("terminal event consumed progress budget: %+v", finished.Snapshot())
			}
		}
	}
	// Cancellation remains available even if no further progress is allowed.
	s := started(t, 1, 0)
	if next, err := s.Cancel(s.Snapshot().Revision); err != nil || next.Snapshot().Status != Cancelled {
		t.Fatalf("cancel at budget limit = %+v, %v", next, err)
	}
}

func TestRevisionConflicts(t *testing.T) {
	s := started(t, 3, 3)
	updated := apply(t, s, EventProgress)
	for _, revision := range []uint64{0, s.Snapshot().Revision, updated.Snapshot().Revision + 1} {
		next, err := updated.Apply(revision, eventFor(updated, EventSucceeded))
		requireUnchanged(t, updated, next, err, ErrRevision)
		next, err = updated.Cancel(revision)
		requireUnchanged(t, updated, next, err, ErrRevision)
		next, err = updated.Start(revision)
		requireUnchanged(t, updated, next, err, ErrRevision)
		next, err = updated.Retry(revision)
		requireUnchanged(t, updated, next, err, ErrRevision)
	}
	// A caller serializes state ownership: after accepting one command, the
	// competing command must be applied to the updated state to detect staleness.
	failed := apply(t, s, EventFailed)
	next, err := failed.Cancel(s.Snapshot().Revision)
	requireUnchanged(t, failed, next, err, ErrRevision)
}

func TestCounterLimitsDoNotWrap(t *testing.T) {
	s := started(t, ^uint32(0), ^uint32(0))
	s.snapshot.Steps = ^uint32(0)
	s.snapshot.EventSequence = uint64(s.snapshot.Steps)
	next, err := s.Apply(s.Snapshot().Revision, eventFor(s, EventProgress))
	requireUnchanged(t, s, next, err, ErrBudget)
	completed := apply(t, s, EventFailed)
	if completed.Snapshot().EventSequence != uint64(^uint32(0))+1 {
		t.Fatal("event sequence wrapped")
	}
	completed.snapshot.Attempt.Number = ^uint32(0)
	next, err = completed.Retry(completed.Snapshot().Revision)
	requireUnchanged(t, completed, next, err, ErrBudget)
	completed.snapshot.Status = Queued
	next, err = completed.Start(completed.Snapshot().Revision)
	requireUnchanged(t, completed, next, err, ErrBudget)
	s.snapshot.Revision = ^uint64(0)
	for _, cmd := range []func(uint64) (State, error){s.Start, s.Retry, s.Cancel} {
		next, err = cmd(s.Snapshot().Revision)
		requireUnchanged(t, s, next, err, ErrRevisionExhausted)
	}
	next, err = s.Apply(s.Snapshot().Revision, eventFor(s, EventSucceeded))
	requireUnchanged(t, s, next, err, ErrRevisionExhausted)
}

// The only scripted execution is this test helper. Production code has no
// runner interface, dispatch loop, or backend, including a fake backend.
func TestScriptedRunOnceRequiresExplicitStart(t *testing.T) {
	var calls int
	runOnce := func(s State) (State, error) {
		if s.Snapshot().Status != Running {
			return s, ErrTransition
		}
		calls++
		return s.Apply(s.Snapshot().Revision, eventFor(s, EventInterrupted))
	}
	queued := newTask(t, 2, 1)
	next, err := runOnce(queued)
	requireUnchanged(t, queued, next, err, ErrTransition)
	if calls != 0 {
		t.Fatal("queued task ran without Start")
	}
	running, err := queued.Start(queued.Snapshot().Revision)
	if err != nil {
		t.Fatal(err)
	}
	interrupted, err := runOnce(running)
	if err != nil || calls != 1 {
		t.Fatalf("run once = %+v, %v, calls %d", interrupted, err, calls)
	}
	next, err = runOnce(interrupted)
	requireUnchanged(t, interrupted, next, err, ErrTransition)
	if calls != 1 {
		t.Fatal("interrupted attempt was automatically resubmitted")
	}
}

func FuzzTransitions(f *testing.F) {
	f.Add([]byte{0, 3, 4, 2})
	f.Add([]byte{0, 6, 1, 0, 5, 1, 0, 4})
	f.Add([]byte{0, 3, 7, 2, 7, 4, 1, 0})
	f.Fuzz(func(t *testing.T, commands []byte) {
		if len(commands) > 512 {
			commands = commands[:512]
		}
		s := newTask(t, 3, 4)
		var lastAccepted Event
		for _, command := range commands {
			before := s
			v := s.Snapshot()
			revision := v.Revision
			if command&0x80 != 0 {
				revision--
			}
			event := eventFor(s, EventProgress)
			operation := func() (State, error) {
				switch command % 12 {
				case 0:
					return s.Start(revision)
				case 1:
					return s.Retry(revision)
				case 2:
					return s.Cancel(revision)
				case 3:
					event.Kind = EventProgress
				case 4:
					event.Kind = EventSucceeded
				case 5:
					event.Kind = EventFailed
				case 6:
					event.Kind = EventInterrupted
				case 7:
					event = lastAccepted
				case 8:
					event.Sequence = 0
				case 9:
					event.Sequence = v.EventSequence + 2
				case 10:
					event.Attempt.TaskID = "wrong-task"
				case 11:
					event.Kind = "unknown"
				}
				return s.Apply(revision, event)
			}
			next, err := operation()
			again, againErr := operation()
			if next != again || err != againErr || s != before {
				t.Fatal("transition is not pure and deterministic")
			}
			if err != nil && next != before {
				t.Fatal("rejected transition mutated state")
			}
			if next != before && next.Snapshot().Revision != v.Revision+1 {
				t.Fatal("revision did not advance exactly once")
			}
			if (v.Status == Cancelled || v.Status == Succeeded) && next != before {
				t.Fatal("terminal state changed")
			}
			nv := next.Snapshot()
			if nv.Attempt.Number < v.Attempt.Number || nv.Attempt.Number > v.Attempt.Number+1 {
				t.Fatal("invalid attempt advancement")
			}
			if nv.Attempt.Number > nv.Budget.MaxAttempts || nv.Steps > nv.Budget.MaxSteps {
				t.Fatal("logical budget exceeded")
			}
			if nv.EventSequence > uint64(nv.Steps)+1 {
				t.Fatal("event sequence skipped a step")
			}
			if nv.Status == Running && nv.EventSequence != uint64(nv.Steps) {
				t.Fatal("running attempt has a terminal event")
			}
			if next.last != (Event{}) {
				lastAccepted = next.last
			}
			s = next
		}
	})
}
