# Local task lifecycle foundation

`internal/taskrun` is an experimental, project-private state machine for a future
assistant. It is not wired into a command, an assistant service, or the Codex
wrapper. All examples and tests use synthetic observations. Importing the package
or creating a task does nothing beyond constructing an in-memory value.

## Explicit boundaries

- `New` creates a queued task with revision 1.
- `Start(expectedRevision)` records one running attempt. It does not invoke work.
- `Apply(expectedRevision, event)` records one ordered observation: progress,
  success, failure, or interruption.
- `Retry(expectedRevision)` moves only failed or interrupted tasks back to
  queued. Starting the next attempt requires a separate explicit `Start`.
- `Cancel(expectedRevision)` permanently stops a queued, running, failed, or
  interrupted task. Cancellation is idempotent. It does not undo an accepted
  success or cancel an external process.

Succeeded and cancelled tasks cannot be restarted. Failed and interrupted tasks
remain stopped until an explicit retry decision. An interruption means the
outcome is unknown; this package neither infers that no work occurred nor makes
resubmission safe. Invalid and unknown events are rejected without changing
state. There is no automatic retry or resumption path.

## Identity, ordering, and budgets

A caller supplies a nonblank task ID that is unique across its task lifecycles.
Do not reuse a task ID to reset a lifecycle: an old event could otherwise have
the same identity. Attempt IDs combine that task ID with a monotonically
increasing attempt number. `Start` advances it, allowing only one current attempt
in a state. Event sequences start at 1 and advance without gaps within an
attempt; starting another attempt resets its event sequence and progress count.

Every accepted state change increments the revision once. Callers must apply
each command to their latest state and provide its expected revision. A stale
command, mismatched attempt, conflicting event, sequence gap, or exhausted
budget leaves the input and returned state unchanged. Rejected events do not
consume a sequence number or budget.

Only the latest event receipt is retained, keeping the state constant-size:

- An exact replay of that receipt succeeds without changing state or revision,
  including when the supplied revision is old.
- That replay remains harmless after cancellation or queuing a retry.
- A changed event at the same sequence is a conflict.
- Earlier events in the same attempt are rejected as stale, even if they are
  duplicates. Events from an earlier attempt are rejected after the next start.

Once cancellation is accepted, a late completion cannot overwrite it. A success
accepted before cancellation remains a success. These are ordered transitions,
not a claim to resolve uncoordinated concurrent operations.

`Budget.MaxAttempts` limits the total number of starts. `Budget.MaxSteps` limits
progress events per attempt; zero permits no progress events. Success, failure,
interruption, and cancellation remain available at the step limit. Exhausting
progress returns an error and leaves the task running so the caller can explicitly
record an outcome or stop it. These are logical transition budgets, not token,
API, dollar, wall-clock, or external side-effect limits.

## Ownership and limits

State and snapshots are value copies with no mutable references. Pure transitions
are deterministic and do not mutate their input. The expected revision is not a
database compare-and-swap or a lock: callers must serialize state ownership and
commit accepted transitions before acting on them. Applying `Start` independently
to two copies cannot prevent two external workers from being launched. No
exactly-once execution or cross-process guarantee is provided.

There is no backend, production runner (including a fake backend), scheduler,
timer, autostart, model call, network request, tool execution, computer use,
credential store, authorization/approval policy, persistence, serialization
contract, history read, or recovery mechanism. No durable-state claim is made.
The test-only scripted runner illustrates the explicit boundary without exposing
an executable assistant. Future integration needs its own reviewed execution,
authorization, cancellation, ownership, and recovery design.

## Local checks

With the repository's Go toolchain:

```sh
go test -race ./internal/taskrun
go test ./internal/taskrun -run '^$' -fuzz '^FuzzTransitions$' -fuzztime 10s
```

The tests cover lifecycle transitions, explicit retries, stale revisions and
attempts, exact/conflicting/older replays, cancellation ordering, logical budgets,
counter limits, and generated command sequences. They require no credentials or
live services.
