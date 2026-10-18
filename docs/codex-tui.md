# Codex workspace TUI: display foundation

This document records the earlier presentation-only foundation. The integrated
`openai codex tui` command is described in [Interactive Codex workspace](codex-tui-runtime.md).

## Status and recommendation

Build the interactive workspace on a dedicated Codex app-server adapter. Keep the
existing exec adapter for non-interactive runs. They are different protocols, not
interchangeable transports for one wire schema.

This change adds only `internal/codextui`: a deterministic, display-only Bubble Tea
v2 model, a responsive item/detail view, semantic styling, keyboard navigation,
and synthetic fixtures. It adds no shipping command or backend connection. The
screen always identifies itself as a preview. No prompts, approvals, commands,
history reads, credentials, model calls, or configuration changes occur.

The visual direction is a compact workspace header, ordered work timeline, and
selected-item detail. At 100 columns the detail and timeline sit side by side;
narrower windows show the selected detail. The footer always keeps help and quit
discoverable. Activity is understated, with a static option. This is a foundation
for a complete TUI, not a claim of parity with another product.

## Existing source and protocol boundary

The inspected repository baseline is `picatz/openai` commit
`b1f212fea615b1ea88053e1005cd6e47219f173e`. The separate exec-command proposal is
[draft PR 37](https://github.com/picatz/openai/pull/37), inspected at
`6a8b820f7c20705087c311c5207e3a652f97de8a`; this slice does not modify it.

- `internal/tui/model.go` already uses Bubble Tea v2, textarea, viewport, a
  spinner, generation-based stale-event rejection, and scrollback preservation.
  Its backend is text-conversation-shaped. Codex tool items and server requests
  should not be squeezed into assistant text messages.
- `internal/tui/run.go` deliberately uses normal `program.Quit()` for context
  cancellation because the pinned Bubble Tea 2.0.10 force-kill path raced terminal
  cleanup. Preserve that lifecycle design when integrating this new model.
- `cmd/openai/styles.go` still uses Lip Gloss v1. The new package uses the already
  installed v2 dependency and does not create a cross-version theme dependency.
- The `codex` package exposes one-process-per-turn exec JSONL. Item updates are
  snapshots. It has explicit stream-drain/Wait/Close ownership, unknown-event
  preservation, and cancellation support. It cannot answer interactive requests.

The app-server reference is pinned to official Codex `rust-v0.163.0`, commit
`83086188ae0614bd882818ff1aa23f3194962012`. Its source is authoritative when current
web documentation differs. In particular, the pinned input-request type calls
`isBlocking` authoritative and deprecates `autoResolutionMs`; the input feature
is marked experimental. No timer-based approval or automatic answer is proposed.

## Capability matrix

| Capability | Current exec package / PR 37 | This UI-only slice | Proposed app-server slice |
| --- | --- | --- | --- |
| Final text and usage | Supported | Displays supplied text; usage UI deferred | Normalize turn/items and token events |
| Assistant streaming | Item snapshots when emitted | Snapshot replacement and text deltas | `item/agentMessage/delta`; final item authoritative |
| Commands and tools | Typed status/output items | Typed presentation, selected detail | Normalize item lifecycle and command output deltas |
| Changed files | Paths/kinds/status | Supplied paths | Normalize file-change items |
| Unified diff | Not supplied by exec file item | Renders only an explicitly supplied diff | `turn/diff/updated`; do not infer from file names |
| Plans | Exec todo items | Supplied plan text | Normalize `turn/plan/updated` |
| Approval/input answers | Unsupported | No controls or response API | Separate reviewed request/response implementation |
| Turn cancellation | Context/Close; direct child cleanup | Receives interruption state; no process control | `turn/interrupt`, then terminal turn notification |
| Mid-turn steering | Unsupported | No control | Separate `turn/steer`, with expected active turn ID |
| Session start/resume | Supported by exec package; explicit ID in PR 37 | No session/history access | Explicit start/resume; listing is separate work |
| Model/auth settings | Installed Codex owns them | Supplied display label only | Read effective settings; changes require separate UX |
| Resize and navigation | Existing conversation UI | Implemented with fixed-size fixtures | Reuse |
| No-color / reduced motion | No explicit policy in existing model | Explicit Config options; host env wiring deferred | Map nonempty NO_COLOR and documented motion option |
| Screen-reader / ASCII mode | Not established | No claim of full support | Separate append-only accessible mode and terminal testing |

Capabilities in this package describe available display data, never permission
to act. Unsupported content is labeled unavailable, and unknown item kinds still
have a readable fallback. The UI does not advertise unimplemented actions.

## Presentation API and state ownership

`Config` contains workspace/model labels, three display capabilities, and
NoColor/ReducedMotion options. `New` returns a Bubble Tea model. `Init` does no
work. There is no runner or I/O-bearing backend interface in this slice.

- `BeginTurnMsg` establishes an opaque thread/turn identity and strictly newer
  local generation. It clears the current-turn display; this is not history.
- `ItemMsg` replaces a full item snapshot by ID. `TextDeltaMsg` appends only to an
  existing non-terminal item. Final snapshots replace accumulated text; they are
  not appended again. Running snapshots cannot resurrect retained terminal items.
  The adapter must enforce lifecycle monotonicity for IDs already evicted from
  the bounded display window; the view does not keep unbounded tombstones.
- `Scope.Sequence` is assigned monotonically by a future adapter. It is not a
  claimed upstream sequence field. Generation, IDs, and sequence reject stale,
  duplicate, foreign-turn, and post-terminal events.
- `EndTurnMsg` accepts only the defined terminal status constants and freezes the
  turn. Unfinished items become incomplete, or
  interrupted on explicit interruption; success is never invented for them.
- `FrameMsg` changes only the cosmetic indicator. The host must schedule it only
  during activity. Plain or reduced-motion modes ignore it. View reads no clock.
- Update owns state; View only renders. No goroutine writes directly into the
  model. Transport decoding, cancellation, and persistence belong outside it.

The retained display is capped at 128 items, 16 KiB per text/diff field, and 64
file paths per item. Truncation and evicted older items are labeled. Opaque IDs
are never sanitized into potentially colliding keys; oversized IDs are rejected.
External text loses terminal control sequences and bidi formatting controls.
This is a display bound, not a durable event log or a transport message limit.

Keyboard behavior in this slice: arrows/j/k select items; Tab changes item/detail
focus; PgUp/PgDn scroll detail; Ctrl+Home/Ctrl+End go to the beginning/end; ? or F1
opens help; Esc returns to items; Ctrl+Q quits. Streaming does not steal selection
or scrollback. Scrolling to the bottom opts back into following selected detail.

## Next adapter contract and release gates

Implement the transport separately, initially with an in-memory/fake subprocess
test server. Use stdio JSON-RPC framing, one serialized writer and one reader,
bounded messages/queues, request IDs, a pending-request map, and owned shutdown.
Complete the initialize/initialized handshake before creating a thread. Default
to the stable surface; experimental input requires explicit reviewed opt-in.

Normalize exec snapshots and app-server deltas into separate adapter code; do not
rename exec events to look like app-server notifications. The interactive
production route should use app-server, with no silent fallback to exec if
approval or input capabilities are unavailable. Keep `openai codex exec` intact.

Control requests need a separate channel/state from decorative stream updates.
Never coalesce or drop an approval, error, completion, or request-resolution
event. Text deltas may be coalesced only within the same item and epoch. Resolve
pending requests by JSON-RPC request ID, not only item ID: multiple approval
callbacks can belong to one item. Handle `serverRequest/resolved` and interruption
so a late key cannot approve an obsolete request.

Before exposing approval controls, implement every relevant request shape or
fail closed with a clear unsupported-request result. Command, file-change,
network/permission, and MCP elicitation requests have different semantics. Show
the actual destination/command/path and requested scope. Persistent or
session-wide grants need distinct deliberate choices. Do not auto-select
accept, change sandbox defaults, or expose broad permission passthrough as a
shortcut. Dynamic-tool requests are not permission to execute arbitrary client
code. Secret entry, login, and account-management flows remain separate work.

For cancellation, show "interrupt requested" until the server confirms terminal
turn state. An interrupt RPC acknowledgement is not completion. Handle a lost
connection as uncertain, never success. The host owns request cancellation,
closing/draining pipes, waiting for the child, and terminal restoration. Reuse
the current normal-quit lifecycle rather than force-killing the Bubble Tea
context. Add PTY tests for Ctrl+C during a turn, repeated cancel, Ctrl+Q,
transport failure, output backpressure, and context cancellation. Descendant
process-tree termination remains unpromised unless explicitly implemented/tested.

Only after that gate should a command/composer be wired: Enter sends, Alt+Enter
and Ctrl+J add a line, bracketed paste never submits, and Ctrl+C cancels the
active turn. Add model selection and session pickers only after their data and
safe ownership semantics are implemented. Do not label a configured model as an
effective model without a server response.

Broader CLI expansion should first reuse theme, key hints, sanitized text, and
responsive layout primitives. Keep Codex work items distinct from Responses
messages, Live audio state, and image/audio artifacts. Extract a common shell
only once two implemented consumers prove the shared contract useful.

## Verification and files

Production files: `internal/codextui/{types,model,update,view,styles}.go`.
Tests: `model_test.go`, `fixtures_test.go`, and four fixed terminal golden frames.
No dependencies, existing commands, or existing session-reading code change.

Tests cover snapshot/delta precedence, terminal-state non-regression, stale and
duplicate events, cross-turn isolation, bounded retention, byte ownership,
opaque IDs, selection/scroll preservation, help/quit, absent capabilities,
Unicode/control-sequence inputs, tiny layouts, no-color, and motion suppression.
Fuzzing checks plain-frame bounds. A local View microbenchmark measures this
implementation only; it is not a comparison to another product.

Fixed 120×32, 80×24, 40×16, and plain 80×24 fixtures are rendered and inspected.
These are synthetic display frames, not screenshots of an authenticated or
live-running Codex session. No native Codex, API credentials, paid requests,
Mac settings, private history, or permission overrides are used.

## Primary references

- [Existing conversation model](https://github.com/picatz/openai/blob/b1f212fea615b1ea88053e1005cd6e47219f173e/internal/tui/model.go)
- [Existing terminal runner](https://github.com/picatz/openai/blob/b1f212fea615b1ea88053e1005cd6e47219f173e/internal/tui/run.go)
- [Exec package contract](https://github.com/picatz/openai/blob/b1f212fea615b1ea88053e1005cd6e47219f173e/codex/README.md)
- [Pinned app-server request/notification methods](https://github.com/openai/codex/blob/83086188ae0614bd882818ff1aa23f3194962012/codex-rs/app-server-protocol/src/protocol/common.rs)
- [Pinned item, delta, approval, and input types](https://github.com/openai/codex/blob/83086188ae0614bd882818ff1aa23f3194962012/codex-rs/app-server-protocol/src/protocol/v2/item.rs)
- [Pinned turn, steering, interruption, and diff types](https://github.com/openai/codex/blob/83086188ae0614bd882818ff1aa23f3194962012/codex-rs/app-server-protocol/src/protocol/v2/turn.rs)
- [Pinned server-request schema](https://github.com/openai/codex/blob/83086188ae0614bd882818ff1aa23f3194962012/codex-rs/app-server-protocol/schema/json/ServerRequest.json)
- [Pinned command approval decisions](https://github.com/openai/codex/blob/83086188ae0614bd882818ff1aa23f3194962012/codex-rs/app-server-protocol/schema/json/CommandExecutionRequestApprovalResponse.json)
- [Official terminal style guidance](https://github.com/openai/codex/blob/83086188ae0614bd882818ff1aa23f3194962012/codex-rs/tui/styles.md)
- [Current app-server overview](https://learn.chatgpt.com/docs/app-server)
