# Fresh-thread Codex workspace runtime

This narrow adapter owns an installed `codex app-server` process and connects
one fresh thread to the workspace UI. It adds no generated full-protocol SDK,
history discovery, reconnect, credentials UI, model picker, approval controls,
or client-side tool execution. Existing `codex` exec APIs remain separate.

## Configuration and operation

`Open` resolves the working directory and looks up `codex` on PATH. There is no
shell, automatic installation, version probe, or fallback to `codex exec`.
The process receives exactly `app-server`, the selected directory as its working
directory, and the normal inherited environment. Installed Codex configuration,
authentication, permission policies, integrations, and persistence remain in
effect. This client makes no additional sandbox guarantee.

After initialization, `thread/start` supplies only `cwd` and an explicitly
selected `model` (omitted otherwise). A turn supplies only `threadId` and one
text input with `text_elements: []`. No config, sandbox, approval, instructions,
experimental capability, or persistence override is sent. The server's returned
model is a configured label, not per-turn model telemetry. Returned history paths
are never opened. This client does not record local transcripts or diagnostics.

All incoming server-request methods are unregistered. The transport rejects them
with -32601; the adapter emits a visible unsupported-request notice without
rendering request payloads. This includes approval, input, dynamic-tool,
credential, and other requests. Configured server behavior that does not request
client approval remains server-owned; rejection is not a blanket prohibition on
all server actions. Use the native Codex client for approval/input workflows.

## Ordering and completion

One reader consumes transport events. One turn is admitted at a time.
`StartTurn` locks the reducer while awaiting the response; early notifications
remain in the bounded transport queue until the authoritative turn ID is known.
Responses bypass that queue, so no event consumption is required to receive the
start reply. This deliberately favors correct identity over displaying activity
before admission. Overflow or failure produces an explicitly incomplete result.

Only matching `turn/completed` establishes completed, failed, or interrupted
status. The `turn/start` response, EOF, interrupt acknowledgment, and process exit
never establish turn success. Turn completion items can be only a last-message
summary; they are merged without replacing the existing timeline. Final item
snapshots supersede deltas. Terminal item IDs are retained for the whole turn,
including items evicted from the smaller display window.

Recognized notifications: `turn/started`, `turn/completed`, `item/started`,
`item/completed`, `item/agentMessage/delta`,
`item/commandExecution/outputDelta`, `item/fileChange/patchUpdated`,
`turn/diff/updated`, `turn/plan/updated`, and a payload-free `error` notice.
Known items are user/assistant messages, command executions, file changes,
and MCP tool names/status. MCP arguments/results are not displayed. Other item
variants get an explicit unavailable-details fallback; unrelated fields are
not decoded through another variant's schema. Other notifications are ignored.
Plan and diff updates are labeled supplied snapshots, not unfinished tool calls.

`Interrupt` sends one explicit `turn/interrupt` for the known active turn.
Repeated attempts share its result; cancellation of its local wait remains
uncertain. A normal completion can win the race. A canceled/failed dispatched
start closes this session rather than automatically retrying an unknown outcome.
A definite JSON-RPC rejection permits a new explicit user submission.

## Resource and process ownership

- Transport limits remain the independent appserver defaults.
- Adapter queue: 128 bounded events; full queue terminates, never drops silently.
- IDs: at most 512 bytes; item type: at most 128 bytes.
- Lifecycle tracking: at most 4,096 items per turn and 4,096 turns per connection.
- Snapshot/delta text and diff: 16 KiB per field, with truncation labels.
- File snapshots: at most 64 paths, 512 bytes per path, 80 bytes per kind.
- Prompt API: 64 KiB; the interactive composer uses a stricter 16 KiB cap.
- Text deltas retain a capped byte counter, not an unbounded accumulation.
- Raw error/request payloads and child stderr are never logged. Stderr is sent
  directly to the null device, retaining zero bytes. Diagnose authentication or
  configuration separately in the installed Codex client.

The owned duplex uses real pipe file descriptors. Close interrupts concurrent
reads and writes before waiting for the child. The child gets 250 ms to exit,
then is killed if necessary and always reaped. Reaping is mandatory rather than
abandoning the child after a wall-clock deadline. Stdout EOF and process exit are
separate; buffered final stdout is not discarded by `Wait`. No descendant-tree
termination guarantee is made. Local Close is not a remote interrupt handshake.

## Protocol and verification

The wire target is official Codex `rust-v0.163.0`, commit
[`83086188ae0614bd882818ff1aa23f3194962012`](https://github.com/openai/codex/tree/83086188ae0614bd882818ff1aa23f3194962012).
See `testdata/provenance.json` for exact schema/source URLs and hashes.
The conversation fixture is synthetic and covers both response/notification
orderings, two sequential turns, file/command/message streaming, and interruption.
Its 33 frames passed 69 schema checks against the pinned upstream schemas.

Tests use synthetic pipe peers and this test executable as a fake child, including
an actual `Open` process-path check. No real Codex process, model call, account,
private history, or host setting was used. This is protocol-derived synthetic
verification; authenticated real-service interoperability is untested.
