# Codex app-server pipe transport

This package is a transport foundation, not an interactive Codex application.
It accepts an already supplied `io.ReadWriteCloser`. It does not launch Codex,
dial sockets, load credentials, inspect the server's `codexHome`, read history,
run tools, or change the existing exec SDK, CLI, or TUI. Tests use synthetic
`net.Pipe` and paired `io.Pipe` peers only; real-service interoperability has not
been tested.

## Connection and API

`Connect(ctx, pipe, Options{ClientInfo: ClientInfo{Name: "my_client", Version: "1"}})`
takes ownership of the pipe immediately, including cleanup on failure. Its
`Close` must be concurrency-safe and interrupt blocked reads and writes. The
connection context lives for the entire connection. Initialization has its own
10-second default timeout.

Connect sends `initialize`, validates its metadata result, and writes
`initialized` before returning a client. This is not protocol-version negotiation.
No optional capability is advertised by default. `ExperimentalAPI` and exact
`OptOutNotificationMethods` are explicit caller choices; neither implies a
server capability acknowledgment. The initialize request is enqueued before ingress can enqueue any reply.
Early notifications and registered requests are retained under normal bounds;
unregistered requests receive -32601 even during initialization. Application
calls remain unavailable until initialized has been written.

- `Initialization()` returns informational metadata.
- `Call(ctx, method, json.RawMessage)` correlates one raw result or `*RPCError`.
  Nil parameters are omitted; `json.RawMessage("null")` sends JSON null.
  `initialize` and `initialized` are reserved. There are no automatic retries.
- `Next(ctx)` serves a single ordered event consumer. Unknown notification
  methods, parameters, and envelope extensions are preserved. Returned JSON
  bytes are owned by the caller. Results bypass the event queue.
- `Close()` terminates the pipe and joins fixed I/O workers. `Done()` closes only
  after they exit; `Wait()` returns the first terminal cause. Intentional Close
  yields `ErrClosed`; peer EOF stays EOF and never implies turn success.

Call cancellation removes a queued, not-started frame. Once writing begins, it
only abandons local waiting: the server may execute the request. An abandoned
correlation entry remains until its response arrives or the connection closes.
There is no invented cancellation, shutdown, or exit protocol message.

## Server requests and reply authority

Every unregistered request method is rejected with error `-32601`, including
approval, dynamic-tool, credential, elicitation, and attestation requests. The
ordered event records that rejection decision; delivery can still fail when the
connection closes. Registration in `ServerRequestMethods`
means the application explicitly undertakes to inspect and answer that exact
method; it does not advertise a capability or authorize any tool execution.

Registered requests carry a connection-bound `ServerRequest` handle. Its
`Respond`/`Reject` methods permit one attempt; they cannot be redirected to a new
connection or retried after failure. Local payload validation and an already
canceled context do not consume the handle. `Done()` invalidates UI actions on
attempt, server resolution, or disconnect. The resolution notification is
processed before it enters the event stream. A queued reply is removed if
resolution wins; a reply whose write has begun cannot be unsent.

A byte-identical pending request frame (excluding its LF/CRLF delimiter) is
coalesced onto the existing state and is not emitted twice. A changed frame with
the same pending ID terminates the connection. A later ID reuse after the prior
request is finished produces fresh reply authority; no cached approval is ever
replayed automatically.

## Bounds and failure behavior

All defaults are client policy, not claimed upstream protocol limits:

| Limit | Default |
| --- | ---: |
| Frame bytes, excluding CRLF | 16 MiB |
| Queued events | 256 |
| Queued event payload bytes | 32 MiB |
| Queued/active writes | 128 |
| Queued/active write bytes, including LF | 32 MiB |
| Pending outbound calls, including abandoned | 128 |
| Abandoned outbound calls | 128 |
| Pending inbound requests | 128 |
| Retained inbound ID/key bytes | 32 MiB |

Zero fields use defaults; negative or integer-overflowing values are rejected.
Event byte budgets include queued handle ID/key copies even after resolution
removes the request from the pending-request budget. Pending handles may be
conservatively counted in both budgets. Byte budgets cover retained
JSON/method/ID data, not Go object/allocator overhead
or caller-held events. Initialization metadata is additionally bounded by one
frame. Parser scratch space is additionally bounded by the
configured frame size, with nesting limited to 64 levels. Every object rejects
duplicate decoded keys; invalid UTF-8 and unpaired UTF-16 escapes are rejected.
IDs and RPC error codes retain exact signed 64-bit precision. String and integer
IDs are distinct. Batches, null envelopes, ambiguous variants, malformed frames,
and unexpected, duplicate, or premature responses are terminal errors. Unknown extension
fields remain allowed. LF, CRLF, and complete final frames without LF are read.

Event overflow is terminal and explicitly marks an incomplete stream. Previously
accepted events remain drainable. Pending-call admission exhaustion is a local
`ErrPendingLimit`; abandoned-call, incoming-request, or write-queue overflow
terminates the connection. Active writes count against write budgets. Any short
or failed write terminates without continuing that frame or writing another.
Calls and Wait never require the event stream to be drained.

This transport logs no payloads. `RPCError.Error()` includes only its code;
explicit inspection of `Message`, `Data`, events, or initialization metadata may
expose sensitive server data. Applications own rendering, policy, and redaction.

## Protocol provenance and verification

The wire target is official Codex `rust-v0.163.0`, commit
[`83086188ae0614bd882818ff1aa23f3194962012`](https://github.com/openai/codex/tree/83086188ae0614bd882818ff1aa23f3194962012).
The small fixture set is synthetic, not a recording. See
[testdata/README.md](testdata/README.md) for its exact schema/source provenance.
No broad typed API is generated. Typed thread/turn methods, process ownership,
approval UI, authentication, reconnect, and history remain separate work.
