// Package appserver implements a bounded, bidirectional Codex app-server JSONL
// connection over a supplied pipe. It never starts a process, dials a socket,
// discovers credentials, retries requests, or reconnects.
//
// Connect takes ownership of its transport, even when initialization fails.
// The transport must support concurrent Read, Write, and Close, and Close must
// promptly unblock both Read and Write. A net.Pipe endpoint satisfies this
// contract. The connection context controls the entire lifetime; a Call context
// only controls that caller's wait. Canceling a dispatched call does not cancel
// remote execution. Closing the connection does not establish remote success,
// cancellation, or process termination.
//
// Next delivers one ordered stream for a single consumer. Responses bypass that
// stream. Exhausting a queue terminates the connection rather than silently
// dropping events; already accepted events remain drainable. Incoming requests
// are rejected unless their exact method was registered in Options. Registration
// is a commitment by the application to inspect and answer the request, not
// permission for this package to execute it. No approval policy is implemented.
// Raw payloads and errors may contain sensitive data; this package logs nothing.
package appserver
