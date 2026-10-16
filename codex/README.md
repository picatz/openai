# Codex Go process SDK

An unofficial, thin wrapper around an installed `codex exec --json` binary. This
package does not install Codex, manage login, or require the OpenAI Go HTTP SDK.
It searches `PATH` unless `Options.CodexPathOverride` / `NewExec` supplies a path.

## Run and resume

```go
thread, err := codex.NewThread(codex.Options{}, codex.ThreadOptions{
    SandboxMode:    codex.SandboxModeReadOnly,
    ApprovalPolicy: codex.ApprovalModeNever,
    WebSearchMode:  codex.WebSearchModeDisabled,
})
if err != nil {
    return err
}
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
defer cancel()
turn, err := thread.RunText(ctx, "Summarize this repository", nil)
if err != nil {
    return err
}
fmt.Println(turn.FinalResponse)
fmt.Println(thread.ID()) // Save this to resume in another process.
```

A subsequent `RunText` on the same thread resumes it. `ResumeThread(id, options,
threadOptions)` prepares a saved conversation; Codex checks the ID when run. Run
turns serially. `TurnOptions.OutputSchema` accepts a JSON-marshalable object; the
final structured response remains JSON text in `Turn.FinalResponse`. Temporary
schema files are private and removed when the run finishes, fails, or is canceled.

`RunStreamed` returns an event channel. Drain it, then call `Wait`; call `Close` or
cancel its context if you stop reading early. `Wait` alone cannot drain the
channel for you. `Close` waits for process and schema cleanup. Lower-level callers
can use `NewExec`, `Exec.Run`, and the `EventStream` iterator; breaking the iterator
stops and reaps the direct CLI process. `ExecStream.Wait` must follow draining
stdout, or use `Close` to abandon output. Cancellation errors support `errors.Is`
with `context.Canceled` / `context.DeadlineExceeded`; CLI exit errors retain
`*exec.ExitError` and at most the final 64 KiB of stderr. Descendant process-tree
termination is not promised; use the CLI sandbox and a bounded context for runs.

## Options and permissions

Zero values leave installed Codex defaults in place. This package does not change
approval, sandbox, network, or web-search policy unless the caller explicitly
sets an option. Saved Codex configuration and authentication remain in effect.
Even read-only agent work can make model requests and incur charges.

- `ApprovalPolicy` maps to `approval_policy` configuration. This exec wrapper
  cannot answer interactive approval requests; it does not implement an approval UI
- `SandboxMode`, `WorkingDirectory`, and `AdditionalDirectories` map to the CLI
  flags. Writable additional directories expand the agent's filesystem access
- `NetworkAccessEnabled` is a pointer so `nil`, `false`, and `true` are distinct
- `WebSearchMode` accepts disabled, cached, or live search
- `ConfigOverrides` takes ordered `KEY=VALUE` TOML assignments, not file paths.
  Typed settings are applied afterward. `ConfigFile` is a deprecated single
  assignment retained for source compatibility despite its misleading name
- `FullAuto` and `IncludePlanTool` remain deprecated legacy flag passthroughs;
  support depends on the installed CLI. Prefer explicit approval/sandbox settings
- `Args.Ephemeral` requests a run without persisting session files
- Images are passed after `resume` when resuming. The thread ID follows the `--`
  option terminator, so even a dash-prefixed ID stays literal; prompts always go
  through stdin

## Compatibility and verification

Reviewed against [Codex rust-v0.160.1](https://github.com/openai/codex/releases/tag/rust-v0.160.1)
on 2026-10-06. The compatibility sources are the tagged
[exec CLI](https://github.com/openai/codex/blob/rust-v0.160.1/codex-rs/exec/src/cli.rs),
[shared options](https://github.com/openai/codex/blob/rust-v0.160.1/codex-rs/utils/cli/src/shared_options.rs),
[Rust JSONL schema](https://github.com/openai/codex/blob/rust-v0.160.1/codex-rs/exec/src/exec_events.rs),
and [TypeScript exec wrapper](https://github.com/openai/codex/blob/rust-v0.160.1/sdk/typescript/src/exec.ts).
This is a source- and fixture-verified compatibility target, not a claim that an
authenticated Codex session or every historical CLI release was exercised.

The decoder includes MCP arguments/results/errors, web-search action/results,
cache-write/reasoning token usage, declined commands, and in-progress patches.
Missing new counters remain zero for older streams. Unknown event types preserve
an owned complete `ThreadEvent.Raw` payload; they are not treated as today's event
schema. All decoded events also retain `Raw` so extra fields on known events are
available. Unknown items (including collaboration-tool items in this increment)
retain their type and owned bytes in `UnknownThreadItem`, which round-trips its
original JSON. Required event envelopes and malformed known field types fail
rather than being silently treated as success. Invalid JSONL is terminal; blank
lines, CRLF, large events, and a final line without a newline are supported.

[`codex app-server`](https://learn.chatgpt.com/docs/app-server) is a separate,
bidirectional JSON-RPC protocol with initialization, request IDs, and interactive
approval handling. It is not interchangeable with exec JSONL. This increment
keeps the one-process-per-turn design; it does not add app-server transport,
authentication APIs, remote execution, or an interactive approval service.

Default tests use only a synthetic helper built from the test executable and
hand-authored fixtures. The fixture tool names/commands are data, never executed.
No installed Codex, credentials, or live API calls are used. Existing live tests
remain behind both the `integration` build tag and `OPENAI_LIVE_TESTS=1`; do not
enable them for routine verification.

```sh
go test -race -count=3 ./codex
go test ./...
go vet ./...
# Compile the live examples without running any of them:
go test -tags=integration -run '^$' ./...
```
