# Codex CLI entrypoint

`openai codex exec` runs one foreground, non-interactive Codex turn through the
existing Go process SDK. Install and authenticate the separate
[official Codex CLI](https://github.com/openai/codex) first; `codex` must be on
`PATH`. This command does not install Codex, sign you in, or use the OpenAI HTTP
client. Help and flag validation work without Codex or an API key.

```sh
openai codex exec 'Summarize this repository'
cat question.txt | openai codex exec
cat question.txt | openai codex exec -
openai codex exec -C ./project --timeout 5m 'Explain the test layout'
openai codex exec --output json 'Summarize this repository'
openai codex exec --resume THREAD_ID 'Explain the next step'
```

## Input and configuration

- Prompt arguments are joined with spaces. When arguments are supplied, stdin is
  not read or appended. With no arguments, or a single `-`, the prompt is read
  from stdin until EOF. Empty prompts and prompts over 16 MiB are rejected.
- Use `--` before a prompt beginning with a dash. Prompt text is sent to Codex on
  stdin, never interpreted by a shell. Legacy file, URL, and clipboard shortcuts
  are not expanded.
- `--cd` / `-C` selects Codex's working directory. Omit it to use the current
  directory. The installed CLI retains its normal repository checks.
- `--model` is optional. Omit it to let Codex select its configured model; this
  command does not apply the Responses model default or `OPENAI_MODEL` override.
- `--timeout` bounds the whole operation, including reading stdin. The default
  is `0` (no deadline); `5m`, for example, gives the operation five minutes. This
  is different from the HTTP request timeout on API commands.
- `--output text` (the default) prints the last completed agent message after
  both the turn and process succeed. `--output json` prints a single object with
  `thread_id`, `final_response`, and `usage`. It is a completed-result summary,
  not the raw Codex JSONL stream.
- `--resume ID` continues a specific Codex thread. Use `thread_id` from a previous
  JSON result. Codex validates the ID when the turn starts. This is separate from
  `openai --session`; there is no automatic history lookup, picker, or `--last`.

`--model`, `--timeout`, and `--output` work before or after `codex exec`. Other
inherited root flags (`--base-url`, `--stream`, `--web-search`, `--legacy`,
`--temporary` / `-t`, `--history-dir`, and `--session`) are rejected when explicitly
supplied, including explicit false values. Use Codex configuration or the native
Codex CLI for additional settings. In particular, `--temporary` does not disable
Codex session persistence.

## Permissions, errors, and scope

No sandbox, approval, network, or web-search policy is changed by this wrapper.
The installed Codex defaults and saved configuration remain in effect, and the
process inherits its environment. Depending on that configuration, Codex may
read or change files and execute commands. Even read-only work can make model
requests and incur charges. This is not an interactive approval interface and
does not implement the separate `codex app-server` protocol.

Failures go to stderr and return a nonzero exit status. There is no completed
result on stdout after a failed turn, incomplete stream, malformed event, failed
process exit, or cancellation. Native process failures retain the package's
bounded stderr diagnostics; successful native stderr is not forwarded. Ctrl-C
cancels the operation and returns exit status 130. Deadlines return a nonzero
status. The SDK stops and reaps its direct subprocess; descendant process-tree
termination is not promised. See the [package lifecycle guide](../codex/README.md).

Input uses the existing bounded CLI prompt reader. Cancellation returns control
without closing caller-owned stdin. If embedding the command tree in a
long-lived process, its caller must close or unblock a still-pending input read;
the standalone CLI exits after cancellation.

The initial interface intentionally handles text prompts and explicit thread
IDs. Interactive sessions, live event display, images, JSON output schemas,
session discovery, and additional native CLI switches remain available through
the Go package or native CLI where supported.

## Verification

Command tests execute a synthetic helper copied from the Go test executable.
They cover prompts, stdin, literal arguments, exact native options, JSON results,
resume IDs, flag placement, help, missing binaries, validation, malformed and
incomplete streams, turn and process failures, cancellation, and deadlines.
They do not require installed Codex, credentials, live API calls, or paid runs.
The package's source-verified compatibility target and live-testing limits are
documented in its [README](../codex/README.md).
