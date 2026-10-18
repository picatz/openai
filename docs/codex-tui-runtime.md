# Interactive Codex workspace

`openai codex tui` opens a fresh app-server thread through the separately
installed Codex CLI. It requires terminal input and output; piping a prompt is
supported by `openai codex exec` instead. This workspace streams actual
app-server events when a user launches it. Test fixtures are synthetic; they are
not the implementation's runtime data source.

```sh
openai codex tui --cd ./project
openai codex tui --model MODEL --no-color --reduced-motion
```

## Controls and current boundary

- Enter submits the current draft as a text turn.
- Alt+Enter or Ctrl+J inserts a newline. Bracketed paste inserts without submitting.
- F6 switches between the composer and workspace. Esc focuses the workspace.
- In the workspace, arrows or j/k select an item; Tab switches item/detail focus;
  PgUp/PgDn scroll; Ctrl+Home/End goes to the beginning or follows the latest detail.
- F1 opens keyboard help. Typing `?`, j, or k in the composer inserts ordinary text.
- Ctrl+C requests interruption only for a known active turn. While a start reply
  is pending, the UI reports uncertainty instead of inventing a remote cancel.
- Ctrl+Q closes the local connection, terminates/reaps its direct app-server child,
  and restores the terminal. Local close is not a confirmed remote interruption.

Each completed turn permits another prompt on the same fresh thread. Only the
current turn is displayed. Partial output remains visible after a failed
connection, with an incomplete/unknown outcome. There is no automatic retry,
reconnect, history picker, resume, mid-turn steering, approval UI, or login flow.
The draft is capped at 16 KiB, with visible truncation notice, and is not saved by
this client. System clipboard reads and writes are not implicit composer actions.

The workspace adapts to wide and narrow terminals. It requires at least 30×17
for the composer layout and shows a resize hint below that size. `--no-color` or
nonempty `NO_COLOR` suppresses color; `--reduced-motion` requests static activity.
This slice does not claim a complete screen-reader or ASCII-only mode.

## Permissions, settings, and costs

Installed Codex configuration, authentication, integrations, permission policies,
and persistence remain in effect. This client does not supply sandbox or approval
overrides and makes no stronger safety guarantee than the installed configuration.
The returned model label is configured state, not per-turn execution telemetry.
Turns may incur charges under the installed Codex account/provider configuration.

Incoming server requests are rejected and visibly reported. This includes
approvals and interactive input; no approval controls are presented. Use the
native Codex client for those workflows. This client does not inspect existing
history or credentials, log request payloads, or retain child stderr. If startup
fails, check the installed Codex client's configuration/authentication separately.

`--cd`, `--model`, and an optional whole-session `--timeout` are supported.
Responses API flags, local-session flags, `--output`, and permission passthrough
flags are rejected. No automatic Codex installation or shell expansion occurs.

## Verification and provenance

The adapter targets the pinned official `rust-v0.163.0` protocol; see
[the runtime contract](../internal/codexsession/README.md) for exact methods,
resource limits, process ownership, and source provenance.

All verification uses synthetic pipe peers and fake child executables. It covers
both start-response orderings, repeated turns, final snapshots, stale events,
explicit interrupt versus local close, unsupported requests, queue exhaustion,
EOF and uncertain outcomes, bounded text, and process cleanup. PTY tests exercise
paste/newlines, repeated Ctrl+C, Ctrl+Q during activity and pending start, context
cancellation, and restoration of terminal state, alternate screen, cursor, and
bracketed-paste mode. Fixed wide, standard, compact, and plain frames are synthetic.
Authenticated Codex/app-server interoperability has not been tested.
