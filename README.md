# OpenAI [![Go Reference](https://pkg.go.dev/badge/github.com/picatz/openai.svg)](https://pkg.go.dev/github.com/picatz/openai) [![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

An unofficial community-maintained OpenAI CLI and Go wrapper for the Codex CLI.

## Install

Requires Go 1.27 or newer.

```sh
go install github.com/picatz/openai/cmd/openai@latest
```

Set `OPENAI_API_KEY` in your environment. Obtain a key from the [OpenAI platform](https://platform.openai.com/). Requests use your account and may incur API charges. Never commit keys or paste them into bug reports.

## Responses

With a terminal attached, `openai` or `openai responses chat` opens the Bubble Tea v2 Responses chat. `openai chat` opens the same interface using Chat Completions. One-shot commands and piped input use plain stdout without terminal control sequences:

```sh
openai responses create 'Explain Go contexts'
printf 'Explain Go contexts' | openai
cat question.txt | openai responses create -
openai responses create 'Explain Go contexts' --stream
openai responses create 'Explain Go contexts' --output json
openai responses create 'Explain Go contexts' --stream --output json
openai responses get resp_123 --output json
openai responses delete resp_123
```

`--stream` prints text deltas as they arrive. With `--output json`, it emits one complete API response after successful completion, including usage and response ID. Failures go to stderr with a nonzero exit code; canceled commands exit with code 130. A truncated, failed, or incomplete stream is an error, even if partial text has already reached stdout.

One-shot creation sets `store: false`. Retrieval/deletion requires an existing response stored by another request. Web search is off by default for one-shot requests; add `--web-search` to enable it. The new interactive Responses chat also uses `store: false`, keeps native output items for same-endpoint replay, and only enables web search when requested. `--legacy` retains the previous temporary server-storage and web-search behavior and attempts bounded deletion on exit.

**Migration:** `responses get` now retrieves a response by ID. To generate a response from a prompt, replace the old `responses get 'prompt'` form with `responses create 'prompt'`. The removed Assistants commands have been deprecated in favor of Responses.

## Configuration

All commands use the same SDK client configuration:

- `OPENAI_API_KEY`: API authentication
- `OPENAI_MODEL` or `--model`: text model (default `gpt-4o`, retained for compatibility)
- `OPENAI_BASE_URL` or `--base-url`: alternate API base URL
- `OPENAI_API_URL`: supported legacy base URL alias, only used when `OPENAI_BASE_URL` is absent
- `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID`: SDK organization/project options
- `--timeout`: per-request timeout (default `2m`; `0` disables it)

Flags take precedence over environment variables. The image command has its own image-model flag. `--help` and shell completion do not require credentials.

### Legacy Chat Completions and local models

`openai chat --legacy` keeps the previous Chat Completions terminal and Pebble history format. Add `--temporary` for its memory-only history. Existing history at `~/.openai-cli-chat-pebble-storage-cache` is preserved; no migration or automatic deletion is performed.

An OpenAI-compatible Chat Completions server can be selected with:

```sh
OPENAI_MODEL='your-local-model' OPENAI_BASE_URL='http://localhost:11434/v1/' openai chat
```

Responses support depends on the chosen server. Both `openai chat 'prompt'` and `openai responses create 'prompt'` accept piped input, text/JSON output, and streaming. There is no automatic fallback or translation between the two APIs. Chat `--stream --output json` emits a normalized `{id,text,usage}` result; nonstream JSON preserves the full API response. Tool calls in the text-only Chat interface are rejected explicitly. The legacy terminal and image interfaces retain their existing output behavior.

## Interactive chat and local sessions

The responsive Bubble Tea v2 UI has a multiline composer, scrollback, request status, response IDs, token usage, and a local session picker.

- Enter sends; Alt+Enter, Shift+Enter, or Ctrl+J adds a line
- PgUp/PgDown scroll; Ctrl+Home/Ctrl+End jump to the beginning/end
- Ctrl+C cancels the current request and restores its draft; when idle it clears the draft or exits
- Ctrl+O selects a saved session for the same API and endpoint; Ctrl+N starts a new one
- Ctrl+S retries a failed history save; Ctrl+Q quits (asks again before discarding an unsaved reply)

The UI saves successful turns locally under the OS configuration directory (`openai/sessions`). Use `--history-dir` to choose another directory or `--temporary` to disable saving. Files are created with owner-only permissions on Unix. Session history contains prompts and replies, so protect it as private data. The previous Pebble history is untouched and remains accessible with `openai chat --legacy`.

Responses sessions retain the full native output items, including unknown fields and encrypted reasoning when supplied, rather than rebuilding history from display text. Sessions are bound to their API and endpoint; the CLI refuses to send an existing session to another endpoint. No credentials are stored with sessions. A failed/canceled request does not append to saved history. Atomic writes and revision checks prevent silently overwriting another process's changes.

The same sessions work without the UI:

```sh
openai responses create 'Start a discussion' --session new
openai responses create 'Continue it' --session SESSION_ID
openai chat 'Start a local-model discussion' --session new --model your-model
openai sessions list
openai sessions show SESSION_ID --output json
```

Text mode prints the saved session ID on stderr. Session mode JSON emits `{session_id,response_id,text,usage,saved}`. Resume with the original model, API command, and base URL. Only successful responses are committed; save failures report an error and leave the reply available in output/the UI. Provider support for native item replay varies; unsupported features surface as API errors, without silently stripping state or switching providers.

`--legacy` preserves the former terminal commands, including file/URL/clipboard expansion and Codex delegation. Those shortcuts are not automatically interpreted by the new text-focused UI.

## Development and verification

```sh
go test ./...
go test -race ./...
go vet ./...
go build ./cmd/openai
```

The default suite is hermetic: API calls use synthetic transports and Codex process tests execute the test binary as a fake helper. No API key, installed Codex, microphone, or paid model access is needed. CI runs those checks on Linux, macOS, and Windows.

Historical live integration examples are separately gated. Running them requires both the `integration` build tag and `OPENAI_LIVE_TESTS=1`, plus deliberately configured credentials and an installed Codex CLI. They can incur charges, clone external repositories, and let Codex change temporary checkouts. They are not part of CI. Never enable them merely to run the normal test suite.

This modernization is staged: the CLI/SDK/test foundations and Bubble Tea interface precede provider proxy and newer API capabilities. Live-service and real-device verification is separate from mock coverage.

## Decisions (public beta)

The native Decisions endpoint evaluates typed predicates, choices, and ordered scores. This first interface accepts text, including stdin; it does not translate Decisions into Chat/Responses requests on other providers.

```sh
openai decisions predicate 'This item is damaged' --question 'Is damage reported?'
openai decisions choice 'I need a refund' --question 'Which team should help?' \
  --choice 'billing=Payments and refunds' --choice 'support=Product usage'
openai decisions score 'The service is unavailable' --question 'How severe is the issue?' \
  --level 'low=Minor inconvenience' --level 'high=Work is blocked' --output json
```

The default is `gpt-6-luna`, the model currently documented for the public beta. Predicate values are probabilities, choice values select one supplied option, and scores are probability-weighted averages of zero-based level indices. JSON output retains the complete probability distributions and unknown response fields. Confidence is a model estimate, not a guarantee; validate thresholds against your own labeled data. Availability and beta behavior can change; see the [official Decisions guide](https://developers.openai.com/api/docs/guides/decisions).

## File audio

```sh
openai audio transcribe recording.wav
openai audio transcribe recording.wav --output json
openai audio speech 'Hello from the CLI' --file speech.mp3
printf 'Read this aloud' | openai audio speech --voice cedar --format wav --file speech.wav
openai audio speech 'Hello' --format pcm --file - > speech.pcm
```

Transcription defaults to `gpt-transcribe`; speech defaults to `gpt-4o-mini-tts` with the `marin` voice. Both can be selected with the subcommand's `--model`. Transcription accepts a regular file up to 25 MB. Speech accepts at most 4096 characters and supports MP3, Opus, AAC, FLAC, WAV, and PCM. Raw PCM is not a WAV container.

Speech output never overwrites an existing file. A complete owner-readable temporary file is published without replacing another writer's destination; request or write errors remove the temporary data. Binary stdout must be redirected away from a terminal. These commands do not record from a microphone or play audio automatically. Make clear to listeners that generated speech is AI-generated. See [file transcription](https://developers.openai.com/api/docs/guides/speech-to-text) and [text to speech](https://developers.openai.com/api/docs/guides/text-to-speech) for model-specific capabilities.

The tests use in-memory mock HTTP transports and a generated silent WAV fixture. No recording, playback, real API call, or paid-service validation is part of the default tests. GPT-Live is a separate session protocol and is not emulated by combining these file-audio commands.

## Bounded GPT-Live file transport

`openai live` is an explicit, headless file-audio client for the GPT-Live primary WebSocket protocol. It does not open a microphone, play speakers, or use the different Realtime API event protocol.

```sh
openai live --input-pcm input.pcm --output-pcm reply.pcm \
  --sample-rate 24000 --duration 30s --listen-after 5s --output json
```

Input/output are raw mono signed 16-bit little-endian PCM at 16 or 24 kHz, not WAV containers. Convert/resample input beforehand; changing `--sample-rate` does not resample bytes. Input is paced in 20 ms frames. After EOF, the client sends silence for `--listen-after`, then requests a close. The explicit duration limit can cut off a reply; GPT-Live has no output-audio-done event to infer when it is safe to stop listening.

Defaults: `gpt-live-1`, voice `marin`, a `gpt-6-luna` Responses backend with a 1024 output-token cap, a 1-minute session duration (maximum 10 minutes), and 5 seconds after EOF. No backend tools or server recording storage are configured. Voice is billed by duration, with backend model usage charged separately. The transport has 15-second startup/finalization bounds, no automatic reconnect/retry, capped frames/usage metadata, and same-origin-only authentication: redirects are rejected, and non-TLS connections are limited to loopback.

Add `--controls` to read line commands from stdin:

- `mute`: stop forwarding file samples locally and request server mute
- `unmute`: resume only after the matching server acknowledgment
- `close`: stop input and finalize the session, then save successful output

Muting input does not stop backend work or assistant speech. Ctrl+C requests graceful finalization but returns cancellation and does not publish the output file. Other errors also discard temporary audio. A successful explicit/EOF/duration close publishes a complete new file without replacing existing data. Microphone capture, playback interruption, and hardware/device integration are later work, not implied by these controls.

Control/transcript/backend events are logged only when `--event-log events.jsonl` selects a new private regular file; audio remains in its selected PCM file. A slow event consumer triggers bounded shutdown rather than blocking the session. An unread stderr pipe cannot block the Live protocol loop. JSON stdout reports whether final usage was confirmed and whether audio was saved. `session.usage.updated` is cumulative, and `session.closed` supplies the final voice duration; backend token usage is kept separately. A socket failure or timeout before `session.closed` is reported as unconfirmed finalization, not successful completion.

Tests cover synthetic PCM, protocol ordering, acknowledged mute/unmute, cancellation/backpressure, final-usage collection, truncation/timeouts, and loopback WebSocket/auth/redirect contracts. Real GPT-Live service behavior, audio quality, microphone access, and playback are untested. See [GPT-Live WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live) and [session lifecycle](https://developers.openai.com/api/docs/guides/live-conversations).
