# OpenAI [![Go Reference](https://pkg.go.dev/badge/github.com/picatz/openai.svg)](https://pkg.go.dev/github.com/picatz/openai) [![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

An unofficial community-maintained OpenAI CLI and Go wrapper for the Codex CLI.

## Install

Requires Go 1.27 or newer.

```sh
go install github.com/picatz/openai/cmd/openai@latest
```

Set `OPENAI_API_KEY` in your environment. Obtain a key from the [OpenAI platform](https://platform.openai.com/). Requests use your account and may incur API charges. Never commit keys or paste them into bug reports.

## Responses

With a terminal attached, `openai` or `openai responses chat` opens the Responses chat. One-shot commands and piped input use plain stdout without terminal control sequences:

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

One-shot creation sets `store: false`. Retrieval/deletion requires an existing response stored by another request. Web search is off by default for one-shot requests; add `--web-search` to enable it. Interactive Responses chat currently retains its previous temporary server-storage and web-search behavior; it attempts to delete its responses on exit.

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

`openai chat` keeps the existing Chat Completions terminal and Pebble history format. `openai chat --temporary` uses memory-only history. Existing history at `~/.openai-cli-chat-pebble-storage-cache` is preserved; no migration or automatic deletion is performed.

An OpenAI-compatible Chat Completions server can be selected with:

```sh
OPENAI_MODEL='your-local-model' OPENAI_BASE_URL='http://localhost:11434/v1/' openai chat
```

Responses support depends on the chosen server. The text/JSON/stream one-shot output flags above apply to Responses operations; the legacy chat and image interfaces retain their existing output behavior.

## Codex Go package

The [`codex` package](codex/README.md) wraps `codex exec --json`, with resumable threads, structured output, cancellation, and forward-compatible events. It requires a separately installed Codex CLI. See the package guide for permission options, compatibility scope, and stream cleanup.

## Development and verification

```sh
go test ./...
go test -race ./...
go vet ./...
go build ./cmd/openai
```

The default suite is hermetic: API calls use synthetic transports and Codex process tests execute the test binary as a fake helper. No API key, installed Codex, microphone, or paid model access is needed. CI runs those checks on Linux, macOS, and Windows.

Historical live integration examples are separately gated. Running them requires both the `integration` build tag and `OPENAI_LIVE_TESTS=1`, plus deliberately configured credentials and an installed Codex CLI. They can incur charges, clone external repositories, and let Codex change temporary checkouts. They are not part of CI. Never enable them merely to run the normal test suite.

This modernization is staged: reliable CLI/SDK/test foundations first, followed by the Bubble Tea interface and newer API capabilities. Live-service and real-device verification is separate from mock coverage.
