# Internal provider and proxy contract

This is an internal, opt-in foundation for future CLI routing. It adds no server
command, listener, credential store, automatic provider detection, model download,
or endpoint fallback. Existing CLI requests do not go through it yet.

## Capabilities are declarations, not guesses

`internal/provider` identifies Chat Completions, Responses and Decisions as
separate wire protocols. For each exact model route, declare independently:

- Endpoint availability
- Streaming
- Tools
- Tool choice (accepting tools does not imply accepting `tool_choice`)
- Structured output
- Upstream state/storage/continuation

Each value is `Unknown`, `Supported`, or `Unsupported`; the zero value is
`Unknown`. Endpoint support never implies support for its optional features.
An unknown or unsupported requested capability fails explicitly with HTTP 501,
before any upstream request. Error messages retain the distinction. Provider
validation of particular tool types, schemas, modalities and model behavior
remains authoritative; these declarations are not a complete API schema.

No route implicitly supports Decisions. The reserved `/v1/decisions` path is
forwarded only if the application explicitly declares that exact endpoint
supported from documentation or a separate contract test. Structured output,
Responses support, or a provider's differently named decision-like API are not
such evidence. No Decisions emulation is provided.

There are deliberately no version-blind provider presets. An OpenAI-compatible
base URL does not establish support for both APIs. Local models must use the API
mode they actually implement. For stateless Responses routes, callers must send
`store:false` and replay full prior output items client-side; `previous_response_id`,
`conversation`, `background:true`, and `store:true` require declared state support.
When state support is unknown, `store:false` is also required. The proxy never
silently changes these fields or reconstructs output items from text alone.

## Fixed routing

`internal/proxy.New(Config)` returns an `http.Handler`. `Config.Routes` is trusted
application configuration, with one unique, exact `Model` string per upstream
`BaseURL`. The base URL includes the API prefix, for example
`http://127.0.0.1:11434/v1` or `https://provider.example/api/v1`.

A minimal synthetic configuration is:

```go
handler, err := proxy.New(proxy.Config{
    Routes: []proxy.Route{{
        Model:   "configured-local-model",
        BaseURL: "http://127.0.0.1:11434/v1",
        Capabilities: provider.Capabilities{
            provider.ChatCompletions: {
                Endpoint:  provider.Supported,
                Streaming: provider.Supported,
            },
            provider.Responses: {
                Endpoint:  provider.Supported,
                Streaming: provider.Supported,
                State:     provider.Unsupported,
            },
        },
    }},
})
```

This example is a declaration for a separately verified local server, not a
claim that every server or model has these capabilities. Any API key,
organization or project belongs to the specific route. No environment key is
read by this package. Configure distinct routes with distinct credentials where
needed. Only the configured destination is used; request fields cannot select a
host, base URL, credential or fallback. Model aliases/rewrites and cost-based
selection are intentionally deferred.

Supported downstream requests are exactly POST to:

- `/v1/chat/completions`
- `/v1/responses`
- `/v1/decisions`, subject to its explicit declaration above

These map to the same relative endpoint at the selected base URL. Query strings,
encoded path variants, other methods, retrieval/deletion endpoints, compressed
request bodies and unknown model routes are rejected. The request must be one
JSON object with a nonempty model string. Duplicate top-level keys and
case-ambiguous spellings of routing/capability parameters (including Unicode
case-fold aliases) are rejected to prevent parser-dependent routing or feature
selection. The relevant nested envelopes, Responses `text.format` and each
format's `type`, also reject duplicate or case-ambiguous keys. Only exact keys
determine capability requirements. Unknown fields, provider-specific nested
objects and output items otherwise remain opaque in the original request bytes.

## HTTP behavior and limits

- Successful requests forward the complete original JSON bytes without SDK
  decoding/re-encoding, translation, field removal or model substitution
- Upstream statuses, error bodies, end-to-end response headers, trailers and
  response bytes pass through, including SSE comments, event types, unknown
  events and final usage frames
- Only downstream `Accept` is copied upstream; content type is set to
  `application/json` and authentication headers are supplied from the route
- Caller authorization, cookies, organization/project values, forwarding
  headers, trailers, idempotency keys and other arbitrary headers are not copied
- Hop-by-hop response headers are removed by Go's reverse proxy
- No retries, redirect following or rerouting occur, before or after output
- A failure before headers returns a generic 502 or deadline-related 504;
  transport error details, URLs, credentials and prompts are not logged or
  included in generated errors
- After output begins, transport/read/write/deadline failures abort the stream;
  no replacement JSON error or second provider response is appended
- Upstream response bodies are not scrubbed: an upstream's own response is
  forwarded faithfully, so treat the configured upstream as trusted
- Forwarding uses a bounded copy buffer and flushes immediately; slow downstream
  writes apply backpressure instead of accumulating the upstream response
- Caller cancellation closes the upstream request
- The default input limit is 1 MiB; the default total request timeout is two
  minutes, including request-body reading and streaming. Zero configuration
  selects these defaults; negative values are rejected. There is no unlimited mode
- Native HTTP read/write deadlines also bound stalled sockets, with one second
  of write grace to deliver a timeout error. ResponseWriter wrappers must expose
  `Unwrap` for deadline support or configure equivalent server limits
- The default transport does not use environment HTTP proxies and disables
  automatic decompression; it also bounds dialing, TLS and response-header waits

A custom transport must respect request contexts and must not implement its own
retries or redirects. The handler does not provide authentication, concurrency
quotas, an HTTP listener, or a general-purpose open proxy. A future server entry
point needs loopback-by-default binding, explicit access controls for any broader
exposure, request-header/idle limits and concurrency limits. Do not expose this
internal handler publicly as-is.

## Verification

The ordinary suite uses only synthetic HTTP servers, fake credentials, and
custom in-memory transports:

```sh
go test -race ./internal/provider ./internal/proxy
```

Contracts cover both APIs' opaque requests and outputs, fixed model routing,
credential isolation, unknown and unsupported features, explicit Decisions
opt-in, no redirects or retries, upstream error/status preservation, exact SSE
bytes and trailers, prompt flushing, caller cancellation, body limits, bounded
backpressure, pre-header and mid-stream timeouts, write failure and truncation.
Boundary regressions cover Unicode case-fold aliases, duplicate/escaped keys,
nested structured-format ambiguity, and unchanged unknown-field pass-through.
No paid API, real Codex execution, installed model or API key is required.

### Separate local-model verification

A real local-model job must remain separate from normal pull-request CI. It
should be manual and secret-free, pinned to an immutable server artifact and
model digest, with an explicit download-size budget, job timeout, process
cleanup, loopback binding and short token/context limits. Verify both Chat and
Responses nonstreaming/streaming against the installed server version, then
state/tool/schema capabilities independently. A skipped or failed capability
must not trigger an implicit API fallback. Synthetic coverage is not a claim
that real model execution was tested.

No workflow here downloads or runs a model. A version-pinned optional workflow
can be added once its server/model artifacts and supported contract are chosen.
