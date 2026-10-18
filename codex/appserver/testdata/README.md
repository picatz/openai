# Synthetic pinned wire fixtures

`wire.jsonl` contains 13 manually constructed frames, not server captures. They
exercise client initialize; server initialize result; client initialized;
agent-message delta; null result; structured RPC error; command approval with
maximum int64 ID; file approval with string ID; dynamic tool call; unsupported
request rejection; request resolution; future notification; and future request.

The source target is official Codex `rust-v0.163.0`, commit
`83086188ae0614bd882818ff1aa23f3194962012`. `provenance.json` records exact URLs
and SHA-256 digests of the source/schema bytes used to construct and validate the
frames, plus the fixture digest. These digests identify the researched files;
the tests do not download or execute upstream code.

The frames were checked against `JSONRPCMessage.json` and the applicable pinned
InitializeParams, InitializeResponse, ClientNotification, ServerNotification,
and ServerRequest schemas. `TestPinnedFixtures` checks the local strict parser.
Schema acceptance alone does not prove envelope exclusivity or int64 bounds;
separate negative tests cover these. Command/file approvals include the required
`startedAtMs` field absent from some prose summaries.

Relevant wire definitions were also compared against official current commit
`8c076942cb70fba8e5c4732c23f1faabc4aed484` on October 18, 2026. The envelope,
initialization, stdio lifecycle, reply handles, request schema, and resolution
schema were unchanged. Nearby method payloads had changed, which is why the
transport preserves raw payloads rather than publishing broad typed models.

Synthetic framing, concurrency, race, cancellation, resource-bound, and lifecycle
tests establish the local pipe contract only. They do not establish interaction
with an installed Codex process or an authenticated service.
