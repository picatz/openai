# Synthetic protocol fixture

`conversation.jsonl` is authored synthetic data, not a recording. Its index
identifies direction, payload key, and upstream schema for all 33 frames.
The complete fixture was independently checked against 21 pinned official
schema files, with 69 successful checks and three negative assertions.

`provenance.json` records the pinned URLs, Git blob IDs, and SHA-256 hashes used
for that check. Source/schema files are referenced rather than vendoring or
generating the full app-server protocol. Fixture `thread/start` options select
only a synthetic directory and model; sandbox and approval values in responses
are synthetic returned metadata, never client permission overrides.

The runtime test adjusts only its initialize title. Other client messages are
compared structurally with the fixture, including omitted optional settings.
