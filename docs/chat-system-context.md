# Chat system-context compatibility

The legacy Chat Completions terminal's `system: <instruction>` command replaces the active instruction. It sends one current system message at the beginning of the next request, rather than appending contradictory instructions to the conversation. `system:` with no instruction clears it.

This addresses the CLI-side cause of [issue #11](https://github.com/picatz/openai/issues/11): some model templates inspect only the first system message. It does not claim that different providers/models interpret all instructions identically. No generic API translation or provider-specific model emulation is involved.

User/assistant turns and stored request/response history are not deleted or rewritten by the setter. Generated summaries remain lower-privilege assistant context, and summarization retains the active system instruction. An in-memory summary carrying the legacy `Summary of previous messages for context: ` marker is retained as assistant context when the instruction changes.

Regression tests inspect the actual Chat Completions request against a synthetic transport that models first-system-only template behavior. They also cover repeated updates, clearing, legacy summary preservation, summarization, and unchanged storage. Real Ollama/model behavior remains a separate integration check. Keep issue #11 open until the relevant fix has landed and verification supports closure.
