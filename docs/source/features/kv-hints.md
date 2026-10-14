# KV hint request metadata

The prototype `kv_hints` request field carries a versioned envelope from the
Python LLM API or the OpenAI-compatible completions, chat completions and Responses
endpoints to the PyTorch executor's request metadata.

This is a transport foundation. Receiving an envelope does **not** execute its
actions, contact the named endpoint, activate a backend, choose a source or change
fallback behavior. Source permissions, required-source semantics and action
execution require a separately agreed consumer integration. Do not rely on this
field to enforce a routing requirement.

The wire format matches Dynamo's
[KV hint envelope](https://github.com/ai-dynamo/dynamo/blob/3a4c497204c9ff309735a9a052707159cb99c346/lib/kv-router/src/kv_hints.rs).
The following example transports an action without applying it:

```json
{
  "protocol_version": "0.1",
  "message_id": "msg-123",
  "actions": [
    {
      "action_id": "a1",
      "action_type": "kv.fetch",
      "action_version": "1.0",
      "payload": {
        "source_control_endpoint": "tcp://127.0.0.1:23280",
        "block_hashes": [11, 22]
      }
    }
  ]
}
```

For the Python API, import `KvHintsEnvelope` and `KvHintAction` from either
`tensorrt_llm.llmapi` or `tensorrt_llm.kv_hints`. Pass a typed envelope or raw JSON
object to `llm.generate_async(..., kv_hints=envelope)` or
`llm.generate(..., kv_hints=envelope)`. Python callers are trusted application
code; public applications must establish their own ingress policy before calling
these APIs. Raw envelopes allow admission to ignore a future protocol version
before interpreting that version's fields.

For batched `generate`, a scalar envelope repeats the **same envelope and
message_id** on every prompt. Use an aligned list for distinct messages or to
omit hints on selected prompts:

```python
llm.generate([prompt_a, prompt_b], kv_hints=[envelope_a, None])
```

For HTTP, put the JSON object under `kv_hints` in the body (or the client's
`extra_body`). The completions, chat completions (including Harmony) and Responses
paths carry this metadata; the disaggregated proxy supports its existing
completion/chat endpoints. **gRPC does not carry `kv_hints` in this slice.**

## Ingress and admission

Worker and disaggregated-proxy HTTP servers strip client hints by default,
**before nested schema validation**, and count non-null stripped hints. Malformed
optional hints therefore cannot fail inference when ingress is disabled. To
accept hints, explicitly set `TRTLLM_TRUST_KV_HINTS=1` in each receiving server
process, including proxy and workers. This switch trusts all traffic reaching
that process; it does not authenticate clients. Enable it only behind an
authenticated trusted router or equivalent network restriction. It does not
change authentication or validation of required P/D handoff parameters.

At trusted HTTP and Python admission, unknown envelope versions and envelopes
larger than **65,536 UTF-8 JSON bytes (64 KiB)** become no hint and inference
continues. The limit applies to the compact serialized envelope, not character
count. Raw supported envelopes are bounded before nested validation and the
validated encoding is checked again. Structural errors in supported version
`0.1`, such as a missing `message_id` or non-object action payload, still fail
validation. Typed envelope construction validates its own fields; pass a raw
JSON object when the future envelope shape is unknown.

`trtllm_kv_hints_ignored_total{reason="untrusted|unknown_version|oversized"}`
counts ignored envelopes at the admission edge where they are discarded.
Rejected metadata is removed before queueing, proxy forwarding or rank
broadcast; deserialization and rank consumers do not recount it. The metric is
available through the server's existing Prometheus metrics configuration.

Admission freezes validated JSON before queueing a generation request. Caller
mutation cannot change queued metadata. Serialization and the Python-object
broadcast preserve the snapshot across tensor/context and pipeline parallel
ranks and on child requests. `LlmRequest.py_kv_hints` holds the snapshot; decoding
returns an independent copy. Unknown action types/versions and optional JSON
metadata remain intact.

## Action selection and consumer boundary

The pure `select_kv_fetch` helper selects only the first `kv.fetch`, preserving
the envelope protocol, message identifier, extra metadata and the complete
selected action, including its identifier, version and opaque payload. Its
result reports counts of unknown actions and additional fetch actions for the
orchestration owner to account for once. It performs no global counting, so
calling it on multiple ranks cannot inflate request metrics. Provider rejection
must not try a later action from the envelope.

This PR supplies the selection rule, not action-to-plan wiring or execution.
Orchestration integration must invoke and account for selection once, map the
selected immutable hint to an authorized source/intent, and apply required P/D
handoff separately. The holder and shared transport do not interpret provider
payloads or contact endpoints.

KVCM derives content names locally from the request's own tokens and state.
Producer `block_hashes` guide routing/range selection; they are not content
names or storage object keys. A provider must validate its action version and
payload and match any endpoint against a server-controlled allowlist or trusted
control-plane binding **before** connecting or submitting SDK work. A supplied
address never grants permission to contact it. Activating that consumer requires
provider integration and endpoint-policy tests in addition to this transport.
