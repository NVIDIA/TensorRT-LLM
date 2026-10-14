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

For Dynamo's root-aligned `block_hashes`, each array entry represents a block
starting from the request's first block. With typical decimal `u64` hashes, a
single-action envelope with short metadata holds about **3,100–3,200 hashes**,
or roughly **100,000 tokens of represented prefix at 32 tokens per block**.
This is an estimate, not a fixed token limit. Hash values, identifiers, endpoint
length, extra metadata and additional actions affect the available capacity.

The size limit bounds the represented cache prefix, not the total prompt length.
A longer prompt can still carry a shorter reusable-prefix hint. If an envelope
exceeds the byte limit, admission discards the entire envelope without truncating
its hash array; inference continues without that hint. Larger-prefix support
requires agreement with the producer on the size budget and representation.

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

Action selection is private orchestration policy, separate from the public
envelope format. The configured action consumer supplies its supported action
versions, independently of the envelope protocol version. Selection skips
unsupported versions and chooses the first supported `kv.fetch` in producer
order. Version strings must match exactly; an empty supported set or no matching
action produces no selected action.

Selection preserves the envelope protocol, message identifier, extra metadata
and the complete selected action, including its identifier, version and opaque
payload. It reports separate counts for unknown action types, unsupported fetch
versions and additional supported fetch actions. Orchestration accounts for
these once; selection itself has no global counters.

Keep the consumer's supported-version set fixed for the request. After selection,
runtime authorization failure, payload validation failure or provider rejection
discards optional reuse without selecting another action. Payload validation
remains with the consumer and provider; the selector does not inspect payloads.

The private helper returns the selected action as an immutable one-action
envelope. Action-to-source integration authorizes the source/intent and supplies
the final backend hint. The holder forwards that immutable value without
interpreting it. A provider can accept the selected envelope itself as its
backend hint.

Orchestration must invoke selection and account for ignored actions once.
Required P/D handoff remains separate. Action-to-plan wiring and execution
remain integration work. The holder and shared transport do not interpret
provider payloads or contact endpoints.

KVCM derives content names locally from the request's own tokens and state.
Producer `block_hashes` guide routing/range selection; they are not content
names or storage object keys.

Before any hint-directed connection or SDK submission, orchestration's
action-to-source integration must check the selected `kv.fetch` endpoint
against one server-controlled trusted registry. The consumer adapter interprets
the endpoint field; generic envelope admission and the holder keep provider
payloads opaque. If the endpoint is not listed, discard the optional hint and
continue inference. This does not relax required P/D handoff rules.

The provider must independently validate its action version, payload and
destination before connecting or submitting SDK work. It must not rely solely
on runtime approval. A supplied address never grants permission to contact it.

Consumer activation requires tests that bypass each check separately. The
runtime check must block an unauthorized destination when provider validation is
bypassed. The provider check must block it when runtime validation is bypassed.
Verify that neither case starts a connection or SDK operation, and that rejection
of optional reuse lets a runnable request continue inference. This metadata-only
PR defines these requirements; consumer integration must implement and test them
before activation.
