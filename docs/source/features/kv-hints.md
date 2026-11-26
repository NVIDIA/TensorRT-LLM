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
paths carry this metadata, including through the disaggregated proxy.

## Known limitations

**Current gRPC request paths do not forward `kv_hints` to the executor.** This
affects clients and routers that submit requests to workers over gRPC, including
routers that accept HTTP requests but use gRPC for the worker connection.
TensorRT-LLM emits no hint-specific warning or drop metric for this missing
transport support; the admission counters below do not report this gap.
gRPC parity remains follow-up work. Use a supported HTTP or Python path when
hint transport is required.

## Ingress and admission

Worker and disaggregated-proxy HTTP servers strip client hints by default,
**before nested schema validation**, and count non-null stripped hints. Malformed
optional hints therefore cannot fail inference when ingress is disabled. To
accept hints, explicitly set `TRTLLM_TRUST_KV_HINTS=1` in each receiving server
process, including proxy and workers. This process-wide switch is
**prototype-only**. It trusts all traffic that reaches the process and does not
authenticate senders.

Before enabling a hint consumer in production, require an authenticated, trusted
sender at ingress and enforce the
[endpoint registry check](#candidate-collection-and-consumer-boundary) before any
hint-directed lookup, connection or SDK submission. The switch does not provide
either control. Required P/D handoff keeps its existing authentication and
validation rules.

At trusted HTTP and Python admission, unknown envelope versions, envelopes
larger than **65,536 UTF-8 JSON bytes (64 KiB)** and malformed hints become no
hint, so inference can continue. The limit applies to the compact serialized
envelope, not character count. Raw supported envelopes are bounded before
nested validation and the validated encoding is checked again.

Structural errors in supported version `0.1`, such as a missing `message_id`
or non-object action payload, discard the entire envelope and count reason
`malformed`. Non-object hint values and non-JSON Python values follow the same
policy. Admission does not salvage individual actions from an invalid envelope.
Accepted envelopes must also fit the proxy's MessagePack transport. Admission
discards an envelope if any integer in its payload or extra metadata is outside
the range `-2**63` through `2**64 - 1`, inclusive, and counts it as `malformed`.
Admission also checks the serialized envelope with the snapshot's JSON decoder.
Envelopes that exceed the decoder's nesting limit are discarded as `malformed`,
even when they fit the byte limit. Snapshots retain the checked JSON bytes.
This policy applies only to optional hints; invalid request bodies, other
request fields and required P/D handoff parameters retain their validation.
Direct construction of a typed envelope still validates its fields and can
raise an error. Pass a raw JSON object when the future envelope shape is unknown.

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

`trtllm_kv_hints_ignored_total{reason="untrusted|unknown_version|oversized|malformed"}`
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

HTTP request text representations omit `kv_hints`, including when a request is
formatted in an error log. JSON serialization and proxy forwarding retain the
metadata. Explicitly logging a serialized request or the hints themselves still
requires separate redaction.

## Candidate collection and consumer boundary

The initial hinted-fetch consumer must ignore hints on requests that use a LoRA
adapter. Orchestration must check request eligibility before candidate
collection. The private helper remains independent of request state. This
restriction applies to optional hinted fetch; it does not change local inference
or required P/D handoff.

Assembly supplies the supported `(action_type, action_version)` pairs for each
configured backend family. Keep this capability snapshot fixed for the request.
A family is a private integration key for action support. It is not a producer
wire field, a concrete backend binding or an addressing mode. Several bindings
can belong to one family. This does not add a public Backend capability API.

The private helper collects every supported `kv.fetch` action in producer order.
Type and version strings must match exactly, independently of the envelope
protocol version. Each candidate contains the unordered set of supporting
families. An action appears once even if several families support it. Repeated
action identifiers do not remove producer entries. Empty capabilities or no
matching actions produce no candidates.

Candidates share the immutable admitted envelope and retain their action
positions. When a consumer needs one candidate, it can obtain an immutable
one-action envelope. This preserves the envelope protocol, message identifier,
extra metadata and complete action, including its identifier, version and opaque
payload, without copying envelope metadata for every retained candidate.

The helper reports separate counts for unknown action types and fetch actions
with no matching version. Each action is counted once, regardless of the number
of configured families. Orchestration accounts for these counts once per
request. The helper has no global counters and does not count retained
candidates as ignored.

The initial orchestration policy selects the first candidate. A trusted mapping
must resolve exactly one configured, authorized binding within its supporting
families. Check that binding's actual capability; a family match does not prove
that every binding in the family can process the action. Family registration
order does not select a binding. Missing, ambiguous or unauthorized mappings,
payload validation failure and provider rejection discard optional reuse. They
must not select another candidate, family or binding. Retained candidates are
not an automatic retry list.

The helper performs no I/O and does not interpret provider payloads.
Action-to-source integration authorizes the source and intent, then supplies
the final immutable backend hint. The holder forwards that value unchanged.
A provider can accept the selected one-action envelope itself as its hint.

For generation-only requests, a hint opts into named fetch before the paired
source only when the request is eligible and the selected action resolves to
one configured, authorized backend binding. A `kv_hints` field alone does not
enable named fetch. Without a usable hint, keep the existing paired path and
add no named lookup or transfer work. This includes hints discarded by
admission, unsupported actions, excluded requests and missing, ambiguous or
unauthorized bindings. The hint does not replace required P/D handoff.

Action-to-plan wiring and execution remain integration work. The holder and
shared transport do not interpret provider payloads or contact endpoints.

Every `kv.fetch` action in `kv_hints` requests KV by content name. Request-paired
handoff uses separate P/D parameters and never comes from these actions.
Orchestration must reject a `kv.fetch` mapping to a request-paired binding before
provider contact. Discard that optional reuse and keep the permitted no-hint
path. A peer worker can serve named KV; the distinction is the addressing mode,
not the worker location or transport library.

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
of optional reuse lets a runnable request continue inference.

Before consumer activation, map a supported `kv.fetch` action to a request-paired
binding in a test. Verify that orchestration rejects optional reuse without
starting a hint-directed lookup, connection or SDK operation. Verify that
required P/D handoff still uses its original parameters.

Before consumer activation, test LoRA requests with a supported `kv.fetch` hint.
Verify that orchestration does not collect candidates, open a hint-directed
connection or submit hint-directed SDK work. Verify that eligible requests
without a LoRA adapter can still reach candidate collection. The LoRA gate must
preserve local inference and required P/D handoff behavior.

Before consumer activation, test generation-only requests with and without a
usable hint. Verify that an eligible request with a selected, resolved and
authorized hint attempts named fetch before the paired source. Cover absent,
discarded and unsupported hints, excluded requests and failed binding resolution;
verify that these cases keep the paired path without added named-fetch work.
Verify that required P/D handoff rules still apply when named fetch succeeds or
fails.

Before consumer activation, test overlapping family support and several
bindings in one family. Verify that trusted resolution selects one supported,
authorized binding without using registration order. Reject missing or
ambiguous mappings. Verify that authorization, payload or provider rejection
does not select another candidate, family or binding.

This metadata-only PR defines these requirements; consumer integration must
implement and test them before activation.
