<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multimodal Encoder Runtime Scheduling

| Field | Value |
| --- | --- |
| Status | Proposed |
| Scope | TensorRT-LLM PyTorch backend |
| Follow-up | [PR #13503: encoder sizing controls](https://github.com/NVIDIA/TensorRT-LLM/pull/13503) |
| Last updated | 2026-07-09 |

## Summary

This document proposes runtime scheduling for multimodal (MM) encoders. It makes
`encoder_max_batch_size` and `encoder_max_num_tokens` effective limits on the work submitted to MM
encoder forwards while preserving the existing LLM scheduling behavior and the same-step MM
encoder-to-LLM execution path.

The scheduler operates on **original MM items** such as one image or one video. An item is atomic in
this design. It may combine multiple compatible items into one physical encoder forward, but it does
not split one item across forwards. The item cost is the number of physical encoder attention tokens
before a connector or merger reduces the representation. This cost is intentionally different from
the number of placeholder embeddings inserted into the LLM prompt.

The default admission policy follows the vLLM model: an MM request rejected by LLM capacity does not
run its encoder independently. An opt-in environment variable enables an experimental independent
policy for A/B testing. That policy lets an already-admitted request make encoder progress before the
current iteration's LLM-capacity filtering; it does not pre-encode waiting requests.

This is a design specification. A prototype on the development branch may contain request-level
scheduling, native encoder request states, or video temporal splitting that this document explicitly
supersedes.

## Motivation

[PR #13503](https://github.com/NVIDIA/TensorRT-LLM/pull/13503) introduced two user-facing controls and
used them to size encoder attention metadata and dummy inputs deterministically:

- `encoder_max_batch_size`: the maximum number of MM items intended for one encoder batch.
- `encoder_max_num_tokens`: the maximum number of encoder attention tokens intended for one encoder
  batch.

That PR deliberately left scheduler enforcement as follow-up work. Consequently, sizing and runtime
work admission can disagree. An encoder forward can receive more items or physical attention tokens
than the configured workspace was designed for. Conversely, using LLM placeholder lengths as the
encoder cost can severely underestimate models with token-reducing connectors, such as Qwen-VL's
vision merger.

LLM chunked prefill does not solve this problem. A long text context can advance through several LLM
iterations, but an MM encoder input is generally bidirectional and cannot be split at arbitrary token
boundaries. The scheduler therefore needs an MM-specific unit of work and budget.

## Goals

1. Enforce a per-iteration MM encoder item budget and physical attention-token budget.
2. Keep one MM item atomic and preserve numerically equivalent encoder and end-to-end results.
3. Preserve the existing MM behavior in which encoder output can feed an LLM forward in the same
   iteration.
4. Preserve the exact non-MM scheduling path and avoid measurable overhead for non-MM models.
5. Reuse existing request metadata, schedule distribution, and model-specific input processors where
   practical.
6. Support Qwen2-VL/Qwen2.5-VL, Qwen3-VL, and Mistral3/Pixtral in the initial implementation.
7. Provide a controlled experiment for encoder-independent admission without making it the default.

## Non-goals

- Splitting a single image, video, or audio item into spatial, temporal, window, or patch microbatches.
- Introducing a user-facing partial precomputed-embedding API.
- Implementing cross-request MM encoder output caching. The design reserves stable item identities and
  lifecycle hooks so a separate cache effort can integrate later.
- Bounding, evicting, or offloading request-local encoder outputs in the first implementation.
- Changing native encoder-decoder scheduling or its encoder stream.
- Supporting MM tower LoRA. The target encoders currently do not accept request-level LoRA parameters.
- Redesigning overlap scheduler or MM prefetch compatibility in the first implementation.
- Extending the new scheduler to models that do not inherit `MultimodalModelMixin`.

## Terminology and units

The word "token" is ambiguous across the MM pipeline. The implementation and metrics must use the
following explicit terms.

| Term | Definition | Example |
| --- | --- | --- |
| MM item | One original user-visible input, such as one image or one video | The second image in a prompt |
| Encoder attention token | One row participating in the encoder's attention before a merger or connector | A Qwen-VL pre-merger vision patch |
| Encoder output embedding | One row emitted after the encoder connector/merger | A row inserted into the LLM embedding stream |
| Placeholder token | A position or run in the text prompt replaced by MM output embeddings | Image placeholder positions |
| Internal encoder sequence | A sequence/window/segment represented in encoder `AttentionMetadata` | One temporal or window-attention sequence |

The scheduler cost of item `i` is:

```text
encoder_cost(i) = sum(item_encoder_attention_metadata.seq_lens)
```

It is not `multimodal_embedding_lengths[i]`, the placeholder count, or the number returned by an
LLM-side context chunk. A connector may make these values equal, but the scheduler must not assume
that identity.

### Configuration semantics

`None` means fallback to the resolved LLM value, not unlimited or legacy behavior:

```text
resolved_encoder_max_batch_size =
    encoder_max_batch_size if configured else max_batch_size

configured_encoder_token_budget =
    encoder_max_num_tokens if configured else max_num_tokens

effective_encoder_token_budget = max(
    configured_encoder_token_budget,
    model_max_atomic_item_tokens,
)
```

The item-count budget counts original MM items. It does not count requests, internal attention
sequences, video frames, temporal units, or physical encoder forwards.

The effective token budget is raised when a valid atomic item can be larger than the configured
budget. This is necessary because the current scope does not split an item. The implementation must:

1. retain both configured and effective values;
2. log a warning once when it raises the value;
3. use the effective value consistently for scheduling, profiling, attention metadata, and workspace
   sizing; and
4. not raise `encoder_max_batch_size` except to enforce the existing positive-value validation.

An individual request whose item exceeds the model's startup-declared maximum is invalid. Input
processing should normally resize or sample the media to remain within model limits. A defensive
runtime check fails only that request; it must not dynamically resize metadata or abort the server.

## Scope and model gating

The executor computes a model-level flag once during initialization:

```python
is_multimodal = isinstance(model, MultimodalModelMixin)
```

The new scheduler path is constructed only when this flag is true. It must not scan `model.modules()`
on every iteration. Models outside the mixin continue through the legacy path even if they accept
some form of multimodal data.

The initial implementation requires complete item-level support for:

- `Qwen2VLModelBase`, covering Qwen2-VL and Qwen2.5-VL;
- `Qwen3VLModelBase`; and
- `Mistral3VLM`, covering Mistral3 and Pixtral variants.

All supported mixin families must provide item costs, item identities, a model-specific encoder
capacity bound, and item-level execution. There is no silent generic fallback for a supported mixin
family.

For a non-MM model, the executor must instantiate and call the existing scheduler object directly.
There is no MM wrapper, MM request scan, MM metadata allocation, or schedule-payload population. For
an MM model with no pending item, the MM wrapper takes a constant-time fast path into the existing
scheduler.

## Request classification and state

Native `LlmRequestState.ENCODER_INIT` belongs to encoder-decoder scheduling and has next-iteration
semantics that do not match the existing MM workflow. The implementation must not reuse it and must
not add a native `MM_INIT` state.

An MM request remains in `CONTEXT_INIT`. Python attaches an immutable request-kind flag:

```text
py_is_multimodal_encoder_request =
    model_is_multimodal_mixin
    and request_has_raw_mm_payload
    and request_has_no_complete_precomputed_mm_embedding
```

mRoPE metadata without raw MM content is not an encoder request. A request with a complete externally
precomputed `multimodal_embedding` bypasses encoder compute and consumes no encoder item or token
budget. Partial external embeddings remain unsupported.

The flag identifies request kind; readiness is stored separately. A partially encoded request stays in
the context state but is filtered out before the LLM microbatch until its required embeddings are
ready or will become ready in the current iteration.

## Static request metadata

Model-specific input processing must produce scheduling metadata before waiting-queue admission and
before the initial request broadcast. The metadata lives in `request.py_multimodal_data`, alongside
existing MM fields:

```python
{
    # Existing: output/placeholder rows per item.
    "multimodal_embedding_lengths": list[int],

    # New: pre-connector physical encoder attention tokens per item.
    "multimodal_encoder_token_lengths": list[int],

    # New: stable lookup into the model-specific raw MM data.
    "multimodal_item_refs": list[tuple[str, int]],
}
```

Examples of item references are `("image", 0)` and `("video", 1)`. All three lists are aligned to the
canonical prompt-placeholder order, not dictionary iteration order, modality grouping, or completion
order. Input processing must reject ambiguous or inconsistent ordering.

These fields are CPU scheduling data and belong in `_CPU_ONLY_MULTIMODAL_DATA_KEYS`. The scheduler
should also reuse existing `MultimodalInput` information where applicable: positions, lengths, hashes,
UUIDs, and item-run metadata. It must not require a hash to schedule. The current hashing path contains
a one-modality flattening assumption, so ordering must be validated before a future cache uses hashes
as item identities.

`MultimodalRuntimeData` is not a suitable scheduling store. It is created for an already selected LLM
chunk and therefore exists too late for waiting admission or encoder item selection.

### Cost generation

Each supported input processor computes physical token lengths from the same normalized media
metadata that it uses to build model inputs. Shared model helpers should prevent the input processor
and encoder forward from carrying duplicate formulas.

At execution, the model compares the declared cost with the actual sum of encoder attention sequence
lengths. A mismatch is a request validation failure. This guards against processor/model drift and
ensures that the scheduler never submits more physical work than it reserved.

## Mutable request-local state

The Python request owns fixed-size item slots, one contiguous output buffer, and precomputed offsets:

```python
py_mm_encoder_outputs: list[Optional[torch.Tensor]]
py_mm_encoder_output_buffer: Optional[torch.Tensor]
py_mm_encoder_output_offsets: list[int]
```

An item has exactly one of two persistent states:

| State | Output slot | Meaning |
| --- | --- | --- |
| `PENDING` | `None` | Eligible for selection |
| `READY` | tensor | Reusable request-local result |

Selection does not mutate request state. The current executor consumes one schedule result before
constructing another result that could select the same item. Keeping selection in the serialized
scheduler output avoids reservations that could leak when a result is canceled or discarded.

On the first successful item, the executor allocates one contiguous final output buffer using the
declared per-item embedding lengths. Each completed item is copied directly into its canonical slice;
the output slots hold views into this buffer rather than independent allocations. Once all slots are
ready, that same buffer becomes `multimodal_embedding` without a final `torch.cat` allocation. On
request cancellation, validation failure, or execution error, the buffer and slots are released with
the request.

## Scheduler architecture

The MM scheduler is a Python pre-pass around the existing CapacityScheduler and MicroBatchScheduler.
It is not a replacement for the C++ LLM capacity scheduler.

```text
waiting requests
      |
      | MM progress eligibility for new admission
      v
existing LLM CapacityScheduler
      |
      | fitting/admitted requests in original policy order
      v
MM atomic-item budget packing
      |
      | partial MM requests are withheld
      v
existing LLM MicroBatchScheduler
      |
      | ready or completing-this-step MM requests retain original order
      v
MM encoder forward(s) -> assemble embeddings -> LLM forward, same iteration
```

### Per-iteration budget

Each call that creates one `ScheduledRequests` result starts with:

```text
remaining_items = resolved_encoder_max_batch_size
remaining_tokens = effective_encoder_token_budget
```

All physical MM encoder forward groups selected for that result share these counters. Creating several
shape-compatible groups does not create several budgets. Budgets are per data-parallel replica; TP,
PP, and CP ranks executing the same replica share the selected item IDs.

The item-count budget and token budget are independent. An item fits only when both counters can pay
its complete cost. Future cache hits use neither compute counter.

### Item selection order

The selection rules are deliberately simple and deterministic:

1. Within one request, process items in canonical prompt order.
2. Never skip an earlier pending item to select a cheaper later item from the same request.
3. Running/admitted requests may backfill across requests: if one request's next item cannot fit, the
   scheduler may consider another active request.
4. New waiting MM admission is strict FCFS. If the head request cannot make initial progress with its
   first pending item, stop admitting all later waiting requests in that turn to preserve strict FCFS.
   An item larger than the effective budget is admitted so normal validation can fail that request
   instead of permanently blocking the queue.
5. Preserve the existing scheduler-policy order when passing ready requests to the
   MicroBatchScheduler. Do not append newly completed MM requests at the end.

These rules trade some packing efficiency for predictable fairness, stable output ordering, and a
small implementation surface. A best-fit or modality-aware policy can be evaluated later from traces.

### Same-step eligibility

An admitted request is eligible for the current LLM microbatch when every required item is either:

- already `READY`; or
- selected for encoder execution in this same schedule result.

If the request is selected by the MicroBatchScheduler, execution runs its selected encoder items,
assembles the full
embedding, and runs the LLM forward in the same iteration. This preserves the existing MM latency
behavior.

If MM execution completes but the LLM microbatch cannot include the request, its outputs remain in
request-local slots and the request retries LLM scheduling later. Under the default policy, the
CapacityScheduler must have admitted the request before any encoder work is launched; a capacity
rejection does not leave behind speculative encoder output.

## Admission policies

### Default: LLM-coupled admission

The default follows vLLM's essential admission behavior:

- A request rejected by LLM capacity does not independently run the MM encoder.
- A new waiting MM request moves into the active set only if it can make progress: it is already MM
  ready, or its first required atomic item fits the remaining encoder budget.
- A waiting request that cannot make any prefill progress remains waiting and consumes no active LLM
  slot.
- Once admitted and making MM progress, a request may occupy an active LLM slot over several encoder
  iterations until all items are ready.

This coupling protects the LLM from an unbounded set of encoder-only active requests and keeps request
lifecycle semantics close to the existing scheduler. The cost is possible LLM slot occupancy while a
large multi-item request advances through encoder iterations.

The implementation order must avoid encoding a request that the final capacity decision rejects. The
waiting progress check is an eligibility gate, while the existing CapacityScheduler remains the
authority that reserves LLM resources.

### Experimental: independent encoder admission

The following process environment variable enables an A/B experiment:

```text
TLLM_MM_ENCODER_INDEPENDENT_SCHEDULING=1
```

The default is `0`. This is intentionally not a public `LlmArgs` field in the initial experiment.

The implementation also has two diagnostic/rollout environment variables:

```text
TLLM_MM_ENCODER_RUNTIME_SCHEDULING=0  # disable item scheduling; default is 1
TLLM_MM_ENCODER_FORWARD_LOG=1         # log Qwen-style grid item/token counts
```

Independent mode selects encoder items from the existing `active_requests` before LLM capacity
filtering. It lets an already-admitted request continue encoder progress in an iteration where LLM
capacity rejects it; it does not create a separate pre-admission collection or increase the active
request limit. Completed outputs stay request-local until normal LLM scheduling.

Independent mode changes only admission. Both policies use the same serial execution order, item
selection, model hooks, correctness checks, and output assembly, which makes throughput and latency
comparisons meaningful.

Initial compatibility is limited to configurations in which the existing schedule distribution gives
all TP/PP/CP ranks the same selected item list. Attention data parallelism and disaggregated serving
are unsupported in independent mode until ownership, memory accounting, and transfer semantics are
defined. MM side-stream prefetch and overlap compatibility is deferred; the first implementation must
not silently change the existing feature's semantics.

## Schedule distribution

TensorRT-LLM already distributes scheduling results from the scheduling rank. This design extends the
existing payload; it does not add a second collective or broadcast.

`ScheduledRequests` and `SerializableSchedulerOutput` gain an optional mapping:

```python
scheduled_mm_encoder_items: dict[int, list[int]] | None
```

The key is request ID and each value contains canonical item indices. Static costs and item references
arrive through the existing initial `RequestBroadcaster`. The per-iteration payload sends only the
selection. Non-MM schedules leave the field as `None`, so their serialized payload and hot path remain
unchanged apart from backward-compatible field handling.

The rank that already owns scheduling applies the MM policy. Other TP/PP/CP ranks reconstruct the same
request/item selection from the serialized result before model execution. No rank independently
re-packs the items.

## Model execution API

`MultimodalModelMixin` gains an internal item-level hook:

```python
def encode_multimodal_items(
    self,
    selected_items: Sequence[tuple[MultimodalParams, int]],
) -> list[torch.Tensor]:
    """Return one post-connector tensor per selected item, in input order."""
```

The scheduler knows only request IDs, item indices, and costs. The executor resolves the selected
request/item pairs and calls the model. Item slicing happens while the original payload is still on
CPU; only the prepared microbatch is transferred to GPU. The model owns:

- slicing raw model-specific tensors by item;
- determining physical forward compatibility;
- grouping compatible consecutive items;
- constructing exact attention metadata;
- executing the encoder and connector; and
- splitting outputs back into one tensor per selected item.

The executor owns the selected-only H2D transfer, direct copies into the final contiguous output
buffer, and removal of raw modality payloads after all items complete. Completed outputs remain on
GPU for same-step LLM consumption; this design does not offload them.

The existing `encode_multimodal_inputs` remains available for the legacy path and existing prefetch
behavior.

### Physical grouping

Scheduler packing is modality- and shape-agnostic. Current supported models group **consecutive**
selected items by modality; their preparation contracts produce compatible fields, dtype/device, and
shapes within each run. A future model that cannot guarantee this must override the encoding hook or
use a stricter compatibility key. All physical forwards share one per-iteration budget.

Nonconsecutive items are not reordered to make a larger batch. Reordering would require an additional
permutation contract and increases the risk of attaching outputs to the wrong placeholders. The
initial implementation favors correctness and a transparent execution trace.

Qwen2/Qwen3 implementations derive item slices from their grid and patch offsets. Mistral derives
image slices from its image metadata. A video remains one item; it is not transformed from `(t, h, w)`
into independent `(1, h, w)` forwards. That transformation can change temporal attention and output,
so it is outside this design.

### LoRA

The target Qwen and Mistral encoder entry points do not currently accept per-request LoRA parameters;
LoRA is applied on the LLM side. Therefore the initial grouping key does not contain `lora_task_id`.
If encoder/tower LoRA is introduced, the adapter identity must become part of both the physical
grouping key and any cache key before the feature is enabled.

## Attention metadata capacity

`encoder_max_batch_size` counts original items, while `AttentionMetadata.max_num_requests` may count
internal sequences, temporal segments, or attention windows. These values must not be equated. The
generic expression below is invalid because the quantities use different units:

```python
# Do not use.
max(encoder_max_batch_size, encoder_max_num_tokens)
```

Each supported model family provides a static upper-bound hook conceptually equivalent to:

```python
def get_encoder_attention_metadata_capacity(
    self,
    effective_max_num_tokens: int,
    max_batch_size: int,
) -> dict[str, int]:
    ...
```

The returned capacities describe the encoder's actual attention backends. A simple encoder may map
one item to one sequence and return `max_batch_size`. A Qwen encoder may require separate full- and
window-attention sequence bounds derived from its grid rules and token budget.

Startup metadata allocation, profiling/dummy inputs, CUDA graph preparation where applicable, and
runtime validation must all consume the same model-specific capacity. Supported mixin models must
not fall back to a fixed constant such as 8192 or a dimensionally inconsistent conservative maximum.

## Execution ordering and streams

The first implementation is serial from the scheduler's perspective:

```text
selected MM encoder forward group(s)
    -> direct commit into request-local final embedding buffer
    -> selected LLM forward
```

This ordering is required for same-step consumption. It does not imply a new CUDA stream. The design
does not modify the native encoder-decoder `encoder_stream`, and it does not enable MM/LLM concurrent
execution by default. Existing optional MM prefetch or overlap behavior requires a separate
compatibility review because in-flight reservation and output lifetime must be integrated explicitly.

## Error handling and cancellation

Errors before GPU launch, including missing metadata, inconsistent item ordering, an item larger than
the model maximum, and declared/actual cost mismatch, fail the affected request.

A GPU encoder-forward exception follows the executor's conservative fatal/error path. The initial
implementation does not attempt strong per-item fault isolation or continue the same iteration after
a potentially invalid CUDA state. Successfully completed request-local outputs are released when the
request is torn down unless owned by a future cache.

Cancellation removes the request from waiting or active collections and releases request-local output
tensors. A scheduled request that is no longer active is treated as an explicit scheduler lifecycle
error.

## Memory ownership and future cache integration

The initial implementation intentionally has no separate byte or item limit for accumulated
request-local outputs. This is acceptable for the first correctness/performance evaluation but is a
known risk, especially in independent mode where requests can finish encoding before LLM admission.

The following follow-up is required before independent mode becomes a supported default:

1. Track resident MM output embeddings by items and bytes.
2. Reserve resident capacity before scheduling an item, not after its forward.
3. Stop independent selection when the resident budget is exhausted.
4. Release capacity on LLM consumption, cancellation, validation failure, and executor error.
5. Define whether completed outputs remain on GPU, move to pinned CPU memory, or are evicted.
6. Define per-DP-replica ownership and distributed accounting.
7. Add stress tests for waiting-request accumulation and cancellation storms.

A separate cross-request cache can integrate without changing scheduler identity. A hit fills the
corresponding request-local item slot and consumes no encoder item or compute-token budget. Cache
insertion happens only after a successful item result is committed. Before enabling the cache:

- validate canonical item ordering against hashes/UUIDs;
- include model, preprocessing, dtype, and future encoder-adapter identity in the key;
- account cached tensors against resident memory even though they use no compute budget;
- define eviction and reference release on cancel; and
- ensure the cache does not rely on the current one-modality hashing assumption.

## Correctness requirements

Accuracy is a hard invariant. Scheduling may change batching and launch boundaries, but not semantic
item order or model computation.

Required comparisons include:

1. Full-batch encoder output versus micro-scheduled, request-local reassembly for each supported model.
2. Multiple images with different shapes and multiple physical forward groups.
3. Mixed image/video prompts in placeholder order.
4. A request spanning multiple encoder scheduling iterations.
5. Several requests backfilled under both item and token pressure.
6. Complete precomputed embeddings bypassing the encoder budget.
7. Cancellation while an item is pending and while it is in flight.
8. Distributed schedule serialization and reconstruction.
9. Default versus independent admission with identical encoder output and final greedy tokens.

Tests compare encoder outputs using dtype-appropriate numerical tolerances. Bitwise equality is not
always possible when a kernel sees a different physical batch, but deterministic greedy generation
must produce the same tokens. A tolerance-only encoder test is insufficient if the end-to-end greedy
result changes.

No implementation may split Qwen video temporal units merely to satisfy a budget without a separate
model-level proof and accuracy suite.

## Performance and observability

The implementation should expose enough information to understand utilization without logging every
item at normal verbosity:

- configured and effective encoder token budget;
- per-step selected item count and physical token count;
- number of physical encoder forward groups;
- requests blocked by atomic item/token budget;
- request-local ready bytes, especially in independent mode;
- selected H2D bytes and peak allocated memory around each encoder forward;
- waiting time to first encoder item and time from final item to first LLM prefill;
- cache hits/misses when cache integration lands; and
- whether the experimental independent policy is enabled.

Performance evaluation should compare at least:

- non-MM throughput and latency before/after, which should be statistically unchanged;
- current full MM batch behavior against default item scheduling;
- default coupled admission against independent admission;
- single-item and multi-item prompts;
- compatible and incompatible item shapes; and
- encoder-heavy workloads mixed with long LLM prefill/decode workloads.

The primary A/B question is not only encoder utilization. It is whether independent encoder progress
improves end-to-end latency or throughput after accounting for resident-output memory and contention
with LLM work.

## Implementation plan

### Phase 1: metadata and model contracts

1. Finalize physical token accounting helpers for Qwen2/2.5, Qwen3, and Mistral3/Pixtral.
2. Emit `multimodal_encoder_token_lengths` and `multimodal_item_refs` in canonical prompt order.
3. Add CPU-only metadata registration and request validation.
4. Add model-specific attention metadata capacity hooks and use the effective budget everywhere.
5. Add `encode_multimodal_items` implementations with consecutive compatibility grouping.

### Phase 2: request-local state and schedule payload

1. Add immutable MM request classification at request attachment time.
2. Add fixed output slots and in-flight item reservations.
3. Extend `ScheduledRequests` and `SerializableSchedulerOutput` with optional selected item IDs.
4. Reconstruct the same item selection on all participating ranks.
5. Add cancellation and error cleanup.

### Phase 3: default runtime enforcement

1. Construct the MM wrapper only for `MultimodalModelMixin` models.
2. Add FCFS waiting progress eligibility without changing CapacityScheduler authority.
3. Pack atomic items for admitted/active requests under both budgets.
4. Filter partial requests before MicroBatchScheduler while preserving policy order.
5. Execute selected items and preserve same-iteration LLM forward.
6. Slice items before H2D and transfer only the selected microbatch.
7. Assemble outputs directly into one contiguous final buffer without `torch.cat`.
8. Drop raw encoder payloads after completion so LLM preparation cannot transfer them again.
9. Verify no non-MM regression and no native encoder-decoder behavior change.

### Phase 4: experimental independent admission

1. Add the environment-variable gate, default off.
2. Select items from admitted active requests before LLM capacity filtering.
3. Reuse the same packing, execution, assembly, and correctness path.
4. Keep admission and the active-request capacity unchanged.
5. Run the coupled-versus-independent A/B matrix before considering a public option.

### Phase 5: deferred hardening

1. Resident-output memory budgeting, offload, and eviction.
2. Cross-request encoder cache integration.
3. MM overlap/prefetch compatibility.
4. Optional packing policies beyond FCFS/backfill.
5. Item-internal chunking only where the model explicitly proves equivalence.

## Rejected alternatives

### Charge placeholder/output embedding lengths

Rejected because a merger can reduce physical encoder tokens before insertion into the LLM. The
resulting budget would not control attention workspace or encoder compute.

### Add encoder tokens to the LLM token budget

Rejected because it directly reduces LLM scheduling capacity using a different compute domain and
unit. MM gets a separate per-iteration budget. Default admission remains LLM-coupled for lifecycle and
capacity correctness, not because MM tokens are charged as LLM tokens.

### Run encoder only after all LLM scheduling

Rejected as a general default because it cannot prevent an oversized encoder forward and makes
same-step readiness hard to express. It also risks doing work for a request that LLM capacity did not
admit.

### Reuse `ENCODER_INIT` or add `MM_INIT`

Rejected because native encoder-decoder state transition semantics defer LLM work, while existing MM
requests can encode and prefill in the same iteration. Python readiness and item selection are enough.

### Split video into temporal items

Rejected for the initial design. Independent `(1, h, w)` forwards need not equal one `(t, h, w)`
forward because temporal attention and positional treatment can change. An original video remains
atomic.

### Put encoder-only requests in `active_requests`

Rejected for independent mode because it can exceed LLM capacity, increase scheduler scans, distort
metrics, and interfere with KV/resource lifecycle. A separate collection isolates the experiment.

### Reorder all compatible items

Rejected initially because output permutation and placeholder association become more complex.
Consecutive grouping captures a useful batching case while preserving an obvious order contract.

### Generic attention capacity fallback

Rejected because item count, physical token count, and internal sequence count have different units.
Each supported family must provide its own bound.

## Open and deferred questions

The following are intentionally not required to land the first runtime-enforcement change:

1. What resident-output byte limit and eviction/offload policy should independent mode use?
2. How should MM prefetch and overlap scheduler share item reservations and CUDA streams?
3. Should a later public configuration replace the experimental environment variable?
4. Is strict FCFS waiting admission preferable after production traces, or should a bounded bypass or
   best-fit policy be added?
5. Which model families can prove semantically equivalent item-internal chunking?
6. How should a cross-request cache coordinate ownership across DP and disaggregated deployments?
7. When encoder/tower LoRA is supported, how should adapter residency interact with batching and cache
   keys?

## Reference implementation map

### TensorRT-LLM

- [PR #13503](https://github.com/NVIDIA/TensorRT-LLM/pull/13503): configuration controls, deterministic
  dummy sizing, and the explicit runtime-scheduler follow-up.
- [`llm_args.py`](../../../tensorrt_llm/llmapi/llm_args.py): user-facing encoder limits and LLM fallback
  resolution.
- [`multimodal.py`](../../../tensorrt_llm/inputs/multimodal.py): `MultimodalInput`,
  `MultimodalRuntimeData`, and CPU-only MM fields.
- [`registry.py`](../../../tensorrt_llm/inputs/registry.py): input processing, placeholder metadata, and
  MM hashing.
- [`modeling_multimodal_mixin.py`](../../../tensorrt_llm/_torch/models/modeling_multimodal_mixin.py):
  MM model contract and existing full-request encoder path.
- [`modeling_multimodal_encoder.py`](../../../tensorrt_llm/_torch/models/modeling_multimodal_encoder.py):
  shared encoder sizing and validation support.
- [`modeling_qwen2vl.py`](../../../tensorrt_llm/_torch/models/modeling_qwen2vl.py),
  [`modeling_qwen3vl.py`](../../../tensorrt_llm/_torch/models/modeling_qwen3vl.py), and
  [`modeling_mistral.py`](../../../tensorrt_llm/_torch/models/modeling_mistral.py): initial model-family
  implementations.
- [`scheduler.py`](../../../tensorrt_llm/_torch/pyexecutor/scheduler/scheduler.py): Python scheduling
  interfaces, `ScheduledRequests`, and serialized output.
- [`py_executor.py`](../../../tensorrt_llm/_torch/pyexecutor/py_executor.py): request lifecycle,
  scheduling, and distributed orchestration.
- [`model_engine.py`](../../../tensorrt_llm/_torch/pyexecutor/model_engine.py): MM encoder and LLM model
  execution.
- [`llm_request.py`](../../../tensorrt_llm/_torch/pyexecutor/llm_request.py): Python request metadata and
  helpers.
- [`request_utils.py`](../../../tensorrt_llm/_torch/pyexecutor/request_utils.py): initial request
  broadcast.
- [PyTorch scheduler documentation](scheduler.md): CapacityScheduler and MicroBatchScheduler roles.

### vLLM comparison

The following vLLM files are design references, not APIs that TensorRT-LLM must copy. They were
inspected to understand the current item-level scheduling model:

- [V1 scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py):
  per-step encoder budget, atomic item selection, waiting/running admission, and selected item IDs.
- [Encoder cache manager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/encoder_cache_manager.py):
  selection-time allocation and request cleanup.
- [V1 request](https://github.com/vllm-project/vllm/blob/main/vllm/v1/request.py): per-item encoder
  embedding counts and computed-item tracking.
- [Input processor](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/input_processor.py):
  per-item feature construction and placeholder association.
- [Multimodal inputs](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py) and
  [multimodal utilities](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/utils.py): item
  specifications and prompt-position ordering.
- [Encoder budget](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/encoder_budget.py) and
  [scheduler configuration](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py):
  effective per-item budget and configured defaults.
- [GPU model runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py):
  selected-item execution, compatibility grouping, and encoder-before-LLM same-step ordering.

Important differences from vLLM are intentional. vLLM commonly charges post-encoder embedding spans
for its encoder compute budget, while this design charges physical pre-connector attention tokens to
match TensorRT-LLM's user-facing knob and workspace sizing. TensorRT-LLM also keeps its existing C++
LLM CapacityScheduler as the resource authority and adds a narrowly gated Python MM pre-pass.
