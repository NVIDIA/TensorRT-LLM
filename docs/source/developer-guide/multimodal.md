<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Multimodal Request Data

Multimodal preprocessing and execution share `MultimodalParams`, defined in
[inputs/multimodal.py](source:tensorrt_llm/inputs/multimodal.py).

| Type or member | Responsibility |
|---|---|
| `MultimodalParams.multimodal_data` | Processed model inputs, embeddings or shared-tensor handles, and model-specific auxiliary data. |
| `MultimodalParams.item_metadata` | Reads the single `MultimodalItemMetadata` record in `multimodal_data`. |
| `MultimodalItemMetadata` | Prompt-ordered item references, optional encoder costs and output row counts, and input partitions. |
| `MultimodalItemPartition` | Describes one field's item axis and optional packed lengths, inside `item_metadata.input_partitions`. |
| `MultimodalInput` | Hashes, prompt positions, and exact runs converted into existing C++ request fields. |
| `MultimodalRuntimeData` | Derived information for the current LLM forward or prefill chunk. |
| `MultimodalEncoderRequestState` | Request-owned readiness and cache references; reads lengths from the admitted item metadata. |

## Process Boundaries and Ownership

```text
Main / input processor
  -> prompt token IDs + MultimodalParams (processed data and item metadata)
Worker handoff
  -> restore tensors; convert MultimodalInput to C++ request fields
Executor / LlmRequest.py_multimodal_params
  -> admission creates encoder_state where item scheduling is supported
  -> encoder selection creates Params views with selected data and metadata
  -> LLM forward creates Params views with embeddings and chunk runtime
  -> executor cleanup releases request-owned cache references
```

Only the request Params own `encoder_state` cache references. Execution views
leave that state unset, and request Params leave `multimodal_runtime` unset.
The model owns the `TensorLRUCache`; the executor owns request cleanup.
Text-only requests do not need a Params aggregate. Live cache references and
CUDA events stay in the executor rather than the cross-process payload.

The worker clears the processor-side `multimodal_input` after converting it to
C++ fields. Persistent encoder caching lazily reconstructs one Python
`MultimodalInput` per request. Selected inputs copy this wrapper before slicing
its hashes, and prefill/termination cleanup clears it.

Inline encoding and prefetch write encoder outputs back to the request.
Item-scheduled LLM chunks keep their assembled embeddings local to the forward.
Execution paths retain their CUDA event synchronization responsibilities.

## Item Metadata and Selection

`item_refs` identifies processed items by modality and modality-local index, in
prompt order. A processed item is not necessarily one original media object or
one tensor. Prompt token counts, physical encoder token costs, and output
embedding rows are distinct quantities.

Input preparation publishes the common record directly. The frontend's
`mm_item_order` is consumed during preprocessing; execution reads only
`item_metadata.item_refs`. Output row counts live in
`item_metadata.output_embedding_lengths`. Admission, full-request and selected-item
encoding, cache keys, and encoder output export all read the same record.
Encoded result views reuse the same item record; the transport projection exports per-item lengths.

Item refs, costs, and output rows are immutable tuples; the constructor also
accepts lists. Encoder state references this record without copying its lengths.
Producers fill input partitions before admission and validate contracts at
producer/admission boundaries, avoiding repeated validation per forward.

Partitions describe how to select processed fields: shared fields have no item
axis, batched fields have an item axis, and packed fields also provide host-side
per-item lengths. Contiguous tensor selections can share storage; reordered
selections may copy. Treat shared input storage as read-only. Model hooks declare compatible inputs, padding, masks, and encoder arguments
through `EncoderGroup`; the common dispatcher batches and restores item order.

## Processor Hooks

Processors implement
`get_mm_item_metadata(prompt_token_ids, multimodal_data, mm_item_order=None)` and
return `MultimodalItemMetadata`. Generic preparation supplies prompt order and
known output rows; model processors add encoder costs and input partitions.
Partial model records retain known prompt rows. Conflicting order or rows fail
at input preparation. Gemma4 and Nemotron add partitions; Qwen and Mistral also
provide physical encoder costs.

Unknown costs or row counts remain `None`. Common metadata does not enable item
scheduling: models can supply order and output rows with
`encoder_token_lengths=None` and keep full-request encoding. Model capability
and configuration decide whether items are scheduled separately.

Custom processors must use this common hook and type. The former
`get_mm_encoder_item_metadata` hook, `MultimodalEncoderItemMetadata` tuple, and
Params order/length fallbacks have been removed.

## Onboarding a Model

Follow [Adding a New Model](../models/adding-new-model.md) for model definition,
weight loading, and registration. This section covers the additional multimodal
processor and execution contracts.

Start with full-request encoding and add item scheduling only when the model
can encode its items independently. Use these contracts rather than inventing
encoder costs or inferring item axes from matching tensor sizes.

| Capability | Required information |
|---|---|
| Common request metadata | `item_refs` in prompt order. Unknown encoder costs and output rows stay `None`. |
| Grouped output assembly | Exact `output_embedding_lengths`; `EncoderGroup` needs them to split and reorder rows. |
| Item scheduling | Exact positive `encoder_token_lengths`, output row counts, and an item-selection implementation. |
| Persistent item caching | Output row counts and stable content/processor hashes. Partial hits also require correct item selection. |

Startup profiling calls `get_dummy_mm_data` directly, bypassing the normal
processor wrapper. For grouped encoders, its returned data must include the
common `MultimodalItemMetadata` record with item references and exact output
row counts.

For example, a processor for a full-request audio encoder can return:

```python
return MultimodalItemMetadata(
    item_refs=[("audio", 0)],
    encoder_token_lengths=None,
    output_embedding_lengths=[num_output_rows],
)
```

This supports output assembly without declaring schedulable encoder costs.
If output row counts are unknown too, the model must resolve them before using
grouped assembly or item caching. A model-specific full-request encoder can
compute those counts from its result. Metadata alone never enables scheduling.

To opt in, the model sets `supports_mm_encoder_item_scheduling = True` and
implements `encode_multimodal_inputs` and item selection. The engine also
checks `disable_mm_encoder`, the scheduling policy, and supported combinations
in `mm_item_scheduling_enabled()` and
`validate_mm_encoder_scheduling_compatibility()`. Under item scheduling, missing
costs or output rows fail admission; they do not silently switch one request to
full-request encoding. Keep the capability disabled until the model can supply
the complete scheduling contract.

### Selecting Unusual Payloads

Declare `input_partitions` for ordinary batched, packed, and shared fields.
For example, two audio items with two and one clips use:

```python
input_partitions = {
    "audio": {
        "input_audio_features": MultimodalItemPartition(item_lengths=(2, 1)),
        "feature_attention_mask": MultimodalItemPartition(item_lengths=(2, 1)),
        "audio_num_clips": MultimodalItemPartition(),
    },
}
```

If a payload has multiple item axes, nested variable-length structures, or
selection that must rebuild sibling fields, override
`build_multimodal_encoder_input(param, local_item_indices, modality)`. Return a
new execution `MultimodalParams` view and leave the request's payload and cache
references intact. The mixin then slices the common item metadata and hashes
using prompt-global indices. Payload indices and prompt indices can differ:
for `[('video', 0), ('image', 0)]`, the image has local index `0` and prompt
index `1`.

Returning `None` from `get_multimodal_item_partitions` uses the existing
packed/stacked compatibility selector. New models should declare partitions or
override selection explicitly; the compatibility selector cannot describe an
arbitrary model layout. An empty partition dict means every field is shared,
so it is appropriate only when selecting an item requires no payload changes.

### Item Boundaries and Model Hooks

An item is the smallest independently encodable unit, not necessarily one
uploaded media object. Several crops, frames, or audio clips can belong to one
item. A common item reference names one modality. Keep coupled audio/video
data in one model-owned payload under that item's modality, or keep full-request
encoding if independent item execution is not possible. Separate audio and
video references imply that either may be selected independently.

Use `after_full_multimodal_embeddings` for request-level preparation after
cache lookup or encoding. Its tensors are not always full-request tensors: a
mixed batch containing item-scheduled chunk outputs already gathers active
rows. Use `after_active_multimodal_embeddings` for row-dependent transforms,
such as splitting Qwen3-VL deepstack features, immediately before fusion.

Model-specific auxiliary values belong in `multimodal_data`. Declare their
selection rules alongside the primary fields, or rebuild them in the custom
selection hook. Add a field to `MultimodalItemMetadata` only when shared
preprocessing/execution consumers need it, and update validation, `select()`,
producer merging, and process-handoff tests together. Keep lengths and
partitions as host metadata; CUDA-to-host copies can synchronize even when
written as `.cpu().tolist()`.

## Encoder Results and Batching

Physical tower functions return tensors. The outer
`encode_multimodal_inputs(params)` returns one encoded `MultimodalParams` view
per input view, in the same order. Each result holds the primary
`multimodal_embedding`, item metadata, and model data needed to consume it.
`forward_multimodal_encoder_items()` returns one such view per selected item.
These results borrow output storage and do not retain raw media, input tensors,
request cache references, events, or chunk runtime.

Use `encode_multimodal_by_groups()` for shared assembly, or
`split_multimodal_encoder_outputs()` around an existing numerical implementation.
Keep dependent audio/video inputs together in their model-owned payload.
`EncoderGroup` describes a physical encoder call; `MultimodalEncoderMixin`
continues to own a tower's attention metadata and CUDA graph lifecycle.

An optional `batch_key` separates incompatible padded shapes or masks. An
optional `max_num_tokens` and host-side `get_num_tokens` bound compatible
batches by encoder work, independently of output rows. Gemma4 uses the resolved
encoder token budget and padded input shapes for this purpose. Atomic input
admission remains responsible for limits on one input view. Grouping does not
make an indivisible view independently schedulable.

## Encoder/Prefill Handoff

The execution boundary associates encoded views with request hashes and,
when available, the already processed whole-request prompt. It owns the output
storage before releasing temporary communication buffers.
`export_multimodal_params()` projects this result into
`DisaggregatedParams.multimodal`:

```text
schema_version: 1
items: [item_ref, named embedding handles, output rows, optional hash/model data]
request_layout: optional finalized prompt IDs, MM positions/lengths/runs, model data
decode_inputs: optional model-required generation state
```

Items may carry named auxiliary tensors in addition to the primary embedding;
their shapes need not share its leading axis. Qwen3-VL keeps its existing packed
DeepStack feature width. The current item cache stores the primary tensor, so
models must pack cache-required features there or provide their own cache adapter.
Raw media, runtime objects and cache owners stay local.
Tensor handles use the existing shared-tensor transport: CUDA IPC for CUDA
tensors or host shared memory for CPU tensors. This schema alone does not add
cross-host transfer support.

P imports a completed `request_layout` directly and skips media processing and
prompt expansion. Item-only results instead call
`finalize_multimodal_prompt(inputs, params, mm_hashes=...)`, which returns
`(prompt_token_ids, MultimodalParams)`. Model processors implement the local
`_finalize_multimodal_prompt` expansion hook where needed; transport and handle
import stay outside that hook. Qwen image-only item results use encoder geometry
for M-RoPE; external image embeddings without geometry keep their existing grid
inference. Qwen video handoff requires the completed request layout and M-RoPE
state. EVS finalization still uses model-owned retained
counts and preserves split runs.

For P/D, `prepare_multimodal_decode_inputs()` selects only the model state D
needs. P exports its own retained state rather than forwarding E-owned handles.
Existing HTTP P/D multimodal span fields remain supported and are derived from
the finalized layout when present.

External E/P orchestrators must consume the versioned `multimodal` payload;
the former flat E/P embedding/hash/mrope fields and
`build_disagg_prefill_multimodal_inputs()` hook have been removed. Deploy the
matching Dynamo adapter with this change. Aggregated TRT-LLM execution remains
independent of Dynamo. These contracts do not enable unsupported models,
modalities, or item scheduling combinations.
