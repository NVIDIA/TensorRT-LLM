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
The encoder-result boundary exports lengths as lists.

Item refs, costs, and output rows are immutable tuples; the constructor also
accepts lists. Encoder state references this record without copying its lengths.
Producers fill input partitions before admission and validate contracts at
producer/admission boundaries, avoiding repeated validation per forward.

Partitions describe how to select processed fields: shared fields have no item
axis, batched fields have an item axis, and packed fields also provide host-side
per-item lengths. Contiguous tensor selections can share storage; reordered
selections may copy. Treat shared input storage as read-only. Model hooks and
`EncoderGroup` still own compatible batching, padding, masks, encoder arguments,
and output fusion.

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

## Encoder/Prefill Handoff

For encoder/prefill disaggregation (E/P), model processor hooks return
`(prompt_token_ids, MultimodalParams)`. Use `MultimodalParams.from_embeddings`
to attach embeddings or shared-tensor handles. See
[inputs/registry.py](source:tensorrt_llm/inputs/registry.py) for the hook contract.

`DisaggregatedParams` remains the transport envelope used by external
orchestrators such as Dynamo. The prefill processor still rebuilds model-specific
prompt layout; its local Params result does not replace the wire protocol.
Custom E/P hooks must follow the tuple return contract. Review Python hook
compatibility separately from transport fields when changing this interface.

These contracts do not change the supported model/modality matrix or impose a
common batching policy on every model.
