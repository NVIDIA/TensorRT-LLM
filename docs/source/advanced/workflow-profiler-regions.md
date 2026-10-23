<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# PyTorch inference workflow regions

Workflow scopes are selected by `TLLM_PROFILE_CUSTOM_REGIONS` and use the shared
[profiler metadata contract](profiler-metadata.md). They are enabled by the default
selection, but emit events only while a PyTorch profiler is active. Selection is
cached at process startup; `none` disables custom regions. Existing profiling
endpoints and executor-thread start/stop behavior are unchanged.

## Insertion points

These are nested workflow boundaries, not replacements for the existing
`step[...]` iteration annotations. Count only bare `step[...]` labels as iteration
boundaries; the overlap scheduler's suffixed `step[...] sample` label remains a
separate sub-iteration scope.

| Region | Current owner and boundary |
| --- | --- |
| `trtllm::prepare_batch` | `PyExecutor._prepare_and_schedule_batch`: request fetching and executor-side batch preparation |
| `trtllm::schedule` | `PyExecutor._schedule`: scheduler invocation and scheduled-request conversion |
| `trtllm::forward` | `PyExecutor._forward_step`: stream coordination and existing forward lifecycle |
| `trtllm::prefill`, `trtllm::decode`, `trtllm::mixed` | Inside `_forward_step`, only around the model-engine forward invocation |
| `trtllm::sampling` | `PyExecutor._sample_async`: asynchronous logit handling and sampling |
| `trtllm::update_requests` | `PyExecutor._update_requests`: sampled request-state updates |
| `trtllm::handle_responses` | `PyExecutor._handle_responses`: response construction, enqueueing, and finalization |
| `trtllm::prepare_inputs` | `DecoderInputPreparer.prepare_inputs`: scheduled input and attention-metadata preparation |
| `trtllm::model_forward` | `DecoderRunner.model_forward`: current model-call boundary |
| `trtllm::cuda_graph_capture` | Decoder `CUDAGraphRunner.capture` and `EncoderCUDAGraphRunner.capture`: warmup and capture |
| `trtllm::cuda_graph_replay` | Both graph runners' `replay`: static-input staging and graph submission |

The phase uses the current scheduled context-token count and generation-request
count: positive context tokens alone yield `prefill`, generation work alone yields
`decode`, and both yield `mixed`. Chunked context contributes only the scheduled
chunk's tokens, not the full prompt length. An empty forward follows the decode
fallback. Router Replay preparation, finish/abort, and hang diagnostics retain
their existing ordering and ownership.

## Tensor metadata and graph limitations

`record_function_with_metadata` supports contexts and decorators. Selected active
scopes lazily inspect only directly supplied tensors, including a decorated call's
positional and keyword arguments. Each tensor contributes `workflow.tensor.N.shape`,
`.dtype`, `.stride`, and `.device`; `workflow.tensor_count` reports the number of
direct tensors. Any tensor shape/dtype is accepted. Containers are not traversed,
and tensor contents are not materialized or copied. Disabled/inactive scopes do
not build metadata. Shared observational setup/cleanup keeps results and body
exceptions unchanged. Profiling still adds measurable Python and profiler overhead.

Owner boundaries can supply a typed `metadata_factory` in addition to direct
tensors. A context factory takes no arguments; a decorator factory receives the
wrapped call's arguments. Its fields merge after direct-tensor fields, inside the
same lazy observational setup. This is not a recursive serializer: request
contents, request identifiers, attention objects, nested model inputs and device
tensor values are not serialized.

## Scheduled logical work and executor coordinates

`trtllm::forward` and its `prefill`/`decode`/`mixed` child emit the following
fields from the owning executor's host state. Counts describe the boundary before
the model engine adds any further graph-padding requests, not graph occupancy.

| Field | Meaning and source |
| --- | --- |
| `workflow.executor.iteration` | Existing `iter_counter`; rank-local executor iteration, not a globally synchronized step ID |
| `workflow.executor.microbatch_slot` | Actual PP ring `microbatch_id`, propagated by both intermediate and last-stage forward paths; reused across iterations, not a globally unique batch ID; `0` for known PP=1; omitted if unavailable |
| `workflow.executor.is_warmup` | Existing model-engine warmup flag, or the decoder runner's `is_dummy` flag at the model call |
| `workflow.attention_dp_enabled` | Existing executor `enable_attention_dp` host bool, when available; TP size alone does not identify attention-DP layout |
| `workflow.rank`, `workflow.world_size` | Existing `dist.mapping` model rank and world size |
| `workflow.tp_rank`, `workflow.tp_size`, `workflow.pp_rank`, `workflow.pp_size`, `workflow.cp_rank`, `workflow.cp_size` | Existing Mapping coordinates/sizes for TP, PP and CP |
| `workflow.moe_ep_rank`, `workflow.moe_ep_size` | Existing Mapping MoE expert-parallel coordinates/size, not a separate inferred global EP group |
| `workflow.batch.scheduled_context_requests`, `workflow.batch.scheduled_generation_requests` | Current context/generation list lengths, including any dummies already present |
| `workflow.batch.live_context_requests`, `workflow.batch.live_generation_requests`, `workflow.batch.live_requests` | Current request-list entries excluding the union of `is_attention_dp_dummy`, `is_cuda_graph_dummy` and `is_dummy_request`, matching `LlmRequest.is_dummy`; total is the sum of the two live counts |
| `workflow.batch.attention_dp_dummy_requests`, `workflow.batch.cuda_graph_dummy_requests`, `workflow.batch.generic_dummy_requests` | Entries carrying each explicit flag (`generic_dummy_requests` is the `is_dummy_request` flag used by synthetic warmup requests); categories can overlap, so their counts must not be summed as a unique dummy total |
| `workflow.batch.scheduled_context_tokens`, `workflow.batch.live_context_tokens` | Sum of current `context_chunk_size` across all context entries or only live context entries; not full prompt lengths |

Unavailable coordinates/state are omitted, never replaced with numeric sentinels.
There is no concrete aggregate worker-role state at these owners, so no worker
role is inferred from requests or a lazily initialized disaggregation coordinator.
There is no configured max-batch or client-concurrency occupancy field. Scheduled
logical counts are rank-local views: TP/PP ranks may share logical work, while CP
can change physical input rows. Do not sum logical counts across ranks without
accounting for that layout.

`trtllm::model_forward` additionally emits `workflow.model.input_token_rows` from
the provided direct `input_ids.shape[0]`, when present, and the engine's
`workflow.executor.is_warmup`. This is the actual physical packed input extent at
the model call, potentially including padding. Embed-only calls with no
`input_ids` omit the row field. Speculative decoding and beam expansion can
produce multiple rows per generation request; rows are not live request counts.

## CUDA graph metadata

Both graph owners emit `workflow.graph.kind` (`decoder` or `encoder`),
`workflow.graph.operation` (`capture`, `warmup_only` or `replay`),
`workflow.graph.padding_enabled` (configured padding), and
`workflow.graph.warmup_steps` (the owner's warmup count for capture/warmup-only,
zero for replay). A `capture` boundary includes its eager warmups and capture;
it does not claim each child operation was captured. `warmup_only` explicitly
identifies the existing path that returns without creating a graph.

Decoder fields are:

| Field | Meaning and source |
| --- | --- |
| `workflow.graph.key.batch_size`, `.draft_len`, `.short_seq_len_mode`, `.is_all_greedy_sample`, `.sample_type`, `.num_contexts`, `.context_query_len`, `.num_encoder_tokens`, `.peft_cache_data_type`, `.use_lora_graph` | Named dimensions of the existing decoder `KeyType`; sampling enum uses its name and optional PEFT dtype uses its string or `None` |
| `workflow.graph.batch_capacity` | Key batch-size request capacity, not a live request count |
| `workflow.graph.max_beam_width` | Owner's configured beam width used in the existing graph token-capacity formula, not observed live beam occupancy |
| `workflow.graph.token_capacity` | Existing `_get_num_tokens_for_key`: `num_contexts * context_query_len + (batch_size * max_beam_width - num_contexts) * (draft_len + 1)` |
| `workflow.graph.input_token_rows` | Supplied `input_ids.shape[0]` at capture/replay entry, when present; capture then substitutes static capacity-shaped tensors |
| `workflow.graph.encoder_hidden_state_rows` | Supplied `encoder_hidden_states.shape[0]` for mixed encoder-decoder input, when present; may be shorter than the key's encoder extent |

Decoder input rows may already be padded. Neither live occupancy nor actual
padding counts can be recovered from these tensors, so decoder graph boundaries
omit those fields. Correlate each replay with its parent forward/phase logical
counts: the same graph key can serve different live batches. Capture inputs can
be all dummy requests; capacity is never evidence of live occupancy.

Encoder fields remain distinct from decoder generation semantics:

| Field | Meaning and source |
| --- | --- |
| `workflow.graph.key.padded_batch_size`, `.padded_total_tokens`, `.max_seq_len_bucket` | The existing three-tuple encoder graph key, including feature-mode fixed-sequence buckets |
| `workflow.graph.batch_capacity`, `workflow.graph.token_capacity` | First and second encoder key dimensions |
| `workflow.graph.input_token_rows` | Supplied packed `input_ids.shape[0]`, when present |
| `workflow.graph.input_feature_shape` | Supplied `input_features.shape`, when present; features are not decoder token rows |
| `workflow.graph.source_requests`, `workflow.graph.source_tokens` | Length and sum of retained host `_encoder_source_seq_lens`, only when available as an integer list |
| `workflow.graph.padding_requests`, `workflow.graph.padding_tokens` | Encoder key capacity minus those retained source counts; graph slots/token positions without source work, not live decoder padding |

Encoder source counts may themselves include executor dummies; they are not
claimed as logical live counts. Without retained original host lengths the source
and actual-padding fields are omitted. No device tensor reads, synchronization,
new communication or shared iteration counters are introduced by extraction.

CUDA work is asynchronous. CPU durations describe workflow/submission time, not
GPU execution time; use correlated GPU events to measure execution. Python scopes
inside a captured model run during warmup/capture and do not re-execute on replay.
The replay annotation covers staging and submission, and does not establish fresh
per-operation Python metadata or a GPU synchronization boundary.
