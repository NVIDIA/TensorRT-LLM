# ModelingV2 runtime features: reference

How to read this, and what to do when a path below has moved, is in `SKILL.md`.
`_torch/` means `tensorrt_llm/_torch/`. Flow paths (`checks/`, `spec/INTAKE.md`,
`task.example.yaml`) are relative to `agent-flow/agent_flow/workflows/trtllm_modeling_v2/`.
Integration node ids are relative to the integration tests' `defs/` directory
and need the checkpoints its conftest's `llm_models_root` resolves. Every flag named here
exists in `examples/llm-api/quickstart_advanced.py`, `trtllm-serve`,
`trtllm-eval` or `trtllm-modeling-v2 gate`.

## 0. Common machinery

- **Turning features on in a spec.** `runtime` keys `cuda_graph`,
  `chunked_prefill`, `overlap_scheduler`, `kv_cache_reuse`, `extra_flags`;
  feature units in `features`; configuration `variants` overlay `runtime`,
  `llm_api_options` or `env` on a unit. `task.example.yaml` documents each key,
  `spec/INTAKE.md` says which variant each request needs.
- **Running one check.** `trtllm-modeling-v2 gate <boot|accuracy|acceptance|serve|disagg>
  --task task.yaml --unit <unit or variant> --member <m>`; `--mode off` (boot
  only) runs the stock path; `--tree <dir>` measures a worktree as it is;
  `--json` prints the verdict.
- **What a gate knows of the configuration.** A gate sends the unit's options
  to the engine and does not read back what the engine resolved.
  `modeling_v2="require"` makes the engine raise rather than fall back to its
  own implementation. Block reuse, chunked prefill and speculation each have
  an observation of their own (sections 1, 2 and 7). Whether a cache dtype, a
  CUDA-graph list or a width took effect is in the engine's own log and in
  the benchmark score, nowhere else.
- **Step contract.** A target that implements the `step_args` method checks the
  metadata it consumes on its first forward when `TRTLLM_MODELING_V2_VALIDATE=1`
  reaches the ranks (pass it through `env_overrides`); see `_router_index.py`.
- **What combines with what** in the engine:
  `docs/source/features/feature-combination-matrix.md`.

## 1. Block reuse (prefix KV-cache reuse)

**What.** Filled KV blocks enter a radix tree; a later request with a matching
prefix resumes from them instead of recomputing. Partial reuse can copy a
partly matched block (`docs/source/features/kvcache.md`).

**Engine owns.** The cache manager: allocation, matching, eviction, host
offload, free and partial-reuse copies. `KVCacheManagerV2`
(`_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`) is the default
(`use_kv_cache_manager_v2` defaults true); `KVCacheManager` in
`resource_manager.py` is the V1 C++ manager. `get_kv_cache_manager_cls`
(`_torch/pyexecutor/_util.py`) picks the class from the model config (hybrid
recurrent, sparse attention, FP4 MLA). Hybrid recurrent models get
`MambaHybridCacheManagerV2` (`kv_cache/mamba_cache_manager/`), which restores
recurrent state only at snapshot boundaries set by
`kv_cache_config.mamba_state_config`; with no policy the loader turns reuse off
and logs "Disabling KV cache block reuse for the hybrid Mamba model".

**Target must honour.**
- A context request may start at a cached offset. Positions, RoPE and the
  attention call take each sequence's cached length from the metadata
  (`num_cached_tokens_per_seq` in `KVCacheParams`), and attention reads the
  prefix from the pool: `TrtllmAttentionMetadata` turns `use_paged_context_fmha`
  on whenever reuse, chunking or draft tokens are configured.
- Per-sequence state beyond K/V (indexer keys, block scales, compressed or
  recurrent state) must live in manager-owned storage so allocation, free,
  slot reuse, prefix reuse and partial-reuse copies apply to it: a V2 buffer
  role registered through the manager's `_extra_buffers_per_layer` hook (`Role`
  in `kv_cache_manager_v2.py`: `INDEX_KEY`, `KEY_BLOCK_SCALE`, ...), or the
  recurrent pool with its snapshots. State in target-owned tensors keyed by
  request is stale after a reuse hit, and no partial-reuse copy carries it.
- With speculative decoding, state written per draft token must be committed
  for the accepted count (`AuxiliarySpeculativeStateHandler`, section 7).
- Nothing may hinge on "first time this request is seen": a resumed request
  never computes its cached prefix.

**Where.** `_torch/pyexecutor/kv_cache/`, `_torch/pyexecutor/resource_manager.py`,
`_torch/pyexecutor/kv_cache_stats.py`, `_torch/attention/backends/fmha/fallback.py`
(the paged-context kernel gate); `enable_block_reuse`, `enable_partial_reuse`,
`copy_on_partial_reuse`, `use_kv_cache_manager_v2`, `BlockReusePolicy`,
`_extra_buffers_per_layer`, `MambaStateConfig`, `periodic_snapshot_interval`,
`use_paged_context_fmha`, `num_cached_tokens_per_seq`, `cached_tokens`, `cache_salt`,
`iterReusedBlocks`.

**Tests to imitate.**
- An op over a cached prefix: the cells with a non-zero `cached` length and
  `use_paged_context_fmha` on in
  `tests/unittest/_torch/modeling_v2/attention/test_modeling_v2_thop_attention.py`.
- Side buffers sharing the K/V lifecycle:
  `tests/unittest/_torch/pyexecutor/kv_cache/test_kv_cache_v2_extra_buffers.py`;
  recurrent state: `tests/unittest/_torch/pyexecutor/kv_cache/test_mamba_cache_manager.py`.
- A silently dropped prefix, gated at dispatch (mostly CPU):
  `tests/unittest/_torch/attention/test_trtllm_paged_context_fmha_gate.py`.
- End to end on one GPU:
  `tests/unittest/_torch/pyexecutor/test_overlap_scheduler.py::test_overlap_scheduler_block_reuse_cache_hit`.
- With disaggregation: `accuracy/test_disaggregated_serving.py::TestGPTOSS::test_auto_dtype`
  (block-reuse parameter), `::TestQwen3_5_4B::test_mismatched_block_reuse`.
- By hand: quickstart with and without `--disable_kv_cache_reuse`, one prompt
  sent twice. While debugging a V2 pool, `TRTLLM_KV_FRESH_PAGE_FILL=nan` and
  `TRTLLM_KV_GUARD_PAGE=nan` (diagnostic only) make a read of a page the
  request does not own visible instead of returning another request's KV.

**Flow proof.** A boot and an acceptance load run the reuse probe
(`checks/harness/reuse_probe.py`: one prompt sent twice on the same cache,
pinned to attention-DP rank 0). With reuse on, the gate fails when the second
response reports zero `cached_tokens`; boot, accuracy and acceptance fail when
the engine logs that it turned reuse off. A probe that cannot run fails
nothing. A `reuse` variant with `gates: [accuracy]` runs no probe: report it
as configured, not verified.

**Blind spots.** The probe shows that blocks matched, not that the forward
over the cached prefix is right; that rests on the benchmark, whose few-shot
prompts share a long prefix. Recurrent-snapshot restoration and attention-DP
ranks past 0 are not probed. Side state outside the pools stays invisible
until a second request reuses a prefix and diverges.

## 2. Chunked prefill

**What.** With `enable_chunked_prefill`, a context longer than the step's
token budget (`max_num_tokens`) is computed over several iterations; every
chunk after the first attends to KV that earlier chunks wrote.

**Engine owns.** The split and its policy (`context_chunking_policy` in
`SchedulerConfig`, `ContextChunkingPolicy`; `FORCE_CHUNK` where block
boundaries must be chunk boundaries), each request's chunk position
(`context_current_position`, `is_last_context_chunk`), and producing a token
only after the last chunk.

**Target must honour.** Every chunk is a context call over a cached prefix --
the section 1 contract (cached length from metadata, paged-context attention).
State that must carry across chunks (recurrent state, running scales) goes
back to manager storage between calls, never held in Python. Multimodal
embeddings are sliced to the chunk (section 6). A forward whose cached-context
path differs from its fresh-context path needs both certified; the catalog's
MLA entries `mla_rope_append_paged_kv_assign_q` and
`load_paged_kv_cache_for_mla` are that second path for MLA.

**Where.** `_torch/pyexecutor/scheduler/`,
`_torch/pyexecutor/engine/runners/decoder/` (its input preparation),
`_torch/attention/backends/`; `enable_chunked_prefill`, `max_num_tokens`,
`ContextChunkingPolicy`, `context_chunking_policy`, `FORCE_CHUNK`,
`context_current_position`, `is_last_context_chunk`, `num_cached_tokens_per_seq`.

**Tests to imitate.**
- `tests/unittest/_torch/pyexecutor/test_overlap_scheduler.py` (chunked prefill
  on with a 128-token budget, so one long request splits).
- `tests/unittest/_torch/multimodal/test_multimodal_runtime.py::TestFindInputMmEmbed::test_chunked_prefill_scenario`.
- `accuracy/modeling_v2/deepseek_v3/r1_0528_nvfp4__sm_10x__default/test_accuracy.py::TestModelingV2DeepseekR10528Nvfp4Sm10xDefault::test_gsm8k_chunked_prefill`:
  the pattern -- a `max_num_tokens` below the few-shot prompt length, so every
  benchmark prompt splits.
- `accuracy/test_disaggregated_serving.py::TestQwen3_8B::test_chunked_prefill`.
- By hand: quickstart `--enable_chunked_prefill --max_num_tokens <n>` with a
  prompt longer than 2n tokens, against a run whose `--max_num_tokens` exceeds
  the prompt.

**Flow proof.** One `chunked` variant (`gates: [accuracy]`) whose
`max_num_tokens` is below the benchmark's prompts (`spec/INTAKE.md`), so the
engine must split every prompt, and the result is scored. A variant whose
gates include `boot` also sends, in that boot, a prompt of about twice
`max_num_tokens`, and fails unless some request computed more than
`max_num_tokens` tokens past what the cache resumed (`chunk_shortfall`,
`checks/boot.py`).

**Blind spots.** The boot's evidence is inferred from lengths, not a chunk
count; a wrong later-chunk path shows only in the variant's score. A budget
above every benchmark prompt leaves chunking configured and never triggered.

## 3. Disaggregated serving

**What.** Prefill and decode run on separate engines. The context engine
computes the prompt and the first token; the cache transceiver moves the
request's state; the generation engine continues decoding without a context
forward (`docs/source/features/disagg-serving.md`,
`docs/source/developer-guide/kv-transfer.md`).

**What crosses the hop.** The pages of every pool role the context-side manager
declares, located by a page table (`_torch/disaggregation/resource/page.py`,
`resource/kv_extractor.py`) and moved by per-role mappers
(`native/mixers/attention/`, `native/mixers/ssm/` for recurrent conv and SSM
state); plus auxiliary per-request data: first generated tokens, draft tokens,
prompt and cached token counts (`native/auxiliary.py`). The Python transceiver
moves one KV slice per request after prefill completes. Nothing else travels.

**Engine owns.** The `KvCacheTransceiver` interface and
`create_kv_cache_transceiver` (`_torch/disaggregation/kv_cache_transceiver.py`;
`BindKvCacheTransceiver` wraps the C++ runtime), the Python runtime
`KvCacheTransceiverV2` (`transceiver.py`), rank mapping when context and
generation use different TP/PP/DP (`native/peer.py`), and
`CacheTransceiverConfig` (`backend`, `transceiver_runtime`, ... in
`tensorrt_llm/llmapi/llm_args.py`). V2 hybrid-recurrent managers and FP4 MLA
require `transceiver_runtime: PYTHON` with `backend: NIXL`; both sides must
run the same build and the same transceiver runtime.

**Target must honour.** All per-sequence state must sit in pools the manager
declares to the transceiver. The manager states transfer semantics through
`get_disagg_role_mapper_kinds` (default: K/V `INDEXED` in the head-major layout
the TRTLLM attention kernels write; `INDEX_KEY` `REPLICATED`),
`get_disagg_role_layouts` (resharding geometry) and `get_disagg_ignored_roles`;
`MapperKind` is in `resource/page.py`. A target that writes K/V in another
layout, or keeps state no role names, decodes on the generation side from wrong
or initial state -- fluently. The generation engine's first forward is a
generation step whose cached length is the prompt length
(`tests/unittest/_torch/pyexecutor/test_gen_only_cached_tokens.py`); it must need
nothing the context forward computed without storing.

**Where.** `_torch/disaggregation/` (`base/`, `native/`, `nixl/`, `resource/`,
`orchestration/`), `_torch/pyexecutor/kv_cache/` (the role hooks),
`tensorrt_llm/commands/serve.py` (the `disaggregated` subcommand);
`KvCacheTransceiverV2`, `create_kv_cache_transceiver`, `CacheTransceiverConfig`,
`transceiver_runtime`, `get_disagg_role_mapper_kinds`, `get_disagg_role_layouts`,
`MapperKind`, `AuxBufferMeta`, `CtxTransferStatus`.

**Tests to imitate.**
- Transfer correctness without serving: `tests/unittest/disaggregated/test_cache_transceiver_single_process.py`
  (V2 manager and Python transceiver in one process, across TP/PP/DP, MLA,
  sliding window), `test_kv_transfer.py`, `test_mamba_transfer.py` (recurrent
  state). A new layout or role usually means a mapper plus a page-table change,
  tested here.
- The transceiver contract with a fake, no GPU:
  `tests/unittest/_torch/disaggregation/test_kv_cache_transceiver_contract.py`.
- Working ctx/gen configurations: `accuracy/test_disaggregated_serving.py`
  (`launch_disaggregated_llm` is its one-node launcher):
  `TestQwen3_8B::test_chunked_prefill`, `TestGPTOSS::test_auto_dtype`,
  `TestQwen3NextInstruct::test_auto_dtype` (hybrid recurrent),
  `TestDeepSeekV3Lite::test_nvfp4_kv_cache_nixl`; worker YAMLs in
  `disaggregated/test_configs/`.

**Do not build** a transport, role service, request protocol or launcher: the
engine's transceiver moves the state, and its disaggregated serving
(`trtllm-serve` workers behind `trtllm-serve disaggregated`;
`launch_disaggregated_llm` in `accuracy/test_disaggregated_serving.py`,
two nodes in `accuracy/test_disaggregated_multinode.py`) runs it. A target's
disaggregation work is the state the transceiver moves. To start one by hand
while debugging, `examples/disaggregated/README.md`.

**Flow proof.** The `disagg` unit owes only its `disagg` check
(`checks/disagg.py`; `gate disagg --unit disagg --member <m>`), and the run's
`task.yaml` says which of two it is. Read `accuracy.disagg` before writing
anything.

*It states `config`, `context` and `generation`:* the three files the tree's
own disaggregated examples use, and the flow serves that deployment itself. You
write no test, launcher or server. Each side is one `trtllm-serve` per instance
with `--server_role` and its side's file as `--config`, as it is but for the
gate's `modeling_v2: require` (`gates/disagg/disagg.<unit>.<member>/options.<ctx|gen>.yaml`
is what it ran; nothing of the unit's options is in it), behind
`trtllm-serve disaggregated -c` with the stated `config`. The gate passes when
every server and the proxy came up and the unit's benchmark through the proxy is at or
above the unit's floor. Its verdict names the server that did not come up and
its log, in the same directory. `gate disagg` submits the job, across nodes too.

*It states `test`:* the proof is a test the run
writes beside the tree's own in `accuracy/test_disaggregated_serving.py`
(`launch_disaggregated_llm` starts both workers and `trtllm-serve
disaggregated`), named by `accuracy.disagg.test`. The gate runs it from its
pytest root as CI does, once, at the member the request validates fully. By
default that is one pytest on one node, with two engines' devices at the
member's width, at most a node's (`dep4` on a 4-GPU node: write the test for two
engines of two; `accuracy.disagg.devices` overrides it, per node). When two
engines do not fit a node, `accuracy.disagg.nodes: N` runs one pytest on each of
N nodes of one gates allocation (`MV2FLOW_JOB_NODE_RANK/HOSTS/NODES`), one
engine's devices per node by default, so the gates session must hold N nodes.
The gate passes only when every test the node id selects passed on every node:
a skip (too few devices, no `LLM_MODELS_ROOT`) fails it.

**Blind spots.** Boot and accuracy never cross the hop; only the `disagg` check does, and
it judges a benchmark score, not token identity with an aggregated engine. A
hop that loses part of the state can still score within the floor on short
answers. The transceiver backend and runtime the workers actually ran are not
read back.

## 4. KV-cache dtype (fp8, nvfp4) and scales

**What.** `kv_cache_config.dtype`: `auto` keeps the checkpoint's
`kv_cache_quant_algo`; `fp8`, `fp8_ds_mla` or `nvfp4` override it; prototype
`mixed_nvfp4_<dtype>` needs the V2 manager. Recurrent state has its own
`mamba_ssm_cache_dtype`.

**Engine owns.** Resolution into `quant_config.kv_cache_quant_algo`
(`validate_and_set_kv_cache_quant` in `_torch/pyexecutor/model_loader.py`, which
logs when it overrides the checkpoint), pool element types and the NVFP4
block-scale roles (`KEY_BLOCK_SCALE`, `VALUE_BLOCK_SCALE`), and the FP4-MLA
manager choice in `get_kv_cache_manager_cls` (V2 and the TRTLLM backend only).

**Target must honour.**
- Read the resolved algorithm from its `ModelConfig`; `route` raises `NoTarget`
  for an algorithm the forward does not implement, so `auto` falls back and
  `require` says why.
- Hand KV scales to the attention op whenever the cache is quantized. On the
  stock path `_torch/attention/attention.py` passes per-layer
  `kv_scale_orig_quant` / `kv_scale_quant_orig` only for an FP4 KV cache, and the
  TRTLLM backend falls back to a unit `kv_cache_scaling_factor` otherwise,
  because its kernel reads a scale for any quantized cache. Checkpoint
  `k_scale` / `v_scale` reach the stock path through `_load_kv_cache_scales`
  (`_torch/modules/linear.py`; switch `TRTLLM_LOAD_KV_SCALES`).
- Every op that writes the cache (append and RoPE-append ops) writes the pool's
  format.

**Where.** `tensorrt_llm/llmapi/llm_args.py` (`KvCacheConfig`),
`_torch/pyexecutor/model_loader.py`, `tensorrt_llm/quantization/mode.py`,
`_torch/attention/attention.py`, `_torch/attention/backends/` (`trtllm.py`,
`fp4_mla/`), `_torch/pyexecutor/kv_cache/` (mixed-precision staging where the
tree has it); `kv_cache_quant_algo`, `has_fp8_kv_cache`, `has_fp4_kv_cache`,
`kv_scale_orig_quant`, `kv_scale_quant_orig`, `kv_cache_scaling_factor`,
`_load_kv_cache_scales`, `Fp4MlaKVCacheManagerV2`.

**Tests to imitate.**
- The FP8 cells in `tests/unittest/_torch/modeling_v2/attention/test_modeling_v2_thop_attention.py`.
- Backends: `tests/unittest/_torch/attention/` (`test_fp4_mla*.py`,
  `test_mixed_precision_backends.py`); pools:
  `tests/unittest/_torch/pyexecutor/kv_cache/` (its mixed-precision cases).
- Accuracy: the rows tagged `kv_cache_quant_algo` in
  `accuracy/references/gsm8k.yaml` and the tests in
  `accuracy/test_llm_api_pytorch.py` that set a KV-cache dtype; across the hop,
  `accuracy/test_disaggregated_serving.py::TestDeepSeekV3Lite::test_nvfp4_kv_cache_nixl`.
- By hand: quickstart `--kv_cache_dtype fp8` (also `trtllm-serve
  --kv_cache_dtype`) against `auto` in the same session: close, not identical,
  for a lossy format.

**Flow proof.** One variant per dtype: `llm_api_options` setting
`kv_cache_config.dtype`, `gates: [accuracy]` (derive's `kv=bf16/fp8/nvfp4`
writes them). Naming no anchor row of its own, a variant is judged against its
unit's own score at the same member on the same target (`anchor: relative`, a
floor 2 SE of the difference under it); with
`accuracy.discriminators.kv_cache_quant_algo` it is judged against that dtype's
own row instead. Feature `kv_quant` (native storage instead of bf16
containers) is a unit like the others: its checks run with the native storage on.

**Blind spots.** No gate reads the dtype the engine resolved: an NVFP4 request
that fell back to FP8 passes unless the score shows it. Report the dtype as
verified only when the loader's KV-quantization log line says so. A missing or
unit scale shows only as an accuracy shift against `auto`, which a relative
floor catches when it is larger than the two runs' noise. Whether `kv_quant`
stores fewer bytes per token is in the engine's pool-allocation log line
(`Allocated <n> GiB for max tokens in paged KV cache`), which no gate reads.

## 5. CUDA graphs and the overlap scheduler

**What.** The model engine captures the forward per batch size (and optional
prefill token counts) and replays it. The overlap scheduler launches step
n+1's forward before step n's host-side processing finishes
(`docs/source/features/overlap-scheduler.md`,
`docs/source/features/torch_compile_and_piecewise_cuda_graph.md`).

**Engine owns.** The plan (`plan_cuda_graphs` / `CudaGraphPlan` in
`_torch/pyexecutor/engine/cuda_graph.py`: batch sizes filtered by the runtime
limits, padding, prefill capture sizes), capture and replay (`CUDAGraphRunner`
in `_torch/pyexecutor/cuda_graph_runner.py`, `DecoderCudaGraphs` in
`engine/runners/decoder/graphs.py`), breakable graphs
(`_torch/pyexecutor/breakable_cuda_graph/`), piecewise compilation
(`_torch/compilation/`), and the overlap loop (`_executor_loop_overlap` in
`py_executor.py`).

**Target must honour.**
- No host sync, and no Python branch on tensor values or per-step sizes, inside
  the captured forward. Every per-step value reaches the forward through
  metadata tensors the engine refreshes in place: replay copies only the static
  inputs, and rebinding a tensor attribute of `attn_metadata` / `spec_metadata`
  leaves the graph reading stale memory (`TLLM_CUDA_GRAPH_STRICT_BUFFERS=1`
  checks this). A Python scalar computed per step is frozen at its capture-time
  value.
- Under breakable graphs (catalog entry `graph/eager_on_graph.py`) the eager
  callable's capture-time arguments are frozen as well; it must read live
  state itself.
- Under overlap, nothing may depend on host-side per-request state that step
  n's processing has not updated yet.

**Where.** `CUDAGraphRunner`, `CudaGraphPlan`, `CudaGraphConfig`,
`cuda_graph_config`, `enable_padding`, `prefill_capture_num_tokens`,
`eager_on_graph`, `TLLM_CUDA_GRAPH_STRICT_BUFFERS`, `disable_overlap_scheduler`,
`_executor_loop_overlap`.

**Tests to imitate.**
- `tests/unittest/_torch/pyexecutor/test_cuda_graph_capture_replay.py` (static
  tensors poisoned between replays; the strict-buffer check; the `use_mrope`
  cases for position state).
- `tests/unittest/_torch/pyexecutor/test_breakable_cuda_graph.py`,
  the graph tests under `tests/unittest/_torch/modeling_v2/` (eager-on-graph).
- `tests/unittest/_torch/pyexecutor/test_overlap_scheduler.py::test_overlap_scheduler_consistency`
  (identical outputs with overlap on and off).
- `tests/unittest/_torch/compilation/`.
- By hand: quickstart with and without `--use_cuda_graph`
  (`--cuda_graph_batch_sizes`, `--cuda_graph_padding_enabled`), and with
  `--disable_overlap_scheduler`, at a batch size that is captured exactly and
  at one that is padded.

**Flow proof.** `runtime.cuda_graph: true` in every gate: the benchmark is the
proof. The overlap scheduler (on in core) has no check of its own.

**Blind spots.** Whether the engine captured the declared batches, and whether
the workload's batches replayed a graph or ran eager, is not observed. A host
value frozen into a graph shows only at batch sizes the benchmark replays. An
extra host sync under overlap costs throughput only, which no gate sees.

## 6. Multimodal inputs

**What.** An input processor prepares media, an encoder embeds it, and the
embeddings replace placeholder tokens in the decoder's input
(`docs/source/features/multi-modality.md`; onboarding hooks in
`docs/source/developer-guide/multimodal.md`).

**Engine owns.** Input processors and their registry
(`tensorrt_llm/inputs/registry.py`: `InputProcessor`, `register_input_processor`),
`MultimodalParams` (`tensorrt_llm/inputs/multimodal.py`), encoder scheduling
(`_torch/pyexecutor/engine/multimodal.py`, `engine/runners/mm_encoder.py`),
the per-step `MultimodalRuntimeData`, and media hashing, so that KV reuse never
matches the same placeholder tokens across different images (`kvcache.md`,
multimodal UUIDs).

**Target must honour.** Fuse only the embeddings of the tokens computed in this
step: under chunking or reuse part of an image is already cached
(`find_input_mm_embeds`, `fuse_input_embeds`, `get_multimodal_embeddings` in
`_torch/models/modeling_multimodal_utils.py`). A `scope: text_only` target
builds the language model alone and must load and route with the tower absent.

**Where.** `_torch/models/modeling_multimodal_mixin.py`,
`_torch/models/modeling_multimodal_utils.py`, `tensorrt_llm/inputs/`,
`_torch/pyexecutor/engine/multimodal.py`; `MultimodalParams`,
`MultimodalRuntimeData`, `past_seen_token_num`, `find_input_mm_embeds`,
`fuse_input_embeds`, `register_input_processor`, `multi_modal_uuids`.

**Tests to imitate.**
- `tests/unittest/_torch/multimodal/` (`test_multimodal_runtime.py`,
  `test_fuse_input_embeds.py`, `test_find_num_image_tokens.py`).
- `tests/unittest/_torch/pyexecutor/test_multimodal_scheduler.py`,
  `tests/unittest/_torch/pyexecutor/kv_cache/test_kv_cache_v2_multimodal_runs.py`.
- Accuracy: `accuracy/test_llm_api_pytorch_multimodal.py`; encoder as its own
  server: `accuracy/test_epd_disagg_multimodal.py`; anchors in
  `accuracy/references/mmmu.yaml`.
- By hand, from `examples/llm-api/` (the script imports its sibling
  `quickstart_advanced.py`): `quickstart_multimodal.py --model_dir <ckpt>
  --modality image --media <image>`; `tests/integration/test_input_files/merlion.png`
  (git-lfs) is a known image.

**Flow proof.** No engine field. A `multimodal` unit's boot sends an image
(default the merlion photograph) and the serve gate sends an image chat: both
show that a request with an image is answered. That the image was seen is
shown only by an image benchmark (`mmmu`, `mmstar`, `chartqa`, ...).

**Blind spots.** A target that ignores the image passes boot and serve, and a
text benchmark says nothing of the vision path. Image tokens split across a
chunk, and image-prefix reuse, are exercised only by a prompt built for them.
For `text_only`, absent by design and absent by accident look alike.

## 7. Speculative decoding

**What.** A drafter proposes tokens, the target verifies them in one forward,
and rejection sampling accepts a prefix
(`docs/source/features/speculative-decoding.md`: MTP, Eagle3, draft/target,
NGram, PARD, DFlash, DSpark).

**Engine owns.** Building the spec metadata, worker, resource manager, drafter
and decoder from the config (`_torch/speculative/utils.py`:
`get_spec_metadata`, `get_spec_worker`, `get_spec_resource_manager`,
`get_spec_drafter`), verification and acceptance, draft KV, and the iteration
statistics (`numAcceptedTokens`).

**Target must honour.**
- For a one-model speculator the draft layers are part of the forward the
  target writes (the ModelingV2 DeepSeek target runs its MTP layer as one more
  decoder layer). Eagle3, DFlash and DSpark read hidden states the target
  captures at the layers `is_layer_capture` names (`maybe_capture_hidden_states`
  on `SpecMetadata` and its subclasses).
- A generation request carries the accepted token plus drafts each step, so
  attention runs the paged-context path (section 1).
- State written per draft token is committed for the accepted count
  (`AuxiliarySpeculativeStateHandler` in `_torch/speculative/interface.py`).
- `route` admits a `speculative_config` only where acceptance was measured.

**Where.** `SpecMetadata`, `SpeculativeDecodingMode`, `max_draft_len`,
`is_layer_capture`, `maybe_capture_hidden_states`,
`AuxiliarySpeculativeStateHandler`, `numAcceptedTokens`.

**Tests to imitate.**
- `tests/unittest/_torch/speculative/` (`test_eagle3.py`, `hw_agnostic/`).
- `accuracy/modeling_v2/deepseek_v3/r1_0528_nvfp4__sm_10x__default/test_accuracy.py::TestModelingV2DeepseekR10528Nvfp4Sm10xDefault::test_mtp3_acceptance`:
  one workload under `modeling_v2` `require` and `off`, iteration statistics on,
  `compute_acceptance_length` / `assert_acceptance_length`
  (`accuracy/accuracy_core.py`) against `accuracy/references/acceptance_length.yaml`,
  populated from the stock leg with `TRTLLM_POPULATE_ACCEPTANCE_LENGTH=1`.
  `::test_gsm8k_identity_vs_mtp3` pairs accuracy with and without drafts in
  one session.
- By hand: quickstart `--spec_decode_algo <algo> --spec_decode_max_draft_len <n>`
  (`--draft_model_dir` for a separate drafter); greedy output with drafting off
  should agree up to near-ties, and acceptance is in the iteration statistics.

**Flow proof.** Features `mtp`, `eagle3`, `dflash` and `dspark` owe the
acceptance gate (`checks/acceptance.py`; `gate acceptance --unit <feature>
--member <m>`): accepted tokens per iteration, iteration-weighted over the
workload (32 prompts and 32 draft iterations by default), in one load. It
fails at or below 1.0 and under the floor of the unit's `accuracy.acceptance`
entry (`min_absolute`, or `measured_reference`, with its `floor_source`). A
unit with no entry of its own takes that of the speculator it builds on, then
`default`; with none of them it is held to 1.0 alone.

**Blind spots.** Boot and accuracy: rejection sampling preserves the output
distribution, so a wrong draft path yields correct text, slower. The floor is
a number the run declares: without one, a drafter that accepts little passes.

## 8. Attention DP, TP and EP where they meet these features

- **Where.** `tensorrt_llm/mapping.py` (`Mapping`), `_torch/distributed/`,
  `_torch/pyexecutor/scheduler/adp_router.py` (`DefaultADPRouter`,
  `KVCacheAwareADPRouter`), `enable_attention_dp` and `AttentionDpConfig` in
  `llm_args.py`. Members are `tp<N>`, `tep<N>`, `dep<N>` (attention DP + EP).
- **Attention DP.** Each attention-DP rank schedules its own requests in its own
  KV cache. A rank with no work runs a dummy request (`is_attention_dp_dummy`)
  so collectives line up; the forward must accept a step whose only request is
  a dummy.
- **Reuse under attention DP** hits only when the repeat lands on the same rank:
  pin it with `SchedulingParams` (`attention_dp_rank`, `attention_dp_relax`
  off), as the flow's probe does, or use the KV-cache-aware router.
- **Disaggregation** may pair different TP/PP/DP on the two sides; `native/peer.py`
  matches ranks and the mixers reshard heads. At a `dep` member a benchmark
  spreads requests over the attention-DP ranks; one request at a time crosses
  only rank 0 to rank 0.
- **Side state under TP.** A role whose bytes are identical on every rank is
  declared `REPLICATED`; a role that splits by head declares its geometry
  (section 3).
- **Tests.** `tests/unittest/_torch/multi_gpu/`, `tests/unittest/_torch/modeling_v2/comm/`,
  `tests/unittest/_torch/pyexecutor/test_adp_router.py`,
  `tests/unittest/disaggregated/test_cache_transceiver_single_process.py`.
- **Flow proof.** Each member's benchmark, or its boot where no benchmark runs
  at it; no gate reads back the widths the engine ran.
- **Blind spots.** The reuse probe covers attention-DP rank 0 only. With
  `benchmark_members: <member>` the benchmark runs at that member alone; the
  other members rest on their boots (`task.example.yaml`).
