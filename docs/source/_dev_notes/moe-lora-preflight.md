# Implementation memo: Cutlass MoE shared-outer LoRA

Internal dev note for [TRTLLM-12507](https://jirasw.nvidia.com/browse/TRTLLM-12507). Not user-facing. Will be migrated into the PR description and discarded post-merge.

Companion to the plan at [`cutlass-moe-shared-outer-lora.plan.md`](../../../cutlass-moe-shared-outer-lora.plan.md). Phase 0 findings are below; per-phase progress logs follow.

## Progress snapshot

| Phase | Status | Notes |
|---|---|---|
| 0 — Pre-flight | done | Verified facts F1–F7 below. |
| 1 — C++ thop op extension | done | `moeOp.cpp` extended; see Phase 1 log. |
| 2 — Python eager wiring | done | `CutlassFusedMoE`, op schema, Qwen-MoE model. See Phase 2 log. |
| 3 — CUDA-graph decode path | partial (slot-indexed eager OK; graph capture blocked by kernel) | Slot-indexed pointer mode lands in `moeOp.cpp` + `CudaGraphLoraParams.token_to_slot_host` and `get_moe_slot_inputs`. Eager equivalence passes. CUDA-graph capture is rejected by the op until the Phase 6 kernel patch; see F7 and Phase 3 log. |
| 4 — Loader + validator | done | `validation.py`, `moe_layout.py`, `create_moe.py` wiring. See Phase 4 log. |
| 5 — Tests | done (MVP) | CPU validator + layout tests, GPU smoke for fused-moe op. CUDA-graph capture+replay test skipped pending Phase 6. See Phase 5 log. |
| 6a — Native shared-outer kernel flag (kernel + op + test fixture) | done | `LoraParams::*_shared_a/b` zeros the per-expert offset in `setupLoraWorkspace`; 6 op kwargs threaded; native helper in `moe_layout.py`; bit-identity GPU test. |
| 6a follow-up — Loader plumbing (`lora_layout.json`) | done (metadata-only) | New `lora_layout_sidecar.py` parser; `LoraManager.get_moe_shared_flags(uid)`; `model_engine` per-request assembler populates `lora_params["moe_shared_flags"]`. Cache row still load-time replicated -- skipping the stack and accepting native unreplicated on-disk shapes requires changing `LoraModule::localTotalSize` and is a separate follow-up PR. |
| 6b — GPU-side LoRA expansion (lift host-sync, enable graph capture) | follow-up PR | Replaces the `setupLoraWorkspace` / `LoraImpl::run` host-CPU branching with a device-side problem-builder kernel. Re-enables the skipped graph-capture parity test. |
| 7 — Docs | done | Routed-Expert MoE LoRA section added to [`docs/source/features/lora.md`](../features/lora.md). |


## Verified facts

### F1. Loader stride matches kernel offset arithmetic

The kernel offsets a flat per-adapter buffer by `weight_index * dim * lora_rank` ([`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`](../../../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu) lines 3694–3733), e.g. for `fc1` A:

```cpp
host_permuted_fc1_weight_ptrs[i * 2]
    = reinterpret_cast<ScaleBiasType const*>(lora_params.fc1_lora_weight_ptrs[source_index * 2])
    + weight_index * hidden_size * lora_rank;
```

So at expert `e` the kernel expects expert-slice `e` to start at `e * dim * rank` elements after the base pointer.

The Python loader produces exactly that layout: [`tensorrt_llm/lora_manager.py`](../../../tensorrt_llm/lora_manager.py) lines 1126–1184 stacks per-expert HF tensors with `torch.stack(t_in_list)` (shape `[E, rank, dim]`) and then `t_in.flatten()`. Element `[e, r, h]` is at offset `e * rank * dim + r * dim + h`, so expert `e`'s slice starts at `e * rank * dim`. Identity holds for all four sides:

| Module | A shape `[rank, in_dim]` | B shape `[out_dim, rank]` |
|---|---|---|
| `moe_h_to_4h` (fc1 main / w1 — gate) | `[r, hidden]`, stride `hidden * r` | `[inter, r]`, stride `inter * r` |
| `moe_gate` (fc1 gated / w3 — up) | `[r, hidden]`, stride `hidden * r` | `[inter, r]`, stride `inter * r` |
| `moe_4h_to_h` (fc2 / w2 — down) | `[r, inter]`, stride `inter * r` | `[hidden, r]`, stride `hidden * r` |

The C++ cache's `[adapterSize, inDim] = [r, dim * E]` view in [`cpp/tensorrt_llm/runtime/loraCache.cpp`](../../../cpp/tensorrt_llm/runtime/loraCache.cpp) line 663 only reinterprets the flat buffer; the kernel ignores the view and uses pointer arithmetic, so the view-vs-pack mismatch is benign.

**Implication for Phase 1**: no loader changes are needed for the per-expert layout. The thop op can pass the existing `weights_in_pointer` / `weights_out_pointer` from `peft_table[task_id]` directly into `LoraParams.fc1_lora_weight_ptrs[i*2 + 0/1]`.

### F2. Kernel rejections to mirror in Python validator

Three hard kernel checks at [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`](../../../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu):

| Combination | Line | Mechanism |
|---|---|---|
| `use_lora && act_fp4` | 3963 | `TLLM_CHECK_WITH_INFO(!use_lora || !act_fp4, "MOE does not support LoRA with FP4 model")` |
| `min_latency_mode && use_lora` | 4024 | `TLLM_CHECK(use_lora == false)` |
| `use_lora` in `loraFC1` with `act_fp4` | 3751 | redundant check inside `loraFC1` |

Phase 2 / Phase 4 implications:
- Python validator MUST reject MOE LoRA combined with NVFP4 base weights.
- Python validator MUST reject MOE LoRA combined with `min_latency_mode=True` (covered also by `runMoeMinLantency` thop entry — we'll reject there too).
- All other quantizations (FP8, INT4, INT8) are NOT explicitly rejected by the kernel for LoRA, but the MVP scope is bf16/fp16 only — validator rejects all quant + MOE LoRA. This is stricter than the kernel allows and gives us a safer initial surface.

### F3. Alltoall + LoRA must be rejected in Python (not enforced by kernel)

The kernel passes `enable_alltoall` alongside `use_lora` to `finalizeMoeRoutingKernelLauncher` without rejecting the combination (`moe_kernels.cu` lines 4060, 4184). However the kernel reads `lora_params.fc1_lora_ranks[source_index]` where `source_index = host_permuted_rows[i] % num_rows` is the **local-rank original token index**. When alltoall reshuffles tokens across ranks, the per-token adapter pointer array stops covering the foreign tokens that landed on the local rank.

**Implication**: Phase 1 (thop) and Phase 4 (validator) reject `enable_alltoall=True && lora_active`. WideEP, which always uses alltoall, is out of scope for MVP regardless.

### F4. `LoraImpl` lifecycle in thop

[`cpp/tensorrt_llm/kernels/lora/lora.cpp`](../../../cpp/tensorrt_llm/kernels/lora/lora.cpp) lines 67–104:

- Constructor needs `in_hidden_size`, `out_hidden_sizes` (vector), `transA`, `transB`, `num_lora_modules`, `nvinfer1::DataType`, `max_low_rank`, and a `std::shared_ptr<CublasGemmWrapper>`.
- `setGemmConfig()` is called once with the dtype before first `run()`; selects cuBLAS gemm config (FP16/BF16/FP32).
- `getWorkspaceSize(numTokens, numReqs, type)` returns grouped-gemm + low-rank intermediate + params workspace.
- `setBestTactic(config)` is optional (heuristic profiling); the MVP can skip and let cuBLAS auto-select.

The TRT plugin constructs LoraImpl in `initialize()` ([`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp`](../../../cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp) lines 347–362):

```cpp
auto cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
mLoraImpl1 = std::make_shared<LoraImpl>(
    mExpertHiddenSize, {mExpertInterSize}, false, true, 1, mLoraType, mMaxLowRank, cublasWrapper);
mLoraImpl2 = std::make_shared<LoraImpl>(
    mExpertInterSize, {mExpertHiddenSize}, false, true, 1, mLoraType, mMaxLowRank, cublasWrapper);
```

**Implication for Phase 1**: `FusedMoeRunner` does NOT know `hidden_size`, `inter_size`, `max_low_rank` at construction (`__init__` only takes dtypes). These are per-call. We will:

1. Add a `CublasMMWrapper` member, lazily constructed on first LoRA call (needs `cublasHandle`, `cublasLtHandle` from `at::cuda::getCurrentCUDABlasHandle` / `getCurrentCUDABlasLtHandle`).
2. Cache `LoraImpl` instances in a `std::map<std::tuple<int64_t, int64_t, c10::ScalarType, int>, std::pair<LoraImplPtr, LoraImplPtr>>` keyed by `(hidden, inter, dtype, max_rank)`.
3. Add a `cudaEvent_t mLoraMemcpyEvent` member, created on first LoRA use (the kernel synchronizes on this event at `moe_kernels.cu` line 3689).

### F5. Workspace sizing

The kernel's `getWorkspaceSize(use_lora=true)` ([`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`](../../../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu) line 3094) already includes the LoRA-side activation buffers (`lora_input_`, `fc1_lora_result_`, `fc2_lora_result_`). However the cuBLAS grouped-gemm scratch passed via `LoraParams.workspace` is **separate**.

The plugin allocates it alongside the MoE workspace ([`cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp`](../../../cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp) line 566):

```cpp
size_t lora_workspace_size = std::max(
    mLoraImpl1->getWorkspaceSize(num_tokens * mExpertsPerToken, num_reqs_lora, mLoraType),
    mLoraImpl2->getWorkspaceSize(num_tokens * mExpertsPerToken, num_reqs_lora, mLoraType));
```

**Implication for Phase 1**: extend `getWorkspaceInfo` to allocate one extra `lora_workspace` slice of that size when LoRA is active, then point `LoraParams.workspace` at it.

### F6. `[i*2, i*2+1]` per-token expansion convention is shared

Both the TRT plugin path (`mixtureOfExpertsPlugin.cpp` lines 726–823) and the standalone `lora_grouped_gemm` thop op ([`cpp/tensorrt_llm/thop/loraOp.cpp`](../../../cpp/tensorrt_llm/thop/loraOp.cpp) lines 97–127) implement the same per-request → per-token expansion logic. We will lift the expansion helper into `moeOp.cpp` rather than depending on either of those (the plugin path is TRT-specific; `loraOp.cpp` is a different op surface). Mirroring the logic locally keeps the diff focused.

### F7. Kernel LoRA path is not CUDA-graph capturable (post-Phase-3 finding)

`setupLoraWorkspace` in [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`](../../../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu) (lines ~3680–3760) builds the per-token adapter pointer arrays on the **host**:

1. `cudaMemcpyAsync(host_permuted_rows, ..., D2H, stream)` — copies the post-permutation token-to-source map to host.
2. `cudaEventRecord(memcpy_event, stream)`.
3. `cudaEventSynchronize(memcpy_event)` — **host-blocking wait** on the event.
4. CPU loop over `host_permuted_rows[i]`, writing into `host_permuted_fc1_weight_ptrs[i*2 + 0/1]`, etc.
5. `cudaMemcpyAsync(device_permuted_*_ptrs, host_*_ptrs, H2D, stream)`.

The `cudaEventSynchronize` step and the host-CPU pointer fan-out cannot be recorded into a CUDA graph: during capture, the event is recorded into the graph (not signalled in real time), so the host-side wait either errors out or deadlocks, and the subsequent CPU branches read uninitialized memory. Empirically this manifests as a hard segfault during `cudaStreamEndCapture`.

Slot-indexed pointer mode (Phase 3) addresses the **input-layout** half of graph-safety (stable pinned-host source buffers, stable device pointer-table memory, no per-step allocations on the Python side), but it does not move the host wait off the critical path. The full fix belongs in the kernel.

**Implication for the MVP**:

- The op rejects `lora_active && isCapturing(stream)` with a clear `TORCH_CHECK`.
- Slot-indexed eager execution is supported and tested (it has independent value: stable pointer tables and pinned-host buffers reduce per-step Python work).
- The CUDA-graph capture-and-replay parity test is `pytest.mark.skip`'d with a reference to this finding.

**Phase 6 follow-up**: replace the host-side expansion with a small GPU kernel that takes `permuted_rows` and the per-adapter pointer tables and writes the per-token pointer arrays directly into device memory. Drops the `cudaEventSynchronize`, makes the LoRA path fully stream-bound, and unblocks graph capture.

## Phase 1 design decisions

- Schema: extend `FusedMoeRunner::runMoe` with optional CPU tensors `fc1_lora_ranks`, `fc1_lora_weight_ptrs`, `fc2_lora_ranks`, `fc2_lora_weight_ptrs`, `gated_lora_ranks`, `gated_lora_weight_ptrs`, `host_request_types`, `host_context_lengths`, plus `lora_max_low_rank: int`. All defaulted to none; non-LoRA callers see no behavior change.
- `LoraImpl` cache is keyed on `(hidden, inter, dtype, max_rank)`. Cache lives on the `FusedMoeRunner` instance; cleared by `clearWorkspaces`.
- Reject in `runMoe`: `lora_active && (act_fp4 || min_latency_mode || enable_alltoall)`.
- Reject in `runMoeMinLantency`: `lora_active` outright.
- `host_request_types` schema: int32 `[num_seqs]`, values `0` (CONTEXT/prefill) or `1` (GENERATION), matching `tensorrt_llm::RequestType`.

## Phase 2 / Phase 3 / Phase 4 design notes (deferred details)

- **Phase 2 model wiring**: only `modeling_qwen_moe.py` currently has shared-expert LoRA, and it does not pass `lora_params` to routed experts. Modify there first; other MoE models inherit the same plumbing via `forward_impl(**kwargs)`. The compile/torch_export branch in `interface.py` line 864 (`moe_custom_op`) drops kwargs and is incompatible with MVP LoRA — emit a clear error.
- **Phase 3 CUDA-graph**: add an inverse `token_to_slot: [max_num_tokens]` int32 tensor to `CudaGraphLoraParams`; populate during `update_sorted_indices`. Plumb into a `slot_lora_*` mode in the thop op.
- **Phase 4 validator**: new file `tensorrt_llm/_torch/peft/lora/validation.py` exporting `validate_lora_target_modules_supported(model_config, lora_config)`. Call from `LoraManager.__init__` after parsing target modules. Reject MOE_* targets unless `moe_backend == 'CUTLASS'` and `quant_config` is None or `kind == FP_AUTO`.

## Risks identified during pre-flight

- **None blocking**. All four verifications passed. The biggest implementation risk is the LoraImpl lifecycle in thop — mitigated by lazy construction + caching.
- **Soft risk**: the `setupLoraWorkspace` function takes a non-const `LoraParams&` and resizes its host vectors. The struct is sized once per call, so concurrent calls into the same `FusedMoeRunner` would race. The runner already holds `std::lock_guard<std::mutex> lock(mMutex)` at the top of `runMoe` (line 279), so this is already serialized.

## Out of scope for MVP

- DoRA + MoE — already rejected at [`tensorrt_llm/lora_manager.py`](../../../tensorrt_llm/lora_manager.py) line 1131. Keep that rejection.
- INT4/INT8/FP8 weight-quantized base + MoE LoRA — kernel allows FP8 paths but MVP scope is bf16/fp16 only.
- WideEP, CuteDsl, DeepGemm, TRTLLMGen, DenseGEMM, Triton, Vanilla, MegaMoE — rejected by validator in Phase 4.
- Native shared-outer (kernel patch) — Phase 6 / follow-up PR.
- `register_to_config + is_torch_compiling` MoE LoRA — emit clear error.

---

## Phase 1 — C++ thop op extension (done)

[`cpp/tensorrt_llm/thop/moeOp.cpp`](../../../cpp/tensorrt_llm/thop/moeOp.cpp) gained the routed-expert LoRA surface:

- Schema additions on `FusedMoeRunner::runMoe` (mirrored in [`tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`](../../../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py) `fused_moe`):
  `fc1_lora_ranks`, `fc1_lora_weight_ptrs`, `fc2_lora_ranks`, `fc2_lora_weight_ptrs`,
  `gated_lora_ranks`, `gated_lora_weight_ptrs`, `host_request_types`, `host_context_lengths`,
  `lora_max_low_rank`. All optional; absence => `lora_active = false`.
- Helper `buildMoeLoraParams(...)` (line ~1138) validates inputs, performs per-request → per-token expansion via `expandPerRequestLoraTo` (line ~1070), and emits a populated `kernels::LoraParams`.
- Lazy infra:
  - `CublasMMWrapper` constructed via `ensureLoraInfra()` (one per runner).
  - `LoraImpl` instances cached in `mLoraImplCache` keyed by `(hidden, inter, dtype, max_rank)` via `getOrCreateLoraImpls`.
  - `mLoraMemcpyEvent` created on first LoRA use; matches the kernel synchronize at `moe_kernels.cu` line 3689.
- Workspace: `computeLoraWorkspaceSize` adds a cuBLAS scratch slice; allocated by an extended `getWorkspaceInfo(use_lora, lora_workspace_size)` and pointed at by `lora_params.workspace`.
- Rejections (Phase-1 implementation matches F2/F3 findings):
  - `lora_active && min_latency_mode` → `TORCH_CHECK` in `runMoe`.
  - `lora_active && enable_alltoall` → `TORCH_CHECK` in `runMoe`.
  - `lora_active && act_fp4` → kernel still has the check; thop layer rejects non-bf16/fp16 activation dtype to keep error close to caller.
  - `runMoeMinLantency` rejects any LoRA outright.

## Phase 2 — Python eager wiring (done)

- [`tensorrt_llm/_torch/peft/lora/layer.py`](../../../tensorrt_llm/_torch/peft/lora/layer.py): added `MoeLoraLayer` sentinel that subclasses `LoraLayer` for discovery by `CudaGraphLoraManager`; its `forward` raises (the kernel handles compute).
- [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`](../../../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py):
  - `__init__`: registers a `MoeLoraLayer` marker when `model_config.lora_config` targets MoE modules (`_maybe_make_lora_marker`).
  - `_extract_moe_lora_tensors` maps the per-layer `lora_params` dict to the op's kwargs, with the canonical mapping `fc1 ↔ moe_h_to_4h`, `fc2 ↔ moe_4h_to_h`, `gated ↔ moe_gate`. Asserts fc1+fc2 presence.
  - `forward_impl` → `forward_chunk` → `run_moe` all accept `lora_params: Optional[Dict]` and forward it.
  - Runtime error if `lora_params` is non-None but the layer wasn't constructed with MoE-LoRA target modules.
- [`tensorrt_llm/_torch/models/modeling_qwen_moe.py`](../../../tensorrt_llm/_torch/models/modeling_qwen_moe.py): passes `lora_params=lora_params` into `self.experts(...)`.

## Phase 4 — Loader & validator (done)

- New module [`tensorrt_llm/_torch/peft/lora/validation.py`](../../../tensorrt_llm/_torch/peft/lora/validation.py):
  - `MOE_LORA_MODULE_NAMES = {"moe_h_to_4h", "moe_4h_to_h", "moe_gate"}`
  - `has_moe_lora_targets(lora_config)`
  - `check_moe_lora_supported(moe_backend_name, lora_config, quant_config, layer_idx=None)`
- New helper [`tensorrt_llm/_torch/peft/lora/moe_layout.py`](../../../tensorrt_llm/_torch/peft/lora/moe_layout.py):
  - `make_per_expert_lora(num_experts, rank, in_dim, out_dim, shared_side="A"|"B"|None, ...)` — produces per-expert-stacked `A:[E, rank, in]` and `B:[E, out, rank]`. Shared-outer is achieved by `expand` → `contiguous` (load-time replication).
  - `reference_moe_lora_delta(...)` — eager PyTorch reference for unit tests.
  - `DEFAULT_SHARED_SIDE`: up-projections (`moe_h_to_4h`, `moe_gate`) share A; down-projection (`moe_4h_to_h`) shares B.
- Validator wired into [`tensorrt_llm/_torch/modules/fused_moe/create_moe.py`](../../../tensorrt_llm/_torch/modules/fused_moe/create_moe.py) `resolve_moe_cls`, after the TRTLLMGen → Cutlass fallback. Raises early if MoE-LoRA target modules are present and either the resolved backend is not `CUTLASS` or quantization is active.

**Deferred** for the MVP: real `lora_layout.json` sidecar parsing and HF-format auto-detect-and-collapse. With load-time replication the user pre-replicates the shared tensor `E` times before saving the adapter, and the loader's existing per-expert path handles it. The `moe_layout` helper plus the validator are sufficient for synthetic adapters used in unit tests; sidecar plumbing can be added incrementally without disturbing the kernel surface.

## Phase 5 — Tests (MVP done)

- [`tests/unittest/_torch/lora/test_moe_lora_validator.py`](../../../tests/unittest/_torch/lora/test_moe_lora_validator.py) — CPU. Asserts accept/reject across all MoE backends and quant variants.
- [`tests/unittest/_torch/lora/test_moe_layout.py`](../../../tests/unittest/_torch/lora/test_moe_layout.py) — CPU. Adapter generator shape/replication/seed/reference-delta tests.
- [`tests/unittest/_torch/lora/test_moe_lora_op.py`](../../../tests/unittest/_torch/lora/test_moe_lora_op.py) — GPU smoke. Skipped unless `torch.cuda.is_available()` and `torch.ops.trtllm.fused_moe` is registered. Cases:
  - Per-expert LoRA changes the output (vs no-LoRA baseline).
  - Shared-outer LoRA (A shared on FC1, B shared on FC2) runs and changes the output.
  - `min_latency_mode=True` + LoRA raises.
  - Incomplete LoRA inputs (missing FC2 / `host_request_types`) raise.

**Deferred** for follow-up: full numerical-equivalence test against a hand-written SwiGLU MoE reference, multi-LoRA in flight, and the CUDA-graph capture-and-replay cases (these depend on Phase 3 landing).

---

## Phase 3 — CUDA-graph decode path (partial; graph capture deferred)

**Status**: slot-indexed input layout and host buffer stability are implemented, but **capture under `torch.cuda.graph` is rejected by the op** because the kernel's LoRA path does a host-side `cudaEventSynchronize` (see F7). Slot-indexed eager execution is supported and exercised in tests.

Implementation:

- [`tensorrt_llm/_torch/peft/lora/cuda_graph_lora_params.py`](../../../tensorrt_llm/_torch/peft/lora/cuda_graph_lora_params.py):
  - Added `token_to_slot_host: [max_num_tokens]` int32 pinned-host buffer; populated in `update_sorted_indices` (both `tokens_per_seq==1` and spec-decode paths).
  - Added `get_moe_slot_inputs(layer_idx, module_id) -> (slot_ranks_host, slot_weight_ptrs_host)`. The packed `[max_lora_size, 3]` pointer tensor is cached per `(layer_idx, module_id)` so its address survives across captures and replays (graph-capture safe).
- [`cpp/tensorrt_llm/thop/moeOp.cpp`](../../../cpp/tensorrt_llm/thop/moeOp.cpp):
  - New op kwargs (mirrored in `torch_custom_ops.py`): `fc1_slot_lora_ranks`, `fc1_slot_lora_weight_ptrs`, `fc2_slot_lora_ranks`, `fc2_slot_lora_weight_ptrs`, `gated_slot_lora_ranks`, `gated_slot_lora_weight_ptrs`, `token_to_slot` (all CPU pinned).
  - `materializeSlotIndexedLoraTo(slot_ranks, slot_weight_ptrs, token_to_slot, num_tokens, expand_ranks, expand_ptrs)` performs the per-token lookup in C++ and writes into the persistent `mLoraExpand*` host vectors.
  - `reserveLoraHostBuffers(max_num_tokens)` (bound as Python `reserve_lora_host_buffers`) pre-reserves the host buffer capacity. The slot-indexed `buildMoeLoraParams` branch also lazily bumps capacity when first called.
  - `buildMoeLoraParams` now accepts both per-request and slot-indexed inputs and dispatches accordingly. Supplying both is rejected with a clear error.
  - Top-level `runMoe` rejects the (per-request + slot-indexed) combination.
  - `runMoe` also rejects `lora_active && tensorrt_llm::common::isCapturing(stream)` with a clear `TORCH_CHECK`. See F7: the fused-MoE kernel's LoRA path performs a host-side `cudaEventSynchronize` after a D2H pointer-expansion copy, which is not capturable. Phase 6 lifts this restriction.
- [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`](../../../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py):
  - `_extract_moe_lora_tensors` now dispatches to a new `_extract_moe_lora_tensors_cuda_graph` when `lora_params["use_cuda_graph_mode"]` is true.
  - The CUDA-graph branch pulls `slot_ranks_host`, `h_b_ptrs`, `h_b_prime_ptrs` and `token_to_slot_host` out of `CudaGraphLoraParams`, packs them, and returns the slot-indexed kwargs accepted by the op.

Tests:

- CPU unit tests in [`tests/unittest/_torch/lora/test_lora.py`](../../../tests/unittest/_torch/lora/test_lora.py):
  - `token_to_slot_host` population for both `tokens_per_seq=1` and `tokens_per_seq=3` (spec decode).
  - `get_moe_slot_inputs` packing layout (A in column 0, B in column 1, 0 in column 2).
  - Pointer-cache stability across calls.
  - `None` return for unknown `(layer_idx, module_id)`.
- GPU end-to-end tests in [`tests/unittest/_torch/lora/test_moe_lora_op.py`](../../../tests/unittest/_torch/lora/test_moe_lora_op.py):
  - Slot-indexed mode bit-identical to per-request mode for a single LoRA covering all tokens.
  - Mixed-batch slot dispatch (half tokens on slot 0, half on slot 1) — each half bit-identical to its single-adapter reference.
  - `torch.cuda.CUDAGraph` capture + replay numerically equivalent to eager — **skipped pending Phase 6** (kernel-side host-sync removal). See F7.
  - Mutual-exclusion error when both per-request and slot-indexed inputs are supplied.

---

## Phase 6a — Native shared-outer kernel flag (done; kernel + op + tests)

Splits the original Phase 6 into a focused "native shared-outer flag" landing now and a separate "GPU-side LoRA expansion" follow-up (Phase 6b). This step removes the load-time replication overhead without touching the host-sync that blocks CUDA-graph capture (F7).

Implementation:

- [`cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h`](../../../cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h) and [`cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h`](../../../cpp/tensorrt_llm/kernels/internal_cutlass_kernels/include/moe_kernels.h): six new `LoraParams` fields, defaulting to `false`:
  - `fc1_shared_a`, `fc1_shared_b`, `fc2_shared_a`, `fc2_shared_b`, `gated_shared_a`, `gated_shared_b`.
- [`cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`](../../../cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu): `setupLoraWorkspace` reads these flags and gates the corresponding `weight_index * dim * lora_rank` offset; when set, the per-token A or B pointer dereferences a single unreplicated buffer (e.g. A: `[rank, in_dim]`, B: `[out_dim, rank]`).
- [`cpp/tensorrt_llm/thop/moeOp.cpp`](../../../cpp/tensorrt_llm/thop/moeOp.cpp): six new optional `bool` kwargs on `runMoe`, propagated through `buildMoeLoraParams` to the returned `LoraParams`.
- [`tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`](../../../tensorrt_llm/_torch/custom_ops/torch_custom_ops.py): matching schema additions on `fused_moe` and its `register_fake`.
- [`tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`](../../../tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py): `_resolve_moe_shared_flags` picks up `lora_params["moe_shared_flags"]` and threads the booleans through both `_extract_moe_lora_tensors` paths (per-request and CUDA-graph).
- [`tensorrt_llm/_torch/peft/lora/moe_layout.py`](../../../tensorrt_llm/_torch/peft/lora/moe_layout.py): new `make_native_shared_lora(shared_side=...)` returns the unreplicated layout plus matching shared flags; `expand_native_shared_for_reference` broadcasts back to `[E, ...]` for the eager reference.

Tests:

- CPU layout tests in [`tests/unittest/_torch/lora/test_moe_layout.py`](../../../tests/unittest/_torch/lora/test_moe_layout.py): unreplicated shapes, seed reproducibility, expand-for-reference equivalence, native-vs-replicated reference delta agreement.
- GPU tests in [`tests/unittest/_torch/lora/test_moe_lora_op.py`](../../../tests/unittest/_torch/lora/test_moe_lora_op.py):
  - `test_moe_native_shared_outer_matches_replicated_bitidentical`: native shared-outer with `fc1_shared_a=True`, `fc2_shared_b=True`, `gated_shared_a=True` produces a bit-identical kernel output to the load-time-replication baseline (atol=rtol=0).
  - `test_moe_native_shared_outer_differs_from_no_lora`: sanity smoke that the native path's adapter actually moves the output.

### Phase 6a follow-up — Loader plumbing (DONE)

The kernel and op already accept native shared-outer adapters; this follow-up
threads the convention through the loader and per-request assembler so the
flag is reachable from a real adapter directory rather than only from
synthetic tests.

What landed:

- New parser at [`tensorrt_llm/lora_layout_sidecar.py`](../../../tensorrt_llm/lora_layout_sidecar.py) reads an optional `lora_layout.json` next to `adapter_config.json` with schema:
  ```json
  {
    "version": 1,
    "moe_shared_outer": {
      "moe_h_to_4h": {"shared_side": "A"},
      "moe_gate":    {"shared_side": "A"},
      "moe_4h_to_h": {"shared_side": "B"}
    }
  }
  ```
  Unknown module names, bad `shared_side` values, or unsupported versions raise `LoraLayoutError` with a path-qualified message.
- [`LoraManager`](../../../tensorrt_llm/lora_manager.py) stashes the resulting six-bool flag dict per uid (`_uid_to_moe_shared_flags`) inside `load_from_model_dir`, and exposes `get_moe_shared_flags(uid)`. Adapters without a sidecar get an all-False entry.
- [`ModelEngine._get_eager_lora_params_from_requests`](../../../tensorrt_llm/_torch/pyexecutor/model_engine.py) computes the union of active uids' flags via the manager and writes the result into `lora_params["moe_shared_flags"]` when non-trivial. Mismatched flags across active uids raise (the fused-MoE op applies one global flag set per call).
- CPU parser tests in [`tests/unittest/_torch/lora/test_lora_layout_sidecar.py`](../../../tests/unittest/_torch/lora/test_lora_layout_sidecar.py); MoE loader-roundtrip tests in [`tests/unittest/others/test_lora_manager.py::TestLoraManagerMoeSharedFlags`](../../../tests/unittest/others/test_lora_manager.py).

**Important scope note (this release is metadata-only).** The C++ LoRA cache row size is determined by `LoraModule::localTotalSize` (see [`cpp/tensorrt_llm/runtime/loraCache.cpp:476`](../../../cpp/tensorrt_llm/runtime/loraCache.cpp)), which for MoE bakes in the `num_experts` factor. The loader therefore still replicates shared sides to `[E, ...]` before packing into `_cpp_lora_weights`. With the flag set, the kernel zero-offsets and reads only one slice; without it, the kernel reads every (identical) slice. Output is bit-identical either way. The sidecar exists to:

1. Establish the public convention so adapter producers can mark their files.
2. Wire the flag end-to-end (loader → manager → assembler → op → kernel) so a single follow-up PR can drop the load-time replication without re-touching the call chain.

Memory-savings follow-up (separate PR): change `LoraModule::localTotalSize` to skip the `num_experts` factor for shared sides (or otherwise size the cache row from the actual packed blob), skip the loader's `torch.stack` for shared sides, and accept unreplicated `[rank, in_dim]` / `[out_dim, rank]` shapes directly from disk. The end-to-end test fixture in `TestLoraManagerMoeSharedFlags` is already structured to pivot to that path.

## Phase 6b — GPU-side LoRA expansion (follow-up PR)

Lifts F7: replaces the host-CPU branching in both `setupLoraWorkspace` and `LoraImpl::run` with a device-side problem-builder kernel that reads `permuted_rows` and the per-adapter pointer tables on the device and writes the `device_permuted_*_ptrs` directly. Drops the host `cudaEventSynchronize`, makes the LoRA path fully stream-bound, and unblocks CUDA-graph capture of LoRA-active MoE layers.

Validation gate: capture+replay numerical parity vs eager (re-enables `test_moe_lora_slot_indexed_cuda_graph_replay_matches_eager`).

### Phase 7 — Docs

User-facing page under `docs/source/features/` summarizing supported backends, the shared-outer adapter convention, and the replication-vs-native trade-off.
