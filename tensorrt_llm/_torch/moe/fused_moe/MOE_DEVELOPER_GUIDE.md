# MoE Developer Guide

## Architecture

### MoE Layer in Model

```text
Input Hidden States
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
   fc_gate (Router)     Shared Expert (optional)
       │                      │
       ▼                      │
  Fused-MoE                   │
  ┌─────────────────────┐     │
  │ Routing (topK, etc) │     │
  │         │           │     │
  │         ▼           │     │
  │   MoE Backends      │     │
  │  (FC1→Act→FC2)      │     │
  │         │           │     │
  │   Apply Weights     │     │
  └─────────────────────┘     │
       │                      │
       ▼                      ▼
    Combine Outputs (sum) ◄───┘
       │
       ▼
  Final Hidden States
```

### ConfigurableMoE: The Orchestrator

`ConfigurableMoE` composes independent components via composition (not inheritance) and **owns module lifecycle** (backend construction, weight loading, comm-strategy creation, `repeat_idx` advancement, DWDP record). Forward-time execution is delegated to a **scheduler**:

```text
ConfigurableMoE
├── Backend           (pure computation: routing → quantize → FC1 → act → FC2)
├── Communication     (distributed, optional: dispatch tokens → compute → combine)
├── EPLB              (optional: dynamic expert migration across GPUs)
├── DWDP              (optional: bind composite-VA weights; no EP communication)
└── MoEScheduler      (forward-execution strategy: chunking, EPLB hook ordering,
                       comm orchestration; selected by backend.scheduler_kind)
```

`forward_impl` is thin — it resolves `output_dtype`, delegates to `self.scheduler.forward(...)`, then runs wrapper-level bookkeeping that both schedulers share:

```python
def forward_impl(self, x, router_logits, ...):
    if self.enable_dwdp:
        self.dwdp_manager.wait_and_bind(self.backend, self.layer_idx)
    outputs = self.scheduler.forward(x, router_logits, ...)
    if self.enable_dwdp:
        self.dwdp_manager.finish_layer(self.layer_idx)
    self.repeat_idx = (self.repeat_idx + 1) % self.repeat_count
    return outputs
```

### Scheduler Selection (`MoESchedulerKind`)

Each backend declares one of two scheduler kinds via the `scheduler_kind` class attribute (defined on `MoE` base, default `EXTERNAL_COMM`):

| Kind | Scheduler class | Used by | Cross-rank EP exchange |
|------|-----------------|---------|------------------------|
| `EXTERNAL_COMM` | `ExternalCommMoEScheduler` | Cutlass, DeepGemm, CuteDSL, DenseGEMM, TRTLLMGen, Marlin | Host issues `Communication.dispatch` / `.combine` outside the MoE kernel; supports per-chunk EPLB hooks and multi-stream chunk overlap |
| `FUSED_COMM` | `FusedCommMoEScheduler` | DeepgemmCudaW4a8Mxfp4Mxfp8Impl, TrtllmCutedslMegaMoeNvfp4Impl | Comm is fused into the backend kernel via SymmBuffer / NVSHMEM-equivalent peer-pointer mapping; no host comm; lockstep chunk launches; EPLB stats AllReduced internally |

The two paths have *deliberately opposite* invariants (`use_dp_padding` honored vs ignored, ADP padding kept vs stripped, empty-chunk substituted vs zero-token kernel launch, multi-stream overlap allowed vs forbidden). See `moe_scheduler.py` class docstrings and `MOE_SCHEDULER_DESIGN.md` for the full contract.

### DWDP execution boundary

DWDP is wrapper-owned lifecycle, not a scheduler or backend variant:

```text
Executor construction
  → register eligible MoE layer indices
  → after weight loading: create VMM transport + composite VA + page pool

PyExecutor forward start
  → DwdpManager.start_forward(): first MoE layers, unless the previous
    pass already prefetched them
  → ConfigurableMoE.wait_and_bind(composite VA)
  → MoEScheduler.forward(... ordinary CuTeDSL kernel ...)
  → main stream joins scheduler auxiliary work
  → DwdpManager records current-slot consumption
  → prefetch the next MoE layer in the cyclic sequence whose slot is now free
    (the last layers of a pass prefetch the first layers of the next pass)

Control action (weight update, sleep/wakeup)
  → DwdpManager.invalidate_prefetch()

Executor shutdown or construction rollback
  → detach model references
  → synchronize outstanding device work
  → release composite VA, page pool, transport, and MPI sub-communicator
```

`modules/dwdp/` owns the shareable CUDA VMM transport, MNNVL fabric
allocations on GB200, composite mappings, page pool, copy stream, and event
protocol. `ConfigurableMoE` owns wait/bind/record placement so those calls
happen once per layer rather than once per scheduler chunk. Schedulers must not
perform DWDP lifecycle calls.

The composite mapping exposes the full expert table through the backend's
ordinary weight attributes, so DWDP returns `None` from
`_create_comm_strategy_auto` and continues through the `EXTERNAL_COMM`
CuTeDSL scheduler without dispatch/combine. `DwdpConfig.contention_opt=True`
changes only remote-copy submission: it uses a cached, peer-interleaved
`cudaMemcpyBatchAsync` plan instead of the default per-peer tensor copies.

The currently supported activation path is NVFP4 + CuTeDSL, TP=1 per context
worker, an MPI worker launch, overlap scheduling disabled, and no EPLB on the
same MoE path. Rubin SM107 also requires `use_fused_finalize`. Treat these as
supported-path constraints; the CuTeDSL/NVFP4 activation predicate is not a
comprehensive early-validation layer for every incompatible feature.

### External-comm execution flow (most backends)

`ExternalCommMoEScheduler._forward_chunk_impl` runs per chunk:

```text
[EPLB start_wait_gpu] → routing → [EPLB done_wait_gpu + update_statistic + route]
  → [comm.prepare_dispatch (NVLink2-sided)] → quantize/dispatch (adaptive order)
  → backend.run_moe → [EPLB start_set_cpu] → [unfinalized_combine_fn]
  → comm.combine → [EPLB done_set_cpu]

Adaptive quantize/dispatch order (gated by comm.supports_post_quant_dispatch()):
  Post-quant flow: quantize_input() → comm.dispatch()   (send quantized data)
  Pre-quant flow:  comm.dispatch() → quantize_input()   (send raw, quantize locally)
```

EPLB hooks fire only at the first/last chunk of the first/last `repeat_idx`. Multi-stream chunk overlap is enabled when `not enable_alltoall and aux_stream is not None`, unless the backend declares `input_requirement.run_moe_reuses_buffers`: its `run_moe` returns a view of a buffer the next call overwrites and keeps scratch across calls (the B12x leaves, whose `B12xMoEWrapper` output buffer is also shared across layers). `ExternalCommMoEScheduler` then runs its chunks on one stream and copies every chunk but the last out before the next chunk runs — the trailing `torch.cat` copies the last one. `FusedCommMoEScheduler` copies no chunk output at all, so it refuses a backend that declares the field rather than aliasing its chunks.

`unfinalized_combine_fn` is a constructor argument (`create_moe(..., unfinalized_combine_fn=...)`) for a model that needs a per-`(token, expert)` transform applied to each expert output *before* the top-k sum. Once registered it runs on **every** `do_finalize=False` forward, with or without a comm strategy, so the forward's return type (dense tensor vs. unfinalized triple) depends only on static config. Under a comm strategy the scheduler runs it on the dispatched rows and the combine reduces already-finalized partials — equal to finalizing after the combine, because each term depends on one row alone and non-local slots contribute exact zeros. That reordering is only sound for a combine that is an unweighted row sum over a dense per-token tensor; `DeepEPLowLatency.combine` is not one (it consumes expert-major rows and applies the routing weights itself), which is what `Communication.supports_finalize_before_combine()` declares. `ConfigurableMoE.validate_backend` rejects at construction: a backend without `supports_unfinalized_output()`, a comm without `supports_finalize_before_combine()`, and `apply_router_weight_on_input`. With no hook registered, `do_finalize=False` plus a comm strategy stays a named runtime refusal instead of a crash inside the comm layer.

### Fused-comm execution flow (MegaMoE-style)

`FusedCommMoEScheduler._forward_chunk` runs per chunk:

```text
[EPLB start_wait_gpu] → routing → [EPLB done_wait_gpu + update_statistic + route]
  → backend.quantize_input → backend.run_moe (fused dispatch+GEMM+act+GEMM+combine)
  → [EPLB start_set_cpu + done_set_cpu]
```

No external `Communication.dispatch` / `.combine`. Zero-token chunks still launch the kernel so peer EP ranks can cross the in-kernel NVLink barrier.

The DeepGEMM kernel's in-kernel barriers trap when a peer is late; their timeout is relaxed during warmup, see [Communication timeouts](#communication-timeouts).

### Communication timeouts

Several MoE communication kernels wait for their peers inside the kernel and trap once a peer is late by more than a device-side timeout. The trap is a sticky CUDA error, so every waiting rank dies. During warmup, one-time per-rank work (kernel JIT, autotuning, module loading) can delay a healthy rank by minutes with no collective in between, so these timeouts are relaxed while the engine warms up:

- `moe_comm_timeout_guard.py` owns the process-wide `MoECommTimeoutGuard`. It resolves two budgets, `TRTLLM_MOE_COMM_WARMUP_TIMEOUT_SEC` (default 1800 s) and `TRTLLM_MOE_COMM_TIMEOUT_SEC` (default: each library's native timeout), and on each transition pushes the current one to every registered `MoECommTimeoutProxy`. `CommunicationFactory.create_strategy` resolves the budgets before it tries any strategy, so an invalid value fails there instead of reading as an unavailable backend. The NVLinkOneSided-only names `TRTLLM_MOE_A2A_[WARMUP_]TIMEOUT_SEC` and `TRTLLM_NVLINK_ONE_SIDED_A2A_[WARMUP_]TIMEOUT_SEC` are deprecated aliases that now apply to every backend.
- The model engine's outermost `warmup_scope()` calls `set_moe_comm_warmup` on entry and on exit. The CUDA-graph pass that captures runs inside `moe_comm_serving_timeouts()`, because captured graphs keep the timeouts that were current at capture. Switching while a CUDA graph is being captured raises.
- Each backend with such a timeout registers a proxy when it is constructed. The guard holds proxies weakly, so the backend must keep its proxy alive:

| Backend | Proxy applies the timeout through | Native serving timeout |
|---------|-----------------------------------|------------------------|
| NVLinkOneSided | `torch.ops.trtllm.moe_a2a_set_timeout`; copied into launch arguments | 300 s |
| DeepGEMM MegaMoE | `deep_gemm.set_barrier_timeout_seconds` (`3rdparty/patches/deepgemm_configurable_barrier_timeout.patch`); a kernel argument | 60 s, or `DG_JIT_BARRIER_TIMEOUT_SECONDS` |
| DeepEP normal kernels | `deep_ep.set_timeout_seconds` (`3rdparty/patches/deep_ep_runtime_timeout.patch`); a device variable read at execution | ~100 s |
| NcclEP | `nccl.ep.Group.set_timeout_ns`, one proxy per group (`3rdparty/patches/nccl_ep_group_timeout.patch`); copied into launch arguments | `NCCL_EP_TIMEOUT_MS`, else `GroupConfig.timeout_ns`, else ~100 s |

Timeouts are nominal: NVLinkOneSided, DeepGEMM and DeepEP convert seconds at an assumed 2 GHz SM clock, while NcclEP uses the device's clock rate.

### Core Design Principles

1. **Composition over inheritance** — Backend, Communication, EPLB, and Scheduler are independent, composable components
2. **Any Backend × Any Communication × EPLB On/Off** — All valid combinations should work (subject to `can_implement` and `scheduler_kind`)
3. **Backend = pure computation** — No communication logic, no EPLB logic inside backends
4. **Communication is pluggable** — `EXTERNAL_COMM` backends pick a strategy via `CommunicationFactory` based on hardware/workload; `FUSED_COMM` backends bypass external comm entirely
5. **Backend declares capabilities** — `can_implement(p, d)` is the single source of truth for what a backend supports, and it is a **pure function** of its two arguments (see [Backend Selection](#backend-selection)); the resolver holds no capability knowledge of its own
6. **Backend declares scheduler** — `scheduler_kind` class attribute selects the forward path; lifecycle code stays generic, forward path stays specialized
7. **One class per identity** — a registered implementation is one `provider.technique.kernel_name.quant` id serving one quantization format for the life of the layer (see [Canonical Examples](#canonical-examples))

## Architecture Transition (IMPORTANT)

The codebase is transitioning between two architectures:

| | Old Path | New Path |
|---|---|---|
| Entry | `XXFusedMoE` (e.g., `TritonFusedMoE`) | `ConfigurableMoE` + `XXBackend` + `MoEScheduler` |
| Communication | Embedded inside each backend | Separated into `communication/` (or fused into kernel for `FUSED_COMM`) |
| Forward execution | Inline in backend | `MoEScheduler` (`moe_scheduler.py`) |
| EPLB | Not supported | Available on EPLB-capable backends |
| Status | Being replaced | Active development |

ConfigurableMoE currently supports these backends (`create_moe.py`):
- the `CutlassFusedMoEBase` leaves, the `TrtllmGenFusedMoEBase` leaves,
  `DeepgemmCudaFp8BlockScalesImpl`, the `CuteDslFusedMoEBase` leaves, the two
  B12x leaves, `TrtllmCutedslFusedFc12Nvfp4Impl`,
  `TrtllmCutedslDenseGemmNvfp4Impl`, `DeepgemmCudaW4a8Mxfp4Mxfp8Impl`,
  `TrtllmCutedslMegaMoeNvfp4Impl`, the `MarlinFusedMoEBase` leaves

Still on old path (standalone, with embedded communication):
- `TritonFusedMoE`, `VanillaMoE`

**Rule: All new features should target ConfigurableMoE + Backend + Scheduler architecture.**

## File Map

### Core (`fused_moe/`)

| File | Role |
|------|------|
| `configurable_moe.py` | Orchestrator — wires Backend + Communication + EPLB + Scheduler; owns lifecycle and `forward_impl` |
| `moe_scheduler.py` | Forward-execution strategies (`MoEScheduler` ABC, `ExternalCommMoEScheduler`, `FusedCommMoEScheduler`, `create_moe_scheduler` factory) |
| `create_moe.py` | Factory — builds the layer once `moe_resolution` has named the class |
| `moe_resolution.py` | **The one selection entry point** (`resolve_moe_impl`) — orders candidates, asks each one's `can_implement`, returns a `MoEResolutionReport` |
| `impl_contract.py` | Selection vocabulary — `MoEProblem`, `MoEDeployment`, `MoEEnvironment`, `MoEEligibility`, `MoERejectReason`, `MoEResolutionReport` |
| `impl_environment.py` | The only place that probes the machine (SM, optional wheels, env flags) and freezes the result |
| `impl_identity.py` | `MoEImplId` / `MoEImplDescriptor` / registry — the stable one-id-per-leaf-class mechanism used after an implementation migrates |
| `interface.py` | Complete-layer base `MoE` and enums (`MoEWeightLoadingMode`, `MoESchedulerKind`) |
| `impl_base.py` | Execution-unit base `MoEImplBase` — weights + `run_moe`, no `forward`; plus `apply_moe_impl_construction_state()`, which every execution unit must call |
| `activation.py` | Activation vocabulary — the `MoEActivation` carrier a model builds, the `MoEActivationSupport` a backend declares, and `install_activation_params` / `materialize_activation_params`, the only place a semantic constant becomes a kernel register |
| `impl_blocks.py` | The blocks `MoE` and `MoEImplBase` share — `MoEExecutionContractMixin` (scheduler-facing declarations, `forward_fake`) and `MoEWeightOwnerMixin` (`create_weights` / `load_weights` / `_check_configs`) |
| `quantization.py` | Quantization method implementations (`FusedMoEMethod` subclasses: weight creation, loading, quant/dequant ops per quant mode) |
| `routing.py` | Routing methods (`TopKRouting`, etc.) |
| `moe_comm_timeout_guard.py` | Device-side timeouts of communication kernels, relaxed during warmup — budget resolution and `MoECommTimeoutProxy` registration (see [Communication timeouts](#communication-timeouts)) |
| `moe_load_balancer.py` | EPLB implementation |
| `per_layer_dynamic_eplb/` | Per-layer dynamic EPLB, created by `maybe_create_moe_load_balancer`: model-level `manager`, per-layer handle `layer`, live-weight `arena`, and HALO-Q/TMA `planner` |
| `moe_op_backend.py` | Op backend registry for TRTLLMGen (flashinfer/trtllm ops) |

### Backends (`fused_moe/`)

| File | Backend | Hardware | Scenario | Scheduler |
|------|---------|----------|----------|-----------|
| `cutlass/` (+ the `fused_moe_cutlass.py` alias module) | the abstract `CutlassFusedMoEBase` and its twelve leaves (`CUTLASS_LEAVES`): one per quantization format, two for FP8 block scales (see [CUTLASS leaves](#cutlass-leaves-fused_moecutlass)) | SM80+ | High throughput, the widest quant support; the fallback family | `EXTERNAL_COMM` |
| `trtllm_gen/` (+ the `fused_moe_trtllm_gen.py` alias module) | the abstract `TrtllmGenFusedMoEBase` and its leaves (see [TRTLLM-Gen leaves](#trtllm-gen-leaves-fused_moetrtllm_gen)) | SM100/SM103/SM107 | Min-latency and high-throughput on Blackwell/Rubin; also serves unquantized BF16 through FlashInfer, on a gate of its own (`§` under [Quantization Support](#quantization-support)) | `EXTERNAL_COMM` |
| `fused_moe_deepgemm.py` | `DeepgemmCudaFp8BlockScalesImpl` (aliased as `DeepGemmFusedMoE`) | SM100/SM103/SM107 | FP8 Block Scales on Blackwell/Rubin | `EXTERNAL_COMM` |
| `cutedsl/dense_gemm.py` (+ the `fused_moe_densegemm.py` alias module) | `TrtllmCutedslDenseGemmNvfp4Impl` (aliased as `DenseGEMMFusedMoE`) | SM100/SM103 | NVFP4 min-latency; CuTe DSL dense GEMM packs all experts into one matrix (vs Cutlass per-expert scatter), efficient for small token counts | `EXTERNAL_COMM` |
| `cutedsl/grouped_gemm/` (+ the `fused_moe_cute_dsl.py` alias module) | the abstract `CuteDslFusedMoEBase` (aliased as `CuteDslFusedMoE`) and its two leaves, `TrtllmCutedslGroupedGemmNvfp4Impl` (`trtllm.cutedsl.grouped_gemm.nvfp4`) and `TrtllmCutedslGroupedGemmBf16Impl` (`trtllm.cutedsl.grouped_gemm.none`) | SM100/SM103/SM107; BF16 SM107 only | High throughput NVFP4, generally faster than Cutlass; unquantized BF16 SwiGLU on Rubin | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl_b12x.py` | `CuteDslB12xNvfp4FusedMoE` and `CuteDslB12xW4a16Nvfp4FusedMoE` (shared mixin `CuteDslB12xFusedMoEBase`, aliased as `CuteDslB12xFusedMoE`) | SM120/SM121 | NVFP4 (CUTLASS prefill, FlashInfer b12x decode) and W4A16 NVFP4 (b12x for every call), best on RTX PRO 6000 and DGX Spark. They head the `CUTEDSL` family, so they win on SM120/121 when flashinfer is present and yield to the grouped-GEMM leaves otherwise. BF16 activations only (`B12xMoEWrapper` reads and writes bfloat16), so a float16 layer falls back to Cutlass. The NVFP4 leaf also declines `force_dynamic_quantization` (`DYNAMIC_QUANT_UNSUPPORTED`): b12x quantizes decode activations against load-time scales, so such a layer falls back to Cutlass too. Single-GPU-shaped topology only: with no dispatch/combine kernel they reject `ep_size > 1` and attention-DP | `EXTERNAL_COMM` |
| `cutedsl/fused_fc12.py` (+ the `fused_moe_cute_dsl_fc12.py` alias module) | `TrtllmCutedslFusedFc12Nvfp4Impl` (`trtllm.cutedsl.fused_fc12.nvfp4`, aliased as `CuteDslFc12FusedMoE`) | SM107 (Rubin) | NVFP4 fused FC1+FC2 in one persistent kernel (`cute_dsl_nvfp4_fc12_fused_rubin`) with a trace-time SwiGLU or SiTU epilogue, keeping the FC1→FC2 intermediate on-chip; shares `TrtllmCutedslGroupedGemmNvfp4Impl`'s NVFP4 weight layout and autotune runner; reachable through the `CUTEDSL_FC12` family or a pinned `impl_id`; needs `MoEDep.CUTEDSL_RUBIN`; non-uGPU | `EXTERNAL_COMM` |
| `mega_moe/mega_moe_deepgemm.py` | `DeepgemmCudaW4a8Mxfp4Mxfp8Impl` (aliased as `MegaMoEDeepGemm`) | SM100/SM103/SM107 | W4A8_MXFP4_MXFP8 via DeepGEMM `fp8_fp4_mega_moe` fused dispatch+GEMM+act+GEMM+combine kernel; requires `hidden_size % 512 == 0` | `FUSED_COMM` |
| `mega_moe/mega_moe_cute_dsl.py` | `TrtllmCutedslMegaMoeNvfp4Impl` (aliased as `MegaMoECuteDsl`) | SM100/SM103/SM107 | NVFP4 fused dispatch+FC1+act+FC2+combine; internally selects one of its per-architecture kernel variants. Requires CUDA 13 Cutlass DSL and a symmetric-memory provider. Uses uniform activation constants and applies routing weights before FC2 quantization for DeepSeek-V4, after FC2 for other models | `FUSED_COMM` |
| `marlin/` (+ the `fused_moe_marlin.py` alias module) | the abstract `MarlinFusedMoEBase` and its two leaves, `MarlinCudaNvfp4Impl` and `MarlinCudaW4a16Nvfp4Impl` | SM89-SM99 | W4A16 NVFP4 on Ada/Hopper (BF16 activations + FP4 weights, single-launch `marlin_nvfp4_moe_gemm`); attention-DP + EP via external comm (plain BF16 dispatch payload); non-NVFP4 layers (e.g. unquantized MTP draft layers) degrade to Cutlass, recorded in the `MoEResolutionReport`; no dynamic EPLB | `EXTERNAL_COMM` |
| `fused_moe_triton.py` | `TritonFusedMoE` | SM90 only | GPT-OSS on Hopper (requires `swiglu_gptoss_style=True`) | (legacy path) |
| `fused_moe_vanilla.py` | `VanillaMoE` | All devices | Reference / debugging only | (legacy path) |

### Communication (`fused_moe/communication/`)

`CommunicationFactory` (`communication_factory.py`) picks a strategy at runtime
from hardware and configuration; `FUSED_COMM` backends skip it. `base.py` holds
the `Communication` ABC.

A strategy whose native workspace must follow executor sleep/wakeup implements
the runtime-checkable `CheckpointableCommunication` protocol (`base.py`).
`checkpoint_resource_key()` identifies wrappers that share one workspace, so the
executor calls `checkpoint_prepare()` / `checkpoint_restore()` once per
resource. Discover strategies through the protocol, never through concrete
strategy checks or private workspace attributes. The executor request queue
owns admission around the checkpoint and reopens it only after a complete
wakeup.

### CUTLASS leaves (`fused_moe/cutlass/`)

Everything CUTLASS lives in this package. `fused_moe_cutlass.py` above it
defines the family name `CutlassFusedMoE` (the abstract `CutlassFusedMoEBase`,
parent of every leaf) and re-exports `raise_moe_lora_multichunk_unsupported`.

| File | Role |
|------|------|
| `identity.py` | CUTLASS's values of the identity types: the segment constants, the shared contracts (`CUTLASS_CAPABILITIES`, `CUTLASS_INPUT_REQUIREMENT`) and the `cutlass_descriptor` factory, whose `supports_moe_lora` and `supports_expert_bias` are the per-leaf contract differences |
| `base.py` | `CutlassFusedMoEBase` — construction, the layer-format check in `create_weights`, and the defaults: a pass-through `quantize_input` and a `run_moe` that calls the op with the leaf's `FUSED_MOE_OP_FLAGS` (the weight lifecycle itself is `MoEWeightOwnerMixin`'s). Also the one `torch.ops.trtllm.fused_moe` call (`_run_fused_moe_op`, which a leaf's `run_moe` calls with the operands the `run_moe` contract does not carry), `FusedMoeOpFlags`, and the `quantize_input` implementations several formats bind (`quantize_input_static_e4m3`, `quantize_input_mxfp8`) |
| `eligibility.py` | The checks each leaf's `can_implement` composes |
| `lora.py` | `CutlassMoELoraMixin` — routed-expert LoRA, mixed in only by the two leaves the op can fuse it for |
| `<format>.py` | One registered leaf each, named after its format: `unquantized.py` for `none`, `fp8_block_scales_hopper.py` / `fp8_block_scales_triton.py` for the two `fp8_block_scales` leaves, the `quant` segment otherwise |

| `quant` segment | Leaf | Id | What differs from the base |
|---|---|---|---|
| `none` | `TrtllmCutlassUnquantizedImpl` | `trtllm.cutlass.grouped_gemm.none` | Fuses routed-expert LoRA |
| `fp8` | `TrtllmCutlassFp8Impl` | `trtllm.cutlass.grouped_gemm.fp8` | Static per-tensor e4m3 input; fuses routed-expert LoRA |
| `fp8_block_scales` | `DeepgemmCudaHopperFp8BlockScalesImpl` | `deepgemm.cuda.hopper_grouped_gemm.fp8_block_scales` | SM90; `CutlassMoeFCRunner` hands FC1/FC2 to the vendored DeepGEMM |
| `fp8_block_scales` | `TrtllmTritonFp8BlockScalesImpl` | `trtllm.triton.grouped_gemm.fp8_block_scales` | SM120/121, BF16 activations; a Triton grouped GEMM that never enters C++ (`issues_fused_moe_op = False`, so no build gate); runs SwiGLU/GeGLU only, takes no expert bias and no activation constant but a uniform pre-SiLU SwiGLU clamp |
| `nvfp4` | `TrtllmCutlassNvfp4Impl` | `trtllm.cutlass.grouped_gemm.nvfp4` | Quantizes activations to NVFP4 itself |
| `w4a16_nvfp4` | `TrtllmCutlassW4a16Nvfp4Impl` | `trtllm.cutlass.grouped_gemm.w4a16_nvfp4` | Dequantizes the weights every forward, then runs unquantized |
| `mxfp8` | `TrtllmCutlassMxfp8Impl` | `trtllm.cutlass.grouped_gemm.mxfp8` | Dynamic MXFP8 input |
| `w4a8_mxfp4_mxfp8` | `TrtllmCutlassW4a8Mxfp4Mxfp8Impl` | `trtllm.cutlass.grouped_gemm.w4a8_mxfp4_mxfp8` | Dynamic MXFP8 input |
| `w4a8_mxfp4_fp8` | `TrtllmCutlassW4a8Mxfp4Fp8Impl` | `trtllm.cutlass.grouped_gemm.w4a8_mxfp4_fp8` | Static per-tensor e4m3 input |
| `w4a16_mxfp4` | `TrtllmCutlassW4a16Mxfp4Impl` | `trtllm.cutlass.grouped_gemm.w4a16_mxfp4` | Rounds hidden and intermediate up to 128 at construction; `run_moe` pads the activation to match |
| `w4a8_awq` | `TrtllmCutlassW4a8AwqImpl` | `trtllm.cutlass.grouped_gemm.w4a8_awq` | — |
| `w8a16` | `TrtllmCutlassW8a16Impl` | `trtllm.cutlass.grouped_gemm.w8a16` | — |

- **A leaf declares; the base derives.** A leaf states its `descriptor`, its
  SM and dtype sets, gpt-oss support and `FUSED_MOE_OP_FLAGS`, and implements
  `can_implement` / `_get_quant_method`. `MoEImplBase.__init_subclass__`
  copies `scheduler_kind`, `capabilities` and `input_requirement` off the
  descriptor, so the contract has one source, and nothing in the base branches
  on `quant_config`.
- **FP8 block scales splits by SM.** Its two leaves carry the providers and
  techniques of the kernels they run (DeepGEMM, Triton), yet stay in the family
  because the `CUTLASS` literal serves FP8 block scales on SM90 and SM120/121.
  `find_cutlass_leaf(quant_algo, *, sm=None)` (and `cutlass_leaf`, which
  raises instead of returning `None`) is a lookup, not an eligibility check, and
  reads the SM only to choose between those two.

### TRTLLM-Gen leaves (`fused_moe/trtllm_gen/`)

Everything TRTLLM-Gen lives in this package. `fused_moe_trtllm_gen.py` above it
holds no code: it defines `TRTLLMGenFusedMoE` (the abstract
`TrtllmGenFusedMoEBase`) and re-exports `trtllm_gen_leaf` and
`find_trtllm_gen_leaf`.

| File | Role |
|------|------|
| `identity.py` | TRTLLM-Gen's values of the types `impl_identity.py` defines: the identity constants (`PROVIDER_*`, `TECHNIQUE_TRTLLM_GEN`, `KERNEL_FUSED_MOE`), the two published contracts (`TRTLLM_GEN_CAPABILITIES`, `TRTLLM_GEN_INPUT_REQUIREMENT`), and the `trtllm_gen_descriptor` factory that stamps those segments into what the registry reads |
| `base.py` | `TrtllmGenFusedMoEBase` — the abstract root every leaf shares: construction, the weight-creation skeleton, the two routing predicates the framework reads off a module, the fake-output shapes. Reads both axes as leaf-declared attributes and overridable hooks, never as a branch on `quant_config` or the provider string |
| `fp4_block_scale.py` | `run_fp4_block_scale_moe` and the formats that reach it — NVFP4, W4A16_MXFP4, W4A8_MXFP4_MXFP8 — each supplying only its weight/input preparation, its SiTu alignment, and its scale group size. Two leaves per format, one per provider |
| `fp8_block_scale.py` | DeepSeek-style FP8 with 1x128 block scales: `run_fp8_block_scale_moe`, the narrowed activation ABI (scalar clamp), and shared-expert fusion. Two leaves. Shares nothing with the fp4 module, which is why the two sit side by side rather than under a common quant layer |
| `eligibility.py` | the eligibility checks each leaf's `can_implement` composes, as free functions — every caller is a leaf |
| `kernel_inputs.py` | the `run_moe` prologue the kernel bodies share, as free functions taking the impl |
| `trtllm_<quant>.py` | one registered native leaf each: `nvfp4`, `fp8_block_scales`, `w4a16_mxfp4`, `w4a8_mxfp4_mxfp8`, `w4a8_nvfp4_fp8`, `w4a8_mxfp4_fp8` |
| `flashinfer_<quant>.py` | one registered FlashInfer leaf each: `nvfp4`, `fp8_block_scales`, `w4a16_mxfp4`, `w4a8_mxfp4_mxfp8`, `bf16` |

#### The leaves

Module names carry the two identity segments that differ: `provider` — the
in-tree `trtllm` ops or the FlashInfer package — and `quant`. Technique and
kernel are `trtllm_gen` / `fused_moe` for every leaf, which the package name
says once. The grid is not full: FlashInfer has no `w4a8_nvfp4_fp8` or
`w4a8_mxfp4_fp8` runner and alone serves the unquantized format, so some
cells of the provider x format grid have no leaf.

| `quant` segment | `trtllm.` leaf | `flashinfer.` leaf |
|---|---|---|
| `nvfp4` | `TrtllmTrtllmGenNvfp4Impl` | `FlashinferTrtllmGenNvfp4Impl` |
| `fp8_block_scales` | `TrtllmTrtllmGenFp8BlockScalesImpl` | `FlashinferTrtllmGenFp8BlockScalesImpl` |
| `w4a16_mxfp4` | `TrtllmTrtllmGenW4a16Mxfp4Impl` | `FlashinferTrtllmGenW4a16Mxfp4Impl` |
| `w4a8_mxfp4_mxfp8` | `TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl` | `FlashinferTrtllmGenW4a8Mxfp4Mxfp8Impl` |
| `w4a8_nvfp4_fp8` | `TrtllmTrtllmGenW4a8Nvfp4Fp8Impl` | — |
| `w4a8_mxfp4_fp8` | `TrtllmTrtllmGenW4a8Mxfp4Fp8Impl` | — |
| `none` (BF16) | — | `FlashinferTrtllmGenBf16Impl` |

Ids follow the class names (`trtllm.trtllm_gen.fused_moe.nvfp4`,
`flashinfer.trtllm_gen.fused_moe.none`), keeping the technique and kernel
segments so a future non-TRTLLM-Gen kernel from the same wheel stays
distinguishable. The two native-only formats subclass the family base directly:
a per-quant parent implementing three methods for one subclass buys nothing.

The provider split is a policy call, not a capability one. With the FlashInfer
wheel installed a `TRTLLM` literal still picks the native leaf, and only
`TRTLLM_GEN_FUSED_MOE_USE_FLASHINFER=1` moves it — read through
`MoEEnvironment.env_flag`, so `can_implement` never touches `os.environ`. A
FlashInfer leaf asked for without the flag declines with `PATH_NOT_ENABLED`,
which distinguishes "not chosen" from "not installed" (`DEP_MISSING`).
`IMPL_PRIORITY` lists each FlashInfer leaf ahead of the native leaf of the same
format, so the flag alone decides the order the two are asked in.
`FlashinferTrtllmGenBf16Impl` is outside that policy — see the `§` footnote
under [Quantization Support](#quantization-support).

Only a registered class may be resolved to, and importing the package is what
registers, which `moe_resolution` guarantees. `trtllm_gen_leaf` sits at the
bottom of `__init__.py` because the lookup is only correct once every leaf
module has been imported. `create_moe_backend` dispatches with `issubclass`,
not `==`.

### Marlin leaves (`fused_moe/marlin/`)

Same shape as the TRTLLM-Gen package, one size smaller.
`fused_moe_marlin.py` above it defines `MarlinFusedMoE` as a module-level alias
of `MarlinFusedMoEBase` and re-exports `marlin_leaf` / `find_marlin_leaf`; the
implementation lives in the package.

| File | Role |
|------|------|
| `identity.py` | Marlin's values of the identity types: `PROVIDER_MARLIN` / `TECHNIQUE_CUDA` / `KERNEL_FUSED_MOE`, the two published contracts (`MARLIN_CAPABILITIES`, `MARLIN_INPUT_REQUIREMENT`), and the `marlin_descriptor` factory |
| `base.py` | `MarlinFusedMoEBase` — the abstract root both leaves share: construction, `run_moe`, `quantize_input`, `_get_quant_method`, the activation epilogue and the workspace |
| `eligibility.py` | the checks each leaf's `can_implement` composes, as free functions |
| `nvfp4.py` / `w4a16_nvfp4.py` | one registered leaf each, named after the `quant` segment of its identity |

| `quant` segment | leaf | id |
|---|---|---|
| `nvfp4` | `MarlinCudaNvfp4Impl` | `marlin.cuda.fused_moe.nvfp4` |
| `w4a16_nvfp4` | `MarlinCudaW4a16Nvfp4Impl` | `marlin.cuda.fused_moe.w4a16_nvfp4` |

Unlike TRTLLM-Gen there is no per-kernel-ABI layer between base and leaves,
because Marlin has exactly one: `marlin_nvfp4_moe_gemm` is W4A16 and reads the
weights the same way whichever algorithm the checkpoint declared. So `run_moe`
and the weight method sit on the base and a leaf is an identity plus a
`can_implement`. They are still two identities because the `quant` a leaf
publishes has to be the `quant` the request named — `check_quant_matches_identity`
(in `impl_contract.py`, shared with TRTLLM-Gen) is the gate that enforces it.

`marlin` is the provider and `cuda` the technique: the kernel is vendored into
this repo but its lineage is upstream's, which the device code still says
(`MARLIN_NAMESPACE_NAME marlin_moe_wna16` in
`cpp/tensorrt_llm/kernels/marlin/marlin_nvfp4_moe_gemm.cu`).

### CuteDSL leaves (`fused_moe/cutedsl/`)

The CuteDSL implementations that share code. An id here is
`trtllm.cutedsl.<kernel>.<quant>`, and the layout follows it: one subpackage or
module per kernel, one module per quant where a kernel serves several. The
kernel segment says how much of the expert MLP one launch covers:
`grouped_gemm` runs FC1 and FC2 as two grouped GEMMs, `fused_fc12` runs both in
one persistent kernel, and `dense_gemm` packs every expert into one dense matrix
for small token counts. Inheritance stays inside a kernel; kernels share only the
package-level modules. `fused_moe_cute_dsl.py`, `fused_moe_cute_dsl_fc12.py` and
`fused_moe_densegemm.py` above it hold no code: they define `CuteDslFusedMoE`,
`CuteDslFc12FusedMoE` and `DenseGEMMFusedMoE`. The grouped-GEMM lookups,
`cutedsl_grouped_gemm_leaf` / `find_cutedsl_grouped_gemm_leaf`, are exported by
the package.

| File | Role |
|------|------|
| `identity.py` | the identity segments (`PROVIDER_TRTLLM`, `TECHNIQUE_CUTEDSL`, `KERNEL_*`), and the grouped-GEMM query and id helper |
| `kernel_inputs.py` | what more than one kernel takes: the `NvFp4WeightView` dataclass, and free functions taking the impl, `finalize_routing_scales` and `quantize_input_nvfp4`, which every NVFP4 leaf here binds as `quantize_input` |
| `runners.py` | the outer autotune runners: NVFP4, which the fused-FC12 runner narrows to its tiles, and BF16 |
| `grouped_gemm/base.py` | `CuteDslFusedMoEBase`, the abstract root of the two grouped-GEMM leaves: construction, the output-memset overlap stream, the locality-domain plan, the weight lifecycle |
| `grouped_gemm/eligibility.py` | the checks both grouped-GEMM leaves' `can_implement` compose, as free functions |
| `grouped_gemm/nvfp4.py` / `grouped_gemm/bf16.py` | one registered leaf each |
| `fused_fc12.py` / `dense_gemm.py` | one registered leaf each, with what only that kernel needs |

| kernel / `quant` | leaf | id |
|---|---|---|
| `grouped_gemm` / `nvfp4` | `TrtllmCutedslGroupedGemmNvfp4Impl` | `trtllm.cutedsl.grouped_gemm.nvfp4` |
| `grouped_gemm` / `none` | `TrtllmCutedslGroupedGemmBf16Impl` | `trtllm.cutedsl.grouped_gemm.none` |
| `fused_fc12` / `nvfp4` | `TrtllmCutedslFusedFc12Nvfp4Impl` | `trtllm.cutedsl.fused_fc12.nvfp4` |
| `dense_gemm` / `nvfp4` | `TrtllmCutedslDenseGemmNvfp4Impl` | `trtllm.cutedsl.dense_gemm.nvfp4` |

`trtllm.cutedsl.mega_moe.nvfp4` is CuteDSL too but lives in `mega_moe/`: it is a
fused-communication kernel that shares its scheduler with the DeepGEMM MegaMoE
and no code with this package. The B12x leaves, which the `CUTEDSL` backend
literal also selects, descend from the CUTLASS family and stay in
`fused_moe_cute_dsl_b12x.py`.

### MegaMoE (`fused_moe/mega_moe/`)

| File | Role |
|------|------|
| `mega_moe_deepgemm.py` | `DeepgemmCudaW4a8Mxfp4Mxfp8Impl` backend (DeepGEMM `fp8_fp4_mega_moe` wrapper), aliased as `MegaMoEDeepGemm` |
| `mega_moe_cute_dsl.py` | `TrtllmCutedslMegaMoeNvfp4Impl` backend (architecture-specific CuteDSL MegaMoE wrapper, NVFP4), aliased as `MegaMoECuteDsl` |
| `CHUNKING_DESIGN.md` | Chunking design for MegaMoE (sequential multi-chunk, in-kernel barrier semantics) |
| `COMMUNICATION_COMPARISON.md` | Comparison of fused-comm SymmBuffer vs external comm strategies |
| `KERNEL_INTERNALS.html` | Reference for the underlying DeepGEMM kernel layout |

The exported CuteDSL kernel sources for `TrtllmCutedslMegaMoeNvfp4Impl` live under
`tensorrt_llm/_torch/moe/kernels/cutedsl_megamoe/`, a copy of the upstream
cutedsl_megamoe sources maintained in-tree; update it by copying newer upstream
code in. `moe/kernels/megamoe_kernels.py` exposes the `KernelClass` entry
points. The runner constructs
`ProblemDesc` and `ImplDesc` and dispatches internally; the public backend
remains `MEGAMOE_CUTEDSL`. Rubin's smaller token buckets use the genphase
kernel when supported, otherwise the generic kernel. During warmup, the
default multi-rank path primes its adaptive buckets with zero-token launches.

### Design Documents

| File | Topic |
|------|-------|
| `MOE_SCHEDULER_DESIGN.md` | Scheduler refactor design + `MoEScheduler` contract |
| `mega_moe/CHUNKING_DESIGN.md` | MegaMoE chunking invariants |

### C++ (`cpp/`)

The C++ side mirrors this layout inside each link layer: `kernels/moe/`
(`libtensorrt_llm.so`), `thop/moe/` (`libth_common.so`) and
`tests/unit_tests/kernels/moe/`. Backend directories own their CMake targets;
semantic directories compile into `kernels_src`.

| Path | What lives there | Build target |
|------|------------------|--------------|
| `cpp/tensorrt_llm/kernels/moe/cutlass/` | CUTLASS fused-MoE runner (`CutlassMoeFCRunner`), grouped-GEMM launchers, LoRA helpers; public headers under `include/` | `moe_gemm_src` (+ per-arch OBJECT libraries, defined in `kernels/cutlass_kernels/CMakeLists.txt`) |
| `cpp/tensorrt_llm/kernels/moe/trtllmGen/` | TRTLLM-Gen block-scale MoE runner and device kernels; `routing/` holds its routing kernels | `trtllm_gen_fp8_block_scale_moe`, `trtllm_gen_fp8_block_scale_moe_routing` |
| `cpp/tensorrt_llm/kernels/moe/cuteDsl/` | Host-side utilities for the CuTe DSL MoE backends | `cute_dsl_src` |
| `cpp/tensorrt_llm/kernels/moe/communication/` | MoE all-to-all, all-reduce fusion, fused-comm and prepare kernels | `kernels_src` |
| `cpp/tensorrt_llm/kernels/moe/loadBalance/` | EPLB device kernels (host side: `runtime/moeLoadBalancer/`) | `kernels_src` |
| `cpp/tensorrt_llm/kernels/moe/routing/` | Custom routing kernels | `kernels_src` |
| `cpp/tensorrt_llm/kernels/moe/utils/` | `moe_align_block_size` and other small helpers | `kernels_src` |
| `cpp/tensorrt_llm/thop/moe/{cutlass,trtllmGen,cuteDsl,marlin}/` | `torch.ops.trtllm` MoE ops grouped by the backend they serve; `cutlass/moeOp.cpp` holds the primary `TORCH_LIBRARY(trtllm)` block. `marlin/` exists only here: the op is MoE-only, its kernel stays in `kernels/marlin/` | `th_common` |
| `cpp/tensorrt_llm/thop/moe/{communication,loadBalance,routing,utils}/` | `torch.ops.trtllm` MoE ops grouped by role, same names as the `kernels/moe/` directories they wrap | `th_common` |
| `cpp/tensorrt_llm/runtime/moeLoadBalancer/`, `nanobind/runtime/moeBindings.*` | EPLB host runtime and its Python binding | `runtime_src`, nanobind module |
| `cpp/tests/unit_tests/kernels/moe/` | gtests (`mixtureOfExpertsTest`, `routingKernelsTest`, ...) | per-test executables |

Deliberately outside `moe/`: the shared headers
`kernels/{moeCommKernelsCommon.h,moeTopKFuncs.cuh,moe_utils.cuh}` (they have
non-MoE consumers), the MoE files inside mixed directories (`marlin/`,
`llama4MinLatencyKernels/`, `dsv3MinLatencyKernels/`,
`cutlass_kernels/fp8_blockscale_gemm/sm120_*`), `trtllmGenKernels/batchedGemm/`
(shared GEMM backend), `internal_cutlass_kernels/` (prebuilt; mirrors
`moe/cutlass/include/`), `deep_ep/` and `deep_gemm/`.

### Tests

| File | Tests | Status |
|------|-------|--------|
| `test_moe_backend.py` | Backend unit tests (`run_moe`, `can_implement`, weight loading, numerics) | Active |
| `test_moe_impl.py` | Selection and identity: what a resolution can get wrong (see the Identity Tests row under [Canonical Examples](#canonical-examples)) | Active |
| `test_moe_module.py` | ConfigurableMoE integration tests (Backend × Comm × EPLB) | Active |
| `test_fused_moe.py` | Legacy MoE tests | Being replaced, do NOT add new tests here |
| `test_moe.py` | Legacy TRTLLM backend tests | Being replaced, do NOT add new tests here |

## Backend Selection

One function decides which implementation runs: `moe_resolution.resolve_moe_impl`.
It holds no capability knowledge: it orders candidates, asks each one, and
returns the first that accepts. So "which impl will run" has one answer no
matter who asks, and the factory cannot admit something a backend rejects.

```python
report = resolve_moe_impl(model_config, layer_idx=layer_idx)
impl_cls = impl_class_for(report)      # raises, with the full trail, if nothing fits
```

`MoeConfig.backend` is the only user-facing selector, and it names a family.
Pinning one identity (`impl_id=`) is an argument of `resolve_moe_impl` alone,
for callers that must measure one implementation; neither `MoeConfig` nor
`create_moe` accepts one.

### `can_implement` must be pure

```python
@classmethod
def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
    if d.env.sm not in (100, 103):
        return _reject(MoERejectReason.SM_UNSUPPORTED, f"... got SM{d.env.sm}")
    if not d.env.has_dep(MoEDep.FLASHINFER):
        return _reject(MoERejectReason.DEP_MISSING, "... requires flashinfer")
    return MoEEligibility.ok()
```

Read `p` and `d`, nothing else. Specifically: **no** `get_sm_version()`, **no**
`import` to test whether a wheel exists, **no** `os.environ`. Those probes happen
once in `impl_environment.collect_moe_environment()` and arrive frozen as
`d.env`. A gate that probes the host instead answers a different question on
every machine, which is exactly the irreproducibility the frozen environment
exists to remove.

The same rule is why a gate reads `d.eplb_enabled` and `d.moe_lora_enabled`
instead of `self._supports_load_balancer()` — a predicate on `self` can only be
consulted after the object it might have to reject already exists.

Unknown is not false. A shape field can be `None` when the caller does not know
it yet, and a gate that reads `None` must abstain rather than reject, or a
missing `pretrained_config` attribute turns into a backend downgrade.

Abstaining has no state of its own: `MoEEligibility` is two-valued, so a gate
abstains by skipping its check and returning `MoEEligibility.ok()`. Read that
literally — `resolve_moe_impl` counts the candidate as eligible and may pick it
without the shape constraint ever having been proven. That is the accepted cost:
rejecting on absent information downgrades a backend that would have been
perfectly legal, and a caller who wants the constraint checked has to supply the
shape.

### Adding a new probe

Add a member to `MoEDep` or `MoEEnvFlag` and a probe function to the table in
`impl_environment.py`. Both enums are closed on purpose: a name not declared
there cannot be read during selection, which is what keeps the environment an
explicit input instead of a growing set of implicit ones.

### Degradation is allowed, silence is not

`moe_backend` is a **preference, not a pin**. A requested backend that cannot
serve the layer is turned down and a substitute runs — production depends on
this, because an unquantized MTP draft layer in a MIXED_PRECISION checkpoint
must not take down a model whose other layers are NVFP4.

What is not allowed is doing it quietly. Every resolution returns a
`MoEResolutionReport` naming the winner, every rejected candidate, its
`MoERejectReason`, and the environment fingerprint; a degradation additionally
logs a warning once per layer. `report.degraded_from` is the answer to "why did
my `moe_backend` not take effect". It is taken from the requested
implementations' rejections only (the fallback candidates are asked after
them): the highest-priority one that declined for a reason other than the
layer's format or the SM. On SM100 the SM120-only b12x leaves lead the `CUTEDSL`
family, so without the SM exclusion their `SM_UNSUPPORTED` would stand in for
the CuteDSL grouped-GEMM leaf's rejection. The same exclusion means a leaf that
serves the format but not a deployment setting must not answer
`QUANT_UNSUPPORTED`, or the report blames another candidate: the B12x NVFP4
leaf declines `force_dynamic_quantization` as `DYNAMIC_QUANT_UNSUPPORTED`.

Two things fail hard, and they fail in different places. An unknown or retired
backend literal raises `ValueError` before any candidate is considered, so there
is no report to inspect — a misspelled backend is a config error to fix, not
something to route around. Nothing being able to serve the layer is the opposite:
`resolve_moe_impl` still returns a full report, with `winner is None` and every
rejection recorded, and `impl_class_for` is what raises. Catch the first, read
the second.

`NO_FALLBACK_BACKENDS` is the one exception to "degradation is allowed", and
`VANILLA` is its only member. Vanilla exists to produce reference numerics, so a
caller comparing a kernel against it is not helped by getting Cutlass back with a
warning — that silently answers a different question than the one asked. A
`VANILLA` request whose gates reject therefore takes the `winner is None` path and
raises with the whole rejection trail.

The report is diagnostic rather than pinnable: `_legacy_backend_name` fills
`winner`, `eligible` and `rejected[].legacy_backend` from `__name__`, so the
report (including the `to_dict()` artifact a tuning result is replayed from)
records class names. `TritonFusedMoE`, `VanillaMoE` and the two B12x leaves
publish no `MoEImplId`. A class name is not an identity: read
`descriptor.identity`.

### Cross-rank agreement

Every rank resolves independently, so a wheel installed on some nodes and not
others makes ranks pick different impls. The symptom is a hang, not an error:
the ranks allocate differently shaped expert weights and then wait for each
other in a collective that no longer matches.

Selection does not police this itself. `MoEEnvironment.fingerprint()` is recorded
in every `MoEResolutionReport`, so comparing two ranks' reports names the
divergence immediately — but that is a diagnosis after the fact, not a guard.
An automatic check has to be a collective, and selection is the wrong place to
start one: `resolve_moe_impl` runs per MoE layer, so its participants are
"the ranks that happen to build a MoE layer", which under pipeline parallelism
is not every rank (a stage holding only dense layers never calls it). A
collective entered by that set on the world group deadlocks. If such a check is
added later it belongs at an initialization point every rank reaches
unconditionally, not here.

### A layer's format is fixed when it is built

Resolution picks an implementation from the quantization the layer has when
`create_moe` runs, and a leaf *is* one format: its weight method, input
quantization and kernel flags do not change afterwards. So a model whose
experts can end up in another format than the model-level `quant_config`
(MIXED_PRECISION checkpoints, `exclude_modules`) passes the layer's final
quantization to `create_moe` as `override_quant_config`.
`modeling_utils.moe_experts_quant_config(model_config, experts_prefix)` computes
it the way the `__post_init__` passes would (the `quant_config_dict` entry, by
the match `apply_layerwise_quant_config` uses, then the `exclude_modules`
match); every MoE model without a lookup of its own passes it at its
`create_moe` call. The `exclude_modules` patterns must already name TRT-LLM
modules by then, so GptOss translates its checkpoint names before it builds
its layers.

The model's `__post_init__` passes (`apply_layerwise_quant_config`,
`apply_quant_config_exclude_modules`) can still rewrite a layer's
`quant_config` before its weights are allocated. The CUTLASS, TRTLLM-Gen, B12x
and CuteDSL grouped-GEMM leaves check the final value in `create_weights`
through `impl_contract.require_layer_quant_format` and raise if it no longer
matches the format they were resolved for, instead of allocating weights the
checkpoint cannot fill.

## Backend Capability Matrix

### Quantization Support

Each backend's `can_implement(p, d)` classmethod declares what it supports. Source of truth: the `can_implement` classmethod in each backend file.

| Quantization | Cutlass | TRTLLMGen | DeepGemm | DenseGEMM | CuteDSL | MegaMoE-DG | MegaMoE-CuteDSL | Triton | Marlin | Vanilla |
|---|---|---|---|---|---|---|---|---|---|---|
| Unquantized (BF16/FP16) | Y (SM80+) | Y (SM100/103/107, BF16, needs FlashInfer)§ | N | N | Y (SM107, BF16, SwiGLU only)¶ | N | N | Y (SM90, BF16) | N | Y |
| FP8 QDQ | Y (SM89+) | N | N | N | N | N | N | Y (SM90) | N | Y |
| FP8 Block Scales | Y (SM90, SM120) | Y (SM100/103/107) | Y (SM100/103/107) | N | N‡ | N | N | N | N | Y |
| NVFP4 | Y (SM100/103/107/120/121) | Y (SM100/103/107) | N | Y (SM100/103) | Y (SM100/103/107/120/121)¶ | N | Y (SM100/103/107, cu13 cutlass-dsl + symmetric-memory provider; per-expert alpha/norm_const + SwiGLU clamp) | N | Y (SM89-SM99) | Y |
| W4A16 NVFP4 | Y (SM80+, dequant-on-the-fly) | N | N | N | Y (SM120/121, BF16, via `CuteDslB12xW4a16Nvfp4FusedMoE`, needs flashinfer) | N | N | N | Y (SM89-SM99, BF16) | Y |
| W4A8 NVFP4 FP8 | N | Y (SM100/103) | N | N | N | N | N | N | N | N |
| W4A16 MXFP4 | Y (SM90) | Y (SM100/103/107) | N | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 FP8 | Y (SM100/103/107) | Y (SM100/103/107) | N | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 MXFP8 | Y (SM100/103/107/120/121) | Y (SM100/103/107) | N | N | N | Y (SM100/103/107, requires `hidden_size % 512 == 0`) | N | N | N | N |
| W8A8 MXFP8 MXFP8 | Y (SM100/103/107) | N | N | N | N | N | N | N | N | N |
| W4A8 AWQ | Y (SM89/90) | N | N | N | N | N | N | N | N | N |
| W8A16 | Y (SM80+) | N | N | N | N | N | N | N | N | N |
| INT4 WoQ (W4AFP8) | N | N | N | N | N | N | N | N | N | N |

§ The unquantized TRTLLM-Gen path is `FlashinferTrtllmGenBf16Impl`, and it is
not a TRTLLM-Gen kernel at all: it calls FlashInfer's `trtllm_bf16_moe` /
`trtllm_bf16_routed_moe`, which is why it is gated on
`MoEDep.FLASHINFER_BF16_MOE` rather than on a quant algo or on the
`TRTLLM_GEN_FUSED_MOE_USE_FLASHINFER` opt-in, and why `TRTLLMOpBackend` raises
`NotImplementedError` for it. It is the only leaf reachable on a host with no
quantization configured at all. The row reads `Y` because `can_implement`
really can select it; without the FlashInfer symbols the layer degrades to
Cutlass with `DEP_MISSING` recorded in `degraded_from`, where the pre-resolver
code raised `RuntimeError` instead. One shape gate applies:
`Bf16MoeLauncher::check_moe` requires a multiple of 128, and what the wrapper
passes it is `intermediate_size_per_partition`, so a non-aligned shard is
`SHAPE_UNALIGNED` and falls back to Cutlass rather than dying in the kernel
launcher — Qwen3.5-35B BF16 TP8 shards 512 → 64 and takes that path.

¶ On SM107 both CuteDSL grouped-GEMM leaves need `MoEDep.CUTEDSL_RUBIN` (the
installed CuTe DSL exposes the Rubin helpers), and the family carries three
constraints the other SMs do not:

- **Fused finalize is mandatory.** There is no unfused FC2 — NVFP4 has no plain
  grouped GEMM there, and the BF16 op fuses finalize unconditionally. Disabling
  finalize fusion, explicitly or by configuring LoRA, is
  `FINALIZE_FUSION_REQUIRED` rather than a late `NotImplementedError`.
- **Unquantized is SwiGLU-only, without a clamp.**
  `cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin` takes no activation argument
  or constant, so another activation or a clamped SwiGLU is
  `ACTIVATION_UNSUPPORTED`; NVFP4 forwards `activation_type` and the clamp, and
  serves Relu2 and SiTU too.
- **Locality domain excludes EPLB and DWDP.** Localized weight shards are built
  once from the loaded weights, so they cannot follow expert migration
  (`EPLB_UNSUPPORTED`) or parameter rebinding. DWDP is not an error: with
  `uses_locality_domain` true, `_should_enable_dwdp` simply returns False.
  Both locality-domain kernels also fuse SwiGLU, so `plan_moe` declines any other
  activation: an NVFP4 Relu2 layer runs unpartitioned rather than being rejected.

Cutlass covers `W4A16 NVFP4` on a wider SM range than plain `NVFP4` because the
two run different kernels: `W4A16NVFP4CutlassFusedMoEMethod` dequantizes the FP4
weights into the activation dtype each forward and then calls the unquantized
kernel, so it inherits that path's `SM80+` floor instead of needing NVFP4
tensor cores. This is what makes Cutlass the landing spot when a W4A16 NVFP4
layer finds no specialized backend — the B12x W4A16 leaf without flashinfer
or with float16 activations, or any SM outside Marlin's 89-99 and B12x's
120/121.

‡ No CuteDSL leaf publishes `fp8_block_scales`; the DeepGEMM and TRTLLM-Gen
leaves own the format on SM100/103/107 and Cutlass on SM90/SM120/SM121. Because a
`CUTEDSL` request considers only the CuteDSL family plus the Cutlass fallback,
an explicit `moe_backend="CUTEDSL"` on an FP8-block checkpoint at SM100 fails
resolution; request `DEEPGEMM` or `TRTLLM` for that checkpoint instead.

### Activation Support

Activation is **declared, never inspected**: no factory or selection code asks
what class a backend is in order to decide whether it can run a checkpoint's
activation. Three types in `activation.py` carry that instead.

| Layer | Type | Written by |
|-------|------|------------|
| Carrier | `MoEActivation` = `SwigluActivation` \| `SwigluBiasActivation` \| `SiTuActivation` \| `SimpleActivation` | The **model**, passed as `create_moe(activation=...)`; the default `DEFAULT_MOE_ACTIVATION` is plain SwiGLU. One frozen dataclass per kind, so a kind and its constants cannot disagree, and a constant the kind does not have cannot be written down at all |
| Declaration | `MoEActivationSupport` — `kinds`, an `ActivationParamShape` for `alpha_beta` and `limit`, plus `clamp_after_silu` | The **backend**, as an `activation_support` class attribute. Every backend declares one; `resolve_activation_support` raises if a module does not |
| Adapter | `install_activation_params` → `materialize_activation_params` | Not the backend: the `apply_moe_impl_construction_state()` every execution unit already calls installs the slots, and `ConfigurableMoE` re-installs after the EPLB sync and inside `create_weights`. A complete layer that owns kernels directly (`TritonFusedMoE`, `VanillaMoE`) calls it itself — `MoE.__init__` deliberately does not, because a wrapper's declaration is the backend's |

The carrier names constants **semantically**, per kind; only the declaration and
the adapter speak the ABI. `SwigluActivation` has `clamp` and the
`clamp_after_silu` mode (false by default, preserving the historical
pre-activation clamp);
`SwigluBiasActivation` has `gate_sigmoid_scale` / `linear_offset` / `clamp`;
`SiTuActivation` has `gate_softcap` / `linear_softcap` and no clamp at all;
`SimpleActivation(kind)` covers the kinds that take no constants. `constants()`
is the single seam where those become the functor's `alpha` / `beta` / `limit`
registers — and it has to be a seam, because the registers are borrowed for
unrelated jobs: `SwigluBias` reads `alpha` as a scale inside the sigmoid and
`beta` as an additive offset (neutral `0.0`), while `SiTu` reads both as tanh
soft-cap magnitudes the kernel divides by, so they must be positive (neutral
`1.0`). A single nullable `beta` has no coherent default without first finding
the kind, which is exactly the lookup this split removes.

`kinds` is what the kernels **execute**, not what the gate tolerates — several
backends quietly narrow an accepted kind to SwiGLU or SiLU, and those kinds must
stay out of the declaration.

The adapter assigns the three constant slots — `act_alpha`, `act_beta`,
`act_clamp` — plus the scalar `act_clamp_after_silu` mode, and those are the
only names a forward path or a quantization method reads. The
op schemas keep their historical spelling, so a backend passes
`torch.ops.trtllm.fused_moe(swiglu_alpha=self.act_alpha, ...)`; only the Python
plumbing was renamed. The slots' *types* come from the backend's declaration,
never from the checkpoint:

- `PER_EXPERT_TENSOR` → `float32[expert_size_per_partition]`. A scalar is
  broadcast; a caller tensor of the right length is cloned, because
  `quantization.py` divides these buffers in place by the FC31 scale.
- `UNIFORM_SCALAR` → one `float`. A per-expert tensor whose values are not
  *exactly* equal is rejected rather than averaged: the kernel bakes one value,
  so a tolerance here would discard the experts that differ instead of
  approximating them.
- `UNSUPPORTED` → the candidate is declined during resolution
  (`MoERejectReason.ACTIVATION_UNSUPPORTED`) whenever the layer's activation
  fills that register, so a backend that can serve it is still reachable.

`clamp_after_silu=True` is a separate execution capability rather than a
register shape. The resolver declines a backend unless its declaration opts in,
and materialization enforces the same contract. The CUTLASS grouped-GEMM leaves
opt in because their SwiGLU adaptor consumes the mode. The SM120
FP8-block-scale leaf runs a Triton kernel that clamps before the SiLU only and
takes the clamp by value, so it declares `kinds={Swiglu, Geglu}` with
`limit=UNIFORM_SCALAR` and nothing else.

With `USING_OSS_CUTLASS_MOE_GEMM=OFF`, the Torch runner uses the bundled
internal kernel archive's legacy interfaces. That build supports pre-SiLU
SwiGLU clamping and rejects post-SiLU clamping, SiTu, Relu2, routed-expert LoRA,
MXFP8 activations or weights, NVFP4 weight-only MoE, FP8-input W4A8,
dynamic FC2 scaling, per-expert W4A8 activation scales,
all-to-all, unpadded output, a separate valid-token count, and unswizzled
input scales. These features require the open-source kernels. The shared
MoE kernel tests use the public quantization utilities with either runner;
post-SiLU, unpadded output, and MXFP8 x MXFP4 cases skip for the internal runner.
The FP8-input W4A8 fixture and all-to-all variants require the OSS build. The Torch
boundary translates public activation IDs to the internal enum values.
When SwiGLU parameters are present, it supplies missing alpha/beta defaults
as tensors so the legacy archive consumes the clamp limit.
Public routing helpers remain available in either build.
The build mode is collected once into `MoEEnvironment.oss_cutlass_moe` and
included in its fingerprint. Libraries without the build-mode query preserve
the OSS default. One family gate, `cutlass/eligibility.check_oss_cutlass_build`,
runs after the support gates in every CUTLASS leaf that issues
`torch.ops.trtllm.fused_moe` (`issues_fused_moe_op`; not the SM120 Triton leaf)
and rejects SiTu and Relu2,
the formats whose leaf sets `requires_oss_cutlass_build` (`MXFP8`,
`W4A8_MXFP4_MXFP8`, `W4A16_MXFP4`, `W4A8_AWQ`), dynamic FC2 scaling on the leaf
that sets `uses_dynamic_fc2_scale` (`NVFP4`), post-SiLU clamping, and
routed-expert LoRA. The B12x NVFP4 leaf runs the same gate for the prefill
chunks it routes through CUTLASS; it declines dynamic quantization before
that gate on any build, because its decode route cannot honour it. W4A8 AWQ
uses per-expert activation-scale buffers and is unavailable in the legacy build. `CutlassFusedMoEBase` exposes
`supports_alltoall` before communication selection; it is False only for a
leaf that issues the op on a legacy build. Unsupported auto-selection
falls back to AllGather/ReduceScatter, and
explicit all-to-all requests fail before allocation. Construction validation
retains an all-to-all backstop. Input layout and token-count options supplied
to native operators directly retain runtime guards.
The neutral SwiGLU tensors are cached by device, expert count and CUDA stream;
warm eager forwards reuse them without allocations or fill kernels. During CUDA
graph capture, each forward creates its own defaults and records their fills in
that graph instead of retaining graph-pool tensors in the runner cache.

The CPU resolution tests run in `l0_cpu.yml`; routing and activation numerics
run in the H100 and B200 single-GPU lists. The H100 C++ kernel stage discovers
both OSS and bundled-internal MoE variants through CTest. With OSS kernels
disabled, `mixtureOfExpertsTest` exercises the bundled runner directly.

Declare `limit_when_absent` only when the ABI has no encoding for "no clamp":
`TrtllmCutedslGroupedGemmNvfp4Impl` passes `float("inf")` because its epilogue
always applies the clamp functor. It is substituted *before* coercion, so a
`PER_EXPERT_TENSOR` backend gets it broadcast like any other scalar, and
`MoEActivationSupport` rejects it outright next to `limit=UNSUPPORTED`.

TRTLLM-Gen is the one exception, and it is a per-class one: its clamp is a
per-expert tensor for the FP4 fused-activation cubins but a by-value `float`
for the FP8 block-scale kernel (`DevKernel.h`, `float swigluLimit`), so
`TRTLLMGenFp8BlockScalesBase` overrides `resolve_activation_support` to narrow
the shape while the rest of the family falls through to the class attribute.
`resolve_activation_support` is probed with `getattr` for exactly that reason.
`ConfigurableMoE.create_weights` installs the slots ahead of each allocation,
guarded so a pass that is not allocating does not re-run the in-place division
`NVFP4TRTLLMGenFusedMoEBaseMethod` applies to `beta` and `clamp`.

Selection keeps both halves of the picture in the tuning key —
`MoEProblem.activation` (the kind), `MoEProblem.activation_constants` (which
registers the carrier actually fills), and `MoEProblem.clamp_after_silu` (the
clamp order) — because the kind alone does not separate a clamped layer from an
unclamped one or distinguish the two clamp orders, and those compile to
different kernels.

**gpt-oss / MiniMax SwiGLU** is a separate axis: a backend can execute the kind
and still reject the format it arrives with. The gpt-oss package (per-expert
bias plus filled `alpha` / `beta` / `limit`, summarized as
`MoEProblem.swiglu_gptoss_style`) is declined by every specialized backend whose
`kinds` omit `SwigluBias` (the CuteDSL grouped-GEMM leaves, the B12x leaves,
`DeepGemmFusedMoE`, `TrtllmCutedslDenseGemmNvfp4Impl`, the Marlin leaves). The
TRTLLM-Gen and CUTLASS leaves admit it per format through
`supports_gptoss_style` (on CUTLASS: unquantized, MXFP8, NVFP4 and the three
MXFP4 formats). The kernel takes the three constants for every format; what
limits the set is the weight methods. W8A16 and W4A8_AWQ keep a bias shape that
is wrong for their transposed layouts, and NVFP4 sets
`rejects_gptoss_expert_bias` because its weight pad assumes 2-D, so MiniMax-M3
(no expert bias) is in and a 1-D gpt-oss bias is out. Each way of getting the
set wrong has a test that catches it:

- `ModelConfig.get_mxfp4_quant_algo` maps gpt-oss to `W4A16_MXFP4` below SM100
  and to the `W4A8_MXFP4_*` pair at SM100+, so a gate on `W4A8_MXFP4_MXFP8`
  alone leaves gpt-oss unservable on Hopper (Cutlass is the fallback and every
  other backend abstains).
- Dropping the gate un-skips the `test_configurable_moe_single_gpu` gpt-oss ×
  CUTLASS matrix, which then fails inside weight loading.
- Leaving out the unquantized leaf rejects `test_gpt_oss_trtllmgen[CUTLASS]`.
- Excluding NVFP4 as "gpt-oss bias" rejects `TestMiniMaxM3::test_nvfp4`.

### Scheduler / EPLB Constraints

- `FUSED_COMM` backends (`DeepgemmCudaW4a8Mxfp4Mxfp8Impl`, `TrtllmCutedslMegaMoeNvfp4Impl`) **must not** layer host-side `Communication.dispatch` / `.combine` on top of the fused kernel — `ConfigurableMoE._create_comm_strategy_auto` returns `None` for them.
- `FusedCommMoEScheduler` calls `backend.quantize_input(...)` for every chunk including zero-token chunks (so peer ranks can cross the in-kernel NVLink barrier). Each fused-comm backend therefore MUST make `quantize_input` tolerate `x.shape[0] == 0` and return its own empty tensor layout; the scheduler does NOT synthesize backend-specific empty tensors.
- Dynamic EPLB requires backend and quantization-method support; backends gate wrapper-level constraints in `validate_configurable_moe`. `MegaMoEDeepGemm` routes to slot IDs and migrates the transformed DG weights its quantization method registers (`num_slots % ep_size == 0`). `TrtllmCutedslMegaMoeNvfp4Impl` declares `supports_eplb=True`; its quantization method registers the MegaMoE-format derived params and `fc1_norm_const` with the load balancer alongside the raw NVFP4 family, so per-slot migration stays byte-consistent.
- `FUSED_COMM` backends use `ignore_allreduce=False` for EPLB statistic update because the fused kernel AllReduces routing stats internally.
- Per-layer dynamic EPLB uses `MoeConfig.load_balancer.mode=per_layer_dynamic` with
  `num_slots` greater than the model expert count. Backend resolution selects
  `MEGAMOE_CUTEDSL` when no concrete backend is requested and rejects an
  incompatible explicit backend before model construction.
- Per-layer dynamic EPLB requires `CUDA_SCALE_LAUNCH_QUEUES=4x` before CUDA context
  initialization. TensorRT-LLM applies the setting before `TorchLlmArgs`
  validation and CUDA initialization/device probes, and propagates it to MPI,
  direct, and Ray workers. Applications that initialize CUDA before TensorRT-LLM
  must set it at process startup.
- HALO-Q and the in-switch TMA copy run as one kernel on the higher-priority
  copy stream, one CTA per auxiliary SM. Each CTA raises a device route epoch
  once its routes are written. The main stream may enqueue shared-expert work
  and input quantization before its first routed-expert use, where a stream
  memory wait on that epoch orders it after the routes. Each layer binds one
  main stream, the executor's execution stream, on which warmup and serving
  submit from their own threads; the path does not require a CUDA green
  context or a steady-state host wait.
- HALO-Q waits for every EP rank inside the kernel and traps once a peer is
  later than its spin bound. Each launch passes the bound, which
  `SchedulerSpinBound` takes from the MoE communication timeout guard: the
  warmup timeout during warmup and AutoTuner profiling, the native bound
  otherwise.
- `helper_slot_mode=per_layer_resident` gives each layer its own helper-slot
  set, and the copy keeps a per-layer table of the expert each helper slot
  holds so that it skips experts a whole team already holds. Any other writer
  of those helper rows would invalidate the table.
- HALO-Q and TMA follow the native-operator path under
  `cpp/tensorrt_llm/kernels/moe/loadBalance/dynamicEplb` and
  `cpp/tensorrt_llm/thop/moe/loadBalance`. The scheduler accepts the original
  contiguous CUDA `int32` route tensor; do not restore route staging, tail
  fills, or a host-generated copy plan.
- Rebalance resources are model-scoped. Worker shutdown and construction
  rollback must release the scheduler, TMA bindings, shared-slot mappings,
  and fabric allocations collectively. Forward execution must not retain
  activation tensors after their consumer has been enqueued.
- Per-layer dynamic EPLB lives in `per_layer_dynamic_eplb/` (`manager`,
  `layer`, `arena`, `planner`) and reuses the EPLB seams.
  `maybe_create_moe_load_balancer` builds `PerLayerDynamicMoeLoadBalancer`,
  which shares the `BaseMoeLoadBalancer` lifecycle with `MoeLoadBalancer`, and
  `add_layer` returns a `PerLayerDynamicMoeLayerLoadBalancer` with
  `produces_replica_plan = True`. In every chunk `_load_balancer_route` plans,
  the backend reads `MoERunContext.replica_plan`, and the post-`run_moe` hook
  releases the plan. The planner consumes logical routes, so the expert
  statistics and the layer-wise calibrator record (or replay) them before the
  route; standard EPLB records the slots after it. When the forward raises,
  `MoeLoadBalancerIterContext` calls `abort_iter`, which releases the pending
  plans. Plans run during warmup too, so there is no warmup switch; only the
  HALO-Q spin bound changes, as described above. Model code passes independent
  main-stream work such as shared experts through
  `MoE.load_balancer_overlap(fn)`; the route runs it once, after submitting the
  copy stream.
- The quant method owns the seven live kernel planes. Its `create_weights`
  allocates them from its plane spec through the layer handle's
  `allocate_arena` and binds them; its `cache_derived_state` checks them and
  calls the handle's `bind_backend`, which builds the HALO-Q/TMA producer. The
  backend only declares `supports_per_layer_dynamic_eplb`, uses
  `compute_slot_count` (H + S, with S from the layer handle) for the planes,
  symmetric workspace, and launch geometry while `expert_size_per_partition`
  stays H, and consumes the replica plan, including the SMs its input
  quantization leaves to the copy kernel. `bind_backend` reads the backend's
  `ep_process_group` and `max_tokens_per_rank_capacity`.
- DeepSeek-V4 selects the fused shared FC12 kernel when the routed backend
  declares per-layer dynamic support, the final projection quantization and
  hardware (SM100 family) support that kernel, and the model runs eagerly
  (`cuda_graph_config: null`). Per-layer dynamic EPLB is not a condition, but it
  requires eager mode, so an EPLB-off baseline runs the same shared kernel only
  with `cuda_graph_config: null`. Other runs keep the model's existing shared-MLP
  backend. With per-layer dynamic EPLB, the active configuration supplies the
  fused kernel an SM budget that leaves the copy kernel its auxiliary SMs.

## Canonical Examples

When adding new components, use these reference implementations:

| Task | Reference | Key methods to implement |
|------|-----------|--------------------------|
| New `EXTERNAL_COMM` Backend | One format: `fused_moe_deepgemm.py` (`DeepgemmCudaFp8BlockScalesImpl`). Several formats: `cutlass/` (`CutlassFusedMoEBase` and a leaf such as `cutlass/nvfp4.py`) | Declare `MoEImplBase`; implement `capabilities`, `activation_support`, `can_implement`, `_get_quant_method`, `quantize_input`, `run_moe`; call `apply_moe_impl_construction_state()` in `__init__` (`create_weights` / `load_weights` come from `MoEWeightOwnerMixin` — override only if allocation needs more); then add the class to `moe_resolution.IMPL_PRIORITY` and `BACKEND_FAMILY`, and add a branch in `create_moe_backend`. Declare the `descriptor` and `@register_moe_impl` on the class that runs (for a family, the leaf). Split an abstract parent out of it only when several quantization formats are planned — see the one-class-per-identity note below |
| New `FUSED_COMM` Backend | `mega_moe/mega_moe_deepgemm.py` (`DeepgemmCudaW4a8Mxfp4Mxfp8Impl`), `mega_moe/mega_moe_cute_dsl.py` (`TrtllmCutedslMegaMoeNvfp4Impl`) | Same as above + override `scheduler_kind = MoESchedulerKind.FUSED_COMM` and `validate_configurable_moe` for backend-specific constraints. For NVFP4 CuteDSL specifically, mirror the `TrtllmCutedslMegaMoeNvfp4Impl` pattern: capability probe for the CUDA 13 Cutlass DSL runtime, JSON-friendly tactic dict, lazy kernel imports from `moe/kernels/cutedsl_megamoe/`, and `quantize_input` that short-circuits zero-token input. |
| New Quantization Method | `quantization.py` → `FP8QDQFusedMoEMethod` | Subclass `FusedMoEMethod`, implement quant/dequant ops |
| New Communication Strategy | `communication/nvlink_one_sided.py` (`NVLinkOneSided`) | Subclass `Communication`, implement `prepare_dispatch`, `dispatch`, `combine`; also implement `CheckpointableCommunication` when native workspace mappings must follow executor sleep/wakeup |
| New Scheduler | `moe_scheduler.py` (`ExternalCommMoEScheduler` / `FusedCommMoEScheduler`) | Subclass `MoEScheduler`, implement `forward`; add new `MoESchedulerKind` value and wire into `create_moe_scheduler` factory |
| Backend Tests | `test_moe_backend.py` | Follow existing parametrize patterns |
| Identity Tests | `test_moe_impl.py` | Test what selection can get wrong: the backend literal or a pinned id reaches the intended leaf, a leaf declines what its kernel does not run, a pin fails rather than degrades, and a leaf refuses a layer whose format moved after it was picked. Do not restate a declaration (an id string, a support set, a family's membership) across tests; a family with several leaves pins its leaves' declarations once, in one hand-written table (`test_cutlass_leaf_declarations`, `test_cutedsl_leaf_declarations`). Tests that merely *use* the class belong in `test_moe_backend.py` |
| Integration Tests | `test_moe_module.py` | Test Backend × Communication × EPLB combinations |

**Note on backend inheritance:** new execution units inherit `MoEImplBase` (in
`impl_base.py`), never `MoE` and never another implementation for its
constructor. `MoE` is the complete-layer type (`ConfigurableMoE`,
`TritonFusedMoE`). Implementations have no `forward`, so they only run as
`ConfigurableMoE.backend`; anything that needs a callable layer wraps one
(`Llama4MinLatencyFusedMoE` extends `ConfigurableMoE` and picks its leaf with
`cutlass_leaf`). A family's leaves inherit `MoEImplBase` through its
abstract root (`CutlassFusedMoEBase`, `TrtllmGenFusedMoEBase`,
`MarlinFusedMoEBase`, `CuteDslFusedMoEBase`).

The B12x pair is the one exception. `CuteDslB12xNvfp4FusedMoE`
inherits the CUTLASS NVFP4 leaf because its prefill chunk *is* that grouped
GEMM, reached through `super()`. `CuteDslB12xW4a16Nvfp4FusedMoE` takes only
`CutlassFusedMoEBase`, for construction and the CUTLASS branch of
`create_moe_backend`, because W4A16_NVFP4 never leaves the b12x kernel. Their
shared mixin countermands what the parents hand over: it sets
`descriptor = None` so the NVFP4 leaf does not publish its parent's identity,
checks the layer's format against its own `b12x_quant_algo`, and declines EPLB
in its gate although the inherited `_supports_load_balancer()` says True. That
gate replaces the parent's, so it restates the one parent check both leaves
depend on, because both consume the FC1 buffer the CUTLASS NVFP4 weight pipeline
padded: the FC1 row alignment (`nvfp4_fc1_row_alignment_rejection`).

**Note on one class per identity:** an identity and the code it names belong on
the same class. That class declares the `descriptor`, takes `scheduler_kind` /
`capabilities` / `input_requirement` off it, carries `@register_moe_impl`, and
is concrete: the four abstract methods of `MoEImplBase` (`can_implement`,
`_get_quant_method`, `quantize_input`, `run_moe`) are implemented on it, or on
its family parent where formats share them. `DeepgemmCudaFp8BlockScalesImpl`,
`TrtllmCutedslMegaMoeNvfp4Impl` and `TrtllmCutedslDenseGemmNvfp4Impl` implement
all four themselves. One identity may still
fan out to several kernels as long as the four id segments stay the same:
`TrtllmCutedslMegaMoeNvfp4Impl` picks a per-architecture kernel inside one
class, because architecture is not an id segment.

Split an abstract parent out only once several quantization formats share an
implementation. The parent holds what they share, leaves at least
`can_implement` to the leaves, stays abstract, and carries no `descriptor`,
because a descriptor is exactly one identity. How much else it implements
follows how much the formats share: `TrtllmGenFusedMoEBase` implements none of
the four (its formats differ in kernel ABI, so each leaf gets its methods
through a per-ABI layer), `CuteDslFusedMoEBase` none either (its two formats
share construction, the locality-domain plan and the weight lifecycle, but run
different ops), `MarlinFusedMoEBase` three (one ABI serves both formats), and
`CutlassFusedMoEBase` defaults for `quantize_input` and `run_moe` that most
leaves keep.

Such a parent must not branch on the axis its subclasses divide: an
`if self.<quant or provider flag>:` is a subclass difference expressed as data,
and belongs in an override or a leaf-declared attribute. `TrtllmGenFusedMoEBase`
keeps only flags read off the class at resolution (`supports_gptoss_style`,
`supports_situ`) and turns the rest into hooks; `CutlassFusedMoEBase` takes its
per-format kernel configuration as `FUSED_MOE_OP_FLAGS`, its padding as
`pad_hidden_and_intermediate_to`, and its build requirements as
`requires_oss_cutlass_build` / `uses_dynamic_fc2_scale`; `CuteDslFusedMoEBase`
reaches what only one format needs through the `_init_format_state` and
`_locality_domain_scale_shard` hooks and the leaf-declared
`_locality_domain_full_scale_params`.

**Note on implementation class names:** the name is derived from the identity
as `<Provider><Technique><Quant>Impl`. The kernel segment enters when two ids
differ in that segment alone: the class that would otherwise collide spells it
out between technique and quant, in id order
(`DeepgemmCudaHopperFp8BlockScalesImpl` next to
`DeepgemmCudaFp8BlockScalesImpl`). The `trtllm.cutedsl.*.nvfp4` ids
(`mega_moe`, `fused_fc12`, `dense_gemm`, `grouped_gemm`) differ in
`kernel_name` alone, so their classes spell the kernel out;
`TrtllmCutedslGroupedGemmBf16Impl` keeps it to match its NVFP4 sibling. Quant
`none` is `Bf16` on a BF16-only kernel (`FlashinferTrtllmGenBf16Impl`,
`TrtllmCutedslGroupedGemmBf16Impl`) and `Unquantized` on
`TrtllmCutlassUnquantizedImpl`, which also takes FP16. A bare `cutedsl` token
matches several leaves; name the kernel segment to pin one. A partial pin can
also span techniques: `grouped_gemm.nvfp4` matches the CUTLASS and the CuteDSL
grouped GEMM and resolves to the CuteDSL leaf by `IMPL_PRIORITY`; name the
technique to pin either.

**Note on family names:** the `XXFusedMoE` names are module-level aliases.
`CutlassFusedMoE`, `TRTLLMGenFusedMoE` and `MarlinFusedMoE` name a family's
abstract root, the parent of every leaf (the B12x leaves also descend from
`CutlassFusedMoE`); `CuteDslFusedMoE` names the root of the two grouped-GEMM
leaves only, not of the `CUTEDSL` family, which also holds the B12x leaves;
`CuteDslB12xFusedMoE` names the B12x mixin; `DeepGemmFusedMoE`,
`DenseGEMMFusedMoE`, `CuteDslFc12FusedMoE`, `MegaMoEDeepGemm` and
`MegaMoECuteDsl` name single leaves. A family name publishes no descriptor,
and every gate against it must be `issubclass` / `isinstance`: resolution hands
over a leaf, so `type(x) is TRTLLMGenFusedMoE` matches nothing. Classes without
an identity (`TritonFusedMoE`, `VanillaMoE`) carry the `XXFusedMoE` form as
their real class name.

## Anti-Patterns

- **Do NOT add communication logic inside backends** — Communication belongs in `communication/`, backends do pure computation (exception: `FUSED_COMM` backends own the SymmBuffer collective inside their fused kernel)
- **Do NOT add forward-execution policy inside backends** — chunking, EPLB hook ordering, dispatch/combine sequencing belong in `MoEScheduler`
- **Do NOT modify old `XXFusedMoE` files for new features** — Use ConfigurableMoE + Backend + Scheduler architecture
- **Do NOT add new tests to `test_fused_moe.py` or `test_moe.py`** — Use `test_moe_backend.py` and `test_moe_module.py`
- **Do NOT skip `can_implement()` checks** — Every backend must declare what it supports; an unsupported combination returns `MoEEligibility.no(MoERejectReason.<CODE>, detail)`, never a bare `False` and never a free-form string a test would have to pattern-match
- **Do NOT probe the machine inside `can_implement()`** — No `get_sm_version()`, no `import` as a presence test, no `os.environ`. Read `d.env`; add the probe to `impl_environment.py` if it does not exist yet
- **Do NOT gate an activation on a class name** — No `isinstance`, no `moe_cls in [...]`, no ad-hoc `swiglu_gptoss_style` branch in a factory. A backend states what its kernels execute in `activation_support`; `_reject_unsupported_activation` reads that declaration for every candidate, and `materialize_activation_params` enforces it again at construction
- **Do NOT add a second selection entry point** — `resolve_moe_impl` is the only one. A helper that picks a class on the side is how `get_moe_cls` and the old `resolve_moe_cls` drifted apart in the first place
- **Do NOT substitute a backend without recording it** — A degradation must be visible in the `MoEResolutionReport`, not only in a log line
- **Do NOT pick `scheduler_kind` opportunistically** — Use `EXTERNAL_COMM` (default) unless your backend's fused kernel genuinely owns cross-rank exchange via SymmBuffer / equivalent in-kernel collective; `FUSED_COMM` brings hard invariants (no host comm, lockstep launches, no multi-stream overlap)
- **Schedulers MUST NOT write `moe.repeat_idx`** — `repeat_idx` is wrapper state advanced once per `forward_impl` regardless of chunk count
- **Do NOT set a communication kernel's timeout directly or through environment variables at runtime** — A backend whose kernels wait for peers under a timeout registers a `MoECommTimeoutProxy` with the timeout guard when it is constructed and unregisters it before releasing what the proxy controls; only the guard decides the value (see [Communication timeouts](#communication-timeouts)). Writing environment variables at runtime races with NCCL threads that read them
- **Do NOT allocate symmetric memory from `run_moe` in `FUSED_COMM` backends** — Symmetric-memory rendezvous is a build-time collective and is unsafe under PP / layer-skip or CUDA graph capture; allocate from `create_weights()` after `ConfigurableMoE` has synchronized EPLB-derived attributes. See `mega_moe/mega_moe_deepgemm.py` for the DG pattern and `mega_moe/mega_moe_cute_dsl.py:_alloc_symm_provider` for the NVSHMEM-equivalent provider.
- **Do NOT add a new `FUSED_COMM` backend without a zero-token `quantize_input` regression test** — `FusedCommMoEScheduler` calls `quantize_input` for every chunk (including zero-token chunks) so each backend must return its own empty-tensor layout.
- **Do NOT use a dataclass for an autotuner tactic without a tested `__repr__` round-trip** — `AutoTuner` serializes tactic values through `json.dumps`/`json.loads` and `eval(repr(tactic))`; a plain dataclass fails the `eval(repr(...))` check. Prefer a JSON-friendly **tuple of primitives or lists of primitives** (lists are JSON-friendly; tuples round-trip via `eval(repr(...))`). See the tactic-representation comment block in `tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py` for the canonical 11-field tactic pattern (with legacy 10- and 8-field compatibility), including token-back READY granularity; `_unpack_tactic` is the single source of truth for field order and compatibility defaults. The fallback tactic is the token-aware `default_megamoe_tactic(num_tokens)` helper, selected by `Sm100MegaMoENvfp4Runner.forward(tactic=-1)`, not a separate `fallback_tactic()` method.
- **Use `distributed_tuning_strategy=DistributedTuningStrategy.MERGE` on a multi-rank `FUSED_COMM` backend's `TuningConfig`** — Every EP rank must converge on the same compiled tactic for every chunk, otherwise the in-kernel NVLink dispatch barrier deadlocks. `PARALLEL` can profile different tactics on different ranks and is unsafe for fused collectives. Reference: `Sm100MegaMoENvfp4Runner.get_tuning_config`.

### CUTLASS eager scratch reclamation

The scheduled decoder engine supplies a `CutlassWorkspaceReclaimer` scope around
one complete model forward, including warmup. Each decoder runner keeps one
reclaimer for its lifetime, including repeated warmups. Dummy forwards, including
KV-capacity profiling, enter the scope as warmup and contribute to its capacity floor.
The Python scope enters and exits
a native `EagerMoeWorkspaceScope`; inference calls discover runners directly in
C++, after autotuning, without per-layer Python registration or stream queries.
The native scope retains each runner and groups calls by runner/device/stream.
Cleanup restores each allocation stream, including after model or cleanup errors.
Scopes are thread-local and reject overlapping or cross-thread use. If warmup
finds no runners, subsequent serving forwards bypass native scopes; later warmup
can still discover runners. The native
`EagerMoeWorkspace` accumulates exact workspace requirements across layers and
chunks and replaces excess backing after three successful forwards using at most half
the retained capacity. The replacement retains the larger of the window demand and the capacity
established by successful warmup. This warmup floor is never reduced by serving
forwards; at the floor, the shrink-window checks are bypassed.
The engine also supplies its host token count. Known runners on the current
stream begin before the model executes, reserving a previously observed demand
before embedding/attention intermediates can split a released burst-sized block.
This reservation is opportunistic: exact kernel sizes still determine the demand
window, and a failed reservation does not reject a smaller actual allocation. Routing pointers are rebound by
`getWorkspaceInfo` on the next call. A new runner/stream without warmup does not
reclaim; capture, another engine owner or unscoped use disables reclamation for
that workspace. Graph storage and LoRA-specific persistent scratch are excluded.
`TRTLLM_RECLAIM_WORKSPACE=0`, set before engine initialization, disables both
attention and CUTLASS MoE reclamation scopes. All cleanup runs on each
allocation's original CUDA stream; no device synchronization or payload copy is
needed for pure scratch. The replacement is allocated before releasing the old
block so a small allocation does not split and pin the burst-sized segment;
on allocation OOM, release the old scratch and retry without requiring extra
headroom. Do not count individual layer/chunk calls as forwards.
