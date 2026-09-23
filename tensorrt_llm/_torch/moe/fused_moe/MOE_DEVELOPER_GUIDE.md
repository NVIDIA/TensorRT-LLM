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
        self.dwdp_manager.record_compute_and_prefetch_next(self.layer_idx)
    self.repeat_idx = (self.repeat_idx + 1) % self.repeat_count
    return outputs
```

### Scheduler Selection (`MoESchedulerKind`)

Each backend declares one of two scheduler kinds via the `scheduler_kind` class attribute (defined on `MoE` base, default `EXTERNAL_COMM`):

| Kind | Scheduler class | Used by | Cross-rank EP exchange |
|------|-----------------|---------|------------------------|
| `EXTERNAL_COMM` | `ExternalCommMoEScheduler` | Cutlass, DeepGemm, CuteDSL, DenseGEMM, TRTLLMGen | Host issues `Communication.dispatch` / `.combine` outside the MoE kernel; supports per-chunk EPLB hooks and multi-stream chunk overlap |
| `FUSED_COMM` | `FusedCommMoEScheduler` | DeepgemmCudaW4a8Mxfp4Mxfp8Impl, TrtllmCutedslMegaMoeNvfp4Impl | Comm is fused into the backend kernel via SymmBuffer / NVSHMEM-equivalent peer-pointer mapping; no host comm; lockstep chunk launches; EPLB stats AllReduced internally |

The two paths have *deliberately opposite* invariants (`use_dp_padding` honored vs ignored, ADP padding kept vs stripped, empty-chunk substituted vs zero-token kernel launch, multi-stream overlap allowed vs forbidden). See `moe_scheduler.py` class docstrings and `MOE_SCHEDULER_DESIGN.md` for the full contract.

### DWDP execution boundary

DWDP is wrapper-owned lifecycle, not a scheduler or backend variant:

```text
Executor construction
  → register eligible MoE layer indices
  → after weight loading: create VMM transport + composite VA + page pool

PyExecutor forward start
  → DwdpManager.prefetch_first_layers() for the first two MoE layers
  → ConfigurableMoE.wait_and_bind(composite VA)
  → MoEScheduler.forward(... ordinary CuTeDSL kernel ...)
  → main stream joins scheduler auxiliary work
  → DwdpManager records current-slot consumption
  → prefetch the MoE layer two positions ahead

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
  → backend.run_moe → [EPLB start_set_cpu] → comm.combine → [EPLB done_set_cpu]

Adaptive quantize/dispatch order (gated by comm.supports_post_quant_dispatch()):
  Post-quant flow: quantize_input() → comm.dispatch()   (send quantized data)
  Pre-quant flow:  comm.dispatch() → quantize_input()   (send raw, quantize locally)
```

EPLB hooks fire only at the first/last chunk of the first/last `repeat_idx`. Multi-stream chunk overlap is enabled when `not enable_alltoall and aux_stream is not None`.

### Fused-comm execution flow (MegaMoE-style)

`FusedCommMoEScheduler._forward_chunk` runs per chunk:

```text
[EPLB start_wait_gpu] → routing → [EPLB done_wait_gpu + update_statistic + route]
  → backend.quantize_input → backend.run_moe (fused dispatch+GEMM+act+GEMM+combine)
  → [EPLB start_set_cpu + done_set_cpu]
```

No external `Communication.dispatch` / `.combine`. Zero-token chunks still launch the kernel so peer EP ranks can cross the in-kernel NVLink barrier.

The DeepGEMM kernel's in-kernel barrier traps after `DG_BARRIER_TIMEOUT_SECONDS` (60 s by default), and first-touch work in warmup can leave one rank that far behind its peers. `DeepgemmCudaW4a8Mxfp4Mxfp8Impl.run_moe` therefore waits on a host-side EP barrier before each launch while the model engine is in warmup (`set_warmup_launch_fence`, switched by the engine's `is_warmup` setter); serving launches are not fenced.

### Core Design Principles

1. **Composition over inheritance** — Backend, Communication, EPLB, and Scheduler are independent, composable components
2. **Any Backend × Any Communication × EPLB On/Off** — All valid combinations should work (subject to `can_implement` and `scheduler_kind`)
3. **Backend = pure computation** — No communication logic, no EPLB logic inside backends
4. **Communication is pluggable** — `EXTERNAL_COMM` backends pick a strategy via `CommunicationFactory` based on hardware/workload; `FUSED_COMM` backends bypass external comm entirely
5. **Backend declares capabilities** — `can_implement(p, d)` is the single source of truth for what a backend supports, and it is a **pure function** of its two arguments (see [Backend Selection](#backend-selection)); the resolver holds no capability knowledge of its own
6. **Backend declares scheduler** — `scheduler_kind` class attribute selects the forward path; lifecycle code stays generic, forward path stays specialized

## Architecture Transition (IMPORTANT)

The codebase is transitioning between two architectures:

| | Old Path | New Path |
|---|---|---|
| Entry | `XXFusedMoE` (e.g., `CutlassFusedMoE`) | `ConfigurableMoE` + `XXBackend` + `MoEScheduler` |
| Communication | Embedded inside each backend | Separated into `communication/` (or fused into kernel for `FUSED_COMM`) |
| Forward execution | Inline in backend | `MoEScheduler` (`moe_scheduler.py`) |
| EPLB | Not supported | Available on EPLB-capable backends |
| Status | Being replaced | Active development |

ConfigurableMoE currently supports these backends (`create_moe.py`):
- `CutlassFusedMoE`, the `TrtllmGenFusedMoEBase` leaves,
  `DeepgemmCudaFp8BlockScalesImpl`, `CuteDslFusedMoE`, `CuteDslB12xFusedMoE`,
  `TrtllmCutedslFusedFc12Nvfp4Impl`, `TrtllmCutedslDenseGemmNvfp4Impl`,
  `DeepgemmCudaW4a8Mxfp4Mxfp8Impl`, `TrtllmCutedslMegaMoeNvfp4Impl`, the
  `MarlinFusedMoEBase` leaves

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
| `moe_load_balancer.py` | EPLB implementation |
| `moe_op_backend.py` | Op backend registry for TRTLLMGen (flashinfer/trtllm ops) |

### Backends (`fused_moe/`)

| File | Backend | Hardware | Scenario | Scheduler |
|------|---------|----------|----------|-----------|
| `fused_moe_cutlass.py` | `CutlassFusedMoE` | SM80+ | High throughput, most comprehensive quant support | `EXTERNAL_COMM` |
| `trtllm_gen/` (+ the `fused_moe_trtllm_gen.py` alias module) | the abstract `TrtllmGenFusedMoEBase` and its leaves (see [TRTLLM-Gen leaves](#trtllm-gen-leaves-fused_moetrtllm_gen)) | SM100/SM103 | Min-latency and high-throughput on Blackwell; also serves unquantized BF16 through FlashInfer, on a gate of its own (`§` under [Quantization Support](#quantization-support)) | `EXTERNAL_COMM` |
| `fused_moe_deepgemm.py` | `DeepgemmCudaFp8BlockScalesImpl` (aliased as `DeepGemmFusedMoE`) | SM100/SM103/SM107 | FP8 Block Scales on Blackwell/Rubin | `EXTERNAL_COMM` |
| `fused_moe_densegemm.py` | `TrtllmCutedslDenseGemmNvfp4Impl` (aliased as `DenseGEMMFusedMoE`) | SM100/SM103 | NVFP4 min-latency; CuTe DSL dense GEMM packs all experts into one matrix (vs Cutlass per-expert scatter), efficient for small token counts | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl.py` | `CuteDslFusedMoE` | SM100/SM103 | High throughput NVFP4, generally faster than Cutlass | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl_b12x.py` | `CuteDslB12xFusedMoE` | SM120/SM121 | NVFP4 hybrid CUTLASS-prefill / FlashInfer NVFP4 MoE decode — best perf on RTX PRO 6000 (SM120) and DGX Spark (SM121); select via the `CUTEDSL` backend path (it heads that family's candidate list, so it wins on SM120/121 when flashinfer is present and yields to `CuteDslFusedMoE` otherwise); single-GPU-shaped topology only — it rejects both `ep_size > 1` and attention-DP, because it has no dispatch/combine kernel and has never been exercised behind a DP allgather | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl_fc12.py` | `TrtllmCutedslFusedFc12Nvfp4Impl` (`trtllm.cutedsl.fused_fc12.nvfp4`, aliased as `CuteDslFc12FusedMoE`) | SM107 (Rubin) | NVFP4 fused FC1+FC2: a `MoEImplBase` leaf sharing `CuteDslFusedMoE`'s NVFP4 weight layout and outer autotune runner, driving the single persistent `cute_dsl_nvfp4_fc12_fused_rubin` kernel (keeps the FC1->FC2 intermediate on-chip, removing the global round-trip and one launch); reachable through the `CUTEDSL_FC12` backend family or a pinned `impl_id`; requires CuTe DSL Rubin support (`MoEDep.CUTEDSL_RUBIN` in the collected environment); routing tile 128 (1-CTA) or 256 (2-CTA, cluster (2,1)), chosen by the autotuner per shape; non-uGPU | `EXTERNAL_COMM` |
| `mega_moe/mega_moe_deepgemm.py` | `DeepgemmCudaW4a8Mxfp4Mxfp8Impl` (aliased as `MegaMoEDeepGemm`) | SM100/SM103 | W4A8_MXFP4_MXFP8 via DeepGEMM `fp8_fp4_mega_moe` fused dispatch+GEMM+act+GEMM+combine kernel; requires `hidden_size % 512 == 0` | `FUSED_COMM` |
| `mega_moe/mega_moe_cute_dsl.py` | `TrtllmCutedslMegaMoeNvfp4Impl` (aliased as `MegaMoECuteDsl`) | SM100/SM103/SM107 | NVFP4 fused dispatch+FC1+act+FC2+combine; internally selects one of its per-architecture kernel variants. Requires CUDA 13 Cutlass DSL and a symmetric-memory provider. Uses uniform activation constants and applies routing weights before FC2 quantization for DeepSeek-V4, after FC2 for other models. | `FUSED_COMM` |
| `marlin/` (+ the `fused_moe_marlin.py` alias module) | the abstract `MarlinFusedMoEBase` and its two leaves, `MarlinCudaNvfp4Impl` and `MarlinCudaW4a16Nvfp4Impl` | SM89-SM99 | W4A16 NVFP4 on Ada/Hopper (BF16 activations + FP4 weights, fused single-launch `marlin_nvfp4_moe_gemm` kernel); supports attention-DP + EP via external comm (scheduler precomputes routing; dispatch payload is plain BF16, no activation scales); non-NVFP4 layers (e.g. unquantized MTP draft layers) degrade to Cutlass in `resolve_moe_impl`, recorded in the layer's `MoEResolutionReport`; no dynamic EPLB | `EXTERNAL_COMM` |
| `fused_moe_triton.py` | `TritonFusedMoE` | SM90 only | GPT-OSS on Hopper (requires `swiglu_gptoss_style=True`) | (legacy path) |
| `fused_moe_vanilla.py` | `VanillaMoE` | All devices | Reference / debugging only | (legacy path) |

### Communication (`fused_moe/communication/`)

Communication strategies are auto-selected at runtime by `CommunicationFactory` based on hardware and configuration. Skipped for `FUSED_COMM` backends. See `communication_factory.py` for selection logic and `base.py` for the `Communication` ABC.

Communication strategies whose native workspace must participate in executor
sleep/wakeup implement the runtime-checkable `CheckpointableCommunication`
protocol from `communication/base.py`. The public
`checkpoint_resource_key()` identifies wrappers that share one underlying
workspace so the executor invokes `checkpoint_prepare()` and
`checkpoint_restore()` once per resource. Discovery must use this protocol;
do not add concrete strategy checks or inspect private workspace attributes.
The executor request queue owns the persistent admission state around the
checkpoint operation and reopens admission only after a complete wakeup.

### TRTLLM-Gen leaves (`fused_moe/trtllm_gen/`)

Everything TRTLLM-Gen lives in this package: the registered leaves and
every layer under them. `fused_moe_trtllm_gen.py` above it defines two names
and no code — `TRTLLMGenFusedMoE`, a module-level alias of
`TrtllmGenFusedMoEBase`, and a re-export of `trtllm_gen_leaf` — so that name
stays importable from its own path while the implementation lives here.

The package is self-contained, bottom to top:

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

### MegaMoE (`fused_moe/mega_moe/`)

| File | Role |
|------|------|
| `mega_moe_deepgemm.py` | `DeepgemmCudaW4a8Mxfp4Mxfp8Impl` backend (DeepGEMM `fp8_fp4_mega_moe` wrapper), aliased as `MegaMoEDeepGemm` |
| `mega_moe_cute_dsl.py` | `TrtllmCutedslMegaMoeNvfp4Impl` backend (architecture-specific CuteDSL MegaMoE wrapper, NVFP4), aliased as `MegaMoECuteDsl` |
| `CHUNKING_DESIGN.md` | Chunking design for MegaMoE (sequential multi-chunk, in-kernel barrier semantics) |
| `COMMUNICATION_COMPARISON.md` | Comparison of fused-comm SymmBuffer vs external comm strategies |
| `KERNEL_INTERNALS.html` | Reference for the underlying DeepGEMM kernel layout |

The exported CuteDSL kernel sources for `TrtllmCutedslMegaMoeNvfp4Impl` live under
`tensorrt_llm/_torch/cute_dsl_kernels/cutedsl_megamoe/`, preserving the upstream
package structure and `KernelClass` entry points. The runner constructs
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
| `test_moe_impl.py` | Identity, registration and resolution-by-`impl_id`, one section per migrated implementation | Active |
| `test_moe_module.py` | ConfigurableMoE integration tests (Backend × Comm × EPLB) | Active |
| `test_fused_moe.py` | Legacy MoE tests | Being replaced, do NOT add new tests here |
| `test_moe.py` | Legacy TRTLLM backend tests | Being replaced, do NOT add new tests here |

## Backend Selection

One function decides which implementation runs: `moe_resolution.resolve_moe_impl`.
It owns no capability knowledge. It orders candidates, asks each one, and returns
the first that accepts — so "which impl will run" has a single answer no matter who
asks, and the factory can no longer admit something a backend rejects.

```python
report = resolve_moe_impl(model_config, layer_idx=layer_idx)
impl_cls = impl_class_for(report)      # raises, with the full trail, if nothing fits
```

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
my `moe_backend` not take effect".

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

Until every implementation owns an identity, the report is diagnostic rather
than pinnable: `_legacy_backend_name` derives `winner`, `eligible`, and
`rejected[].backend` from `__name__`, so the report — including the `to_dict()`
artifact a tuning result is replayed from — records a class name rather than an
id, alongside `problem.quant` and `deployment.env.env_flags`. A class that
still spans several quantization formats deliberately does not synthesize a
`MoEImplId`; a canonical one is attached only where a class owns a fixed
`MoEImplDescriptor.identity`. The TRTLLM-Gen leaves, the DeepGEMM
implementations, `TrtllmCutedslFusedFc12Nvfp4Impl` and
`TrtllmCutedslMegaMoeNvfp4Impl` do today, each admitting exactly one format.
Until the rest follow and those fields switch to `descriptor.impl_id`, a class
name is not an identity: read `descriptor.identity`.

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

## Backend Capability Matrix

### Quantization Support

Each backend's `can_implement(p, d)` classmethod declares what it supports. Source of truth: the `can_implement` classmethod in each backend file.

| Quantization | Cutlass | TRTLLMGen | DeepGemm | DenseGEMM | CuteDSL | MegaMoE-DG | MegaMoE-CuteDSL | Triton | Marlin | Vanilla |
|---|---|---|---|---|---|---|---|---|---|---|
| Unquantized (BF16/FP16) | Y (SM80+) | Y (SM100/103, BF16, needs FlashInfer)§ | N | N | Y (SM107, BF16, SwiGLU only)¶ | N | N | Y (SM90, BF16) | N | Y |
| FP8 QDQ | Y (SM89+) | N | N | N | N | N | N | Y (SM90) | N | Y |
| FP8 Block Scales | Y (SM90, SM120) | Y (SM100/103) | Y (SM100/103) | N | N‡ | N | N | N | N | Y |
| NVFP4 | Y (SM100/103/107/120/121) | Y (SM100/103) | N | Y (SM100/103) | Y (SM100/103/107/120/121)¶ | N | Y (SM100/103/107, cu13 cutlass-dsl + symmetric-memory provider; per-expert alpha/norm_const + SwiGLU clamp) | N | Y (SM89-SM99) | Y |
| W4A16 NVFP4 | Y (SM80+, dequant-on-the-fly) | N | N | N | Y (SM120/121 via `CuteDslB12xFusedMoE`, needs flashinfer) | N | N | N | Y (SM89-SM99, BF16) | Y |
| W4A8 NVFP4 FP8 | N | Y (SM100/103) | N | N | N | N | N | N | N | N |
| W4A16 MXFP4 | Y (SM90) | Y (SM100/103) | N | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 FP8 | Y (SM100/103/107) | Y (SM100/103) | N | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 MXFP8 | Y (SM100/103/107/120/121) | Y (SM100/103) | N | N | N | Y (SM100/103, requires `hidden_size % 512 == 0`) | N | N | N | N |
| W8A8 MXFP8 MXFP8 | Y (SM100/103) | N | N | N | N | N | N | N | N | N |
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

¶ `CuteDslFusedMoE` on SM107 needs `MoEDep.CUTEDSL_RUBIN` (the installed CuTe DSL
exposes the Rubin helpers) and carries three constraints the other SMs do not:

- **Fused finalize is mandatory.** There is no unfused FC2 — NVFP4 has no plain
  grouped GEMM there, and the BF16 op fuses finalize unconditionally. Disabling
  finalize fusion, explicitly or by configuring LoRA, is
  `FINALIZE_FUSION_REQUIRED` rather than a late `NotImplementedError`.
- **Unquantized is SwiGLU-only.** `cute_dsl_bf16_gather_grouped_gemm_swiglu_rubin`
  takes no activation argument, so any other activation is
  `ACTIVATION_UNSUPPORTED`. NVFP4 forwards `activation_type` and serves Relu2 too.
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
layer finds no specialized backend — `CuteDslB12xFusedMoE` without flashinfer,
or any SM outside Marlin's 89-99 and B12x's 120/121.

‡ `CuteDslFusedMoE` has FP8-block-scale *plumbing* but no FP8-block-scale kernel:
`run_moe_fp8_block_scales` ends in `cute_dsl_fp8_group_blockwise_gemm_ref`, a
local pure-PyTorch helper that upcasts to fp32, materializes the expanded scales,
and loops `torch.einsum` per expert. The only CuteDSL runners the file imports
are the `Sm100BlockScaledContiguous*` NVFP4 ones. `can_implement` therefore
declines FP8 block scales rather than claiming a reference path as a backend, and
the algorithm's real owners are the DeepGEMM and TRTLLM-Gen leaves on
SM100/103 and Cutlass on SM90/SM120. Consequence worth knowing before changing
this: because a `CUTEDSL` request only ever considers the CuteDSL family plus the
Cutlass fallback, and Cutlass's FP8-block kernel stops at SM90/SM120, an explicit
`moe_backend="CUTEDSL"` on an FP8-block checkpoint at SM100 now fails resolution
instead of silently running the reference GEMM. `test_cute_dsl_fp8_block_scales`
and `test_cute_dsl_fp8_block_scales_4gpus` in
`tests/integration/defs/accuracy/test_llm_api_pytorch.py` are exactly that
configuration and are `pytest.mark.skip`-ed for this reason; they were never
scheduled in any `tests/integration/test_lists/` entry, so the skip changes no CI
stage. Point them at `DEEPGEMM` / `TRTLLM` if this checkpoint needs coverage
again.

### Activation Support

Activation is **declared, never inspected**: no factory or selection code asks
what class a backend is in order to decide whether it can run a checkpoint's
activation. Three types in `activation.py` carry that instead.

| Layer | Type | Written by |
|-------|------|------------|
| Carrier | `MoEActivation` = `SwigluActivation` \| `SwigluBiasActivation` \| `SiTuActivation` \| `SimpleActivation` | The **model**, passed as `create_moe(activation=...)`; the default `DEFAULT_MOE_ACTIVATION` is plain SwiGLU. One frozen dataclass per kind, so a kind and its constants cannot disagree, and a constant the kind does not have cannot be written down at all |
| Declaration | `MoEActivationSupport` — `kinds`, plus an `ActivationParamShape` for `alpha_beta` and one for `limit` | The **backend**, as an `activation_support` class attribute. Every backend declares one; `resolve_activation_support` raises if a module does not |
| Adapter | `install_activation_params` → `materialize_activation_params` | Not the backend: the `apply_moe_impl_construction_state()` every execution unit already calls installs the slots, and `ConfigurableMoE` re-installs after the EPLB sync and inside `create_weights`. A complete layer that owns kernels directly (`TritonFusedMoE`, `VanillaMoE`) calls it itself — `MoE.__init__` deliberately does not, because a wrapper's declaration is the backend's |

The carrier names constants **semantically**, per kind; only the declaration and
the adapter speak the ABI. `SwigluActivation` has just `clamp`;
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

The adapter assigns exactly three slots — `act_alpha`, `act_beta`, `act_clamp` —
and those are the only names a forward path or a quantization method reads. The
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

Declare `limit_when_absent` only when the ABI has no encoding for "no clamp":
`CuteDslFusedMoE` passes `float("inf")` because its epilogue always applies the
clamp functor. It is substituted *before* coercion, so a `PER_EXPERT_TENSOR`
backend gets it broadcast like any other scalar, and `MoEActivationSupport`
rejects it outright next to `limit=UNSUPPORTED`.

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
`MoEProblem.activation` (the kind) and `MoEProblem.activation_constants` (which
registers the carrier actually fills) — because the kind alone does not separate
a clamped layer from an unclamped one, and those compile to different kernels.

Quantization support (the matrix above) is a separate axis, and a backend can
execute a kind while still rejecting the quant algorithm it arrives with. The
gpt-oss SwiGLU package — per-expert bias plus a filled `alpha` / `beta` /
`limit` triple, summarized for selection as `MoEProblem.swiglu_gptoss_style` —
is declined by every specialized backend (`CuteDslFusedMoE`,
`CuteDslB12xFusedMoE`, `DeepGemmFusedMoE`, `TrtllmCutedslDenseGemmNvfp4Impl`,
the Marlin leaves) for the plain reason that none of them lists `SwigluBias` in
`kinds`, while the TRTLLM-Gen leaves execute it and admit only the formats
whose class sets `supports_gptoss_style` (checked by
`check_trtllm_gen_capabilities`, off the class rather than an instance because
it runs at resolution time).

Cutlass gates gpt-oss / MiniMax SwiGLU on unquantized, MXFP8, NVFP4, and the
MXFP4 family (`CutlassFusedMoE._GPTOSS_SUPPORTED_ALGOS` = `None`, `MXFP8`,
`NVFP4`, `W4A16_MXFP4`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`). The CUDA kernel
is not the constraint — `torch.ops.trtllm.fused_moe` takes `swiglu_alpha` /
`swiglu_beta` / `swiglu_limit` on the same call for every path, including
NVFP4 (`CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1>`), and TMA-WS GEMM1
applies `SwigluBiasAdaptor` in `doActivation`. NVFP4 is eligible only when
there is no expert bias (`MoEProblem.bias is not True`): MiniMax-M3 NVFP4
builds a `SwigluBiasActivation` (`gate_sigmoid_scale` / `linear_offset` /
`clamp`) with `bias=False`.
gpt-oss 1-D bias still goes through `NVFP4CutlassFusedMoEMethod`'s 2-D
weight pad and is rejected at selection. Unquantized and the MXFP4 family
can load that 1-D bias. W8A16 / W4A8_AWQ stay rejected because they inherit
the base `w3_w1_weight_shape[:2]` default (wrong for transposed layouts).
Widening the set without that distinction converts a selection-time
rejection into a weight-loading crash.

Four things make this easy to get wrong in either direction. First, the SM
asymmetry: `ModelConfig.get_mxfp4_quant_algo` maps a gpt-oss checkpoint to
`W4A16_MXFP4` below SM100 and to the `W4A8_MXFP4_*` pair at SM100+, so a gate
keyed on `W4A8_MXFP4_MXFP8` alone excludes Hopper entirely and — because Cutlass
is `FALLBACK_IMPL` and every other backend abstains — leaves gpt-oss unservable
there. Second, dropping the gate altogether is equally wrong: it un-skips the
`test_configurable_moe_single_gpu` gpt-oss × CUTLASS matrix, which then fails
inside weight loading rather than being rejected up front. Third, omitting
`None` rejects dummy / unquantized gpt-oss (`test_gpt_oss_trtllmgen[CUTLASS]`)
even though the kernel path is valid. Fourth, treating MiniMax SwigluBias as
"gpt-oss bias load" and excluding NVFP4 rejects
`TestMiniMaxM3::test_nvfp4` (`MoeConfig(backend="CUTLASS")`): MiniMax has no
expert bias, and the NVFP4 TMA-WS runner already applies `SwigluBiasAdaptor`.

### Scheduler / EPLB Constraints

- `FUSED_COMM` backends (`DeepgemmCudaW4a8Mxfp4Mxfp8Impl`, `TrtllmCutedslMegaMoeNvfp4Impl`) **must not** layer host-side `Communication.dispatch` / `.combine` on top of the fused kernel — `ConfigurableMoE._create_comm_strategy_auto` returns `None` for them.
- `FusedCommMoEScheduler` calls `backend.quantize_input(...)` for every chunk including zero-token chunks (so peer ranks can cross the in-kernel NVLink barrier). Each fused-comm backend therefore MUST make `quantize_input` tolerate `x.shape[0] == 0` and return its own empty tensor layout; the scheduler does NOT synthesize backend-specific empty tensors.
- Dynamic EPLB requires backend and quantization-method support. Backends gate
  wrapper-level constraints via `validate_configurable_moe`; `MegaMoEDeepGemm`
  supports dynamic EPLB by routing to slot IDs and migrating transformed DG
  weight tensors registered by its quantization method, with the constraint
  `num_slots % ep_size == 0`. `TrtllmCutedslMegaMoeNvfp4Impl` publishes `capabilities.supports_eplb=True` on its descriptor, and its quantization method declares `eplb_support_status = SUPPORTED` and registers the four MegaMoE-format derived params (`mega_fc{1,2}_weight{,_sf}`) and the per-expert `fc1_norm_const` with the load balancer alongside the raw NVFP4 family, so per-slot migration stays byte-consistent.
- `FUSED_COMM` backends use `ignore_allreduce=False` for EPLB statistic update because the fused kernel AllReduces routing stats internally.

## Canonical Examples

When adding new components, use these reference implementations:

| Task | Reference | Key methods to implement |
|------|-----------|--------------------------|
| New `EXTERNAL_COMM` Backend | `fused_moe_cutlass.py` (`CutlassFusedMoE`) | Declare `MoEImplBase`; implement `capabilities`, `activation_support`, `can_implement`, `_get_quant_method`, `quantize_input`, `run_moe`; call `apply_moe_impl_construction_state()` in `__init__` (`create_weights` / `load_weights` come from `MoEWeightOwnerMixin` — override only if allocation needs more); then add the class to `moe_resolution.IMPL_PRIORITY` and `BACKEND_FAMILY`, and add a branch in `create_moe_backend`. Declare the `descriptor` and `@register_moe_impl` on the same class that implements those four methods. Split an abstract parent out of it only when several quantization formats are planned — see the one-class-per-identity note below |
| New `FUSED_COMM` Backend | `mega_moe/mega_moe_deepgemm.py` (`DeepgemmCudaW4a8Mxfp4Mxfp8Impl`), `mega_moe/mega_moe_cute_dsl.py` (`TrtllmCutedslMegaMoeNvfp4Impl`) | Same as above + override `scheduler_kind = MoESchedulerKind.FUSED_COMM` and `validate_configurable_moe` for backend-specific constraints. For NVFP4 CuteDSL specifically, mirror the `TrtllmCutedslMegaMoeNvfp4Impl` pattern: capability probe for the CUDA 13 Cutlass DSL runtime, JSON-friendly tactic dict, lazy kernel imports from `cute_dsl_kernels/cutedsl_megamoe/`, and `quantize_input` that short-circuits zero-token input. |
| New Quantization Method | `quantization.py` → `FP8QDQFusedMoEMethod` | Subclass `FusedMoEMethod`, implement quant/dequant ops |
| New Communication Strategy | `communication/nvlink_one_sided.py` (`NVLinkOneSided`) | Subclass `Communication`, implement `prepare_dispatch`, `dispatch`, `combine`; also implement `CheckpointableCommunication` when native workspace mappings must follow executor sleep/wakeup |
| New Scheduler | `moe_scheduler.py` (`ExternalCommMoEScheduler` / `FusedCommMoEScheduler`) | Subclass `MoEScheduler`, implement `forward`; add new `MoESchedulerKind` value and wire into `create_moe_scheduler` factory |
| Backend Tests | `test_moe_backend.py` | Follow existing parametrize patterns |
| Identity Tests | `test_moe_impl.py` | Add one section per implementation: the id round-trips through the registry, the leaf admits only its own format, and a pin fails rather than degrades. Tests that merely *use* the class belong in `test_moe_backend.py` |
| Integration Tests | `test_moe_module.py` | Test Backend × Communication × EPLB combinations |

**Note on backend inheritance:** New execution-unit backends should inherit from `MoEImplBase` (in `impl_base.py`), NOT from `CutlassFusedMoE` and NOT from `MoE`. `MoE` is the complete-layer type (`ConfigurableMoE`, `TritonFusedMoE`).

`CutlassFusedMoE` is itself an execution unit now: it is no longer a `MoE` and has **no `forward`**, so it can only run as `ConfigurableMoE.backend`. Anything that needs a callable layer must wrap it (see `Llama4MinLatencyFusedMoE`, which extends `ConfigurableMoE` and pins `moe_cls=CutlassFusedMoE`).

These backends declare `MoEImplBase` directly — `CutlassFusedMoE`, `TrtllmGenFusedMoEBase`, `MarlinFusedMoEBase`, `TrtllmCutedslDenseGemmNvfp4Impl`, `TrtllmCutedslMegaMoeNvfp4Impl`, `CuteDslFusedMoE`, and the two DeepGEMM implementations. `TrtllmGenFusedMoEBase` and `MarlinFusedMoEBase` declare it as abstract family roots and their leaves inherit through them, which is the shape the next note describes. Only `CuteDslB12xFusedMoE` still reaches it through another implementation, `CutlassFusedMoE`, as a historical shortcut; that concrete inheritance is broken in its own follow-up item, not here.

**Note on one class per identity:** an identity and the code it names belong on the same class. That class declares the `descriptor`, takes its three contract attributes off it, carries `@register_moe_impl`, and implements the four abstract methods of `MoEImplBase`: `can_implement`, `_get_quant_method`, `quantize_input`, `run_moe`. Both DeepGEMM implementations, `TrtllmCutedslMegaMoeNvfp4Impl` and `TrtllmCutedslDenseGemmNvfp4Impl` are shaped that way — each holds its identity, its construction, and its whole contract in one place.

One identity may still fan out to several kernels underneath, as long as the four id segments stay the same for all of them. `TrtllmCutedslMegaMoeNvfp4Impl` dispatches its per-architecture kernel variants from one registered class, because the architecture it runs on is not an id segment — `can_implement` gates the SM, and the runner picks the kernel.

Splitting an abstract parent out below `MoEImplBase` is worth it only once several quantization formats actually share an implementation. The parent then holds what they share — construction, buffers, workspace sizing, the weight lifecycle — leaves at least `can_implement` to the leaves, stays abstract, and carries no `descriptor`, because a descriptor is exactly one identity and an abstract class has nothing to publish it for; each format is a registered leaf that narrows `can_implement` to itself. How much else the parent keeps depends on how much the formats share. `TrtllmGenFusedMoEBase` implements none of the four, because its formats differ in kernel ABI as well as in eligibility, so each leaf supplies its own `_get_quant_method`, `quantize_input` and `run_moe` through a per-ABI layer. `MarlinFusedMoEBase` is the opposite end: one kernel ABI serves both its formats, so it implements three of the four and a leaf is an identity plus a `can_implement`. Below one format the parent would carry no identity, implement nothing, and have exactly one subclass, which is why the other backends do not have one.

What such a parent must also not do is branch on the axis its subclasses divide. Every `if self.<quant or provider flag>:` in it is a subclass difference expressed as data, so it belongs in an override — which is why `TrtllmGenFusedMoEBase` keeps `supports_gptoss_style` and `supports_situ` (read off the *class* at resolution, where there is no instance to ask) but has no flag whose only reader is one of its own methods. Those became hooks: `_create_quant_method_weights`, `_configure_shared_expert_fusion`, `_requires_separated_routing`, `_situ_tp_weight_alignment`, `resolve_activation_support`. `needs_zero_expert_bias` is the one declarative flag left, because its four setters sit on two different inheritance levels and an override would repeat the same body four times.

**Note on implementation class names:** the name is derived from the identity as `<Provider><Technique><Quant>Impl`, with the kernel segment entering only when two ids differ in that segment alone — and when it does, it sits where the id puts it, between technique and quant, so the name reads in id order. `DeepgemmCudaFp8BlockScalesImpl` and `DeepgemmCudaW4a8Mxfp4Mxfp8Impl` share a provider and a technique but differ in `quant` as well as `kernel_name`, so neither needs its kernel segment to stay unambiguous.

`TrtllmCutedslMegaMoeNvfp4Impl` carries its kernel segment for the reason `TrtllmCutedslFusedFc12Nvfp4Impl` already does, and it carries it ahead of the collision: `trtllm.cutedsl.mega_moe.nvfp4`, `trtllm.cutedsl.fused_fc12.nvfp4`, `trtllm.cutedsl.dense_gemm.nvfp4` and the `trtllm.cutedsl.grouped_gemm.nvfp4` that `CuteDslFusedMoE` will claim differ in `kernel_name` alone, so the shorter `TrtllmCutedslNvfp4Impl` can only ever belong to one of them. Spelling the segment out now keeps that last migration a pure rename of the other class. Note that `cutedsl` as a bare token already names more than one leaf — a pin on it matches this kernel, the fused FC1+FC2 one and the dense-GEMM one, and `IMPL_PRIORITY` plus the SM gates decide which answers; name the kernel segment to pin one.

The `XXFusedMoE` names remain supported as module-level aliases — `DeepGemmFusedMoE`, `MegaMoEDeepGemm`, `MegaMoECuteDsl`, and `TRTLLMGenFusedMoE` are assignments onto the classes above, not base classes — so call sites, `issubclass` checks, and comments across the MoE tree keep resolving. An alias is deliberately not a parent: a parent under the old name would give one kernel a name that resolves and a name that does not, and for the family root it would leave a second class to keep in step with the first. `TRTLLMGenFusedMoE` is the one alias that stands for a family rather than a single leaf, which is why no descriptor can publish it: a descriptor is exactly one identity. It also means every gate against it must be `issubclass` / `isinstance` and never an equality check — resolution hands over a leaf, so `type(x) is TRTLLMGenFusedMoE` matches nothing.

**Known debt:** backends that have not migrated still carry the `XXFusedMoE` form as their real name, and that rename is deliberately not paid one backend at a time. It belongs with the change that switches the report's `winner` / `eligible` / `rejected[].backend` fields to `descriptor.impl_id` — see [Degradation is allowed, silence is not](#degradation-is-allowed-silence-is-not).

## Anti-Patterns

- **Do NOT add communication logic inside backends** — Communication belongs in `communication/`, backends do pure computation (exception: `FUSED_COMM` backends own the SymmBuffer collective inside their fused kernel)
- **Do NOT add forward-execution policy inside backends** — chunking, EPLB hook ordering, dispatch/combine sequencing belong in `MoEScheduler`
- **Do NOT modify old `XXFusedMoE` files for new features** — Use ConfigurableMoE + Backend + Scheduler architecture
- **Do NOT add new tests to `test_fused_moe.py` or `test_moe.py`** — Use `test_moe_backend.py` and `test_moe_module.py`
- **Do NOT skip `can_implement()` checks** — Every backend must declare what it supports; an unsupported combination returns `MoEEligibility.no(MoERejectReason.<CODE>, detail)`, never a bare `False` and never a free-form string a test would have to pattern-match
- **Do NOT probe the machine inside `can_implement()`** — No `get_sm_version()`, no `import` as a presence test, no `os.environ`. Read `d.env`; add the probe to `impl_environment.py` if it does not exist yet
- **Do NOT gate an activation on a class name** — No `isinstance`, no `moe_cls in [...]`, no ad-hoc `swiglu_gptoss_style` branch in a factory. A backend states what its kernels execute in `activation_support`; `_reject_unsupported_activation` reads that one declaration for every candidate, and `materialize_activation_params` enforces it again at construction. A per-class check re-creates the `assert moe_cls in [...]` this replaced, in a place the backend's own author will not find
- **Do NOT add a second selection entry point** — `resolve_moe_impl` is the only one. A helper that picks a class on the side is how `get_moe_cls` and the old `resolve_moe_cls` drifted apart in the first place
- **Do NOT substitute a backend without recording it** — A degradation must be visible in the `MoEResolutionReport`, not only in a log line
- **Do NOT pick `scheduler_kind` opportunistically** — Use `EXTERNAL_COMM` (default) unless your backend's fused kernel genuinely owns cross-rank exchange via SymmBuffer / equivalent in-kernel collective; `FUSED_COMM` brings hard invariants (no host comm, lockstep launches, no multi-stream overlap)
- **Schedulers MUST NOT write `moe.repeat_idx`** — `repeat_idx` is wrapper state advanced once per `forward_impl` regardless of chunk count
- **Do NOT allocate symmetric memory from `run_moe` in `FUSED_COMM` backends** — Symmetric-memory rendezvous is a build-time collective and is unsafe under PP / layer-skip or CUDA graph capture; allocate from `create_weights()` after `ConfigurableMoE` has synchronized EPLB-derived attributes. See `mega_moe/mega_moe_deepgemm.py` for the DG pattern and `mega_moe/mega_moe_cute_dsl.py:_alloc_symm_provider` for the NVSHMEM-equivalent provider.
- **Do NOT add a new `FUSED_COMM` backend without a zero-token `quantize_input` regression test** — `FusedCommMoEScheduler` calls `quantize_input` for every chunk (including zero-token chunks) so each backend must return its own empty-tensor layout.
- **Do NOT use a dataclass for an autotuner tactic without a tested `__repr__` round-trip** — `AutoTuner` serializes tactic values through `json.dumps`/`json.loads` and `eval(repr(tactic))`; a plain dataclass fails the `eval(repr(...))` check. Prefer a JSON-friendly **tuple of primitives or lists of primitives** (lists are JSON-friendly; tuples round-trip via `eval(repr(...))`). See the tactic-representation comment block in `tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py` for the 10-field tactic pattern (with legacy 8-field compatibility) (mma_tiler/cluster_shape as `list[int]`, `epi_flag_batch` as a nested `(int, int)` tuple, the rest as `bool`/`int`/`str`; `_unpack_tactic` is the single source of truth for the field order). The fallback tactic is the token-aware `default_megamoe_tactic(num_tokens)` helper, selected by `Sm100MegaMoENvfp4Runner.forward(tactic=-1)`, not a separate `fallback_tactic()` method.
- **Use `distributed_tuning_strategy=DistributedTuningStrategy.MERGE` on a multi-rank `FUSED_COMM` backend's `TuningConfig`** — Every EP rank must converge on the same compiled tactic for every chunk, otherwise the in-kernel NVLink dispatch barrier deadlocks. `PARALLEL` can profile different tactics on different ranks and is unsafe for fused collectives. Reference: `Sm100MegaMoENvfp4Runner.get_tuning_config`.
