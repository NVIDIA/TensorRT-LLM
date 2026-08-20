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
└── MoEScheduler      (forward-execution strategy: chunking, EPLB hook ordering,
                       comm orchestration; selected by backend.scheduler_kind)
```

`forward_impl` is thin — it resolves `output_dtype`, delegates to `self.scheduler.forward(...)`, then runs wrapper-level bookkeeping that both schedulers share:

```python
def forward_impl(self, x, router_logits, ...):
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
| `FUSED_COMM` | `FusedCommMoEScheduler` | MegaMoEDeepGemm, MegaMoECuteDsl | Comm is fused into the backend kernel via SymmBuffer / NVSHMEM-equivalent peer-pointer mapping; no host comm; lockstep chunk launches; EPLB stats AllReduced internally |

The two paths have *deliberately opposite* invariants (`use_dp_padding` honored vs ignored, ADP padding kept vs stripped, empty-chunk substituted vs zero-token kernel launch, multi-stream overlap allowed vs forbidden). See `moe_scheduler.py` class docstrings and `MOE_SCHEDULER_DESIGN.md` for the full contract.

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
- `CutlassFusedMoE`, `TRTLLMGenFusedMoE`, `DeepGemmFusedMoE`, `CuteDslFusedMoE`,
  `CuteDslB12xFusedMoE`, `DenseGEMMFusedMoE`, `MegaMoEDeepGemm`,
  `MegaMoECuteDsl`, `MarlinFusedMoE`

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
| `impl_blocks.py` | The blocks `MoE` and `MoEImplBase` share — `MoEExecutionContractMixin` (scheduler-facing declarations, `forward_fake`) and `MoEWeightOwnerMixin` (`create_weights` / `load_weights` / `_check_configs`) |
| `quantization.py` | Quantization method implementations (`FusedMoEMethod` subclasses: weight creation, loading, quant/dequant ops per quant mode) |
| `routing.py` | Routing methods (`TopKRouting`, etc.) |
| `moe_load_balancer.py` | EPLB implementation |
| `moe_op_backend.py` | Op backend registry for TRTLLMGen (flashinfer/trtllm ops) |

### Backends (`fused_moe/`)

| File | Backend | Hardware | Scenario | Scheduler |
|------|---------|----------|----------|-----------|
| `fused_moe_cutlass.py` | `CutlassFusedMoE` | SM80+ | High throughput, most comprehensive quant support | `EXTERNAL_COMM` |
| `fused_moe_trtllm_gen.py` | `TRTLLMGenFusedMoE` | SM100/SM103 | Min-latency and high-throughput on Blackwell; also serves unquantized BF16 through FlashInfer's `trtllm_bf16_moe` (gated on `MoEDep.FLASHINFER_BF16_MOE`, not on a quant algo) | `EXTERNAL_COMM` |
| `fused_moe_deepgemm.py` | `DeepGemmFusedMoE` | SM100/SM103 | FP8 Block Scales on Blackwell | `EXTERNAL_COMM` |
| `fused_moe_densegemm.py` | `DenseGEMMFusedMoE` | SM100/SM103 | NVFP4 min-latency; CuTe DSL dense GEMM packs all experts into one matrix (vs Cutlass per-expert scatter), efficient for small token counts | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl.py` | `CuteDslFusedMoE` | SM100/SM103 | High throughput NVFP4, generally faster than Cutlass | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl_b12x.py` | `CuteDslB12xFusedMoE` | SM120/SM121 | NVFP4 hybrid CUTLASS-prefill / FlashInfer NVFP4 MoE decode — best perf on RTX PRO 6000 (SM120) and DGX Spark (SM121); select via the `CUTEDSL` backend path (it heads that family's candidate list, so it wins on SM120/121 when flashinfer is present and yields to `CuteDslFusedMoE` otherwise); single-GPU-shaped topology only — it rejects both `ep_size > 1` and attention-DP, because it has no dispatch/combine kernel and has never been exercised behind a DP allgather | `EXTERNAL_COMM` |
| `mega_moe/mega_moe_deepgemm.py` | `MegaMoEDeepGemm` | SM100/SM103 | W4A8_MXFP4_MXFP8 via DeepGEMM `fp8_fp4_mega_moe` fused dispatch+GEMM+act+GEMM+combine kernel; requires `hidden_size % 512 == 0` | `FUSED_COMM` |
| `mega_moe/mega_moe_cute_dsl.py` | `MegaMoECuteDsl` | SM100/SM103/SM107 | NVFP4 via ported architecture-specific CuteDSL fused dispatch+FC1+act+FC2+combine kernels; SM107 uses the native 2x-K MMA path and requires a Rubin-capable Cutlass DSL build; requires CUDA 13 Cutlass DSL runtime (PR #14354) and NVSHMEM provider (hard gate); threads per-expert `fc31_alpha`/`fc2_alpha`/`fc1_norm_const` through the kernel ABI and supports SwiGLU clamp via `swiglu_limit`; default deepgemm graph (topk score folded before fc1-out quant, host `combine_output.sum(dim=1)`) | `FUSED_COMM` |
| `fused_moe_marlin.py` | `MarlinFusedMoE` | SM89-SM99 | W4A16 NVFP4 on Ada/Hopper (BF16 activations + FP4 weights, fused single-launch `marlin_nvfp4_moe_gemm` kernel); supports attention-DP + EP via external comm (scheduler precomputes routing; dispatch payload is plain BF16, no activation scales); non-NVFP4 layers (e.g. unquantized MTP draft layers) degrade to Cutlass in `resolve_moe_impl`, recorded in the layer's `MoEResolutionReport`; no dynamic EPLB | `EXTERNAL_COMM` |
| `fused_moe_triton.py` | `TritonFusedMoE` | SM90 only | GPT-OSS on Hopper (requires `swiglu_gptoss_style=True`) | (legacy path) |
| `fused_moe_vanilla.py` | `VanillaMoE` | All devices | Reference / debugging only | (legacy path) |

### Communication (`fused_moe/communication/`)

Communication strategies are auto-selected at runtime by `CommunicationFactory` based on hardware and configuration. Skipped for `FUSED_COMM` backends. See `communication_factory.py` for selection logic and `base.py` for the `Communication` ABC.

### MegaMoE (`fused_moe/mega_moe/`)

| File | Role |
|------|------|
| `mega_moe_deepgemm.py` | `MegaMoEDeepGemm` backend (DeepGEMM `fp8_fp4_mega_moe` wrapper) |
| `mega_moe_cute_dsl.py` | `MegaMoECuteDsl` backend (SM100/SM103/SM107 CuteDSL kernel wrapper, NVFP4) |
| `CHUNKING_DESIGN.md` | Chunking design for MegaMoE (sequential multi-chunk, in-kernel barrier semantics) |
| `COMMUNICATION_COMPARISON.md` | Comparison of fused-comm SymmBuffer vs external comm strategies |
| `KERNEL_INTERNALS.html` | Reference for the underlying DeepGEMM kernel layout |

The ported CuteDSL kernel sources for `MegaMoECuteDsl` live under
`tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/` (flattened from the
upstream `moe_nvfp4_swapab/` + `src/` split). The package is loaded lazily
by `MegaMoECuteDsl` through `import_kernel()` so the heavyweight kernel
module is imported only on a supported GPU when a compatible Cutlass DSL
runtime is available.

### Design Documents

| File | Topic |
|------|-------|
| `MOE_SCHEDULER_DESIGN.md` | Scheduler refactor design + `MoEScheduler` contract |
| `mega_moe/CHUNKING_DESIGN.md` | MegaMoE chunking invariants |

### Tests

| File | Tests | Status |
|------|-------|--------|
| `test_moe_backend.py` | Backend unit tests (`run_moe`, `can_implement`) | Active |
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

Until the one-class-per-implementation migration is complete, the report is
diagnostic rather than pinnable: it records the legacy backend class name,
`problem.quant`, and `deployment.env.env_flags`. Legacy classes deliberately do
not synthesize a `MoEImplId`, because one class still spans several quantization
formats. A canonical ID is attached only when a leaf class owns one fixed
`MoEImplDescriptor.identity`.

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
| Unquantized (BF16/FP16) | Y (SM80+) | Y (SM100/103, BF16, needs FlashInfer `trtllm_bf16_moe`)§ | N | N | N | N | N | Y (SM90, BF16) | N | Y |
| FP8 QDQ | Y (SM89+) | N | N | N | N | N | N | Y (SM90) | N | Y |
| FP8 Block Scales | Y (SM90, SM120) | Y (SM100/103) | Y (SM100/103) | N | N‡ | N | N | N | N | Y |
| NVFP4 | Y (SM100/103/120/121) | Y (SM100/103) | N | Y (SM100/103) | Y (SM100/103/120/121) | N | Y (SM100/103/107, compatible cu13 cutlass-dsl; Rubin-capable build on SM107; per-expert alpha/norm_const + SwiGLU clamp) | N | Y (SM89-SM99) | Y |
| W4A16 NVFP4 | Y (SM80+, dequant-on-the-fly) | N | N | N | Y (SM120/121 via `CuteDslB12xFusedMoE`, needs flashinfer) | N | N | N | Y (SM89-SM99, BF16) | Y |
| W4A8 NVFP4 FP8 | N | Y (SM100/103) | N | N | N | N | N | N | N | N |
| W4A16 MXFP4 | Y (SM90) | Y (SM100/103) | N | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 FP8 | Y (SM100/103) | Y (SM100/103) | N | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 MXFP8 | Y (SM100/103) | Y (SM100/103) | N | N | N | Y (SM100/103, requires `hidden_size % 512 == 0`) | N | N | N | N |
| W8A8 MXFP8 MXFP8 | Y (SM100/103) | N | N | N | N | N | N | N | N | N |
| W4A8 AWQ | Y (SM89/90) | N | N | N | N | N | N | N | N | N |
| W8A16 | Y (SM80+) | N | N | N | N | N | N | N | N | N |
| INT4 WoQ (W4AFP8) | N | N | N | N | N | N | N | N | N | N |

§ The unquantized `TRTLLMGenFusedMoE` path is not a TRTLLM-Gen kernel at all: it
calls FlashInfer's `trtllm_bf16_moe` / `trtllm_bf16_routed_moe`, which is why it
is gated on `MoEDep.FLASHINFER_BF16_MOE` rather than on a quant algo, and why
`TRTLLMOpBackend` raises `NotImplementedError` for it. The row reads `Y` because
`can_implement` really can select it; without the FlashInfer symbols the layer
degrades to Cutlass with `DEP_MISSING` recorded in `degraded_from`, where the
pre-resolver code raised `RuntimeError` instead. The same path also requires
`intermediate_size_per_partition % 128 == 0` (`Bf16MoeLauncher::check_moe`);
a non-aligned shard is `SHAPE_UNALIGNED` and falls back to Cutlass.

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
the algorithm's real owners are `DeepGemmFusedMoE` / `TRTLLMGenFusedMoE` on
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

The matrix above is quantization only; activation style is a separate axis. The
gpt-oss SwiGLU package (per-expert bias plus `swiglu_alpha` / `swiglu_beta` /
`swiglu_limit`, surfaced as `MoEProblem.swiglu_gptoss_style`) is rejected by
every specialized backend — `CuteDslFusedMoE`, `CuteDslB12xFusedMoE`,
`DeepGemmFusedMoE`, `DenseGEMMFusedMoE`, `MarlinFusedMoE` — while
`TRTLLMGenFusedMoE` accepts only the algorithms in its `_GPTOSS_SUPPORTED_ALGOS`.

Cutlass gates gpt-oss / MiniMax SwiGLU on unquantized, MXFP8, NVFP4, and the
MXFP4 family (`CutlassFusedMoE._GPTOSS_SUPPORTED_ALGOS` = `None`, `MXFP8`,
`NVFP4`, `W4A16_MXFP4`, `W4A8_MXFP4_FP8`, `W4A8_MXFP4_MXFP8`). The CUDA kernel
is not the constraint — `torch.ops.trtllm.fused_moe` takes `swiglu_alpha` /
`swiglu_beta` / `swiglu_limit` on the same call for every path, including
NVFP4 (`CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1>`), and TMA-WS GEMM1
applies `SwigluBiasAdaptor` in `doActivation`. NVFP4 is eligible only when
there is no expert bias (`MoEProblem.bias is not True`): MiniMax-M3 NVFP4
passes `ActivationType.SwigluBias` + alpha/beta/limit with `bias=False`.
gpt-oss 1-D bias still goes through `NVFP4CutlassFusedMoEMethod`'s 2-D
weight pad and is rejected at selection. Unquantized and the MXFP4 family
can load that 1-D bias. W8A16 / W4A8_AWQ stay rejected because they inherit
the base `w3_w1_weight_shape[:2]` default (wrong for transposed layouts).
Widening the set without that distinction converts a selection-time
rejection into a weight-loading crash.

Three things make this easy to get wrong in either direction. First, the SM
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

The unquantized `TRTLLMGenFusedMoE` FlashInfer path has a separate shape
gate: `Bf16MoeLauncher::check_moe` requires
`intermediate_size % 128 == 0`, and the wrapper passes
`intermediate_size_per_partition`. Qwen3.5-35B BF16 TP8 shards 512 → 64 and
must degrade to Cutlass instead of dying in the kernel launcher.

### Scheduler / EPLB Constraints

- `FUSED_COMM` backends (`MegaMoEDeepGemm`, `MegaMoECuteDsl`) **must not** layer host-side `Communication.dispatch` / `.combine` on top of the fused kernel — `ConfigurableMoE._create_comm_strategy_auto` returns `None` for them.
- `FusedCommMoEScheduler` calls `backend.quantize_input(...)` for every chunk including zero-token chunks (so peer ranks can cross the in-kernel NVLink barrier). Each fused-comm backend therefore MUST make `quantize_input` tolerate `x.shape[0] == 0` and return its own empty tensor layout; the scheduler does NOT synthesize backend-specific empty tensors.
- Dynamic EPLB requires backend and quantization-method support. Backends gate
  wrapper-level constraints via `validate_configurable_moe`; `MegaMoEDeepGemm`
  supports dynamic EPLB by routing to slot IDs and migrating transformed DG
  weight tensors registered by its quantization method, with the constraint
  `num_slots % ep_size == 0`. `MegaMoECuteDsl` declares `eplb_support_status = SUPPORTED`: its quantization method registers the four MegaMoE-format derived params (`mega_fc{1,2}_weight{,_sf}`) and the per-expert `fc1_norm_const` with the load balancer alongside the raw NVFP4 family, so per-slot migration stays byte-consistent.
- `FUSED_COMM` backends use `ignore_allreduce=False` for EPLB statistic update because the fused kernel AllReduces routing stats internally.

## Canonical Examples

When adding new components, use these reference implementations:

| Task | Reference | Key methods to implement |
|------|-----------|--------------------------|
| New `EXTERNAL_COMM` Backend | `fused_moe_cutlass.py` (`CutlassFusedMoE`) | Declare `MoEImplBase`; implement `capabilities`, `can_implement`, `_get_quant_method`, `quantize_input`, `run_moe`; call `apply_moe_impl_construction_state()` in `__init__` (`create_weights` / `load_weights` come from `MoEWeightOwnerMixin` — override only if allocation needs more); then add the class to `moe_resolution.IMPL_PRIORITY` and `BACKEND_FAMILY`, and add a branch in `create_moe_backend`. Add a fixed `descriptor.identity` only for a one-implementation leaf class |
| New `FUSED_COMM` Backend | `mega_moe/mega_moe_deepgemm.py` (`MegaMoEDeepGemm`), `mega_moe/mega_moe_cute_dsl.py` (`MegaMoECuteDsl`) | Same as above + override `scheduler_kind = MoESchedulerKind.FUSED_COMM` and `validate_configurable_moe` for backend-specific constraints. For NVFP4 CuteDSL specifically, mirror the `MegaMoECuteDsl` pattern: capability probe for the CUDA 13 Cutlass DSL runtime, JSON-friendly tactic dict, lazy kernel import via `cute_dsl_kernels/mega_moe_nvfp4/import_kernel()`, and `quantize_input` that short-circuits zero-token input. |
| New Quantization Method | `quantization.py` → `FP8QDQFusedMoEMethod` | Subclass `FusedMoEMethod`, implement quant/dequant ops |
| New Communication Strategy | `communication/nvlink_one_sided.py` (`NVLinkOneSided`) | Subclass `Communication`, implement `prepare_dispatch`, `dispatch`, `combine` |
| New Scheduler | `moe_scheduler.py` (`ExternalCommMoEScheduler` / `FusedCommMoEScheduler`) | Subclass `MoEScheduler`, implement `forward`; add new `MoESchedulerKind` value and wire into `create_moe_scheduler` factory |
| Backend Tests | `test_moe_backend.py` | Follow existing parametrize patterns |
| Integration Tests | `test_moe_module.py` | Test Backend × Communication × EPLB combinations |

**Note on backend inheritance:** New execution-unit backends should inherit from `MoEImplBase` (in `impl_base.py`), NOT from `CutlassFusedMoE` and NOT from `MoE`. `MoE` is the complete-layer type (`ConfigurableMoE`, `TritonFusedMoE`).

`CutlassFusedMoE` is itself an execution unit now: it is no longer a `MoE` and has **no `forward`**, so it can only run as `ConfigurableMoE.backend`. Anything that needs a callable layer must wrap it (see `Llama4MinLatencyFusedMoE`, which extends `ConfigurableMoE` and pins `moe_cls=CutlassFusedMoE`).

Five backends declare `MoEImplBase` directly — `CutlassFusedMoE`, `TRTLLMGenFusedMoE`, `DenseGEMMFusedMoE`, `MegaMoECuteDsl`, `MegaMoEDeepGemm`. The remaining four (`CuteDslFusedMoE`, `CuteDslB12xFusedMoE`, `DeepGemmFusedMoE`, `MarlinFusedMoE`) still reach it through `CutlassFusedMoE` as a historical shortcut; that concrete inheritance is broken in each backend's own follow-up item, not here.

## Anti-Patterns

- **Do NOT add communication logic inside backends** — Communication belongs in `communication/`, backends do pure computation (exception: `FUSED_COMM` backends own the SymmBuffer collective inside their fused kernel)
- **Do NOT add forward-execution policy inside backends** — chunking, EPLB hook ordering, dispatch/combine sequencing belong in `MoEScheduler`
- **Do NOT modify old `XXFusedMoE` files for new features** — Use ConfigurableMoE + Backend + Scheduler architecture
- **Do NOT add new tests to `test_fused_moe.py` or `test_moe.py`** — Use `test_moe_backend.py` and `test_moe_module.py`
- **Do NOT skip `can_implement()` checks** — Every backend must declare what it supports; an unsupported combination returns `MoEEligibility.no(MoERejectReason.<CODE>, detail)`, never a bare `False` and never a free-form string a test would have to pattern-match
- **Do NOT probe the machine inside `can_implement()`** — No `get_sm_version()`, no `import` as a presence test, no `os.environ`. Read `d.env`; add the probe to `impl_environment.py` if it does not exist yet
- **Do NOT add a second selection entry point** — `resolve_moe_impl` is the only one. A helper that picks a class on the side is how `get_moe_cls` and the old `resolve_moe_cls` drifted apart in the first place
- **Do NOT substitute a backend without recording it** — A degradation must be visible in the `MoEResolutionReport`, not only in a log line
- **Do NOT pick `scheduler_kind` opportunistically** — Use `EXTERNAL_COMM` (default) unless your backend's fused kernel genuinely owns cross-rank exchange via SymmBuffer / equivalent in-kernel collective; `FUSED_COMM` brings hard invariants (no host comm, lockstep launches, no multi-stream overlap)
- **Schedulers MUST NOT write `moe.repeat_idx`** — `repeat_idx` is wrapper state advanced once per `forward_impl` regardless of chunk count
- **Do NOT allocate symmetric memory from `run_moe` in `FUSED_COMM` backends** — Symmetric-memory rendezvous is a build-time collective and is unsafe under PP / layer-skip or CUDA graph capture; allocate from `create_weights()` after `ConfigurableMoE` has synchronized EPLB-derived attributes. See `mega_moe/mega_moe_deepgemm.py` for the DG pattern and `mega_moe/mega_moe_cute_dsl.py:_alloc_symm_provider` for the NVSHMEM-equivalent provider.
- **Do NOT add a new `FUSED_COMM` backend without a zero-token `quantize_input` regression test** — `FusedCommMoEScheduler` calls `quantize_input` for every chunk (including zero-token chunks) so each backend must return its own empty-tensor layout. See `tests/unittest/_torch/moe/test_moe_backend.py::test_megamoe_deepgemm_quantize_input_zero_tokens` and `test_megamoe_cutedsl_quantize_input_zero_tokens` for the pattern.
- **Do NOT use a dataclass for an autotuner tactic without a tested `__repr__` round-trip** — `AutoTuner` serializes tactic values through `json.dumps`/`json.loads` and `eval(repr(tactic))`; a plain dataclass fails the `eval(repr(...))` check. Prefer a JSON-friendly **tuple of primitives or lists of primitives** (lists are JSON-friendly; tuples round-trip via `eval(repr(...))`). See the tactic-representation comment block in `tensorrt_llm/_torch/moe/custom_ops/cute_dsl_megamoe_custom_op.py` for the canonical 10-tuple tactic pattern and its legacy 8-tuple compatibility (`_unpack_tactic` is the single source of truth for the field order). For BF16 form-A, `Sm100MegaMoENvfp4Runner.forward(tactic=-1)` first checks the rank-identical shape-tuned table, then falls back to the token-aware `default_megamoe_tactic(num_tokens)` helper. Form-B and quantized combine use the standalone non-bulk tactic. Do not add a separate `fallback_tactic()` method.
- **Use `distributed_tuning_strategy=DistributedTuningStrategy.MERGE` on a multi-rank `FUSED_COMM` backend's `TuningConfig`** — Every EP rank must converge on the same compiled tactic for every chunk, otherwise the in-kernel NVLink dispatch barrier deadlocks. `PARALLEL` can profile different tactics on different ranks and is unsafe for fused collectives. Reference: `Sm100MegaMoENvfp4Runner.get_tuning_config`.
