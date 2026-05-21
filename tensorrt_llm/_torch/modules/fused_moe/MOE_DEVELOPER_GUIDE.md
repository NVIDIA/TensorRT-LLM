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
| `FUSED_COMM` | `FusedCommMoEScheduler` | MegaMoEDeepGemm | Comm is fused into the backend kernel via NVLink SymmBuffer; no host comm; lockstep chunk launches; EPLB stats AllReduced internally |

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
5. **Backend declares capabilities** — `can_implement()` declares supported quant/dtype; ConfigurableMoE adapts flow accordingly
6. **Backend declares scheduler** — `scheduler_kind` class attribute selects the forward path; lifecycle code stays generic, forward path stays specialized

## Architecture Transition (IMPORTANT)

The codebase is transitioning between two architectures:

| | Old Path | New Path |
|---|---|---|
| Entry | `XXFusedMoE` (e.g., `CutlassFusedMoE`) | `ConfigurableMoE` + `XXBackend` + `MoEScheduler` |
| Communication | Embedded inside each backend | Separated into `communication/` (or fused into kernel for `FUSED_COMM`) |
| Forward execution | Inline in backend | `MoEScheduler` (`moe_scheduler.py`) |
| EPLB | Only in WideEPMoE | Available to all backends |
| Status | Being replaced | Active development |

ConfigurableMoE currently supports these backends (`create_moe.py`):
- `CutlassFusedMoE`, `TRTLLMGenFusedMoE`, `DeepGemmFusedMoE`, `CuteDslFusedMoE`, `DenseGEMMFusedMoE`, `MegaMoEDeepGemm`

Still on old path (standalone, with embedded communication):
- `TritonFusedMoE`, `WideEPMoE`, `VanillaMoE`

**Rule: All new features should target ConfigurableMoE + Backend + Scheduler architecture.**

## File Map

### Core (`fused_moe/`)

| File | Role |
|------|------|
| `configurable_moe.py` | Orchestrator — wires Backend + Communication + EPLB + Scheduler; owns lifecycle and `forward_impl` |
| `moe_scheduler.py` | Forward-execution strategies (`MoEScheduler` ABC, `ExternalCommMoEScheduler`, `FusedCommMoEScheduler`, `create_moe_scheduler` factory) |
| `create_moe.py` | Factory — selects MoE class based on `model_config.moe_backend` |
| `interface.py` | Base class `MoE` and enums (`MoEWeightLoadingMode`, `MoESchedulerKind`, `AlltoallMethodType`) |
| `quantization.py` | Quantization method implementations (`FusedMoEMethod` subclasses: weight creation, loading, quant/dequant ops per quant mode) |
| `routing.py` | Routing methods (`TopKRouting`, etc.) |
| `moe_load_balancer.py` | EPLB implementation |
| `moe_op_backend.py` | Op backend registry for TRTLLMGen (flashinfer/trtllm ops) |

### Backends (`fused_moe/`)

| File | Backend | Hardware | Scenario | Scheduler |
|------|---------|----------|----------|-----------|
| `fused_moe_cutlass.py` | `CutlassFusedMoE` | SM80+ | High throughput, most comprehensive quant support | `EXTERNAL_COMM` |
| `fused_moe_trtllm_gen.py` | `TRTLLMGenFusedMoE` | SM100/SM103 | Min-latency and high-throughput on Blackwell | `EXTERNAL_COMM` |
| `fused_moe_deepgemm.py` | `DeepGemmFusedMoE` | SM100/SM103 | FP8 Block Scales on Blackwell | `EXTERNAL_COMM` |
| `fused_moe_densegemm.py` | `DenseGEMMFusedMoE` | SM100/SM103 | NVFP4 min-latency; CuTe DSL dense GEMM packs all experts into one matrix (vs Cutlass per-expert scatter), efficient for small token counts | `EXTERNAL_COMM` |
| `fused_moe_cute_dsl.py` | `CuteDslFusedMoE` | SM100/SM103 | High throughput NVFP4, generally faster than Cutlass | `EXTERNAL_COMM` |
| `mega_moe/mega_moe_deepgemm.py` | `MegaMoEDeepGemm` | SM100 only | W4A8_MXFP4_MXFP8 via DeepGEMM `fp8_fp4_mega_moe` fused dispatch+GEMM+act+GEMM+combine kernel; requires `hidden_size % 512 == 0` | `FUSED_COMM` |
| `fused_moe_triton.py` | `TritonFusedMoE` | SM90 only | GPT-OSS on Hopper (requires `swiglu_gptoss_style=True`) | (legacy path) |
| `fused_moe_wide_ep.py` | `WideEPMoE` | All GPUs | Deprecating — use ConfigurableMoE instead | (legacy path) |
| `fused_moe_vanilla.py` | `VanillaMoE` | All devices | Reference / debugging only | (legacy path) |

### Communication (`fused_moe/communication/`)

Communication strategies are auto-selected at runtime by `CommunicationFactory` based on hardware and configuration. Skipped for `FUSED_COMM` backends. See `communication_factory.py` for selection logic and `base.py` for the `Communication` ABC.

### MegaMoE (`fused_moe/mega_moe/`)

| File | Role |
|------|------|
| `mega_moe_deepgemm.py` | `MegaMoEDeepGemm` backend (DeepGEMM `fp8_fp4_mega_moe` wrapper) |
| `CHUNKING_DESIGN.md` | Chunking design for MegaMoE (sequential multi-chunk, in-kernel barrier semantics) |
| `COMMUNICATION_COMPARISON.md` | Comparison of fused-comm SymmBuffer vs external comm strategies |
| `KERNEL_INTERNALS.html` | Reference for the underlying DeepGEMM kernel layout |

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

## Backend Capability Matrix

### Quantization Support

Each backend's `can_implement(quant_algo, dtype_activation, swiglu_gptoss_style, ...)` method declares supported quantizations. Source of truth: the `can_implement` classmethod in each backend file.

| Quantization | Cutlass | TRTLLMGen | DeepGemm | DenseGEMM | CuteDSL | MegaMoE-DG | Triton | WideEP | Vanilla |
|---|---|---|---|---|---|---|---|---|---|
| Unquantized (BF16/FP16) | Y (SM80+) | N | N | N | N | N | Y (SM90, BF16) | Y | Y |
| FP8 QDQ | Y (SM89+) | N | N | N | N | N | Y (SM90) | Y | Y |
| FP8 Block Scales | Y (SM90, SM120) | Y (SM100/103) | Y (SM100/103) | N | Y (SM100/103) | N | N | Y | Y |
| NVFP4 | Y (SM100/103/120/121) | Y (SM100/103) | N | Y (SM100/103) | Y (SM100/103) | N | N | Y | Y |
| W4A8 NVFP4 FP8 | N | Y (SM100/103) | N | N | N | N | N | N | N |
| W4A16 MXFP4 | Y (SM90) | Y (SM100/103) | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 FP8 | Y (SM100/103) | Y (SM100/103) | N | N | N | N | Y (SM90) | N | N |
| W4A8 MXFP4 MXFP8 | Y (SM100/103) | Y (SM100/103) | N | N | N | Y (SM100, requires `hidden_size % 512 == 0`) | N | N | N |
| W4A8 AWQ | Y (SM89/90) | N | N | N | N | N | N | N | N |
| W8A16 | Y (SM80+) | N | N | N | N | N | N | N | N |
| INT4 WoQ (W4AFP8) | N | N | N | N | N | N | N | Y | N |

### Scheduler / EPLB Constraints

- `FUSED_COMM` backends (`MegaMoEDeepGemm`) **must not** layer host-side `Communication.dispatch` / `.combine` on top of the fused kernel — `ConfigurableMoE._create_comm_strategy_auto` returns `None` for them.
- Dynamic EPLB requires backend and quantization-method support. Backends gate
  wrapper-level constraints via `validate_configurable_moe`; `MegaMoEDeepGemm`
  supports dynamic EPLB by routing to slot IDs and migrating transformed DG
  weight tensors registered by its quantization method, with the constraint
  `num_slots % ep_size == 0`.
- `FUSED_COMM` backends use `ignore_allreduce=False` for EPLB statistic update because the fused kernel AllReduces routing stats internally.

## Canonical Examples

When adding new components, use these reference implementations:

| Task | Reference | Key methods to implement |
|------|-----------|--------------------------|
| New `EXTERNAL_COMM` Backend | `fused_moe_cutlass.py` (`CutlassFusedMoE`) | `can_implement`, `run_moe`, `create_weights`, `load_weights` |
| New `FUSED_COMM` Backend | `mega_moe/mega_moe_deepgemm.py` (`MegaMoEDeepGemm`) | Same as above + override `scheduler_kind = MoESchedulerKind.FUSED_COMM` and `validate_configurable_moe` for backend-specific constraints |
| New Quantization Method | `quantization.py` → `FP8QDQFusedMoEMethod` | Subclass `FusedMoEMethod`, implement quant/dequant ops |
| New Communication Strategy | `communication/nvlink_one_sided.py` (`NVLinkOneSided`) | Subclass `Communication`, implement `prepare_dispatch`, `dispatch`, `combine` |
| New Scheduler | `moe_scheduler.py` (`ExternalCommMoEScheduler` / `FusedCommMoEScheduler`) | Subclass `MoEScheduler`, implement `forward`; add new `MoESchedulerKind` value and wire into `create_moe_scheduler` factory |
| Backend Tests | `test_moe_backend.py` | Follow existing parametrize patterns |
| Integration Tests | `test_moe_module.py` | Test Backend × Communication × EPLB combinations |

**Note on backend inheritance:** New backends should inherit from `MoE` (in `interface.py`), NOT from `CutlassFusedMoE`. Current backends inherit from `CutlassFusedMoE` as a historical shortcut to reuse infrastructure (load balancer, weight management, TP/EP). This will be refactored — a dedicated `MoEBackend` interface will be extracted. `MegaMoEDeepGemm` and `DenseGEMMFusedMoE` already inherit directly from `MoE`.

## Anti-Patterns

- **Do NOT add communication logic inside backends** — Communication belongs in `communication/`, backends do pure computation (exception: `FUSED_COMM` backends own the SymmBuffer collective inside their fused kernel)
- **Do NOT add forward-execution policy inside backends** — chunking, EPLB hook ordering, dispatch/combine sequencing belong in `MoEScheduler`
- **Do NOT modify old `XXFusedMoE` files for new features** — Use ConfigurableMoE + Backend + Scheduler architecture
- **Do NOT add new tests to `test_fused_moe.py` or `test_moe.py`** — Use `test_moe_backend.py` and `test_moe_module.py`
- **Do NOT skip `can_implement()` checks** — Every backend must declare what it supports; unsupported combos must return `(False, reason)`
- **Do NOT pick `scheduler_kind` opportunistically** — Use `EXTERNAL_COMM` (default) unless your backend's fused kernel genuinely owns cross-rank exchange via SymmBuffer / equivalent in-kernel collective; `FUSED_COMM` brings hard invariants (no host comm, lockstep launches, no multi-stream overlap)
- **Schedulers MUST NOT write `moe.repeat_idx`** — `repeat_idx` is wrapper state advanced once per `forward_impl` regardless of chunk count
