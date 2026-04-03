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

ConfigurableMoE composes independent components via composition (not inheritance):

```text
ConfigurableMoE
├── Backend (pure computation): routing → quantize → FC1 → activation → FC2
├── Communication (distributed): dispatch tokens → compute → combine results
├── EPLB (optional): dynamic expert migration across GPUs
└── Multi-chunk: splits tokens into chunks to reduce peak memory usage
```

Execution flow within ConfigurableMoE (`_forward_chunk_impl`):

```text
routing() → [EPLB] → quantize/dispatch (adaptive order) → backend.run_moe() → combine()
                              │                                                    │
                       Communication                                        Communication

Adaptive order (based on comm.supports_post_quant_dispatch()):
  Post-quant flow: quantize_input() → comm.dispatch()   (send quantized data)
  Pre-quant flow:  comm.dispatch() → quantize_input()   (send raw, quantize locally)
```

### Core Design Principles

1. **Composition over inheritance** — Backend, Communication, and EPLB are independent, composable components
2. **Any Backend × Any Communication × EPLB On/Off** — All valid combinations should work
3. **Backend = pure computation** — No communication logic, no EPLB logic inside backends
4. **Communication is pluggable** — Auto-selected at runtime by `CommunicationFactory` based on hardware and workload
5. **Backend declares capabilities** — `can_implement()` declares what it supports; ConfigurableMoE adapts flow accordingly

## Architecture Transition (IMPORTANT)

The codebase is transitioning between two architectures:

| | Old Path | New Path |
|---|---|---|
| Entry | `XXFusedMoE` (e.g., `CutlassFusedMoE`) | `ConfigurableMoE` + `XXBackend` |
| Communication | Embedded inside each backend | Separated into `communication/` |
| EPLB | Only in WideEPMoE | Available to all backends |
| Status | Being replaced | Active development |

ConfigurableMoE currently supports these backends (`create_moe.py`):
- CutlassFusedMoE, TRTLLMGenFusedMoE, DeepGemmFusedMoE, CuteDslFusedMoE

Still on old path (standalone, with embedded communication):
- TritonFusedMoE, WideEPMoE, VanillaMoE

**Rule: All new features should target ConfigurableMoE + Backend architecture.**

## File Map

### Core (`fused_moe/`)

| File | Role |
|------|------|
| `configurable_moe.py` | Orchestrator — wires Backend + Communication + EPLB + multi-chunk |
| `create_moe.py` | Factory — selects MoE class based on `model_config.moe_backend` |
| `interface.py` | Base class `MoE` and enums (`MoEWeightLoadingMode`, `AlltoallMethodType`) |
| `quantization.py` | Quantization method implementations (`FusedMoEMethod` subclasses: weight creation, loading, quant/dequant ops per quant mode) |
| `routing.py` | Routing methods (`TopKRouting`, etc.) |
| `moe_load_balancer.py` | EPLB implementation |
| `moe_op_backend.py` | Op backend registry for TRTLLMGen (flashinfer/trtllm ops) |

### Backends (`fused_moe/`)

| File | Backend | Hardware | Scenario |
|------|---------|----------|----------|
| `fused_moe_cutlass.py` | CutlassFusedMoE | SM80+ | High throughput, most comprehensive quant support |
| `fused_moe_trtllm_gen.py` | TRTLLMGenFusedMoE | SM100/SM103 | Min-latency and high-throughput on Blackwell |
| `fused_moe_deepgemm.py` | DeepGemmFusedMoE | SM100/SM103 | FP8 Block Scales on Blackwell |
| `fused_moe_triton.py` | TritonFusedMoE | SM90 only | GPT-OSS on Hopper (requires `swiglu_gptoss_style=True`) |
| `fused_moe_cute_dsl.py` | CuteDslFusedMoE | SM100/SM103 | High throughput NVFP4, generally faster than Cutlass |
| `fused_moe_wide_ep.py` | WideEPMoE | All GPUs | Deprecating — use ConfigurableMoE instead |
| `fused_moe_vanilla.py` | VanillaMoE | All devices | Reference / debugging only |

### Communication (`fused_moe/communication/`)

Communication strategies are auto-selected at runtime by `CommunicationFactory` based on hardware and configuration. See `communication_factory.py` for selection logic and `base.py` for the `Communication` ABC.

### Tests

| File | Tests | Status |
|------|-------|--------|
| `test_moe_backend.py` | Backend unit tests (run_moe, can_implement) | Active |
| `test_moe_module.py` | ConfigurableMoE integration tests (Backend × Comm × EPLB) | Active |
| `test_fused_moe.py` | Legacy moe tests | Being replaced, do NOT add new tests here |
| `test_moe.py` | Legacy TRTLLM backend tests | Being replaced, do NOT add new tests here |

## Backend Capability Matrix

### Quantization Support

Each backend's `can_implement(quant_algo, dtype_activation, swiglu_gptoss_style)` method declares supported quantizations. Source of truth: the `can_implement` classmethod in each backend file.

| Quantization | Cutlass | TRTLLMGen | DeepGemm | Triton | CuteDSL | WideEP | Vanilla |
|---|---|---|---|---|---|---|---|
| Unquantized (BF16/FP16) | Y (SM80+) | N | N | Y (SM90, BF16) | N | Y | Y |
| FP8 QDQ | Y (SM89+) | N | N | Y (SM90) | N | Y | Y |
| FP8 Block Scales | Y (SM90, SM120) | Y (SM100/103) | Y (SM100/103) | N | N | Y | Y |
| NVFP4 | Y (SM100/103/120/121) | Y (SM100/103) | N | N | Y (SM100/103) | Y | Y |
| W4A8 NVFP4 FP8 | N | Y (SM100/103) | N | N | N | N | N |
| W4A16 MXFP4 | Y (SM90) | Y (SM100/103) | N | Y (SM90) | N | N | N |
| W4A8 MXFP4 FP8 | Y (SM100/103) | Y (SM100/103) | N | Y (SM90) | N | N | N |
| W4A8 MXFP4 MXFP8 | Y (SM100/103) | Y (SM100/103) | N | N | N | N | N |
| W4A8 AWQ | Y (SM89/90) | N | N | N | N | N | N |
| W8A16 | Y (SM80+) | N | N | N | N | N | N |
| INT4 WoQ (W4AFP8) | N | N | N | N | N | Y | N |



## Canonical Examples

When adding new components, use these reference implementations:

| Task | Reference | Key methods to implement |
|------|-----------|------------------------|
| New Backend | `fused_moe_cutlass.py` (CutlassFusedMoE) | `can_implement`, `run_moe`, `create_weights`, `load_weights` |
| New Quantization Method | `quantization.py` → `FP8QDQFusedMoEMethod` | Subclass `FusedMoEMethod`, implement quant/dequant ops |
| New Communication Strategy | `communication/nvlink_one_sided.py` (NVLinkOneSided) | Subclass `Communication`, implement `prepare_dispatch`, `dispatch`, `combine` |
| Backend Tests | `test_moe_backend.py` | Follow existing parametrize patterns |
| Integration Tests | `test_moe_module.py` | Test Backend × Communication × EPLB combinations |

**Note on backend inheritance:** New backends should inherit from `MoE` (in `interface.py`), NOT from `CutlassFusedMoE`. Current backends inherit from `CutlassFusedMoE` as a historical shortcut to reuse infrastructure (load balancer, weight management, TP/EP). This will be refactored — a dedicated `MoEBackend` interface will be extracted.

## Anti-Patterns

- **Do NOT add communication logic inside backends** — Communication belongs in `communication/`, backends do pure computation
- **Do NOT modify old `XXFusedMoE` files for new features** — Use ConfigurableMoE + Backend architecture
- **Do NOT add new tests to `test_fused_moe.py` or `test_moe.py`** — Use `test_moe_backend.py` and `test_moe_module.py`
- **Do NOT skip `can_implement()` checks** — Every backend must declare what it supports; unsupported combos must return `(False, reason)`
