# Deep-Sim: Op-Level Simulation for TensorRT-LLM

**Status**: Vision document (not implementation-ready)
**Author**: venky/hisim-port branch
**Date**: 2026-04-01
**Builds on**: AIC-Sim (Phases 0-3, `slop/specs/2026-03-30-simulation-mode-design.md`)

## Problem Statement

AIConfigurator (AIC) predicts batch-level latency externally. Its collection
pipeline lives outside TRT-LLM and breaks when TRT-LLM's op APIs change.
Every new TRT-LLM version x hardware combination requires a manual collection
run (~30 GPU-hours on 8xH200) and manual `__compat__` version routing updates.

The core issue: **timing data and the ops it describes are maintained in
separate codebases with separate release cycles.**

## Vision

Embed per-op timing collection and simulation directly into TRT-LLM's op
abstraction layer. Two modes:

- **COLLECT**: Real hardware, real ops, records per-op timing as a side effect
- **SIMULATE**: No GPU, reads timing database, replaces compute with lookups

```
┌─────────────────────────────────────────────────────────────┐
│                    TRT-LLM Op Layer                         │
│                                                             │
│  DecoderLayer.forward()                                     │
│    ├── RMSNorm.forward()                                    │
│    ├── Attention.forward()                                  │
│    │   ├── qkv_proj (Linear, COLUMN)                        │
│    │   ├── attention_core (Flash/TRT-LLM backend)           │
│    │   └── o_proj (Linear, ROW) + AllReduce                 │
│    ├── RMSNorm.forward()                                    │
│    └── GatedMLP.forward()                                   │
│        ├── gate_up_proj (Linear, COLUMN)                    │
│        └── down_proj (Linear, ROW) + AllReduce              │
│                                                             │
│  ┌──────────┐    ┌──────────┐                               │
│  │ COLLECT  │    │ SIMULATE │                               │
│  │ mode     │    │ mode     │                               │
│  │          │    │          │                                │
│  │ Run real │    │ Lookup   │                               │
│  │ kernel + │    │ timing + │                               │
│  │ record   │    │ advance  │                               │
│  │ timing   │    │ clock    │                               │
│  └────┬─────┘    └────┬─────┘                               │
│       │               │                                     │
│       ▼               ▼                                     │
│  ┌─────────────────────────┐                                │
│  │   Op Timing Database    │                                │
│  │   {hw}/{version}/       │                                │
│  │   ops.jsonl             │                                │
│  └─────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

## Why Embed in TRT-LLM

| Property | External (AIC today) | Embedded (Deep-Sim) |
|----------|---------------------|---------------------|
| New op added | Write external collector + test cases | Op inherits timing hook from base class |
| New TRT-LLM version | Collector may break, needs `__compat__` update | Hook lives in op code, changes with it |
| New hardware | Manual ~30 GPU-hour collection run | One CI profiling run auto-populates |
| Version matching | External support_matrix.csv routing | DB is generated *by* the version it belongs to |
| API breakage risk | High (external calls into TRT-LLM ops) | Zero (hook is inside the op) |
| Collection automation | Semi-manual, requires collector maintenance | CI-integrated, runs as nightly/release job |

## Two Modes

### COLLECT Mode

Runs on real hardware with real GPUs. Each op executes normally but also
records its timing signature:

```python
# Conceptual — not final API
class TimedLinear(Linear):
    def forward(self, input, ...):
        if self._timing_mode == "collect":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = super().forward(input, ...)
            end.record()
            torch.cuda.synchronize()
            self._timing_db.record("gemm", {
                "m": input.shape[0], "n": self.out_features,
                "k": self.in_features, "dtype": str(input.dtype),
                "tp_mode": self.tp_mode.name,
            }, start.elapsed_time(end))
            return result
        else:
            return super().forward(input, ...)
```

**Output**: `op_timing_db/{hardware}/{trtllm_version}/ops.jsonl`

**Requirements**:
- Real distributed execution (TP allreduce timings depend on actual NCCL)
- Real model weights loaded
- Real GPU memory allocation
- Representative input shapes (from dataset or synthetic)

### SIMULATE Mode

Runs without GPU. Each op's compute is replaced with a timing lookup:

```python
# Conceptual — not final API
class SimulatedLinear(Linear):
    def forward(self, input, ...):
        predicted_ms = self._timing_db.lookup("gemm", {
            "m": input.shape[0], "n": self.out_features,
            "k": self.in_features, "dtype": str(input.dtype),
            "tp_mode": self.tp_mode.name,
        })
        self._sim_clock.step(predicted_ms / 1000.0)
        return torch.zeros(input.shape[0], self.out_features)  # dummy
```

**Requirements**:
- No GPU needed (CPU-only)
- Single process (TP is config parameter for DB lookup, not runtime)
- Model graph must be instantiated (to walk the op tree)
- Timing database for target hardware must exist

## Op Abstraction Boundary

The right hook point is `nn.Module.forward()` on TRT-LLM's core modules:

| Op | Module | File | Timing Key |
|----|--------|------|-----------|
| GEMM | `Linear` | `_torch/modules/linear.py` | `(m, n, k, dtype, tp_mode)` |
| Attention | `Attention` | `_torch/modules/attention.py` | `(batch, seq_len, heads, kv_heads, head_dim, window, dtype)` |
| LayerNorm | `RMSNorm` | `_torch/modules/rms_norm.py` | `(hidden_size, dtype)` |
| MLP | `GatedMLP` | `_torch/modules/gated_mlp.py` | `(hidden_size, inter_size, dtype)` |
| MoE | `FusedMoE` | `_torch/modules/fused_moe/` | `(tokens, experts, topk, hidden, inter, dtype)` |
| AllReduce | `AllReduce` | `_torch/distributed/ops.py` | `(size_bytes, tp_size)` |
| AllGather | `allgather` | `_torch/distributed/ops.py` | `(size_bytes, tp_size)` |
| MLA | MLA variants | `_torch/modules/attention.py` | `(batch, seq_len, latent_dim, ...)` |

These match AIC's existing per-op categories (`gemm_perf.txt`,
`context_attention_perf.txt`, `nccl_perf.txt`, etc.).

## Timing Database Format

```jsonl
{"op": "gemm", "params": {"m": 8192, "n": 65536, "k": 51200, "dtype": "float16", "tp_mode": "COLUMN"}, "latency_ms": 38.787, "hardware": "h200_sxm", "version": "1.3.0"}
{"op": "attention", "params": {"batch": 8, "seq_len": 16384, "heads": 96, "kv_heads": 1, "head_dim": 64, "mode": "context"}, "latency_ms": 5.640, "hardware": "h200_sxm", "version": "1.3.0"}
{"op": "allreduce", "params": {"size_bytes": 16384, "tp_size": 8}, "latency_ms": 0.042, "hardware": "h200_sxm", "version": "1.3.0"}
```

This is a superset of AIC's CSV format, expressed as structured JSON for
extensibility. Conversion from AIC's existing data is straightforward.

## Relationship to AIC-Sim (v1 Roadmap)

AIC-Sim (Phases 0-5) builds the **serving simulation infrastructure**:
scheduler, request lifecycle, metrics output, CLI integration. Deep-Sim
replaces only the **time prediction layer** — everything above it stays.

```
                    AIC-Sim (v1)              Deep-Sim (v2)
                    ─────────────             ─────────────
CLI / Bench         trtllm-bench --mode sim   (same)
Metrics             metrics.json, etc.        (same)
SimClock            SimClock.step()           (same)
Scheduler           Real Python scheduler     (same)
Request lifecycle   Real LlmRequest SM        (same)
                    ─────────────             ─────────────
Time prediction     AIC batch predictor       Op-level timing DB
Model execution     SimModelEngine (dummy)    SimulatedOps (per-op dummy)
Process model       Single-process            Single-process (SIMULATE)
                                              Multi-process (COLLECT)
```

### What's reusable from v1 in v2

| v1 Component | Reusable in v2? | Notes |
|-------------|-----------------|-------|
| `SimClock` | Yes | Accumulates time identically |
| `SimConfig` / `PredictorConfig` | Yes | Extends with `name="op_timing"` |
| `SimSampler` | Yes | Dummy token generation is the same |
| Single-process executor mode | Yes | SIMULATE mode uses same path |
| Metrics output (Phase 4) | Yes | Reads from SimClock regardless of source |
| CLI integration (Phase 5) | Yes | `--mode sim` dispatches to either predictor |
| `InferTimePredictor` ABC | Partially | v2 replaces batch-level predict() with per-op hooks |
| `SimModelEngine` | No (replaced) | v2 uses real model graph with simulated ops |

### What v1 should avoid to minimize throwaway

1. Don't deeply couple `SimModelEngine` to the predictor interface — keep it as a thin shell
2. Don't build elaborate batch→prediction mapping — v2 replaces this entirely
3. Do build robust `SimClock`, metrics, scheduler infrastructure — v2 reuses all of it
4. Do build single-process executor mode — v2 SIMULATE uses it

## Collection Automation Vision

```
TRT-LLM CI Pipeline (nightly or release)
  │
  ├─ Build TRT-LLM
  │
  ├─ Run COLLECT mode on target hardware (CI GPU cluster)
  │   ├─ Load representative models (Llama, DeepSeek, Qwen, ...)
  │   ├─ Run inference with timing hooks enabled
  │   ├─ Output: op_timing_db/{hw}/{version}/ops.jsonl
  │   └─ ~1 GPU-hour per model (vs AIC's ~30 GPU-hours for full sweep)
  │
  ├─ Package timing DB as artifact
  │   ├─ Versioned alongside TRT-LLM release
  │   ├─ Published to NGC / PyPI alongside wheel
  │   └─ Backward-compatible: old DBs still work with new TRT-LLM
  │
  └─ Validate: run SIMULATE mode, compare against COLLECT wall-clock
     └─ Prediction error < 15% → PASS
```

### Why faster than AIC collection

AIC sweeps a combinatorial space of `(M, N, K, dtype)` for each GEMM op —
thousands of test cases per op. Deep-Sim COLLECT records timing from actual
inference runs with representative inputs. A single model run touches all
ops with realistic shapes. Multiple models cover the shape space practically
rather than exhaustively.

## Scope Decisions (Inherited from HiSim)

Deep-Sim intentionally does NOT simulate:

- **Overlap scheduling** (CPU/GPU overlap) — timing assumes sequential execution
- **Mixed prefill + decode** — separate batch predictions (same as AIC-Sim)
- **Faithful KV tensor layout** — only capacity + transfer timing
- **Sampling-dependent behavior** — dummy tokens
- **Real distributed execution in SIMULATE mode** — TP/PP via config
- **Speculative decoding paths** — not modeled
- **Memory allocation dynamics** — CUDA allocator behavior not predicted

## Open Questions

1. **Interpolation strategy**: When SIMULATE encounters an op shape not in the DB
   (e.g., M=7777 but DB has M=8192 and M=4096), how to interpolate? AIC uses
   SOL/roofline for gaps. Deep-Sim could use nearest-neighbor or linear interp.

2. **Communication modeling**: NCCL allreduce timing depends on system topology,
   not just message size. How to parameterize this without running NCCL?

3. **Fusion detection**: TRT-LLM fuses ops (e.g., residual + layernorm). The
   fused kernel has different timing than sum of parts. COLLECT naturally
   captures fused timing; SIMULATE needs to know which ops are fused.

4. **KV cache I/O**: Cache read/write timing depends on memory system state.
   Model as constant per-token cost or measure empirically?

5. **Warm-up effects**: First few iterations may have different timing due to
   CUDA graph capture, JIT compilation, etc. Exclude from DB?

## Timeline Relationship

```
Now ──── AIC-Sim v1 (Phases 3-5) ──── Deep-Sim v2 ────────────►
         │                              │
         ├─ Phase 3: SimClock ✓         ├─ Op timing hooks (COLLECT)
         ├─ Phase 3.5: Single-process   ├─ Op timing DB format
         ├─ Phase 4: Metrics output     ├─ SimulatedOp base class
         ├─ Phase 5: CLI integration    ├─ Per-op SIMULATE forward
         └─ (uses AIC predictor)        └─ CI collection pipeline
                                            (replaces AIC predictor)
```

AIC-Sim is the foundation. Deep-Sim replaces the prediction layer but
keeps everything above it.
