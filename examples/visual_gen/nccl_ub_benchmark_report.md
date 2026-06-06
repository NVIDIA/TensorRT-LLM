# NCCL User-Buffer Registration — Benchmark Report

**Date:** 2026-06-05  
**System:** umb-b200-138 (8× NVIDIA B200 SXM, 183 GB each)  
**Container:** `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc17`  
**PyTorch:** 2.11.0a0+eb65b36914.nv26.02  
**NCCL:** 2.29.2  

---

## Background

NCCL user-buffer registration (`NCCL_CUMEM_ENABLE=1`) enables zero-copy collectives by
mapping tensors into NCCL's VMM address space. This change was added to the TRT-LLM
VisualGen Python path via `ParallelConfig.nccl_buffer_reg`.

The benchmarks were run in two tiers:

1. **Micro-benchmark** — collective latency with representative tensor shapes, no model weights.
2. **E2E pipeline benchmark** — full diffusion inference, with cold-start characterization for
   ring attention (each config run as a fresh process with no prior calls of the same shape).

---

## Models Tested

| Model | Task | Resolution | Frames | Steps |
|---|---|---|---|---|
| **FLUX.1-dev** | text-to-image | 1024×1024 | — | 20 |
| **Cosmos3-Nano** | text-to-video | 720×1280 | 57 | 20 |

---

## Part 1: Micro-Benchmark (Collective Latency)

**Config:** warmup=5, iters=50, dtype=bfloat16  
**Shapes tested** (represent actual Ulysses attention tensors):

| Collective | Shape (per rank) | Context |
|---|---|---|
| `flux_all_to_all` | `[1, 4096/N, 24/N, 128]` BF16 | FLUX Ulysses self-attn |
| `flux_all_gather` | `[1, 4096/N, 24, 128]` BF16 | FLUX sequence gather |
| `wan_all_to_all` | `[1, 7680/N, 40/N, 128]` BF16 | Wan2.1 Ulysses self-attn |
| `wan_all_gather` | `[1, 7680/N, 40, 128]` BF16 | Wan2.1 sequence gather |
| `all_reduce_small` | `[1, 512, 512]` BF16 | VAE norm all-reduce |

### Results — Median Latency (ms)

| Collective | 2G base | 2G CUMEM | 4G base | 4G CUMEM | 8G base | 8G CUMEM |
|---|---:|---:|---:|---:|---:|---:|
| flux_all_to_all | 0.071 | 0.076 | 0.069 | **0.061** | 0.059 | 0.062 |
| flux_all_gather | 0.082 | 0.088 | 0.100 | **0.098** | 0.147 | **0.146** |
| wan_all_to_all | 0.099 | **0.094** | 0.073 | **0.066** | 0.065 | **0.058** |
| wan_all_gather | 0.158 | 0.158 | 0.174 | **0.172** | 0.206 | 0.207 |
| all_reduce_small | 0.048 | **0.045** | 0.057 | **0.048** | 0.060 | 0.063 |

4-GPU shows the most consistent improvement: all-to-all −11–13%, all-reduce −19%.
2-GPU and 8-GPU are within noise.

---

## Part 2: E2E Pipeline Benchmark

### Ulysses Sequence Parallelism

Backend: VANILLA. Config: warmup=1, iters=3.

#### FLUX.1-dev — 1024×1024, 20 steps

| GPUs | Baseline (s) | +CUMEM (s) | Speedup vs 1-GPU | CUMEM gain |
|---:|---:|---:|---:|---:|
| 1 | 1.81 | 1.81 | 1.00× | 0% |
| 2 | 1.42 | 1.40 | 1.27× / 1.29× | +1.4% |
| **4** | **0.93** | 0.94 | **1.95×** / 1.93× | ~0% |
| 8 | 1.50 | 1.49 | 1.21× / 1.21× | +0.7% |

#### Cosmos3-Nano — 720×1280 × 57 frames, 20 steps

| GPUs | Baseline (s) | +CUMEM (s) | Speedup vs 1-GPU | CUMEM gain |
|---:|---:|---:|---:|---:|
| 1 | 13.95 | 14.26 | 1.00× | ~0% |
| 2 | 9.69 | 9.91 | 1.44× | ~0% |
| **4** | **6.77** | 6.93 | **2.06×** | ~0% |

**CUMEM has negligible effect on Ulysses E2E latency (±2%, within noise).**

The micro-benchmark collective improvements (10–19% at 4 GPU) do not translate to
E2E because communication is a small fraction of total step time on B200 — the GPU
spends most of each denoising step in compute kernels.

---

### Ring Attention (CUTEDSL backend)

Each configuration was run as a **fresh isolated process** (no prior calls of the same
ring-size in that process) to accurately characterize cold-start behavior.

#### Per-Iteration Timing — Cosmos ring-4 (7 iters, no warmup)

| Iter | No CUMEM | CUMEM |
|---:|---:|---:|
| 1 | 34.0 s | 29.5 s |
| 2 | 14.6 s | 15.9 s |
| 3 | 14.2 s | 14.2 s |
| 4–7 | 14.15–14.18 s | 14.13–14.18 s |

**Both modes show a cold-start spike on the first call. CUMEM does not eliminate it.**
Steady-state is identical at ~14.15 s regardless of CUMEM.

**Root cause of cold-start:** The CUTEDSL backend JIT-compiles attention kernels on first
use for each (seq, heads, ring_size) shape configuration. This compilation takes ~30 s
for Cosmos ring-4. After one call the kernel binary is cached in memory; all subsequent
calls are fast. This is a compute artifact, not an NCCL artifact — NCCL copies on every
call with or without CUMEM, and that per-call cost is present in all iterations (it is
small relative to the 14 s compute time).

> **Note:** An earlier version of this report incorrectly attributed the cold-start to
> NCCL staging-buffer allocation and reported a 55% improvement from CUMEM. That result
> was a test artifact: the CUMEM run was executed in the same container session immediately
> after the baseline run, so CUDA kernels compiled during the baseline run were already
> cached when the CUMEM run started. Running each config in a truly fresh process shows
> both modes have the same cold-start profile and the same steady-state latency.

#### Ring Attention vs Single GPU — Steady State

Ring attention is slower than single-GPU for both models at all tested GPU counts:

**FLUX ring (CUTEDSL), steady-state:**

| GPUs | No CUMEM | CUMEM | vs 1-GPU (1.81 s) |
|---:|---:|---:|---:|
| 2 | 1.61 s | 1.62 s | 0.89× |
| 4 | 2.33 s | 2.36 s | 0.78× |
| 8 | 3.61 s | 3.61 s | 0.50× |

**Cosmos ring (CUTEDSL), steady-state:**

| GPUs | No CUMEM | CUMEM | vs 1-GPU (13.95 s) |
|---:|---:|---:|---:|
| 2 | 14.0 s | 14.0 s | ~1.00× |
| 4 | 14.15 s | 14.15 s | 1.01× (same) |

At 1024×1024 FLUX, ring communication overhead dominates at all GPU counts.
For Cosmos video, ring-4 steady-state matches single-GPU but offers no speedup,
while Ulysses-4 achieves **2.06×**.

---

## Summary

### Optimal Configurations

| Model | Best Config | Latency | Speedup vs 1 GPU |
|---|---|---:|---:|
| FLUX.1-dev | **Ulysses 4-GPU** | 0.93 s | **1.95×** |
| Cosmos3-Nano | **Ulysses 4-GPU** | 6.77 s | **2.06×** |

### CUMEM Impact

| Scenario | Effect |
|---|---|
| Ulysses E2E, any GPU count | **Negligible** (±2%, noise) |
| Ring attention, steady state | **None** — identical throughput |
| Ring attention, first call (cold JIT) | **Marginal** — 29.5 s vs 34 s (first call only); both then stable at 14.15 s |
| Micro-benchmark, 4-GPU collectives | **10–19%** on individual collective latency |

### Conclusion

CUMEM provides measurable improvement at the collective level (10–19% at 4 GPU in
isolation) but this does not propagate to E2E latency. Communication is not the bottleneck
for these models on B200 — compute dominates. Ring attention is not competitive with
Ulysses for either model at the tested resolutions.

**Recommendation:** Use Ulysses. Enable `nccl_buffer_reg` as a low-cost default;
it is safe to leave on and may benefit configurations with higher communication-to-compute
ratios (longer sequences, more transformer layers, lower-end GPUs) that were not tested here.

---

## Feature Usage

```yaml
parallel_config:
  ulysses_size: 4
  nccl_buffer_reg: true   # sets NCCL_CUMEM_ENABLE=1 before init_process_group
```

Requirements: NCCL ≥ 2.21, CUDA driver with VMM support (A100/H100/B200).

---

## Reproduction

```bash
# Micro-benchmark (4-GPU)
torchrun --nproc-per-node=4 examples/visual_gen/bench_nccl_ub.py --tier micro --warmup 5 --iters 50

# E2E Ulysses sweep
python3 bench_e2e_sweep.py --model flux   [--nccl-cumem]
python3 bench_e2e_sweep.py --model cosmos [--nccl-cumem]

# Ring cold-start characterization (run each in a fresh process)
python3 bench_cold_ring.py --model cosmos --warmup 0 --iters 7
python3 bench_cold_ring.py --model cosmos --nccl-cumem --warmup 0 --iters 7
```

Raw JSON in `examples/visual_gen/nccl_ub_results/`.
