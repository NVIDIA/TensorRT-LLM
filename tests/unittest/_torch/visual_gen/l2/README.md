<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisualGen L2 component tests

L2 tests check one **component x feature** pair: a real-weight model component
runs with a feature on, and its output is compared with a reference computed
in the same job. They sit between kernel tests and end-to-end quality tests:

| Layer | Question | Reference | Gate |
|---|---|---|---|
| L1 kernel / op | Is this kernel numerically correct? | Exact math on the kernel's own inputs (e.g. decode the candidate's FP8/FP4 values and scales, compute in FP32) | Kernel-specific (rel-L2, ULP, error ratio vs a correctly rounded result) |
| **L2 component x feature** | Does the feature behave as intended inside the model? | Same component, same inputs, same job: feature off, or an independent implementation | rel-L2 threshold per feature category (below) |
| E2E | Is the generated media still good? | Golden media | LPIPS (`ENGINEERING_CRITERIA.md` section 5) |

## Metric

`rel-L2 = ||candidate - reference||_2 / ||reference||_2`, computed in FP32 over
the whole output tensor and gated per input case.

Rejected as gates (kept as diagnostics only):

- **Cosine similarity**: no fixed floor works. 0.99 fails correct NVFP4
  output, yet every injected FP8 defect we measured still scored 0.9967-0.9988.
- **p99 error**: ignores the worst 1% of elements, so it missed a defect that
  touched 0.97% of them.
- **Max-abs error**: dominated by a single outlier element.

## Thresholds

One threshold per feature category, measured on a **single DiT block** by
default.

| Category | Examples | Reference | rel-L2 |
|---|---|---|---:|
| Lossless | Parallelism, compile / attention backend, CUDA graph, offload, BF16 regression check | Feature off, or HF diffusers in FP32 for the BF16 check | 0.01 |
| 8-bit quantization | FP8, FP8 blockwise, SageAttention | BF16, same block | 0.05 |
| 4-bit quantization | NVFP4 | BF16, same block | 0.1 |

Exceptions:

- **Static (checkpoint) quantization** leaves the first and last three blocks
  unquantized, so it can only be tested on a full 40-block expert. The lossless
  threshold there is 0.015 (the benign floor grows with depth).
- **FP4 VAE** keeps its own 0.12 (measured max 0.0945).

Measured basis (Wan 2.2 A14B, real-trajectory inputs, Blackwell and Rubin):

- Dynamic quantization loss at one block: FP8 max 0.026, NVFP4 max 0.074.
  FP8 blockwise max 0.032 and SageAttention max 0.021 (Blackwell only).
- BF16 vs HF FP32 at one block: max 0.0068. Injected structural defects
  (RoPE height/width swap or frame stride, dropped cross-attention, rolled
  heads) all score at least 0.0149, so 0.01 catches all 64 cases.
- Lossless one-block features (compile, attention backends, fusion off): max
  0.0087. Ulysses, CUDA graph and offload: at most 1.2e-4.
- Loss is about the same at 1 block and at the full 40-block expert, and every
  block runs the same code, so one block is enough for dynamic features.
  Truncating at intermediate depths inflates both loss and noise and is not
  used.

## What the thresholds do and do not catch

- **Quantization thresholds are loss budgets, not bug checks.** They answer
  "did quantization get worse". Quantization amplifies benign perturbations
  6-12x, so real bugs can hide inside the budget: with NVFP4 on, a
  height/width RoPE swap passed 0.1 in all 32 cases (16 inputs x 2 GPUs). At
  0.05, FP8 caught that swap in 18/32 cases and a dropped cross-attention
  branch in 26/32. Quantization correctness belongs to L1 kernel tests against
  dequantized references.
- **Model-implementation bugs** (wrong tensor layout, wrong RoPE positions, a
  dropped branch) are caught by the BF16 regression check: at one and three
  blocks every injected structural defect was at least 5x above the benign
  noise floor.
- **Very small numeric errors** (one dropped bias, AdaLN modulation rounded to
  BF16) are below the noise floor at L2 and belong to L1. An end-to-end check
  (54 videos) found no measurable quality impact from them.

## Inputs

Synthetic inputs overstate loss badly: random text conditioning has RMS about
1, real UMT5 embeddings 0.001-0.012, and replacing only the conditioning with
real embeddings cut the high-noise expert's static FP8 loss from 0.96 to
0.006-0.010. Tests therefore
use DiT inputs captured from a real denoising trajectory.

`test_data/wan22_t2v_a14b_dit_inputs.safetensors` (Git LFS) holds every DiT
call of two short HF diffusers generations: 2 prompts x 4 steps x 2 CFG
branches = 16 calls, split between the two experts, at 320x320, 9 frames
(1200 tokens). Embeddings are stored without their zero padding and latents
once per step; the safetensors metadata lists every call and the generation
settings.

The fixture is per model, not per test: new feature tests for the same model
reuse it. Regenerate it only for a new model or a changed DiT input contract:

```bash
python generate_wan22_dit_inputs.py $LLM_MODELS_ROOT/Wan2.2-T2V-A14B-Diffusers \
    test_data/wan22_t2v_a14b_dit_inputs.safetensors
```

then update `FIXTURE_SHA256` and re-measure the thresholds.

## Tests

`test_wan22_t2v_a14b_l2.py`, both experts, one GPU:

| Test | Candidate | Reference | rel-L2 |
|---|---|---|---:|
| `test_bf16_matches_hf` | TRT-LLM BF16 | HF diffusers block in FP32 | 0.01 |
| `test_lossless_feature` | torch.compile (per block), CUDA graph runner, CPU offload | Same block, eager, weights on GPU | 0.01 |
| `test_attention_backend_lossless` | TRTLLM, CUTEDSL, FA4 attention | VANILLA (PyTorch SDPA) | 0.01 |
| `test_quantization_loss_budget` | Dynamic FP8, FP8 blockwise / NVFP4 | TRT-LLM BF16 | 0.05 / 0.1 |
| `test_sage_attention_loss_budget` | SageAttention (int8 Q/K, FP8 V, blocks 1/16/1) | Dense TRTLLM attention | 0.05 |

The BF16 regression check uses HF diffusers because CI has no "before the
change" build to compare against. HF runs in FP32 so the score holds only
TRT-LLM's own BF16 error: against a BF16 HF block the two implementations'
rounding adds up (max 0.0103 vs 0.0068 against FP32), and a 0.02 threshold
there would miss 2 of 16 dropped-cross-attention cases. Every feature test also asserts the feature
actually ran (quantized Linear methods, Sage config on the self-attention
backends, a compiled graph, a captured CUDA graph, weights back on the CPU).

GPU gating: NVFP4, CUTEDSL and FA4 need SM100+; the Sage recipe is
registered for SM100 only. Dynamic FP8 blockwise currently aborts in DeepGEMM
on SM107 and is marked xfail there.

```bash
cd tests/unittest
LLM_MODELS_ROOT=... pytest -s _torch/visual_gen/l2/test_wan22_t2v_a14b_l2.py
```

`-s` prints every per-case value next to its threshold.
