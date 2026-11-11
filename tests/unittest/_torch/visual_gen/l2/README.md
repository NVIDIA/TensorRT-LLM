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
the whole output tensor and gated per input case. Shapes must match exactly;
broadcasting a candidate to the reference shape is not allowed.

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

Measured basis (Wan 2.2 A14B high-noise expert, first-step inputs at the
default 720x1280 x 81 frames, GB200 and Rubin, maximum over 32 calls with 8
fresh seeds):

- Dynamic quantization loss at one block: FP8 0.031, FP8 blockwise 0.029,
  NVFP4 0.053, SageAttention 0.006 (GB200 only).
- BF16 vs HF FP32 at one block: max 0.0083. Injected structural defects
  (RoPE height/width swap or frame stride, dropped cross-attention, rolled
  heads) all score at least 0.058, so 0.01 catches all 128 defect cases.
- Lossless one-block features: torch.compile 0.0042, attention backends
  0.0021, CUDA graph and offload 0. Parallelism: see the two-GPU tests below.
- Loss is about the same at 1 block and at the full 40-block expert, and every
  block runs the same code, so block 0 is enough for dynamic features.
  Truncating at intermediate depths inflates both loss and noise and is not
  used.

## What the thresholds do and do not catch

- **Quantization thresholds are loss budgets, not bug checks.** They answer
  "did quantization get worse". Quantization amplifies benign perturbations
  6-12x, so real bugs can hide inside the budget: with NVFP4 on, a dropped
  cross-attention branch passed 0.1 in all 32 calls, and a height/width RoPE
  swap scored 0.099-0.106, on both sides of it. Quantization correctness
  belongs to L1 kernel tests against dequantized references.
- **Model-implementation bugs** (wrong tensor layout, wrong RoPE positions, a
  dropped branch) are caught by the BF16 regression check: every injected
  structural defect scored at least 7x the largest benign BF16 error.
- **Very small numeric errors** (one dropped bias, AdaLN modulation rounded to
  BF16) are below the noise floor at L2 and belong to L1. An end-to-end check
  (54 videos) found no measurable quality impact from them.

## Inputs

Synthetic inputs overstate loss badly: random text conditioning has RMS about
1, real UMT5 embeddings about 0.07 over their valid tokens, and replacing only
the conditioning with real embeddings cut the high-noise expert's static FP8
loss from 0.96 to 0.006-0.010. The tests therefore use real inputs, and the
only real DiT input that needs no DiT to produce is the first denoising step's:
the seeded Gaussian latent the pipeline starts from.

Every run builds them with TRT-LLM's Wan 2.2 A14B defaults
(`get_wan_default_params`): 720x1280, 81 frames, 512-token text, the
scheduler's first timestep for 40 steps, the pipeline's default negative
prompt, and the prompts the repository already uses for Wan T2V. That is 2
prompts x 2 CFG branches = 4 calls (75600 tokens each). Text embeddings come
from the checkpoint's UMT5 encoder (about 11 GB, loaded once per process and
freed; 30 s on GB200, up to 100 s on Rubin).

The seed defaults to 42, matching the existing Wan regression tests, and is
printed with every result. Inputs are still generated in every run. A CPU
generator makes the latent depend on the seed only, not on the GPU;
`TLLM_VISUALGEN_L2_SEED=<seed>` overrides the default for additional coverage
or reproduction. All references are computed in the same run, so no golden
output depends on the seed. A separate multi-seed experiment (8 seeds at the
default size and 16 at 288x512 x 33 frames) found no check varied by more
than 0.005 (FP8; the BF16 check by 0.0016).

Only the high-noise expert is tested. Block 0 of one expert already covers
the code every DiT block of both experts runs, and the low-noise expert's
inputs (from step 13 on) exist only after running the full high-noise expert.
Inputs that only match real statistics are not a substitute: text sampled
from real embedding statistics with a random latent underestimated the
low-noise expert's loss (FP8 by 21%, SageAttention by 47%), because its
latents carry image content from earlier steps.

## Tests

`test_wan22_t2v_a14b_l2.py`, one GPU:

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
rounding adds up (max 0.0103 vs 0.0068 against FP32 on a 4-step capture), and
a 0.02 threshold there would miss 2 of 16 dropped-cross-attention cases. Every
feature test also asserts the feature actually ran (the requested quantized
Linear method, the exact attention backend and Sage recipe, a compiled graph,
a captured CUDA graph, weights back on the CPU). Parallel quantization checks
the method or attention recipe on both ranks, unwrapping sequence-parallel
attention wrappers to check the compute backend.

GPU gating: NVFP4, CUTEDSL and FA4 need SM100+; the Sage recipe is
registered for SM100 only.

`test_wan22_t2v_a14b_l2_multi_gpu.py`, two GPUs. The test process builds the
inputs and the single-GPU reference; the candidate runs on two spawned ranks
with the same inputs:

| Test | Candidate | Reference | rel-L2 |
|---|---|---|---:|
| `test_parallel_lossless` | Ulysses, async Ulysses, TP; Ring and Attention2D with FA4 | Same block on one GPU, same attention backend | 0.01 |
| `test_cfg_parallel_lossless` | The pipeline's CFG-parallel step (one branch per GPU, all-gather, guidance) | Guidance applied to the two branches run on one GPU | 0.01 |
| `test_parallel_quantization_loss_budget` | Ulysses or TP with dynamic FP8, FP8 blockwise, NVFP4, SageAttention | TRT-LLM BF16 on one GPU | 0.05 / 0.1 / 0.05 |

Measured on GB200 and Rubin: Ulysses at most 1e-4 against one GPU, Ring and
Attention2D 0.0021, TP 0.0043 (its all-reduce sums in a different order), CFG
parallel 0.0008.

- **Parallel x quantization is a loss budget.** Dynamic per-tensor
  activation scales are computed per rank, so a parallel quantized block is
  not bitwise comparable with the single-GPU quantized block (on a 4-step
  capture they differed by up to 0.02 for FP8 and 0.047 for NVFP4). Its loss
  against BF16 matches the single-GPU loss within 0.002, so the single-GPU
  budgets apply.
- **CFG reference.** The non-parallel CFG step batches both branches into one
  forward. That changes GEMM shapes (up to 0.002 per branch), and guidance
  multiplies the difference of two close predictions, which raised it to
  0.024 on the high-noise expert of a 4-step capture. The test therefore guides the single-GPU
  branch outputs directly.

Ring and Attention2D merge partial attention by its log-sum-exp, which only
FA4 provides, so they are gated on SM100+ like FA4.

```bash
cd tests/unittest
LLM_MODELS_ROOT=... pytest -s _torch/visual_gen/l2/test_wan22_t2v_a14b_l2.py
LLM_MODELS_ROOT=... pytest -s _torch/visual_gen/l2/test_wan22_t2v_a14b_l2_multi_gpu.py
```

`-s` prints every per-case value next to its threshold.

## CI placement

The single-GPU suite is listed in the `pre_merge` PyTorch group of
`l0_b200.yml`. The two-GPU suite is listed in the exactly-two-GPU
`pre_merge` PyTorch group of `l0_dgx_b200.yml`. Both need
`Wan2.2-T2V-A14B-Diffusers` under `LLM_MODELS_ROOT`. The two-GPU test
spawns its own two ranks, like the existing VisualGen parallel tests.
