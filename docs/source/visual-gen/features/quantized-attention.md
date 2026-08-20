# VisualGen Quantized Attention

```{note}
This page is an unindexed draft until the VisualGen documentation hub is introduced.
```

- [Overview](#overview)
  - [Recipes](#recipes)
  - [Configuration Surface](#configuration-surface)
- [QK16PV8 (CUTEDSL)](#qk16pv8-cutedsl)
- [SageAttention (TRTLLM)](#sageattention-trtllm)
- [Interaction With Other Features](#interaction-with-other-features)

## Overview

Visual generation models spend a large fraction of each denoising step inside attention, and every step is a full-context pass rather than an autoregressive decode. Quantized attention lowers the precision of the tensors the attention kernel itself consumes (Q, K, V), so that BMM1 (`Q·Kᵀ`) and/or BMM2 (`P·V`) run on narrower Tensor Core instructions. This is orthogonal to `VisualGenArgs.quant_config`, which quantizes the linear layers' *weights*: quantized attention quantizes *activations* inside the attention op and leaves the checkpoint untouched, so it needs no calibrated checkpoint and can be switched on for any supported model.

Quantized attention is configured through `VisualGenArgs.attention_config.quant_attention_config` (`QuantAttentionConfig`).

### Recipes

A recipe is the tuple `(qk_dtype, v_dtype, (q_block_size, k_block_size, v_block_size))`. Only the combinations below are accepted; `AttentionConfig` validates the recipe against the selected backend at construction time and raises `ValueError` otherwise (`tensorrt_llm/visual_gen/args.py`, `_validate_quant_attention_config`).

| Backend | `qk_dtype` | `v_dtype` | `(q, k, v)` block sizes | Common name |
|---|---|---|---|---|
| `TRTLLM` | `int8` | `fp8` | `(1, 1, 1)`, `(1, 4, 1)`, `(1, 16, 1)` | SageAttention (INT8 QK) |
| `TRTLLM` | `fp8` | `fp8` | `(1, 1, 1)`, `(1, 4, 1)` | SageAttention (FP8 QK) |
| `CUTEDSL` | `bf16` | `fp8` | `(0, 0, 1)`, `(0, 0, 0)` | QK16PV8 |
| `CUTEDSL` | `mxfp8` | `fp8` | `(0, 0, 0)`, `(0, 0, 1)` | Block-scaled MXFP8 Q/K |
| `CUTEDSL` | `nvfp4` | `fp8` | `(0, 0, 0)`, `(0, 0, 1)` | Block-scaled NVFP4 Q/K |

Notes:

- `v_dtype` only accepts `fp8`. Every quantized-attention kernel currently loads V as FP8 (e4m3), so BMM2 is always FP8; the recipes differ in how BMM1 is handled.
- `qk_dtype: "bf16"` means Q/K are **not** quantized — BMM1 stays in BF16. It's recommended to set `v_block_size` to 1 for this kernel for both accuracy and performance.
- `quant_attention_config` requires `backend` to be `TRTLLM` or `CUTEDSL`.

### Configuration Surface

| Field | Type | Default | Meaning |
|---|---|---|---|
| `qk_dtype` | `"bf16" \| "int8" \| "fp8" \| "mxfp8" \| "nvfp4"` | `"bf16"` | Q/K element format for BMM1. `bf16` leaves Q/K unquantized. |
| `v_dtype` | `"fp8"` | `"fp8"` | V element format for BMM2 (FP8 e4m3). |
| `q_block_size` | int ≥ 0 | `0` | Q tokens per SageAttention quantization block. `0` on the CuTe DSL paths. |
| `k_block_size` | int ≥ 0 | `0` | K tokens per SageAttention quantization block. `0` on the CuTe DSL paths. |
| `v_block_size` | int ≥ 0 | `0` | V block size on the hidden dimension. `0` = one tensor-wide V scale; `1` = one scale per channel. |

Routing (`tensorrt_llm/_torch/visual_gen/attention_backend/utils.py`) forwards the validated `quant_attention_config` into the backend constructor: `TrtllmAttention` for `TRTLLM`, and the dense `CuTeDSLAttention` FMHA backend for `CUTEDSL`.

## QK16PV8 (CUTEDSL)

**What it does.** Q and K stay in BF16 (or FP16), so BMM1 runs at full input precision. Only V is quantized to FP8 e4m3, so BMM2 runs on FP8 Tensor Cores. `v_block_size` selects how V is scaled:

- **`v_block_size: 1` (Recommended)** — one scale per KV head and channel.
- **`v_block_size: 0`** — a single per-tensor scale, folded into the kernel's `scale_output` scalar.

```{note}
Prefer `v_block_size: 1`. With `v_block_size: 0` the per-tensor scale is a *device* scalar, so folding it into `scale_output` requires reading it back to the host (`.item()`) inside `cute_dsl_fmha_fwd`. That readback drains the pipeline once per attention call, adding a device-host synchronization overhead. `v_block_size: 1` avoids the readback entirely, and its amax is additionally cheaper because a reduction to `(H, D)` parallelizes better than a reduction to one scalar. Per-head-per-channel scaling is also finer, so it's recommended to set `v_block_size` to `1` for both accuracy and performance.
```

**Configuration.**

```python
from tensorrt_llm import VisualGenArgs
from tensorrt_llm.visual_gen import AttentionConfig, QuantAttentionConfig

args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention_config=AttentionConfig(
        backend="CUTEDSL",
        quant_attention_config=QuantAttentionConfig(
            qk_dtype="bf16",
            v_dtype="fp8",
            q_block_size=0,
            k_block_size=0,
            v_block_size=1,
        ),
    ),
)
```

```yaml
attention_config:
  backend: CUTEDSL
  quant_attention_config:
    qk_dtype: bf16
    v_dtype: fp8
    q_block_size: 0
    k_block_size: 0
    v_block_size: 1
```

## SageAttention (TRTLLM)

**What it does.** SageAttention quantizes all three tensors with fine-grained scales, so both BMM1 and BMM2 run in low precision:

- **Q and K** are quantized to INT8 or FP8 e4m3 with one scale per *token block* per head. The block size is `q_block_size` for Q and `k_block_size` for K, measured in tokens along the sequence axis; a larger K block amortizes more scales but is coarser.
- **V** is quantized to FP8 e4m3 with `v_block_size` elements per scale along the hidden dimension. All supported recipes use `v_block_size = 1`, i.e. one scale per head per channel.

**Requirements and behavior.**

- Blackwell GPU.
- The recommended `qk_dtype: "int8"` is only supported on `sm_100a`.
  - `sm_103a` can use `qk_dtype: "fp8"` but its accuracy could be worse than `dk_dtype: "int8"`.

**Configuration.**

```python
from tensorrt_llm import VisualGenArgs
from tensorrt_llm.visual_gen import AttentionConfig, QuantAttentionConfig

args = VisualGenArgs(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    attention_config=AttentionConfig(
        backend="TRTLLM",
        quant_attention_config=QuantAttentionConfig(
            qk_dtype="int8",
            v_dtype="fp8",
            q_block_size=1,
            k_block_size=16,
            v_block_size=1,
        ),
    ),
)
```

```yaml
attention_config:
  backend: TRTLLM
  quant_attention_config:
    qk_dtype: int8
    v_dtype: fp8
    q_block_size: 1
    k_block_size: 16
    v_block_size: 1
```

## Interaction With Other Features

- **Linear-layer quantization** (`VisualGenArgs.quant_config`, e.g. FP8 block scales or NVFP4) is independent and can be combined with any attention recipe.
- **Sparse attention.** On `CUTEDSL`, quantized attention and Video Sparse Attention (VSA) are mutually exclusive and rejected by the validator. On `TRTLLM`, Skip Softmax uses the same backend and the SageAttention unit tests exercise the two together.
- **Parallelism.** SageAttention is covered by a multi-GPU Ulysses test (`tests/unittest/_torch/visual_gen/multi_gpu/test_ulysses_sage_attention.py`). The CuTe DSL dense backend produces LSE, so it also composes with Attention2D / Ring context parallelism; the TRTLLM Sage path does not expose LSE through this wrapper.

## Performance and Quality

_(TODO: fill in measured speedups and quality metrics per model, GPU, and recipe — these require
running the stack and are intentionally left blank.)_
