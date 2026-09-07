# VisualGen Quantized Attention

```{note}
This page is an unindexed draft until the VisualGen documentation hub is introduced.
```

- [Overview](#overview)
  - [Recipes](#recipes)
  - [Configuration Surface](#configuration-surface)
- [QK16PV8 (CUTEDSL)](#qk16pv8-cutedsl)
- [SageAttention (TRTLLM)](#sageattention-trtllm)
- [MXFP8 / NVFP4 (CUTEDSL / FlashInfer)](#mxfp8-nvfp4-cutedsl)
- [Interaction With Other Features](#interaction-with-other-features)

## Overview

Visual generation models spend a large fraction of each denoising step inside attention, and every step is a full-context pass rather than an autoregressive decode. Quantized attention lowers the precision of the tensors the attention kernel itself consumes (Q, K, V), so that BMM1 (`Q·Kᵀ`) and/or BMM2 (`P·V`) run on narrower Tensor Core instructions. This is orthogonal to `VisualGenArgs.quant_config`, which quantizes the linear layers' *weights*: quantized attention quantizes *activations* inside the attention op and leaves the checkpoint untouched, so it needs no calibrated checkpoint and can be switched on for any supported model.

Quantized attention is configured through `VisualGenArgs.attention_config.quant_attention_config` (`QuantAttentionConfig`).

### Recipes

A recipe is the tuple `(qk_dtype, v_dtype, (q_block_size, k_block_size, v_block_size))`. Only the combinations below are accepted; `AttentionConfig` validates the recipe against the selected backend at construction time and raises `ValueError` otherwise (`tensorrt_llm/visual_gen/args.py`, `_validate_quant_attention_config`).

| Backend | `qk_dtype` | `v_dtype` | `(q, k, v)` block sizes | Common name | Supported hardware |
|---|---|---|---|---|---|
| `TRTLLM` | `int8` | `fp8` | `(1, 1, 1)`, `(1, 4, 1)`, `(1, 16, 1)` | SageAttention (INT8 QK) | Blackwell (SM100) |
| `TRTLLM` | `fp8` | `fp8` | `(1, 1, 1)`, `(1, 4, 1)` | SageAttention (FP8 QK) | Blackwell family (SM10X) |
| `CUTEDSL` | `bf16` | `fp8` | `(0, 0, 0)` | QK16PV8 | Blackwell family (SM10X) |
| `CUTEDSL` | `mxfp8` | `fp8` | `(0, 0, 0)`, `(0, 0, 1)` | MXFP8 Q/K | Blackwell family (SM10X) |
| `CUTEDSL` | `nvfp4` | `fp8` | `(0, 0, 0)`, `(0, 0, 1)` | NVFP4 Q/K | Blackwell family (SM10X) |
| `FLASHINFER` | `mxfp8` | `fp8` | `(0, 0, 0)` | MXFP8 Q/K | Blackwell family (SM10X) |
| `FLASHINFER` | `nvfp4` | `fp8` | `(0, 0, 0)` | NVFP4 Q/K | Blackwell family (SM10X) |
| `FLASHINFER` | `nvfp4` | `nvfp4` | `(0, 0, 0)` | NVFP4 Attention | RTX Blackwell family (SM12X) |

### Configuration Surface

| Field | Type | Default | Meaning |
|---|---|---|---|
| `qk_dtype` | `"bf16" \| "int8" \| "fp8" \| "mxfp8" \| "nvfp4"` | `"bf16"` | Q/K element format for BMM1. `bf16` leaves Q/K unquantized. |
| `v_dtype` | `"fp8" \| "nvfp4"` | `"fp8"` | V element format for BMM2. |
| `q_block_size` | int ≥ 0 | `0` | Q tokens per SageAttention quantization block. `0` outside SageAttention. |
| `k_block_size` | int ≥ 0 | `0` | K tokens per SageAttention quantization block. `0` outside SageAttention. |
| `v_block_size` | int ≥ 0 | `0` | V block size on the hidden dimension. `0` = one tensor-wide V scale; `1` = one scale per channel. |

Routing (`tensorrt_llm/_torch/visual_gen/attention_backend/utils.py`) forwards the validated `quant_attention_config` into the backend constructor: `TrtllmAttention` for `TRTLLM`, `FlashInferAttention` for `FLASHINFER`, and the dense `CuTeDSLAttention` FMHA backend for `CUTEDSL`.

## QK16PV8 (CUTEDSL)

**What it does.** Q and K stay in BF16 (or FP16), so BMM1 runs at full input precision. Only V is quantized to FP8 e4m3, so BMM2 runs on FP8 Tensor Cores.

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
            v_block_size=0,
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
    v_block_size: 0
```

## SageAttention (TRTLLM)

**What it does.** SageAttention quantizes all three tensors with fine-grained scales, so both BMM1 and BMM2 run in low precision:

- **Q and K** are quantized to INT8 or FP8 e4m3 with one scale per *token block* per head. The block size is `q_block_size` for Q and `k_block_size` for K, measured in tokens along the sequence axis; a larger K block amortizes more scales but is coarser.
- **V** is quantized to FP8 e4m3 with `v_block_size` elements per scale along the hidden dimension. All supported recipes use `v_block_size = 1`, i.e. one scale per head per channel.

**Requirements and behavior.**

- Blackwell GPU.
- The recommended `qk_dtype: "int8"` is only supported on `sm_100a`.
  - `sm_103a` can use `qk_dtype: "fp8"` but its accuracy could be worse than `qk_dtype: "int8"`.

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

## MXFP8 / NVFP4 (CUTEDSL / FlashInfer)

**What it does.** MXFP8 and NVFP4 quantize Q/K with fixed blocks of 32 and 16 elements, respectively. `q_block_size` and `k_block_size` remain `0` because MXFP8 and NVFP4 follows specialized block-scaling schema which divide into both token dimensions and channel dimensions. V uses FP8 on `CUTEDSL` and FlashInfer SM10X, or NVFP4 on FlashInfer SM12X.

**Requirements and behavior.**

- `CUTEDSL` requires a head dimension of 128 and supports `v_block_size` of `0` (tensor-wide scale) or `1` (per-head, per-channel scale).
- FlashInfer SM100/SM103 requires a head dimension of 128 and supports MXFP8 or NVFP4 Q/K with FP8 V.
- FlashInfer SM120/SM121 supports NVFP4 Q/K/V and requires self-attention, equal Q/K/V shapes and head counts, a head dimension of 64 or 128, and a sequence length divisible by 128.

**Configuration.**

CUTEDSL with MXFP8 Q/K and FP8 V:

```python
from tensorrt_llm import VisualGenArgs
from tensorrt_llm.visual_gen import AttentionConfig, QuantAttentionConfig

args = VisualGenArgs(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    attention_config=AttentionConfig(
        backend="CUTEDSL",
        quant_attention_config=QuantAttentionConfig(
            qk_dtype="mxfp8",
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
    qk_dtype: mxfp8
    v_dtype: fp8
    q_block_size: 0
    k_block_size: 0
    v_block_size: 1
```

FlashInfer with NVFP4 Q/K/V:

```python
from tensorrt_llm import VisualGenArgs
from tensorrt_llm.visual_gen import AttentionConfig, QuantAttentionConfig

args = VisualGenArgs(
    model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    attention_config=AttentionConfig(
        backend="FLASHINFER",
        quant_attention_config=QuantAttentionConfig(
            qk_dtype="nvfp4",
            v_dtype="nvfp4",
            q_block_size=0,
            k_block_size=0,
            v_block_size=0,
        ),
    ),
)
```

```yaml
attention_config:
  backend: FLASHINFER
  quant_attention_config:
    qk_dtype: nvfp4
    v_dtype: nvfp4
    q_block_size: 0
    k_block_size: 0
    v_block_size: 0
```

## Interaction With Other Features

- **Linear-layer quantization** (`VisualGenArgs.quant_config`, e.g. FP8 block scales or NVFP4) is independent and can be combined with any attention recipe.
- **Sparse attention.** On `CUTEDSL`, quantized attention and Video Sparse Attention (VSA) are mutually exclusive and rejected by the validator. On `TRTLLM`, Skip Softmax uses the same backend and the SageAttention unit tests exercise the two together.
- **Parallelism.** SageAttention is covered by a multi-GPU Ulysses test (`tests/unittest/_torch/visual_gen/multi_gpu/test_ulysses_sage_attention.py`). The CuTe DSL dense backend produces LSE, so it also composes with Attention2D / Ring context parallelism; the TRTLLM Sage path does not expose LSE through this wrapper.
