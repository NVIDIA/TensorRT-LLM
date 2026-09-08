# VisualGen Sparse Attention

```{note}
This page is an unindexed draft until the VisualGen documentation hub is introduced.
```

- [Overview](#overview)
  - [Algorithms](#algorithms)
- [Skip Softmax Attention](#skip-softmax-attention)
- [Video Sparse Attention (VSA)](#video-sparse-attention-vsa)

## Overview

Visual generation models naturally operate on long image or video token sequences. Each denoising step is closer to a full-context prefill pass than to autoregressive decoding, and attention can dominate runtime for high-resolution image generation or long video generation.

Sparse attention in VisualGen is configured through `VisualGenArgs.attention_config.sparse_attention_config`. The user-facing config stays in VisualGen args or model config. Checkpoint calibration metadata remains internal and is lowered into per-attention-backend `SparseParams` when each attention module is constructed.

### Algorithms

| `algorithm` | Config class | Status |
|---|---|---|
| `skip_softmax` | `SkipSoftmaxAttentionConfig` | Supported |
| VSA | TBD | TODO |

## Skip Softmax Attention

Skip Softmax Attention is a kernel-level method, also known as BLASST, that dynamically skips computation in a FlashAttention-style kernel. It can accelerate existing full-attention VisualGen models in a plug-and-play manner.

The value actually consumed by the kernel is **`threshold_scale_factor`**. The kernel combines it with the **sequence length** to compute the **threshold** at runtime. Other configuration paths resolve to that scalar before the attention backend is constructed.

### Checkpoint Config

[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt) can perform calibration and store metadata for Skip Softmax Attention in the model checkpoint's `config.json`. The checkpoint config provides the formula that maps `target_sparsity` to `threshold_scale_factor`.

This checkpoint config is **optional**. It is only required when using `target_sparsity`, which is a [0, 1] scalar that is more intuitive than directly choosing the kernel-facing `threshold_scale_factor`. `target_sparsity` only serves as guidance; the actual **achieved** sparsity in the kernel can vary.

Example checkpoint config:

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "algorithm": "skip_softmax",
        "threshold_scale_factor": {
          "formula": "a * exp(b * target_sparsity)",
          "coefficients": {
            "a": 1000.0,
            "b": 5.0
          }
        },
        "target_sparsity": 0.5,
        "disabled_until_timestep": 0.8,
        "ignore": [
          "blocks.0.attn1",
          "blocks.0.attn2"
        ]
      }
    }
  }
}
```

The checkpoint config may contain multiple `config_groups` for different sparse attention algorithms. At most one group may configure Skip Softmax Attention. Multiple groups whose `algorithm` is `skip_softmax` are invalid.

- `formula` — an **arbitrary** [numexpr](https://numexpr.readthedocs.io/) expression of `threshold_scale_factor` using `target_sparsity` and one or more named coefficients. Standard math functions such as `exp`, `log`, `sqrt`, `pow`, and `**` are available. The runtime parses and evaluates it directly, so calibration is not locked to a fixed functional form.
- `coefficients` — scalar coefficient values referenced by `formula`.
- `target_sparsity` — optional checkpoint-provided target value. User-provided `target_sparsity` overrides this checkpoint default.
- `disabled_until_timestep` — optional normalized `[0, 1]` transformer-forward timestep cutoff. Denoising starts near 1 and moves toward 0, so Skip Softmax Attention is disabled while `timestep >= disabled_until_timestep` and enabled after the timestep drops below the cutoff.
- `ignore` — optional fnmatch layer patterns where the calibrated Skip Softmax Attention config should not apply. Patterns match both full module names and component-relative names, so `blocks.0.attn1` matches `transformer.blocks.0.attn1` and `transformer_2.blocks.0.attn1`.

Diffusers checkpoints with multiple transformer components keep calibration per component:

```text
checkpoint/
  model_index.json
  transformer/config.json
  transformer_2/config.json
```

Each component reads its own `config.json`, so formulas and `ignore` patterns can differ between `transformer` and `transformer_2`.

### User Configuration

User configuration is supplied through Python or YAML and controls how the checkpoint metadata is consumed:

- Set `threshold_scale_factor` directly to pass a concrete threshold to the kernel. This does not require checkpoint calibration metadata.
- Set `target_sparsity` to request a sparsity target. The runtime resolves it to `threshold_scale_factor` using the checkpoint calibration formula. If the checkpoint does not provide the required Skip Softmax Attention metadata, the runtime raises an error.
- Set `disabled_until_timestep` to disable Skip Softmax Attention at the beginning of denoising. The cutoff is expressed in normalized scheduler time; the number of dense steps it produces depends on the scheduler and the number of inference steps.

`threshold_scale_factor` and `target_sparsity` are alternatives: if both are present, `threshold_scale_factor` takes precedence and the calibration formula is not used. User-provided `target_sparsity` and `disabled_until_timestep` override checkpoint defaults. Checkpoint `ignore` patterns always disable Skip Softmax Attention for matching layers.

Skip Softmax Attention works with both the **TRTLLM** and **CUTEDSL** attention backends in VisualGen. Set `attention_config.backend` to either when enabling it. On CUTEDSL, Skip Softmax Attention can also be combined with `quant_attention_config`'s block-scaled Q/K recipes (MXFP8, NVFP4); VSA is the only CUTEDSL sparse-attention algorithm that is mutually exclusive with quantized attention.

#### Mapping `disabled_until_timestep` to Actual Denoising Steps

VisualGen passes each transformer a normalized scheduler timestep `t` in `[0, 1]`. Denoising proceeds from high to low `t`. Skip Softmax Attention is disabled while `t >= disabled_until_timestep` and enabled once `t < disabled_until_timestep`. Equality therefore belongs to the dense phase.

For a scheduler sequence `t[0], ..., t[N-1]`, the number of initial dense-attention steps is the number of entries whose normalized timestep is greater than or equal to `disabled_until_timestep`:

```text
dense_steps = count(t[i] >= disabled_until_timestep)
skip_softmax_steps = N - dense_steps
```

The mapping must be computed from the actual scheduler sequence and can differ across models, schedulers, scheduler settings, and numbers of inference steps. `disabled_until_timestep` is not simply a fraction of `N` when the schedule is nonlinear. The following 40-step Wan 2.2 UniPC schedule is one example. Here, `s[i]` is the unshifted normalized flow sigma at denoising step `i`: it represents the base noise level before `flow_shift` is applied. The checkpoint uses 1,000 training timesteps and `flow_shift=3.0`:

```text
s[i] = 1 - i * (1 - 1/1000) / 40,              i = 0, ..., 39
shifted_sigma[i] = 3 * s[i] / (1 + 2 * s[i])
```

The flow shift concentrates timesteps near the high-noise end of the schedule. UniPC converts these shifted sigmas into the normalized runtime timestep sequence `t[i]`. Each row below is obtained from that actual sequence by evaluating `t[i] >= disabled_until_timestep` for all 40 steps and counting how many comparisons are true:

| `disabled_until_timestep` | Initial dense steps | Skip Softmax steps |
| :---: | ---: | ---: |
| `1.00` | 0 | 40 |
| `0.97` | 4 | 36 |
| `0.94` | 7 | 33 |
| `0.93` | 8 | 32 |
| `0.90` | 10 | 30 |
| `0.86` | 14 | 26 |

For example, the actual sequence has `t[13]` at approximately `0.861` and `t[14]` at approximately `0.848`. Therefore, `disabled_until_timestep=0.86` is true for `i=0, ..., 13`, giving 14 initial dense steps and 26 Skip Softmax steps.

VisualGen defines this cutoff using the **scheduler-derived normalized timestep** `t[i]`, rather than the **denoising-step index** `i`, so the same configuration interface works across models, schedulers, and different numbers of inference steps while preserving each scheduler's denoising trajectory.

This control is specific to iterative visual generation: the same attention layers run repeatedly while the denoising state changes, so early high-noise steps can remain dense before sparsity is enabled later.

#### Python API

```python
from tensorrt_llm.visual_gen import (
    AttentionConfig,
    SkipSoftmaxAttentionConfig,
    VisualGen,
    VisualGenArgs,
)

# Direct threshold:
args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention_config=AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
        ),
    ),
)

pipe = VisualGen(args)
```

```python
# Target sparsity (requires a calibrated checkpoint):
args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention_config=AttentionConfig(
        backend="TRTLLM",
        sparse_attention_config=SkipSoftmaxAttentionConfig(
            target_sparsity=0.5,
            disabled_until_timestep=0.6,
        ),
    ),
)
```

```python
# CUTEDSL backend:
args = VisualGenArgs(
    model="<path_or_hf_id>",
    attention_config=AttentionConfig(
        backend="CUTEDSL",
        sparse_attention_config=SkipSoftmaxAttentionConfig(
            threshold_scale_factor=5000.0,
        ),
    ),
)
```

#### YAML

```yaml
# Direct threshold:
attention_config:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 5000.0
```

```yaml
# Target sparsity (requires a calibrated checkpoint):
attention_config:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    target_sparsity: 0.5
    disabled_until_timestep: 0.6
```

```yaml
# CUTEDSL backend:
attention_config:
  backend: CUTEDSL
  sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 5000.0
```

### CUDA Graphs

`disabled_until_timestep` creates two sparse-attention phases when it is set: the high-timestep disabled phase and the enabled phase after the cutoff. VisualGen includes that phase in CUDA graph keys so graph capture does not reuse a graph across different Skip Softmax Attention settings. See [VisualGen CUDA Graphs](cuda-graph.md) for the general capture and replay design.

Graphs are captured lazily. The first denoising step seen for a given tensor shape and sparse-attention phase captures a graph; later steps with the same shape and phase replay that graph. When denoising crosses the cutoff, the phase key changes, so VisualGen captures a second graph for the enabled phase instead of replaying the graph from the disabled phase.

## Video Sparse Attention (VSA)

TODO
