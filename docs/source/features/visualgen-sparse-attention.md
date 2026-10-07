# VisualGen Sparse Attention (Beta)

```{note}
This feature is in **beta** stage. APIs, supported models, and optimization options are actively evolving and may change in future releases.
```

- [Overview](#overview)
  - [Algorithms](#algorithms)
- [Skip Softmax Attention](#skip-softmax-attention)
- [SOL Attention](#sol-attention)
- [Video Sparse Attention (VSA)](#video-sparse-attention-vsa)

## Overview

Visual generation models naturally operate on long image or video token sequences. Each denoising step is closer to a full-context prefill pass than to autoregressive decoding, and attention can dominate runtime for high-resolution image generation or long video generation.

Sparse attention in VisualGen is configured through `VisualGenArgs.attention_config.sparse_attention_config`. The user-facing config stays in VisualGen args or model config, while `attention_config.backend` selects the kernel family. Algorithms produce their block-sparse routes through the core `block_sparse_attn_predict` hook: a backend either predicts inside that hook from the flattened Q/K/V, or predicts before the core forward and hands the complete `BlockSparseForwardInputs` through `AttentionForwardArgs.sparse_backend_args`, which the default hook passes through. The VisualGen TRTLLM wrapper owns the timestep schedule those backends consult: it prepares the denoising timestep once per eager call and answers whether a layer runs sparse for it (dense layers and dense timestep phases do not). `SparseRuntimeParams` is the single lowered runtime carrier passed as `AttentionForwardArgs.sparse_runtime_params`; its optional `block_sparse_inputs` field nests the algorithm-neutral routes for the general block-sparse FMHA. `None` means prediction has not run, while an empty `SparseRuntimeParams()` records that prediction ran without a sparse payload.

### Algorithms

| `algorithm` | Config class | Status |
|---|---|---|
| `skip_softmax` | `SkipSoftmaxAttentionConfig` | Supported |
| `vsa` | `VideoSparseAttentionConfig` | Supported (`CUTEDSL`, `TRTLLM`) |
| `sol_attn` | `SolAttentionConfig` | Supported (`TRTLLM`, `CUTEDSL`) |

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

TRT-LLM imports NumExpr only when it needs to consume a checkpoint formula. During package bootstrap, TRT-LLM defaults `NUMEXPR_NUM_THREADS` to `1` without overriding an explicit environment setting. This evaluates the scalar formulas without creating a NumExpr worker pool while allowing applications with substantial NumExpr work to opt into parallel evaluation. This setting controls only NumExpr and does not replace workload-specific OpenMP tuning. Applications that import NumExpr before TRT-LLM must configure it before process startup.

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

Skip Softmax Attention works with both the **TRTLLM** and **CUTEDSL** attention backends in VisualGen. Set `attention_config.backend` to either when enabling it. On CUTEDSL, Skip Softmax Attention can also be combined with `quant_attention_config`'s block-scaled Q/K recipes (MXFP8, NVFP4); VSA and SOL cannot be combined with quantized attention on any backend.

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

`disabled_until_timestep` creates two sparse-attention phases when it is set: the high-timestep disabled phase and the enabled phase after the cutoff. VisualGen includes that phase in CUDA graph keys so graph capture does not reuse a graph across different Skip Softmax Attention settings. See [VisualGen CUDA Graphs](visualgen-cuda-graph.md) for the general capture and replay design.

Graphs are captured lazily. The first denoising step seen for a given tensor shape and sparse-attention phase captures a graph; later steps with the same shape and phase replay that graph. When denoising crosses the cutoff, the phase key changes, so VisualGen captures a second graph for the enabled phase instead of replaying the graph from the disabled phase.

## SOL Attention

SOL ([arXiv:2607.24027](https://arxiv.org/abs/2607.24027)) routes attention
blocks dynamically: blocks whose scores stand out are computed exactly, the
rest are approximated from compact K/V proxies and folded back into the online
softmax. VisualGen serves it on two backends behind one `SolAttentionConfig`:

- **TRTLLM** runs SOL in two stages. A TRT-LLM-owned predictor first produces
  an exact block bitmask and K/V proxy summaries from Q/K/V; the shared
  `PrimsTSBlockSparseFmha` library then executes that route from the shared
  `SparseRuntimeParams`. `SOLTrtllmAttention` is only the VisualGen bridge: it
  overrides the core `block_sparse_attn_predict` hook, so prediction runs
  inside the core forward from the flattened Q/K/V, the batch layout in the
  attention metadata and the `timestep` in the forward arguments; dense layers
  and dense timestep phases return no routes. The VisualGen wrapper compacts
  fused projection split views once and shares those tensors between
  prediction and the block-sparse FMHA. Cross-attention, context parallelism,
  attention quantization and unsupported tensor envelopes raise an error
  instead of silently falling back to dense attention.
- **CUTEDSL** runs the fused kernel vendored from the reference implementation
  (`SOLCuTeDSLAttention`), which folds routing, sparse computation and the
  approximation correction into one online-softmax pass. It runs on datacenter
  Blackwell, sm100 (B200/GB200) and sm103 (B300/GB300), and requires
  `head_dim=128`, bfloat16 and MHA (`num_kv_heads == num_heads`). On an input
  the kernel is known not to serve, such as an unsupported architecture, a
  `head_dim` other than 128 or a non-bfloat16 dtype, it runs dense attention
  instead (the CuTe DSL FMHA where available, torch SDPA otherwise), logs the
  reason once and counts the fallback; set `TRTLLM_SOL_ATTN_STRICT=1` to raise
  instead, which is useful when benchmarking to confirm the kernel actually
  ran. Errors raised by the kernel itself are not caught. Cross-attention and
  masked calls are delegated to dense attention per call.

Configure SOL with `SolAttentionConfig` and either backend:

```python
from tensorrt_llm.visual_gen import AttentionConfig, SolAttentionConfig

attention_config = AttentionConfig(
    backend="TRTLLM",  # or "CUTEDSL"
    sparse_attention_config=SolAttentionConfig(
        tau=1.0,
        disabled_until_timestep=0.6,
        dense_layers=[0, 2, 3, 4],
    ),
)
```

The equivalent YAML is:

```yaml
attention_config:
  backend: TRTLLM                 # or CUTEDSL
  sparse_attention_config:
    algorithm: sol_attn
    tau: 1.0                      # routing threshold; higher tau routes more blocks sparse
    disabled_until_timestep: 0.6  # dense while the normalized timestep >= cutoff
    dense_layers: [0, 2, 3, 4]    # optional: layer indices forced dense
    thresh_type: diag             # block threshold policy: "diag" or "exact"
```

- `tau` is the routing threshold in standard deviations above the mean block
  score; higher values route more blocks sparse.
- `disabled_until_timestep` has the same meaning as it does for Skip Softmax
  Attention: attention runs dense while the normalized denoising timestep is at
  or above the cutoff, protecting the high-noise prefix, and switches to SOL
  below it. Use `None` rather than `0.0` to disable the prefix.
- `dense_layers` lists zero-based layer indices that always use dense
  attention.
- `thresh_type` selects how the routing threshold models the key blocks:
  `diag` treats every key channel independently, `exact` uses the full key
  covariance. Both backends implement both policies from the same per-block
  statistics.

The TRTLLM envelope is full-mask BF16 self-attention on SM100 or SM103 with 4-D
BSHD Q/K/V tensors, equal Q/K/V shapes and head dimension 128. The predictor is
one graph-visible operator that allocates its route and summary tensors per
call, so it composes with CUDA Graph capture and with torch.compile, including
`fullgraph`, without keeping any state between calls.

When a cutoff is configured, VisualGen includes the dense-or-sparse phase in
the CUDA Graph key. The TRTLLM attention metadata reduces the timestep to a host
value during graph warmup, keeps it in the component attention state and reuses
it during capture for every timestep-scheduled algorithm (Skip Softmax Attention
and SOL), while the SOL predictor's outputs are allocated inside the captured
graph and replayed with it; the CuTeDSL backends read the
phase the CUDA Graph runner resolved for the graph key instead of the device
tensor. Per-token timesteps, such as Wan I2V where the conditioning frame stays
at timestep zero, reduce to their largest live value, so the schedule stays
dense until every token is below the cutoff. A dense capture therefore cannot
be reused for the sparse phase.

## Video Sparse Attention (VSA)

VSA reduces the cost of self-attention in video diffusion models by attending only to the most relevant spatial-temporal blocks. It uses a two-branch design: a lightweight coarse branch mean-pools tokens into (4, 4, 4) cubes and computes cube-level attention scores to select the top-K most relevant cubes per query, then a fine branch runs the selected backend's block-sparse kernel over those cubes only. The two outputs are blended with learned gates, which is why VSA needs a fine-tuned checkpoint. Select either `CUTEDSL` for the CuTe DSL fine-stage kernel or `TRTLLM` for PrimTS block-sparse attention.

Requirements:

- A VSA-fine-tuned checkpoint such as [`FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers`](https://huggingface.co/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers). Standard Wan checkpoints do not carry the learned VSA gates.
- The `CUTEDSL` or `TRTLLM` attention backend. `CUTEDSL` uses the CuTe DSL fine-stage kernel; `TRTLLM` lowers the selected cubes through the generic PrimTS block-sparse FMHA contract.
- A supported CUDA device and tensor shape for the selected block-sparse kernel. When that kernel is unavailable or the input is outside its envelope, the fine branch runs the compact Q/K/V through that backend's dense path (SDPA for `CUTEDSL`, TRTLLM attention for `TRTLLM`) and the VSA post-processing still applies.
- VSA cannot be combined with `quant_attention_config`.
- VSA is not compatible with Ring attention or Attention2D (it does not produce per-split LSE). Ulysses is supported.

Configure VSA with `VideoSparseAttentionConfig` and either backend:

```python
from tensorrt_llm import VisualGenArgs
from tensorrt_llm.visual_gen.args import AttentionConfig, VideoSparseAttentionConfig

args = VisualGenArgs(
    model="FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
    attention_config=AttentionConfig(
        backend="TRTLLM",  # or "CUTEDSL"
        sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=0.9),
    ),
)
```

The equivalent YAML, for `--visual_gen_args` or `trtllm-serve`:

```yaml
attention_config:
  backend: TRTLLM                 # or CUTEDSL
  sparse_attention_config:
    algorithm: vsa
    vsa_sparsity: 0.90            # fraction of K/V cubes skipped by the fine branch
```

`vsa_sparsity` controls the fraction of K/V cubes the fine branch skips (0.0 keeps every cube and reproduces dense attention, 0.9 skips 90% of them). Higher sparsity gives more speedup at the cost of some quality.

VSA retains shape-dependent metadata and route tensors so CUDA Graph replay can reuse stable addresses. A pipeline instance keeps one set of these tensors per distinct shape profile for as long as the CUDA Graphs that reference them, so their footprint grows with the number of served resolution/frame profiles exactly like the graphs do.

Both VSA backends use the same VisualGen-owned predictor implementation, one
instance per attention layer, and identical post-processing. The `TRTLLM` path runs the coarse stage before the core
forward, hands the predicted `BlockSparseForwardInputs` (including the
tile-padding validity bits only the VSA predictor knows) through
`sparse_backend_args`, lets the default core prediction hook pass them to the
general block-sparse FMHA, and then blends the fine and coarse outputs. Its
compact dense fallback passes no sparse inputs, so the core runs dense attention
and VSA post-processing still runs. `CUTEDSL` retains only its backend-specific
fine-attention execution. The core FMHA registry owns the reusable block-sparse
implementation rather than a VSA-specific lifecycle.
