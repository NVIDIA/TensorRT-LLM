<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisualGen Sparse Attention

```{note}
This page is an unindexed draft until the VisualGen documentation hub is
introduced. The sections below are drafts for follow-up work.
```

Sparse attention in VisualGen is configured through
`VisualGenArgs.attention_config.sparse_attention_config`. The user-facing config
stays in VisualGen args or model config. Checkpoint calibration metadata remains
internal and is lowered into per-attention-backend `SparseParams` when each
attention module is constructed.

## Skip Softmax

Skip Softmax is a kernel-level sparse attention method. It skips low-contribution
attention blocks inside the attention kernel and does not introduce a separate
cache manager or model-level predictor.

### Python API

```python
from tensorrt_llm.visual_gen import (
    AttentionConfig,
    SkipSoftmaxAttentionConfig,
    VisualGen,
    VisualGenArgs,
)

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

### YAML API

```yaml
attention_config:
  backend: TRTLLM
  sparse_attention_config:
    algorithm: skip_softmax
    threshold_scale_factor: 5000.0
    initial_disabled_steps: 2
```

### Fields

`threshold_scale_factor` is the raw scalar threshold consumed by the kernel. It
takes precedence over `target_sparsity`.

`target_sparsity` is a semantic target in `[0, 1]`. It requires a calibration
formula from checkpoint `config.json`.

`disabled_layers` is a list of fnmatch patterns. A matching attention module
gets no skip-softmax `SparseParams`. Patterns match both full module names and
component-relative names, so `blocks.0.attn1` matches
`transformer.blocks.0.attn1` and `transformer_2.blocks.0.attn1`.

`initial_disabled_steps` disables skip-softmax for the first N denoising steps of
each request, then enables the configured threshold. It is based on the ordinal
denoising step, not the scheduler timestep value.

### Checkpoint Metadata

VisualGen auto-detects skip-softmax calibration from `sparse_attention_config` in
checkpoint `config.json`. Diffusers checkpoints with multiple transformer
components read each component's own `config.json`.

Single-model `config.json`:

```json
{
  "sparse_attention_config": {
    "threshold_scale_factor": {
      "formula": "exp(log_a + b * target_sparsity)",
      "coefficients": {
        "log_a": -14.14,
        "b": 36.64
      }
    },
    "disabled_layers": ["blocks.0.attn1"]
  }
}
```

Multi-model diffusers checkpoints keep calibration per component:

```text
checkpoint/
  model_index.json
  transformer/config.json
  transformer_2/config.json
```

When both user config and checkpoint metadata are present, checkpoint metadata
supplies formulas per model component. The public config supplies runtime knobs
such as `target_sparsity`, `disabled_layers`, and `initial_disabled_steps`. The
public config object does not store formulas or component sub-configs.

### CUDA Graphs

`initial_disabled_steps` creates two sparse-attention phases when it is positive:
the initial disabled phase and the enabled phase. VisualGen includes that phase
in CUDA graph keys so graph capture does not reuse a graph across different
skip-softmax settings.

## Video Sparse Attention (VSA)
TODO
