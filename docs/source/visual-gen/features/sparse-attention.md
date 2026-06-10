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
    # Normalized transformer-forward timestep cutoff. Denoising starts near 1,
    # so skip-softmax stays disabled while timestep >= 0.6.
    disabled_until_timestep: 0.6
```

### Fields

`threshold_scale_factor` is the raw scalar threshold consumed by the kernel. It
takes precedence over `target_sparsity`.

`target_sparsity` is a semantic target in `[0, 1]`. It requires a calibration
formula from checkpoint `config.json`. If checkpoint metadata also provides
`target_sparsity`, the user config value wins.

`disabled_until_timestep` is a normalized `[0, 1]` transformer-forward timestep
cutoff. Denoising starts near 1 and moves toward 0, so skip-softmax is disabled
while `timestep >= disabled_until_timestep` and enabled after the timestep drops
below the cutoff. This keeps the API independent of the user-selected number of
denoising steps.

### Checkpoint Metadata

VisualGen auto-detects skip-softmax calibration from the ModelOpt-generated
`sparse_attention_config` in checkpoint `config.json`. Diffusers checkpoints
with multiple transformer components read each component's own `config.json`.

Single-model `config.json`:

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "algorithm": "skip_softmax",
        "ignore": [
          "blocks.0.attn1",
          "blocks.0.attn2"
        ],
        "disabled_until_timestep": 0.6,
        "threshold_scale_factor": {
          "formula": "a * exp(b * target_sparsity)",
          "prefill": {
            "a": 1443.4853294366435,
            "b": 4.303654042880227
          }
        },
        "target_sparsity": {
          "prefill": 0.5
        }
      }
    }
  }
}
```

The ModelOpt-generated `config_groups` block can contain many groups.
Skip-softmax scans groups whose `algorithm` is `"skip_softmax"`. `ignore`
carries checkpoint-provided fnmatch patterns for layers that should not receive
skip-softmax `SparseParams`. Patterns match both full module names and
component-relative names, so `blocks.0.attn1` matches
`transformer.blocks.0.attn1` and `transformer_2.blocks.0.attn1`. Calibration
defaults come from the first matching group with `threshold_scale_factor`.

Multi-model diffusers checkpoints keep calibration per component:

```text
checkpoint/
  model_index.json
  transformer/config.json
  transformer_2/config.json
```

When both user config and checkpoint metadata are present, checkpoint metadata
supplies formulas per model component. The public config can override runtime
knobs such as `target_sparsity` and `disabled_until_timestep`; layer-disable
patterns come from ModelOpt `config.json` `ignore`. The public config object
does not store formulas or component sub-configs.

### CUDA Graphs

`disabled_until_timestep` creates two sparse-attention phases when it is set:
the high-timestep disabled phase and the enabled phase after the cutoff.
VisualGen includes that phase in CUDA graph keys so graph capture does not reuse
a graph across different skip-softmax settings.

## Video Sparse Attention (VSA)
TODO
