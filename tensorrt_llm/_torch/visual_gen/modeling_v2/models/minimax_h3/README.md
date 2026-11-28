<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developing MiniMax-H3 with Modeling V2

Base H3 and FastH3 run on Modeling V2 through the standard VisualGen API, CLI and checkpoint names. [VisualGen Modeling V2 developer guide](../../README.md)

## Run a target

Set `H3_CHECKPOINT` to the checkpoint directory and run from the repository root.

```bash
python examples/visual_gen/models/minimax_h3.py \
    --model "$H3_CHECKPOINT" \
    --visual_gen_args examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml
```

| Checkpoint | V2 target | Denoising schedule |
| --- | --- | --- |
| Base H3 | [h3_pipeline/](h3_pipeline/) supports BF16, FP8, block-scaled FP8 and NVFP4. | The default 50 sigma points include zero and give 49 updates. |
| FastH3 | [fasth3/](fasth3/) uses the checkpoint's learned compression gate and video sparse attention. | The released eight-step checkpoint supplies nine sigma points. |

The [H3 router](pipeline_registry.py) checks `fastvideo_inference.json`. A `fasth3-` schema selects FastH3, and an ordinary base checkpoint selects the base pipeline. Each worker logs its target in `AutoPipeline: Creating`. The selected pipeline builds the transformer.

## Attention and parallelism

The [H3 attention component](components/attention.py) owns the projections and normalization weights. Its [Q/K norm and RoPE op](ops/rope.py) takes projected Q/K, weights and rotation tables. The [op test](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/models/minimax_h3/ops/test_rope.py) compares Q and K with a PyTorch reference. Changing which channels rotate changes that reference. Tuning the [Triton kernel](kernels/fused_rope.py) for the same rotation uses the existing reference.

```text
TP       project local heads -> attention on those heads -> output projection and sum
Ulysses  local tokens -> exchange Q/K/V -> attention on all tokens for local heads -> return exchange
```

For TP, attention assigns heads and [Linear](components/linear.py) owns the projection shards and output reduction. The [Ulysses component](components/ulysses_attention.py) coordinates both exchanges around attention. Its async path starts transfers as each projection becomes ready.

The family also decides which layers and denoising steps use a sparse prediction. The [catalog SOL op](../../catalog/attention/sol_predictor.py) takes prepared Q/K/V and calls the [existing predictor](../../../attention/backends/sparse/sol/kernels.py). Apply the same [V1 dependency review](../../README.md#allowed-dependencies) to new providers.

## Test a change

H3 [unit tests](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/models/minimax_h3/) and [checkpoint quality tests](../../../../../../tests/integration/defs/visual_gen/modeling_v2/minimax_h3/) mirror family and target ownership. Use the [catalog tests](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/catalog/) for model-independent ops. Record shared backend imports in the [dependency test](../../../../../../tests/unittest/_torch/visual_gen/modeling_v2/models/minimax_h3/test_dependencies.py). Map native dependencies that CI cannot infer in the [VisualGen CI rule](../../../../../../jenkins/scripts/cbts/cbts/rules/visual_gen.py).

A generation run can verify that the requested parallel or attention path produced the expected media. A lifecycle feature needs its own check. The [base H3 sleep/wake test](../../../../../../tests/integration/defs/visual_gen/modeling_v2/minimax_h3/h3_pipeline/test_minimax_h3_sleep.py) checks request admission, memory release and restored video and audio. For a loading fix, check the weights against the checkpoint and repeat the original request.
