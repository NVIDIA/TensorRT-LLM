<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# MiniMax-H3 Modeling V2

VisualGen loads the [V1 H3 family](../minimax_h3/) when `TLLM_VISUALGEN_MODELING2_H3` is unset or `0`. Set it to `1` before starting the Python process or server to use V2. Set it on every worker for a distributed run. Both implementations use the same checkpoint identifiers, VisualGen API and serving configuration.

From the repository root, set `H3_CHECKPOINT` to your base H3 checkpoint directory and run the example.

```bash
TLLM_VISUALGEN_MODELING2_H3=1 python examples/visual_gen/models/minimax_h3.py \
    --model "$H3_CHECKPOINT" \
    --visual_gen_args examples/visual_gen/configs/minimax-h3-bf16-1gpu.yaml
```

## How a checkpoint selects a pipeline

The VisualGen loader identifies the H3 family and resolves the checkpoint directory. With V2 enabled, the [family registry](pipeline_registry.py) reads `fastvideo_inference.json` there and selects the target before constructing the transformer. Check the fully qualified target in the loader log when validating a run.

| Checkpoint contract | V2 target | Generation defaults |
| --- | --- | --- |
| Base H3 | [default_pipeline/](default_pipeline/) | Both implementations use 50 sigma points including the terminal zero, giving 49 joint audio/video updates. |
| FastH3 manifest with a `fasth3-` schema version | [fasth3/](fasth3/) | The checkpoint supplies the schedule and VSA configuration. The FastH3 8-Step V2 release uses nine points and eight updates in both implementations. |

## Where a change belongs

| Change | Owner |
| --- | --- |
| Base workflow or reference conditioning, TP text encoder | `default_pipeline/` |
| FastH3 trained schedule and selection of gated VSA for its transformer | `fasth3/` |
| Joint denoising and decode, packed-row layout, checkpoint loading shared by both targets | [pipeline.py](pipeline.py), [packing.py](packing.py), [weights.py](weights.py) |
| H3 attention layer, projections, head layout, normalization and attention policy | [components/](components/) and [ops/](ops/) |
| H3 tensor sharding, Ulysses projection order and stream synchronization, VAE tile placement | [parallel/](parallel/) |
| An interchangeable communication or attention operation used across families | The [Modeling V2 catalog](../../../modeling_v2/catalog/). A communication operation selects a compatible kernel for its tensor and layout contract. |
| A kernel specific to H3 | [kernels/](kernels/) or the target that consumes it |

For example, H3 issues asynchronous Ulysses transfers in V, Q, K order so one projection can overlap a preceding transfer. The family orders the existing prepare, push and barrier operations on its CUDA streams and joins them before attention. The [Ulysses communication operations](../../../modeling_v2/catalog/comm/ulysses.py) select eligible fused layout kernels and provide the general exchange and layout paths. Another model can reuse an operation and compose a different attention layer. Base H3 and FastH3 can share an H3 component when its inputs, layout and semantics agree. See the [LLM Modeling V2 guide](../../../modeling_v2/README.md).

The Python-registered H3 V2 Triton operations use the `trtllm::minimax_h3_v2_` prefix because both implementations share a process-wide PyTorch dispatcher. For a new native operation, follow the [Modeling V2 native code guide](../../../../../cpp/tensorrt_llm/modeling_v2/README.md) for its family or target namespace.

For a deployment-specific pipeline, add a sibling target directory named `<checkpoint-and-quantization>__<sm>__<parallel>/`. Add its match to the family registry under the relevant checkpoint contract. Unmatched deployments continue to use that checkpoint's functional target. Keep code used by sibling targets at family scope.

## Testing and rollout

Put family and target [unit tests](../../../../../tests/unittest/_torch/visual_gen/models/minimax_h3_v2/) under the path that mirrors the implementation. Keep a separate [whole-pipeline quality case](../../../../../tests/integration/defs/visual_gen/minimax_h3_v2/) for each target. A family-registry change exercises both targets and both settings of the V2 switch. A catalog change also exercises the operation's tests and its other consumers.

Compare V1 and V2 with the same checkpoint revision, seed, input, precision, backend, topology and explicit inference-step count. Use `50` on both sides for the base-H3 quality reference (49 updates). Also exercise an omitted step count separately to verify each version's default.

This rollout keeps V1 available for direct comparison. Retire V1 in a subsequent change after base H3 and FastH3 pass functional and audio/video quality checks and warmed latency and peak-memory comparisons against V1 on representative deployments.
