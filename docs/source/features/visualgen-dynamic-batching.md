<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# VisualGen Dynamic Batching

VisualGen can merge compatible queued text-to-image requests into one pipeline
execution. Pipelines batch compatible computation while preserving any
numerically sensitive request boundaries; each caller receives its own output
and request ID. Requests run through all
denoising steps together. New requests join the next pipeline call.

This can improve throughput for concurrent image generation, particularly when
a single image does not keep the GPU busy. Larger batches consume more memory,
and a collection window adds latency under light traffic. Choose limits using
measurements on the target model, resolution and GPU.

## Configuration

Use `VisualGenArgs.batching_config` in Python or the same field in
`--visual_gen_args` YAML. `max_batch_size=1` disables merging by default.
`max_queue_delay_ms=0` merges only requests already available when the worker
collects its next batch. A positive delay lets the worker wait briefly for peers;
the full batch starts immediately. The delay bounds collection time, not total
queueing or generation latency.

An example serving config is
[`dynamic_batching.yml`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/visual_gen/serve/configs/dynamic_batching.yml):

```bash
trtllm-serve black-forest-labs/FLUX.1-dev \
  --visual_gen_args examples/visual_gen/serve/configs/dynamic_batching.yml
```

Submit independent asynchronous requests before waiting for their results:

```python
from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams

args = VisualGenArgs(
    batching_config={"max_batch_size": 4, "max_queue_delay_ms": 5},
)
with VisualGen("black-forest-labs/FLUX.1-dev", args=args) as generator:
    handles = [
        generator.generate_async(prompt, VisualGenParams(seed=seed))
        for prompt, seed in [("A red fox", 42), ("A blue bird", 17)]
    ]
    for index, handle in enumerate(handles):
        handle.result().save(f"image_{index}.png")
```

Requests can share a batch when their resolution and sampling settings match;
prompt text and seeds may differ. Queue order is preserved, so an incompatible
request ends the current batch. Different resolutions run in separate batches.
The cap counts generated images, including prompt lists and multiple images per
prompt. A single request larger than the cap still runs independently.

FLUX.1 and FLUX.2 text-to-image support batching. Qwen-Image supports unquantized,
single-GPU execution with `VANILLA` attention. Unsupported configurations,
image-edit requests and denoising-cache configurations run independently.
FLUX requests also run independently when a transformer layer uses dynamic
NVFP4 activation scales, which would otherwise depend on other requests in a batch.
Each request keeps its seed and its own outputs, although batched computation
can introduce small floating-point differences.

## Measuring the benefit

Use the [serving benchmark](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/BENCHMARKING_VISUAL_GEN.md)
to compare `max_batch_size=1` with batching enabled. Keep the model, precision,
attention, compilation settings and workload identical. Warm the measured
shapes before comparing throughput, mean/P99 end-to-end latency and peak device
memory. Include varied prompt lengths and more than one batch size, and keep
generated samples for quality checks. Per-request generation timings describe
the shared pipeline call; do not sum them as GPU busy time.

[Regression tests](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/integration/defs/visual_gen/test_dynamic_batching.py)
cover output quality and the request-merging path. A measured improvement on
one workload does not imply a speedup for every model, resolution or load level.
