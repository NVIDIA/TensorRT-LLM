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

## Compatibility and request semantics

The engine preserves FIFO order. It stops collecting at an incompatible request
and retains that request for the next pipeline call. Sampling parameters must
match, including resolution, steps, guidance, text length limits, negative prompt,
per-prompt output count and model-specific extras. Prompt text and seeds may differ.
Explicit versus default parameter intent is preserved when checking compatibility.

The cap counts generated images across requests, including prompt lists and
`num_images_per_prompt`. An individual request exceeding the cap runs independently;
the cap does not change existing single-request limits. Each original request
retains its own random-number stream and noise tensor shape. Batched kernels can
still introduce small floating-point differences in the generated image.

Pipelines opt in through `supports_dynamic_batching` in their implementation.
Currently FLUX.1 and FLUX.2 text-to-image pipelines opt in. Qwen-Image opts in
for unquantized, single-GPU execution with the `VANILLA` attention backend.
Qwen preserves the original request shapes for text encoding, attention, text
MLPs and the final output projection; its large image projections and image
MLPs use the combined batch. This limits BF16 numerical drift that can amplify
across denoising steps. Other Qwen configurations and image-edit pipelines run
requests independently.
Unsupported pipelines, reference-conditioned requests, prepared inputs, and
requests with non-serializable extra parameters run independently. Enabling a
denoising cache through `cache_config` also keeps requests independent because
cache decisions can depend on the batch. An inactive pipeline/cache combination
logs a warning at worker startup.

Rank zero chooses the batch and broadcasts its request boundaries and seeds to
all participating ranks. Output splitting happens before IPC. A pipeline error
is delivered to every request in that batch. A caller timing out discards only
its own late response; it does not cancel its peers. Per-request generation and
phase timings describe the shared pipeline call, so do not sum them to estimate
GPU busy time. Existing client iteration statistics count dispatched requests
as in flight; they do not distinguish the worker's queue from its running batch.

[CUDA graphs](visualgen-cuda-graph.md) distinguish batch sizes through tensor-shape
keys. Qwen graph keys also include the original request sizes, text lengths and
mask presence. New shapes or request layouts can pay capture or compilation
cost and retain additional graph buffers. Warm the measured shapes and layouts
before collecting steady-state performance data.

## Validation and benchmarking

CPU scheduler, noise, IPC routing and failure-path tests live in
`tests/unittest/_torch/visual_gen/test_dynamic_batching.py`. The GPU test in
`tests/integration/defs/visual_gen/test_dynamic_batching.py` exercises the public
API and worker path, compares batched images to singleton runs and existing
FLUX goldens with LPIPS, and includes CUDA graph, torch.compile, multiple-output
and Qwen 50-step regression cases. Qwen request-shape, mask and graph-key tests live in
`tests/unittest/_torch/visual_gen/test_qwen_image_batching.py`.

For an A/B throughput comparison, run the existing serving benchmark once with
`max_batch_size=1` and once with the example config, using otherwise identical
model revision, precision, attention, compilation settings and workload:

```bash
python -m tensorrt_llm.serve.scripts.benchmark_visual_gen \
  --backend openai-images --port 8000 \
  --requests '[{"prompt":"A red fox","seed":42},{"prompt":"A blue bird","seed":17}]' \
  --num-requests 64 --width 512 --height 512 --num-inference-steps 28 \
  --max-concurrency 8 --request-rate inf \
  --save-result --save-detailed --result-filename dynamic-batching.json
```

Repeat the workload to exclude cold compilation/capture, then compare request
throughput, mean/P99 end-to-end latency and peak device memory. Use varied prompt
lengths and more than one batch size. Keep generated samples for quality checks;
this feature does not imply a measured speedup for every workload.
