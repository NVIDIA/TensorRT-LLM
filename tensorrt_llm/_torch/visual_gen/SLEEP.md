<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental Pipeline Sleep

The internal single-GPU MiniMax-H3 pipeline can retain persistent GPU allocations
in host RAM while idle. This is process-local offload, not a durable snapshot:
the process, host memory, CUDA context, and node must remain alive.

Enable it explicitly at load time:

```python
pipeline = PipelineLoader(args).load(sleep_restore_mode="PINNED")
# Waits for an active forward call before releasing its weights.
pipeline.sleep()
pipeline.wake_up()
```

`CPU` selects pageable host backing; `PINNED` selects pinned host backing. Both
reuse TensorRT-LLM's virtual-memory allocator and restore the original device
addresses. Reserve host RAM for the captured allocation pool in addition to
ordinary process memory. The allocator retains host backing after wake, so this
RAM budget is required for the lifetime of the pipeline after its first sleep.

Loading captures persistent allocations; inference and warmup allocations are
outside that pool. Sleep also empties PyTorch's unused caching-allocator blocks,
but does not release the CUDA context or guarantee that all device memory is
free. Live outputs and allocations from other users of the same process remain
outside this pipeline's unique allocation tag.

Calls are idempotent in their completed state. Sleep closes admission, waits for
the active `forward` call (including calls through `infer`), and synchronizes CUDA
before releasing memory. New generation is rejected while sleep is pending,
asleep, or waking. Sleep and wake transitions are serialized per instance.
Overlapping generation calls are rejected on sleep-enabled pipelines; they are
not queued. Calling sleep or wake from inside a generation callback raises
instead of waiting on itself. A failed or interrupted transition disables further
generation; restart the worker rather than attempting to reuse partially restored
allocations.

## Scope

- Disabled by default; no changes to the text LLM executor or shared allocator.
- Sleep state and controls belong only to MiniMaxH3Pipeline; BasePipeline and
  other model implementations are unchanged.
- H3 already supports only one GPU and rejects CUDA graphs, CPU stage offload,
  and cache acceleration, independently of sleep.
- Sleep additionally rejects runtime LoRA configurations because that combination
  has not been validated. Loading without sleep retains its existing behavior.
- Internal direct-pipeline API only. Public VisualGen, request-queue draining,
  HTTP endpoints, and Dynamo lifecycle integration are not implemented.
- Loading must finish before the pipeline is shared. Callers still manage their
  external request queues and must not bypass `forward` to access model weights
  during sleep/wake. The pipeline is not a concurrent request scheduler.

## Tests and Validation

Unit tests are in `tests/unittest/_torch/visual_gen/test_sleep.py`; small real-GPU
allocator tests are in `test_sleep_gpu.py` alongside it, including two independent
allocation pools alternating sleep/wake with CPU and PINNED backing. The
checkpoint-backed test is
`tests/integration/defs/visual_gen/test_minimax_h3_sleep.py`. Set
`LLM_MODELS_ROOT` (or `MINIMAX_H3_CHECKPOINT` for the local checkpoint override).
The full-model test needs a GPU with at least 140 GiB and sufficient host RAM
for H3 plus its allocation backup; the development run reserves 320 GiB.
The state/loader unit tests run in CPU CI, and the small GPU allocator tests run
in A10 CI. The checkpoint-backed test is listed in
`tests/integration/test_lists/qa/llm_function_core.txt`, not in pre-merge CI.
Use a large-memory single GPU with the host-RAM budget above for that test.
The QA list also selects the non-H3 regressions below.
`test_sleep_non_h3.py` creates tiny local Wan/FLUX checkpoints and exercises
real default loading, weight materialization, transformer forwards, and rejection
of unsupported sleep requests without downloading pretrained checkpoints.
Full-model coverage currently exercises text-to-video/audio; keyframe-conditioned
generation and non-FP8 model configurations have not been validated with sleep.

The sleep test checks lifecycle integrity: post-wake video and audio must match
the same pipeline's pre-sleep output exactly. This is not an independent
golden-output quality comparison. H3's existing Diffusers-reference LPIPS and
audio tests in `tests/integration/defs/examples/visual_gen/test_minimax_h3_e2e.py`
are separately registered in B200 CI and do not enable sleep.
