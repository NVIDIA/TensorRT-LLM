<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sleep and Wakeup (Prototype)

TensorRT-LLM can temporarily release selected GPU allocations while keeping an
LLM instance alive. This supports GPU time-sharing between inference and
training workloads, such as reinforcement learning and agentic workflows.

Sleep can preserve allocation contents in host memory and release their GPU
physical memory. Wakeup allocates GPU physical memory again, maps it at the same
virtual addresses, and restores the saved contents. The virtual addresses
remain reserved while the instance sleeps.

```{warning}
Sleep and wakeup are prototype APIs and are subject to change. They are
available only with the PyTorch backend. AutoDeploy is not supported.
```

## Python API

The public API is asynchronous and is provided by `AsyncLLM`. Enable it with
`SleepConfig`, then call `release()` and `resume()` with the memory tags to
operate on.

```python
from tensorrt_llm import AsyncLLM
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType, SleepConfig

sleep_tags = [
    ExecutorMemoryType.MODEL_ENGINE_MAIN.value,
    ExecutorMemoryType.MODEL_WEIGHTS_MAIN.value,
]

async with AsyncLLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    sleep_config=SleepConfig(),
) as llm:
    # Block new requests, abort requests in flight, and wait for idle workers.
    await llm.pause_generation()

    await llm.release(sleep_tags)
    try:
        # The selected GPU physical memory is available to other workloads.
        ...
    finally:
        await llm.resume(sleep_tags)
        await llm.resume_generation()
```

`AsyncLLM` uses the Ray orchestrator. `release()` and `resume()` complete after
the operation completes on every worker. Use the same tags for both calls and
do not submit requests between them.

Calling `pause_generation()` first is recommended when another task can submit
requests concurrently. It blocks the `AsyncLLM` frontend, aborts requests in
flight, and drains the workers. Sleep also enters a worker control action to
ensure the engines are idle before allocations change.

The PyTorch workers also coordinate sleep and wakeup across tensor-parallel and
pipeline-parallel MPI ranks. `LLM` does not currently expose public synchronous
`release()` and `resume()` methods; do not rely on its private collective-RPC
interface.

## Memory tags

Only allocations captured by the TensorRT-LLM virtual-memory allocator and
registered with a requested tag are affected.

| `ExecutorMemoryType` | Tag value | Contents |
| --- | --- | --- |
| `MODEL_WEIGHTS_MAIN` | `model_weights` | Main-model parameters |
| `MODEL_ENGINE_MAIN` | `model` | Other main-model engine allocations |
| `MODEL_WEIGHTS_DRAFT` | `draft_model_weights` | Draft-model parameters |
| `MODEL_ENGINE_DRAFT` | `draft_model` | Other draft-model engine allocations |
| `KV_CACHE` | `kv_cache` | KV-cache allocations |
| `SAMPLER` | `sampler` | Sampling resources |
| `DRAFTER` | `drafter` | Drafting resources |
| `GUIDED_DECODER` | `guided_decoder` | Guided-decoding resources |
| `SPEC_RESOURCES` | `spec_resource_manager` | Speculative-decoding resources |
| `MODEL_EXTRA` | `model_extra` | Additional model resources |
| `EXTRA_RESOURCES` | `executor_extra` | Additional executor resources |

The enum also has `_no_capture_init_kv_cache` and
`_no_capture_init_extra_resources`. These mark initialization stages whose
allocations are intentionally not captured in sleep-enabled pools, so they are
not useful sleep targets.

Selecting `model` does not imply `model_weights`. Select both to release model
parameters and the other main-model engine allocations.

## Restore modes

`SleepConfig.restore_modes` controls how each tagged allocation is restored:

| Mode | Sleep behavior | Wakeup behavior |
| --- | --- | --- |
| `CPU` | Copy to pageable host RAM, then release GPU physical memory | Copy the host backup to new GPU memory |
| `PINNED` | Copy to pinned host RAM, then release GPU physical memory | Copy the pinned backup to new GPU memory |
| `NONE` | Release GPU physical memory without preserving contents | Map new memory without restoring old contents |
| `MEMSET` | Release GPU physical memory without a host copy | Map and zero new GPU memory |

For example, preserve model weights in pinned host memory and discard the KV
cache:

```python
sleep_config = SleepConfig(
    restore_modes={
        ExecutorMemoryType.MODEL_WEIGHTS_MAIN: "PINNED",
        ExecutorMemoryType.KV_CACHE: "NONE",
    }
)
```

The KV-cache restore mode defaults to `NONE`. Other unlisted tags default to
`PINNED` when the runtime prefers pinned backup memory, and otherwise to `CPU`.
Model weights are therefore preserved in host RAM by default. Host memory must
have enough capacity for all allocations using `CPU` or `PINNED`.

Enabling `sleep_config` changes model loading because TensorRT-LLM must create
tagged CUDA virtual-memory pools and route allocations through them. Enable it
only for instances that will use sleep and wakeup.

## GPU Memory Service weights

Weights loaded with `load_format="gms"` are managed by GPU Memory Service
(GMS), not by the TensorRT-LLM sleep virtual-memory allocator. They have no
`model_weights` or `draft_model_weights` sleep tag.

When a sleep request contains a weight tag:

- Ordinary virtual-memory-managed weights follow their restore mode, and their
  GPU physical memory is released.
- GMS-managed weights are skipped. They remain in the shared GMS GPU memory
  pool and are not copied to host RAM by TensorRT-LLM sleep.
- Other requested, non-GMS allocations remain eligible for release.

GPU-memory usage might therefore not fall by the full model-weight size with
GMS loading. GMS mappings and sessions have a separate lifecycle.

## Failure and lifecycle behavior

- Invalid tag strings raise `ValueError`.
- Sleep and wakeup require `sleep_config`.
- Calls synchronize CUDA work before changing mappings and before returning.
- Distributed operations coordinate all ranks. An error after ranks begin
  changing memory can put the worker in a terminal failed state because a
  partially changed distributed state cannot be safely reconciled.
- Put `resume()` in a `finally` block if the LLM must remain usable when the
  caller's intervening work fails.

## Control-plane HTTP endpoints

`OpenAIServer` can optionally register authenticated reinforcement-learning
control endpoints when its generator is an `AsyncLLM`:

```text
POST /release_memory
POST /resume_memory
```

Both accept:

```json
{
  "tags": ["model", "model_weights"]
}
```

They return `{"status": "success"}` after all workers complete. The routes are
disabled by default. A custom `OpenAIServer` integration must set
`enable_rl_control_endpoints=True`, provide `rl_control_api_key`, and place an
HMAC-SHA256 signature of the exact request body in the
`x-trtllm-rl-control-auth` header. The standard `trtllm-serve` command does not
currently expose these server settings.

The caller must stop generation traffic before releasing memory and resume
traffic only after memory is restored.

## Implementation overview

At model creation, eligible allocations enter tagged CUDA virtual-memory pools.
Sleep selects allocations by tag, optionally copies contents to CPU or pinned
storage, unmaps them, and calls `cuMemRelease` on their CUDA physical allocation
handles. Wakeup calls `cuMemCreate`, remaps new handles at the reserved virtual
addresses, and restores or initializes contents according to `SleepConfig`.

Principal implementation locations are:

- Python API: `tensorrt_llm/_torch/async_llm.py`
- Workers: `tensorrt_llm/executor/base_worker.py` and
  `tensorrt_llm/executor/ray/gpu_worker.py`
- Allocation tagging: `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`
  and `tensorrt_llm/_torch/pyexecutor/model_loader.py`
- Python VMM wrapper: `tensorrt_llm/_torch/virtual_memory.py`
- CUDA VMM: `cpp/tensorrt_llm/runtime/virtualMemory.cpp` and
  `cpp/include/tensorrt_llm/runtime/virtualMemory.h`
- GMS adapter: `tensorrt_llm/_torch/memory/gpu_memory_backend.py`
