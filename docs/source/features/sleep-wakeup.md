<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime Memory Checkpointing (Prototype)

TensorRT-LLM can temporarily release selected GPU allocations while keeping an
LLM runtime process alive. This enables a different GPU workload to run while
the inference runtime is parked.

Release can preserve allocation contents in host memory and free their GPU
physical memory. Resume allocates GPU physical memory again, maps it at the
same virtual addresses, and restores the saved contents. The virtual addresses
remain reserved while the runtime is parked. This is an in-process memory
checkpoint: it is not written to disk and does not survive process termination
or support restart in a different process.

```{warning}
Sleep and wakeup are prototype APIs and are subject to change. They are
available only with the PyTorch backend. AutoDeploy is not supported.
```

## Python API

Enable runtime memory checkpointing with `SleepConfig`. Both synchronous
`LLM` and asynchronous `AsyncLLM` expose `release()`, `resume()`, and
`get_memory_status()`.

```python
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import ExecutorMemoryType, SleepConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    sleep_config=SleepConfig(),
)
llm.release()
try:
    # The selected GPU physical memory is available to another workload.
    ...
finally:
    llm.resume()
```

With no tags, `release()` selects every usable `ExecutorMemoryType` and
`resume()` restores every tag still parked. You can release or restore a
specific subset:

```python
tags = [
    ExecutorMemoryType.MODEL_ENGINE_MAIN,
    ExecutorMemoryType.MODEL_WEIGHTS_MAIN,
]
llm.release(tags)
llm.resume([ExecutorMemoryType.MODEL_WEIGHTS_MAIN])

status = llm.get_memory_status()
assert status.state == "parked"
assert status.parked_tags == [ExecutorMemoryType.MODEL_ENGINE_MAIN]

llm.resume()  # Restore the remaining parked tag and reopen request admission.
```

Tags can be enum members or their string values. TensorRT-LLM normalizes them,
removes duplicates while retaining their first-seen order, and rejects an
explicitly empty list. The two `_no_capture_init_*` enum values are
initialization-only markers and cannot be selected.

The asynchronous API has the same arguments and results:

```python
from tensorrt_llm import AsyncLLM
from tensorrt_llm.llmapi import SleepConfig

async with AsyncLLM(model="meta-llama/Llama-3.1-8B-Instruct",
                    sleep_config=SleepConfig()) as llm:
    await llm.release()
    status = await llm.get_memory_status()
    await llm.resume()
```

`release()` first closes request admission and drains work already in flight.
Generation remains rejected until all parked tags have been restored. The
synchronous MPI/local and asynchronous Ray paths use the same admission states:

| State | Meaning |
| --- | --- |
| `running` | Requests are admitted and no tags are parked |
| `parking` | Admission is closed and release is in progress |
| `parked` | At least one tag remains released |
| `waking` | A restore is in progress |
| `failed` | A memory operation might have partially mutated runtime state |

Distributed status replies must agree. A disagreement is reported as a runtime
failure instead of choosing one worker's view.

Retries are idempotent. Releasing tags that are already parked or resuming tags
that are already active succeeds without repeating a memory operation.
Releasing additional tags while already parked is rejected; resume the runtime
first. New operations are also rejected during transitional and failed states.

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

- Invalid or explicitly empty tag lists raise `ValueError`.
- Release and resume require `sleep_config`.
- Calls synchronize CUDA work before changing mappings and before returning.
- Distributed operations coordinate all ranks. An error after ranks begin
  changing memory can put the worker in a terminal failed state because a
  partially changed distributed state cannot be safely reconciled.
- A host-memory backup is usable only by the same live runtime process.
- Put `resume()` in a `finally` block if the LLM must remain usable when the
  caller's intervening work fails.

## Control-plane HTTP endpoints

`trtllm-serve` can expose authenticated runtime-control routes for the
PyTorch HTTP text-generation server. The feature is opt-in and requires the
secret to come from the environment:

```bash
export TRTLLM_RUNTIME_CONTROL_API_KEY='replace-with-a-long-random-secret'
trtllm-serve meta-llama/Llama-3.1-8B-Instruct --enable_sleep_mode
```

If the YAML configuration has no `sleep_config`, the command creates a default
`SleepConfig`. An explicitly configured `sleep_config` is preserved. Sleep
mode is rejected for AutoDeploy and gRPC.

The enabled routes are:

```text
POST /release_memory
POST /resume_memory
GET  /memory_status
```

The POST body is optional. An omitted body or `{}` selects the defaults
described in the Python API. To select specific resources, send
`{"tags":["model","model_weights"]}`.

Every request must include `x-trtllm-runtime-control-auth` containing
`sha256=<hex digest>`, where the digest is HMAC-SHA256 of the exact transmitted
body. Sign an empty byte string for a GET or a body-less POST. This Python
example signs and sends matching bytes:

```python
import hashlib
import hmac
import json
import os

import requests

key = os.environ["TRTLLM_RUNTIME_CONTROL_API_KEY"].encode()


def auth(body: bytes) -> dict[str, str]:
    digest = hmac.new(key, body, hashlib.sha256).hexdigest()
    return {"x-trtllm-runtime-control-auth": f"sha256={digest}"}


body = json.dumps(
    {"tags": ["model", "model_weights"]},
    separators=(",", ":"),
).encode()
response = requests.post(
    "http://localhost:8000/release_memory",
    data=body,
    headers={"content-type": "application/json", **auth(body)},
)
response.raise_for_status()

status = requests.get(
    "http://localhost:8000/memory_status",
    headers=auth(b""),
)
print(status.json())

resume_body = json.dumps({"tags": ["model_weights"]},
                         separators=(",", ":")).encode()
requests.post(
    "http://localhost:8000/resume_memory",
    data=resume_body,
    headers={"content-type": "application/json", **auth(resume_body)},
).raise_for_status()
```

A status response has this shape:

```json
{
  "state": "parked",
  "parked_tags": ["model"]
}
```

Successful POST requests return `{"status":"success"}`. Invalid tags return
HTTP 400, conflicting transitions return 409, authentication failures return
401, and unexpected worker failures return 500.

Treat the control key as an administrative secret and expose these routes only
to a trusted control plane. Coordinate traffic draining before release when
clients can race with the operation, and wait for `running` before resuming
normal traffic. Host memory must remain available until restore completes.

The older `enable_rl_control_endpoints`, `x-trtllm-rl-control-auth`, and
`/update_weights` surface remains as a deprecated compatibility path for
custom `AsyncLLM` integrations. It cannot be enabled together with generic
runtime control, and `--enable_sleep_mode` never exposes `/update_weights`.

## Implementation overview

At model creation, eligible allocations enter tagged CUDA virtual-memory pools.
Sleep selects allocations by tag, optionally copies contents to CPU or pinned
storage, unmaps them, and calls `cuMemRelease` on their CUDA physical allocation
handles. Wakeup calls `cuMemCreate`, remaps new handles at the reserved virtual
addresses, and restores or initializes contents according to `SleepConfig`.

Principal implementation locations are:

- Synchronous API: `tensorrt_llm/llmapi/llm.py`
- Asynchronous API: `tensorrt_llm/_torch/async_llm.py`
- Serving and authentication: `tensorrt_llm/serve/openai_server.py` and
  `tensorrt_llm/serve/runtime_control_auth.py`
- Workers: `tensorrt_llm/executor/base_worker.py` and
  `tensorrt_llm/executor/ray/gpu_worker.py`
- Allocation tagging: `tensorrt_llm/_torch/pyexecutor/py_executor_creator.py`
  and `tensorrt_llm/_torch/pyexecutor/model_loader.py`
- Python VMM wrapper: `tensorrt_llm/_torch/virtual_memory.py`
- CUDA VMM: `cpp/tensorrt_llm/runtime/virtualMemory.cpp` and
  `cpp/include/tensorrt_llm/runtime/virtualMemory.h`
- GMS adapter: `tensorrt_llm/_torch/memory/gpu_memory_backend.py`
