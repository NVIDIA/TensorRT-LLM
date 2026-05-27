# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-rank (TP=2) sleep/wakeup tests for the MPI/IPC executor path.

Verifies that sleep() and wakeup() correctly release and restore GPU memory on
*all* ranks, not just rank-0.  Uses TinyLlama with tensor_parallel_size=2 so
the PyExecutor starts two MPI worker processes; the control-listener thread on
rank-1 is exercised by every sleep/wakeup call.

GPU memory assertions query usage per visible CUDA device (not just GPU 0),
so a rank-1 leak on GPU 1 will fail the assertion.
"""

import os

import psutil
import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType, SleepConfig

_LLAMA_MODEL_PATH = str(
    llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0")

_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

_SAMPLING_PARAMS = SamplingParams(temperature=0)


def _per_device_gpu_memory() -> dict:
    """Return ``{device_index: used_bytes}`` for the current process tree
    across all visible CUDA devices.

    Querying per device (rather than summing) lets callers assert that
    *each* participating GPU's memory changed; an aggregate sum can mask
    a rank-1 leak if rank-0 releases enough memory to compensate.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return {}

    root_pid = os.getpid()
    targets = frozenset(
        [root_pid] +
        [p.pid for p in psutil.Process(root_pid).children(recursive=True)])

    result: dict = {}
    device_count = pynvml.nvmlDeviceGetCount()
    for idx in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            result[idx] = sum(
                p.usedGpuMemory for p in procs if p.pid in targets)
        except Exception:
            result[idx] = 0
    return result


@pytest.mark.gpu2
def test_mpi_sleep_wakeup_tp2(process_gpu_memory_info_available):
    """Sleep/wakeup with TP=2 releases GPU memory on both ranks.

    Sequence:
    1. Generate prompts to populate KV cache and exercise the model.
    2. Measure active GPU memory across all devices (covers GPU 0 + GPU 1).
    3. Call sleep() -- should free model weights + KV cache on *all* ranks.
    4. Confirm each active GPU's memory has decreased.
    5. Call wakeup() -- re-materialises all tags.
    6. Confirm each active GPU's memory has recovered.
    7. Re-generate the same prompts; outputs must match pre-sleep results,
       proving that non-rank-0 VMM was correctly restored (a partial restore
       would corrupt forward-pass results or crash NCCL).
    """
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=16384)

    llm = LLM(
        model=_LLAMA_MODEL_PATH,
        tensor_parallel_size=2,
        sleep_config=SleepConfig(),
        kv_cache_config=kv_cache_config,
    )

    sleep_tags = [
        ExecutorMemoryType.MODEL_ENGINE_MAIN,
        ExecutorMemoryType.MODEL_WEIGHTS_MAIN,
    ]

    with llm:
        outputs_before = llm.generate(_PROMPTS, _SAMPLING_PARAMS)
        generated_before = [o.outputs[0].text for o in outputs_before]

        mem_active = _per_device_gpu_memory()

        llm._collective_rpc("sleep", (sleep_tags,))

        mem_sleep = _per_device_gpu_memory()
        if process_gpu_memory_info_available:
            for dev, active_bytes in mem_active.items():
                if active_bytes == 0:
                    continue  # device not in use by this job
                assert mem_sleep.get(dev, 0) < active_bytes, (
                    f"GPU {dev} memory did not decrease after sleep: "
                    f"active={active_bytes / 2**20:.1f} MiB, "
                    f"sleep={mem_sleep.get(dev, 0) / 2**20:.1f} MiB; "
                    f"rank assigned to GPU {dev} may not have called "
                    f"release_with_tag()"
                )

        llm._collective_rpc("wakeup", (sleep_tags,))

        mem_wakeup = _per_device_gpu_memory()
        if process_gpu_memory_info_available:
            for dev, sleep_bytes in mem_sleep.items():
                if mem_active.get(dev, 0) == 0:
                    continue
                assert mem_wakeup.get(dev, 0) > sleep_bytes, (
                    f"GPU {dev} memory did not recover after wakeup: "
                    f"sleep={sleep_bytes / 2**20:.1f} MiB, "
                    f"wakeup={mem_wakeup.get(dev, 0) / 2**20:.1f} MiB; "
                    f"rank assigned to GPU {dev} may not have called "
                    f"materialize_with_tag()"
                )

        outputs_after = llm.generate(_PROMPTS, _SAMPLING_PARAMS)
        generated_after = [o.outputs[0].text for o in outputs_after]

    for before, after in zip(generated_before, generated_after, strict=True):
        assert before == after, (
            f"Output mismatch after sleep/wakeup with TP=2.\n"
            f"  Before: {before!r}\n"
            f"  After:  {after!r}\n"
            "This indicates VMM was not correctly restored on all ranks."
        )


@pytest.mark.gpu2
def test_mpi_sleep_wakeup_kv_cache_only_tp2(process_gpu_memory_info_available):
    """Sleep/wakeup releasing only the KV cache on TP=2.

    Ensures that a partial tag set (KV_CACHE only) propagates correctly to
    rank-1's control listener and that generation still works after wakeup.
    """
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=16384)

    llm = LLM(
        model=_LLAMA_MODEL_PATH,
        tensor_parallel_size=2,
        sleep_config=SleepConfig(),
        kv_cache_config=kv_cache_config,
    )

    sleep_tags = [ExecutorMemoryType.KV_CACHE]

    with llm:
        outputs_before = llm.generate(_PROMPTS, _SAMPLING_PARAMS)
        generated_before = [o.outputs[0].text for o in outputs_before]

        llm._collective_rpc("sleep", (sleep_tags,))
        llm._collective_rpc("wakeup", (sleep_tags,))

        outputs_after = llm.generate(_PROMPTS, _SAMPLING_PARAMS)
        generated_after = [o.outputs[0].text for o in outputs_after]

    for before, after in zip(generated_before, generated_after, strict=True):
        assert before == after, (
            f"KV-cache-only sleep/wakeup corrupted output on TP=2.\n"
            f"  Before: {before!r}\n"
            f"  After:  {after!r}"
        )
