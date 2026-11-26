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
"""cublas_mm on a new thread must run on that thread's PyTorch cuBLAS handle.

The test reads device-wide free memory, so it lives in a directory that most GPU stages run
serially. Where other workers share the GPU, their allocations can fail an attempt, which the
retries absorb, and their frees can hide a regression in an attempt.
"""

import threading

import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

# A cuBLAS handle that cublas_mm created for itself would take tens of MiB of device memory
# (64 MiB on H100 with cuBLAS 13.8). The measured call's workspace and output come from PyTorch's
# cache and do not lower free memory.
_MAX_FREE_MEMORY_DROP = 16 << 20
# Allocated and freed before each attempt, so that the caching allocator holds enough free memory
# for the new thread's PyTorch cuBLAS workspace and TRT-LLM's GEMM workspace (32 MiB each).
_CACHE_REFILL_BYTES = 256 << 20
_ATTEMPTS = 3
_TIMEOUT_S = 120


def _free_device_memory():
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def test_cublas_mm_on_new_thread_creates_no_cublas_handle():
    """A thread that starts late, like the serving thread after KV cache sizing, gets no cuBLAS
    handle from cublas_mm beyond its PyTorch one."""
    x = torch.randn((16, 1024), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16)
    # Loads the GEMM kernels, which happens once per process, outside the measurement.
    torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=None)

    drops, errors = [], []
    release = threading.Event()

    def measure(done):
        try:
            # Creates this thread's PyTorch cuBLAS handle and workspace and runs a first GEMM on
            # it, all before measuring.
            torch.cuda.current_blas_handle()
            torch.matmul(x, w.t())
            free_before = _free_device_memory()
            torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=None)
            drops.append(free_before - _free_device_memory())
        except BaseException as error:  # Re-raised on the test thread below.
            errors.append(error)
        finally:
            done.set()
        # An exiting thread frees its thread-local state, which would show up in a later
        # attempt's measurement.
        release.wait()

    threads = []
    try:
        for _ in range(_ATTEMPTS):
            # Allocated and freed at once.
            torch.empty(_CACHE_REFILL_BYTES, dtype=torch.uint8, device="cuda")
            done = threading.Event()
            thread = threading.Thread(target=measure, args=(done,), daemon=True)
            threads.append(thread)
            thread.start()
            assert done.wait(_TIMEOUT_S), "the measuring thread did not finish"
            if errors:
                raise errors[0]
            if drops[-1] < _MAX_FREE_MEMORY_DROP:
                break
    finally:
        release.set()
        for thread in threads:
            thread.join(_TIMEOUT_S)
    assert min(drops) < _MAX_FREE_MEMORY_DROP, (
        f"cublas_mm on a new thread lowered free device memory by "
        f"{[drop >> 20 for drop in drops]} MiB (limit {_MAX_FREE_MEMORY_DROP >> 20} MiB); "
        "it likely created a cuBLAS handle of its own"
    )
