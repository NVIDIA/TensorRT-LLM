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

"""KVCM2 objects can be destroyed on a thread that has no current CUDA context.

Python decides which thread runs a destructor: the one that drops the last reference, or, for an
object kept alive by a reference cycle, whichever thread's allocation triggers the cyclic garbage
collector -- a pytest-timeout or watchdog timer thread, for example. Such a thread has never
touched CUDA, so it has no current context, and every driver call the teardown makes (recording
the finish event, freeing device memory) used to fail with CUDA_ERROR_INVALID_CONTEXT and poison
KVCM2 for the rest of the process.

Each test builds the object on the main thread, finalizes it on a fresh thread that never calls
into CUDA, and checks that the latch stays clear and the thread still has no context afterwards.
The conftest's check_kvcm2_not_poisoned fixture also fails a test that poisons KVCM2.
"""

import ctypes
import gc
import threading

import pytest
import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    Batch,
    BufferConfig,
    CudaStream,
    GpuCacheTierConfig,
    KVCacheManager,
    KVCacheManagerConfig,
    poison_reason,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

JOIN_TIMEOUT_S = 60.0


@pytest.fixture(scope="module", autouse=True)
def initialize_cuda_context() -> None:
    torch.empty(1, device="cuda")


def _driver_current_context() -> int:
    """The calling thread's current CUDA context handle (0 when it has none)."""
    libcuda = ctypes.CDLL("libcuda.so.1")
    ctx = ctypes.c_void_p()
    assert libcuda.cuCtxGetCurrent(ctypes.byref(ctx)) == 0
    return ctx.value or 0


def _make_config() -> KVCacheManagerConfig:
    return KVCacheManagerConfig(
        tokens_per_block=4,
        cache_tiers=[GpuCacheTierConfig(quota=8 << 20)],
        layers=[AttentionLayerConfig(layer_id=0, buffers=[BufferConfig(role="key", size=4096)])],
    )


def _finalize_on_contextless_thread(holder: list) -> None:
    """Drop the only references in `holder` on a new thread that never calls into CUDA.

    The thread checks that it has no context before the drop (otherwise the test proves nothing)
    and that the teardown did not leave one current behind.
    """
    observed: dict = {}

    def run() -> None:
        observed["before"] = _driver_current_context()
        holder.clear()
        gc.collect()
        observed["after"] = _driver_current_context()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(JOIN_TIMEOUT_S)
    assert not thread.is_alive(), f"teardown thread did not finish in {JOIN_TIMEOUT_S}s"
    assert observed["before"] == 0, "the teardown thread already had a CUDA context"
    assert observed["after"] == 0, "teardown left a CUDA context current on the thread"


def _assert_not_poisoned() -> None:
    assert poison_reason() is None, poison_reason()


def _active_cache(manager: KVCacheManager, stream: torch.cuda.Stream, num_tokens: int = 16):
    cache = manager.create_kv_cache(None, list(range(num_tokens)))
    assert cache.resume(CudaStream(stream.cuda_stream))
    assert cache.resize(num_tokens)
    return cache


def test_kv_cache_and_manager_destroyed_without_cuda_context() -> None:
    """An open, active KvCache and its manager freed together, as a leaked test fixture is."""
    stream = torch.cuda.Stream()
    manager = KVCacheManager(_make_config())
    cache = _active_cache(manager, stream)
    torch.cuda.synchronize()
    holder = [cache, manager]
    del cache, manager

    _finalize_on_contextless_thread(holder)

    _assert_not_poisoned()
    # The process can still build and use a manager afterwards.
    manager = KVCacheManager(_make_config())
    cache = _active_cache(manager, stream)
    cache.close()
    manager.shutdown()


def test_kv_cache_destroyed_without_cuda_context() -> None:
    """Only the KvCache is freed on the other thread; the manager stays usable."""
    stream = torch.cuda.Stream()
    manager = KVCacheManager(_make_config())
    try:
        cache = _active_cache(manager, stream)
        torch.cuda.synchronize()
        holder = [cache]
        del cache

        _finalize_on_contextless_thread(holder)

        _assert_not_poisoned()
        cache = _active_cache(manager, stream)
        cache.close()
    finally:
        manager.shutdown()


def test_batch_destroyed_without_cuda_context() -> None:
    """A published Batch owns device memory and readiness events of its own."""
    stream = torch.cuda.Stream()
    manager = KVCacheManager(_make_config())
    try:
        cache = _active_cache(manager, stream)
        batch = Batch(manager, max_rows=2, max_blocks=8)
        batch.add(cache)
        batch.publish(stream.cuda_stream)
        batch.record_read(stream.cuda_stream)
        torch.cuda.synchronize()
        holder = [batch]
        del batch

        _finalize_on_contextless_thread(holder)

        _assert_not_poisoned()
        cache.close()
    finally:
        manager.shutdown()
