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
"""Round trips through the KVCacheManagerV2 host<->GPU copy helpers on both sides of the copy-engine policy.

copy_host_to_device / copy_device_to_host (kvCacheManagerV2Utils.cu) send a call whose tasks are at
least TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES each (default 4 MiB) to the copy engines, as one
cuMemcpyBatchAsync (one memcpy per task on the legacy NULL stream, which the batched call does not
accept), and every other call to the batched copy kernel; use_copy_engine_for_host_copy() reports the
decision for a task size. The knob is read once per process, so the shapes below sit on both sides of
the default instead of toggling the environment variable, and the expected path is computed here from
the environment and the kernel version instead of being taken from the predicate under test. Needs one
GPU.
"""

import os
import platform
import re

import pytest
import torch

copy_helpers = pytest.importorskip(
    "tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils",
    reason="requires the tensorrt_llm C++ bindings (kv_cache_manager_v2_utils is not importable)",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MemToMemTask = copy_helpers.MemToMemTask

DEFAULT_DMA_MIN_BYTES = 4 << 20

# (num_bytes, num_tasks). Kernel path at the default: 884,736 B attention pages, one request's
# onboard batch of 140 and 300 (more than the 256 tasks of one launch), and 300 x 16 B. Copy-engine
# path: 44 pages of 4 MiB in one batched call (44 descriptors), one 15,261,696 B conv-state page, one
# 217,055,232 B SSM-state page, and 3 x (4 MiB + 8 B), a size that is not a multiple of 16 and that
# only the copy engines accept.
SHAPES = [
    (884_736, 140),
    (884_736, 300),
    (16, 300),
    (DEFAULT_DMA_MIN_BYTES, 44),
    (15_261_696, 1),
    (217_055_232, 1),
    (DEFAULT_DMA_MIN_BYTES + 8, 3),
]


@pytest.fixture(autouse=True, scope="module")
def _release_pinned_host_cache():
    """The raw copies leave pinned blocks in torch's host cache; return them when the module is done."""
    yield
    empty_host_cache = getattr(torch.cuda, "host_empty_cache", None) or getattr(
        torch._C, "_host_emptyCache", None
    )
    if empty_host_cache is not None:
        empty_host_cache()


def _host_registers_in_chunks() -> bool:
    """Whether this host pins the tier in 2 GiB chunks, which disables the copy-engine path.

    Mirrors HostMem::shouldUseChunkedRegistration() in hostMem.cpp (Linux 6.11 to 6.13); keep the
    list in step with that function.
    """
    return platform.release()[:4] in ("6.11", "6.12", "6.13")


def _expected_threshold() -> int | None:
    """What hostCopyDmaMinBytes() returns for this environment: None when the path is disabled.

    Mirrors the C++ parser (strtoll with an end-pointer check): unset, empty, non-integer and
    out-of-range values fall back to the default, a negative value means never.
    """
    raw = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES", "")
    if re.fullmatch(r"\s*[+-]?[0-9]+", raw) is None:
        return DEFAULT_DMA_MIN_BYTES
    value = int(raw)
    if not -(1 << 63) <= value < (1 << 63):
        return DEFAULT_DMA_MIN_BYTES
    return None if value < 0 else value


def _expected_copy_engine(num_bytes: int) -> bool:
    """useCopyEngineForHostCopy(num_bytes), computed without calling the predicate under test."""
    threshold = _expected_threshold()
    return not _host_registers_in_chunks() and threshold is not None and num_bytes >= threshold


def _pinned(num_bytes: int) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)


def _device(num_bytes: int) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, device="cuda")


def _random_pinned(num_bytes: int) -> torch.Tensor:
    """Random bytes generated straight into pinned memory: one allocation, no host-side staging copies."""
    info = torch.iinfo(torch.int64)
    words = (num_bytes + 7) // 8
    buf = torch.empty(words, dtype=torch.int64, pin_memory=True).random_(info.min, info.max)
    return buf.view(torch.uint8)[:num_bytes]


def _tasks(dst: torch.Tensor, src: torch.Tensor, num_bytes: int, num_tasks: int) -> list:
    """One task per slot, the slot layout of a slot pool: base + slot * slot_size."""
    return [
        MemToMemTask(dst=dst.data_ptr() + i * num_bytes, src=src.data_ptr() + i * num_bytes)
        for i in range(num_tasks)
    ]


def _check(result: int) -> None:
    assert result == 0, f"copy helper returned CUresult {result}"


def _round_trip(num_bytes: int, num_tasks: int, stream_handle: int | None = None) -> None:
    """Pinned host -> device -> pinned host through the helpers, byte-exact.

    stream_handle None means a fresh non-default stream; 0 is the legacy NULL stream.
    """
    total = num_bytes * num_tasks
    stream = torch.cuda.Stream() if stream_handle is None else None
    handle = stream.cuda_stream if stream is not None else stream_handle
    src = _random_pinned(total)
    dst = _pinned(total)
    dev = _device(total)
    dst.zero_()
    host_to_device = _tasks(dev, src, num_bytes, num_tasks)
    device_to_host = _tasks(dst, dev, num_bytes, num_tasks)
    try:
        _check(copy_helpers.copy_host_to_device(host_to_device, num_bytes, handle))
        _check(copy_helpers.copy_device_to_host(device_to_host, num_bytes, handle))
    finally:
        # torch records no event for the raw driver copies, so drain before the pinned and device
        # tensors return to its caches, also when a check above fails.
        if stream is not None:
            stream.synchronize()
        else:
            torch.cuda.synchronize()
    assert torch.equal(dst, src), f"{num_tasks} x {num_bytes} B round trip is not byte-exact"


def test_policy_follows_the_environment() -> None:
    """The predicate is exactly 'no chunked registration and num_bytes >= threshold' on every host."""
    selected = copy_helpers.use_copy_engine_for_host_copy
    sizes = [16, 884_736, DEFAULT_DMA_MIN_BYTES - 1, DEFAULT_DMA_MIN_BYTES, 15_261_696, 217_055_232]
    for num_bytes in sizes:
        assert selected(num_bytes) == _expected_copy_engine(num_bytes), num_bytes


@pytest.mark.parametrize(("num_bytes", "num_tasks"), SHAPES)
def test_round_trip(num_bytes: int, num_tasks: int) -> None:
    if num_bytes % 16 != 0 and not _expected_copy_engine(num_bytes):
        # The kernel path requires a multiple of 16 and rejects the call before enqueuing anything,
        # so the addresses need not be real.
        tasks = [MemToMemTask(dst=0x1000, src=0x2000)] * num_tasks
        with pytest.raises(RuntimeError, match="multiple of 16"):
            copy_helpers.copy_host_to_device(
                tasks, num_bytes, torch.cuda.current_stream().cuda_stream
            )
        return
    _round_trip(num_bytes, num_tasks)


def test_size_that_is_not_a_multiple_of_16() -> None:
    """24 B: the kernel's check throws before any CUDA work, the copy engines accept the size."""
    num_bytes = 24
    if _expected_copy_engine(num_bytes):
        _round_trip(num_bytes, 3)
        return
    # Rejected before the addresses are used, so they need not be real.
    tasks = [MemToMemTask(dst=0x1000, src=0x2000)] * 3
    for helper in (copy_helpers.copy_host_to_device, copy_helpers.copy_device_to_host):
        with pytest.raises(RuntimeError, match="multiple of 16"):
            helper(tasks, num_bytes, torch.cuda.current_stream().cuda_stream)


def test_null_stream_round_trips() -> None:
    """Stream 0 takes one memcpy per task on the copy-engine path and a plain launch on the kernel path."""
    _round_trip(15_261_696, 3, stream_handle=0)


def test_consumer_stream_sees_copy_through_event() -> None:
    """A stream that waits on an event recorded after the copy must observe the copied bytes.

    That is the only completion contract the storage manager uses: it records a ready event on the
    copy stream and makes consumers wait on it, never on the host.
    """
    num_bytes = (
        15_261_696  # copy engines at the default threshold, kernel on chunked-registration hosts
    )
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    src = _random_pinned(num_bytes)
    dst = _pinned(num_bytes)
    dev = _device(num_bytes)
    dev_check = _device(num_bytes)
    dst.zero_()

    try:
        _check(
            copy_helpers.copy_host_to_device(
                _tasks(dev, src, num_bytes, 1), num_bytes, producer.cuda_stream
            )
        )
        ready = torch.cuda.Event()
        ready.record(producer)
        consumer.wait_event(ready)
        # GPU -> GPU on the consumer, then GPU -> host, all stream-ordered behind the event.
        _check(
            copy_helpers.copy_device_to_device(
                _tasks(dev_check, dev, num_bytes, 1), num_bytes, consumer.cuda_stream
            )
        )
        _check(
            copy_helpers.copy_device_to_host(
                _tasks(dst, dev_check, num_bytes, 1), num_bytes, consumer.cuda_stream
            )
        )
    finally:
        # Both streams carry raw driver copies torch knows nothing about; drain them before the
        # tensors go back to the caches, also when a check above fails.
        torch.cuda.synchronize()
    assert torch.equal(dst, src)
