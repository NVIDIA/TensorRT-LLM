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
cuMemcpyBatchAsync, and every other call to the batched copy kernel; use_copy_engine_for_host_copy()
reports the decision for a task size. The knob is read once per process, so the shapes below sit on
both sides of the default instead of toggling the environment variable, and every expectation that
depends on the path asks the predicate first. Needs one GPU.
"""

import os
from random import randbytes

import pytest
import torch

copy_helpers = pytest.importorskip(
    "tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils",
    reason="the device copy helpers are C++-backend only",
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MemToMemTask = copy_helpers.MemToMemTask

DEFAULT_DMA_MIN_BYTES = 4 << 20

# (num_bytes, num_tasks). Kernel path at the default: 884,736 B attention pages, one request's
# onboard batch of 140 and 300 (more than the 256 tasks of one launch), and 300 x 16 B. Copy-engine
# path: one 15,261,696 B conv-state page, 12 of them in one batched call, one 217,055,232 B SSM-state
# page, and 3 x (4 MiB + 8 B), a size that is not a multiple of 16 and that only the copy engines
# accept.
SHAPES = [
    (884_736, 140),
    (884_736, 300),
    (16, 300),
    (15_261_696, 12),
    (15_261_696, 1),
    (217_055_232, 1),
    (DEFAULT_DMA_MIN_BYTES + 8, 3),
]


def _expected_threshold() -> int | None:
    """The threshold the C++ policy read at start-up: None when the path is disabled, 0 for any size."""
    raw = os.environ.get("TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES", "")
    if raw == "":
        return DEFAULT_DMA_MIN_BYTES
    try:
        value = int(raw, 10)
    except ValueError:
        # The policy logs a warning and keeps the default.
        return DEFAULT_DMA_MIN_BYTES
    return None if value < 0 else value


def _pinned(num_bytes: int) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True)


def _device(num_bytes: int) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, device="cuda")


def _random_payload(num_bytes: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(randbytes(num_bytes)), dtype=torch.uint8)


def _tasks(dst: torch.Tensor, src: torch.Tensor, num_bytes: int, num_tasks: int) -> list:
    """One task per slot, the slot layout of a slot pool: base + slot * slot_size."""
    return [
        MemToMemTask(dst=dst.data_ptr() + i * num_bytes, src=src.data_ptr() + i * num_bytes)
        for i in range(num_tasks)
    ]


def _check(result: int) -> None:
    assert result == 0, f"copy helper returned CUresult {result}"


def _round_trip(num_bytes: int, num_tasks: int) -> None:
    """Pinned host -> device -> pinned host through the helpers, byte-exact."""
    total = num_bytes * num_tasks
    stream = torch.cuda.Stream()
    src = _pinned(total)
    dst = _pinned(total)
    dev = _device(total)
    payload = _random_payload(total)
    src.copy_(payload)
    dst.zero_()
    host_to_device = _tasks(dev, src, num_bytes, num_tasks)
    device_to_host = _tasks(dst, dev, num_bytes, num_tasks)

    _check(copy_helpers.copy_host_to_device(host_to_device, num_bytes, stream.cuda_stream))
    _check(copy_helpers.copy_device_to_host(device_to_host, num_bytes, stream.cuda_stream))
    # The copies are stream-ordered on the stream they were given, so this is the only wait needed.
    stream.synchronize()
    assert torch.equal(dst, payload), f"{num_tasks} x {num_bytes} B round trip is not byte-exact"


def test_policy_follows_the_environment() -> None:
    """The predicate mirrors the threshold this process started with.

    A host that registers pinned memory in chunks reports the kernel for every size instead, so that
    answer is accepted as well.
    """
    selected = copy_helpers.use_copy_engine_for_host_copy
    sizes = [16, 884_736, DEFAULT_DMA_MIN_BYTES - 1, DEFAULT_DMA_MIN_BYTES, 15_261_696, 217_055_232]
    actual = [selected(n) for n in sizes]
    threshold = _expected_threshold()
    if threshold is None:
        assert not any(actual), actual
        return
    expected = [n >= threshold for n in sizes]
    assert actual == expected or not any(actual), (actual, expected)


@pytest.mark.parametrize(("num_bytes", "num_tasks"), SHAPES)
def test_round_trip(num_bytes: int, num_tasks: int) -> None:
    if num_bytes % 16 != 0 and not copy_helpers.use_copy_engine_for_host_copy(num_bytes):
        # The kernel path requires a multiple of 16 and rejects the call before enqueuing anything.
        tasks = [MemToMemTask(dst=0x1000, src=0x2000)] * num_tasks
        with pytest.raises(Exception, match="multiple of 16"):
            copy_helpers.copy_host_to_device(
                tasks, num_bytes, torch.cuda.current_stream().cuda_stream
            )
        return
    _round_trip(num_bytes, num_tasks)


def test_size_that_is_not_a_multiple_of_16() -> None:
    """24 B: the kernel's check throws before any CUDA work, the copy engines accept the size."""
    num_bytes = 24
    if copy_helpers.use_copy_engine_for_host_copy(num_bytes):
        _round_trip(num_bytes, 3)
        return
    # Rejected before the addresses are used, so they need not be real.
    tasks = [MemToMemTask(dst=0x1000, src=0x2000)] * 3
    for helper in (copy_helpers.copy_host_to_device, copy_helpers.copy_device_to_host):
        with pytest.raises(Exception, match="multiple of 16"):
            helper(tasks, num_bytes, torch.cuda.current_stream().cuda_stream)


def test_consumer_stream_sees_copy_through_event() -> None:
    """A stream that waits on an event recorded after the copy must observe the copied bytes.

    That is the only completion contract the storage manager uses: it records a ready event on the
    copy stream and makes consumers wait on it, never on the host.
    """
    num_bytes = 15_261_696  # copy engines at the defaults, kernel on chunked-registration hosts
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    src = _pinned(num_bytes)
    dst = _pinned(num_bytes)
    dev = _device(num_bytes)
    dev_check = _device(num_bytes)
    payload = _random_payload(num_bytes)
    src.copy_(payload)
    dst.zero_()

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
    consumer.synchronize()
    assert torch.equal(dst, payload)
