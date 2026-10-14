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

copy_host_to_device / copy_device_to_host (kvCacheManagerV2Utils.cu) send a call of at most
TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MAX_TASKS tasks (default 8) of at least
TLLM_KV_CACHE_MANAGER_V2_HOST_COPY_DMA_MIN_BYTES each (default 4 MiB) to the copy engines and every
other call to the batched copy kernel; use_copy_engine_for_host_copy() reports the decision for a
shape. Both knobs are read once per process, so the shapes below sit on both sides of the defaults
instead of toggling the environment variables. Needs one GPU.
"""

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

# (num_bytes, num_tasks). Kernel path at the defaults: 884,736 B attention pages, one request's
# onboard batch of 140 and 300 (more than the 256 tasks of one launch), 300 x 16 B, and
# 12 x 15,261,696 B (large tasks that the task limit keeps on the kernel). Copy-engine path: one
# 15,261,696 B conv-state page, one 217,055,232 B SSM-state page, and 3 x (4 MiB + 8 B), a size that
# is not a multiple of 16 and that only the copy engines accept.
SHAPES = [
    (884_736, 140),
    (884_736, 300),
    (16, 300),
    (15_261_696, 12),
    (15_261_696, 1),
    (217_055_232, 1),
    (DEFAULT_DMA_MIN_BYTES + 8, 3),
]


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


def test_policy_follows_the_documented_defaults() -> None:
    """Below the size threshold or above the task limit the kernel is used; the predicate says which."""
    selected = copy_helpers.use_copy_engine_for_host_copy
    assert not selected(884_736, 1)
    assert not selected(DEFAULT_DMA_MIN_BYTES - 1, 1)
    # Large tasks, but more of them than the default task limit allows.
    assert not selected(15_261_696, 12)
    # The two state pages of one request, one task each, take the copy engines unless this host
    # registers pinned memory in chunks, in which case the predicate is false for every shape.
    assert selected(15_261_696, 1) == selected(217_055_232, 1)


@pytest.mark.parametrize(("num_bytes", "num_tasks"), SHAPES)
def test_round_trip(num_bytes: int, num_tasks: int) -> None:
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

    if num_bytes % 16 != 0 and not copy_helpers.use_copy_engine_for_host_copy(num_bytes, num_tasks):
        # The kernel path requires a multiple of 16 and rejects the call before enqueuing anything.
        with pytest.raises(Exception, match="multiple of 16"):
            copy_helpers.copy_host_to_device(host_to_device, num_bytes, stream.cuda_stream)
        return

    _check(copy_helpers.copy_host_to_device(host_to_device, num_bytes, stream.cuda_stream))
    _check(copy_helpers.copy_device_to_host(device_to_host, num_bytes, stream.cuda_stream))
    # The copies are stream-ordered on the stream they were given, so this is the only wait needed.
    stream.synchronize()
    assert torch.equal(dst, payload), f"{num_tasks} x {num_bytes} B round trip is not byte-exact"


def test_kernel_path_rejects_a_size_that_is_not_a_multiple_of_16() -> None:
    """A size below the threshold always takes the kernel, whose check throws before any CUDA work."""
    num_bytes = 24
    assert not copy_helpers.use_copy_engine_for_host_copy(num_bytes, 3)
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
