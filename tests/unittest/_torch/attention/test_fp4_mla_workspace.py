# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side tests of the shared FP4 MLA decode scratch (CPU tensors, fake buffer pool)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fp4_mla import layout, packed_plan
from tensorrt_llm._torch.attention.backends.fp4_mla.config import _FP4_MLA_SPLIT_KV_ENV
from tensorrt_llm._torch.attention.backends.fp4_mla.state import Fp4MlaState

CPU = torch.device("cpu")


class _Block:
    def __init__(self, buffer: torch.Tensor, reserved: bool) -> None:
        self.buffer = buffer
        self.is_reserved = reserved


class _FakePool:
    """CPU stand-in for memory_buffer_utils.Buffers with its best-fit / reserve rules."""

    def __init__(self) -> None:
        self.buffers: dict[str, list[_Block]] = {}
        self.allocations = 0

    def get_buffer(self, shape, dtype, name, reserve):
        nbytes = dtype.itemsize
        for dim in shape:
            nbytes *= int(dim)
        best = None
        for block in self.buffers.get(name, []):
            if block.buffer.numel() >= nbytes and (
                best is None or block.buffer.numel() < best.buffer.numel()
            ):
                best = block
        if best is None:
            self.allocations += 1
            best = _Block(torch.empty((nbytes,), dtype=torch.uint8), reserve)
            self.buffers.setdefault(name, []).append(best)
        elif reserve:
            best.is_reserved = True
        return best.buffer[:nbytes].view(dtype).view(list(shape))


def _metadata(max_sequences: int = 128, *, pool=None) -> SimpleNamespace:
    return SimpleNamespace(
        fp4_mla_state=Fp4MlaState(),
        max_num_sequences=max_sequences,
        max_num_requests=max_sequences,
        is_cuda_graph=True,
        forward_scratch_buffers=pool,
        get_empty=lambda buffers, shape, dtype, name, capture_graph: buffers.get_buffer(
            shape, dtype, name, capture_graph
        ),
    )


@pytest.fixture
def pool(monkeypatch):
    fake = _FakePool()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return fake


def test_growth_past_capacity_keeps_the_old_block(pool):
    a, b = _metadata(pool=pool), _metadata(pool=pool)
    ta = layout._ensure_workspace_tensor(
        a, "ws", (10,), dtype=torch.int32, device=CPU, capacity_numel=16, shared=True
    )
    tb = layout._ensure_workspace_tensor(
        b, "ws", (64,), dtype=torch.int32, device=CPU, capacity_numel=16, shared=True
    )
    assert pool.allocations == 2
    (key,) = pool.buffers
    # the first block is still there, reserved: a captured graph may point at it
    assert sorted(block.buffer.numel() for block in pool.buffers[key]) == [16 * 4, 64 * 4]
    assert all(block.is_reserved for block in pool.buffers[key])
    assert ta.data_ptr() != tb.data_ptr()


def test_capture_never_allocates(pool, monkeypatch):
    a = _metadata(pool=pool)
    layout._ensure_workspace_tensor(
        a, "ws", (8,), dtype=torch.float32, device=CPU, capacity_numel=32, shared=True
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    b = _metadata(pool=pool)
    # a hit inside the existing block is fine, a miss raises
    layout._ensure_workspace_tensor(
        b, "ws", (32,), dtype=torch.float32, device=CPU, capacity_numel=32, shared=True
    )
    with pytest.raises(ValueError, match="capturing"):
        layout._ensure_workspace_tensor(
            b, "ws", (33,), dtype=torch.float32, device=CPU, capacity_numel=32, shared=True
        )
    with pytest.raises(ValueError, match="capturing"):
        layout._ensure_workspace_tensor(
            b, "other", (1,), dtype=torch.float32, device=CPU, capacity_numel=1, shared=True
        )
    with pytest.raises(ValueError, match="capturing"):
        layout._ensure_workspace_tensor(
            b, "unbounded", (1,), dtype=torch.float32, device=CPU, shared=False
        )
    with pytest.raises(ValueError, match="capturing"):
        layout._ensure_workspace_tensor(b, "vp", (1,), dtype=torch.uint8, device=CPU, shared=False)
    assert pool.allocations == 1


def test_scratch_stays_with_its_runner(pool):
    other = _FakePool()
    a, b = _metadata(pool=pool), _metadata(pool=other)
    ta = layout._ensure_workspace_tensor(
        a, "ws", (8,), dtype=torch.float32, device=CPU, shared=True, capacity_numel=32
    )
    tb = layout._ensure_workspace_tensor(
        b, "ws", (8,), dtype=torch.float32, device=CPU, shared=True, capacity_numel=32
    )
    assert ta.data_ptr() != tb.data_ptr()
    assert pool.allocations == other.allocations == 1


@pytest.mark.parametrize("m_tiles", [1, 2, 4])
@pytest.mark.parametrize("max_requests", [1, 7, 23, 64, 128, 256])
def test_packed_split_task_bound_covers_every_batch(monkeypatch, m_tiles, max_requests):
    monkeypatch.delenv(_FP4_MLA_SPLIT_KV_ENV, raising=False)
    monkeypatch.setattr(packed_plan, "_get_sm_count", lambda device: 212)
    bound = packed_plan.packed_split_task_bound(max_requests, m_tiles, CPU)
    worst = max(
        packed_plan._select_packed_split_count(
            requests, m_tiles, CPU, max_kv_len=1 << 20, is_cuda_graph=True
        )
        * requests
        for requests in range(1, max_requests + 1)
    )
    assert bound == max(worst, 32)
    # eager calls never exceed the graph rule: they only cap the splits further by the KV length
    for requests in (1, max_requests):
        for max_kv_len in (256, 4096, 1 << 20):
            splits = packed_plan._select_packed_split_count(
                requests, m_tiles, CPU, max_kv_len=max_kv_len, is_cuda_graph=False
            )
            assert max(splits * requests, 32) <= bound
