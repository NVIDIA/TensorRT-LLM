# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `verify_step_setup` against the composed ops of the decode target's step setup
(`pos3` copy, `cu_seqlens_from_seq_lens`, `SpeculativeRollback.begin`'s slot arithmetic and
`qsa_rollback_gather`), bitwise, eagerly and replayed from a CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_rollback_planes import (  # noqa: E501
    plane_table,
    qsa_rollback_gather,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ssm.cu_seqlens_from_seq_lens import (  # noqa: E501
    cu_seqlens_from_seq_lens,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.verify_step_setup import (  # noqa: E501
    verify_step_setup,
)

assert torch.cuda.is_available(), "the verify step setup test requires a CUDA device"

DEV = "cuda"
TPB = 32
PAGES = 96
LAYERS = 12
MAX_BLOCKS = 6


def _case(batch, p, seed, negative=False):
    g = torch.Generator().manual_seed(seed)
    rows = batch * p
    full = torch.randint(0, 1 << 20, (3, 1, rows + 5), generator=g).to(DEV, torch.int32)
    position_ids = full[..., :rows]
    seq_lens = torch.full((batch,), p, dtype=torch.int32, device=DEV)
    slots = torch.randint(0, 64, (batch,), generator=g).to(DEV, torch.int32)
    req = torch.arange(batch).repeat_interleave(p).to(DEV, torch.int32)
    starts = torch.randint(0, MAX_BLOCKS * TPB - p - 20, (batch,), generator=g)
    logical = (starts.repeat_interleave(p) + torch.arange(p).repeat(batch)).to(DEV, torch.int64)
    if negative:
        logical[:p] -= starts[0] + 40  # a padded request: negative positions, one page back
    block_table = torch.randint(0, PAGES, (batch + 3, MAX_BLOCKS + 2), generator=g)
    block_table = block_table.to(DEV, torch.int32)[:, :MAX_BLOCKS]
    planes = [
        torch.randn(PAGES, TPB, 1, 128, generator=g).to(DEV, torch.bfloat16) for _ in range(LAYERS)
    ]
    return position_ids, seq_lens, slots, req, logical, block_table, planes


def _reference(position_ids, seq_lens, slots, req, logical, block_table, ptrs, strides):
    pos3 = torch.empty([3, position_ids.shape[-1]], dtype=torch.int32, device=DEV)
    pos3.copy_(torch.reshape(position_ids, [3, -1]))
    qsl = cu_seqlens_from_seq_lens(seq_lens, torch.int64)
    positions = logical.to(torch.long)
    pages = block_table[req.to(torch.long), positions // TPB].to(torch.long)
    within = positions % TPB
    index = qsa_rollback_gather(ptrs, strides, pages, within, LAYERS)
    return pos3, qsl, slots.to(torch.long), pages, within, index


def _bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


@pytest.mark.parametrize("negative", [False, True], ids=["positions", "negative_positions"])
@pytest.mark.parametrize("batch,p", [(1, 4), (3, 4), (32, 4), (17, 8), (255, 2)])
def test_setup_matches_composed_ops(batch, p, negative):
    position_ids, seq_lens, slots, req, logical, block_table, planes = _case(
        batch, p, batch * 7 + p, negative
    )
    ptrs, strides = plane_table(planes)
    ref = _reference(position_ids, seq_lens, slots, req, logical, block_table, ptrs, strides)
    got = verify_step_setup(
        position_ids, seq_lens, slots, req, logical, block_table, TPB, ptrs, strides, LAYERS
    )
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(_bits(a), _bits(b))


def test_setup_zero_length_rows_in_segmentation():
    position_ids, seq_lens, slots, req, logical, block_table, planes = _case(6, 2, 5)
    seq_lens.copy_(torch.tensor([2, 2, 2, 2, 2, 2], dtype=torch.int32))
    seq_lens[3:] = 0
    ptrs, strides = plane_table(planes)
    got = verify_step_setup(
        position_ids, seq_lens, slots, req, logical, block_table, TPB, ptrs, strides, LAYERS
    )
    assert torch.equal(got[1], cu_seqlens_from_seq_lens(seq_lens, torch.int64))


def test_setup_cuda_graph_follows_inputs():
    position_ids, seq_lens, slots, req, logical, block_table, planes = _case(8, 4, 11)
    ptrs, strides = plane_table(planes)
    args = (position_ids, seq_lens, slots, req, logical, block_table, TPB, ptrs, strides, LAYERS)
    verify_step_setup(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = verify_step_setup(*args)
    g = torch.Generator().manual_seed(12)
    for _ in range(3):
        logical.add_(torch.randint(0, 3, logical.shape, generator=g).to(DEV))
        position_ids.add_(1)
        for plane in planes:
            plane.copy_(torch.randn(plane.shape, generator=g).to(DEV, torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference(position_ids, seq_lens, slots, req, logical, block_table, ptrs, strides)
        for a, b in zip(out, ref):
            assert torch.equal(_bits(a), _bits(b))


@pytest.mark.parametrize(
    "cell", verify_step_setup.CELLS, ids=[c.why for c in verify_step_setup.CELLS]
)
def test_setup_cells(cell) -> None:
    """Each cell against the docstring's outputs written out in torch indexing."""
    batch, p, num_planes = cell.spec["batch"], cell.spec["p"], cell.spec["planes"]
    position_ids, seq_lens, slots, req, logical, block_table, planes = _case(batch, p, 21 * batch)
    planes = planes[:num_planes]
    ptrs, strides = plane_table(planes)
    args = (position_ids, seq_lens, slots, req, logical, block_table, TPB, ptrs, strides,
            num_planes)  # fmt: skip
    verify_step_setup.is_valid(*args)
    pos3, qsl, slots_long, pages, within, index = verify_step_setup(*args)
    rows = batch * p
    assert torch.equal(pos3, position_ids.reshape(3, rows).to(torch.int32))
    want_qsl = torch.zeros(batch + 1, dtype=torch.int64, device=DEV)
    want_qsl[1:] = torch.cumsum(seq_lens.to(torch.int64), 0)
    assert torch.equal(qsl, want_qsl)
    assert torch.equal(slots_long, slots.to(torch.int64))
    want_pages = block_table[req.long(), torch.div(logical, TPB, rounding_mode="floor")].long()
    want_within = torch.remainder(logical, TPB)
    assert torch.equal(pages, want_pages)
    assert torch.equal(within, want_within)
    want_index = torch.stack(
        [plane[want_pages, want_within].reshape(rows, 128) for plane in planes]
    )
    assert torch.equal(_bits(index), _bits(want_index))
