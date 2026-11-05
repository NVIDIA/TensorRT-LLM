# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the spec_rollback_commit entries, driven by their CELLS.

`spec_rollback_stage` is compared bitwise with its reference. `spec_rollback_commit` has no
reference of its own (test_modeling_v2_spec_rollback drives it through the handler); here each
cell is checked bitwise against the docstring's per-request rewrite, written out in torch.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.spec_rollback_commit import (  # noqa: E501
    ROW,
    spec_rollback_commit,
    spec_rollback_stage,
)

assert torch.cuda.is_available(), "spec_rollback_commit requires a CUDA device"

DEV = "cuda"
CTX = 2
CHANNELS = 640
STATE = 3
TPB = 32


def _slots(n: int, g: torch.Generator) -> torch.Tensor:
    return (torch.randperm(2 * n + 3, generator=g)[:n] + 1).to(DEV, torch.int32)


@pytest.mark.parametrize(
    "cell", spec_rollback_stage.CELLS, ids=[c.why for c in spec_rollback_stage.CELLS]
)
def test_stage_cells(cell) -> None:
    n, p = cell.spec["n"], cell.spec["p"]
    g = torch.Generator().manual_seed(n * 31 + p)
    slots = _slots(n, g)
    pool_slots = 2 * n + 4
    ngram_pool = torch.randint(0, 1 << 40, (pool_slots, CTX), generator=g).to(DEV)
    conv_pool = torch.randn(pool_slots, CHANNELS, STATE, generator=g).to(DEV, torch.bfloat16)
    ids = torch.randint(0, 1 << 20, (n * p,), generator=g).to(DEV, torch.int32)
    spec_rollback_stage.is_valid(ngram_pool, conv_pool, slots, ids, p)
    stream, conv_prev = spec_rollback_stage(ngram_pool, conv_pool, slots, ids, p)
    want_stream, want_prev = spec_rollback_stage.reference(ngram_pool, conv_pool, slots, ids, p)
    assert torch.equal(stream, want_stream)
    assert torch.equal(conv_prev, want_prev)


@pytest.mark.parametrize(
    "cell", spec_rollback_commit.CELLS, ids=[c.why for c in spec_rollback_commit.CELLS]
)
def test_commit_cells(cell) -> None:
    n, p, planes = cell.spec["n"], cell.spec["p"], cell.spec["planes"]
    g = torch.Generator().manual_seed(n * 17 + planes)
    rows = n * p
    slots = _slots(n, g)
    pool_slots = 2 * n + 4
    accepted = torch.randint(1, p + 1, (n,), generator=g).to(DEV, torch.int32)
    ngram_pool = torch.randint(0, 1 << 40, (pool_slots, CTX), generator=g).to(DEV)
    stream = torch.randint(0, 1 << 40, (n, CTX + p), generator=g).to(DEV)
    conv_pool = torch.randn(pool_slots, CHANNELS, STATE, generator=g).to(DEV, torch.bfloat16)
    conv_prev = torch.randn(n, CHANNELS, STATE, generator=g).to(DEV, torch.bfloat16)
    conv_x = torch.randn(n, p, CHANNELS, generator=g).to(DEV, torch.bfloat16)
    # Each row its own (page, slot within the page).
    flat = torch.randperm(rows * 2, generator=g)[:rows]
    pages, within = (flat // TPB).to(DEV), (flat % TPB).to(DEV)
    num_pages = int(pages.max()) + 1
    position = torch.randint(0, 1 << 20, (num_pages, TPB, 3), generator=g).to(DEV, torch.int32)
    position_old = torch.randint(0, 1 << 20, (rows, 3), generator=g).to(DEV, torch.int32)
    plane_list = [
        torch.randn(num_pages, TPB, ROW, generator=g).to(DEV, torch.bfloat16) for _ in range(planes)
    ]
    plane_ptrs = torch.tensor([t.data_ptr() for t in plane_list], dtype=torch.int64, device=DEV)
    plane_strides = torch.tensor([t.stride(0) for t in plane_list], dtype=torch.int64, device=DEV)
    index_old = torch.randn(planes, rows, ROW, generator=g).to(DEV, torch.bfloat16)

    want_ngram, want_conv = ngram_pool.clone(), conv_pool.clone()
    want_position, want_planes = position.clone(), [t.clone() for t in plane_list]
    for i in range(n):
        a, slot = int(accepted[i]), int(slots[i])
        want_ngram[slot] = stream[i, a : a + CTX]
        cat = torch.cat([conv_prev[i], conv_x[i].T], 1)
        want_conv[slot] = cat[:, a : a + STATE]
        for t in range(a, p):
            r = i * p + t
            pg, tok = int(pages[r]), int(within[r])
            want_position[pg, tok] = position_old[r]
            for plane, want in zip(range(planes), want_planes):
                want[pg, tok] = index_old[plane, r]

    args = (accepted, slots, ngram_pool, stream, conv_pool, conv_prev, conv_x, position,
            position_old, pages, within, plane_ptrs, plane_strides, index_old)  # fmt: skip
    spec_rollback_commit.is_valid(*args)
    spec_rollback_commit(*args)
    torch.cuda.synchronize()
    assert torch.equal(ngram_pool, want_ngram)
    assert torch.equal(conv_pool, want_conv)
    assert torch.equal(position, want_position)
    for got, want in zip(plane_list, want_planes):
        assert torch.equal(got, want)
