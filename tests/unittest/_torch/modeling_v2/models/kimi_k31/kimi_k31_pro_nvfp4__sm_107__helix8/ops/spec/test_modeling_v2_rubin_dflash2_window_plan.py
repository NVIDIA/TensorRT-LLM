# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window plan entry (``rubin_dflash2_window_plan``), on a sample of its cells.

CPU (any host): on every sampled cell and every CP rank the reference plan gives each block query
exactly its window: over the ranks its body, head and block keys cover ``[lo_j, L + j]`` once,
each key on the rank whose pages hold it, at the slot ``HelixDraftShard.local_slots`` gives it
there; the chunks partition each body, whose pages are valid up to its length and -1 after.

GPU (SM107): the plan kernel equals the reference exactly on every sampled cell.
"""

from __future__ import annotations

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    plan_buffers,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    rubin_dflash2_window_plan as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.helix_draft_shard import (
    HelixDraftShard,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

gpu = on_sm(107, reason="the window entries run on SM107 (Rubin)")

#: A sample of the entry's cells: the smallest and the median batch.
CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


def _build(spec: dict, seed: int, device):
    """Lengths, a global page table with a distinct pool page per (request, page), and the
    request -> table row map when the cell reads rows of a shared table."""
    g = torch.Generator().manual_seed(seed)
    lengths = list(spec["lengths"])
    batch, blk, page, cp = len(lengths), spec["block_size"], spec["page_size"], spec["cp_size"]
    span = page * cp
    gcols = -(-(max(lengths) + blk + span) // span) * cp
    table = torch.randperm(batch * gcols + 3, generator=g)[: batch * gcols].view(batch, gcols)
    rows = None
    if spec["table_rows"]:
        rows = torch.randperm(batch + 5, generator=g)[:batch]
    num_ctx = torch.tensor(lengths, dtype=torch.int64)
    return num_ctx.to(device), table.to(device), None if rows is None else rows.to(device)


def _local(table: torch.Tensor, rows, cp: int, rank: int, num_rows: int) -> torch.Tensor:
    """Rank ``rank``'s table (local page ``c`` = global page ``c * cp + rank``), placed at the
    requests' rows of a larger shared table when ``rows`` is given."""
    local = table[:, rank::cp].to(torch.int32)
    if rows is None:
        return local.contiguous()
    shared = torch.full((num_rows, local.shape[1]), 2**30, dtype=torch.int32, device=table.device)
    shared[rows] = local
    return shared


def _args(spec, num_ctx, table, rows, rank):
    cp = spec["cp_size"]
    num_rows = 0 if rows is None else int(rows.max()) + 3
    return (
        num_ctx,
        _local(table, rows, cp, rank, num_rows),
        rows,
        spec["block_size"],
        spec["window"],
        spec["page_size"],
        cp,
        rank,
        spec["num_splits"],
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _ids(cells):
    return [c.why[:56] for c in cells]


# CPU


@pytest.mark.cpu_only
@pytest.mark.parametrize("cell", CELLS, ids=_ids(CELLS))
def test_reference_plan_is_each_querys_window(cell):
    spec = cell.spec
    num_ctx, table, rows = _build(spec, _seed(cell), "cpu")
    blk, page, window, cp = spec["block_size"], spec["page_size"], spec["window"], spec["cp_size"]
    # pool page -> (request, global page)
    owner_req = torch.full((int(table.max()) + 1,), -1, dtype=torch.long)
    owner_page = torch.full_like(owner_req, -1)
    owner_req[table.reshape(-1)] = torch.arange(table.shape[0]).repeat_interleave(table.shape[1])
    owner_page[table.reshape(-1)] = torch.arange(table.shape[1]).repeat(table.shape[0])
    seen = {}
    for rank in range(cp):
        plan = op.reference(*_args(spec, num_ctx, table, rows, rank), None)
        shard = HelixDraftShard(cp, rank, page)
        local = table[:, rank::cp].long()
        for b, length in enumerate(num_ctx.tolist()):
            body_len = int(plan.body_len[b])
            t = torch.arange(body_len)
            body_rows = plan.body_pages[b].long()[t // page] * page + t % page
            bounds = plan.chunk_bounds[b].tolist()
            assert bounds[0] == 0 and bounds[-1] == body_len, "the chunks partition the body"
            assert all(lo <= hi and lo % page == 0 for lo, hi in zip(bounds, bounds[1:]))
            pages = int((plan.body_pages[b] >= 0).sum())
            assert pages == -(-body_len // page) and (plan.body_pages[b, pages:] == -1).all()
            head = plan.head_rows[b].long()
            for j in range(blk):
                key_rows = torch.cat(
                    (
                        body_rows,
                        head[(head >= 0) & (torch.arange(len(head)) >= plan.head_first[b, j])],
                    )
                )
                assert (owner_req[key_rows // page] == b).all(), "a key of another request"
                gpage = owner_page[key_rows // page]
                assert (gpage % cp == rank).all(), "a key on another rank's page"
                positions = (gpage * page + key_rows % page).tolist()
                bits = int(plan.block_bits[b])
                positions += [length + k for k in range(j + 1) if (bits >> k) & 1]
                assert all(
                    ((length + k) // page) % cp == rank for k in range(blk) if (bits >> k) & 1
                )
                # The store's rule: the rank owns each key at the slot local_slots names.
                pos = torch.tensor(positions, dtype=torch.long)
                slots, owned = shard.local_slots(pos)
                assert bool(owned.all()), f"rank {rank} reads a position it does not own"
                hist = slots[pos < length]
                stored = local[b][hist // page] * page + hist % page
                assert sorted(stored.tolist()) == sorted(key_rows.tolist())
                seen.setdefault((b, j), []).extend(positions)
    for (b, j), positions in seen.items():
        p = int(num_ctx[b]) + j
        assert sorted(positions) == list(range(max(0, p - window + 1), p + 1)), (b, j)


# GPU


def _filled_buffers(spec, batch, device):
    buffers = plan_buffers(
        batch + 2,
        spec["block_size"],
        spec["window"],
        spec["page_size"],
        spec["cp_size"],
        spec["num_splits"],
        device,
    )
    for t in buffers:
        t.fill_(-7)  # what a larger earlier step could have left
    return buffers


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=_ids(CELLS))
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    num_ctx, table, rows = _build(spec, _seed(cell), "cuda")
    for rank in range(spec["cp_size"]):
        args = _args(spec, num_ctx, table, rows, rank)
        got = op(*args, _filled_buffers(spec, num_ctx.shape[0], "cuda"))
        want = op.reference(*args, None)
        for name, a, b in zip(want._fields, got, want):
            assert torch.equal(a, b), f"rank {rank}: {name} differs"
