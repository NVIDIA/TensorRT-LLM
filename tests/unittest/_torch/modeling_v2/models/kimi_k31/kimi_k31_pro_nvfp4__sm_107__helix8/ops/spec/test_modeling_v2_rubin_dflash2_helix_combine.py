# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's helix combine entry (``rubin_dflash2_helix_combine``).

Every CP rank attends its own disjoint key shard; this entry merges the ranks' normalized
partials by their ``(max, sum)`` stats and returns the rank's head shard.

CPU (any host), on a sample of the cells: the reference equals the softmax over all shards
together, with rows some ranks hold no key of.

GPU (SM107, one GPU): the sampled cells through the entry itself, all CP ranks simulated in one
process. The engine's all-to-all is replaced by the exchange it performs (rank ``r`` receives
chunk ``r`` of every rank), so the entry's reformat and the engine's combine kernel run as in
a multi-rank forward. The real exchange runs in the helix8 end-to-end tests.
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.ops.spec.rubin_dflash2_helix_combine import (  # noqa: E501
    rubin_dflash2_helix_combine as op,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

gpu = on_sm(107, reason="the window entries run on SM107 (Rubin)")

#: A sample of the entry's cells: the smallest and the median rows.
CELLS = sample_cells(op.CELLS, size_key="rows")[:2]


def _shards(spec: dict, seed: int, device="cpu", dtype=torch.float64):
    """Per-rank partials and stats of attention over disjoint key shards, and the full result.

    Rows 0 and 1 have no key on some ranks; every row has a key somewhere (its own block key in
    the drafter). Returns ``(partials [cp, R, H, V], stats [cp, R, H, 2], empty [cp, R], full
    [R, H, V])`` with empty rows already sanitized (zero partial, ``(-inf, 0)``).
    """
    g = torch.Generator().manual_seed(seed)
    cp, rows, heads, value_dim = spec["cp"], spec["rows"], spec["heads"], spec["value_dim"]
    keys = 6
    scores = 3 * torch.randn(cp, rows, heads, keys, generator=g, dtype=torch.float64)
    values = torch.randn(cp, rows, keys, value_dim, generator=g, dtype=torch.float64)
    empty = torch.zeros(cp, rows, dtype=torch.bool)
    empty[1:, 0] = True
    empty[: cp // 2, 1] = True
    masked = scores.masked_fill(empty[:, :, None, None], float("-inf"))
    full = torch.softmax(masked.permute(1, 2, 0, 3).reshape(rows, heads, cp * keys), -1)
    full_v = values.permute(1, 0, 2, 3).reshape(rows, cp * keys, value_dim)
    full = torch.einsum("rhk,rkv->rhv", full, full_v)
    peak = masked.amax(-1)
    has = torch.isfinite(peak)
    probs = torch.exp(masked - torch.where(has, peak, 0)[..., None])
    total = probs.sum(-1)
    partials = (
        torch.einsum("crhk,crkv->crhv", probs, values) / torch.where(has, total, 1)[..., None]
    )
    partials = torch.where(has[..., None], partials, 0)
    stats = torch.stack((torch.where(has, peak, float("-inf")), torch.where(has, total, 0)), dim=-1)
    return (
        partials.to(device=device, dtype=dtype),
        stats.to(device=device, dtype=torch.float32),
        empty.to(device),
        full.to(device),
    )


def _mapping(cp: int, rank: int, nccl: bool = True):
    return SimpleNamespace(
        cp_size=cp,
        cp_rank=rank,
        cp_group=list(range(cp)),
        cp_config={"use_nccl_for_alltoall": nccl},
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _ids(cells):
    return [c.why[:56] for c in cells]


@pytest.mark.cpu_only
@pytest.mark.parametrize("cell", CELLS, ids=_ids(CELLS))
def test_reference_is_the_softmax_over_every_shard(cell):
    spec = cell.spec
    partials, stats, empty, full = _shards(spec, _seed(cell))
    cp, heads, value_dim = spec["cp"], spec["heads"], spec["value_dim"]
    mask = None
    if spec["mask"]:
        partials = partials.masked_fill(empty[..., None, None], float("nan"))
        mask = empty
    width = heads // cp
    for rank in range(cp):
        got = op.reference(partials, stats, _mapping(cp, rank), heads, value_dim, mask)
        want = full[:, rank * width : (rank + 1) * width].reshape(spec["rows"], -1)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=_ids(CELLS))
def test_certified_cells_with_the_exchange_simulated(cell, certifying, monkeypatch):  # noqa: F811
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention import helix as attention

    certifying(op)
    spec = cell.spec
    cp, rows, heads, value_dim = spec["cp"], spec["rows"], spec["heads"], spec["value_dim"]
    partials, stats, empty, _ = _shards(spec, _seed(cell), device="cuda", dtype=torch.bfloat16)
    mask = None
    if spec["mask"]:
        partials = partials.masked_fill(empty[..., None, None], float("nan"))
        mask = empty
    # What every rank sends: the reformat of its own partial and stats.
    sent = [
        attention._helix_nccl_pre_alltoall_eager(
            partials[s].reshape(rows, heads * value_dim),
            stats[s],
            None if mask is None else mask[s],
            cp,
        )
        for s in range(cp)
    ]
    for rank in range(cp):
        received = [
            torch.stack([sent[s][rank] for s in range(cp)]),
            torch.stack([sent[s][cp + rank] for s in range(cp)]),
        ]
        monkeypatch.setattr(attention, "alltoall_helix", lambda chunks, group, r=received: r)
        got = op(
            partials[rank],
            stats[rank],
            _mapping(cp, rank),
            heads,
            value_dim,
            None if mask is None else mask[rank],
        )
        want = op.reference(partials, stats, _mapping(cp, rank), heads, value_dim, mask)
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)
