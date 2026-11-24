# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's helix combine entry (``rubin_dflash2_helix_combine``).

Every CP rank attends its own disjoint key shard; this entry merges the ranks' normalized
partials by their ``(max, sum)`` stats and returns the rank's head shard.

CPU (any host): the reference equals the softmax over all shards together, with rows some
ranks hold no key of; sanitized rows without a mask give exactly what the masked path gives
for unsanitized (NaN) rows; and the reference agrees with ``helix_target_combine.reference``.

GPU (SM107, one GPU): every cell through the entry itself, all CP ranks simulated in one
process. The engine's all-to-all is replaced by the exchange it performs (rank ``r`` receives
chunk ``r`` of every rank), so the entry's reformat and the engine's combine kernel run as in
a multi-rank forward. The real exchange is the multi-rank test's
(``rubin_multirank/test_rubin_dflash2_window_helix.py``).
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_helix_combine import (  # noqa: E501
    rubin_dflash2_helix_combine as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

_ON_SM107 = torch.cuda.is_available() and get_sm_version() in (103, 107)
gpu = pytest.mark.skipif(
    not _ON_SM107, reason="the window entries run on SM103 (GB300) and SM107 (Rubin)"
)


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
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
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


@pytest.mark.cpu_only
def test_sanitized_rows_without_a_mask_equal_the_masked_path():
    spec = dict(op.CELLS[0].spec)
    partials, stats, empty, _ = _shards(spec, 3)
    poisoned = partials.masked_fill(empty[..., None, None], float("nan"))
    poisoned_stats = stats.clone()
    poisoned_stats[..., 0].masked_fill_(empty[..., None], 7.0)  # whatever a kernel left there
    poisoned_stats[..., 1].masked_fill_(empty[..., None], float("nan"))
    for rank in range(spec["cp"]):
        mapping = _mapping(spec["cp"], rank)
        sanitized = op.reference(partials, stats, mapping, spec["heads"], spec["value_dim"])
        masked = op.reference(
            poisoned, poisoned_stats, mapping, spec["heads"], spec["value_dim"], empty
        )
        assert torch.equal(sanitized, masked)


@pytest.mark.cpu_only
def test_is_valid_refuses_unsanitized_rows_without_a_mask():
    spec = dict(op.CELLS[2].spec)
    partials, stats, empty, _ = _shards(spec, 5, dtype=torch.bfloat16)
    mapping = _mapping(spec["cp"], 0)
    op.is_valid(partials[0], stats[0], mapping, spec["heads"], spec["value_dim"])
    poisoned = partials[0].masked_fill(empty[0][:, None, None], float("nan"))
    with pytest.raises(AssertionError, match="unsanitized"):
        op.is_valid(poisoned, stats[0], mapping, spec["heads"], spec["value_dim"])
    op.is_valid(poisoned, stats[0], mapping, spec["heads"], spec["value_dim"], empty[0])


def test_reference_agrees_with_the_target_combine_reference():
    try:
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.helix import (  # noqa: E501
            helix_target_combine,
        )
    except Exception as error:  # noqa: BLE001 - the engine attention import chain
        pytest.skip(f"the engine attention module does not import here: {error!r}"[:200])
    spec = dict(op.CELLS[1].spec)
    partials, stats, _, _ = _shards(spec, 9, dtype=torch.float32)
    for rank in range(spec["cp"]):
        ours = op.reference(
            partials, stats, _mapping(spec["cp"], rank), spec["heads"], spec["value_dim"]
        )
        theirs = helix_target_combine.reference(partials, stats, rank)
        torch.testing.assert_close(ours, theirs.reshape(ours.shape), rtol=1e-6, atol=1e-6)


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
def test_certified_cells_with_the_exchange_simulated(cell, certifying, monkeypatch):  # noqa: F811
    import tensorrt_llm._torch.attention.attention as attention

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
