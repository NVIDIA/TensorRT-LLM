# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family library's ``minimax_m3_msa_kv_update_tail``: the MSA KV-length correction of ``minimax_m3_msa_kv_update``
plus the metadata's out_cache_loc tail fill in one launch.

On every certified cell of the msa_kv_update entry it writes -1 over ``out_cache_loc[tail_start:]`` and leaves every
other output byte-equal to the base op's, which itself matches the entry's reference.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.msa_kv_update import (  # noqa: E501
    TOKENS_PER_BLOCK,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.msa_kv_update import (  # noqa: E501
    msa_kv_update as entry,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the family library's fixed MiniMax-M3 ops are tested on sm_107 (Rubin)",
)

#: Spare capacity past the live rows, as the metadata's buffers have.
_SPARE = 5
_STALE = -12345


def _case(spec, seed):
    """The metadata's buffers for one step, with stale values in every output so a skipped write shows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    batch, qlen = spec["batch"], spec["query_len"]
    total_q = batch * qlen
    staged = torch.randint(qlen, 5000, (batch,), generator=g, dtype=torch.int32)
    accept = spec["accept"]
    if accept == "full":
        kv = staged.clone()
    elif accept == "page_edge":
        staged = torch.tensor([130, 257, 2 * TOKENS_PER_BLOCK + qlen][:batch], dtype=torch.int32)
        kv = staged - torch.tensor([0, 1, qlen - 1][:batch], dtype=torch.int32)
    else:
        kv = staged - torch.randint(0, qlen, (batch,), generator=g, dtype=torch.int32)
        if accept == "padded":
            kv[batch // 2 :] = 0
            staged[batch // 2 :] = 0
    ncols = (5000 + qlen) // TOKENS_PER_BLOCK + 2
    table = (
        torch.randperm(batch * ncols + 7, generator=g)[: batch * ncols]
        .to(torch.int32)
        .reshape(batch, ncols)
    )
    dev = dict(device="cuda", dtype=torch.int32)
    return [
        torch.cat([kv, torch.full((_SPARE,), 99, dtype=torch.int32)]).cuda(),
        torch.cat([staged, torch.full((_SPARE,), 99, dtype=torch.int32)]).cuda(),
        torch.full((batch + _SPARE,), qlen, **dev),
        torch.arange(batch, **dev).repeat_interleave(qlen),
        torch.arange(qlen, **dev).repeat(batch),
        table.cuda(),
        TOKENS_PER_BLOCK,
        batch,
        total_q,
        torch.full((batch + _SPARE,), _STALE, **dev),
        torch.full((batch + 1 + _SPARE,), _STALE, **dev) if spec["cu_kv"] else None,
        torch.full((total_q + _SPARE,), _STALE, **dev),
        torch.full((total_q + _SPARE,), _STALE, **dev) if spec["n_valid"] else None,
    ]


def _clone(args):
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


def _bytes_equal(a, b):
    return (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))


@pytest.mark.parametrize("cell", entry.CELLS, ids=[c.why[:44] for c in entry.CELLS])
@pytest.mark.parametrize("tail", ["at_total_q", "inside_spare", "at_end"])
def test_the_tail_fill_and_nothing_else_differs(cell, tail):
    base = _case(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    fused = _clone(base)
    pristine = _clone(base)
    total_q, out_len = base[8], base[11].numel()
    tail_start = {"at_total_q": total_q, "inside_spare": total_q + 2, "at_end": out_len}[tail]
    family_ops().minimax_m3_msa_kv_update(*base)
    family_ops().minimax_m3_msa_kv_update_tail(*fused, tail_start)
    torch.cuda.synchronize()
    # The base op is the entry's correction.
    ref_seq, ref_cu, ref_loc, ref_nv = entry.reference(
        *_case(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    )
    assert torch.equal(base[9][: base[7]], ref_seq) and torch.equal(base[11][:total_q], ref_loc)
    # Every output but the tail is byte-equal to the base op's.
    for i in (9, 10, 12):
        assert _bytes_equal(fused[i], base[i]), i
    assert torch.equal(fused[11][:tail_start], base[11][:tail_start])
    assert bool((fused[11][tail_start:] == -1).all())
    # The inputs are not written.
    for i in range(6):
        assert torch.equal(base[i], pristine[i]) and torch.equal(fused[i], pristine[i]), i


def test_a_tail_inside_the_live_slots_is_refused():
    args = _case(entry.CELLS[0].spec, seed=3)
    with pytest.raises(RuntimeError, match="tail_start"):
        family_ops().minimax_m3_msa_kv_update_tail(*args, args[8] - 1)
    with pytest.raises(RuntimeError, match="tail_start"):
        family_ops().minimax_m3_msa_kv_update_tail(*args, args[11].numel() + 1)
