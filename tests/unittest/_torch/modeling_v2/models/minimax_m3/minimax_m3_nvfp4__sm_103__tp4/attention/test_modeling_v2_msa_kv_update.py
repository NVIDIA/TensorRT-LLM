# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the msa_kv_update entry (the one-launch MSA KV-length correction)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.msa_kv_update import (  # noqa: E501
    TOKENS_PER_BLOCK,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.attention.msa_kv_update import (  # noqa: E501
    msa_kv_update as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.msa_kv_update import (  # noqa: E501
    msa_kv_update as hook,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401 — needs the path declared above

assert torch.cuda.is_available(), "msa_kv_update requires a CUDA device"

#: Spare capacity past the live rows, as the metadata's buffers have: the op must leave it alone.
_SPARE = 3


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
    stale = -12345
    return (
        torch.cat([kv, torch.full((_SPARE,), 99, dtype=torch.int32)]).cuda(),
        torch.cat([staged, torch.full((_SPARE,), 99, dtype=torch.int32)]).cuda(),
        torch.full((batch + _SPARE,), qlen, **dev),
        torch.arange(batch, **dev).repeat_interleave(qlen),
        torch.arange(qlen, **dev).repeat(batch),
        table.cuda(),
        TOKENS_PER_BLOCK,
        batch,
        total_q,
        torch.full((batch + _SPARE,), stale, **dev),
        torch.full((batch + 1 + _SPARE,), stale, **dev) if spec["cu_kv"] else None,
        torch.full((total_q + _SPARE,), stale, **dev),
        torch.full((total_q + _SPARE,), stale, **dev) if spec["n_valid"] else None,
    )


def _check(args, ref):
    seq, cu_kv, loc, n_valid = args[9], args[10], args[11], args[12]
    batch, total_q = args[7], args[8]
    ref_seq, ref_cu, ref_loc, ref_nv = ref
    # Integer bookkeeping: exact, and nothing past the live rows is touched.
    assert torch.equal(seq[:batch], ref_seq)
    assert torch.equal(loc[:total_q], ref_loc)
    assert bool((seq[batch:] == -12345).all()) and bool((loc[total_q:] == -12345).all())
    assert (cu_kv is None) == (ref_cu is None) and (n_valid is None) == (ref_nv is None)
    if cu_kv is not None:
        assert torch.equal(cu_kv[: batch + 1], ref_cu)
        assert bool((cu_kv[batch + 1 :] == -12345).all())
    if n_valid is not None:
        assert torch.equal(n_valid[:total_q], ref_nv)
        assert bool((n_valid[total_q:] == -12345).all())


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    args = _case(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    op(*args)
    _check(args, op.reference(*args))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_the_hook_takes_every_cell(cell) -> None:
    """The target's hook (what the metadata calls) takes each certified step and writes what the op does."""
    args = _case(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31 + 1)
    assert hook(*args) is True
    _check(args, op.reference(*args))


def test_the_hook_declines_what_the_op_would_misread() -> None:
    """A strided operand is read wrongly by the op, so the hook declines it (the metadata then runs its own
    torch update) and writes nothing."""
    args = list(_case(op.CELLS[2].spec, seed=5))
    args[0] = torch.stack([args[0], args[0]], dim=1)[:, 0]
    before = args[11].clone()
    assert hook(*args) is False
    assert torch.equal(args[11], before)


def test_guard_refuses_a_strided_operand() -> None:
    args = list(_case(op.CELLS[2].spec, seed=6))
    args[0] = torch.stack([args[0], args[0]], dim=1)[:, 0]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(*args)


def test_guard_refuses_a_request_index_past_the_batch() -> None:
    args = list(_case(op.CELLS[2].spec, seed=7))
    args[3] = args[3].clone()
    args[3][-1] = args[7]
    with pytest.raises(AssertionError, match="q_batch_row"):
        with validating(op):
            op(*args)
