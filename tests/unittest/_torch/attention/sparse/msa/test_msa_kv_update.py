# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MSA's one-launch KV-length correction (kernels.msa_kv_update) against its torch formulation."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_kv_update as kv

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not kv.kernel_available(),
    reason="needs CUDA and the trtllm::minimax_m3_msa_kv_update kernel",
)

#: The MSA pool's page: one sparse block is one 128-token KV page.
TOKENS_PER_BLOCK = 128

#: The steps the metadata hands the kernel: (why, spec).
_CASES = (
    (
        "decode without speculation at one request",
        dict(batch=1, query_len=1, accept="full", cu_kv=True, n_valid=True),
    ),
    (
        "decode without speculation at batch 8",
        dict(batch=8, query_len=1, accept="full", cu_kv=True, n_valid=True),
    ),
    (
        "Eagle3 verify (draft 3) with partial acceptance",
        dict(batch=8, query_len=4, accept="partial", cu_kv=True, n_valid=True),
    ),
    ("without cu_kv_lens", dict(batch=4, query_len=4, accept="partial", cu_kv=False, n_valid=True)),
    (
        "CUDA-graph padding rows and no n_valid_blocks",
        dict(batch=8, query_len=4, accept="padded", cu_kv=True, n_valid=False),
    ),
    (
        "positions crossing a page boundary",
        dict(batch=3, query_len=4, accept="page_edge", cu_kv=True, n_valid=True),
    ),
    (
        "a batch past one thread block's 256 lanes",
        dict(batch=300, query_len=2, accept="partial", cu_kv=True, n_valid=True),
    ),
)


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


def _reference(args):
    return kv.reference(*args[:9], cu_kv=args[10] is not None, n_valid=args[12] is not None)


@pytest.mark.parametrize("why, spec", _CASES, ids=[c[0] for c in _CASES])
def test_the_kernel_matches_the_torch_formulation(why, spec) -> None:
    args = _case(spec, seed=zlib.crc32(why.encode()) % 2**31)
    assert kv.msa_kv_update(*args) is True
    _check(args, _reference(args))


def test_a_strided_operand_is_declined_and_nothing_is_written() -> None:
    """The kernel reads operands densely, so a strided one is declined (the metadata then runs its own torch
    update) with nothing written."""
    args = list(_case(_CASES[2][1], seed=5))
    args[0] = torch.stack([args[0], args[0]], dim=1)[:, 0]
    before = args[11].clone()
    assert kv.msa_kv_update(*args) is False
    assert torch.equal(args[11], before)


def test_a_non_int32_operand_is_declined() -> None:
    args = list(_case(_CASES[0][1], seed=6))
    args[2] = args[2].to(torch.int64)
    assert kv.msa_kv_update(*args) is False
