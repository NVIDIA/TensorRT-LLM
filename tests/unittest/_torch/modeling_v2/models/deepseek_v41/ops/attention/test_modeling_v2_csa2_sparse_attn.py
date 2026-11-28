# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_sparse_attn entry against a dense, token-at-a-time transcription of the reference.

Both sides run under IEEE fp32 matmuls. The op accumulates in fp32, the reference in float64, and
each side rounds the softmax numerators to bf16 before the value product like the checkpoint's
kernel. The two differ only where a numerator lands within an fp32 ulp of a bf16 rounding tie
(a one-bf16-ulp flip of one probability, which moves the output by at most 2^-9 of that term), and
by the bf16 rounding of the output itself. Gate: 2 bf16 ulp of each row's largest value
elementwise, 0.5 ulp relative RMS.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _precision import true_fp32_matmul  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    csa2_pack_rows,
    csa2_unpack_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rows_io import (  # noqa: E501
    csa2_row_addr,
    csa2_scatter_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_sparse_attn import (  # noqa: E501
    csa2_sparse_attn,
)

D, WIN = 512, 128


def _build(spec, cell):
    """Pools and inputs of one request that has `hist` tokens of history and runs `n` more."""
    gen = gen_for(cell)
    hist, n, ratio, heads = spec["hist"], spec["n"], spec["ratio"], spec["heads"]
    total = hist + n
    kv_all = csa2_unpack_rows.reference(
        csa2_pack_rows.reference(torch.randn(total, D, generator=gen, device=DEV), "swa"), "swa"
    )  # the rows the model would have written: already rotated and quantize-dequantized
    nblk = (total + 127) // 128 + 1
    swa_pool = torch.zeros(nblk + 2, 128, 528, dtype=torch.uint8, device=DEV)
    swa_table = torch.arange(nblk, dtype=torch.int32, device=DEV).flip(0).unsqueeze(0) + 1
    zeros = lambda t: torch.zeros_like(t, dtype=torch.long)  # noqa: E731
    pos_hist = torch.arange(max(0, hist - WIN), hist, device=DEV)
    if len(pos_hist):
        a = csa2_row_addr.reference(swa_table, zeros(pos_hist), pos_hist, 128)
        csa2_scatter_rows.reference(
            swa_pool,
            a,
            csa2_pack_rows.reference(kv_all[pos_hist], "swa"),
            torch.ones_like(a, dtype=torch.bool),
        )
    main_pool = main_table = topk = None
    if ratio:
        ncomp = total // ratio
        rpp = 128 // ratio
        mblk = (ncomp + rpp - 1) // rpp + 1
        main_pool = torch.zeros(mblk + 2, rpp, 288, dtype=torch.uint8, device=DEV)
        main_table = torch.arange(mblk, dtype=torch.int32, device=DEV).flip(0).unsqueeze(0) + 2
        rows = torch.randn(ncomp, D, generator=gen, device=DEV)
        idx = torch.arange(ncomp, device=DEV)
        a = csa2_row_addr.reference(main_table, zeros(idx), idx, rpp)
        csa2_scatter_rows.reference(
            main_pool,
            a,
            csa2_pack_rows.reference(rows, "main"),
            torch.ones_like(a, dtype=torch.bool),
        )
        k = min(8, ncomp)
        topk = torch.stack(
            [torch.randperm(ncomp, generator=gen, device=DEV)[:k].sort().values for _ in range(n)]
        ).int()
        topk[0, -1] = -1  # an unreachable pick
    pos = torch.arange(hist, total, device=DEV, dtype=torch.int32)
    cs = torch.full((n,), hist, device=DEV, dtype=torch.int32)
    rid = torch.zeros(n, dtype=torch.long, device=DEV)
    q = torch.randn(n, heads, D, generator=gen, device=DEV).bfloat16()
    sink = torch.randn(heads, generator=gen, device=DEV)
    return (
        q,
        kv_all[hist:],
        sink,
        pos,
        cs,
        rid,
        swa_pool,
        swa_table,
        WIN,
        D**-0.5,
        main_pool,
        main_table,
        topk,
        ratio,
        hist > 0,
    )


@pytest.mark.parametrize(
    "cell", csa2_sparse_attn.CELLS, ids=[c.why[:44] for c in csa2_sparse_attn.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_sparse_attn)
    args = _build(cell.spec, cell)
    with true_fp32_matmul():
        got = csa2_sparse_attn(*args)
        ref = csa2_sparse_attn.reference(*args)
    assert got.dtype == torch.bfloat16 and got.shape == args[0].shape
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.5)


def test_floor_masks_window_rows_below_it(certifying):  # noqa: F811
    cell = csa2_sparse_attn.CELLS[3]  # ratio 1, 500 tokens of history, a 70-row chunk
    args = _build(cell.spec, cell)
    floor = torch.full_like(
        args[3], 440, dtype=torch.int32
    )  # rows of the pool window below 440 are not readable
    with true_fp32_matmul():
        got = csa2_sparse_attn(*args, floor=floor)
        ref = csa2_sparse_attn.reference(*args, floor=floor)
        plain = csa2_sparse_attn(*args)
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.5)
    assert (got.float() - plain.float()).abs().max() > 1e-2, "the floor must change the result"
    with pytest.raises(AssertionError, match="floor"):
        csa2_sparse_attn.is_valid(*args, floor=floor[:5])


class _Case:
    """A cell-like spec outside the certified cells (`gen_for` seeds from `why`)."""

    def __init__(self, why, **spec):
        self.why, self.spec = why, spec


def test_unmapped_history_page_is_no_key():
    """A window row of an earlier step on an unmapped SWA page is excluded from the softmax (it used to be admitted as
    a zero key with a zero logit, changing the denominator; the reference read a row from the end of the pool): the
    result equals the mapped table's with those positions floored away."""
    case = _Case(
        "decode at 300 with SWA page 1 (positions 128-255) unmapped",
        hist=300,
        n=3,
        ratio=0,
        heads=8,
    )
    args = list(_build(case.spec, case))
    unmapped = args[7].clone()
    unmapped[0, 1] = -1
    floor = torch.full_like(
        args[3], 256, dtype=torch.int32
    )  # the same keys, masked by position instead
    with true_fp32_matmul():
        got = csa2_sparse_attn(*args[:7], unmapped, *args[8:])
        ref = csa2_sparse_attn.reference(*args[:7], unmapped, *args[8:])
        want = csa2_sparse_attn(*args, floor=floor)
        plain = csa2_sparse_attn(*args)
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.5)
    assert torch.equal(got, want)
    assert (got.float() - plain.float()).abs().max() > 1e-2, (
        "the unmapped page must change the result"
    )


def test_step_rows_do_not_depend_on_the_page_table():
    """A chunk starting on a page boundary (position 256) whose own page is unmapped: its rows come from `kv`, so the
    result is the mapped table's."""
    case = _Case("chunk at 256 whose own SWA page 2 is unmapped", hist=256, n=3, ratio=0, heads=8)
    args = list(_build(case.spec, case))
    unmapped = args[7].clone()
    unmapped[0, 2] = -1
    with true_fp32_matmul():
        got = csa2_sparse_attn(*args[:7], unmapped, *args[8:])
        want = csa2_sparse_attn(*args)
        ref = csa2_sparse_attn.reference(*args[:7], unmapped, *args[8:])
    assert torch.equal(got, want)
    assert_within_ulp(got, ref, element_ulp=2.0, rms_ulp=0.5)


def test_rejections():
    cell = csa2_sparse_attn.CELLS[2]
    args = list(_build(cell.spec, cell))
    csa2_sparse_attn.is_valid(*args)
    bad = list(args)
    bad[6] = torch.zeros(4, 64, 528, dtype=torch.uint8, device=DEV)  # 64-row SWA pages
    with pytest.raises(AssertionError, match="128"):
        csa2_sparse_attn.is_valid(*bad)
    bad = list(args)
    bad[10] = torch.zeros(4, 128, 288, dtype=torch.uint8, device=DEV)  # ratio 2 needs 64-row pages
    with pytest.raises(AssertionError, match="main pool"):
        csa2_sparse_attn.is_valid(*bad)
    bad = list(args)
    bad[14] = False  # history exists (cs > 0) but the SWA pool would be skipped
    with pytest.raises(AssertionError, match="history"):
        csa2_sparse_attn.is_valid(*bad)
    bad = list(args)
    bad[12] = None  # a main pool with no top-k indices is ignored
    with pytest.raises(AssertionError, match="ignored"):
        csa2_sparse_attn.is_valid(*bad)


def test_fully_masked_row_gives_zeros():
    """A row whose every key is masked (here by a floor past its position) gives zeros in the op and the reference, as
    the kernel (the reference used to give NaN)."""
    cell = csa2_sparse_attn.CELLS[1]  # decode, window only
    args = _build(cell.spec, cell)
    floor = (args[3] + 1).int()
    got = csa2_sparse_attn(*args, floor=floor)
    ref = csa2_sparse_attn.reference(*args, floor=floor)
    assert not got.any() and not ref.isnan().any() and not ref.any()
