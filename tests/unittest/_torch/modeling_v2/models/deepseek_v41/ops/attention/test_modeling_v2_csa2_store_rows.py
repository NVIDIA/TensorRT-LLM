# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_store_rows entries: the engine's fused quantize-store (`csa2_store_rows`) and in-place
rope (`csa2_rope_inplace`) against the plain-torch references.

The fused kernel and the references may land one quantization step apart on rare ties, so the store is
gated on the decoded rows (max abs difference below 0.6, one fp8 / fp4 step at these magnitudes) and on
the share of differing bytes (below 1%); the in-place rope is gated at 0.05 absolute on bf16 values of
magnitude ~1 (one bf16 ulp is 2^-8 ~ 0.004, the rotation adds two products and a sum).
"""

import pytest
import torch
from _v41_util import DEV

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    csa2_unpack_rows,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rope import (  # noqa: E501
    csa2_rope_table,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rows_io import (  # noqa: E501
    _footer_offsets,
    csa2_gather_rows_footer,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_store_rows import (  # noqa: E501
    csa2_rope_inplace,
    csa2_stock_rope_table,
    csa2_store_rows,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the fused store kernels are the SM100 family's",
)


def _table(seq=4096):
    return csa2_rope_table(64, seq, 65536, 160000.0, 16.0, 32, 1, DEV)


@pytest.mark.parametrize(
    "cell", csa2_rope_inplace.CELLS, ids=[c.why[:44] for c in csa2_rope_inplace.CELLS]
)
def test_rope_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_rope_inplace)
    s = cell.spec
    torch.manual_seed(0)
    n, table = s["n"], _table()
    x = torch.randn(n, s["heads"], s["dim"], device=DEV).bfloat16()
    pos = torch.randint(0, 4000, (n,), device=DEV, dtype=torch.int32)
    cos_sin = csa2_stock_rope_table(table)
    ref = csa2_rope_inplace.reference(x, pos, cos_sin, s["inverse"])
    y = x.clone()
    got = csa2_rope_inplace(y, pos, cos_sin, s["inverse"])
    assert got is y, "the rotation is in place and returns its input"
    assert (y.float() - ref.float()).abs().max() < 0.05


def test_stock_table_is_cached_per_table():
    table = _table(256)
    assert csa2_stock_rope_table(table) is csa2_stock_rope_table(table)
    assert torch.equal(csa2_stock_rope_table(table)[:, :32], table[..., 0])
    assert torch.equal(csa2_stock_rope_table(table)[:, 32:], table[..., 1])


def test_stock_table_is_released_with_its_table():
    """The stock layout lives as long as the table it was built from: once a target drops its table, neither the
    table nor the converted copy stays allocated (a process-global cache kept both for every table ever seen)."""
    import gc
    import weakref

    table = _table(256)
    stock = weakref.ref(csa2_stock_rope_table(table))
    owner = weakref.ref(table)
    assert stock() is not None
    del table
    gc.collect()
    assert owner() is None and stock() is None


@pytest.mark.parametrize(
    "cell", csa2_store_rows.CELLS, ids=[c.why[:44] for c in csa2_store_rows.CELLS]
)
def test_store_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_store_rows)
    s = cell.spec
    torch.manual_seed(1)
    n, table = s["n"], _table()
    rows = 64 * 40
    pools = {
        k: torch.zeros(rows, w, dtype=torch.uint8, device=DEV)
        for k, w in (("swa", 528), ("main", 288))
    }
    pages = torch.zeros(rows // 64 * 64 * 68, dtype=torch.uint8, device=DEV)
    pos = torch.randint(0, 4000, (n,), device=DEV, dtype=torch.int32)
    slots = torch.randperm(rows, device=DEV)[: min(n, rows)].long()
    slots = torch.cat(
        [slots, torch.full((n - len(slots),), -1, device=DEV, dtype=torch.long)]
    )  # rows past the pool are skipped
    slots[::7] = -1
    vals = torch.randn(n, 512, device=DEV).bfloat16()
    kv = torch.randn(n, 128, device=DEV).bfloat16()
    w = torch.rand(512, device=DEV).bfloat16() + 0.5
    wk = torch.rand(128, device=DEV).bfloat16() + 0.5
    rs = csa2_stock_rope_table(table)
    q = torch.randn(n, 32, 128, device=DEV).bfloat16() if s["index_q"] else None

    def specs(p, g):
        swa = (p["swa"], slots, vals, (w, 1e-6), (pos, rs, 64)) if s.get("swa", True) else None
        main = (p["main"], slots, vals, None, (pos, rs, 64)) if s["main"] else None
        idx = (g, slots, kv, (wk, 1e-6), (pos, rs, 64)) if s["index"] else None
        return swa, main, idx, q

    got_pools = {k: v.clone() for k, v in pools.items()}
    got_pages = pages.clone()
    got_q = csa2_store_rows(*specs(got_pools, got_pages))
    ref_pools = {k: v.clone() for k, v in pools.items()}
    ref_pages = pages.clone()
    ref_q = csa2_store_rows.reference(*specs(ref_pools, ref_pages))
    for key in ("swa", "main"):
        if (key == "main" and not s["main"]) or (key == "swa" and not s.get("swa", True)):
            continue
        a = csa2_unpack_rows(got_pools[key], key, torch.float32)
        b = csa2_unpack_rows(ref_pools[key], key, torch.float32)
        assert (a - b).abs().max() < 0.6 and (
            got_pools[key] != ref_pools[key]
        ).float().mean() < 0.01, key
    if s["index"]:
        # every written index row decodes like the reference's (a skipped or wrong row fails here), and no byte
        # outside the written rows changes
        live = slots[slots >= 0]
        a = csa2_unpack_rows(csa2_gather_rows_footer(got_pages, live), "index", torch.float32)
        b = csa2_unpack_rows(csa2_gather_rows_footer(ref_pages, live), "index", torch.float32)
        # a quantization tie moves one element by at most one fp4 step (a third of the row's largest value), a skipped
        # row is off by the whole row
        assert bool(((a - b).abs().amax(-1) <= 0.5 * b.abs().amax(-1)).all())
        assert (a != b).any(dim=-1).float().mean() < 0.01
        d, sc = _footer_offsets(live)
        written = torch.zeros_like(got_pages, dtype=torch.bool)
        written[d.reshape(-1)] = True
        written[sc.reshape(-1)] = True
        assert torch.equal(got_pages[~written], pages[~written]) and torch.equal(
            ref_pages[~written], pages[~written]
        )
    if s["index_q"]:
        assert (got_q[0] != ref_q[0]).float().mean() < 0.01 and (
            got_q[1] != ref_q[1]
        ).float().mean() < 0.01
    else:
        assert got_q is None and ref_q is None


def test_rejections():
    n = 8
    pool = torch.zeros(64, 528, dtype=torch.uint8, device=DEV)
    slots = torch.arange(n, device=DEV)
    vals = torch.randn(n, 512, device=DEV).bfloat16()
    swa = (pool, slots, vals, None, None)
    csa2_store_rows.is_valid(swa)
    csa2_store_rows.is_valid(
        None, (pool[:, :288].contiguous(), slots, vals, None, None)
    )  # the main rows alone
    with pytest.raises(AssertionError, match="nothing"):
        csa2_store_rows.is_valid(None)
    with pytest.raises(AssertionError, match="bf16"):
        csa2_store_rows.is_valid((pool, slots, vals.float(), None, None))
    with pytest.raises(AssertionError, match="int64"):
        csa2_store_rows.is_valid((pool, slots.int(), vals, None, None))
    x = torch.zeros(4, 8, 512, device=DEV).bfloat16()
    cos_sin = csa2_stock_rope_table(_table(64))
    pos = torch.zeros(4, dtype=torch.int32, device=DEV)
    csa2_rope_inplace.is_valid(x, pos, cos_sin)
    with pytest.raises(AssertionError, match="contiguous"):
        csa2_rope_inplace.is_valid(x.transpose(0, 1), pos, cos_sin)
    with pytest.raises(AssertionError, match="int32"):
        csa2_rope_inplace.is_valid(x, pos.long(), cos_sin)


def test_a_table_in_the_engines_layout_is_kept_as_is():
    """The targets keep only the engine's `[P, rd]` layout of their RoPE tables (the `[P, rd/2, 2]` pair tables they are
    built from are not kept: 256 MiB each at a 1M-position checkpoint): converting such a table again returns it."""
    pairs = torch.randn(64, 32, 2, device=DEV)
    stock = csa2_stock_rope_table(pairs)
    assert stock.shape == (64, 64) and torch.equal(stock[:, :32], pairs[..., 0])
    assert csa2_stock_rope_table(stock) is stock
