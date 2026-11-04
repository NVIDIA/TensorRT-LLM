# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ds_mla_dequant_gather catalog entry: bitwise against the reference
(e4m3 times a power of two is exact in bf16, and the rope half is a copy)."""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.ds_mla_dequant_gather import (
    ds_mla_dequant_gather as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

assert torch.cuda.is_available(), "ds_mla_dequant_gather requires a CUDA device"

#: Every pool byte the cell's requests do not hold: NaN as an e4m3 code, as an fp32 tile scale
#: and as a bf16 rope value, so a read from another page, owner row or offset shows in the bits.
_UNHELD = 0xFF
#: The int16 bits of every output-buffer element the call must leave alone (a bf16 NaN).
_UNWRITTEN = -1


def ds_mla_rows(n: int, g: torch.Generator, nope: bool = False) -> torch.Tensor:
    """`n` uint8 [656] rows on g's device: e4m3 codes drawn over (-448, 448), power-of-two
    tile scales 2^-13..2^0, bf16 rope (zero for a NoPE layer). The shared tests'
    `_ds_mla_pool.ds_mla_rows`, drawing where the generator lives."""
    dev = g.device
    codes = (torch.rand(n, 512, generator=g, device=dev) * 2 - 1) * 448
    codes = codes.to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.exp2(torch.randint(-13, 1, (n, 4), generator=g, device=dev).float())
    if nope:
        rope = torch.zeros(n, 64, device=dev)
    else:
        rope = torch.randn(n, 64, generator=g, device=dev) * 3
    return torch.cat([codes, scale.view(torch.uint8), rope.to(torch.bfloat16).view(torch.uint8)], 1)


def _build(spec, seed):
    """The call's arguments, built on the device, and the buffer `out` leads.

    The pool is `_UNHELD` but for the owner `row` of the pages the requests hold, which
    hold drawn rows: distinct pages anywhere in the pool, or with `far` its top ones. The
    block table has `table_rows` x `table_width` int32 entries (the requests' rows first,
    as the engine's context requests, a rank's under attention DP, lead md.block_table),
    random block ids past the requests' pages. `out` is the first N rows of a
    `_UNWRITTEN` buffer of `out_rows`.
    """
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    tpb, blocks, owners = spec["tokens_per_block"], spec["blocks"], spec["owners"]
    totals = [c + n for c, n in zip(spec["cached"], spec["new"])]
    counts = [math.ceil(t / tpb) for t in totals]
    need = sum(counts)
    if spec.get("far"):
        pages = blocks - need + torch.randperm(need, generator=g, device=dev)
    else:
        pages = torch.randperm(blocks, generator=g, device=dev)[:need]
    rows_, width = spec.get("table_rows", len(totals)), spec.get("table_width", max(counts))
    table = torch.randint(0, blocks, (rows_, width), generator=g, device=dev, dtype=torch.int32)
    used = 0
    for r, k in enumerate(counts):
        table[r, :k] = pages[used : used + k]
        used += k
    pool = torch.full((blocks, owners, tpb, 656), _UNHELD, dtype=torch.uint8, device=dev)
    pool[pages, spec["row"]] = ds_mla_rows(need * tpb, g, spec.get("nope", False)).view(
        need, tpb, 656
    )
    n = sum(totals)
    lens = torch.tensor(totals, device=dev)
    kv_req = torch.repeat_interleave(
        torch.arange(len(totals), dtype=torch.int32, device=dev), lens, output_size=n
    )
    kv_pos = (torch.arange(n, device=dev) - (torch.cumsum(lens, 0) - lens)[kv_req]).int()
    buf = torch.full((spec.get("out_rows", n), 576), _UNWRITTEN, dtype=torch.int16, device=dev)
    buf = buf.view(torch.bfloat16)
    return (pool, spec["row"], kv_req, kv_pos, table, buf[:n]), buf


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor) -> None:
    assert out.shape == ref.shape and out.dtype == ref.dtype, (out.shape, ref.shape)
    bad = _bits(out) != _bits(ref)
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} elements differ"


def _untouched(rows: torch.Tensor) -> bool:
    return bool((_bits(rows) == _UNWRITTEN).all())


def _prefix_counts(n: int) -> list[int]:
    """The row counts below `n` the prefix check drives: 1..64, 2**k - 1 / 2**k / 2**k + 1
    from 2**7, strides of 4093 and of 4099 (the tep4 and the tp4 cells' certifying runs
    each drove one; the tep8 cells' drove 4093), and n - 1."""
    counts = {*range(1, 65), *(2**k + d for k in range(7, 20) for d in (-1, 0, 1))}
    counts |= {*range(4093, n, 4093), *range(4099, n, 4099), n - 1}
    return sorted(m for m in counts if 0 < m < n)


def _check_prefixes(args, full: torch.Tensor) -> int:
    """Each count of `_prefix_counts(N)` gathers the leading kv_req / kv_pos rows into a fresh
    `_UNWRITTEN` buffer: its rows are the full call's leading rows bitwise and the buffer's
    later rows stay unwritten. One sync for all counts; a failure re-runs them one by one to
    name the count. Returns how many counts ran."""
    pool, row, kv_req, kv_pos, table, _ = args
    scratch = torch.empty_like(full)
    counts = _prefix_counts(full.shape[0])
    bad = torch.zeros((), dtype=torch.bool, device=full.device)
    for m in counts:
        _bits(scratch).fill_(_UNWRITTEN)
        got = op(pool, row, kv_req[:m], kv_pos[:m], table, scratch[:m])
        bad |= (_bits(got) != _bits(full[:m])).any() | (_bits(scratch[m:]) != _UNWRITTEN).any()
    if bad.item():
        for m in counts:
            _bits(scratch).fill_(_UNWRITTEN)
            op(pool, row, kv_req[:m], kv_pos[:m], table, scratch[:m])
            _assert_bitwise(scratch[:m], full[:m])
            assert _untouched(scratch[m:]), f"N = {m}: rows past N were written"
    return len(counts)


_CELLS = [
    pytest.param(c, id=c.why[:60], marks=[pytest.mark.large] if c.spec.get("large") else [])
    for c in op.CELLS
]


@pytest.mark.parametrize("cell", _CELLS)
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args, buf = _build(cell.spec, zlib.crc32(cell.why.encode()))
    n = args[2].shape[0]
    out = op(*args)
    ref = op.reference(*args)
    assert out.data_ptr() == buf.data_ptr()
    _assert_bitwise(out, ref)
    assert not torch.isnan(ref.float()).any(), "the reference read a page the requests lack"
    del ref
    assert _untouched(buf[n:]), "the call wrote rows of the buffer past N"
    if cell.spec.get("prefixes"):
        assert _check_prefixes(args, out) > 64


def test_is_valid_refuses_a_bf16_pool() -> None:
    args, _ = _build(dict(cached=[0], new=[64], blocks=4, owners=2, tokens_per_block=32, row=1), 1)
    args = list(args)
    args[0] = args[0].view(torch.bfloat16)
    with pytest.raises(ValueError, match="uint8"):
        op.is_valid(*args)


def test_is_valid_refuses_an_out_shorter_than_kv_req() -> None:
    """Unguarded, an `out` of 40 rows for 64 kv_req entries comes back right for its 40 rows,
    and the kernel writes the other 24 past its end: here into the rest of the 64-row buffer
    the test owns, where they are the reference's rows 40..63. Guarded, the call raises
    before the kernel runs."""
    args, buf = _build(
        dict(cached=[5], new=[59], blocks=6, owners=3, tokens_per_block=32, row=2), 2
    )
    pool, row, kv_req, kv_pos, table, _ = args
    ref = op.reference(*args)
    short = op(pool, row, kv_req, kv_pos, table, buf[:40])
    assert short.shape[0] == 40
    _assert_bitwise(short, ref[:40])
    _assert_bitwise(buf[40:], ref[40:])
    _bits(buf).fill_(_UNWRITTEN)
    with validating(op), pytest.raises(ValueError, match="40 rows for 64 kv_req entries"):
        op(pool, row, kv_req, kv_pos, table, buf[:40])
    assert _untouched(buf), "refused, yet the kernel wrote"
    with validating(op):
        _assert_bitwise(op(*args), ref)


def test_is_valid_refuses_48_out_rows_for_64_kv_req_entries() -> None:
    """Unguarded, an `out` of 48 rows for 64 kv_req entries comes back right for its 48 rows
    while the kernel writes the other 16 past its end: here into the rest of the 64-row
    buffer the test owns, so the stray rows land where the test can read them."""
    args, buf = _build(
        dict(cached=[0], new=[64], blocks=4, owners=2, tokens_per_block=32, row=1), 2
    )
    pool, row, kv_req, kv_pos, table, _ = args
    ref = op.reference(*args)
    short = op(pool, row, kv_req, kv_pos, table, buf[:48])
    assert short.shape[0] == 48
    _assert_bitwise(short, ref[:48])
    _assert_bitwise(buf[48:], ref[48:])
    _bits(buf).fill_(_UNWRITTEN)
    with validating(op), pytest.raises(ValueError, match="48 rows for 64 kv_req"):
        op(pool, row, kv_req, kv_pos, table, buf[:48])
    assert _untouched(buf), "refused, yet the kernel ran"
    with validating(op):
        _assert_bitwise(op(*args), ref)
