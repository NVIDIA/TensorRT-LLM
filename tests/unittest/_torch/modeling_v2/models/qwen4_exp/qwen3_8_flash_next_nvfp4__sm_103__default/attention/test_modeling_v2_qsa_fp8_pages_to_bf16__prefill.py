# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for this target's qsa_fp8_pages_to_bf16 entry (`__prefill`).

The paged state is the engine's kv_fp8 trunk: e4m3 `[pages, 24, Hkv, tpb, D]`, layer 11's K
and V two of the role planes (a straddle cell sized so the requests' pages lie on both sides
of 2^31 elements of them), and the `[slots, 8196]` table at the tune's max_seq_len, the
listed requests on seeded distinct slots, their pages a seeded permutation, `-1` past each
sequence. Scratch and table are pre-filled with sentinels, so a missing or a stray write
shows. Every cell is bitwise `reference` (a widening copy), and the scratch is what the
e4m3 sparse GQA kernel reads: e4m3 codes at unit scale.
"""

import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.attention.qsa_fp8_pages_to_bf16__prefill import (  # noqa: E501
    qsa_fp8_pages_to_bf16 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_fp8_pages_to_bf16 requires a CUDA device"

DEV = "cuda"
SCRATCH_FILL = 0x7FC1  # a bf16 NaN
TABLE_FILL = -7
_EDGE = 2**31


def _pool_pages(spec, needed: int) -> int:
    if not spec["straddle"]:
        return needed + 5
    role_page = spec["kv_heads"] * spec["tokens_per_block"] * spec["head_dim"]
    edge_page = math.ceil(
        (_EDGE - 2 * spec["layer"] * role_page) / (spec["slot_pages"] * role_page)
    )
    return max(edge_page + needed // 2, needed) + 5


def _build(spec, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d = spec["tokens_per_block"], spec["kv_heads"], spec["head_dim"]
    counts = [math.ceil(n / tpb) for n in spec["kv_lens"]]
    needed = sum(counts) + 7
    pool_pages = _pool_pages(spec, needed)
    role_pages, layer = spec["slot_pages"], spec["layer"]
    # Random e4m3 codes (NaN codes 0x7F / 0xFF excluded), in the layer's two planes only.
    data = torch.zeros((pool_pages, role_pages, kv, tpb, d), dtype=torch.uint8, device=DEV)
    first = pool_pages - needed
    for p0 in range(first, pool_pages, 1024):
        p1 = min(p0 + 1024, pool_pages)
        codes = torch.randint(0, 254, (p1 - p0, 2, kv, tpb, d), generator=gen, device=DEV)
        codes = torch.where(codes >= 0x7F, codes + 1, codes)
        data[p0:p1, 2 * layer : 2 * layer + 2] = codes.to(torch.uint8)
    pool = data.view(torch.float8_e4m3fn)
    width, slots_total = spec["width"], spec["slots"]
    table = torch.full((slots_total, width), -1, dtype=torch.int32, device=DEV)
    slots = torch.randperm(slots_total, generator=torch.Generator().manual_seed(seed))
    slots = slots[: len(counts)]
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    cursor = 0
    for slot, count in zip(slots.tolist(), counts):
        table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count
    if spec["holes"]:
        head = slots[0].item()
        table[head, 1] = -1  # unmapped inside the sequence
        table[head, 2] = pool_pages + 3  # past the pool
    counts_t = torch.tensor(counts, dtype=torch.int64, device=DEV)
    listed = int(counts_t.sum())
    out_shape = (listed, kv, tpb, d)
    return dict(
        k_cache=pool[:, 2 * layer],
        v_cache=pool[:, 2 * layer + 1],
        block_table=table,
        slots=slots.to(DEV, torch.int32),
        counts=counts_t,
        offsets=torch.cumsum(counts_t, 0) - counts_t,
        table=torch.full_like(table, TABLE_FILL),
        k_out=torch.full(out_shape, SCRATCH_FILL, dtype=torch.int16, device=DEV).view(
            torch.bfloat16
        ),
        v_out=torch.full(out_shape, SCRATCH_FILL, dtype=torch.int16, device=DEV).view(
            torch.bfloat16
        ),
    )


def _bitwise(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype == torch.bfloat16:
        a, b = a.view(torch.int16), b.view(torch.int16)
    return a.shape == b.shape and torch.equal(a, b)


def _assert_straddles(args) -> None:
    table = args["block_table"]
    pages = table[(table >= 0) & (table < args["k_cache"].shape[0])].long()
    for name in ("k_cache", "v_cache"):
        plane = args[name]
        ends = plane.storage_offset() + pages * plane.stride(0) + plane[0].numel() - 1
        assert bool((ends >= _EDGE).any()) and bool((ends < _EDGE).any()), name


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 -- fixture by name
    certifying(op)
    args = _build(cell.spec, zlib.crc32(cell.why.encode()))
    if cell.spec["straddle"]:
        _assert_straddles(args)
    ref = op.reference(**{k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()})
    out = op(**args)
    for name, got, want in zip(("k_out", "v_out", "table"), out, ref):
        assert _bitwise(got, want), f"{name} differs from the reference"
    # Every listed page was written, and only the listed requests' table entries.
    assert not bool(out[0].isnan().any()) and not bool(out[1].isnan().any())
    listed_rows = torch.zeros_like(args["block_table"], dtype=torch.bool)
    for slot, count in zip(args["slots"].tolist(), args["counts"].tolist()):
        listed_rows[slot, :count] = True
    assert bool((out[2][~listed_rows] == TABLE_FILL).all())


def test_scratch_is_the_e4m3_values_at_unit_scale() -> None:
    """Every listed page's scratch holds the pool's e4m3 values exactly (e4m3 widens to bf16
    without rounding), and the table addresses it."""
    spec = op.CELLS[3].spec
    args = _build(spec, 11)
    k_out, v_out, table = op(**args)
    for slot, count, base in zip(
        args["slots"].tolist(), args["counts"].tolist(), args["offsets"].tolist()
    ):
        pages = args["block_table"][slot, :count].long()
        for got, plane in ((k_out, args["k_cache"]), (v_out, args["v_cache"])):
            want = plane[pages].float()
            assert torch.equal(got[base : base + count].float(), want)
        assert torch.equal(
            table[slot, :count], torch.arange(base, base + count, device=DEV, dtype=torch.int32)
        )


def test_listing_past_the_scratch_is_dropped() -> None:
    """A request whose pages run past `listed` writes only those below it."""
    args = _build(op.CELLS[3].spec, 5)
    short = args["k_out"].shape[0] - 3
    args["k_out"], args["v_out"] = args["k_out"][:short], args["v_out"][:short]
    ref = op.reference(**{k: v.clone() if torch.is_tensor(v) else v for k, v in args.items()})
    out = op(**args)
    for got, want in zip(out, ref):
        assert _bitwise(got, want)
