# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_TSEL on sm_107: the tile select's table (tile_select.select -> minimax_m31_select_prefill_tile) equals
``minimax_m3_select_blocks(scores, n_valid_blocks, 16, init, local, head_major=True)`` BIT FOR BIT.

The op's contract (selectPrefillTileSm103.h): per (query, head) row, the clamp(nValid, 0, numBlocks) leading blocks; the
last localBlocks (from the unclamped count) forced to 1e29 and the first initBlocks to 1e30, local winning; ranked by
the descending ordered key of the float bits (NaN above +inf, +0 above -0, negative NaN below -inf), ties to the smaller
block; -inf and negative-NaN picks -> -1; ascending ids, -1 tail. The adversarial inputs below hit every clause: ties,
+-0, both NaN signs, +-inf, all-equal rows, short rows, the 5856-block shared-memory bound and the 256-row gate. Skipped
off sm_107.

One exception to the bitwise equality: from 129 blocks select_blocks ranks negative NaN above finite scores, against the
contract above; test_negative_nan_above_128_blocks pins that as a strict xfail.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2ctx import (
    tile_select,
)

BLOCK = 128
on_vr = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
needs_op = pytest.mark.skipif(not on_vr, reason="SM107 (Rubin) only")
POS_NAN, NEG_NAN = 0x7FC00000, -0x00400000  # 0xFFC00000 as int32
POS_NAN_MAX = 0x7FFFFFFF  # ordered key 0, the largest representable rank
I64_MAX = torch.iinfo(torch.int64).max


def _nvb(ctx_end, rows):
    pos = torch.arange(ctx_end - rows, ctx_end, device="cuda")
    return ((pos + BLOCK) // BLOCK).to(torch.int32)


def _scores(kind, heads, blocks, rows, seed, neg_nan=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "random":
        return torch.randn(heads, blocks, rows, generator=g, device="cuda")
    if kind == "ties":
        return torch.randint(0, 8, (heads, blocks, rows), generator=g, device="cuda").float()
    if kind == "all_equal":
        return torch.full((heads, blocks, rows), 0.5, device="cuda")
    if kind == "signed_zero":
        s = torch.zeros(heads, blocks, rows, device="cuda")
        neg = torch.rand(heads, blocks, rows, generator=g, device="cuda") < 0.5
        return torch.where(neg, -s, s)
    if kind == "specials":  # +-inf, both NaN signs, +-0 sprinkled into a tie-heavy field
        s = torch.randint(-3, 4, (heads, blocks, rows), generator=g, device="cuda").float()
        bits = s.view(torch.int32)
        pick = torch.randint(0, 12, (heads, blocks, rows), generator=g, device="cuda")
        nnan = NEG_NAN if neg_nan else POS_NAN
        special = torch.tensor(
            [0x7F800000, -0x00800000, POS_NAN, nnan, 0x00000000, -0x80000000],  # +inf -inf +nan -nan +0 -0
            dtype=torch.int32, device="cuda")  # fmt: skip
        bits = torch.where(pick < 6, special[pick.clamp(max=5)], bits)
        return bits.view(torch.float32)
    if kind == "neg_nan":  # negative NaN in a finite field: it must rank below -inf
        s = torch.randn(heads, blocks, rows, generator=g, device="cuda")
        pick = torch.rand(heads, blocks, rows, generator=g, device="cuda") < 0.15
        return torch.where(
            pick, torch.tensor(NEG_NAN, dtype=torch.int32, device="cuda").view(torch.float32), s
        )
    if (
        kind == "lane_skew"
    ):  # every 8th block wins: one lane of a narrow row owns the whole top 16 (its rescan)
        s = torch.randn(heads, blocks, rows, generator=g, device="cuda")
        s[:, ::8, :] += 10.0
        return s
    raise ValueError(kind)


def _check(scores, nvb, init_blocks, local_blocks):
    scores = scores.contiguous()
    got = tile_select.select(scores, nvb, init_blocks, local_blocks)
    assert got is not None, "the tile select declined a step inside its range"
    ref = torch.ops.trtllm.minimax_m3_select_blocks(
        scores, nvb, 16, init_blocks, local_blocks, True
    )
    assert (
        got.shape == ref.shape
        and got.stride() == ref.stride()
        and got.dtype == ref.dtype == torch.int32
    )
    diff = (got != ref).any(dim=-1)
    assert not diff.any(), f"{int(diff.sum())} rows differ; first {diff.nonzero()[:4].tolist()}"


def _key(bits):
    bits = bits.to(torch.int64) & 0xFFFFFFFF
    return torch.where(bits >= 0x80000000, bits, ~bits & 0x7FFFFFFF)


def _oracle(scores, nvb, init_blocks, local_blocks):
    """The contract above as a host key sort, independent of both kernels and of select_blocks: (rows, heads, 16)."""
    heads, blocks, rows = scores.shape
    key = _key(scores.cpu().contiguous().view(torch.int32))
    raw = nvb.cpu().to(torch.int64).view(1, 1, rows)
    b = torch.arange(blocks).view(1, blocks, 1)
    forced = _key(torch.tensor([1.0e30, 1.0e29]).view(torch.int32))
    key = torch.where(b < init_blocks, forced[0], key)
    key = torch.where(b >= (raw - local_blocks).clamp(min=0), forced[1], key)
    length = raw.clamp(0, blocks)
    top = torch.where(b < length, (key << 31) | b, I64_MAX).sort(dim=1).values[:, :16, :]
    valid = (torch.arange(16).view(1, 16, 1) < length.clamp(max=16)) & ((top >> 31) < 0xFF800000)
    ids = torch.where(valid, top & 0x7FFFFFFF, I64_MAX).sort(dim=1).values
    return torch.where(ids == I64_MAX, -1, ids).permute(2, 0, 1).to(torch.int32)


@needs_op
@pytest.mark.parametrize("kind", ["random", "ties", "all_equal", "signed_zero", "specials"])
@pytest.mark.parametrize(
    "ctx_end,rows", [(16384, 16384), (165000, 1160), (65536, 4096), (2048, 256)]
)
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 3), (4, 8)])
def test_table_equals_select_blocks_bitwise(kind, ctx_end, rows, init_blocks, local_blocks):
    blocks = (ctx_end + BLOCK - 1) // BLOCK
    _check(
        _scores(kind, 1, blocks, rows, ctx_end + rows),
        _nvb(ctx_end, rows),
        init_blocks,
        local_blocks,
    )


@needs_op
@pytest.mark.parametrize("kind", ["random", "specials"])
def test_two_kv_heads_and_short_rows(kind):
    """Early rows have fewer than 16 valid blocks (-1 tail); two KV heads keep the head-major backing."""
    _check(_scores(kind, 2, 16, 2048, 7), _nvb(2048, 2048), 0, 1)


@needs_op
@pytest.mark.parametrize("kind", ["random", "ties", "specials"])
def test_the_shared_memory_bound(kind):
    blocks = tile_select.TSEL_MAX_BLOCKS  # 5856: the largest table the op takes
    _check(_scores(kind, 1, blocks, 256, 11), _nvb(blocks * BLOCK, 256), 2, 3)


@needs_op
@pytest.mark.parametrize("kind", ["random", "ties", "all_equal", "specials", "lane_skew"])
@pytest.mark.parametrize("blocks", [17, 64, 127, 128, 129])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (4, 8)])
def test_narrow_row_dispatch_boundary(kind, blocks, init_blocks, local_blocks):
    """Up to 128 blocks the op takes the narrow-row kernel, from 129 the tile kernel; both sides of that boundary and
    the narrow rescan must give the same table. 4 KV heads and an odd row count
    leave a partial last CTA."""
    rows = 1001
    _check(
        # select_blocks mis-ranks negative NaN from 129 blocks (test_negative_nan_above_128_blocks).
        _scores(kind, 4, blocks, rows, blocks * 7 + init_blocks, neg_nan=blocks <= 128),
        _nvb(blocks * BLOCK, rows),
        init_blocks,
        local_blocks,
    )


@needs_op
@pytest.mark.parametrize("payload", [POS_NAN_MAX, POS_NAN])
@pytest.mark.parametrize("blocks", [121, 128])
def test_lane_skew_positive_nan_extremum(payload, blocks):
    """Lane 0 of each narrow row owns every winner (blocks 0, 8, ...), so the narrow kernel takes its exact rescan.
    Positive NaN 0x7fffffff has ordered key 0, so block 0's rank is the largest representable one and the rescan must
    still take it; POS_NAN is the ordinary-payload control. Checked against the host key sort."""
    rows = 1001
    s = torch.randn(
        4, blocks, rows, generator=torch.Generator(device="cuda").manual_seed(blocks), device="cuda"
    )
    s.view(torch.int32)[:, ::8, :] = payload
    nvb = _nvb(blocks * BLOCK, rows)
    got = tile_select.select(s, nvb, 0, 1)
    assert got is not None, "the tile select declined a step inside its range"
    want = _oracle(s, nvb, 0, 1)
    diff = (got.cpu() != want).any(dim=-1)
    assert not diff.any(), f"{int(diff.sum())} rows differ; first {diff.nonzero()[:4].tolist()}"


@needs_op
@pytest.mark.parametrize("blocks", [64, 128])
def test_negative_nan_at_most_128_blocks(blocks):
    _check(_scores("neg_nan", 4, blocks, 1001, blocks), _nvb(blocks * BLOCK, 1001), 0, 1)


@needs_op
@pytest.mark.xfail(
    strict=True,
    reason="minimax_m3_select_blocks ranks negative NaN above finite scores from 129 blocks; the tile select "
    "ranks it below -inf as documented",
)
@pytest.mark.parametrize("blocks", [129, 256, 1290])
def test_negative_nan_above_128_blocks(blocks):
    _check(_scores("neg_nan", 1, blocks, 1001, blocks), _nvb(blocks * BLOCK, 1001), 0, 1)


@needs_op
def test_host_and_int64_n_valid_blocks():
    scores = _scores("random", 1, 64, 512, 5)
    nvb = _nvb(8192, 512)
    ref = torch.ops.trtllm.minimax_m3_select_blocks(scores, nvb, 16, 0, 1, True)
    assert torch.equal(tile_select.select(scores, nvb.cpu().long(), 0, 1), ref)


@pytest.mark.skipif(not on_vr, reason="sm_107 only")
def test_the_gate_declines_below_256_rows_and_above_the_block_bound():
    assert (
        tile_select.select(
            _scores("random", 1, 64, tile_select.TSEL_MIN_ROWS - 1, 3), _nvb(8192, 255), 0, 1
        )
        is None
    )
    wide = torch.zeros(1, tile_select.TSEL_MAX_BLOCKS + 1, 512, device="cuda")
    assert tile_select.select(wide, torch.ones(512, dtype=torch.int32, device="cuda"), 0, 1) is None
