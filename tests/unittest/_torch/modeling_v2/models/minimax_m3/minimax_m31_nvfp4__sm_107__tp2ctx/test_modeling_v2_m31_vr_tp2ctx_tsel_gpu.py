# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate a for TP2CTX_TSEL on sm_107: the tile select's table (tsel.select -> minimax_m31_select_prefill_tile) equals
``minimax_m3_select_blocks(scores, n_valid_blocks, 16, init, local, head_major=True)`` BIT FOR BIT.

The op's contract (selectPrefillTileSm103.h): per (query, head) row, the clamp(nValid, 0, numBlocks) leading blocks; the
last localBlocks (from the unclamped count) forced to 1e29 and the first initBlocks to 1e30, local winning; ranked by
the descending ordered key of the float bits (NaN above +inf, +0 above -0, negative NaN below -inf), ties to the smaller
block; -inf and negative-NaN picks -> -1; ascending ids, -1 tail. The adversarial inputs below hit every clause: ties,
+-0, both NaN signs, +-inf, all-equal rows, short rows, the 5856-block shared-memory bound and the 256-row gate. Skipped
off sm_107 or without the built op.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    tsel,
)

BLOCK = 128
on_vr = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
needs_op = pytest.mark.skipif(
    not on_vr or tsel._op() is None, reason="sm_107 with minimax_m31_select_prefill_tile only"
)
POS_NAN, NEG_NAN = 0x7FC00000, -0x00400000  # 0xFFC00000 as int32


def _nvb(ctx_end, rows):
    pos = torch.arange(ctx_end - rows, ctx_end, device="cuda")
    return ((pos + BLOCK) // BLOCK).to(torch.int32)


def _scores(kind, heads, blocks, rows, seed):
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
        special = torch.tensor(
            [0x7F800000, -0x00800000, POS_NAN, NEG_NAN, 0x00000000, -0x80000000],  # +inf -inf +nan -nan +0 -0
            dtype=torch.int32, device="cuda")  # fmt: skip
        bits = torch.where(pick < 6, special[pick.clamp(max=5)], bits)
        return bits.view(torch.float32)
    raise ValueError(kind)


def _check(scores, nvb, init_blocks, local_blocks):
    scores = scores.contiguous()
    got = tsel.select(scores, nvb, init_blocks, local_blocks)
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
    blocks = tsel.TSEL_MAX_BLOCKS  # 5856: the largest table the op takes
    _check(_scores(kind, 1, blocks, 256, 11), _nvb(blocks * BLOCK, 256), 2, 3)


@needs_op
def test_host_and_int64_n_valid_blocks():
    scores = _scores("random", 1, 64, 512, 5)
    nvb = _nvb(8192, 512)
    ref = torch.ops.trtllm.minimax_m3_select_blocks(scores, nvb, 16, 0, 1, True)
    assert torch.equal(tsel.select(scores, nvb.cpu().long(), 0, 1), ref)


@pytest.mark.skipif(not on_vr, reason="sm_107 only")
def test_the_gate_declines_below_256_rows_and_above_the_block_bound():
    assert (
        tsel.select(_scores("random", 1, 64, tsel.TSEL_MIN_ROWS - 1, 3), _nvb(8192, 255), 0, 1)
        is None
    )
    wide = torch.zeros(1, tsel.TSEL_MAX_BLOCKS + 1, 512, device="cuda")
    assert tsel.select(wide, torch.ones(512, dtype=torch.int32, device="cuda"), 0, 1) is None
