# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in 8-aligned row tile of ``router_gemm`` (``TRTLLM_ROUTER_GEMM_ALIGNED_TILE``).

The flag is read once at import into ``ALIGNED_ROW_TILE``, so the tests toggle that module
attribute. The tile only changes the MMA's N; the tile count, K splits and per-logit
accumulation order are unchanged, so the logits must match the default tile bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    router_gemm as mod,
)

# Rows that pick different default (power-of-two) and aligned tiles.
ROWS = (9, 44, 72, 128, 136, 200, 257)

# row -> (power-of-two tile, 8-aligned tile)
TILES = {
    1: (8, 8),
    8: (8, 8),
    9: (16, 16),
    44: (64, 48),
    72: (128, 72),
    128: (128, 128),
    136: (128, 72),
    200: (128, 104),
    257: (128, 88),
}


@pytest.mark.parametrize("rows", sorted(TILES))
def test_row_tile_values(monkeypatch, rows):
    pow2, aligned = TILES[rows]
    monkeypatch.setattr(mod, "ALIGNED_ROW_TILE", False)
    assert mod.row_tile(rows) == pow2
    monkeypatch.setattr(mod, "ALIGNED_ROW_TILE", True)
    assert mod.row_tile(rows) == aligned
    assert aligned % 8 == 0 and 8 <= aligned <= mod.ROW_TILE
    # The aligned tiles still cover every row of the call.
    assert aligned * ((rows + mod.ROW_TILE - 1) // mod.ROW_TILE) >= rows


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="router_gemm runs on SM 10.3 (GB300) only",
)
@pytest.mark.parametrize("rows", ROWS)
def test_aligned_tile_logits_bit_identical(monkeypatch, rows):
    torch.manual_seed(rows)
    hidden = torch.randn(rows, mod.HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(mod.EXPERTS, mod.HIDDEN, device="cuda", dtype=torch.float32) * 0.02

    tiles = []
    real_gemm = mod._gemm

    def spy(tile):
        tiles.append(tile)
        return real_gemm(tile)

    monkeypatch.setattr(mod, "_gemm", spy)

    monkeypatch.setattr(mod, "ALIGNED_ROW_TILE", False)
    default = mod.router_gemm(hidden, weight, 0.5)
    monkeypatch.setattr(mod, "ALIGNED_ROW_TILE", True)
    aligned = mod.router_gemm(hidden, weight, 0.5)
    torch.cuda.synchronize()

    # The tile reached the kernel compile; the two arms really ran different tiles.
    assert tiles == list(TILES[rows])
    if rows in (44, 72, 136, 200, 257):
        assert tiles[0] != tiles[1]

    assert default.shape == (rows, mod.EXPERTS)
    assert torch.isfinite(default).all()
    # Independent float64 check of the aligned arm (tolerance of the BF16x3 kernel test), so a
    # wrong result is told apart from a merely non-bitwise one.
    expected = mod.router_gemm.reference(hidden, weight, 0.5)
    torch.testing.assert_close(aligned, expected, rtol=1e-4, atol=2e-5)
    assert torch.equal(aligned, default)
