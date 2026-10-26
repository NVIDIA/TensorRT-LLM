# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp vision tower's call grouping.

A request's frames run in calls of whole frames under `_VISION_CALL_MAX_PATCHES` rows.
The tower here has the checkpoint's geometry (1152 wide, 16 heads of 72, 2x2 merge) with
two random blocks, so every op the real tower runs runs here; the cap is a module
constant, lowered so a few hundred rows span several calls. The reference is the same
tower in one call.

Cutting between frames leaves attention, the position embedding, the rope and every
row-wise op unchanged; what can differ is which kernel a call picks (a GEMM's split, the
fused attention kernel for a call whose longest frame is at most 64 rows), within the ops'
reference bands.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)

assert torch.cuda.is_available(), "the vision tower requires a CUDA device"

HIDDEN = 1152
HEADS = 16
HEAD_DIM = HIDDEN // HEADS
INTER = 4304
MERGE = 2
MERGE_DIM = HIDDEN * MERGE * MERGE
OUT = 2560
POS_ROWS = 48 * 48
PIXEL_WIDTH = 3 * 2 * 16 * 16
DEPTH = 2
DT = torch.bfloat16


def _core():
    g = torch.Generator(device="cuda").manual_seed(0)

    def w(*shape, scale=0.02):
        return (scale * torch.randn(*shape, generator=g, device="cuda")).to(DT)

    def norm_w():
        return (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(DT)

    weights = {
        "embed": torch.zeros(1, dtype=DT, device="cuda"),
        "vis_patch_w": w(HIDDEN, 3, 2, 16, 16),
        "vis_patch_b": w(HIDDEN),
        "vis_pos": w(POS_ROWS, HIDDEN, scale=0.5),
        "vis_m_n_w": norm_w(),
        "vis_m_n_b": w(HIDDEN),
        "vis_m_fc1_w": w(MERGE_DIM, MERGE_DIM),
        "vis_m_fc1_b": w(MERGE_DIM),
        "vis_m_fc2_w": w(OUT, MERGE_DIM),
        "vis_m_fc2_b": w(OUT),
    }
    for b in range(DEPTH):
        weights.update(
            {
                f"l{b}_vis_n1_w": norm_w(),
                f"l{b}_vis_n1_b": w(HIDDEN),
                f"l{b}_vis_qkv_w": w(3 * HIDDEN, HIDDEN),
                f"l{b}_vis_qkv_b": w(3 * HIDDEN),
                f"l{b}_vis_proj_w": w(HIDDEN, HIDDEN),
                f"l{b}_vis_proj_b": w(HIDDEN),
                f"l{b}_vis_n2_w": norm_w(),
                f"l{b}_vis_n2_b": w(HIDDEN),
                f"l{b}_vis_fc1_w": w(INTER, HIDDEN),
                f"l{b}_vis_fc1_b": w(INTER),
                f"l{b}_vis_fc2_w": w(HIDDEN, INTER),
                f"l{b}_vis_fc2_b": w(HIDDEN),
            }
        )
    return SimpleNamespace(
        w=weights,
        v_depth=DEPTH,
        v_hidden=HIDDEN,
        v_heads=HEADS,
        v_head_dim=HEAD_DIM,
        v_merge=MERGE,
        v_merge_dim=MERGE_DIM,
        v_pixel_width=PIXEL_WIDTH,
    )


def _pixels(grid, seed=1):
    patches = sum(t * h * w for t, h, w in grid)
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.empty(patches, PIXEL_WIDTH, device="cuda").uniform_(-1, 1, generator=g).to(DT)


def _count_calls(tower, monkeypatch):
    """Wrap `tower._run`; the returned list gets one `(rows, grid)` entry per call."""
    calls = []
    run = tower._run

    def counted(pixel_values, grid, grid_thw):
        calls.append((pixel_values.shape[0], list(grid)))
        return run(pixel_values, grid, grid_thw)

    monkeypatch.setattr(tower, "_run", counted)
    return calls


def _assert_close(out, ref):
    """Two runs of the tower over different row counts differ by bf16 rounding: another
    call picks another GEMM kernel, and the difference compounds through the blocks and the
    merger. Measured here: 0.6% RMS relative error and at most 8 ulps (of the RMS) in one
    element, against 117% RMS between unrelated rows. The gates sit between the two."""
    assert out.dtype == ref.dtype and out.shape == ref.shape, (out.shape, ref.shape)
    error = out.float() - ref.float()
    rms = ref.float().pow(2).mean().sqrt()
    assert error.pow(2).mean().sqrt() <= 0.02 * rms, (error.pow(2).mean().sqrt(), rms)
    assert error.abs().max() <= 16 * 2**-8 * rms, (error.abs().max(), rms)


@pytest.fixture(scope="module")
def tower():
    return M.VisionTower(_core())


# Frames of 256, 144 and 32 rows under a 512-row cap: three calls -- the first item, the
# second with two of the third's six frames, the other four -- so a cut between items and one
# inside an item; the last call's longest frame is 32 rows, where the fused attention kernel
# is the short-frame one.
GRID, CAP = [(2, 16, 16), (3, 12, 12), (6, 4, 8)], 512


def test_calls_of_whole_frames_equal_one_call(tower, monkeypatch):
    grid, cap = GRID, CAP
    pixels = _pixels(grid)
    grid_thw = torch.tensor(grid)
    monkeypatch.setattr(M, "_VISION_CALL_MAX_PATCHES", 1 << 20)
    whole = tower(pixels, grid_thw)

    monkeypatch.setattr(M, "_VISION_CALL_MAX_PATCHES", cap)
    calls = _count_calls(tower, monkeypatch)
    chunked = tower(pixels, grid_thw)

    expected = M._frame_calls([tuple(row) for row in grid], cap)
    assert len(calls) == len(expected) > 1
    assert all(rows <= cap for rows, _ in calls)
    _assert_close(chunked, whole)


def test_a_request_past_the_old_whole_request_ceiling_runs(tower):
    # 66 frames of 512x512 pixels: 33 temporal groups of 32x32 patches, 33792 rows in two
    # calls (32 groups, then 1). The last group's rows, alone in its call, and the first
    # group's, in a call with 31 others, equal each group run on its own.
    grid = [(33, 32, 32)]
    pixels = _pixels(grid)
    out = tower(pixels, torch.tensor(grid))
    assert out.shape == (33 * 32 * 32 // 4, OUT)
    assert torch.isfinite(out).all()
    group_rows, group_out = 32 * 32, 32 * 32 // 4
    for group in (0, 32):
        alone = tower(
            pixels[group * group_rows : (group + 1) * group_rows], torch.tensor([(1, 32, 32)])
        )
        _assert_close(out[group * group_out : (group + 1) * group_out], alone)
