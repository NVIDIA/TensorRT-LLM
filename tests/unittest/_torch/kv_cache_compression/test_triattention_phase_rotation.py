# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mean-path phase-table rotation vs a direct trigonometric fold.

The mean-aggregation coefficient preparation tabulates the offset-averaged
phase of every possible round-start position once at initialization (float64
accumulation, stored fp32) and rotates the pre-scaled calibration query by
one gathered table row per request. This test rebuilds the same coefficients
with direct float64 trigonometry at the exact round starts and compares.
"""

import torch

from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
    _FixedScoreGroup,
)


def test_phase_table_rotation_matches_direct_trig_fold():
    """Table + rotation == direct trig fold at positions {0, mid, last}.

    The last position exercises phases of tens of thousands of radians,
    where fp32 runtime trigonometry would already have lost several digits
    to argument reduction; the float64-built table must not.
    """
    assert hasattr(torch.ops.trtllm, "tri_attention_rotate_mean_score_coefficients"), (
        "TriAttention rotation op is not loaded"
    )
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(20260720)
    num_layers = 2
    num_q_heads = 4
    head_dim = 16
    num_freqs = head_dim // 2
    # Sizes the phase table: the group tabulates [0, seq_len] inclusive.
    seq_len = 32768
    pools = [
        torch.randn(2, 2, 2, 4, head_dim, device=device).to(torch.bfloat16)
        for _ in range(num_layers)
    ]
    block_offsets = torch.zeros(1, 3, 2, 1, dtype=torch.int32, device=device)
    q_real = torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    q_imag = torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    mlr_coef = torch.randn(num_layers, num_q_heads, num_freqs, device=device)
    freq_scale_sq = torch.rand(num_freqs, device=device) + 0.5
    # RoPE-style inverse frequencies: omega[0] == 1.0 makes the last tabulated
    # position a genuinely large trigonometric argument.
    omega = 10000.0 ** (-torch.arange(num_freqs, device=device, dtype=torch.float32) / num_freqs)
    offsets = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32, device=device)
    group = _FixedScoreGroup(
        pools,
        list(range(num_layers)),
        3,
        1,
        seq_len,
        num_q_heads,
        block_offsets,
        [0] * num_layers,
        q_real,
        q_imag,
        mlr_coef,
        freq_scale_sq,
        omega,
        offsets,
        output_width=4,
    )
    max_position = group._max_position
    assert max_position == seq_len + 1

    positions = [0, max_position // 2, max_position - 1]
    round_starts = torch.tensor(positions, dtype=torch.int32, device=device)
    total = len(positions) * num_layers * num_q_heads * num_freqs
    c_re = torch.full((total,), float("nan"), dtype=torch.float32, device=device)
    c_im = torch.full_like(c_re, float("nan"))
    torch.ops.trtllm.tri_attention_rotate_mean_score_coefficients(
        c_re,
        c_im,
        group._q_real_scaled,
        group._q_imag_scaled,
        group._phase_cos,
        group._phase_sin,
        round_starts,
        len(positions),
        num_layers,
        num_q_heads,
        num_freqs,
        max_position,
    )

    # Direct trigonometric fold: average the offset phases at each round
    # start, then rotate and scale the calibration query (float64 throughout,
    # cast to fp32 only at the end, exactly like the tables were built).
    phase = (
        round_starts.to(torch.float64)[:, None, None] + offsets.to(torch.float64)[None, :, None]
    ) * omega.to(torch.float64)[None, None, :]
    mean_cos = torch.cos(phase).mean(dim=1)[:, None, None, :]
    mean_sin = torch.sin(phase).mean(dim=1)[:, None, None, :]
    fss = freq_scale_sq.to(torch.float64)
    q_re = q_real.to(torch.float64)[None]
    q_im = q_imag.to(torch.float64)[None]
    reference_re = (fss * (q_re * mean_cos - q_im * mean_sin)).to(torch.float32)
    reference_im = (fss * (q_im * mean_cos + q_re * mean_sin)).to(torch.float32)

    shape = (len(positions), num_layers, num_q_heads, num_freqs)
    torch.testing.assert_close(c_re.view(shape), reference_re, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(c_im.view(shape), reference_im, rtol=1e-5, atol=1e-5)
