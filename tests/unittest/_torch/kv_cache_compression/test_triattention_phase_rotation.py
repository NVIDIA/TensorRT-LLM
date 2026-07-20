# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mean-path phase-table rotation: numerical fidelity and path equality.

The mean-aggregation coefficient preparation tabulates the offset-averaged
phase of every possible round-start position once at initialization (float64
accumulation, stored fp32) and rotates the pre-scaled calibration query by
one gathered table row per request. The first test rebuilds the same
coefficients with direct float64 trigonometry at the exact round starts and
compares. The second proves the two places that rotation can run — the
standalone rotation kernel writing global coefficient planes, and the score
kernels' own in-CTA shared-memory prologue (the production default) — yield
BIT-IDENTICAL paged-score outputs, because both compile one shared rotation
expression.
"""

import pytest
import torch
from test_triattention_score_ops import _build_case as _build_score_case

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


# One geometry per code path the in-CTA rotation prologue compiles into:
# both CUDA load paths (16-byte vectorized chunks and strided scalar), GQA
# groups 2/4/8 with dedicated template instantiations, group 3 through the
# per-query-head grid mapping, and every float pool dtype. The shared
# builder's per-request sequence lengths are ragged, so the write mask is
# compared too (sentinel-filled outputs).
_EQUALITY_CASES = [
    pytest.param(
        dict(
            head_dim=128,
            tokens_per_block=32,
            num_q_heads=8,
            num_kv_heads=2,
            dtype=torch.bfloat16,
        ),
        id="vectorized_bf16_group4",
    ),
    pytest.param(
        dict(
            head_dim=128,
            tokens_per_block=32,
            num_q_heads=8,
            num_kv_heads=1,
            dtype=torch.bfloat16,
        ),
        id="vectorized_bf16_group8",
    ),
    pytest.param(
        dict(
            head_dim=128,
            tokens_per_block=32,
            num_q_heads=6,
            num_kv_heads=2,
            dtype=torch.float16,
        ),
        id="vectorized_fp16_group3_per_query_head",
    ),
    pytest.param(
        dict(
            head_dim=8,
            tokens_per_block=4,
            num_q_heads=4,
            num_kv_heads=2,
            dtype=torch.bfloat16,
        ),
        id="scalar_bf16_group2",
    ),
    pytest.param(
        dict(
            head_dim=8,
            tokens_per_block=4,
            num_q_heads=6,
            num_kv_heads=2,
            dtype=torch.float32,
        ),
        id="scalar_fp32_runtime_group3",
    ),
]


@pytest.mark.parametrize("case", _EQUALITY_CASES)
def test_in_cta_rotation_scores_bit_equal_to_standalone_rotation(case):
    """Full mean-path score outputs are torch.equal between rotation flavors.

    Bit-equality (not a tolerance) is the contract: the score kernels' in-CTA
    prologue and the standalone rotation kernel share one rotation
    expression, and the score accumulation downstream is identical, so any
    single-ulp drift means the arithmetic diverged and must fail here.
    """
    assert hasattr(torch.ops.trtllm, "tri_attention_paged_score"), (
        "TriAttention paged score op is not loaded"
    )
    assert hasattr(torch.ops.trtllm, "tri_attention_rotate_mean_score_coefficients"), (
        "TriAttention rotation op is not loaded"
    )
    request_count = 3
    (
        group,
        round_starts,
        token_starts,
        valid_seq_lens,
        _,
        mean_cos,
        mean_sin,
        _,
    ) = _build_score_case(
        request_count=request_count,
        max_requests=4,
        num_layers=2,
        page_count=4,
        prompt_len=5,
        seed=20260723,
        offsets=[1.0, 2.0, 4.0],
        **case,
    )
    device = group.output.device
    sentinel = -54321.0

    def run(mean_rotate_in_cta: bool) -> "tuple[torch.Tensor, torch.Tensor]":
        group.output.fill_(sentinel)
        valid_widths = torch.zeros(request_count, dtype=torch.int32, device=device)
        scores = group.launch(
            request_count,
            valid_seq_lens,
            valid_widths,
            round_starts,
            token_starts,
            mean_cos,
            mean_sin,
            "mean",
            mean_rotate_in_cta=mean_rotate_in_cta,
        ).clone()
        return scores, valid_widths

    # Reference leg: the kernel-round2 preparation (standalone rotation
    # kernel + score kernels reading the global coefficient planes).
    reference_scores, reference_widths = run(mean_rotate_in_cta=False)
    in_cta_scores, in_cta_widths = run(mean_rotate_in_cta=True)
    assert not reference_scores.eq(sentinel).all(), "reference leg scored nothing"
    assert torch.equal(in_cta_scores, reference_scores)
    assert torch.equal(in_cta_widths, reference_widths)
