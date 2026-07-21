# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence coverage for the fused score+stats+union pipeline (two CuTe kernels)."""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention CuTe kernels require SM100",
)
@pytest.mark.parametrize(
    "score_start,valid_lens",
    [
        # Full-range scoring, both requests full length.
        (0, None),
        # Non-page-aligned uniform start with ragged valid lengths.
        (37, [250, 198]),
        # Page-aligned start.
        (128, [250, 230]),
    ],
)
def test_union_fusion_matches_split_pipeline(
    score_start: int,
    valid_lens: "list | None",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused pipeline must reproduce the split score->stats->union rows.

    Geometry is the fused kernel's contract (128-token pages, GQA group 8,
    32 frequencies). The reference runs the production split path: the score
    launch gathers each request's decode window, then ``prepare_union_scores``
    normalizes rows and takes the cross-row union maximum.
    """
    pytest.importorskip("cutlass")
    monkeypatch.setenv("TRTLLM_TRIATTENTION_CUTE_UNION_FUSION", "1")

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        _FixedScoreGroup,
        prepare_union_scores,
    )

    torch.manual_seed(20260721)
    device = torch.device("cuda")
    seq_len = 256
    tokens_per_block = 128
    num_freqs = 32
    num_q_heads = 8
    num_pages = seq_len // tokens_per_block
    pool = (
        0.125 * torch.randn(num_pages, 2, 1, tokens_per_block, 2 * num_freqs, device=device)
    ).to(torch.bfloat16)
    q_real = 0.125 * torch.randn(1, num_q_heads, num_freqs, device=device)
    q_imag = 0.125 * torch.randn_like(q_real)
    mlr_coef = 0.125 * torch.randn_like(q_real)
    freq_scale_sq = torch.linspace(0.5, 1.5, num_freqs, device=device)
    omega = torch.linspace(0.01, 0.03, num_freqs, device=device)
    offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
    round_starts = torch.tensor([float(seq_len), float(seq_len + 1)], device=device)
    phase = (round_starts[:, None, None] + offsets[None, :, None]) * omega[None, None]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()

    k_plane = [2 * page for page in range(num_pages)]
    v_plane = [2 * page + 1 for page in range(num_pages)]
    block_offsets = torch.tensor(
        [[[k_plane, v_plane], [k_plane, v_plane]]], dtype=torch.int32, device=device
    )
    group = _FixedScoreGroup(
        [pool],
        [0],
        2,
        num_pages,
        seq_len,
        num_q_heads,
        block_offsets,
        [0],
        q_real,
        q_imag,
        mlr_coef,
        freq_scale_sq,
        omega,
        offsets,
        output_width=seq_len,
    )
    if valid_lens is None:
        valid_lens = [seq_len, seq_len]
    valid_seq_lens = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    request_count = 2

    # Reference: the split pipeline over the same decode windows.
    split_widths = torch.empty(request_count, dtype=torch.int32, device=device)
    token_starts = torch.full((request_count,), score_start, dtype=torch.int32, device=device)
    per_head = group.launch(
        request_count,
        valid_seq_lens,
        split_widths,
        token_starts,
        mean_cos,
        mean_sin,
    )
    rows = per_head.shape[1] * per_head.shape[2]
    scores_rows = per_head.reshape(request_count, rows, seq_len).contiguous()
    row_mean = torch.empty((request_count, rows, 1), dtype=torch.float32, device=device)
    row_inv_std = torch.empty_like(row_mean)
    expected = torch.empty((request_count, seq_len), dtype=torch.float32, device=device)
    prepare_union_scores(
        scores_rows,
        split_widths,
        row_mean,
        row_inv_std,
        expected,
        request_count,
        normalize_scores=True,
    )

    fused_widths = torch.empty(request_count, dtype=torch.int32, device=device)
    fused_out = torch.full(
        (request_count, seq_len), float("nan"), dtype=torch.float32, device=device
    )
    launched = group.launch_cute_union_fusion(
        request_count,
        valid_seq_lens,
        fused_widths,
        token_starts,
        score_start,
        mean_cos,
        mean_sin,
        fused_out,
    )
    assert launched, "fused union pipeline must engage on its contract geometry"
    assert torch.equal(fused_widths, split_widths)
    for request in range(request_count):
        width = int(valid_lens[request]) - score_start
        torch.testing.assert_close(
            fused_out[request, :width],
            expected[request, :width],
            rtol=5.0e-3,
            atol=5.0e-3,
        )

    # A cohort without one uniform prompt start must decline, not launch.
    assert not group.launch_cute_union_fusion(
        request_count,
        valid_seq_lens,
        fused_widths,
        token_starts,
        None,
        mean_cos,
        mean_sin,
        fused_out,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention CuTe kernels require SM100",
)
def test_union_fusion_declines_off_contract_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """32-token pages sit outside the fused contract: decline, do not fault."""
    pytest.importorskip("cutlass")
    monkeypatch.setenv("TRTLLM_TRIATTENTION_CUTE_UNION_FUSION", "1")

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        _FixedScoreGroup,
    )

    torch.manual_seed(20260721)
    device = torch.device("cuda")
    seq_len = 256
    tokens_per_block = 32
    num_freqs = 32
    num_q_heads = 8
    num_pages = seq_len // tokens_per_block
    pool = (
        0.125 * torch.randn(num_pages, 2, 1, tokens_per_block, 2 * num_freqs, device=device)
    ).to(torch.bfloat16)
    q_real = 0.125 * torch.randn(1, num_q_heads, num_freqs, device=device)
    freq_scale_sq = torch.linspace(0.5, 1.5, num_freqs, device=device)
    omega = torch.linspace(0.01, 0.03, num_freqs, device=device)
    offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
    mean_cos = torch.cos(torch.outer(torch.tensor([256.0, 257.0], device=device), omega))
    mean_sin = torch.sin(torch.outer(torch.tensor([256.0, 257.0], device=device), omega))
    k_plane = [2 * page for page in range(num_pages)]
    v_plane = [2 * page + 1 for page in range(num_pages)]
    block_offsets = torch.tensor(
        [[[k_plane, v_plane], [k_plane, v_plane]]], dtype=torch.int32, device=device
    )
    group = _FixedScoreGroup(
        [pool],
        [0],
        2,
        num_pages,
        seq_len,
        num_q_heads,
        block_offsets,
        [0],
        q_real,
        torch.randn_like(q_real) * 0.125,
        torch.randn_like(q_real) * 0.125,
        freq_scale_sq,
        omega,
        offsets,
        output_width=seq_len,
    )
    valid_seq_lens = torch.full((2,), seq_len, dtype=torch.int32, device=device)
    widths = torch.empty(2, dtype=torch.int32, device=device)
    token_starts = torch.zeros(2, dtype=torch.int32, device=device)
    union_out = torch.empty((2, seq_len), dtype=torch.float32, device=device)
    with pytest.warns(UserWarning, match="union fusion unavailable"):
        launched = group.launch_cute_union_fusion(
            2,
            valid_seq_lens,
            widths,
            token_starts,
            0,
            mean_cos.contiguous(),
            mean_sin.contiguous(),
            union_out,
        )
    assert not launched
