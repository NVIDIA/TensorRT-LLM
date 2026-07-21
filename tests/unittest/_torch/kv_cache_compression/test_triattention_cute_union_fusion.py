# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence coverage for the fused score+stats+union pipeline (two CuTe kernels)."""

import pytest
import torch

_SM100_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention CuTe kernels require SM100",
)


def _check_union_fusion_matches_split_pipeline(
    tokens_per_block: int,
    num_freqs: int,
    num_q_heads: int,
    score_starts: "int | list",
    valid_lens: "list | None",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fused pipeline must reproduce the split score->stats->union rows.

    ``score_starts`` is either one uniform window start or a per-request
    list (the fused kernels read the start per request at runtime). The
    reference runs the production split path: the score launch gathers each
    request's decode window, then ``prepare_union_scores`` normalizes rows
    and takes the cross-row union maximum.
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
    num_pages = seq_len // tokens_per_block
    # 32-token pages: one 128-token compute tile spans four pages, so a
    # shuffled physical-page table catches any fragment/page mix-up. The
    # ragged valid lengths land mid-tile, exercising the clamped tail
    # fragments.
    page_permutation = {128: [0, 1], 32: [3, 1, 4, 7, 5, 0, 2, 6]}[tokens_per_block]
    assert sorted(page_permutation) == list(range(num_pages))
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

    k_plane = [2 * page for page in page_permutation]
    v_plane = [2 * page + 1 for page in page_permutation]
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
    if isinstance(score_starts, int):
        score_starts = [score_starts] * request_count
    assert len(score_starts) == request_count

    # Reference: the split pipeline over the same decode windows.
    split_widths = torch.empty(request_count, dtype=torch.int32, device=device)
    token_starts = torch.tensor(score_starts, dtype=torch.int32, device=device)
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
        mean_cos,
        mean_sin,
        fused_out,
    )
    assert launched, "fused union pipeline must engage on its contract geometry"
    assert torch.equal(fused_widths, split_widths)
    for request in range(request_count):
        width = int(valid_lens[request]) - int(score_starts[request])
        torch.testing.assert_close(
            fused_out[request, :width],
            expected[request, :width],
            rtol=5.0e-3,
            atol=5.0e-3,
        )


@_SM100_ONLY
@pytest.mark.parametrize(
    "tokens_per_block,num_freqs,num_q_heads,score_starts,valid_lens",
    [
        # The originally validated geometry: 32 frequencies (64-element K
        # rows), GQA group 8, across full-range, non-page-aligned ragged,
        # and page-aligned uniform window starts.
        (32, 32, 8, 0, None),
        (32, 32, 8, 37, [250, 198]),
        (32, 32, 8, 128, [250, 230]),
        (128, 32, 8, 0, None),
        (128, 32, 8, 37, [250, 198]),
        (128, 32, 8, 128, [250, 230]),
        # Qwen3 geometry: 128-element K rows (64 frequencies) and GQA group
        # 4, which rides the MMA tile N=8 with zeroed padding columns.
        (32, 64, 4, 0, None),
        (32, 64, 4, 37, [250, 198]),
        (128, 64, 4, 0, None),
        (128, 64, 4, 128, [250, 230]),
        # Mixed-prompt cohorts: each request scores its own window (one
        # start mid-tile, one page-aligned) — the case the fused pipeline
        # previously declined.
        (32, 32, 8, [37, 128], [250, 198]),
        (128, 32, 8, [37, 128], None),
        (32, 64, 4, [37, 128], None),
        (128, 64, 4, [37, 128], [250, 230]),
    ],
)
def test_union_fusion_matches_split_pipeline(
    tokens_per_block: int,
    num_freqs: int,
    num_q_heads: int,
    score_starts: "int | list",
    valid_lens: "list | None",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _check_union_fusion_matches_split_pipeline(
        tokens_per_block, num_freqs, num_q_heads, score_starts, valid_lens, monkeypatch
    )


@_SM100_ONLY
def test_union_fusion_engages_gqa4_narrow_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    """GQA group 4 with 32 frequencies (the formerly declined geometry) engages.

    The fused kernel pads group-4 head columns up to the MMA tile N=8 with
    zeroed weights, the partial-stats epilogue writes only the real heads'
    rows, and the union finalizer maps head rows onto the padded score
    planes — so this mixed geometry must launch and match the split path.
    """
    _check_union_fusion_matches_split_pipeline(128, 32, 4, 0, None, monkeypatch)


def test_union_fusion_rejects_unsupported_frequency_count() -> None:
    """16 frequencies (head size 32) sit outside the fused kernel contract."""
    cutlass = pytest.importorskip("cutlass")

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_cute_score_fused import (  # noqa: E501
        _TriAttentionScoreKernel,
    )

    with pytest.raises(ValueError, match="frequencies"):
        _TriAttentionScoreKernel(
            num_layers=1,
            seq_len=256,
            num_q_heads=8,
            num_kv_heads=1,
            num_freqs=16,
            tokens_per_block=128,
            pool_shape=(2, 2, 1, 128, 32),
            pool_strides=(8192, 4096, 4096, 32, 1),
            pool_dtype=cutlass.BFloat16,
            page_shards=3,
        )
