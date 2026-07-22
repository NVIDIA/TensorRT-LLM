# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence coverage for the fused score+stats+union pipeline (two CuTe kernels).

The reference side gathers the SAME production score rows (the fused pack's
score-only entry) and normalizes + union-reduces them with a pure-torch
float32 oracle. The fused-vs-reference comparison was always tolerance-based
(the fused pipeline's reduction order differs from any reference); the
tolerances are unchanged from the retired Triton reference copies.
"""

import pytest
import torch

_SM100_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention CuTe kernels require SM100",
)


def _reference_union_scores(scores_rows: torch.Tensor, valid_widths: torch.Tensor) -> torch.Tensor:
    """Pure-torch union oracle: z-normalize each row's valid prefix, union-max.

    Mirrors the production union semantics: per-row mean and biased std over
    the valid prefix (std clamped at 1e-6), then the per-token maximum across
    the request's rows; tokens past the valid width stay ``-inf``.
    """
    request_count, _, width = scores_rows.shape
    combined = torch.full(
        (request_count, width), float("-inf"), dtype=torch.float32, device=scores_rows.device
    )
    for request in range(request_count):
        valid_width = int(valid_widths[request])
        if valid_width <= 0:
            continue
        valid = scores_rows[request, :, :valid_width].to(torch.float32)
        mean = valid.mean(dim=1, keepdim=True)
        std = ((valid - mean).square().sum(dim=1, keepdim=True) / valid_width).sqrt()
        combined[request, :valid_width] = ((valid - mean) / std.clamp_min(1e-6)).amax(dim=0)
    return combined


def _check_union_fusion_matches_split_pipeline(
    tokens_per_block: int,
    num_freqs: int,
    num_q_heads: int,
    score_starts: "int | list",
    valid_lens: "list | None",
) -> None:
    """The fused pipeline must reproduce the split score->normalize->union rows.

    ``score_starts`` is either one uniform window start or a per-request
    list (the fused kernels read the start per request at runtime). The
    reference leg runs the production score-only launch over the same decode
    windows, then the pure-torch union oracle.
    """
    pytest.importorskip("cutlass")

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        _FixedScoreGroup,
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

    # Reference: the production score gather over the same decode windows,
    # then the pure-torch union oracle.
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
    expected = _reference_union_scores(scores_rows, split_widths)

    fused_widths = torch.empty(request_count, dtype=torch.int32, device=device)
    fused_out = torch.full(
        (request_count, seq_len), float("nan"), dtype=torch.float32, device=device
    )
    group.launch_cute_union_fusion(
        request_count,
        valid_seq_lens,
        fused_widths,
        token_starts,
        mean_cos,
        mean_sin,
        fused_out,
    )
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
        # GQA group 4 with 32 frequencies: head columns pad up to the MMA
        # tile N=8 with zeroed weights, the partial-stats epilogue writes
        # only the real heads' rows, and the union finalizer maps head rows
        # onto the padded score planes.
        (128, 32, 4, 0, None),
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
) -> None:
    _check_union_fusion_matches_split_pipeline(
        tokens_per_block, num_freqs, num_q_heads, score_starts, valid_lens
    )


@_SM100_ONLY
@pytest.mark.parametrize("guard", ["runner_construction_failure", "frequency_count"])
def test_union_fusion_guards_raise(guard: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """One representative per fused-pipeline guard family raises loudly.

    ``runner_construction_failure``: a fused-runner construction failure
    surfaces as the no-fallback RuntimeError at score setup
    (``prepare_cute_score`` runs before the union runner is built).
    ``frequency_count``: 16 frequencies (head size 32) sit outside the fused
    kernel contract and are rejected at kernel construction.
    """
    cutlass = pytest.importorskip("cutlass")

    if guard == "frequency_count":
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
        return

    import tensorrt_llm._torch.kv_cache_compression.triattention.triattention_cute_score_fused as fused_module  # noqa: E501
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        _FixedScoreGroup,
    )

    torch.manual_seed(20260721)
    device = torch.device("cuda")
    seq_len, tokens_per_block, num_freqs, num_q_heads = 256, 128, 32, 8
    num_pages = seq_len // tokens_per_block
    pool = (
        0.125 * torch.randn(num_pages, 2, 1, tokens_per_block, 2 * num_freqs, device=device)
    ).to(torch.bfloat16)
    q_real = 0.125 * torch.randn(1, num_q_heads, num_freqs, device=device)
    omega = torch.linspace(0.01, 0.03, num_freqs, device=device)
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
        torch.linspace(0.5, 1.5, num_freqs, device=device),
        omega,
        torch.tensor([1.0, 2.0, 4.0], device=device),
        output_width=seq_len,
    )

    def _refuse_construction(**_kwargs):
        raise ValueError("synthetic fused-runner construction failure")

    monkeypatch.setattr(fused_module, "TriAttentionCuteScoreRunner", _refuse_construction)
    mean_cos = torch.cos(torch.outer(torch.tensor([256.0, 257.0], device=device), omega))
    mean_sin = torch.sin(torch.outer(torch.tensor([256.0, 257.0], device=device), omega))
    with pytest.raises(RuntimeError, match="no other score path exists"):
        group.launch_cute_union_fusion(
            2,
            torch.full((2,), seq_len, dtype=torch.int32, device=device),
            torch.empty(2, dtype=torch.int32, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
            mean_cos.contiguous(),
            mean_sin.contiguous(),
            torch.empty((2, seq_len), dtype=torch.float32, device=device),
        )


@_SM100_ONLY
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 32 * 1024**3,
    reason="the giant-scratch geometry needs ~15 GiB of device memory",
)
@pytest.mark.parametrize(
    "max_requests",
    [
        # Qwen3-8B serve geometry at max_batch_size 64 with the 16384-token
        # bucket: the score scratch spans 2,415,919,104 elements, past 2^31.
        # Before the finalizer's fallback loads were folded into a 64-bit
        # pointer, this leg died with an illegal memory access whenever a
        # window start was not lane-aligned (the serve-mode eviction crash).
        64,
        # Same shape at max_batch_size 32 stays below 2^31 and covers the
        # boundary from the always-correct side.
        32,
    ],
)
def test_union_fusion_giant_scratch_unaligned_start(max_requests: int) -> None:
    """Unaligned window starts must survive a past-2^31-element score scratch.

    The union finalizer reads the score scratch at flat offsets up to
    ``plane * capacity * layers * bucket``; the production Qwen3-8B serve
    shape (capacity 64, 36 layers, bucket 16384) pushes those offsets past
    2^31. Requests whose pinned prompt length is not a multiple of the lane
    width take the per-token load branch, which must fold the Int64 offset
    into the pointer instead of the DSL's 32-bit dynamic coordinate. The leg
    launches at full capacity with zero-length tail rows, exactly like a
    production eviction round, and checks the fused rows against the split
    score-gather plus the pure-torch union oracle.
    """
    pytest.importorskip("cutlass")

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_kernels import (
        _FixedScoreGroup,
    )

    torch.manual_seed(20260722)
    device = torch.device("cuda")
    num_layers = 36
    num_q_heads = 32
    num_kv_heads = 8
    num_freqs = 64
    tokens_per_block = 32
    seq_len = 16384
    decode_window = 8192
    # Window starts deliberately off the 4-token lane grid (real prompt
    # lengths are arbitrary), one of them also off the page grid.
    score_starts = [897, 641]
    valid_lens = [start + decode_window for start in score_starts]
    request_count = len(score_starts)

    # Every layer shares one physical pool: the scratch magnitude only needs
    # the segment count, not distinct K content per layer.
    num_pages = (max(valid_lens) + tokens_per_block - 1) // tokens_per_block
    pool = (
        0.125
        * torch.randn(num_pages, 2, num_kv_heads, tokens_per_block, 2 * num_freqs, device=device)
    ).to(torch.bfloat16)
    layer_pools = [pool] * num_layers
    calib_shape = (num_layers, num_q_heads, num_freqs)
    q_real = 0.125 * torch.randn(calib_shape, device=device)
    q_imag = 0.125 * torch.randn_like(q_real)
    mlr_coef = 0.125 * torch.randn_like(q_real)
    freq_scale_sq = torch.linspace(0.5, 1.5, num_freqs, device=device)
    omega = torch.linspace(0.01, 0.03, num_freqs, device=device)
    offsets = torch.tensor([1.0, 2.0, 4.0], device=device)
    round_starts = torch.arange(max_requests, dtype=torch.float32, device=device) + seq_len
    phase = (round_starts[:, None, None] + offsets[None, :, None]) * omega[None, None]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()

    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    block_offsets = torch.zeros(1, max_requests, 2, num_pages, dtype=torch.int32, device=device)
    block_offsets[0, :request_count, 0] = 2 * page_ids
    block_offsets[0, :request_count, 1] = 2 * page_ids + 1
    group = _FixedScoreGroup(
        layer_pools,
        list(range(num_layers)),
        max_requests,
        num_pages,
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
        output_width=decode_window,
    )
    assert (num_kv_heads * 8 * max_requests * num_layers * seq_len > 2**31) == (max_requests == 64)

    valid_seq_lens = torch.zeros(max_requests, dtype=torch.int32, device=device)
    token_starts = torch.zeros(max_requests, dtype=torch.int32, device=device)
    valid_seq_lens[:request_count] = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    token_starts[:request_count] = torch.tensor(score_starts, dtype=torch.int32, device=device)

    # Reference: the split score gather over the same decode windows, then
    # the pure-torch union oracle.
    split_widths = torch.zeros(max_requests, dtype=torch.int32, device=device)
    per_head = group.launch(
        request_count,
        valid_seq_lens,
        split_widths,
        token_starts,
        mean_cos,
        mean_sin,
    )
    rows = per_head.shape[1] * per_head.shape[2]
    scores_rows = per_head.reshape(request_count, rows, decode_window).contiguous()
    expected = _reference_union_scores(scores_rows, split_widths)

    # Fused pipeline at FULL capacity (zero-length tails), like production.
    fused_widths = torch.zeros(max_requests, dtype=torch.int32, device=device)
    fused_out = torch.full(
        (max_requests, seq_len), float("nan"), dtype=torch.float32, device=device
    )
    group.launch_cute_union_fusion(
        max_requests,
        valid_seq_lens,
        fused_widths,
        token_starts,
        mean_cos,
        mean_sin,
        fused_out,
    )
    torch.cuda.synchronize()
    assert torch.equal(fused_widths[:request_count], split_widths[:request_count])
    for request in range(request_count):
        width = valid_lens[request] - score_starts[request]
        torch.testing.assert_close(
            fused_out[request, :width],
            expected[request, :width],
            rtol=5.0e-3,
            atol=5.0e-3,
        )
