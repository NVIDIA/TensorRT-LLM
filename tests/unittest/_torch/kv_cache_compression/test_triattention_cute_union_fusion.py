# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence coverage for the fused score+stats+union pipeline (two CuTe kernels).

The reference side gathers the SAME production score rows (the fused pack's
score-only entry, which every buffer-namespace runner compiles) and normalizes +
union-reduces them with a pure-torch float32 oracle. The fused-vs-reference
comparison was always tolerance-based (the fused pipeline's reduction order
differs from any reference); the tolerances are unchanged from the retired
Triton reference copies.
"""

import pytest
import torch

_SM100_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention CuTe kernels require SM100",
)


def _make_union_buffers(
    *,
    layer_pools,
    max_requests,
    seq_len,
    num_q_heads,
    q_real,
    q_imag,
    mlr_coef,
    freq_scale_sq,
    omega,
    offsets,
    decode_width=None,
):
    """Union-mode buffers over one shared page-table slot.

    The union runner also compiles the score-only entries, so one buffer
    namespace serves both the fused pipeline and the split reference leg.
    """
    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention import (
        init_eviction_buffers,
    )

    num_layers = len(layer_pools)
    return init_eviction_buffers(
        eviction_mode="union",
        layer_pools=layer_pools,
        dense_groups=[list(range(num_layers))],
        dense_layers=list(range(num_layers)),
        page_representatives=[0],
        max_requests=max_requests,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_freqs=int(q_real.shape[-1]),
        keep_count=1,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        offsets=offsets,
        omega=omega,
        decode_width=decode_width,
        layer_group_representative={layer: 0 for layer in range(num_layers)},
        layer_pool_keys=[("pool", 0)] * num_layers,
    )


def _write_block_offsets(bufs, encoded):
    """Load a test page table into the staged block-offset plane."""
    bufs.block_offsets_device.zero_()
    bufs.block_offsets_device[:, : encoded.shape[1], :, : encoded.shape[-1]].copy_(encoded)


def _stage_score_metadata(bufs, request_count, valid_seq_lens, valid_widths, token_starts):
    """Stage the per-round score metadata exactly like production.

    The compiled runner reads valid lengths and window starts straight from
    the staged metadata rows (pointer capture), so stage them like
    ``stage_eviction_cohort`` does; the width subtraction mirrors what the
    production phase-gather launch derives on device.
    """
    torch.sub(
        valid_seq_lens[:request_count],
        token_starts[:request_count],
        out=valid_widths[:request_count],
    )
    bufs.valid_seq_lens_device[:request_count].copy_(valid_seq_lens[:request_count])
    bufs.token_starts_device[:request_count].copy_(token_starts[:request_count])


def _launch_split_scores(
    bufs, request_count, valid_seq_lens, valid_widths, token_starts, mean_cos, mean_sin
):
    """The production score-only leg plus the decode-window gather."""
    _stage_score_metadata(bufs, request_count, valid_seq_lens, valid_widths, token_starts)
    assert request_count in bufs.runner._compiled
    bufs.runner.launch(request_count, mean_cos, mean_sin)
    num_segments = request_count * bufs.num_layers
    group_size = bufs.num_q_heads // bufs.num_kv_heads
    source = (
        bufs.cute_scratch[: bufs.num_kv_heads * 8 * num_segments * bufs.bucket_seq_len]
        .view(bufs.num_kv_heads, 8, request_count, bufs.num_layers, bufs.bucket_seq_len)[
            :, :group_size
        ]
        .permute(2, 3, 0, 1, 4)
    )
    columns = (
        token_starts[:request_count].to(torch.int64).view(-1, 1, 1, 1, 1) + bufs.gather_columns
    )
    columns = columns.clamp_(max=bufs.bucket_seq_len - 1).expand(
        request_count, bufs.num_layers, bufs.num_kv_heads, group_size, bufs.decode_width
    )
    output = torch.full(
        (request_count, bufs.num_layers, bufs.num_q_heads, bufs.decode_width),
        float("nan"),
        dtype=torch.float32,
        device=bufs.device,
    )
    torch.gather(
        source,
        4,
        columns,
        out=output.view(
            request_count, bufs.num_layers, bufs.num_kv_heads, group_size, bufs.decode_width
        ),
    )
    return output


def _launch_union_fusion(
    bufs, request_count, valid_seq_lens, valid_widths, token_starts, mean_cos, mean_sin, union_out
):
    """The fused score+stats+normalized-union pipeline (THE union path)."""
    _stage_score_metadata(bufs, request_count, valid_seq_lens, valid_widths, token_starts)
    assert (
        request_count in bufs.runner._compiled_stats
        and request_count in bufs.runner._compiled_normalize_union
    )
    bufs.runner.launch_union_fusion(request_count, mean_cos, mean_sin, union_out[:request_count])


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

    bufs = _make_union_buffers(
        layer_pools=[pool],
        max_requests=2,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets,
    )
    k_plane = [2 * page for page in page_permutation]
    v_plane = [2 * page + 1 for page in page_permutation]
    _write_block_offsets(
        bufs,
        torch.tensor([[[k_plane, v_plane], [k_plane, v_plane]]], dtype=torch.int32, device=device),
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
    per_head = _launch_split_scores(
        bufs,
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
    _launch_union_fusion(
        bufs,
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
        # Representative rows per axis: the originally validated geometry
        # (32 freqs, GQA group 8) at both page sizes with full-range and
        # ragged page-aligned starts.
        (32, 32, 8, 0, None),
        (128, 32, 8, 128, [250, 230]),
        # Qwen3 geometry: 128-element K rows (64 frequencies) and GQA group
        # 4, which rides the MMA tile N=8 with zeroed padding columns.
        (32, 64, 4, 37, [250, 198]),
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
def test_union_fusion_frequency_count_guard_raises() -> None:
    """16 frequencies (head size 32) sit outside the fused kernel contract
    and are rejected at kernel construction."""
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

    bufs = _make_union_buffers(
        layer_pools=layer_pools,
        max_requests=max_requests,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        q_real=q_real,
        q_imag=q_imag,
        mlr_coef=mlr_coef,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets,
        decode_width=decode_window,
    )
    assert (bufs.cute_scratch.numel() > 2**31) == (max_requests == 64)
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    bufs.block_offsets_device.zero_()
    bufs.block_offsets_device[0, :request_count, 0, :num_pages] = 2 * page_ids
    bufs.block_offsets_device[0, :request_count, 1, :num_pages] = 2 * page_ids + 1

    valid_seq_lens = torch.zeros(max_requests, dtype=torch.int32, device=device)
    token_starts = torch.zeros(max_requests, dtype=torch.int32, device=device)
    valid_seq_lens[:request_count] = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    token_starts[:request_count] = torch.tensor(score_starts, dtype=torch.int32, device=device)

    # Reference: the split score gather over the same decode windows, then
    # the pure-torch union oracle.
    split_widths = torch.zeros(max_requests, dtype=torch.int32, device=device)
    per_head = _launch_split_scores(
        bufs,
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
    _launch_union_fusion(
        bufs,
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
