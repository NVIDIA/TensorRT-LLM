# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivalence coverage for the fused score+stats+union pipeline.

The reference leg gathers the production score rows and normalizes +
union-reduces them with a pure-torch float32 oracle; tolerances are
unchanged from the retired Triton reference copies.
"""

import pytest
import torch
from conftest import launch_split_scores as _launch_split_scores
from conftest import make_cute_buffers as _make_cute_buffers
from conftest import stage_score_metadata as _stage_score_metadata
from conftest import write_block_offsets as _write_block_offsets

_SM100_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="TriAttention CuTe kernels require SM100",
)


def _run_fused_union(
    tri,
    request_count,
    source_lengths,
    decode_lengths,
    prompt_lengths,
    mean_cos,
    mean_sin,
    union_out,
):
    """Run the fused score+stats+normalized-union pipeline."""
    _stage_score_metadata(tri, request_count, source_lengths, decode_lengths, prompt_lengths)
    tri._mean_cos[:request_count].copy_(mean_cos[:request_count])
    tri._mean_sin[:request_count].copy_(mean_sin[:request_count])
    tri._launch_score(request_count)
    columns = min(union_out.shape[1], tri._selection_scores_rows.shape[1])
    union_out[:request_count, :columns].copy_(tri._selection_scores_rows[:request_count, :columns])


def _reference_union_scores(
    scores_rows: torch.Tensor, decode_lengths: torch.Tensor
) -> torch.Tensor:
    """Compute the normalized max-fold union score oracle."""
    request_count, _, width = scores_rows.shape
    combined = torch.full(
        (request_count, width), float("-inf"), dtype=torch.float32, device=scores_rows.device
    )
    for request in range(request_count):
        decode_length = int(decode_lengths[request])
        if decode_length <= 0:
            continue
        valid = scores_rows[request, :, :decode_length].to(torch.float32)
        mean = valid.mean(dim=1, keepdim=True)
        std = ((valid - mean).square().sum(dim=1, keepdim=True) / decode_length).sqrt()
        combined[request, :decode_length] = ((valid - mean) / std.clamp_min(1e-6)).amax(dim=0)
    return combined


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
        # Mixed-prompt request group (one start mid-tile, one page-aligned) — the
        # case the fused pipeline previously declined. Starts are per-request
        # runtime reads, so one representative row covers the family.
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
    """Check fused union rows against the split score-normalize-union path."""
    pytest.importorskip("cutlass")

    torch.manual_seed(20260721)
    device = torch.device("cuda")
    seq_len = 256
    num_pages = seq_len // tokens_per_block
    # Shuffled pages catch fragment/page mix-ups; ragged lengths land
    # mid-tile.
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
    logical_source_lengths = torch.tensor([float(seq_len), float(seq_len + 1)], device=device)
    phase = (logical_source_lengths[:, None, None] + offsets[None, :, None]) * omega[None, None]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()

    common = dict(
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
    tri = _make_cute_buffers(eviction_mode="union", **common)
    # The split reference leg runs on its own score-only buffers.
    ref_tri = _make_cute_buffers(eviction_mode="per_head", **common)
    k_plane = [2 * page for page in page_permutation]
    v_plane = [2 * page + 1 for page in page_permutation]
    encoded = torch.tensor(
        [[[k_plane, v_plane], [k_plane, v_plane]]], dtype=torch.int32, device=device
    )
    _write_block_offsets(tri, encoded)
    _write_block_offsets(ref_tri, encoded)
    if valid_lens is None:
        valid_lens = [seq_len, seq_len]
    source_lengths = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    request_count = 2
    if isinstance(score_starts, int):
        score_starts = [score_starts] * request_count
    assert len(score_starts) == request_count

    # Reference: the production score gather over the same decode windows,
    # then the pure-torch union oracle.
    split_widths = torch.empty(request_count, dtype=torch.int32, device=device)
    prompt_lengths = torch.tensor(score_starts, dtype=torch.int32, device=device)
    per_head = _launch_split_scores(
        ref_tri,
        request_count,
        source_lengths,
        split_widths,
        prompt_lengths,
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
    _run_fused_union(
        tri,
        request_count,
        source_lengths,
        fused_widths,
        prompt_lengths,
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
def test_union_fusion_frequency_count_guard_raises() -> None:
    """Reject unsupported 16-frequency fused-kernel geometry."""
    pytest.importorskip("cutlass")

    from tensorrt_llm._torch.kv_cache_compression.triattention.triattention_cute_score_fused import (  # noqa: E501
        _TriAttentionScoreKernel,
    )

    with pytest.raises(ValueError, match="frequencies"):
        _TriAttentionScoreKernel(
            num_layers=1,
            score_token_capacity=256,
            num_q_heads=8,
            num_freqs=16,
            pool_shape=(2, 2, 1, 128, 32),
            pool_strides=(8192, 4096, 4096, 32, 1),
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
    logical_source_lengths = (
        torch.arange(max_requests, dtype=torch.float32, device=device) + seq_len
    )
    phase = (logical_source_lengths[:, None, None] + offsets[None, :, None]) * omega[None, None]
    mean_cos = torch.cos(phase).mean(dim=1).contiguous()
    mean_sin = torch.sin(phase).mean(dim=1).contiguous()

    common = dict(
        layer_pools=layer_pools,
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
    tri = _make_cute_buffers(eviction_mode="union", max_requests=max_requests, **common)
    assert (tri._score_scratch.numel() > 2**31) == (max_requests == 64)
    # The split reference leg only scores the two live requests; its own
    # small per_head buffers keep the giant scratch on the union side.
    ref_tri = _make_cute_buffers(eviction_mode="per_head", max_requests=request_count, **common)
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    for staged in (tri, ref_tri):
        staged._block_offsets_device.zero_()
        staged._block_offsets_device[0, :request_count, 0, :num_pages] = 2 * page_ids
        staged._block_offsets_device[0, :request_count, 1, :num_pages] = 2 * page_ids + 1

    source_lengths = torch.zeros(max_requests, dtype=torch.int32, device=device)
    prompt_lengths = torch.zeros(max_requests, dtype=torch.int32, device=device)
    source_lengths[:request_count] = torch.tensor(valid_lens, dtype=torch.int32, device=device)
    prompt_lengths[:request_count] = torch.tensor(score_starts, dtype=torch.int32, device=device)

    # Reference: the split score gather over the same decode windows, then
    # the pure-torch union oracle.
    split_widths = torch.zeros(max_requests, dtype=torch.int32, device=device)
    per_head = _launch_split_scores(
        ref_tri,
        request_count,
        source_lengths,
        split_widths,
        prompt_lengths,
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
    _run_fused_union(
        tri,
        max_requests,
        source_lengths,
        fused_widths,
        prompt_lengths,
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
