# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test suite for KV compressor kernels.

Tests cover: prefill/decode corner cases, state updates, varlen, MTP support.
Run: pytest -s tests/unittest/_torch/attention/sparse/test_compressor_kernel.py
"""

import pytest
import torch
import triton
from utils.util import skip_pre_blackwell

_CUDA_SUPPORTED_HEAD_DIMS = (128, 512)


def prefill_kernel(
    kv_score: torch.Tensor,
    ape: torch.Tensor,
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    kv_comp: torch.Tensor,
    paged_kv: torch.Tensor,
    paged_score: torch.Tensor,
    block_table_kv: torch.Tensor,
    max_outputs: int,
    block_table_score: torch.Tensor = None,
    compress_ratio: int = None,
    head_dim: int = None,
    page_size: int = 32,
):
    """CUDA prefill kernel wrapper with head_dim skip guard."""
    if head_dim not in _CUDA_SUPPORTED_HEAD_DIMS:
        pytest.skip(
            f"CUDA prefill kernel only supports head_dim in {_CUDA_SUPPORTED_HEAD_DIMS}, got {head_dim}"
        )
    if block_table_score is None:
        block_table_score = block_table_kv

    paged_kv_2d = paged_kv.flatten(1) if paged_kv.dim() == 3 else paged_kv
    paged_score_2d = paged_score.flatten(1) if paged_score.dim() == 3 else paged_score

    torch.ops.trtllm.compressor_prefill_reduction(
        kv_score,
        ape,
        paged_kv_2d,
        paged_score_2d,
        block_table_kv,
        block_table_score,
        kv_comp,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_new_comp_kv,
        kv_lens.shape[0],
        page_size,
        head_dim,
        compress_ratio,
        max_outputs,
    )


def prepare_compress_output(
    cu_new_comp_kv: torch.Tensor,
    batch_size: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Pre-allocate output tensor for compression kernels.

    Args:
        cu_new_comp_kv: [bsz+1] cumulative output offsets
        batch_size: Number of batches
        head_dim: Dimension of KV head
        device: Target device
        dtype: Output dtype (default bfloat16)

    Returns:
        kv_comp: [total_outputs, head_dim] output buffer
    """
    total_outputs = cu_new_comp_kv[-1].item()
    kv_comp = torch.empty(total_outputs, head_dim, device=device, dtype=dtype)
    return kv_comp


def decode_kernel(
    kv_score: torch.Tensor,
    ape: torch.Tensor,
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    cu_new_comp_kv: torch.Tensor,
    kv_comp: torch.Tensor,
    paged_kv: torch.Tensor,
    paged_score: torch.Tensor,
    block_table_kv: torch.Tensor,
    block_table_score: torch.Tensor = None,
    compress_ratio: int = None,
    head_dim: int = None,
    page_size: int = 32,
    next_n: int = 1,
):
    """CUDA decode kernel wrapper with head_dim skip guard."""
    if head_dim not in _CUDA_SUPPORTED_HEAD_DIMS:
        pytest.skip(
            f"CUDA decode kernel only supports head_dim in {_CUDA_SUPPORTED_HEAD_DIMS}, got {head_dim}"
        )
    if block_table_score is None:
        block_table_score = block_table_kv

    torch.ops.trtllm.compressor_paged_kv_compress(
        kv_score,
        ape,
        paged_kv,
        paged_score,
        block_table_kv,
        block_table_score,
        kv_comp,
        kv_lens,
        cu_seq_lens,
        cu_new_comp_kv,
        kv_lens.shape[0],
        page_size,
        head_dim,
        compress_ratio,
        next_n,
    )


# ============================================================================
# PyTorch References (mirror model.py logic)
# ============================================================================


def run_pytorch_reference(
    new_kv, new_score, ape, kv_state, score_state, token_idx, compress_ratio, head_dim, overlap
):
    """Decode reference: single token update + conditional compression."""
    bsz = kv_state.shape[0]
    should_compress = (token_idx + 1) % compress_ratio == 0
    ape_ratio = token_idx % compress_ratio

    score_val = new_score + ape[ape_ratio]
    kv_val = new_kv
    output = None

    if overlap:
        update_idx = compress_ratio + ape_ratio
        kv_state[:bsz, update_idx] = kv_val.squeeze(1)
        score_state[:bsz, update_idx] = score_val.squeeze(1)
        if should_compress:
            d = head_dim
            kv_cat = torch.cat(
                [kv_state[:bsz, :compress_ratio, :d], kv_state[:bsz, compress_ratio:, d:]], dim=1
            )
            score_cat = torch.cat(
                [score_state[:bsz, :compress_ratio, :d], score_state[:bsz, compress_ratio:, d:]],
                dim=1,
            )
            output = (kv_cat * score_cat.softmax(dim=1)).sum(dim=1, keepdim=True)
            kv_state[:bsz, :compress_ratio] = kv_state[:bsz, compress_ratio:].clone()
            score_state[:bsz, :compress_ratio] = score_state[:bsz, compress_ratio:].clone()
    else:
        kv_state[:bsz, ape_ratio] = kv_val.squeeze(1)
        score_state[:bsz, ape_ratio] = score_val.squeeze(1)
        if should_compress:
            output = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

    return output


def run_pytorch_prefill_reference(
    kv, score, ape, kv_state, score_state, compress_ratio, head_dim, overlap
):
    """Prefill reference: bulk compression + state update."""
    bsz, seqlen, _ = kv.size()
    ratio = compress_ratio
    remainder = seqlen % ratio
    cutoff = seqlen - remainder
    offset = ratio if overlap else 0

    # State update
    if overlap and cutoff >= ratio:
        kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
        score_state[:bsz, :ratio] = score[:, cutoff - ratio : cutoff] + ape

    if remainder > 0:
        kv, kv_state[:bsz, offset : offset + remainder] = kv.split([cutoff, remainder], dim=1)
        score_state[:bsz, offset : offset + remainder] = score[:, cutoff:] + ape[:remainder]
        score = score[:, :cutoff]

    if cutoff == 0:
        return None

    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + ape

    if overlap:
        b, s, r, _ = kv.size()
        d = head_dim

        kv_transformed = torch.zeros(b, s, 2 * ratio, d, device=kv.device, dtype=kv.dtype)
        score_transformed = torch.full(
            (b, s, 2 * ratio, d), float("-inf"), device=score.device, dtype=score.dtype
        )

        kv_transformed[:, :, ratio:] = kv[:, :, :, d:]
        score_transformed[:, :, ratio:] = score[:, :, :, d:]
        kv_transformed[:, 1:, :ratio] = kv[:, :-1, :, :d]
        score_transformed[:, 1:, :ratio] = score[:, :-1, :, :d]

        kv, score = kv_transformed, score_transformed

    output = (kv * score.softmax(dim=2)).sum(dim=2)
    return output if seqlen >= ratio else None


def run_pytorch_prefill_reference_varlen(
    kv_score, ape, kv_lens, start_pos, compress_ratio, head_dim, overlap, kv_state, score_state
):
    """Varlen prefill reference: process each batch independently."""
    bsz = kv_lens.shape[0]
    coff = 2 if overlap else 1
    state_dim = coff * head_dim
    ratio = compress_ratio
    offset = ratio if overlap else 0

    # Compute sequence lengths and cumulative offsets
    seq_lens = kv_lens - start_pos
    cu_seq_lens = torch.zeros(bsz + 1, device=kv_score.device, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

    outputs = []

    for b in range(bsz):
        seqlen = seq_lens[b].item()
        input_start = cu_seq_lens[b].item()

        kv_score_b = kv_score[input_start : input_start + seqlen]
        kv = kv_score_b[:, :state_dim].unsqueeze(0)
        score = kv_score_b[:, state_dim:].unsqueeze(0)

        remainder = seqlen % ratio
        cutoff = seqlen - remainder

        # State update
        if overlap and cutoff >= ratio:
            kv_state[b : b + 1, :ratio] = kv[:, cutoff - ratio : cutoff]
            score_state[b : b + 1, :ratio] = score[:, cutoff - ratio : cutoff] + ape

        if remainder > 0:
            kv_state[b : b + 1, offset : offset + remainder] = kv[:, cutoff:]
            score_state[b : b + 1, offset : offset + remainder] = (
                score[:, cutoff:] + ape[:remainder]
            )

        if cutoff == 0:
            continue

        kv_comp = kv[:, :cutoff].unflatten(1, (-1, ratio))
        score_comp = score[:, :cutoff].unflatten(1, (-1, ratio)) + ape

        if overlap:
            _, s, r, _ = kv_comp.size()
            d = head_dim

            kv_transformed = torch.zeros(1, s, 2 * ratio, d, device=kv.device, dtype=kv.dtype)
            score_transformed = torch.full(
                (1, s, 2 * ratio, d), float("-inf"), device=score.device, dtype=score.dtype
            )
            kv_transformed[:, :, ratio:] = kv_comp[:, :, :, d:]
            score_transformed[:, :, ratio:] = score_comp[:, :, :, d:]
            kv_transformed[:, 1:, :ratio] = kv_comp[:, :-1, :, :d]
            score_transformed[:, 1:, :ratio] = score_comp[:, :-1, :, :d]
            kv_comp, score_comp = kv_transformed, score_transformed

        output = (kv_comp * score_comp.softmax(dim=2)).sum(dim=2)
        outputs.append(output.squeeze(0))

    if outputs:
        return torch.cat(outputs, dim=0)  # [total_outputs, head_dim]
    return torch.empty(0, head_dim, device=kv_score.device, dtype=torch.float32)


# ============================================================================
# Test Utilities
# ============================================================================


def fuse_kv_score(kv, score):
    """Fuse kv and score into [*, 2*dim]."""
    return torch.cat([kv, score], dim=-1)


def prepare_decode_metadata(
    batch_size: int,
    compress_ratio: int,
    head_dim: int,
    device: torch.device,
    next_n: int = 1,
):
    """Prepare decode metadata for tests."""
    max_compressions = (next_n + compress_ratio - 1) // compress_ratio
    cu_seq_lens = torch.arange(
        0, (batch_size + 1) * next_n, next_n, device=device, dtype=torch.int32
    )
    cu_outputs = torch.arange(
        0, (batch_size + 1) * max_compressions, max_compressions, device=device, dtype=torch.int32
    )
    return cu_seq_lens, cu_outputs


def prepare_prefill_metadata(
    kv_lens: torch.Tensor,
    start_pos: torch.Tensor,
    compress_ratio: int,
    head_dim: int,
    device: torch.device = None,
):
    """Prepare prefill metadata for tests."""
    batch_size = kv_lens.shape[0]
    if device is None:
        device = kv_lens.device

    if start_pos is None:
        start_pos = torch.zeros(batch_size, device=device, dtype=torch.int32)

    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(
        kv_lens // compress_ratio - start_pos // compress_ratio, min=1
    )

    cu_seq_lens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)

    cu_outputs = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_outputs[1:] = torch.cumsum(num_outputs_per_batch, dim=0)

    return cu_seq_lens, cu_outputs


def create_paged_cache(batch_size, seqlen, compress_ratio, head_dim, overlap, page_size=4):
    """Create paged cache tensors."""
    coff = 2 if overlap else 1
    state_dim = coff * head_dim
    total_positions = seqlen + (compress_ratio if overlap else 0)
    max_blocks = (total_positions + page_size - 1) // page_size
    num_blocks = batch_size * max_blocks

    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    return paged_kv, paged_score, block_table, page_size, max_blocks


def pack_prefill_inputs(kv_list, score_list):
    """Pack variable-length inputs into [m, 2*state_dim] format."""
    seq_lens = torch.tensor([kv.shape[0] for kv in kv_list], device="cuda", dtype=torch.int32)
    kv_score = fuse_kv_score(torch.cat(kv_list, dim=0), torch.cat(score_list, dim=0))
    return kv_score, seq_lens


# ============================================================================
# Correctness Tests
# ============================================================================

PREFILL_CONFIGS = [
    # Overlap mode (ratio=4)
    pytest.param(4, 32, 4, 128, True, id="overlap_large_head_dim"),
    pytest.param(1, 20, 4, 512, True, id="overlap_hd512_5chunks"),
    # Basic mode (ratio=128)
    pytest.param(1, 128, 128, 128, False, id="basic_hd128_eq_ratio"),
    pytest.param(1, 256, 128, 512, False, id="basic_hd512_2chunks"),
]


@pytest.mark.parametrize("batch_size,seqlen,compress_ratio,head_dim,overlap", PREFILL_CONFIGS)
def test_prefill_corner_cases(batch_size, seqlen, compress_ratio, head_dim, overlap):
    """Test prefill kernel corner cases."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim

    kv = torch.randn(batch_size, seqlen, state_dim, device="cuda")
    score = torch.randn(batch_size, seqlen, state_dim, device="cuda")
    ape = torch.randn(compress_ratio, state_dim, device="cuda")
    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")
    paged_kv, paged_score, block_table, page_size, _ = create_paged_cache(
        batch_size, seqlen, compress_ratio, head_dim, overlap
    )

    out_py = run_pytorch_prefill_reference(
        kv.clone(),
        score.clone(),
        ape,
        kv_state_py,
        score_state_py,
        compress_ratio,
        head_dim,
        overlap,
    )

    kv_score = fuse_kv_score(kv.view(-1, state_dim), score.view(-1, state_dim))
    kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    prefill_kernel(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        page_size,
    )

    num_chunks = seqlen // compress_ratio
    if out_py is None or num_chunks == 0:
        pass  # No compression expected
    elif kv_comp.numel() == 0:
        pytest.fail("cuTile returned empty output but PyTorch returned valid output")
    else:
        out_reshaped = kv_comp.view(batch_size, num_chunks, head_dim)
        assert torch.allclose(out_py.to(kv_comp.dtype), out_reshaped, rtol=2e-3, atol=5e-3), (
            f"Output mismatch: max diff = {(out_py.to(kv_comp.dtype) - out_reshaped).abs().max():.6f}"
        )


DECODE_CONFIGS = [
    pytest.param(1, 4, 128, True, 12, id="overlap_large_head_dim"),
    pytest.param(1, 4, 512, True, 8, id="overlap_hd512"),
    pytest.param(1, 128, 128, False, 256, id="basic_hd128_2compressions"),
    pytest.param(1, 128, 512, False, 256, id="basic_hd512_2compressions"),
]


@pytest.mark.parametrize("batch_size,compress_ratio,head_dim,overlap,num_steps", DECODE_CONFIGS)
def test_decode_corner_cases(batch_size, compress_ratio, head_dim, overlap, num_steps):
    """Test decode kernel corner cases."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim
    page_size = 8
    max_blocks = (num_steps + compress_ratio + page_size - 1) // page_size

    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Pre-compute decode metadata
    cu_seq_lens, cu_outputs = prepare_decode_metadata(
        batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
    )

    # Pre-fill for overlap mode (all batches)
    if overlap:
        init_kv = torch.randn(compress_ratio, state_dim, device="cuda")
        init_score = torch.randn(compress_ratio, state_dim, device="cuda")
        kv_state_py[:, :compress_ratio] = init_kv
        score_state_py[:, :compress_ratio] = init_score
        for b in range(batch_size):
            for r in range(compress_ratio):
                log_block, offset = r // page_size, r % page_size
                phys_block = block_table[b, log_block].item()
                paged_kv[phys_block, offset] = init_kv[r]
                paged_score[phys_block, offset] = init_score[r]

    for step in range(num_steps):
        new_kv = torch.randn(batch_size, state_dim, device="cuda")
        new_score = torch.randn(batch_size, state_dim, device="cuda")

        # For overlap mode, account for initial compress_ratio tokens in cache
        # token_idx is the absolute position: compress_ratio + step for overlap, step for basic
        token_idx = (compress_ratio + step) if overlap else step
        total_tokens = token_idx + 1

        # PyTorch reference
        out_py = run_pytorch_reference(
            new_kv.unsqueeze(1),
            new_score.unsqueeze(1),
            ape,
            kv_state_py,
            score_state_py,
            token_idx,
            compress_ratio,
            head_dim,
            overlap,
        )

        # CUDA decode kernel
        kv_score = fuse_kv_score(new_kv, new_score)  # [bsz, 2*state_dim]
        kv_lens = torch.full((batch_size,), total_tokens, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), token_idx, device="cuda", dtype=torch.int32)

        decode_kernel(
            kv_score,
            ape,
            kv_lens,
            start_pos,
            cu_seq_lens,
            cu_outputs,
            kv_comp,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            page_size,
            next_n=1,
        )

        should_compress = (step + 1) % compress_ratio == 0
        if should_compress:
            if out_py is not None:
                for b in range(batch_size):
                    out_idx = cu_outputs[b].item()
                    diff = out_py[b, 0, :head_dim].to(kv_comp.dtype) - kv_comp[out_idx, :]
                    assert torch.allclose(
                        out_py[b, 0, :head_dim].to(kv_comp.dtype),
                        kv_comp[out_idx, :],
                        rtol=1e-2,
                        atol=1e-3,
                    ), f"Step {step}, Batch {b}: mismatch diff={(diff).abs().max():.6f}"


def test_prefill_accepts_bf16_kv_score_with_fp32_state():
    """Prefill accepts bf16 kv_score while preserving fp32 compressor state."""
    batch_size, seqlen, compress_ratio, head_dim, overlap = 1, 4, 4, 128, True
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim

    kv_bf16 = torch.randn(batch_size, seqlen, state_dim, device="cuda").bfloat16()
    score_bf16 = torch.randn(batch_size, seqlen, state_dim, device="cuda").bfloat16()
    kv = kv_bf16.float()
    score = score_bf16.float()
    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")
    out_py = run_pytorch_prefill_reference(
        kv.clone(),
        score.clone(),
        ape,
        kv_state_py,
        score_state_py,
        compress_ratio,
        head_dim,
        overlap,
    )

    paged_kv, paged_score, block_table, page_size, _ = create_paged_cache(
        batch_size, seqlen, compress_ratio, head_dim, overlap, page_size=8
    )
    assert paged_kv.dtype == torch.float32

    kv_score = fuse_kv_score(kv_bf16.view(-1, state_dim), score_bf16.view(-1, state_dim))
    assert kv_score.dtype == torch.bfloat16

    kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )

    prefill_kernel(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        paged_kv,
        paged_score,
        block_table,
        max_outputs=1,
        block_table_score=block_table,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        page_size=page_size,
    )

    assert out_py is not None
    assert torch.allclose(out_py.to(kv_comp.dtype).view_as(kv_comp), kv_comp, rtol=2e-3, atol=5e-3)

    for p in range(seqlen):
        phys = block_table[0, p // page_size].item()
        off = p % page_size
        assert torch.allclose(paged_kv[phys, off], kv[0, p], atol=1e-5)
        assert torch.allclose(
            paged_score[phys, off], score[0, p] + ape[p % compress_ratio], atol=1e-5
        )


def test_decode_accepts_bf16_kv_score_with_fp32_state():
    """Decode accepts bf16 kv_score and computes start_pos as kv_len - next_n."""
    batch_size, compress_ratio, head_dim, overlap = 1, 4, 128, True
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim
    page_size = 8
    decode_steps = 4
    max_blocks = 1

    ape = torch.randn(compress_ratio, state_dim, device="cuda")
    paged_kv = torch.zeros(batch_size * max_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros_like(paged_kv)
    assert paged_kv.dtype == torch.float32
    block_table = torch.zeros(batch_size, max_blocks, device="cuda", dtype=torch.int32)

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")
    init_kv = torch.randn(compress_ratio, state_dim, device="cuda")
    init_score = torch.randn(compress_ratio, state_dim, device="cuda")
    kv_state_py[:, :compress_ratio] = init_kv
    score_state_py[:, :compress_ratio] = init_score
    paged_kv[0, :compress_ratio] = init_kv
    paged_score[0, :compress_ratio] = init_score

    cu_seq_lens, cu_outputs = prepare_decode_metadata(
        batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
    )

    for step in range(decode_steps):
        token_idx = compress_ratio + step
        total_tokens = token_idx + 1
        new_kv_bf16 = torch.randn(batch_size, state_dim, device="cuda").bfloat16()
        new_score_bf16 = torch.randn(batch_size, state_dim, device="cuda").bfloat16()
        new_kv = new_kv_bf16.float()
        new_score = new_score_bf16.float()

        out_py = run_pytorch_reference(
            new_kv.unsqueeze(1),
            new_score.unsqueeze(1),
            ape,
            kv_state_py,
            score_state_py,
            token_idx,
            compress_ratio,
            head_dim,
            overlap,
        )

        kv_score = fuse_kv_score(new_kv_bf16, new_score_bf16)
        assert kv_score.dtype == torch.bfloat16
        kv_lens = torch.full((batch_size,), total_tokens, device="cuda", dtype=torch.int32)
        stale_start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

        decode_kernel(
            kv_score,
            ape,
            kv_lens,
            stale_start_pos,
            cu_seq_lens,
            cu_outputs,
            kv_comp,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            page_size,
            next_n=1,
        )

        phys = block_table[0, token_idx // page_size].item()
        off = token_idx % page_size
        assert torch.allclose(paged_kv[phys, off], new_kv[0], atol=1e-5)
        assert torch.allclose(
            paged_score[phys, off], new_score[0] + ape[token_idx % compress_ratio], atol=1e-5
        )

        if out_py is not None:
            assert torch.allclose(
                out_py[0, 0, :head_dim].to(kv_comp.dtype),
                kv_comp[0],
                rtol=2e-3,
                atol=5e-3,
            )


STATE_UPDATE_CONFIGS = [
    pytest.param(1, 12, 4, 128, True, id="overlap_hd128_3chunks"),
    pytest.param(1, 13, 4, 512, True, id="overlap_hd512_3chunks_1remainder"),
    pytest.param(1, 256, 128, 128, False, id="basic_hd128_2chunks"),
    pytest.param(1, 260, 128, 512, False, id="basic_hd512_2chunks_4remainder"),
]


@pytest.mark.parametrize("batch_size,seqlen,compress_ratio,head_dim,overlap", STATE_UPDATE_CONFIGS)
def test_prefill_state_update(batch_size, seqlen, compress_ratio, head_dim, overlap):
    """Verify prefill state updates match reference — all chunks, not just the last one.

    Checks two things:
      1. All full-chunk positions [0, cutoff) are written to paged cache with
         the correct kv and score+APE values (block reuse requirement: this
         section would FAIL against the old kernel that only wrote the last chunk).
      2. The last-chunk + remainder state matches the PyTorch reference state
         (continuity requirement for decode following prefill).
    """
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim

    kv = torch.arange(batch_size * seqlen * state_dim, dtype=torch.float32, device="cuda").reshape(
        batch_size, seqlen, state_dim
    )
    score = torch.arange(
        batch_size * seqlen * state_dim, dtype=torch.float32, device="cuda"
    ).reshape(batch_size, seqlen, state_dim)
    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    paged_kv, paged_score, block_table, page_size, max_blocks = create_paged_cache(
        batch_size, seqlen, compress_ratio, head_dim, overlap
    )

    # Run PyTorch reference (sets last-chunk + remainder in kv_state_py/score_state_py)
    _ = run_pytorch_prefill_reference(
        kv.clone(),
        score.clone(),
        ape,
        kv_state_py,
        score_state_py,
        compress_ratio,
        head_dim,
        overlap,
    )

    # Run cuTile kernel
    kv_packed = kv.view(-1, state_dim)
    score_packed = score.view(-1, state_dim)
    kv_score = fuse_kv_score(kv_packed, score_packed)
    kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    prefill_kernel(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        page_size,
    )

    remainder = seqlen % compress_ratio
    cutoff = seqlen - remainder
    offset = compress_ratio if overlap else 0

    # --- Check 1: ALL full-chunk positions [0, cutoff) ---
    # Expected: paged_kv[phys, blk_off] == kv[b, p]
    #           paged_score[phys, blk_off] == score[b, p] + ape[p % compress_ratio]
    # This check FAILS against the old kernel (which only wrote the last chunk),
    # proving these positions are now correctly populated for block reuse.
    for b in range(batch_size):
        for p in range(cutoff):
            log_block = p // page_size
            block_offset = p % page_size
            phys_block = block_table[b, log_block].item()
            r = p % compress_ratio

            got_kv = paged_kv[phys_block, block_offset].float()
            exp_kv = kv[b, p].float()
            assert torch.allclose(got_kv, exp_kv, atol=1e-5), (
                f"batch={b} pos={p}: paged_kv mismatch max_diff={(got_kv - exp_kv).abs().max():.6f}"
            )

            got_sc = paged_score[phys_block, block_offset].float()
            exp_sc = (score[b, p] + ape[r]).float()
            assert torch.allclose(got_sc, exp_sc, atol=1e-5), (
                f"batch={b} pos={p}: paged_score mismatch "
                f"max_diff={(got_sc - exp_sc).abs().max():.6f}"
            )

    # --- Check 2: last-chunk + remainder match the PyTorch reference state ---
    # The PyTorch reference stores the "virtual" state needed for decode continuation.
    # Verify the kernel's paged cache agrees for those positions.
    kv_state_kernel = torch.zeros_like(kv_state_py)
    score_state_kernel = torch.full_like(score_state_py, float("-inf"))

    for b in range(batch_size):
        if overlap and cutoff >= compress_ratio:
            for r in range(compress_ratio):
                abs_pos = cutoff - compress_ratio + r
                log_block = abs_pos // page_size
                block_offset = abs_pos % page_size
                phys_block = block_table[b, log_block].item()
                kv_state_kernel[b, r] = paged_kv[phys_block, block_offset]
                score_state_kernel[b, r] = paged_score[phys_block, block_offset]

        if remainder > 0:
            for r in range(remainder):
                abs_pos = cutoff + r
                log_block = abs_pos // page_size
                block_offset = abs_pos % page_size
                phys_block = block_table[b, log_block].item()
                kv_state_kernel[b, offset + r] = paged_kv[phys_block, block_offset]
                score_state_kernel[b, offset + r] = paged_score[phys_block, block_offset]

    assert torch.allclose(kv_state_py, kv_state_kernel, atol=1e-5), (
        f"KV state mismatch: {(kv_state_py - kv_state_kernel).abs().max():.6f}"
    )
    assert torch.allclose(score_state_py, score_state_kernel, atol=1e-5), (
        f"Score state mismatch: {(score_state_py - score_state_kernel).abs().max():.6f}"
    )


PREFILL_ALL_BLOCKS_CONFIGS = [
    pytest.param(1, 12, 4, 128, True, id="overlap_hd128_3chunks"),
    pytest.param(1, 20, 4, 512, True, id="overlap_hd512_5chunks"),
    pytest.param(1, 256, 128, 128, False, id="basic_hd128_2chunks"),
    pytest.param(1, 260, 128, 512, False, id="basic_hd512_2chunks_4remainder"),
]


@pytest.mark.parametrize(
    "batch_size,seqlen,compress_ratio,head_dim,overlap", PREFILL_ALL_BLOCKS_CONFIGS
)
def test_prefill_paged_state_all_blocks(batch_size, seqlen, compress_ratio, head_dim, overlap):
    """Verify prefill writes ALL token positions to paged cache (block reuse requirement).

    Before the fix, only the last full chunk + remainder were written to paged cache.
    Every chunk block must now be written so reused requests can read any prefix slice.

    Checks: for every full-chunk position p in [0, cutoff):
      paged_kv[phys, blk_off]    == kv[b, p]
      paged_score[phys, blk_off] == score[b, p] + ape[p % compress_ratio]
    """
    coff = 2 if overlap else 1
    state_dim = coff * head_dim

    kv = torch.randn(batch_size, seqlen, state_dim, device="cuda")
    score = torch.randn(batch_size, seqlen, state_dim, device="cuda")
    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    paged_kv, paged_score, block_table, page_size, _ = create_paged_cache(
        batch_size, seqlen, compress_ratio, head_dim, overlap
    )

    kv_score = fuse_kv_score(kv.view(-1, state_dim), score.view(-1, state_dim))
    kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )
    max_outputs = torch.clamp((kv_lens - start_pos) // compress_ratio, min=1).max().item()

    prefill_kernel(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        page_size,
    )

    # Verify every full-chunk position is correctly written
    cutoff = (seqlen // compress_ratio) * compress_ratio
    for b in range(batch_size):
        for p in range(cutoff):
            log_blk = p // page_size
            blk_off = p % page_size
            phys = block_table[b, log_blk].item()
            r = p % compress_ratio  # APE index: position within window

            got_kv = paged_kv[phys, blk_off].float()
            exp_kv = kv[b, p].float()
            assert torch.allclose(got_kv, exp_kv, atol=1e-5), (
                f"batch={b} pos={p}: paged_kv mismatch max_diff={(got_kv - exp_kv).abs().max():.6f}"
            )

            got_sc = paged_score[phys, blk_off].float()
            exp_sc = (score[b, p] + ape[r]).float()
            assert torch.allclose(got_sc, exp_sc, atol=1e-5), (
                f"batch={b} pos={p}: paged_score mismatch "
                f"max_diff={(got_sc - exp_sc).abs().max():.6f}"
            )


BLOCK_REUSE_CONFIGS = [
    # (batch_size, prefix_len, compress_ratio, head_dim, overlap, reuse_at)
    # reuse_at must be on a chunk boundary (reuse_at % compress_ratio == 0)
    # so the decode uses an EARLIER chunk as overlap window, not the last one.
    pytest.param(1, 12, 4, 128, True, 4, id="overlap_hd128_reuse_after_chunk0"),
    pytest.param(1, 12, 4, 512, True, 8, id="overlap_hd512_reuse_after_chunk1"),
    pytest.param(1, 512, 128, 128, False, 128, id="basic_hd128_reuse_after_chunk0"),
    pytest.param(1, 512, 128, 512, False, 256, id="basic_hd512_reuse_after_chunk1"),
]


@pytest.mark.parametrize(
    "batch_size,prefix_len,compress_ratio,head_dim,overlap,reuse_at",
    BLOCK_REUSE_CONFIGS,
)
def test_prefill_block_reuse_decode(
    batch_size, prefix_len, compress_ratio, head_dim, overlap, reuse_at
):
    """End-to-end block reuse: decode from mid-prefix using paged cache written during prefill.

    Simulates the block reuse scenario:
      1. Request A prefills the full prefix [0 .. prefix_len-1], populating paged cache.
      2. Request B reuses A's paged blocks: decodes starting at position reuse_at,
         reading the overlap window from earlier blocks (positions < reuse_at).

    With the old code only the last chunk was in paged cache, so decode triggered at
    reuse_at + compress_ratio - 1 would read garbage for the overlap window.
    With the fix, all blocks are present and decode outputs must match the reference.
    """
    assert reuse_at % compress_ratio == 0, "reuse_at must be on a chunk boundary"
    assert reuse_at < prefix_len, "reuse_at must be within the prefix"

    coff = 2 if overlap else 1
    state_len = coff * compress_ratio
    state_dim = coff * head_dim
    page_size = 8
    decode_steps = compress_ratio * 2  # enough to trigger two compressions
    total_len = prefix_len + decode_steps
    max_blocks = (total_len + (compress_ratio if overlap else 0) + page_size - 1) // page_size

    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    # --- Step 1: Prefill the full prefix ---
    kv_prefill = torch.randn(batch_size, prefix_len, state_dim, device="cuda")
    score_prefill = torch.randn(batch_size, prefix_len, state_dim, device="cuda")

    kv_score_prefill = fuse_kv_score(
        kv_prefill.view(-1, state_dim), score_prefill.view(-1, state_dim)
    )
    kv_lens_pre = torch.full((batch_size,), prefix_len, device="cuda", dtype=torch.int32)
    start_pos_pre = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens_pre, start_pos_pre, compress_ratio, head_dim, "cuda"
    )
    kv_comp_pre = prepare_compress_output(cu_outputs, batch_size, head_dim, "cuda", torch.bfloat16)
    max_outputs = torch.clamp((kv_lens_pre - start_pos_pre) // compress_ratio, min=1).max().item()

    prefill_kernel(
        kv_score_prefill,
        ape,
        kv_lens_pre,
        start_pos_pre,
        cu_seq_lens,
        cu_outputs,
        kv_comp_pre,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        page_size,
    )

    # --- Step 2: Build PyTorch reference state at reuse_at ---
    # The state at reuse_at is: the overlap window = tokens [reuse_at-CR .. reuse_at-1],
    # and any remainder tokens within [reuse_at-CR .. reuse_at).
    # Since reuse_at is on a chunk boundary, remainder = 0.
    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    if overlap and reuse_at >= compress_ratio:
        # Overlap window = last full chunk before reuse_at: [reuse_at-CR .. reuse_at-1]
        kv_state_py[:, :compress_ratio] = kv_prefill[:, reuse_at - compress_ratio : reuse_at]
        score_state_py[:, :compress_ratio] = score_prefill[
            :, reuse_at - compress_ratio : reuse_at
        ] + ape.unsqueeze(0)
    # (non-overlap, or overlap with reuse_at < compress_ratio: state stays zero/-inf)

    # --- Step 3: Decode from reuse_at and verify outputs match reference ---
    cu_seq_lens_dec, cu_outputs_dec = prepare_decode_metadata(
        batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
    )
    kv_comp_dec = prepare_compress_output(
        cu_outputs_dec, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
    )

    for i in range(decode_steps):
        token_idx = reuse_at + i
        total_tokens = token_idx + 1

        new_kv = torch.randn(batch_size, state_dim, device="cuda")
        new_score = torch.randn(batch_size, state_dim, device="cuda")

        # PyTorch reference (uses ground-truth kv_state_py)
        out_py = run_pytorch_reference(
            new_kv.unsqueeze(1),
            new_score.unsqueeze(1),
            ape,
            kv_state_py,
            score_state_py,
            token_idx,
            compress_ratio,
            head_dim,
            overlap,
        )

        # CUDA decode (reads overlap window from paged cache written during prefill)
        kv_score = fuse_kv_score(new_kv, new_score)
        kv_lens = torch.full((batch_size,), total_tokens, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), token_idx, device="cuda", dtype=torch.int32)

        decode_kernel(
            kv_score,
            ape,
            kv_lens,
            start_pos,
            cu_seq_lens_dec,
            cu_outputs_dec,
            kv_comp_dec,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            page_size,
            next_n=1,
        )

        should_compress = (token_idx + 1) % compress_ratio == 0
        if should_compress and out_py is not None:
            for b in range(batch_size):
                out_idx = cu_outputs_dec[b].item()
                diff = out_py[b, 0, :head_dim].to(kv_comp_dec.dtype) - kv_comp_dec[out_idx, :]
                assert torch.allclose(
                    out_py[b, 0, :head_dim].to(kv_comp_dec.dtype),
                    kv_comp_dec[out_idx, :],
                    rtol=1e-2,
                    atol=1e-3,
                ), (
                    f"Block reuse decode step {i} (token_idx={token_idx}), "
                    f"batch {b}: diff={(diff).abs().max():.6f}"
                )


PREFILL_VARLEN_CONFIGS = [
    pytest.param([8, 12, 4, 16], 4, 128, True, id="varlen_hd128_overlap"),
    pytest.param([128, 256, 64, 192], 128, 128, False, id="varlen_hd128_basic"),
    pytest.param([8, 12, 4, 16], 4, 512, True, id="varlen_hd512_overlap"),
    pytest.param([128, 256, 64, 192], 128, 512, False, id="varlen_hd512_basic"),
]


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("seq_lens_list,compress_ratio,head_dim,overlap", PREFILL_VARLEN_CONFIGS)
def test_prefill_varlen(seq_lens_list, compress_ratio, head_dim, overlap):
    """Test prefill with variable-length sequences."""
    batch_size = len(seq_lens_list)
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim

    # Create variable-length packed input
    total_tokens = sum(seq_lens_list)
    kv_packed = torch.randn(total_tokens, state_dim, device="cuda")
    score_packed = torch.randn(total_tokens, state_dim, device="cuda")
    kv_score = fuse_kv_score(kv_packed, score_packed)
    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    # Per-batch metadata
    kv_lens = torch.tensor(seq_lens_list, device="cuda", dtype=torch.int32)
    start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

    # PyTorch reference state
    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Paged cache setup
    page_size = 4
    max_seqlen = max(seq_lens_list)
    total_positions = max_seqlen + (compress_ratio if overlap else 0)
    max_blocks = (total_positions + page_size - 1) // page_size
    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    # PyTorch reference (varlen)
    out_py = run_pytorch_prefill_reference_varlen(
        kv_score.clone(),
        ape,
        kv_lens,
        start_pos,
        compress_ratio,
        head_dim,
        overlap,
        kv_state_py,
        score_state_py,
    )

    # cuTile kernel
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
    )
    seq_lens = kv_lens - start_pos
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    prefill_kernel(
        kv_score,
        ape,
        kv_lens,
        start_pos,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        page_size,
    )

    # Check output - extract only valid outputs from packed output
    # cu_outputs uses min=1 for CUDA graph, but kernel only writes where seqlen >= ratio
    actual_outputs_per_batch = [s // compress_ratio for s in seq_lens_list]
    total_actual_outputs = sum(actual_outputs_per_batch)

    if total_actual_outputs == 0:
        # No compression expected
        assert out_py.numel() == 0, "Expected empty output when no compression"
    elif out_py.numel() == 0:
        pytest.fail("PyTorch returned empty output but expected valid output")
    else:
        # Extract valid outputs from kernel's packed output
        # cu_outputs[b] gives offset for batch b, but includes min=1 padding
        valid_outputs = []
        offset = 0
        for b, actual_count in enumerate(actual_outputs_per_batch):
            # cu_outputs uses clamped count, so we need to compute actual offset
            clamped_count = max(seq_lens_list[b] // compress_ratio, 1)
            if actual_count > 0:
                valid_outputs.append(kv_comp[offset : offset + actual_count])
            offset += clamped_count

        if valid_outputs:
            out_kernel_valid = torch.cat(valid_outputs, dim=0)
            assert torch.allclose(
                out_py.to(out_kernel_valid.dtype), out_kernel_valid, rtol=2e-3, atol=5e-3
            ), (
                f"Output mismatch: max diff = {(out_py.to(out_kernel_valid.dtype) - out_kernel_valid).abs().max():.6f}"
            )
        else:
            assert out_py.numel() == 0, "Expected empty output"


PREFILL_DECODE_CONFIGS = [
    pytest.param(1, 20, 4, 128, True, 12, id="overlap_hd128_prefill20_decode12"),
    pytest.param(1, 256, 128, 128, False, 128, id="basic_hd128_prefill256_decode128"),
    pytest.param(1, 20, 4, 512, True, 12, id="overlap_hd512_prefill20_decode12"),
    pytest.param(1, 256, 128, 512, False, 128, id="basic_hd512_prefill256_decode128"),
]


@pytest.mark.parametrize(
    "batch_size,prefill_len,compress_ratio,head_dim,overlap,decode_steps", PREFILL_DECODE_CONFIGS
)
def test_prefill_then_decode(
    batch_size, prefill_len, compress_ratio, head_dim, overlap, decode_steps
):
    """Test prefill followed by decode (simulates inference)."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim
    page_size = 8
    total_len = prefill_len + decode_steps
    max_blocks = (total_len + (compress_ratio if overlap else 0) + page_size - 1) // page_size

    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    # PyTorch reference state
    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Paged cache
    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    # 1. Prefill
    kv_prefill = torch.randn(batch_size, prefill_len, state_dim, device="cuda")
    score_prefill = torch.randn(batch_size, prefill_len, state_dim, device="cuda")

    # PyTorch prefill
    out_py_prefill = run_pytorch_prefill_reference(
        kv_prefill.clone(),
        score_prefill.clone(),
        ape,
        kv_state_py,
        score_state_py,
        compress_ratio,
        head_dim,
        overlap,
    )

    # cuTile prefill
    kv_packed = kv_prefill.view(-1, state_dim)
    score_packed = score_prefill.view(-1, state_dim)
    kv_score_prefill = fuse_kv_score(kv_packed, score_packed)
    kv_lens_prefill = torch.full((batch_size,), prefill_len, device="cuda", dtype=torch.int32)
    start_pos_prefill = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

    # Pre-compute prefill metadata
    cu_seq_lens, cu_outputs = prepare_prefill_metadata(
        kv_lens_prefill, start_pos_prefill, compress_ratio, head_dim, kv_score_prefill.device
    )
    kv_comp = prepare_compress_output(
        cu_outputs, batch_size, head_dim, kv_score_prefill.device, torch.bfloat16
    )
    seq_lens = kv_lens_prefill - start_pos_prefill
    num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
    max_outputs = num_outputs_per_batch.max().item()

    prefill_kernel(
        kv_score_prefill,
        ape,
        kv_lens_prefill,
        start_pos_prefill,
        cu_seq_lens,
        cu_outputs,
        kv_comp,
        paged_kv,
        paged_score,
        block_table,
        max_outputs,
        block_table,
        compress_ratio,
        head_dim,
        page_size,
    )

    if out_py_prefill is not None and kv_comp.numel() > 0:
        num_chunks = prefill_len // compress_ratio
        out_reshaped = kv_comp.view(batch_size, num_chunks, head_dim)
        assert torch.allclose(
            out_py_prefill.to(kv_comp.dtype), out_reshaped, rtol=2e-3, atol=5e-3
        ), (
            f"Prefill output mismatch: {(out_py_prefill.to(kv_comp.dtype) - out_reshaped).abs().max():.6f}"
        )

    # 2. Decode (continue from where prefill left off)
    decode_start = (prefill_len // compress_ratio) * compress_ratio
    remainder = prefill_len % compress_ratio

    # Pre-compute decode metadata
    cu_seq_lens_decode, cu_outputs_decode = prepare_decode_metadata(
        batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
    )
    kv_comp_decode = prepare_compress_output(
        cu_outputs_decode, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
    )

    for i in range(decode_steps):
        step = decode_start + remainder + i  # Continue from where prefill left off

        new_kv = torch.randn(batch_size, state_dim, device="cuda")
        new_score = torch.randn(batch_size, state_dim, device="cuda")

        # Both prefill and decode use absolute token positions in the state cache.
        # The prefill kernel writes at [cutoff-ratio:cutoff] and [cutoff:cutoff+remainder],
        # so the decode kernel continues from position step = prefill_len + i.
        token_idx = step
        total_tokens = token_idx + 1

        # PyTorch reference
        out_py = run_pytorch_reference(
            new_kv.unsqueeze(1),
            new_score.unsqueeze(1),
            ape,
            kv_state_py,
            score_state_py,
            token_idx,
            compress_ratio,
            head_dim,
            overlap,
        )

        # CUDA decode
        kv_score = fuse_kv_score(new_kv, new_score)
        kv_lens = torch.full((batch_size,), total_tokens, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), token_idx, device="cuda", dtype=torch.int32)

        decode_kernel(
            kv_score,
            ape,
            kv_lens,
            start_pos,
            cu_seq_lens_decode,
            cu_outputs_decode,
            kv_comp_decode,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            page_size,
            next_n=1,
        )

        should_compress = (token_idx + 1) % compress_ratio == 0
        if should_compress and out_py is not None:
            for b in range(batch_size):
                out_idx = cu_outputs_decode[b].item()
                diff = out_py[b, 0, :head_dim].to(kv_comp_decode.dtype) - kv_comp_decode[out_idx, :]
                assert torch.allclose(
                    out_py[b, 0, :head_dim].to(kv_comp_decode.dtype),
                    kv_comp_decode[out_idx, :],
                    rtol=2e-3,
                    atol=5e-3,
                ), (
                    f"Decode step {i} (token_idx={token_idx}), Batch {b}: mismatch diff={(diff).abs().max():.6f}"
                )


# MTP (Multi-Token Prediction) Tests
MTP_CONFIGS = [
    pytest.param(1, 4, 128, True, 4, id="overlap_hd128_next4"),
    pytest.param(1, 4, 512, True, 3, id="overlap_hd512_next3"),
    pytest.param(2, 128, 128, False, 4, id="basic_hd128_multi_batch_next4"),
    pytest.param(1, 128, 512, False, 4, id="basic_hd512_next4"),
]


@pytest.mark.parametrize("batch_size,compress_ratio,head_dim,overlap,next_n", MTP_CONFIGS)
def test_decode_mtp(batch_size, compress_ratio, head_dim, overlap, next_n):
    """Test decode with multiple tokens per request (MTP)."""
    coff = 2 if overlap else 1
    state_len, state_dim = coff * compress_ratio, coff * head_dim
    page_size = 8
    num_steps = compress_ratio * 4  # Enough to trigger multiple compressions
    max_blocks = (num_steps + compress_ratio + page_size - 1) // page_size

    ape = torch.randn(compress_ratio, state_dim, device="cuda")

    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, max_blocks
    )

    kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
    score_state_py = torch.full((batch_size, state_len, state_dim), float("-inf"), device="cuda")

    # Pre-fill for overlap mode
    if overlap:
        init_kv = torch.randn(compress_ratio, state_dim, device="cuda")
        init_score = torch.randn(compress_ratio, state_dim, device="cuda")
        kv_state_py[:, :compress_ratio] = init_kv
        score_state_py[:, :compress_ratio] = init_score
        for b in range(batch_size):
            for r in range(compress_ratio):
                log_block, offset = r // page_size, r % page_size
                phys_block = block_table[b, log_block].item()
                paged_kv[phys_block, offset] = init_kv[r]
                paged_score[phys_block, offset] = init_score[r]

    # Process multiple tokens at once
    # For overlap mode, account for initial compress_ratio tokens in cache
    base_token_idx = compress_ratio if overlap else 0

    step = 0
    while step < num_steps:
        actual_n = min(next_n, num_steps - step)

        # Generate next_n tokens
        new_kv = torch.randn(batch_size * actual_n, state_dim, device="cuda")
        new_score = torch.randn(batch_size * actual_n, state_dim, device="cuda")

        # PyTorch reference: process one token at a time
        py_outputs = []
        for t in range(actual_n):
            token_idx = base_token_idx + step + t
            kv_t = new_kv[t::actual_n].unsqueeze(1)  # Get token t for all batches
            score_t = new_score[t::actual_n].unsqueeze(1)
            out_py = run_pytorch_reference(
                kv_t,
                score_t,
                ape,
                kv_state_py,
                score_state_py,
                token_idx,
                compress_ratio,
                head_dim,
                overlap,
            )
            if out_py is not None:
                py_outputs.append((token_idx, out_py))

        # CUDA decode: process all tokens at once
        kv_score = fuse_kv_score(new_kv, new_score)
        abs_start = base_token_idx + step
        kv_lens = torch.full((batch_size,), abs_start + actual_n, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), abs_start, device="cuda", dtype=torch.int32)

        # Compute decode metadata for actual_n (may differ from next_n at end of loop)
        cu_seq_lens, cu_outputs = prepare_decode_metadata(
            batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=actual_n
        )
        kv_comp = prepare_compress_output(
            cu_outputs, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
        )

        decode_kernel(
            kv_score,
            ape,
            kv_lens,
            start_pos,
            cu_seq_lens,
            cu_outputs,
            kv_comp,
            paged_kv,
            paged_score,
            block_table,
            block_table,
            compress_ratio,
            head_dim,
            page_size,
            next_n=actual_n,
        )

        # Verify outputs match (packed [total_outputs, head_dim] format)
        if len(py_outputs) > 0:
            for i, (token_idx, out_py) in enumerate(py_outputs):
                for b in range(batch_size):
                    out_idx = cu_outputs[b].item() + i
                    diff = out_py[b, 0, :head_dim].to(kv_comp.dtype) - kv_comp[out_idx, :]
                    assert torch.allclose(
                        out_py[b, 0, :head_dim].to(kv_comp.dtype),
                        kv_comp[out_idx, :],
                        rtol=2e-3,
                        atol=2e-3,
                    ), f"Token {token_idx}, Batch {b}: mismatch diff={(diff).abs().max():.6f}"

        step += actual_n


CHUNKED_PREFILL_CONFIGS = [
    # (compress_ratio, head_dim, overlap, batch_size, start_pos, new_seqlen)
    # overlap=True, aligned start_pos
    pytest.param(4, 512, True, 1, 4, 4, id="overlap_sp4_seq4"),
    pytest.param(4, 512, True, 1, 4, 8, id="overlap_sp4_seq8"),
    pytest.param(4, 512, True, 1, 8, 8, id="overlap_sp8_seq8"),
    pytest.param(4, 512, True, 2, 4, 4, id="overlap_batch2_sp4_seq4"),
    pytest.param(4, 512, True, 1, 4, 5, id="overlap_sp4_seq5_remainder"),
    # overlap=True, unaligned start_pos (sp % ratio != 0)
    pytest.param(4, 512, True, 1, 6, 4, id="overlap_sp6_seq4_unaligned"),
    pytest.param(4, 512, True, 1, 6, 5, id="overlap_sp6_seq5_unaligned"),
    pytest.param(4, 512, True, 1, 5, 7, id="overlap_sp5_seq7_unaligned"),
    pytest.param(4, 512, True, 1, 3, 9, id="overlap_sp3_seq9_unaligned"),
    # overlap=False, aligned start_pos
    pytest.param(128, 128, False, 1, 128, 128, id="nonoverlap_sp128_seq128"),
    # overlap=False, unaligned start_pos
    pytest.param(128, 128, False, 1, 50, 206, id="nonoverlap_sp50_seq206_unaligned"),
    pytest.param(128, 128, False, 1, 5, 251, id="nonoverlap_sp5_seq251_2windows"),
]


@pytest.mark.parametrize(
    "compress_ratio,head_dim,overlap,batch_size,start_pos_val,new_seqlen",
    CHUNKED_PREFILL_CONFIGS,
)
def test_chunked_prefill(compress_ratio, head_dim, overlap, batch_size, start_pos_val, new_seqlen):
    """End-to-end chunked prefill: reference (one-shot full prefill) vs
    kernel (initial prefill + chunked prefill on same paged cache).

    Validates that kernel Phase 1 correctly persists state and Phase 2
    correctly reads it back in a subsequent call.
    """
    device = torch.device("cuda")
    coff = 2 if overlap else 1
    state_dim = coff * head_dim
    ratio = compress_ratio
    page_size = 32
    total_seqlen = start_pos_val + new_seqlen

    total_ref_outputs = total_seqlen // ratio
    if total_ref_outputs == 0:
        return

    # Shared paged cache (zeros — no prior state)
    total_positions = total_seqlen + (ratio if overlap else 0)
    max_blocks = (total_positions + page_size - 1) // page_size
    num_blocks = batch_size * max_blocks
    paged_kv = torch.zeros(num_blocks, page_size, state_dim, device=device)
    paged_score = torch.zeros(num_blocks, page_size, state_dim, device=device)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        batch_size, max_blocks
    )
    ape = torch.randn(ratio, state_dim, device=device)

    # Generate full sequence data
    kv_full = torch.randn(batch_size, total_seqlen, state_dim, device=device)
    score_full = torch.randn(batch_size, total_seqlen, state_dim, device=device)

    # ---- Reference: one-shot full prefill (sp=0) ----
    state_len = coff * ratio
    kv_state_ref = torch.zeros(batch_size, state_len, state_dim, device=device)
    score_state_ref = torch.full((batch_size, state_len, state_dim), float("-inf"), device=device)
    ref_out = run_pytorch_prefill_reference(
        kv_full.clone(),
        score_full.clone(),
        ape,
        kv_state_ref,
        score_state_ref,
        ratio,
        head_dim,
        overlap,
    )
    assert ref_out is not None, "Reference should produce output"

    # ---- Kernel call 1: initial prefill (sp=0, kv_len=start_pos_val) ----
    actual_num_out_1 = start_pos_val // ratio
    kv_comp_1 = torch.empty(0, head_dim, device=device, dtype=torch.bfloat16)

    if start_pos_val > 0:
        kv_score_1 = fuse_kv_score(
            kv_full[:, :start_pos_val].reshape(-1, state_dim),
            score_full[:, :start_pos_val].reshape(-1, state_dim),
        )
        kv_lens_1 = torch.full((batch_size,), start_pos_val, device=device, dtype=torch.int32)
        start_pos_1 = torch.zeros(batch_size, device=device, dtype=torch.int32)
        cu_seq_lens_1, cu_outputs_1 = prepare_prefill_metadata(
            kv_lens_1,
            start_pos_1,
            ratio,
            head_dim,
            device,
        )
        kv_comp_1 = prepare_compress_output(
            cu_outputs_1,
            batch_size,
            head_dim,
            device,
            torch.bfloat16,
        )
        max_outputs_1 = max(actual_num_out_1, 1)
        prefill_kernel(
            kv_score_1,
            ape,
            kv_lens_1,
            start_pos_1,
            cu_seq_lens_1,
            cu_outputs_1,
            kv_comp_1,
            paged_kv,
            paged_score,
            block_table,
            max_outputs_1,
            block_table,
            ratio,
            head_dim,
            page_size,
        )

    # ---- Kernel call 2: chunked prefill (sp=start_pos_val, kv_len=total_seqlen) ----
    actual_num_out_2 = total_seqlen // ratio - start_pos_val // ratio
    kv_score_2 = fuse_kv_score(
        kv_full[:, start_pos_val:].reshape(-1, state_dim),
        score_full[:, start_pos_val:].reshape(-1, state_dim),
    )
    kv_lens_2 = torch.full((batch_size,), total_seqlen, device=device, dtype=torch.int32)
    start_pos_2 = torch.full((batch_size,), start_pos_val, device=device, dtype=torch.int32)
    cu_seq_lens_2, cu_outputs_2 = prepare_prefill_metadata(
        kv_lens_2,
        start_pos_2,
        ratio,
        head_dim,
        device,
    )
    kv_comp_2 = prepare_compress_output(
        cu_outputs_2,
        batch_size,
        head_dim,
        device,
        torch.bfloat16,
    )
    max_outputs_2 = max(actual_num_out_2, 1)
    prefill_kernel(
        kv_score_2,
        ape,
        kv_lens_2,
        start_pos_2,
        cu_seq_lens_2,
        cu_outputs_2,
        kv_comp_2,
        paged_kv,
        paged_score,
        block_table,
        max_outputs_2,
        block_table,
        ratio,
        head_dim,
        page_size,
    )

    # ---- Compare per batch: concat(call1_out, call2_out) vs reference ----
    for b in range(batch_size):
        parts = []
        if start_pos_val > 0 and actual_num_out_1 > 0:
            off_1 = cu_outputs_1[b].item()
            parts.append(kv_comp_1[off_1 : off_1 + actual_num_out_1])
        if actual_num_out_2 > 0:
            off_2 = cu_outputs_2[b].item()
            parts.append(kv_comp_2[off_2 : off_2 + actual_num_out_2])

        if not parts:
            continue
        kernel_out_b = torch.cat(parts, dim=0)
        ref_out_b = ref_out[b, :total_ref_outputs, :head_dim].to(torch.bfloat16)

        assert torch.allclose(ref_out_b, kernel_out_b, rtol=2e-3, atol=5e-3), (
            f"batch={b} sp={start_pos_val} seqlen={new_seqlen} "
            f"max_diff={(ref_out_b - kernel_out_b).abs().max():.6f}"
        )


# ============================================================================
# Postprocess + Scatter Kernel Tests
# ============================================================================

try:
    _HAS_POSTPROCESS_SCATTER = hasattr(torch.ops.trtllm, "compressor_postprocess_scatter")
except Exception:
    _HAS_POSTPROCESS_SCATTER = False


def _scatter_reference_to_paged_cache(
    kv: torch.Tensor,
    num_comp_tokens: torch.Tensor,
    cu_kv_comp: torch.Tensor,
    kv_cache: torch.Tensor,
    block_offsets: torch.Tensor,
    tokens_per_block: int,
    head_dim: int,
):
    """Reference paged scatter without calling any CuTile kernel."""
    batch_size = num_comp_tokens.shape[0]
    for b in range(batch_size):
        num_tokens_b = num_comp_tokens[b].item()
        src_start = cu_kv_comp[b].item()
        for t in range(num_tokens_b):
            blk = t // tokens_per_block
            off = t % tokens_per_block
            phys = block_offsets[b, blk].item()
            dst = off * head_dim
            kv_cache[phys, dst : dst + head_dim] = kv[src_start + t]


@pytest.mark.parametrize("rotate_activation", [True, False], ids=["rotate", "no_rotate"])
@pytest.mark.parametrize(
    "batch_size,num_tokens,head_dim,nope_dim,rope_dim,tokens_per_block",
    [
        pytest.param(1, 16, 128, 64, 64, 32, id="b1_t16_hd128"),
        pytest.param(1, 64, 128, 64, 64, 32, id="b1_t64_hd128"),
        pytest.param(2, 32, 128, 64, 64, 32, id="b2_t32_hd128"),
        pytest.param(1, 128, 512, 256, 256, 128, id="b1_t128_hd512"),
        pytest.param(2, 64, 512, 256, 256, 128, id="b2_t64_hd512"),
        pytest.param(4, 32, 512, 256, 256, 128, id="b4_t32_hd512"),
    ],
)
@pytest.mark.skipif(
    not _HAS_POSTPROCESS_SCATTER, reason="Postprocess/scatter CUDA ops not available"
)
def test_fused_postprocess_scatter(
    batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block, rotate_activation
):
    """Compare fused RMSNorm+RoPE+(optional Hadamard)+Scatter vs sequential unfused path.

    Ensures the fused CUDA kernel produces identical paged KV cache output as
    the reference pipeline: RMSNorm -> RoPE -> (Hadamard) -> Scatter.
    """
    torch.manual_seed(42)
    device = "cuda"

    tokens_per_batch = num_tokens
    total_tokens = batch_size * tokens_per_batch

    kv_comp = torch.randn(total_tokens, head_dim, device=device, dtype=torch.bfloat16)
    rms_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16) * 0.1 + 1.0
    rms_eps = 1e-5

    max_pos = total_tokens + 64
    cos_sin_table = torch.randn(max_pos, 2, rope_dim // 2, device=device, dtype=torch.float32)
    position_ids = torch.arange(total_tokens, device=device, dtype=torch.int32)

    max_comp_len = tokens_per_batch + 4
    max_blocks = (max_comp_len + tokens_per_block - 1) // tokens_per_block
    num_blocks = batch_size * max_blocks
    kv_cache_fused = torch.zeros(
        num_blocks, tokens_per_block * head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache_ref = torch.zeros_like(kv_cache_fused)

    block_offsets = torch.zeros(batch_size, max_blocks, device=device, dtype=torch.int32)
    for b in range(batch_size):
        block_offsets[b] = torch.arange(b * max_blocks, (b + 1) * max_blocks, dtype=torch.int32)

    num_comp_tokens = torch.full((batch_size,), tokens_per_batch, device=device, dtype=torch.int32)
    cu_kv_comp = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_kv_comp[1:] = num_comp_tokens.cumsum(0)
    start_pos = torch.zeros(batch_size, device=device, dtype=torch.int32)

    # --- Reference: unfused pipeline ---
    x = _build_postprocess_reference(
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        head_dim,
        rotate_activation,
    )

    _scatter_reference_to_paged_cache(
        x,
        num_comp_tokens,
        cu_kv_comp,
        kv_cache_ref,
        block_offsets,
        tokens_per_block,
        head_dim,
    )

    # --- Fused postprocess + scatter kernel ---
    compressed_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
    torch.ops.trtllm.compressor_postprocess_scatter(
        kv_comp,
        None,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        kv_cache_fused,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        block_offsets,
        compressed_mask,
        tokens_per_block,
        0,
        rotate_activation,
        None,
        None,
    )
    torch.cuda.synchronize()

    max_diff = (kv_cache_fused.float() - kv_cache_ref.float()).abs().max().item()
    assert max_diff < 0.05, f"Postprocess+scatter vs ref max diff = {max_diff}"


@pytest.mark.parametrize(
    "batch_size, tokens_per_batch_list, head_dim, nope_dim, rope_dim, tokens_per_block, mask_pattern",
    [
        # Variable token counts + non-contiguous mask: skip batches 1 and 3
        (4, [8, 4, 6, 2], 128, 64, 64, 4, [True, False, True, False]),
        # All masked except one
        (4, [4, 4, 4, 4], 128, 64, 64, 4, [False, False, True, False]),
        # Alternating mask with head_dim=512
        (4, [4, 8, 4, 8], 512, 384, 128, 4, [True, False, True, False]),
        # Single batch masked (edge case)
        (1, [8], 128, 64, 64, 4, [False]),
        # All unmasked (baseline sanity)
        (3, [6, 4, 8], 128, 64, 64, 4, [True, True, True]),
    ],
)
@pytest.mark.skipif(
    not _HAS_POSTPROCESS_SCATTER, reason="Postprocess/scatter CUDA ops not available"
)
def test_fused_postprocess_scatter_masked_batches(
    batch_size, tokens_per_batch_list, head_dim, nope_dim, rope_dim, tokens_per_block, mask_pattern
):
    """Verify compressed_mask skips scatter for masked batches with variable token counts.

    Tests:
    - Masked batches have zero cache
    - Unmasked batches have correct postprocessed values matching reference
    - Non-contiguous and all-masked/all-unmasked patterns
    """
    torch.manual_seed(42)
    device = "cuda"

    num_comp_tokens_list = tokens_per_batch_list
    num_comp_tokens = torch.tensor(num_comp_tokens_list, device=device, dtype=torch.int32)
    total_tokens = int(num_comp_tokens.sum().item())

    kv_comp = torch.randn(total_tokens, head_dim, device=device, dtype=torch.bfloat16) * 0.1
    rms_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16) * 0.1 + 1.0
    rms_eps = 1e-5

    max_pos = total_tokens + 64
    cos_sin_table = torch.randn(max_pos, 2, rope_dim // 2, device=device, dtype=torch.float32)
    position_ids = torch.arange(total_tokens, device=device, dtype=torch.int32)

    max_tokens = max(num_comp_tokens_list)
    max_comp_len = max_tokens + 4
    max_blocks = (max_comp_len + tokens_per_block - 1) // tokens_per_block
    num_blocks = batch_size * max_blocks
    kv_cache_fused = torch.zeros(
        num_blocks, tokens_per_block * head_dim, device=device, dtype=torch.bfloat16
    )

    block_offsets = torch.zeros(batch_size, max_blocks, device=device, dtype=torch.int32)
    for b in range(batch_size):
        block_offsets[b] = torch.arange(b * max_blocks, (b + 1) * max_blocks, dtype=torch.int32)

    cu_kv_comp = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_kv_comp[1:] = num_comp_tokens.cumsum(0)
    start_pos = torch.zeros(batch_size, device=device, dtype=torch.int32)

    # Build per-token compressed_mask from per-batch mask_pattern
    compressed_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
    for b in range(batch_size):
        if mask_pattern[b]:
            start = cu_kv_comp[b].item()
            end = cu_kv_comp[b + 1].item()
            compressed_mask[start:end] = True

    # Build reference for unmasked batches
    ref = _build_postprocess_reference(
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        head_dim,
        rotate_activation=True,
    )

    torch.ops.trtllm.compressor_postprocess_scatter(
        kv_comp,
        None,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        kv_cache_fused,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        block_offsets,
        compressed_mask,
        tokens_per_block,
        0,
        True,
        None,
        None,
    )
    torch.cuda.synchronize()

    for b in range(batch_size):
        n_tokens = num_comp_tokens_list[b]
        if not mask_pattern[b]:
            # Masked: all cache blocks must be zero
            for blk_idx in range(max_blocks):
                phys = block_offsets[b, blk_idx].item()
                assert kv_cache_fused[phys].abs().max().item() == 0.0, (
                    f"Masked batch {b} block {blk_idx} should be zero"
                )
        else:
            # Unmasked: cache values should match reference
            token_offset = cu_kv_comp[b].item()
            for t in range(n_tokens):
                blk = t // tokens_per_block
                off = t % tokens_per_block
                phys = block_offsets[b, blk].item()
                cache_row = kv_cache_fused[phys, off * head_dim : (off + 1) * head_dim]
                ref_row = ref[token_offset + t]
                max_diff = (cache_row.float() - ref_row.float()).abs().max().item()
                assert max_diff < 0.05, f"Unmasked batch {b} token {t}: max_diff={max_diff}"


def _build_postprocess_reference(
    kv_comp,
    rms_weight,
    rms_eps,
    cos_sin_table,
    position_ids,
    nope_dim,
    rope_dim,
    head_dim,
    rotate_activation=True,
    keep_fp32=False,
):
    """Reference: RMSNorm -> RoPE -> optional Hadamard on kv_comp.

    Stays fp32 between steps and only truncates to ``kv_comp.dtype`` at the
    end -- matches the kernel which keeps activations in fp32 registers
    throughout postprocess (no V4-Pro fake-quant bf16 round-trips).
    """
    x = kv_comp.clone().float()
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + rms_eps)
    x = rms_weight.float() * x
    xn = x[:, :nope_dim]
    xp = x[:, nope_dim:]
    half_rope = rope_dim // 2
    cos_v = cos_sin_table[position_ids.long(), 0, :]
    sin_v = cos_sin_table[position_ids.long(), 1, :]
    xp = xp.view(-1, half_rope, 2)
    x_even, x_odd = xp[..., 0], xp[..., 1]
    xp = torch.stack([x_even * cos_v - x_odd * sin_v, x_odd * cos_v + x_even * sin_v], dim=-1)
    xp = xp.view(-1, rope_dim)
    x = torch.cat([xn, xp], dim=-1)
    if rotate_activation:
        try:
            from fast_hadamard_transform import hadamard_transform
        except ImportError:
            pytest.skip("fast_hadamard_transform not installed")
        x = hadamard_transform(x, scale=head_dim**-0.5)
    if keep_fp32:
        return x
    return x.to(kv_comp.dtype)


def _ceil_pow2_scale(amax: torch.Tensor, max_value_inv: float, min_amax: float) -> torch.Tensor:
    """V4-Pro fast_round_scale: 2^ceil(log2(amax * max_value_inv)).

    Uses fp32 bit manipulation to match V4's fast_log2_ceil + fast_pow2
    byte-for-byte (avoids log2/exp2 fp32 rounding, so the resulting
    power-of-2 is exact even at 2^k boundaries).
    """
    scaled = torch.clamp(amax.float(), min=min_amax) * max_value_inv
    bits = scaled.contiguous().view(torch.int32)
    exp_part = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF).ne(0).to(torch.int32)
    log2_ceil = exp_part + has_mantissa
    pow2_bits = (log2_ceil + 127) << 23
    return pow2_bits.view(torch.float32)


def _e2m1_nibbles_reference(x: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    abs_x = x.abs().clamp(max=6.0)
    idx = torch.zeros_like(abs_x, dtype=torch.uint8)
    for level_idx in range(1, 8):
        take_upper = (abs_x > thresholds[level_idx]) | (
            (abs_x == thresholds[level_idx]) & (level_idx % 2 == 0)
        )
        idx = torch.where(take_upper, torch.full_like(idx, level_idx), idx)
    sign = torch.signbit(x).to(torch.uint8) << 3
    return idx | sign


def _mxfp4_reference(x: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape[-1] % block_size == 0
    packed = torch.empty(*x.shape[:-1], x.shape[-1] // 2, device=x.device, dtype=torch.uint8)
    scales = torch.empty(
        *x.shape[:-1], x.shape[-1] // block_size, device=x.device, dtype=torch.uint8
    )
    fp4_min_amax = 6.0 * torch.finfo(torch.float32).tiny
    for start in range(0, x.shape[-1], block_size):
        end = start + block_size
        chunk = x[:, start:end].float()
        scale = _ceil_pow2_scale(chunk.abs().amax(dim=1, keepdim=True), 1.0 / 6.0, fp4_min_amax)
        q_nibbles = _e2m1_nibbles_reference(chunk / scale)
        packed[:, start // 2 : end // 2] = q_nibbles[:, 0::2] | (q_nibbles[:, 1::2] << 4)
        scale_bits = scale.contiguous().view(torch.int32)
        scales[:, start // block_size] = ((scale_bits[:, 0] >> 23) & 0xFF).to(torch.uint8)
    return packed, scales


def _setup_fused_test_inputs(
    batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block
):
    """Common setup for fused postprocess scatter tests."""
    torch.manual_seed(42)
    device = "cuda"
    total_tokens = batch_size * num_tokens

    kv_comp = torch.randn(total_tokens, head_dim, device=device, dtype=torch.bfloat16) * 0.1
    rms_weight = torch.randn(head_dim, device=device, dtype=torch.bfloat16) * 0.1 + 1.0
    rms_eps = 1e-5
    max_pos = total_tokens + 64
    cos_sin_table = torch.randn(max_pos, 2, rope_dim // 2, device=device, dtype=torch.float32)
    position_ids = torch.arange(total_tokens, device=device, dtype=torch.int32)

    max_comp_len = num_tokens + 4
    max_blocks = (max_comp_len + tokens_per_block - 1) // tokens_per_block
    num_blocks = batch_size * max_blocks

    block_offsets = torch.zeros(batch_size, max_blocks, device=device, dtype=torch.int32)
    for b in range(batch_size):
        block_offsets[b] = torch.arange(b * max_blocks, (b + 1) * max_blocks, dtype=torch.int32)

    num_comp_tokens = torch.full((batch_size,), num_tokens, device=device, dtype=torch.int32)
    cu_kv_comp = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_kv_comp[1:] = num_comp_tokens.cumsum(0)
    start_pos = torch.zeros(batch_size, device=device, dtype=torch.int32)

    return (
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        num_blocks,
        max_blocks,
        block_offsets,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        total_tokens,
    )


@pytest.mark.parametrize("rotate_activation", [True, False], ids=["rotate", "no_rotate"])
@pytest.mark.parametrize(
    "batch_size,num_tokens,head_dim,nope_dim,rope_dim,tokens_per_block",
    [
        pytest.param(1, 16, 128, 64, 64, 32, id="b1_t16_hd128"),
        pytest.param(2, 32, 128, 64, 64, 32, id="b2_t32_hd128"),
        pytest.param(1, 128, 512, 256, 256, 128, id="b1_t128_hd512"),
        pytest.param(2, 64, 512, 256, 256, 128, id="b2_t64_hd512"),
    ],
)
@pytest.mark.skipif(
    not _HAS_POSTPROCESS_SCATTER, reason="Postprocess/scatter CUDA ops not available"
)
def test_fused_postprocess_scatter_fp8_pertensor(
    batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block, rotate_activation
):
    """Test fused RMSNorm+RoPE+(optional Hadamard)+FP8PerTensor scatter vs reference.

    FP8 per-tensor uses scale=1.0 (direct float->fp8_e4m3fn cast).
    Validates fp8 bytes in cache match reference pipeline.
    """
    (
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        num_blocks,
        max_blocks,
        block_offsets,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        total_tokens,
    ) = _setup_fused_test_inputs(
        batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block
    )

    # FP8 per-tensor: cache_stride_blk_bytes = tpb * hd (1 byte per element)
    cache_stride_blk_bytes = tokens_per_block * head_dim
    kv_cache_fused = torch.zeros(
        num_blocks * cache_stride_blk_bytes, device="cuda", dtype=torch.uint8
    )

    # Reference: postprocess (returns bf16), then cast to fp8
    ref = _build_postprocess_reference(
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        head_dim,
        rotate_activation,
    )
    ref_fp8 = ref.float().to(torch.float8_e4m3fn)

    # Fused postprocess + scatter (cache_dtype=fp8, scale_type=fp8_pertensor)
    compressed_mask = torch.ones(total_tokens, dtype=torch.bool, device="cuda")
    torch.ops.trtllm.compressor_postprocess_scatter(
        kv_comp,
        None,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        kv_cache_fused,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        block_offsets,
        compressed_mask,
        tokens_per_block,
        1,
        rotate_activation,
        None,
        None,
    )
    torch.cuda.synchronize()

    # Read back and compare fp8 bytes
    for b in range(batch_size):
        for t in range(num_tokens):
            blk = t // tokens_per_block
            off = t % tokens_per_block
            phys = block_offsets[b, blk].item()
            base = phys * cache_stride_blk_bytes + off * head_dim
            fused_bytes = kv_cache_fused[base : base + head_dim]
            ref_bytes = ref_fp8[b * num_tokens + t].view(torch.uint8)
            max_diff = (fused_bytes.int() - ref_bytes.int()).abs().max().item()
            assert max_diff <= 1, (
                f"FP8 pertensor mismatch at b={b}, t={t}: max byte diff={max_diff}"
            )


@pytest.mark.parametrize("rotate_activation", [True, False], ids=["rotate", "no_rotate"])
@pytest.mark.parametrize(
    "batch_size,num_tokens,head_dim,nope_dim,rope_dim,tokens_per_block",
    [
        pytest.param(1, 16, 128, 64, 64, 32, id="b1_t16_hd128"),
        pytest.param(2, 32, 128, 64, 64, 32, id="b2_t32_hd128"),
        pytest.param(1, 128, 512, 256, 256, 128, id="b1_t128_hd512"),
        pytest.param(2, 64, 512, 256, 256, 128, id="b2_t64_hd512"),
    ],
)
@pytest.mark.skipif(
    not _HAS_POSTPROCESS_SCATTER, reason="Postprocess/scatter CUDA ops not available"
)
def test_fused_postprocess_scatter_fp8_blockwise(
    batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block, rotate_activation
):
    """Test fused RMSNorm+RoPE+(optional Hadamard)+FP8Blockwise scatter vs reference.

    FP8 blockwise uses per-128-element scales. Validates:
    1. FP8 data in cache matches reference quantization
    2. Scales in cache are correct
    3. Optional fp8_output and scale_output buffers are populated
    """
    (
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        num_blocks,
        max_blocks,
        block_offsets,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        total_tokens,
    ) = _setup_fused_test_inputs(
        batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block
    )

    num_scale_blocks = head_dim // 128
    cache_stride_blk_bytes = tokens_per_block * head_dim + tokens_per_block * num_scale_blocks * 4
    kv_cache_fused = torch.zeros(
        num_blocks * cache_stride_blk_bytes, device="cuda", dtype=torch.uint8
    )

    # Optional output buffers
    fp8_output = torch.zeros(total_tokens, head_dim, device="cuda", dtype=torch.uint8)
    scale_output = torch.zeros(total_tokens, num_scale_blocks, device="cuda", dtype=torch.float32)

    # Reference: postprocess (returns bf16) then blockwise quantize
    ref = _build_postprocess_reference(
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        head_dim,
        rotate_activation,
    )

    # Per-128-element quantization reference (from bf16 values)
    ref_fp8_list = []
    ref_scale_list = []
    for token_idx in range(total_tokens):
        row = ref[token_idx]
        fp8_row = torch.zeros(head_dim, dtype=torch.uint8, device="cuda")
        scales = torch.zeros(num_scale_blocks, dtype=torch.float32, device="cuda")
        for s in range(num_scale_blocks):
            chunk = row[s * 128 : (s + 1) * 128].float()
            amax = chunk.abs().max()
            scale = amax / 448.0
            inv_scale = (448.0 / amax) if amax > 0 else 1.0
            quantized = (chunk * inv_scale).to(torch.float8_e4m3fn)
            fp8_row[s * 128 : (s + 1) * 128] = quantized.view(torch.uint8)
            scales[s] = scale
        ref_fp8_list.append(fp8_row)
        ref_scale_list.append(scales)
    ref_fp8_all = torch.stack(ref_fp8_list)
    ref_scale_all = torch.stack(ref_scale_list)

    # Fused postprocess + scatter (cache_dtype=fp8, scale_type=fp8_blockwise)
    compressed_mask = torch.ones(total_tokens, dtype=torch.bool, device="cuda")
    torch.ops.trtllm.compressor_postprocess_scatter(
        kv_comp,
        None,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        kv_cache_fused,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        block_offsets,
        compressed_mask,
        tokens_per_block,
        2,
        rotate_activation,
        fp8_output,
        scale_output,
    )
    torch.cuda.synchronize()

    # Validate optional output buffers
    max_byte_diff = (fp8_output.int() - ref_fp8_all.int()).abs().max().item()
    assert max_byte_diff <= 1, f"fp8_output buffer max byte diff = {max_byte_diff}"

    max_scale_diff = (scale_output - ref_scale_all).abs().max().item()
    assert max_scale_diff < 5e-3, f"scale_output buffer max scale diff = {max_scale_diff}"

    # Validate cache contents
    for b in range(batch_size):
        for t in range(num_tokens):
            blk = t // tokens_per_block
            off = t % tokens_per_block
            phys = block_offsets[b, blk].item()
            block_base = phys * cache_stride_blk_bytes

            # Check FP8 data
            fp8_base = block_base + off * head_dim
            fused_bytes = kv_cache_fused[fp8_base : fp8_base + head_dim]
            ref_bytes = ref_fp8_all[b * num_tokens + t]
            byte_diff = (fused_bytes.int() - ref_bytes.int()).abs().max().item()
            assert byte_diff <= 1, (
                f"FP8 blockwise data mismatch at b={b}, t={t}: max byte diff={byte_diff}"
            )

            # Check scales
            scale_base = block_base + tokens_per_block * head_dim
            for s in range(num_scale_blocks):
                scale_off = scale_base + (off * num_scale_blocks + s) * 4
                fused_scale_bytes = kv_cache_fused[scale_off : scale_off + 4]
                fused_scale = fused_scale_bytes.view(torch.float32).item()
                ref_scale = ref_scale_all[b * num_tokens + t, s].item()
                assert abs(fused_scale - ref_scale) < 5e-3, (
                    f"Scale mismatch at b={b}, t={t}, s={s}: "
                    f"fused={fused_scale:.6f} ref={ref_scale:.6f}"
                )


@pytest.mark.parametrize("rotate_activation", [True, False], ids=["rotate", "no_rotate"])
@pytest.mark.parametrize(
    "batch_size,num_tokens,head_dim,nope_dim,rope_dim,tokens_per_block",
    [
        pytest.param(1, 16, 128, 64, 64, 32, id="b1_t16_indexer"),
        pytest.param(2, 32, 128, 64, 64, 32, id="b2_t32_indexer"),
        pytest.param(1, 64, 512, 256, 256, 64, id="b1_t64_hd512"),
    ],
)
@pytest.mark.skipif(
    not _HAS_POSTPROCESS_SCATTER, reason="Postprocess/scatter CUDA ops not available"
)
def test_fused_postprocess_scatter_mxfp4(
    batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block, rotate_activation
):
    """Optimized indexer mode: packed FP4 cache data plus per-32 UE8M0 scale bytes."""
    (
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        num_blocks,
        _,
        block_offsets,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        total_tokens,
    ) = _setup_fused_test_inputs(
        batch_size, num_tokens, head_dim, nope_dim, rope_dim, tokens_per_block
    )

    packed_head_dim = head_dim // 2
    num_scale_blocks = head_dim // 32
    cache_stride_blk_bytes = (
        tokens_per_block * packed_head_dim + tokens_per_block * num_scale_blocks
    )
    kv_cache_fused = torch.zeros(
        num_blocks * cache_stride_blk_bytes, device="cuda", dtype=torch.uint8
    )
    fp4_output = torch.zeros(total_tokens, packed_head_dim, device="cuda", dtype=torch.uint8)
    scale_output = torch.zeros(total_tokens, num_scale_blocks, device="cuda", dtype=torch.uint8)

    ref = _build_postprocess_reference(
        kv_comp,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        head_dim,
        rotate_activation,
        keep_fp32=True,
    )
    ref_packed, ref_scales = _mxfp4_reference(ref)

    compressed_mask = torch.ones(total_tokens, dtype=torch.bool, device="cuda")
    torch.ops.trtllm.compressor_postprocess_scatter(
        kv_comp,
        None,
        rms_weight,
        rms_eps,
        cos_sin_table,
        position_ids,
        nope_dim,
        rope_dim,
        kv_cache_fused,
        num_comp_tokens,
        cu_kv_comp,
        start_pos,
        block_offsets,
        compressed_mask,
        tokens_per_block,
        3,
        rotate_activation,
        fp4_output,
        scale_output,
    )
    torch.cuda.synchronize()

    assert torch.equal(fp4_output, ref_packed), "packed FP4 output bytes must match reference"
    assert torch.equal(scale_output, ref_scales), "UE8M0 scale bytes must match reference"

    for b in range(batch_size):
        for t in range(num_tokens):
            blk = t // tokens_per_block
            off = t % tokens_per_block
            phys = block_offsets[b, blk].item()
            block_base = phys * cache_stride_blk_bytes
            fp4_base = block_base + off * packed_head_dim
            scale_base = block_base + tokens_per_block * packed_head_dim + off * num_scale_blocks
            token_idx = b * num_tokens + t
            assert torch.equal(
                kv_cache_fused[fp4_base : fp4_base + packed_head_dim],
                ref_packed[token_idx],
            )
            assert torch.equal(
                kv_cache_fused[scale_base : scale_base + num_scale_blocks],
                ref_scales[token_idx],
            )


# ============================================================================
# Benchmarks: cuTile Kernels vs PyTorch Reference
# ============================================================================


def benchmark_compress_kernel():
    """Benchmark CUDA vs PyTorch compress (decode) kernels using triton.testing.do_bench."""

    print("\n" + "=" * 70)
    print("Compress Kernel Benchmark: CUDA vs PyTorch (decode)")
    print("=" * 70)

    configs = [
        # (batch_size, compress_ratio, head_dim, overlap, page_size, name)
        (1, 4, 512, True, 32, "b1_r4_d512_overlap"),
        (8, 4, 512, True, 32, "b8_r4_d512_overlap"),
        (32, 4, 512, True, 32, "b32_r4_d512_overlap"),
        (1, 128, 512, False, 32, "b1_r128_d512"),
        (8, 128, 512, False, 32, "b8_r128_d512"),
        (32, 128, 512, False, 32, "b32_r128_d512"),
        (1, 4, 128, True, 8, "b1_r4_d128_overlap"),
        (8, 4, 128, True, 8, "b8_r4_d128_overlap"),
        (32, 4, 128, True, 8, "b32_r4_d128_overlap"),
    ]

    results = []
    for batch_size, compress_ratio, head_dim, overlap, page_size, name in configs:
        coff = 2 if overlap else 1
        state_dim = coff * head_dim
        max_blocks = (compress_ratio * 2 + page_size - 1) // page_size

        # Prepare inputs
        ape = torch.randn(compress_ratio, state_dim, device="cuda")
        new_kv = torch.randn(batch_size, state_dim, device="cuda")
        new_score = torch.randn(batch_size, state_dim, device="cuda")
        kv_score = fuse_kv_score(new_kv, new_score)

        # PyTorch state
        state_len = coff * compress_ratio
        kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
        score_state_py = torch.full(
            (batch_size, state_len, state_dim), float("-inf"), device="cuda"
        )

        # Paged cache
        num_blocks = batch_size * max_blocks
        paged_kv = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
        paged_score = torch.zeros(num_blocks, page_size, state_dim, device="cuda")
        block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
            batch_size, max_blocks
        )

        # Use step = compress_ratio - 1 to trigger compression.
        step = compress_ratio - 1
        token_idx = (compress_ratio + step) if overlap else step
        kv_lens = torch.full((batch_size,), token_idx + 1, device="cuda", dtype=torch.int32)
        start_pos = torch.full((batch_size,), token_idx, device="cuda", dtype=torch.int32)
        cu_seq_lens, cu_outputs = prepare_decode_metadata(
            batch_size, compress_ratio, head_dim, torch.device("cuda"), next_n=1
        )
        kv_comp = prepare_compress_output(
            cu_outputs, batch_size, head_dim, torch.device("cuda"), torch.bfloat16
        )

        paged_kv_cuda = paged_kv.clone()
        paged_score_cuda = paged_score.clone()
        kv_comp_cuda = kv_comp.clone()

        def cuda_fn():
            decode_kernel(
                kv_score,
                ape,
                kv_lens,
                start_pos,
                cu_seq_lens,
                cu_outputs,
                kv_comp_cuda,
                paged_kv_cuda,
                paged_score_cuda,
                block_table,
                block_table,
                compress_ratio,
                head_dim,
                page_size,
                next_n=1,
            )

        def pytorch_fn():
            run_pytorch_reference(
                new_kv.unsqueeze(1),
                new_score.unsqueeze(1),
                ape,
                kv_state_py.clone(),
                score_state_py.clone(),
                step,
                compress_ratio,
                head_dim,
                overlap,
            )

        cuda_ms = triton.testing.do_bench(cuda_fn, warmup=25, rep=100)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=25, rep=100)

        cuda_us = cuda_ms * 1000
        pytorch_us = pytorch_ms * 1000

        results.append(
            {
                "name": name,
                "cuda_us": cuda_us,
                "pytorch_us": pytorch_us,
                "speedup": pytorch_us / cuda_us if cuda_us > 0 else float("inf"),
            }
        )

    # Print results
    print(f"\n{'Config':<30} {'CUDA (us)':>12} {'PyTorch (us)':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<30} {r['cuda_us']:>12.2f} {r['pytorch_us']:>12.2f} {r['speedup']:>9.2f}x"
        )
    print("=" * 70)

    return results


def benchmark_compress_prefill_kernel():
    """Benchmark CUDA vs PyTorch compress (prefill) kernels using triton.testing.do_bench."""

    print("\n" + "=" * 70)
    print("Compress Prefill Kernel Benchmark: CUDA vs PyTorch")
    print("=" * 70)

    configs = [
        # (batch_size, seqlen, compress_ratio, head_dim, overlap, page_size, name)
        (1, 128, 4, 512, True, 32, "prefill_b1_s128_r4_d512"),
        (4, 256, 4, 512, True, 32, "prefill_b4_s256_r4_d512"),
        (8, 512, 4, 512, True, 32, "prefill_b8_s512_r4_d512"),
        (1, 512, 128, 512, False, 32, "prefill_b1_s512_r128_d512"),
        (4, 512, 128, 512, False, 32, "prefill_b4_s512_r128_d512"),
        (8, 1024, 128, 512, False, 32, "prefill_b8_s1024_r128_d512"),
        (1, 128, 4, 128, True, 8, "prefill_b1_s128_r4_d128"),
        (8, 512, 4, 128, True, 8, "prefill_b8_s512_r4_d128"),
        (32, 512, 4, 128, True, 8, "prefill_b32_s512_r4_d128"),
    ]

    results = []
    for batch_size, seqlen, compress_ratio, head_dim, overlap, page_size, name in configs:
        coff = 2 if overlap else 1
        state_dim = coff * head_dim

        # Prepare inputs
        kv = torch.randn(batch_size, seqlen, state_dim, device="cuda")
        score = torch.randn(batch_size, seqlen, state_dim, device="cuda")
        ape = torch.randn(compress_ratio, state_dim, device="cuda")

        # PyTorch state
        state_len = coff * compress_ratio
        kv_state_py = torch.zeros(batch_size, state_len, state_dim, device="cuda")
        score_state_py = torch.full(
            (batch_size, state_len, state_dim), float("-inf"), device="cuda"
        )

        # Shared inputs
        kv_score = fuse_kv_score(kv.view(-1, state_dim), score.view(-1, state_dim))
        kv_lens = torch.full((batch_size,), seqlen, device="cuda", dtype=torch.int32)
        start_pos = torch.zeros(batch_size, device="cuda", dtype=torch.int32)
        cu_seq_lens, cu_outputs = prepare_prefill_metadata(
            kv_lens, start_pos, compress_ratio, head_dim, kv_score.device
        )
        kv_comp = prepare_compress_output(
            cu_outputs, batch_size, head_dim, kv_score.device, torch.bfloat16
        )
        paged_kv, paged_score, block_table, _, _ = create_paged_cache(
            batch_size, seqlen, compress_ratio, head_dim, overlap, page_size
        )

        paged_kv_kernel = paged_kv.clone()
        paged_score_kernel = paged_score.clone()
        kv_comp_kernel = kv_comp.clone()

        def kernel_fn():
            seq_lens = kv_lens - start_pos
            num_outputs_per_batch = torch.clamp(seq_lens // compress_ratio, min=1)
            max_outputs = num_outputs_per_batch.max().item()
            prefill_kernel(
                kv_score,
                ape,
                kv_lens,
                start_pos,
                cu_seq_lens,
                cu_outputs,
                kv_comp_kernel,
                paged_kv_kernel,
                paged_score_kernel,
                block_table,
                max_outputs,
                block_table,
                compress_ratio,
                head_dim,
                page_size,
            )

        def pytorch_fn():
            run_pytorch_prefill_reference(
                kv.clone(),
                score.clone(),
                ape,
                kv_state_py.clone(),
                score_state_py.clone(),
                compress_ratio,
                head_dim,
                overlap,
            )

        kernel_ms = triton.testing.do_bench(kernel_fn, warmup=25, rep=100)
        pytorch_ms = triton.testing.do_bench(pytorch_fn, warmup=25, rep=100)

        kernel_us = kernel_ms * 1000
        pytorch_us = pytorch_ms * 1000

        results.append(
            {
                "name": name,
                "kernel_us": kernel_us,
                "pytorch_us": pytorch_us,
                "speedup": pytorch_us / kernel_us if kernel_us > 0 else float("inf"),
            }
        )

    # Print results
    print(f"\n{'Config':<30} {'CUDA (us)':>12} {'PyTorch (us)':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<30} {r['kernel_us']:>12.2f} "
            f"{r['pytorch_us']:>12.2f} {r['speedup']:>9.2f}x"
        )
    print("=" * 70)

    return results


def run_all_benchmarks():
    """Run all kernel benchmarks."""
    print("\n" + "=" * 80)
    print("Compressor Kernel Benchmarks")
    print("=" * 80)

    benchmark_compress_kernel()
    benchmark_compress_prefill_kernel()


if __name__ == "__main__":
    run_all_benchmarks()
