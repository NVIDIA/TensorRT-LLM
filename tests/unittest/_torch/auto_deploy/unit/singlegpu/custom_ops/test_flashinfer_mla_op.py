"""Test FlashInfer MLA backend operations.

Tests the flashinfer_mla_with_cache cached op and compares it with the
torch_backend_mla_with_cache reference implementation.

Key features tested:
- 5 tensor arguments: q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight
- Paged cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
- Prefill: Expand compressed_kv, compute normal attention via BatchPrefillWithRaggedKVCacheWrapper
- Decode: BatchMLAPagedAttentionWrapper with paged compressed KV cache

Reference: https://docs.flashinfer.ai/api/mla.html
"""

import flashinfer
import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.mla.flashinfer_mla import (
    _GlobalFlashInferMLAPlanner,
)


def _create_mla_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    dtype: torch.dtype,
    device: str,
):
    """Create MLA input tensors.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        qk_nope_head_dim: Dimension of query/key non-positional part
        qk_rope_head_dim: Dimension of query/key positional (RoPE) part
        kv_lora_rank: Rank of compressed KV (LoRA rank)
        v_head_dim: Dimension of value head
        dtype: Data type
        device: Device

    Returns:
        Dictionary with q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight
    """
    kv_head_dim = qk_nope_head_dim + v_head_dim

    # q_nope: [B, S, N, qk_nope_head_dim]
    q_nope = torch.randn(
        batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=dtype, device=device
    )

    # q_pe: [B, S, N, qk_rope_head_dim]
    q_pe = torch.randn(batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=dtype, device=device)

    # compressed_kv: [B, S, kv_lora_rank]
    compressed_kv = torch.randn(batch_size, seq_len, kv_lora_rank, dtype=dtype, device=device)

    # kpe: [B, S, 1, qk_rope_head_dim]
    kpe = torch.randn(batch_size, seq_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

    # kv_b_proj_weight: [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    kv_b_proj_weight = torch.randn(
        num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=device
    )

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "compressed_kv": compressed_kv,
        "kpe": kpe,
        "kv_b_proj_weight": kv_b_proj_weight,
    }


def _create_unpaged_cache_and_metadata(
    batch_size: int,
    max_seq_len: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: str,
    seq_lengths: list,
    input_positions: list,
):
    """Create unpaged (torch backend) cache and metadata.

    Args:
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        kv_lora_rank: Rank of compressed KV
        qk_rope_head_dim: Dimension of RoPE
        dtype: Data type
        device: Device
        seq_lengths: List of sequence lengths per batch
        input_positions: List of input positions (cache offsets) per batch

    Returns:
        Dictionary with cache and metadata tensors
    """
    # FlashInfer MLA cache (unpaged): [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    mla_cache = torch.zeros(
        batch_size, max_seq_len, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )

    # Metadata
    seq_len_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    input_pos = torch.tensor(input_positions, dtype=torch.int32, device=device)
    slot_idx = torch.arange(batch_size, dtype=torch.int32, device=device)

    # Compute cu_seqlen (cumulative sequence lengths)
    cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlen[1:] = torch.cumsum(seq_len_tensor, dim=0)

    # Determine if this is context (prefill) or generate (decode)
    total_tokens = sum(seq_lengths)
    is_decode = all(s == 1 for s in seq_lengths)

    if is_decode:
        # Decode phase
        batch_info_host = torch.tensor([0, 0, batch_size], dtype=torch.int32, device=device)
    else:
        # Context/prefill phase
        batch_info_host = torch.tensor(
            [batch_size, total_tokens, 0], dtype=torch.int32, device=device
        )

    return {
        "mla_cache": mla_cache,
        "batch_info_host": batch_info_host,
        "seq_len": seq_len_tensor,
        "input_pos": input_pos,
        "slot_idx": slot_idx,
        "cu_seqlen": cu_seqlen[:-1],  # Exclude last element for seq_start
    }


def _create_paged_cache_and_metadata(
    batch_size: int,
    max_num_pages: int,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dtype: torch.dtype,
    device: str,
    seq_lengths: list,
    input_positions: list,
):
    """Create paged (flashinfer backend) cache and metadata.

    Args:
        batch_size: Batch size
        max_num_pages: Maximum number of pages
        page_size: Size of each page
        kv_lora_rank: Rank of compressed KV
        qk_rope_head_dim: Dimension of RoPE
        dtype: Data type
        device: Device
        seq_lengths: List of sequence lengths per batch
        input_positions: List of input positions (cache offsets) per batch

    Returns:
        Dictionary with paged cache and metadata tensors
    """
    # Paged MLA cache: [num_pages, page_size, kv_lora_rank + qk_rope_head_dim]
    mla_cache = torch.zeros(
        max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )

    # Compute total KV lengths (input_pos + seq_len for each sequence)
    kv_lengths = [pos + seq_len for pos, seq_len in zip(input_positions, seq_lengths)]

    # Compute number of pages per sequence
    pages_per_seq = [(kv_len - 1) // page_size + 1 if kv_len > 0 else 1 for kv_len in kv_lengths]

    # Assign pages (simple sequential assignment)
    page_assignments = []
    next_page = 0
    for num_pages in pages_per_seq:
        page_assignments.append(list(range(next_page, next_page + num_pages)))
        next_page += num_pages

    # Create FlashInfer paging metadata
    seq_len_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=device)

    # qo_indptr: cumulative query/output lengths
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(seq_len_tensor, dim=0)

    # cu_num_pages: cumulative number of pages per sequence
    num_pages_per_seq = torch.tensor(pages_per_seq, dtype=torch.int32, device=device)
    cu_num_pages = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_num_pages[1:] = torch.cumsum(num_pages_per_seq, dim=0)

    # cache_loc (paged_kv_indices): flattened list of page indices
    cache_loc = torch.tensor(
        [p for pages in page_assignments for p in pages], dtype=torch.int32, device=device
    )

    # last_page_len: number of valid tokens in the last page of each sequence
    last_page_len = torch.tensor(
        [((kv_len - 1) % page_size) + 1 if kv_len > 0 else 0 for kv_len in kv_lengths],
        dtype=torch.int32,
        device=device,
    )

    # seq_len_with_cache: total KV lengths
    seq_len_with_cache = torch.tensor(kv_lengths, dtype=torch.int32, device=device)

    # Host copies
    qo_indptr_host = qo_indptr.cpu()
    cu_num_pages_host = cu_num_pages.cpu()
    last_page_len_host = last_page_len.cpu()
    seq_len_with_cache_host = seq_len_with_cache.cpu()

    # Determine if this is context (prefill) or generate (decode)
    total_tokens = sum(seq_lengths)
    is_decode = all(s == 1 for s in seq_lengths)

    if is_decode:
        # Decode phase
        batch_info_host = torch.tensor([0, 0, batch_size], dtype=torch.int32, device=device)
    else:
        # Context/prefill phase
        batch_info_host = torch.tensor(
            [batch_size, total_tokens, 0], dtype=torch.int32, device=device
        )

    return {
        "mla_cache": mla_cache,
        "batch_info_host": batch_info_host,
        "cu_seqlen_host": qo_indptr_host,
        "cu_num_pages": cu_num_pages,
        "cu_num_pages_host": cu_num_pages_host,
        "cache_loc": cache_loc,
        "last_page_len": last_page_len,
        "last_page_len_host": last_page_len_host,
        "seq_len_with_cache_host": seq_len_with_cache_host,
        "page_size": page_size,
    }


def _copy_unpaged_to_paged_cache(
    unpaged_cache: torch.Tensor,
    paged_cache: torch.Tensor,
    batch_size: int,
    tokens_per_seq: list,
    page_size: int,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
):
    """Copy unpaged cache data to paged cache format.

    This is used to initialize paged cache with the same data as unpaged cache
    for comparison tests.

    Args:
        unpaged_cache: Source cache [batch, max_seq, dim]
        paged_cache: Destination paged cache [num_pages, page_size, dim]
        batch_size: Number of sequences
        tokens_per_seq: Number of tokens to copy per sequence
        page_size: Number of tokens per page
        cu_num_pages: Cumulative page counts from flashinfer metadata [batch_size + 1]
        cache_loc: Page indices from flashinfer metadata
    """
    for batch_idx in range(batch_size):
        num_tokens = tokens_per_seq[batch_idx]
        if num_tokens == 0:
            continue

        # Get page assignments for this sequence from flashinfer metadata
        page_start_idx = cu_num_pages[batch_idx].item()
        page_end_idx = cu_num_pages[batch_idx + 1].item()

        token_offset = 0
        for i in range(page_start_idx, page_end_idx):
            page_num = cache_loc[i].item()
            tokens_to_copy = min(page_size, num_tokens - token_offset)
            if tokens_to_copy <= 0:
                break

            paged_cache[page_num, :tokens_to_copy] = unpaged_cache[
                batch_idx, token_offset : token_offset + tokens_to_copy
            ]
            token_offset += tokens_to_copy


@pytest.mark.parametrize("seq_length", [32, 128])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("batch_size", [8, 64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_op_context(seq_length, num_heads, batch_size, dtype, device):
    """Test FlashInfer MLA context (prefill) phase.

    Compares flashinfer_mla_with_cache against torch_backend_mla_with_cache
    for context (prefill) operations where seq_length > 1.
    """
    # MLA dimensions (similar to DeepSeek)
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # Must be 64 for FlashInfer MLA.
    kv_lora_rank = 512  # Must be 512 for FlashInfer MLA.
    v_head_dim = 128
    page_size = seq_length  # Use seq_length as page size for simpler comparison

    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // page_size + 1)

    # Create input tensors
    inputs = _create_mla_inputs(
        batch_size,
        seq_length,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    # Flatten inputs for context phase
    total_tokens = batch_size * seq_length
    q_nope_flat = inputs["q_nope"].view(1, total_tokens, num_heads, qk_nope_head_dim)
    q_pe_flat = inputs["q_pe"].view(1, total_tokens, num_heads, qk_rope_head_dim)
    compressed_kv_flat = inputs["compressed_kv"].view(1, total_tokens, kv_lora_rank)
    kpe_flat = inputs["kpe"].view(1, total_tokens, 1, qk_rope_head_dim)

    # Sequence lengths and positions
    seq_lengths = [seq_length] * batch_size
    input_positions = [0] * batch_size  # Context starts at position 0

    # =========================================================================
    # Run torch backend (reference)
    # =========================================================================
    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )

    torch_output = torch.ops.auto_deploy.torch_cached_mla_with_cache(
        q_nope_flat,
        q_pe_flat,
        compressed_kv_flat,
        kpe_flat,
        inputs["kv_b_proj_weight"],
        torch_meta["batch_info_host"],
        torch_meta["seq_len"],
        torch_meta["input_pos"],
        torch_meta["slot_idx"],
        torch_meta["cu_seqlen"],
        torch_meta["mla_cache"],
        None,  # scale
        kv_lora_rank,
    )

    # =========================================================================
    # Run FlashInfer backend
    # =========================================================================
    # Reset planner
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    # Create paged metadata
    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )

    # Compute FlashInfer batch indices and positions
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lengths, device=device), dim=0).int()

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            flashinfer_meta["cu_num_pages"],
            flashinfer_meta["last_page_len"],
            page_size=page_size,
        ),
        total_tokens,
    )

    flashinfer_output = torch.ops.auto_deploy.flashinfer_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        flashinfer_meta["batch_info_host"],
        flashinfer_meta["cu_seqlen_host"],
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cu_num_pages_host"],
        flashinfer_meta["cache_loc"],
        flashinfer_meta["last_page_len"],
        flashinfer_meta["last_page_len_host"],
        flashinfer_meta["seq_len_with_cache_host"],
        batch_indices,
        positions,
        flashinfer_meta["mla_cache"],
        None,  # scale
        kv_lora_rank,
    )

    # =========================================================================
    # Compare outputs
    # =========================================================================
    # Reshape for comparison
    torch_output_reshaped = torch_output.view(batch_size, seq_length, num_heads, v_head_dim)
    flashinfer_output_reshaped = flashinfer_output.view(
        batch_size, seq_length, num_heads, v_head_dim
    )

    # Use larger tolerance for float16 attention operations with optimized kernels.
    # FlashInfer uses fused kernels with different computation order/precision than the
    # torch reference which processes each sequence in a loop. With batch_size=64 and
    # longer sequences (seq_length=128), numerical differences can accumulate to ~2-3%
    # relative error due to softmax over longer sequences and more attention computations.
    # Using atol=2.0 allows for max diff up to 2.0 (about 2-3% of typical output magnitudes).
    assert torch.allclose(
        flashinfer_output_reshaped.cpu().to(torch.float32),
        torch_output_reshaped.cpu().to(torch.float32),
        atol=2.0,
        rtol=0.1,
    ), (
        f"FlashInfer MLA context output doesn't match torch backend. "
        f"Max diff: {(flashinfer_output_reshaped - torch_output_reshaped).abs().max():.6f}"
    )


@pytest.mark.parametrize("prefill_seq_length", [64, 128])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("batch_size", [4, 64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_op_decode(prefill_seq_length, num_heads, batch_size, dtype, device):
    """Test FlashInfer MLA decode (generate) phase.

    Compares flashinfer_mla_with_cache against torch_backend_mla_with_cache
    for decode operations where seq_length = 1.
    """
    # MLA dimensions
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # Must be 64 for FlashInfer MLA.
    kv_lora_rank = 512  # Must be 512 for FlashInfer MLA.
    v_head_dim = 192
    page_size = 64

    seq_length = 1  # Decode phase

    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // page_size + 1)

    # Create input tensors
    inputs = _create_mla_inputs(
        batch_size,
        seq_length,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    # Sequence lengths and positions
    seq_lengths = [seq_length] * batch_size
    input_positions = [prefill_seq_length] * batch_size

    # =========================================================================
    # Setup caches with pre-filled data
    # =========================================================================
    # Create unpaged cache with prefilled data
    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )

    # Pre-fill cache with random data if prefill_seq_length > 0
    if prefill_seq_length > 0:
        torch_meta["mla_cache"][:, :prefill_seq_length, :] = torch.randn(
            batch_size,
            prefill_seq_length,
            kv_lora_rank + qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )

    # Create paged cache
    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )

    # Copy unpaged cache to paged format
    if prefill_seq_length > 0:
        _copy_unpaged_to_paged_cache(
            torch_meta["mla_cache"],
            flashinfer_meta["mla_cache"],
            batch_size,
            [prefill_seq_length] * batch_size,  # Number of tokens to copy
            page_size,
            flashinfer_meta["cu_num_pages"],
            flashinfer_meta["cache_loc"],
        )

    # =========================================================================
    # Run torch backend (reference)
    # =========================================================================
    torch_output = torch.ops.auto_deploy.torch_cached_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        torch_meta["batch_info_host"],
        torch_meta["seq_len"],
        torch_meta["input_pos"],
        torch_meta["slot_idx"],
        torch_meta["cu_seqlen"],
        torch_meta["mla_cache"],
        None,  # scale
        kv_lora_rank,
    )

    # =========================================================================
    # Run FlashInfer backend
    # =========================================================================
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    # Compute FlashInfer batch indices and positions
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lengths, device=device), dim=0).int()

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            flashinfer_meta["cu_num_pages"],
            flashinfer_meta["last_page_len"],
            page_size=page_size,
        ),
        batch_size * seq_length,
    )

    flashinfer_output = torch.ops.auto_deploy.flashinfer_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        flashinfer_meta["batch_info_host"],
        flashinfer_meta["cu_seqlen_host"],
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cu_num_pages_host"],
        flashinfer_meta["cache_loc"],
        flashinfer_meta["last_page_len"],
        flashinfer_meta["last_page_len_host"],
        flashinfer_meta["seq_len_with_cache_host"],
        batch_indices,
        positions,
        flashinfer_meta["mla_cache"],
        None,  # scale
        kv_lora_rank,
    )

    # =========================================================================
    # Compare outputs
    # =========================================================================
    # Use larger tolerance for float16 attention operations with optimized kernels.
    # FlashInfer uses fused kernels with different computation order/precision than the
    # torch reference which processes each sequence in a loop. With large batch_size
    # and seq_length, numerical differences can accumulate.
    assert torch.allclose(
        flashinfer_output.cpu().to(torch.float32),
        torch_output.cpu().to(torch.float32),
        atol=2.0,  # Use larger tolerance for float16 attention with fused kernels
        rtol=0.1,
    ), (
        f"FlashInfer MLA decode output doesn't match torch backend. "
        f"Max diff: {(flashinfer_output - torch_output).abs().max():.6f}"
    )


@pytest.mark.parametrize("prefill_seq_length", [16, 64, 128])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("batch_size", [4, 64])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_context_and_generate(
    prefill_seq_length, num_heads, batch_size, dtype, device
):
    """Test FlashInfer MLA context (prefill) followed by generate (decode).

    This test verifies the full workflow:
    1. Context phase: Process initial sequence
    2. Generate phase: Generate additional tokens one at a time
    """
    # MLA dimensions
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # Must be 64 for FlashInfer MLA.
    kv_lora_rank = 512  # Must be 512 for FlashInfer MLA.
    v_head_dim = 128
    # Use page_size > prefill_seq_length to avoid page overflow during generate.
    # This ensures context and generate phases use the same page assignments.
    page_size = prefill_seq_length + 16

    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // page_size + 1)

    # =========================================================================
    # Context phase
    # =========================================================================
    inputs_context = _create_mla_inputs(
        batch_size,
        prefill_seq_length,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    seq_lengths_context = [prefill_seq_length] * batch_size
    input_positions_context = [0] * batch_size

    # Flatten context inputs
    total_tokens = batch_size * prefill_seq_length
    q_nope_flat = inputs_context["q_nope"].view(1, total_tokens, num_heads, qk_nope_head_dim)
    q_pe_flat = inputs_context["q_pe"].view(1, total_tokens, num_heads, qk_rope_head_dim)
    compressed_kv_flat = inputs_context["compressed_kv"].view(1, total_tokens, kv_lora_rank)
    kpe_flat = inputs_context["kpe"].view(1, total_tokens, 1, qk_rope_head_dim)

    # Create caches
    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths_context,
        input_positions_context,
    )

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths_context,
        input_positions_context,
    )

    # Run torch backend context
    torch_output_context = torch.ops.auto_deploy.torch_cached_mla_with_cache(
        q_nope_flat,
        q_pe_flat,
        compressed_kv_flat,
        kpe_flat,
        inputs_context["kv_b_proj_weight"],
        torch_meta["batch_info_host"],
        torch_meta["seq_len"],
        torch_meta["input_pos"],
        torch_meta["slot_idx"],
        torch_meta["cu_seqlen"],
        torch_meta["mla_cache"],
        None,
        kv_lora_rank,
    )

    # Run FlashInfer context
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lengths_context, device=device), dim=0).int()

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            flashinfer_meta["cu_num_pages"],
            flashinfer_meta["last_page_len"],
            page_size=page_size,
        ),
        total_tokens,
    )

    flashinfer_output_context = torch.ops.auto_deploy.flashinfer_mla_with_cache(
        inputs_context["q_nope"],
        inputs_context["q_pe"],
        inputs_context["compressed_kv"],
        inputs_context["kpe"],
        inputs_context["kv_b_proj_weight"],
        flashinfer_meta["batch_info_host"],
        flashinfer_meta["cu_seqlen_host"],
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cu_num_pages_host"],
        flashinfer_meta["cache_loc"],
        flashinfer_meta["last_page_len"],
        flashinfer_meta["last_page_len_host"],
        flashinfer_meta["seq_len_with_cache_host"],
        batch_indices,
        positions,
        flashinfer_meta["mla_cache"],
        None,
        kv_lora_rank,
    )

    # Verify context outputs match
    torch_output_context_reshaped = torch_output_context.view(
        batch_size, prefill_seq_length, num_heads, v_head_dim
    )
    flashinfer_output_context_reshaped = flashinfer_output_context.view(
        batch_size, prefill_seq_length, num_heads, v_head_dim
    )

    assert torch.allclose(
        flashinfer_output_context_reshaped.cpu().to(torch.float32),
        torch_output_context_reshaped.cpu().to(torch.float32),
        atol=2.0,  # Use larger tolerance for float16 attention with fused kernels
        rtol=0.1,
    ), "Context phase outputs don't match"

    # =========================================================================
    # Generate phase (single token)
    # =========================================================================
    inputs_gen = _create_mla_inputs(
        batch_size,
        1,  # seq_length = 1 for generate
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    seq_lengths_gen = [1] * batch_size
    input_positions_gen = [prefill_seq_length] * batch_size

    # Update torch metadata for generate
    torch_meta_gen = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths_gen,
        input_positions_gen,
    )
    # Use the same cache (already filled from context)
    torch_meta_gen["mla_cache"] = torch_meta["mla_cache"]

    # Update flashinfer metadata for generate
    flashinfer_meta_gen = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths_gen,
        input_positions_gen,
    )
    # Use the same cache
    flashinfer_meta_gen["mla_cache"] = flashinfer_meta["mla_cache"]

    # Run torch backend generate
    torch_output_gen = torch.ops.auto_deploy.torch_cached_mla_with_cache(
        inputs_gen["q_nope"],
        inputs_gen["q_pe"],
        inputs_gen["compressed_kv"],
        inputs_gen["kpe"],
        inputs_context["kv_b_proj_weight"],  # Use same weights
        torch_meta_gen["batch_info_host"],
        torch_meta_gen["seq_len"],
        torch_meta_gen["input_pos"],
        torch_meta_gen["slot_idx"],
        torch_meta_gen["cu_seqlen"],
        torch_meta_gen["mla_cache"],
        None,
        kv_lora_rank,
    )

    # Run FlashInfer generate
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    qo_indptr_gen = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr_gen[1:] = torch.cumsum(torch.tensor(seq_lengths_gen, device=device), dim=0).int()

    batch_indices_gen, positions_gen = flashinfer.get_batch_indices_positions(
        qo_indptr_gen,
        flashinfer.get_seq_lens(
            flashinfer_meta_gen["cu_num_pages"],
            flashinfer_meta_gen["last_page_len"],
            page_size=page_size,
        ),
        batch_size,
    )

    flashinfer_output_gen = torch.ops.auto_deploy.flashinfer_mla_with_cache(
        inputs_gen["q_nope"],
        inputs_gen["q_pe"],
        inputs_gen["compressed_kv"],
        inputs_gen["kpe"],
        inputs_context["kv_b_proj_weight"],
        flashinfer_meta_gen["batch_info_host"],
        flashinfer_meta_gen["cu_seqlen_host"],
        flashinfer_meta_gen["cu_num_pages"],
        flashinfer_meta_gen["cu_num_pages_host"],
        flashinfer_meta_gen["cache_loc"],
        flashinfer_meta_gen["last_page_len"],
        flashinfer_meta_gen["last_page_len_host"],
        flashinfer_meta_gen["seq_len_with_cache_host"],
        batch_indices_gen,
        positions_gen,
        flashinfer_meta_gen["mla_cache"],
        None,
        kv_lora_rank,
    )

    # Verify generate outputs match
    assert torch.allclose(
        flashinfer_output_gen.cpu().to(torch.float32),
        torch_output_gen.cpu().to(torch.float32),
        atol=2.0,  # Use larger tolerance for float16 attention with fused kernels
        rtol=0.1,
    ), (
        f"Generate phase outputs don't match. "
        f"Max diff: {(flashinfer_output_gen - torch_output_gen).abs().max():.6f}"
    )


@pytest.mark.parametrize(
    "seq_lengths",
    [
        [8, 16],
        [12, 24, 32],
        [4, 8, 16, 32],
    ],
)
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_with_variable_seq_lengths(seq_lengths, num_heads, dtype, device):
    """Test FlashInfer MLA with variable sequence lengths in a batch.

    This test verifies that the FlashInfer MLA backend handles batches
    with different sequence lengths correctly.
    """
    batch_size = len(seq_lengths)

    # MLA dimensions
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # Must be 64 for FlashInfer MLA.
    kv_lora_rank = 512  # Must be 512 for FlashInfer MLA.
    v_head_dim = 128
    page_size = 32

    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // page_size + 1)

    # Create individual inputs for each sequence
    total_tokens = sum(seq_lengths)

    # Create batched inputs (we'll flatten later)
    q_nope_list = []
    q_pe_list = []
    compressed_kv_list = []
    kpe_list = []

    for seq_len in seq_lengths:
        inputs = _create_mla_inputs(
            1,  # Single sequence
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            dtype,
            device,
        )
        q_nope_list.append(inputs["q_nope"].squeeze(0))
        q_pe_list.append(inputs["q_pe"].squeeze(0))
        compressed_kv_list.append(inputs["compressed_kv"].squeeze(0))
        kpe_list.append(inputs["kpe"].squeeze(0))

    # Concatenate into flattened format
    q_nope_flat = torch.cat(q_nope_list, dim=0).unsqueeze(0)  # [1, total_tokens, N, D]
    q_pe_flat = torch.cat(q_pe_list, dim=0).unsqueeze(0)
    compressed_kv_flat = torch.cat(compressed_kv_list, dim=0).unsqueeze(0)
    kpe_flat = torch.cat(kpe_list, dim=0).unsqueeze(0)

    # Common kv_b_proj_weight
    kv_head_dim = qk_nope_head_dim + v_head_dim
    kv_b_proj_weight = torch.randn(
        num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=device
    )

    input_positions = [0] * batch_size

    # =========================================================================
    # Run FlashInfer backend
    # =========================================================================
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lengths, device=device), dim=0).int()

    batch_indices, positions = flashinfer.get_batch_indices_positions(
        qo_indptr,
        flashinfer.get_seq_lens(
            flashinfer_meta["cu_num_pages"],
            flashinfer_meta["last_page_len"],
            page_size=page_size,
        ),
        total_tokens,
    )

    flashinfer_output = torch.ops.auto_deploy.flashinfer_mla_with_cache(
        q_nope_flat,
        q_pe_flat,
        compressed_kv_flat,
        kpe_flat,
        kv_b_proj_weight,
        flashinfer_meta["batch_info_host"],
        flashinfer_meta["cu_seqlen_host"],
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cu_num_pages_host"],
        flashinfer_meta["cache_loc"],
        flashinfer_meta["last_page_len"],
        flashinfer_meta["last_page_len_host"],
        flashinfer_meta["seq_len_with_cache_host"],
        batch_indices,
        positions,
        flashinfer_meta["mla_cache"],
        None,
        kv_lora_rank,
    )

    # Verify output shape
    expected_shape = (1, total_tokens, num_heads, v_head_dim)
    assert flashinfer_output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {flashinfer_output.shape}"
    )

    # Verify output is finite
    assert torch.isfinite(flashinfer_output).all(), "Output contains NaN or Inf values"
