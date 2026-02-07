"""Test FlashInfer MLA backend operations.

Tests the flashinfer_mla_with_cache cached op and compares it with the
torch_backend_mla_with_cache reference implementation.

Key features tested:
- 5 tensor arguments: q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight
- Paged caches: ckv_cache [num_pages, page_size, kv_lora_rank] and kpe_cache [num_pages, page_size, qk_rope_head_dim]
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
from tensorrt_llm._torch.auto_deploy.utils.cuda_graph import CudaGraphWarmUpPhase


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

    # Scale factor for Xavier-like initialization to keep values bounded
    # This helps reduce numerical differences by keeping output magnitudes smaller
    q_scale = 1.0 / (qk_nope_head_dim**0.5)
    kv_scale = 1.0 / (kv_lora_rank**0.5)

    # q_nope: [B, S, N, qk_nope_head_dim]
    q_nope = (
        torch.randn(batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=dtype, device=device)
        * q_scale
    )

    # q_pe: [B, S, N, qk_rope_head_dim]
    q_pe = (
        torch.randn(batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=dtype, device=device)
        * q_scale
    )

    # compressed_kv: [B, S, kv_lora_rank]
    compressed_kv = (
        torch.randn(batch_size, seq_len, kv_lora_rank, dtype=dtype, device=device) * kv_scale
    )

    # kpe: [B, S, 1, qk_rope_head_dim]
    kpe = (
        torch.randn(batch_size, seq_len, 1, qk_rope_head_dim, dtype=dtype, device=device) * q_scale
    )

    # kv_b_proj_weight: [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    # Xavier initialization for the projection weight
    weight_scale = 1.0 / (kv_lora_rank**0.5)
    kv_b_proj_weight = (
        torch.randn(num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=device)
        * weight_scale
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
    # Paged MLA caches (two separate caches)
    ckv_cache = torch.zeros(max_num_pages, page_size, kv_lora_rank, dtype=dtype, device=device)
    kpe_cache = torch.zeros(max_num_pages, page_size, qk_rope_head_dim, dtype=dtype, device=device)

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
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
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
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    batch_size: int,
    tokens_per_seq: list,
    page_size: int,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    kv_lora_rank: int,
):
    """Copy unpaged cache data to paged cache format.

    This is used to initialize paged cache with the same data as unpaged cache
    for comparison tests.

    Args:
        unpaged_cache: Source cache [batch, max_seq, dim]
        ckv_cache: Destination paged ckv cache [num_pages, page_size, kv_lora_rank]
        kpe_cache: Destination paged kpe cache [num_pages, page_size, qk_rope_head_dim]
        batch_size: Number of sequences
        tokens_per_seq: Number of tokens to copy per sequence
        page_size: Number of tokens per page
        cu_num_pages: Cumulative page counts from flashinfer metadata [batch_size + 1]
        cache_loc: Page indices from flashinfer metadata
        kv_lora_rank: Rank of compressed KV (split dimension)
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

            # Split unpaged cache into ckv and kpe portions
            unpaged_data = unpaged_cache[batch_idx, token_offset : token_offset + tokens_to_copy]
            ckv_cache[page_num, :tokens_to_copy] = unpaged_data[:, :kv_lora_rank]
            kpe_cache[page_num, :tokens_to_copy] = unpaged_data[:, kv_lora_rank:]
            token_offset += tokens_to_copy


@pytest.mark.parametrize("seq_length", [32, 128])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("batch_size", [8, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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
        flashinfer_meta["ckv_cache"],
        flashinfer_meta["kpe_cache"],
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

    # FlashInfer uses fused kernels with different computation order/precision than the
    # torch reference. With bfloat16 and scaled inputs, tighter tolerances are achievable.
    assert torch.allclose(
        flashinfer_output_reshaped.cpu().to(torch.float32),
        torch_output_reshaped.cpu().to(torch.float32),
        atol=0.05,
        rtol=0.02,
    ), (
        f"FlashInfer MLA context output doesn't match torch backend. "
        f"Max diff: {(flashinfer_output_reshaped - torch_output_reshaped).abs().max():.6f}"
    )


@pytest.mark.parametrize("prefill_seq_length", [64, 128])
@pytest.mark.parametrize("num_heads", [1])
@pytest.mark.parametrize("batch_size", [4, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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
            flashinfer_meta["ckv_cache"],
            flashinfer_meta["kpe_cache"],
            batch_size,
            [prefill_seq_length] * batch_size,  # Number of tokens to copy
            page_size,
            flashinfer_meta["cu_num_pages"],
            flashinfer_meta["cache_loc"],
            kv_lora_rank,
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
        flashinfer_meta["ckv_cache"],
        flashinfer_meta["kpe_cache"],
        None,  # scale
        kv_lora_rank,
    )

    # =========================================================================
    # Compare outputs
    # =========================================================================
    # FlashInfer uses fused kernels with different computation order/precision than the
    # torch reference. With bfloat16 and scaled inputs, tighter tolerances are achievable.
    assert torch.allclose(
        flashinfer_output.cpu().to(torch.float32),
        torch_output.cpu().to(torch.float32),
        atol=0.05,
        rtol=0.02,
    ), (
        f"FlashInfer MLA decode output doesn't match torch backend. "
        f"Max diff: {(flashinfer_output - torch_output).abs().max():.6f}"
    )


@pytest.mark.parametrize("prefill_seq_length", [16, 128, 1024])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("batch_size", [4, 64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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
    # Use a fixed page_size of 64 for FlashInfer MLA.
    page_size = 64

    max_seq_len = 2048
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
        flashinfer_meta["ckv_cache"],
        flashinfer_meta["kpe_cache"],
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
        atol=0.01,
        rtol=0.01,
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
    # Use the same caches
    flashinfer_meta_gen["ckv_cache"] = flashinfer_meta["ckv_cache"]
    flashinfer_meta_gen["kpe_cache"] = flashinfer_meta["kpe_cache"]

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
        flashinfer_meta_gen["ckv_cache"],
        flashinfer_meta_gen["kpe_cache"],
        None,
        kv_lora_rank,
    )

    # Verify generate outputs match
    assert torch.allclose(
        flashinfer_output_gen.cpu().to(torch.float32),
        torch_output_gen.cpu().to(torch.float32),
        atol=0.05,
        rtol=0.02,
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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
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
        flashinfer_meta["ckv_cache"],
        flashinfer_meta["kpe_cache"],
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


@pytest.mark.parametrize(
    "seq_lengths",
    [
        [8, 16, 32],
        [12, 24, 48, 64],
        [16, 32, 64, 96, 128],
    ],
)
@pytest.mark.parametrize("num_decode_steps", [3, 5])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_variable_seq_multi_decode(
    seq_lengths, num_decode_steps, num_heads, dtype, device
):
    """Test FlashInfer MLA with variable sequence lengths and multiple decode steps.

    This test verifies the full workflow with variable sequence lengths:
    1. Context phase: Process initial sequences with different lengths
    2. Multiple decode steps: Generate multiple tokens, updating the cache each step

    Compares torch_backend_mla_with_cache against flashinfer_mla_with_cache.
    """
    batch_size = len(seq_lengths)

    # MLA dimensions
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # Must be 64 for FlashInfer MLA.
    kv_lora_rank = 512  # Must be 512 for FlashInfer MLA.
    v_head_dim = 128
    page_size = 64

    max_seq_len = max(seq_lengths) + num_decode_steps + 128  # Extra headroom
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

    # Create individual inputs for each sequence (context phase)
    total_tokens = sum(seq_lengths)

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

    # Concatenate into flattened format for context phase
    q_nope_flat = torch.cat(q_nope_list, dim=0).unsqueeze(0)  # [1, total_tokens, N, D]
    q_pe_flat = torch.cat(q_pe_list, dim=0).unsqueeze(0)
    compressed_kv_flat = torch.cat(compressed_kv_list, dim=0).unsqueeze(0)
    kpe_flat = torch.cat(kpe_list, dim=0).unsqueeze(0)

    # Common kv_b_proj_weight
    kv_head_dim = qk_nope_head_dim + v_head_dim
    weight_scale = 1.0 / (kv_lora_rank**0.5)
    kv_b_proj_weight = (
        torch.randn(num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=device)
        * weight_scale
    )

    input_positions_context = [0] * batch_size

    # =========================================================================
    # Context phase - Setup both backends
    # =========================================================================

    # Create torch unpaged cache
    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions_context,
    )

    # Create flashinfer paged cache
    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions_context,
    )

    # Run torch backend context
    torch_output_context = torch.ops.auto_deploy.torch_cached_mla_with_cache(
        q_nope_flat,
        q_pe_flat,
        compressed_kv_flat,
        kpe_flat,
        kv_b_proj_weight,
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

    flashinfer_output_context = torch.ops.auto_deploy.flashinfer_mla_with_cache(
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
        flashinfer_meta["ckv_cache"],
        flashinfer_meta["kpe_cache"],
        None,
        kv_lora_rank,
    )

    # Verify context outputs match
    assert torch.allclose(
        flashinfer_output_context.cpu().to(torch.float32),
        torch_output_context.cpu().to(torch.float32),
        atol=0.05,
        rtol=0.02,
    ), (
        f"Context phase outputs don't match. "
        f"Max diff: {(flashinfer_output_context - torch_output_context).abs().max():.6f}"
    )

    # =========================================================================
    # Multiple decode steps
    # =========================================================================
    current_positions = list(seq_lengths)  # Track current position for each sequence

    for decode_step in range(num_decode_steps):
        # Create decode inputs for this step
        inputs_decode = _create_mla_inputs(
            batch_size,
            1,  # seq_length = 1 for decode
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            v_head_dim,
            dtype,
            device,
        )

        seq_lengths_decode = [1] * batch_size
        input_positions_decode = current_positions.copy()

        # Update torch metadata for decode
        torch_meta_decode = _create_unpaged_cache_and_metadata(
            batch_size,
            max_seq_len,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype,
            device,
            seq_lengths_decode,
            input_positions_decode,
        )
        # Use the same cache (accumulated from context and previous decode steps)
        torch_meta_decode["mla_cache"] = torch_meta["mla_cache"]

        # Update flashinfer metadata for decode
        flashinfer_meta_decode = _create_paged_cache_and_metadata(
            batch_size,
            max_num_pages,
            page_size,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype,
            device,
            seq_lengths_decode,
            input_positions_decode,
        )
        # Use the same caches
        flashinfer_meta_decode["ckv_cache"] = flashinfer_meta["ckv_cache"]
        flashinfer_meta_decode["kpe_cache"] = flashinfer_meta["kpe_cache"]

        # Run torch backend decode
        torch_output_decode = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            inputs_decode["q_nope"],
            inputs_decode["q_pe"],
            inputs_decode["compressed_kv"],
            inputs_decode["kpe"],
            kv_b_proj_weight,
            torch_meta_decode["batch_info_host"],
            torch_meta_decode["seq_len"],
            torch_meta_decode["input_pos"],
            torch_meta_decode["slot_idx"],
            torch_meta_decode["cu_seqlen"],
            torch_meta_decode["mla_cache"],
            None,
            kv_lora_rank,
        )

        # Run FlashInfer decode
        _GlobalFlashInferMLAPlanner.reset(torch.device(device))

        qo_indptr_decode = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        qo_indptr_decode[1:] = torch.cumsum(
            torch.tensor(seq_lengths_decode, device=device), dim=0
        ).int()

        batch_indices_decode, positions_decode = flashinfer.get_batch_indices_positions(
            qo_indptr_decode,
            flashinfer.get_seq_lens(
                flashinfer_meta_decode["cu_num_pages"],
                flashinfer_meta_decode["last_page_len"],
                page_size=page_size,
            ),
            batch_size,
        )

        flashinfer_output_decode = torch.ops.auto_deploy.flashinfer_mla_with_cache(
            inputs_decode["q_nope"],
            inputs_decode["q_pe"],
            inputs_decode["compressed_kv"],
            inputs_decode["kpe"],
            kv_b_proj_weight,
            flashinfer_meta_decode["batch_info_host"],
            flashinfer_meta_decode["cu_seqlen_host"],
            flashinfer_meta_decode["cu_num_pages"],
            flashinfer_meta_decode["cu_num_pages_host"],
            flashinfer_meta_decode["cache_loc"],
            flashinfer_meta_decode["last_page_len"],
            flashinfer_meta_decode["last_page_len_host"],
            flashinfer_meta_decode["seq_len_with_cache_host"],
            batch_indices_decode,
            positions_decode,
            flashinfer_meta_decode["ckv_cache"],
            flashinfer_meta_decode["kpe_cache"],
            None,
            kv_lora_rank,
        )

        # Verify decode outputs match
        assert torch.allclose(
            flashinfer_output_decode.cpu().to(torch.float32),
            torch_output_decode.cpu().to(torch.float32),
            atol=0.05,
            rtol=0.02,
        ), (
            f"Decode step {decode_step + 1} outputs don't match. "
            f"Max diff: {(flashinfer_output_decode - torch_output_decode).abs().max():.6f}"
        )
        # Update positions for next decode step
        current_positions = [pos + 1 for pos in current_positions]

    # Final verification: all outputs should be finite
    assert torch.isfinite(flashinfer_output_decode).all(), (
        "Final decode output contains NaN or Inf values"
    )


@pytest.mark.parametrize(
    "chunk_config",
    [
        # Each config has list of chunk sizes per sequence
        # e.g., [[32, 16, 8], [64, 32, 16]] means 2 sequences with 3 chunks each
        {"chunks_per_seq": [[32, 16], [64, 32]]},  # 2 sequences, 2 chunks each
        {"chunks_per_seq": [[32, 16, 8], [64, 32, 16]]},  # 2 sequences, 3 chunks each
        {"chunks_per_seq": [[64, 32, 16, 8]]},  # 1 sequence, 4 chunks
        {
            "chunks_per_seq": [[32, 32, 32], [48, 48, 48], [64, 64, 64]]
        },  # 3 sequences, 3 chunks each
        {"chunks_per_seq": [[16, 16, 16, 16, 16]]},  # 1 sequence, 5 chunks
    ],
)
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_chunked_prefill(chunk_config, num_heads, dtype, device):
    """Test FlashInfer MLA chunked prefill (incremental prefill) with multiple chunks.

    This test verifies that chunked prefill works correctly when:
    1. First chunk is processed (input_pos == 0) - uses BatchPrefillWithRaggedKVCacheWrapper
    2. Subsequent chunks are processed (input_pos > 0) - uses BatchMLAPagedAttentionWrapper

    In chunked prefill, the Q tokens attend to all KV tokens (cached + current),
    which is different from regular prefill where Q and KV lengths are equal.

    Compares flashinfer_mla_with_cache against torch_backend_mla_with_cache.
    """
    chunks_per_seq = chunk_config["chunks_per_seq"]
    batch_size = len(chunks_per_seq)
    num_chunks = len(chunks_per_seq[0])  # Assume all sequences have same number of chunks

    # MLA dimensions
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # Must be 64 for FlashInfer MLA.
    kv_lora_rank = 512  # Must be 512 for FlashInfer MLA.
    v_head_dim = 128
    page_size = 32

    # Calculate total sequence lengths
    total_seq_lengths = [sum(chunks) for chunks in chunks_per_seq]
    max_seq_len = max(total_seq_lengths) + 128
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

    # Common kv_b_proj_weight
    kv_head_dim = qk_nope_head_dim + v_head_dim
    weight_scale = 1.0 / (kv_lora_rank**0.5)
    kv_b_proj_weight = (
        torch.randn(num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=device)
        * weight_scale
    )

    # Initialize caches (will be reused across chunks)
    torch_mla_cache = torch.zeros(
        batch_size, max_seq_len, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )
    flashinfer_ckv_cache = torch.zeros(
        max_num_pages, page_size, kv_lora_rank, dtype=dtype, device=device
    )
    flashinfer_kpe_cache = torch.zeros(
        max_num_pages, page_size, qk_rope_head_dim, dtype=dtype, device=device
    )

    # Track cumulative positions per sequence
    cumulative_positions = [0] * batch_size

    # Process each chunk
    for chunk_idx in range(num_chunks):
        # Get current chunk lengths
        current_chunk_lengths = [
            chunks_per_seq[seq_idx][chunk_idx] for seq_idx in range(batch_size)
        ]
        total_tokens = sum(current_chunk_lengths)
        input_positions = cumulative_positions.copy()

        # Create inputs for this chunk
        q_nope_list = []
        q_pe_list = []
        compressed_kv_list = []
        kpe_list = []

        for chunk_len in current_chunk_lengths:
            inputs = _create_mla_inputs(
                1,
                chunk_len,
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

        q_nope_flat = torch.cat(q_nope_list, dim=0).unsqueeze(0)
        q_pe_flat = torch.cat(q_pe_list, dim=0).unsqueeze(0)
        compressed_kv_flat = torch.cat(compressed_kv_list, dim=0).unsqueeze(0)
        kpe_flat = torch.cat(kpe_list, dim=0).unsqueeze(0)

        # =====================================================================
        # Torch backend
        # =====================================================================
        torch_meta = _create_unpaged_cache_and_metadata(
            batch_size,
            max_seq_len,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype,
            device,
            current_chunk_lengths,
            input_positions,
        )
        torch_meta["mla_cache"] = torch_mla_cache  # Use shared cache

        torch_output = torch.ops.auto_deploy.torch_cached_mla_with_cache(
            q_nope_flat,
            q_pe_flat,
            compressed_kv_flat,
            kpe_flat,
            kv_b_proj_weight,
            torch_meta["batch_info_host"],
            torch_meta["seq_len"],
            torch_meta["input_pos"],
            torch_meta["slot_idx"],
            torch_meta["cu_seqlen"],
            torch_meta["mla_cache"],
            None,
            kv_lora_rank,
        )

        # =====================================================================
        # FlashInfer backend
        # =====================================================================
        _GlobalFlashInferMLAPlanner.reset(torch.device(device))

        flashinfer_meta = _create_paged_cache_and_metadata(
            batch_size,
            max_num_pages,
            page_size,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype,
            device,
            current_chunk_lengths,
            input_positions,
        )
        # Use shared caches
        flashinfer_meta["ckv_cache"] = flashinfer_ckv_cache
        flashinfer_meta["kpe_cache"] = flashinfer_kpe_cache

        qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(
            torch.tensor(current_chunk_lengths, device=device), dim=0
        ).int()

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
            flashinfer_meta["ckv_cache"],
            flashinfer_meta["kpe_cache"],
            None,
            kv_lora_rank,
        )

        # Verify outputs match
        is_first_chunk = chunk_idx == 0
        chunk_type = "regular prefill" if is_first_chunk else "chunked prefill"
        assert torch.allclose(
            flashinfer_output.cpu().to(torch.float32),
            torch_output.cpu().to(torch.float32),
            atol=0.05,
            rtol=0.02,
        ), (
            f"Chunk {chunk_idx + 1}/{num_chunks} ({chunk_type}) outputs don't match. "
            f"Max diff: {(flashinfer_output - torch_output).abs().max():.6f}"
        )

        # Verify outputs are finite
        assert torch.isfinite(flashinfer_output).all(), (
            f"Chunk {chunk_idx + 1}/{num_chunks} ({chunk_type}) output contains NaN or Inf values"
        )

        # Update cumulative positions for next chunk
        for seq_idx in range(batch_size):
            cumulative_positions[seq_idx] += current_chunk_lengths[seq_idx]


# =============================================================================
# CUDA Graph Tests
# =============================================================================
# Tests for CUDA graph functionality of the FlashInfer MLA planner to verify
# that wrappers are correctly created and cached with use_cuda_graph=True.


@pytest.mark.parametrize("prefill_seq_length", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_cuda_graph_wrapper_creation(
    prefill_seq_length, num_heads, batch_size, dtype, device
):
    """Test that CUDA graph wrappers are created with use_cuda_graph=True during warm-up.

    This test verifies that:
    1. During CudaGraphWarmUpPhase, the planner creates a wrapper with use_cuda_graph=True
    2. The wrapper is cached in cached_cuda_graph_decode_wrappers
    3. The wrapper has correct buffer tensors attached
    """
    # MLA dimensions
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # Must be 64 for FlashInfer MLA.
    kv_lora_rank = 512  # Must be 512 for FlashInfer MLA.
    v_head_dim = 128
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

    # Pre-fill cache with random data
    if prefill_seq_length > 0:
        total_prefill_pages = (prefill_seq_length - 1) // page_size + 1
        for batch_idx in range(batch_size):
            page_start = batch_idx * total_prefill_pages
            for page_offset in range(total_prefill_pages):
                page_idx = page_start + page_offset
                tokens_in_page = min(page_size, prefill_seq_length - page_offset * page_size)
                if tokens_in_page > 0:
                    flashinfer_meta["ckv_cache"][page_idx, :tokens_in_page] = torch.randn(
                        tokens_in_page, kv_lora_rank, dtype=dtype, device=device
                    ) / (kv_lora_rank**0.5)
                    flashinfer_meta["kpe_cache"][page_idx, :tokens_in_page] = torch.randn(
                        tokens_in_page, qk_rope_head_dim, dtype=dtype, device=device
                    ) / (qk_rope_head_dim**0.5)

    # Reset planner
    _GlobalFlashInferMLAPlanner.workspace_buffer = None
    _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers = {}
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

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

    # Verify no wrappers exist before warm-up
    assert len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers) == 0, (
        "Expected no cached wrappers before warm-up"
    )

    # Warm-up phase: This triggers wrapper creation with use_cuda_graph=True
    with CudaGraphWarmUpPhase():
        output = torch.ops.auto_deploy.flashinfer_mla_with_cache(
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
            flashinfer_meta["ckv_cache"],
            flashinfer_meta["kpe_cache"],
            None,  # scale
            kv_lora_rank,
        )

    # Verify a CUDA graph wrapper was created
    assert len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers) == 1, (
        f"Expected 1 cached wrapper after warm-up, "
        f"got {len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers)}"
    )

    # Verify the wrapper has the correct plan params
    for (
        plan_params,
        wrapper,
    ) in _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers.items():
        assert plan_params.num_seq == batch_size, (
            f"Plan params num_seq={plan_params.num_seq} doesn't match batch_size={batch_size}"
        )
        assert plan_params.num_heads == num_heads, (
            f"Plan params num_heads={plan_params.num_heads} doesn't match num_heads={num_heads}"
        )
        assert plan_params.kv_lora_rank == kv_lora_rank, (
            f"Plan params kv_lora_rank={plan_params.kv_lora_rank} doesn't match "
            f"kv_lora_rank={kv_lora_rank}"
        )
        assert plan_params.page_size == page_size, (
            f"Plan params page_size={plan_params.page_size} doesn't match page_size={page_size}"
        )

        # Verify wrapper is not None
        assert wrapper is not None, "CUDA graph wrapper should not be None"

    # Verify output is valid
    expected_shape = (batch_size, seq_length, num_heads, v_head_dim)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_cuda_graph_wrapper_caching_per_batch_size(batch_size, dtype, device):
    """Test that CUDA graph wrappers are cached per batch size.

    This test verifies that:
    1. Each batch size gets its own cached wrapper
    2. Wrappers are keyed by MLADecodePlanParams which includes num_seq
    """
    # MLA dimensions
    num_heads = 8
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    page_size = 64
    prefill_seq_length = 64

    seq_length = 1  # Decode phase

    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // page_size + 1)

    # Create inputs
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

    seq_lengths = [seq_length] * batch_size
    input_positions = [prefill_seq_length] * batch_size

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

    # Pre-fill cache
    if prefill_seq_length > 0:
        total_prefill_pages = (prefill_seq_length - 1) // page_size + 1
        for batch_idx in range(batch_size):
            page_start = batch_idx * total_prefill_pages
            for page_offset in range(total_prefill_pages):
                page_idx = page_start + page_offset
                tokens_in_page = min(page_size, prefill_seq_length - page_offset * page_size)
                if tokens_in_page > 0:
                    flashinfer_meta["ckv_cache"][page_idx, :tokens_in_page] = torch.randn(
                        tokens_in_page, kv_lora_rank, dtype=dtype, device=device
                    ) / (kv_lora_rank**0.5)
                    flashinfer_meta["kpe_cache"][page_idx, :tokens_in_page] = torch.randn(
                        tokens_in_page, qk_rope_head_dim, dtype=dtype, device=device
                    ) / (qk_rope_head_dim**0.5)

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

    # Reset planner
    _GlobalFlashInferMLAPlanner.workspace_buffer = None
    _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers = {}
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    # Warm-up to create CUDA graph wrapper for this batch size
    with CudaGraphWarmUpPhase():
        _ = torch.ops.auto_deploy.flashinfer_mla_with_cache(
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
            flashinfer_meta["ckv_cache"],
            flashinfer_meta["kpe_cache"],
            None,
            kv_lora_rank,
        )

    # Verify wrapper was created for this batch size
    assert len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers) == 1, (
        f"Expected 1 cached wrapper for batch_size={batch_size}, "
        f"got {len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers)}"
    )

    # Verify the wrapper has the correct num_seq
    for plan_params in _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers:
        assert plan_params.num_seq == batch_size, (
            f"Plan params num_seq={plan_params.num_seq} doesn't match batch_size={batch_size}"
        )


@pytest.mark.parametrize("prefill_seq_length", [64])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_plan_generate_only(
    prefill_seq_length, num_heads, batch_size, dtype, device
):
    """Test plan_generate_only function for re-planning decode-only batches.

    This test verifies that:
    1. plan_generate_only can re-plan cached CUDA graph wrappers
    2. The wrappers can be used after re-planning
    """
    # MLA dimensions
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    page_size = 64

    seq_length = 1  # Decode phase

    max_seq_len = 256
    max_num_pages = batch_size * (max_seq_len // page_size + 1)

    # Create inputs
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

    seq_lengths = [seq_length] * batch_size
    input_positions = [prefill_seq_length] * batch_size

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

    # Pre-fill cache
    if prefill_seq_length > 0:
        total_prefill_pages = (prefill_seq_length - 1) // page_size + 1
        for batch_idx in range(batch_size):
            page_start = batch_idx * total_prefill_pages
            for page_offset in range(total_prefill_pages):
                page_idx = page_start + page_offset
                tokens_in_page = min(page_size, prefill_seq_length - page_offset * page_size)
                if tokens_in_page > 0:
                    flashinfer_meta["ckv_cache"][page_idx, :tokens_in_page] = torch.randn(
                        tokens_in_page, kv_lora_rank, dtype=dtype, device=device
                    ) / (kv_lora_rank**0.5)
                    flashinfer_meta["kpe_cache"][page_idx, :tokens_in_page] = torch.randn(
                        tokens_in_page, qk_rope_head_dim, dtype=dtype, device=device
                    ) / (qk_rope_head_dim**0.5)

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

    # Reset planner
    _GlobalFlashInferMLAPlanner.workspace_buffer = None
    _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers = {}
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    # First warm-up to create the wrapper
    with CudaGraphWarmUpPhase():
        output1 = torch.ops.auto_deploy.flashinfer_mla_with_cache(
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
            flashinfer_meta["ckv_cache"],
            flashinfer_meta["kpe_cache"],
            None,
            kv_lora_rank,
        )

    # Verify wrapper was created
    assert len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers) == 1

    # Now test plan_generate_only - this is called by the host-side preparation
    # to re-plan the cached wrappers before graph replay
    _GlobalFlashInferMLAPlanner.plan_generate_only(
        batch_size,
        flashinfer_meta["cu_num_pages"][: batch_size + 1],
        flashinfer_meta["cache_loc"],
        flashinfer_meta["last_page_len"][:batch_size],
    )

    # Run again (not in warm-up, so it should use cached wrapper)
    # First, update the inputs to simulate new tokens
    new_inputs = _create_mla_inputs(
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

    # Create new metadata for position+1
    new_input_positions = [prefill_seq_length + 1] * batch_size
    new_flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        new_input_positions,
    )
    # Reuse the same cache
    new_flashinfer_meta["ckv_cache"] = flashinfer_meta["ckv_cache"]
    new_flashinfer_meta["kpe_cache"] = flashinfer_meta["kpe_cache"]

    new_qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    new_qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lengths, device=device), dim=0).int()

    new_batch_indices, new_positions = flashinfer.get_batch_indices_positions(
        new_qo_indptr,
        flashinfer.get_seq_lens(
            new_flashinfer_meta["cu_num_pages"],
            new_flashinfer_meta["last_page_len"],
            page_size=page_size,
        ),
        batch_size * seq_length,
    )

    output2 = torch.ops.auto_deploy.flashinfer_mla_with_cache(
        new_inputs["q_nope"],
        new_inputs["q_pe"],
        new_inputs["compressed_kv"],
        new_inputs["kpe"],
        inputs["kv_b_proj_weight"],  # Use same weights
        new_flashinfer_meta["batch_info_host"],
        new_flashinfer_meta["cu_seqlen_host"],
        new_flashinfer_meta["cu_num_pages"],
        new_flashinfer_meta["cu_num_pages_host"],
        new_flashinfer_meta["cache_loc"],
        new_flashinfer_meta["last_page_len"],
        new_flashinfer_meta["last_page_len_host"],
        new_flashinfer_meta["seq_len_with_cache_host"],
        new_batch_indices,
        new_positions,
        new_flashinfer_meta["ckv_cache"],
        new_flashinfer_meta["kpe_cache"],
        None,
        kv_lora_rank,
    )

    # Verify output is valid
    expected_shape = (batch_size, seq_length, num_heads, v_head_dim)
    assert output2.shape == expected_shape, f"Expected shape {expected_shape}, got {output2.shape}"
    assert torch.isfinite(output2).all(), "Output contains NaN or Inf values"

    # Outputs should be different since inputs are different
    assert not torch.allclose(output1, output2, atol=1e-6), (
        "Outputs should differ since inputs are different"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_cuda_graph_multiple_batch_sizes(dtype, device):
    """Test that multiple batch sizes can have their own CUDA graph wrappers.

    This test verifies that the planner correctly caches wrappers for
    different batch sizes, which is important for supporting multiple
    CUDA graph configurations.
    """
    # MLA dimensions
    num_heads = 8
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512
    v_head_dim = 128
    page_size = 64
    prefill_seq_length = 64

    seq_length = 1  # Decode phase

    batch_sizes = [4, 8, 16]

    # Reset planner
    _GlobalFlashInferMLAPlanner.workspace_buffer = None
    _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers = {}
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    for batch_size in batch_sizes:
        max_num_pages = batch_size * (256 // page_size + 1)

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

        seq_lengths = [seq_length] * batch_size
        input_positions = [prefill_seq_length] * batch_size

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
            batch_size * seq_length,
        )

        # Warm-up to create wrapper for this batch size
        with CudaGraphWarmUpPhase():
            _ = torch.ops.auto_deploy.flashinfer_mla_with_cache(
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
                flashinfer_meta["ckv_cache"],
                flashinfer_meta["kpe_cache"],
                None,
                kv_lora_rank,
            )

    # Verify we have a wrapper for each batch size
    assert len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers) == len(batch_sizes), (
        f"Expected {len(batch_sizes)} cached wrappers, "
        f"got {len(_GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers)}"
    )

    # Verify each batch size has a wrapper
    cached_num_seqs = {
        params.num_seq for params in _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers
    }
    assert cached_num_seqs == set(batch_sizes), (
        f"Expected wrappers for batch_sizes {batch_sizes}, got {cached_num_seqs}"
    )


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cuda"])
def test_flashinfer_mla_init_decode_wrapper_with_buffers(batch_size, dtype, device):
    """Test that _init_decode_wrapper correctly passes buffer tensors with use_cuda_graph=True.

    This test directly tests the _init_decode_wrapper method to verify buffer handling.
    """
    # Reset planner
    _GlobalFlashInferMLAPlanner.workspace_buffer = None
    _GlobalFlashInferMLAPlanner.cached_cuda_graph_decode_wrappers = {}
    _GlobalFlashInferMLAPlanner.reset(torch.device(device))

    # Create buffer tensors
    qo_indptr = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    kv_indptr = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * 2
    kv_indices = torch.arange(batch_size * 2, device=device, dtype=torch.int32)
    kv_len_arr = torch.ones(batch_size, device=device, dtype=torch.int32) * 64

    # Test creating wrapper without CUDA graph (no buffers needed)
    wrapper_no_cg = _GlobalFlashInferMLAPlanner._init_decode_wrapper(use_cuda_graph=False)
    assert wrapper_no_cg is not None, "Should create wrapper without CUDA graph"

    # Test creating wrapper with CUDA graph (buffers required)
    wrapper_with_cg = _GlobalFlashInferMLAPlanner._init_decode_wrapper(
        use_cuda_graph=True,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
    )
    assert wrapper_with_cg is not None, "Should create wrapper with CUDA graph"

    # Both wrappers should be valid BatchMLAPagedAttentionWrapper instances
    assert isinstance(wrapper_no_cg, flashinfer.mla.BatchMLAPagedAttentionWrapper), (
        "Should be BatchMLAPagedAttentionWrapper"
    )
    assert isinstance(wrapper_with_cg, flashinfer.mla.BatchMLAPagedAttentionWrapper), (
        "Should be BatchMLAPagedAttentionWrapper"
    )
