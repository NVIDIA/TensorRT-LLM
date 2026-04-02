import math

import torch
import triton
import triton.language as tl

########################################################
# RocketKV-specific Triton kernels
########################################################


@triton.jit
def rocket_qk_split_kernel(
    input_ptr,
    q_output_ptr,
    k_output_ptr,
    k_output_offsets_ptr,
    context_lens_ptr,
    context_cumsum_ptr,
    valid_seq_indices_ptr,
    window_size,
    num_heads,
    num_kv_heads,
    head_dim,
    q_total_output_tokens,
    k_total_output_tokens,
    valid_batch_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    valid_seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    dim_offset = tl.program_id(2) * BLOCK_SIZE

    if valid_seq_idx >= valid_batch_size:
        return

    # Determine if this is a Q head or K head based on head_idx
    is_q_head = head_idx < num_heads
    is_k_head = head_idx >= num_heads and head_idx < (num_heads + num_kv_heads)

    if not (is_q_head or is_k_head):
        return

    orig_seq_idx = tl.load(valid_seq_indices_ptr + valid_seq_idx)

    seq_start_offset = tl.load(context_cumsum_ptr + orig_seq_idx)

    context_len = tl.load(context_lens_ptr + orig_seq_idx)

    if is_q_head:
        # Process Q head: extract last window_size tokens
        actual_head_idx = head_idx
        extract_start_offset = context_len - window_size
        extract_length = window_size
        output_offset = valid_seq_idx * window_size
        output_ptr = q_output_ptr
        total_output_tokens = q_total_output_tokens
        input_dim_offset = 0
    else:
        # Process K head: extract first (context_len - window_size) tokens
        actual_head_idx = head_idx - num_heads
        extract_start_offset = 0
        extract_length = context_len - window_size
        output_offset = tl.load(k_output_offsets_ptr + valid_seq_idx)
        output_ptr = k_output_ptr
        total_output_tokens = k_total_output_tokens
        input_dim_offset = num_heads * head_dim

    dim_indices = dim_offset + tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_indices < head_dim

    for token_block_start in tl.range(0, extract_length, BLOCK_SIZE_M):
        token_indices = token_block_start + tl.arange(0, BLOCK_SIZE_M)
        token_mask = token_indices < extract_length

        src_token_pos = seq_start_offset + extract_start_offset + token_indices  # [BLOCK_SIZE_M]
        dst_token_pos = output_offset + token_indices  # [BLOCK_SIZE_M]

        # Calculate input indices with proper dimension offset for Q/K separation
        input_dim_indices = input_dim_offset + actual_head_idx * head_dim + dim_indices
        src_indices = (
            src_token_pos[:, None] * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
            + input_dim_indices[None, :]
        )

        # Calculate output indices
        dst_indices = (
            actual_head_idx * total_output_tokens + dst_token_pos[:, None]
        ) * head_dim + dim_indices[None, :]

        full_mask = token_mask[:, None] & dim_mask[None, :]  # [BLOCK_SIZE_M, BLOCK_SIZE]

        data = tl.load(input_ptr + src_indices, mask=full_mask, other=0.0)
        tl.store(output_ptr + dst_indices, data, mask=full_mask)


def triton_rocket_qk_split(
    input_tensor: torch.Tensor,
    input_lens: torch.Tensor,
    input_lens_cumsum: torch.Tensor,
    valid_seq_indices: torch.Tensor,
    k_output_offsets: torch.Tensor,
    total_rocket_k_ctx_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_size: int,
    valid_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Splits input flattened tensor along along token dimension into:
    - Q: last window_size tokens from each sequence
    - K: first (context_len - window_size) tokens from each sequence
    It will ignore the invalid sequences, the length of which is less than a threshold.
    So we should prepare the valid_seq_indices, valid_batch_size and k_output_offsets in advance.

    Args:
        input_tensor: Input tensor [total_tokens, num_heads*head_dim + 2*num_kv_heads*head_dim]
        input_lens: Context length for each sequence [batch_size]
        input_lens_cumsum: Cumulative sum of context lengths [batch_size + 1]
        valid_seq_indices: Indices of valid sequences [valid_batch_size]
        k_output_offsets: Offset for each valid sequence [valid_batch_size]
        total_rocket_k_ctx_tokens: Total number of RocketKV key context tokens
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension of each head
        window_size: Size of the window for queries
        valid_batch_size: Number of valid sequences in batch

    Returns:
        q_window: Window queries [num_heads, window_size * valid_batch_size, head_dim]
        k_context: Context keys [num_kv_heads, sum(valid_context_lens - window_size), head_dim]
    """

    q_total_output_tokens = window_size * valid_batch_size
    k_total_output_tokens = total_rocket_k_ctx_tokens

    q_window = torch.empty(
        (num_heads, q_total_output_tokens, head_dim),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    k_context = torch.empty(
        (num_kv_heads, k_total_output_tokens, head_dim),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    BLOCK_SIZE = 128  # Dimension block size
    BLOCK_SIZE_M = 128  # Token block size for parallel processing

    # Grid: (valid_batch_size, num_heads + num_kv_heads, triton.cdiv(head_dim, BLOCK_SIZE))
    total_heads = num_heads + num_kv_heads
    grid = (valid_batch_size, total_heads, triton.cdiv(head_dim, BLOCK_SIZE))

    rocket_qk_split_kernel[grid](
        input_tensor,
        q_window,
        k_context,
        k_output_offsets,
        input_lens,
        input_lens_cumsum,
        valid_seq_indices,
        window_size,
        num_heads,
        num_kv_heads,
        head_dim,
        q_total_output_tokens,
        k_total_output_tokens,
        valid_batch_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )

    return q_window, k_context


########################################################
# BMM kernel
########################################################


@triton.jit
def rocket_batch_to_flatten_kernel(
    prefix_indices_ptr,
    output_indices_ptr,
    context_lens_ptr,
    valid_seq_indices_ptr,
    sparse_offsets_ptr,
    batch_size,
    valid_batch_size,
    num_kv_heads,
    prefix_budget,
    window_size,
    prompt_budget,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if batch_idx >= batch_size or head_idx >= num_kv_heads:
        return

    # Get context length for this batch
    context_len = tl.load(context_lens_ptr + batch_idx)

    # Check if this batch is valid (appears in valid_seq_indices)
    is_valid = 0
    valid_idx_in_selected = -1

    # Search for current batch_idx in valid_seq_indices
    for valid_idx in tl.range(0, valid_batch_size):
        orig_idx = tl.load(valid_seq_indices_ptr + valid_idx)
        if orig_idx == batch_idx:
            is_valid = 1
            valid_idx_in_selected = valid_idx

    # Calculate output offset for this batch
    output_offset = tl.load(sparse_offsets_ptr + batch_idx)
    total_sparse_tokens = tl.load(sparse_offsets_ptr + batch_size)

    if is_valid:
        # Valid batch: copy prefix indices and compute window indices
        # Process prefix tokens
        for token_block_start in tl.range(0, prefix_budget, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < prefix_budget

            # Load from prefix_indices
            flattened_idx = valid_idx_in_selected * num_kv_heads + head_idx
            prefix_indices = flattened_idx * prefix_budget + token_offsets
            prefix_values = tl.load(prefix_indices_ptr + prefix_indices, mask=token_mask, other=0)

            # Store to output
            output_indices = head_idx * total_sparse_tokens + output_offset + token_offsets
            tl.store(output_indices_ptr + output_indices, prefix_values, mask=token_mask)

        # Process window tokens
        for token_block_start in tl.range(0, window_size, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < window_size

            # Compute window indices: [context_len - window_size, context_len - window_size + 1, ...]
            window_values = context_len - window_size + token_offsets

            # Store to output at prefix_budget offset
            output_indices = (
                head_idx * total_sparse_tokens + output_offset + prefix_budget + token_offsets
            )
            tl.store(output_indices_ptr + output_indices, window_values, mask=token_mask)
    else:
        # Invalid batch: generate [0, 1, ..., context_len-1]
        num_tokens = context_len
        for token_block_start in tl.range(0, num_tokens, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < num_tokens

            # Generate sequential indices
            sequential_indices = token_offsets

            # Store to output
            output_indices = head_idx * total_sparse_tokens + output_offset + token_offsets
            tl.store(output_indices_ptr + output_indices, sequential_indices, mask=token_mask)


def triton_rocket_batch_to_flatten(
    prefix_indices: torch.Tensor,
    input_lens: torch.Tensor,
    valid_seq_indices: torch.Tensor,
    output_offsets: torch.Tensor,
    batch_size: int,
    total_output_tokens: int,
    window_size: int,
    prompt_budget: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten indices considering both valid and invalid batches.
    For valid sequences, combines prefix_indices with dynamically computed window indices.
    For invalid sequences, generates sequential indices.

    Args:
        prefix_indices: Selected prefix indices [valid_batch_size * num_kv_heads, prefix_budget]
        input_lens: Lengths for all sequences [batch_size]
        valid_seq_indices: Valid sequence indices [valid_batch_size]
        output_offsets: Offset for each batch [batch_size + 1]
        batch_size: Number of batches
        total_output_tokens: Total number of output tokens
        window_size: Size of sliding window at the end
        prompt_budget: Total number of tokens for valid sequences (prefix_budget + window_size)
        num_kv_heads: Number of KV heads

    Returns:
        sparse_indices: Flattened sparse indices [num_kv_heads, total_output_tokens]
    """
    total_tasks, prefix_budget = prefix_indices.shape
    valid_batch_size = total_tasks // num_kv_heads

    # Create output tensor
    sparse_indices = torch.empty(
        (num_kv_heads, total_output_tokens),
        dtype=prefix_indices.dtype,
        device=prefix_indices.device,
    )

    # Launch kernel
    BLOCK_SIZE = 512
    grid = (batch_size, num_kv_heads)

    rocket_batch_to_flatten_kernel[grid](
        prefix_indices,
        sparse_indices,
        input_lens,
        valid_seq_indices,
        output_offsets,
        batch_size,
        valid_batch_size,
        num_kv_heads,
        prefix_budget,
        window_size,
        prompt_budget,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return sparse_indices


########################################################
# KT cache update kernel
########################################################


@triton.jit
def rocket_update_kt_cache_gen_kernel(
    k_ptr,
    kt_cache_tensor_ptr,
    kt_cache_block_offsets_ptr,
    kv_lens_ptr,
    num_gen_tokens,
    num_kv_heads,
    head_dim,
    kt_page_size,
    tokens_per_block,
    max_kt_blocks_per_seq,
    k_stride_0,
    k_stride_1,
    DIM_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    dim_block_idx = tl.program_id(2)

    if batch_idx >= num_gen_tokens or kv_head_idx >= num_kv_heads:
        return

    dim_block_start = dim_block_idx * DIM_BLOCK_SIZE
    dim_offsets = tl.arange(0, DIM_BLOCK_SIZE)
    dim_indices = dim_block_start + dim_offsets
    dim_mask = dim_indices < head_dim

    k_base = batch_idx * k_stride_0 + kv_head_idx * head_dim * k_stride_1
    k_indices = k_base + dim_indices * k_stride_1
    k_values = tl.load(k_ptr + k_indices, mask=dim_mask, other=0.0)

    kv_len = tl.load(kv_lens_ptr + batch_idx)
    if kv_len <= 0:
        return

    # Determine which kt_token to update (the last one)
    last_token_idx = kv_len - 1
    last_kt_token_idx = last_token_idx // kt_page_size
    kt_token_idx_in_page = last_token_idx % kt_page_size

    block_offset_in_seq = last_kt_token_idx // tokens_per_block
    if block_offset_in_seq >= max_kt_blocks_per_seq:
        return

    block_idx = tl.load(
        kt_cache_block_offsets_ptr + batch_idx * max_kt_blocks_per_seq + block_offset_in_seq
    )
    token_idx_in_block = last_kt_token_idx % tokens_per_block

    cache_base = (
        block_idx * tokens_per_block + token_idx_in_block
    ) * num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim

    cache_min_indices = cache_base + dim_indices
    cache_max_indices = cache_base + head_dim + dim_indices

    kt_mask = dim_mask & (kt_token_idx_in_page > 0)

    k_min_existing = tl.load(
        kt_cache_tensor_ptr + cache_min_indices, mask=kt_mask, other=float("inf")
    )
    k_max_existing = tl.load(
        kt_cache_tensor_ptr + cache_max_indices, mask=kt_mask, other=float("-inf")
    )

    k_min_new = tl.minimum(k_min_existing, k_values)
    k_max_new = tl.maximum(k_max_existing, k_values)
    k_min_new = k_min_new.to(kt_cache_tensor_ptr.dtype.element_ty)
    k_max_new = k_max_new.to(kt_cache_tensor_ptr.dtype.element_ty)

    tl.store(kt_cache_tensor_ptr + cache_min_indices, k_min_new, mask=dim_mask)
    tl.store(kt_cache_tensor_ptr + cache_max_indices, k_max_new, mask=dim_mask)


def triton_rocket_update_kt_cache_gen(
    k: torch.Tensor,
    kt_cache_tensor: torch.Tensor,
    kt_cache_block_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    kt_page_size: int,
    tokens_per_block: int,
    max_kt_blocks_per_seq: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    """
    Update KT cache with new key values for generation phase.

    Args:
        k: Key tensor [num_gen_tokens, num_kv_heads * head_dim]
        kt_cache_tensor: KT cache tensor
        kt_cache_block_offsets: Block offsets [num_gen_tokens, max_kt_blocks_per_seq]
        kv_lens: Sequence lengths [num_gen_tokens]
        kt_page_size: Page size for KT tokens
        tokens_per_block: Tokens per cache block
        max_kt_blocks_per_seq: Maximum KT blocks per sequence
    """
    num_gen_tokens = k.shape[0]

    grid = (num_gen_tokens, num_kv_heads, 1)

    DIM_BLOCK_SIZE = triton.next_power_of_2(head_dim)

    rocket_update_kt_cache_gen_kernel[grid](
        k,
        kt_cache_tensor,
        kt_cache_block_offsets,
        kv_lens,
        num_gen_tokens,
        num_kv_heads,
        head_dim,
        kt_page_size,
        tokens_per_block,
        max_kt_blocks_per_seq,
        k.stride(0),
        k.stride(1),
        DIM_BLOCK_SIZE=DIM_BLOCK_SIZE,
    )


@triton.jit
def rocket_update_kt_cache_ctx_kernel(
    k_ptr,
    kt_cache_tensor_ptr,
    kt_cache_block_offsets_ptr,
    context_cumsum_ptr,
    sparse_kv_indices_ptr,
    sparse_kv_offsets_ptr,
    batch_size,
    num_heads,
    num_kv_heads,
    head_dim,
    kt_page_size: tl.constexpr,
    tokens_per_block,
    max_kt_blocks_per_seq,
    total_sparse_tokens,
    BLOCK_SIZE_KT: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    """
    Triton kernel for updating KT cache during context phase with sparse indices.
    """
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    kt_block_idx = tl.program_id(2)

    context_start = tl.load(context_cumsum_ptr + batch_idx)

    sparse_start = tl.load(sparse_kv_offsets_ptr + batch_idx)
    sparse_end = tl.load(sparse_kv_offsets_ptr + batch_idx + 1)
    num_sparse_tokens = sparse_end - sparse_start

    if num_sparse_tokens <= 0:
        return

    q_hidden_size = num_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim
    k_dim_base = q_hidden_size + kv_head_idx * head_dim

    BLOCK_SIZE_KV: tl.constexpr = kt_page_size * BLOCK_SIZE_KT

    total_kt_tokens = (num_sparse_tokens + kt_page_size - 1) // kt_page_size
    kt_offsets = kt_block_idx * BLOCK_SIZE_KT + tl.arange(0, BLOCK_SIZE_KT)
    kt_mask = kt_offsets < total_kt_tokens

    kv_start = kt_block_idx * BLOCK_SIZE_KT * kt_page_size
    kv_offsets = kv_start + tl.arange(0, BLOCK_SIZE_KV)
    kv_mask = kv_offsets < num_sparse_tokens
    kv_indices = kv_head_idx * total_sparse_tokens + sparse_start + kv_offsets

    for dim_block_start in tl.range(0, head_dim, BLOCK_SIZE_DIM):
        dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
        dim_indices = dim_block_start + dim_offsets
        dim_mask = dim_indices < head_dim

        kv_token_indices = tl.load(sparse_kv_indices_ptr + kv_indices, mask=kv_mask, other=0)
        # Calculate indices for loading keys [BLOCK_SIZE_DIM, BLOCK_SIZE_KV]
        k_base_indices = (kv_token_indices[None, :] + context_start) * (
            q_hidden_size + 2 * kv_hidden_size
        ) + k_dim_base
        k_indices = k_base_indices + dim_indices[:, None]

        combined_mask = kv_mask[None, :] & dim_mask[:, None]

        # Load key values [BLOCK_SIZE_DIM, BLOCK_SIZE_KV]
        k_values = tl.load(k_ptr + k_indices, mask=combined_mask, other=0.0)

        k_values = tl.reshape(k_values, (BLOCK_SIZE_DIM, BLOCK_SIZE_KT, kt_page_size))

        k_min = tl.min(k_values, axis=-1).to(kt_cache_tensor_ptr.dtype.element_ty)
        k_max = tl.max(k_values, axis=-1).to(kt_cache_tensor_ptr.dtype.element_ty)

        # Calculate cache locations [BLOCK_SIZE_KT]
        block_offsets_in_seq = kt_offsets // tokens_per_block
        valid_block_mask = (block_offsets_in_seq < max_kt_blocks_per_seq) & kt_mask

        # Load block indices [BLOCK_SIZE_KT]
        block_offset_addrs = batch_idx * max_kt_blocks_per_seq + block_offsets_in_seq
        block_indices = tl.load(
            kt_cache_block_offsets_ptr + block_offset_addrs, mask=valid_block_mask, other=0
        )

        tokens_in_block = kt_offsets % tokens_per_block

        # Calculate cache base addresses [BLOCK_SIZE_KT]
        cache_bases = (
            block_indices * tokens_per_block + tokens_in_block
        ) * num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim

        cache_min_addrs = cache_bases[None, :] + dim_indices[:, None]
        cache_max_addrs = cache_min_addrs + head_dim

        store_mask = valid_block_mask[None, :] & dim_mask[:, None]

        tl.store(kt_cache_tensor_ptr + cache_min_addrs, k_min, mask=store_mask)
        tl.store(kt_cache_tensor_ptr + cache_max_addrs, k_max, mask=store_mask)


def triton_rocket_update_kt_cache_ctx(
    qkv_input: torch.Tensor,
    kt_cache_tensor: torch.Tensor,
    kt_cache_block_offsets: torch.Tensor,
    context_cumsum: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_kv_offsets: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    kt_page_size: int,
    prompt_budget: int,
    tokens_per_block: int,
    max_kt_blocks_per_seq: int,
):
    """
    Update KT cache during context phase using sparse indices.

    Args:
        qkv_input: QKV input tensor
            [total_sparse_tokens, num_heads*head_dim + num_kv_heads*head_dim + num_kv_heads*head_dim]
        kt_cache_tensor: KT cache [num_blocks, tokens_per_block, num_kv_heads, 2*head_dim]
        kt_cache_block_offsets: Block offsets [batch_size, max_kt_blocks_per_seq]
        sparse_kv_indices: Sparse KV indices [num_kv_heads, total_sparse_tokens]
        sparse_kv_offsets: Sparse offsets [batch_size + 1]
        num_heads: Number of Q heads
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        kt_page_size: Page size for KT tokens
        prompt_budget: Prompt budget
        tokens_per_block: Tokens per cache block
        max_kt_blocks_per_seq: Maximum KT blocks per sequence
    """
    batch_size = sparse_kv_offsets.size(0) - 1
    total_sparse_tokens = sparse_kv_indices.size(1)

    total_kt_tokens = (prompt_budget + kt_page_size - 1) // kt_page_size

    BLOCK_SIZE_KT = 8
    BLOCK_SIZE_DIM = triton.next_power_of_2(head_dim)

    grid = (batch_size, num_kv_heads, (total_kt_tokens + BLOCK_SIZE_KT - 1) // BLOCK_SIZE_KT)

    rocket_update_kt_cache_ctx_kernel[grid](
        qkv_input,
        kt_cache_tensor,
        kt_cache_block_offsets,
        context_cumsum,
        sparse_kv_indices,
        sparse_kv_offsets,
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        kt_page_size,
        tokens_per_block,
        max_kt_blocks_per_seq,
        total_sparse_tokens,
        BLOCK_SIZE_KT=BLOCK_SIZE_KT,
        BLOCK_SIZE_DIM=BLOCK_SIZE_DIM,
    )


########################################################
# Paged KT cache BMM kernel
########################################################


@triton.jit
def rocket_paged_kt_cache_bmm_kernel(
    q_ptr,
    kt_cache_tensor_ptr,
    kt_cache_block_offsets_ptr,
    kv_lens_ptr,
    output_ptr,
    output_offsets_ptr,
    num_gen_tokens,
    num_kv_heads,
    num_heads_per_kv: tl.constexpr,
    head_dim,
    kt_page_size,
    tokens_per_block,
    max_kt_blocks_per_seq,
    total_kt_tokens,
    sm_scale,
    q_stride_0,
    q_stride_1,
    q_stride_2,
    q_stride_3,
    Q_BLOCK_SIZE: tl.constexpr,
    KT_BLOCK_SIZE: tl.constexpr,
    DIM_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if batch_idx >= num_gen_tokens or kv_head_idx >= num_kv_heads:
        return
    kv_len = tl.load(kv_lens_ptr + batch_idx)
    num_kt_tokens = (kv_len + kt_page_size - 1) // kt_page_size

    q_base = batch_idx * q_stride_0 + kv_head_idx * q_stride_1

    q_head_offsets = tl.arange(0, Q_BLOCK_SIZE)
    q_head_mask = q_head_offsets < num_heads_per_kv

    output_offset = tl.load(output_offsets_ptr + batch_idx)

    dim_indices = tl.arange(0, DIM_BLOCK_SIZE)
    dim_mask = dim_indices < head_dim

    q_indices = q_base + q_head_offsets[:, None] * q_stride_2 + dim_indices[None, :] * q_stride_3
    q_values = tl.load(q_ptr + q_indices, mask=q_head_mask[:, None] & dim_mask[None, :])

    dim_pos_values = tl.sum(q_values, axis=0) > 0
    dim_pos_values = tl.broadcast_to(dim_pos_values[None, :], (KT_BLOCK_SIZE, DIM_BLOCK_SIZE))

    q_values = q_values.to(kt_cache_tensor_ptr.dtype.element_ty)

    for kt_block_idx_start in tl.range(
        0,
        num_kt_tokens,
        KT_BLOCK_SIZE,
    ):
        kt_block_idx_start = tl.multiple_of(kt_block_idx_start, KT_BLOCK_SIZE)

        kt_token_indices = kt_block_idx_start + tl.arange(0, KT_BLOCK_SIZE)
        kt_token_mask = kt_token_indices < num_kt_tokens

        block_offsets = batch_idx * max_kt_blocks_per_seq + kt_token_indices // tokens_per_block

        block_indices = tl.load(
            kt_cache_block_offsets_ptr + block_offsets, mask=kt_token_mask, other=0
        )
        token_indices_in_block = kt_token_indices % tokens_per_block

        cache_bases = (
            block_indices * tokens_per_block + token_indices_in_block
        ) * num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim

        combined_mask = (
            dim_mask[None, :] & kt_token_mask[:, None]
        )  # Shape: [KT_BLOCK_SIZE, DIM_BLOCK_SIZE]

        kt_cache_indices_min = cache_bases[:, None] + dim_indices[None, :]
        kt_cache_indices_max = kt_cache_indices_min + head_dim
        kt_cache_values_min = tl.load(
            kt_cache_tensor_ptr + kt_cache_indices_min, mask=combined_mask, other=0.0
        )
        kt_cache_values_max = tl.load(
            kt_cache_tensor_ptr + kt_cache_indices_max, mask=combined_mask, other=0.0
        )

        kt_cache_values = tl.where(dim_pos_values > 0, kt_cache_values_max, kt_cache_values_min)

        results = tl.dot(q_values, kt_cache_values.T)  # [Q_BLOCK_SIZE, KT_BLOCK_SIZE]

        output_mask = q_head_mask[:, None] & kt_token_mask[None, :]
        output_indices = (
            kv_head_idx * num_heads_per_kv * total_kt_tokens
            + q_head_offsets[:, None] * total_kt_tokens
            + output_offset
            + kt_token_indices[None, :]
        )

        tl.store(output_ptr + output_indices, results * sm_scale, mask=output_mask)


def triton_rocket_paged_kt_cache_bmm(
    q: torch.Tensor,
    kt_cache_tensor: torch.Tensor,
    kt_cache_block_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    output_offsets: torch.Tensor,
    kt_page_size: int,
    tokens_per_block: int,
    max_kt_blocks_per_seq: int,
    total_kt_tokens: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Perform BMM with KT cache for generation phase.

    Args:
        q: Query tensor [num_gen_tokens, num_kv_heads, num_heads_per_kv, head_dim]
        kt_cache_tensor: KT cache tensor
        kt_cache_block_offsets: Block offsets [num_gen_tokens, max_kt_blocks_per_seq]
        kv_lens: Sequence lengths [num_gen_tokens]
        output_offsets: Output offsets [num_gen_tokens + 1]
        kt_page_size: Page size for KT tokens
        tokens_per_block: Tokens per cache block
        max_kt_blocks_per_seq: Maximum KT blocks per sequence
        total_kt_tokens: Total number of KT tokens (fixed size)
        sm_scale: Scale factor for softmax
    Returns:
        output: BMM results [num_kv_heads, num_heads_per_kv, total_kt_tokens]
    """
    num_gen_tokens, num_kv_heads, num_heads_per_kv, head_dim = q.shape
    num_kv_heads * num_heads_per_kv

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Create output tensor with shape [num_kv_heads, num_heads_per_kv, total_kt_tokens]
    output = torch.empty(
        (num_kv_heads, num_heads_per_kv, total_kt_tokens), dtype=torch.float32, device=q.device
    )

    grid = lambda meta: (num_gen_tokens, num_kv_heads)  # noqa: E731

    Q_BLOCK_SIZE = num_heads_per_kv
    KT_BLOCK_SIZE = 64
    DIM_BLOCK_SIZE = triton.next_power_of_2(head_dim)

    rocket_paged_kt_cache_bmm_kernel[grid](
        q,
        kt_cache_tensor,
        kt_cache_block_offsets,
        kv_lens,
        output,
        output_offsets,
        num_gen_tokens,
        num_kv_heads,
        num_heads_per_kv,
        head_dim,
        kt_page_size,
        tokens_per_block,
        max_kt_blocks_per_seq,
        total_kt_tokens,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KT_BLOCK_SIZE=KT_BLOCK_SIZE,
        DIM_BLOCK_SIZE=DIM_BLOCK_SIZE,
    )

    return output


########################################################
# Triton TopK kernel
########################################################


# Adapted from https://github.com/triton-lang/triton/issues/3698#issuecomment-2067681396


@triton.jit
def rocket_reduce_scores_kernel(
    input_ptr,
    output_ptr,
    cum_lens_ptr,
    batch_size,
    num_kv_heads,
    num_heads_per_kv,
    total_kt_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    if batch_idx >= batch_size or kv_head_idx >= num_kv_heads:
        return

    # Get KT token boundaries for this batch
    kt_start = tl.load(cum_lens_ptr + batch_idx)
    kt_end = tl.load(cum_lens_ptr + batch_idx + 1)
    kt_len = kt_end - kt_start

    if kt_len <= 0:
        return

    # Process KT tokens in blocks
    for kt_block_start in tl.range(0, kt_len, BLOCK_SIZE):
        kt_offsets = kt_block_start + tl.arange(0, BLOCK_SIZE)
        kt_mask = kt_offsets < kt_len
        kt_global_offsets = kt_start + kt_offsets

        # Accumulate over num_heads_per_kv
        sum_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for head_idx in range(num_heads_per_kv):
            global_head_idx = kv_head_idx * num_heads_per_kv + head_idx

            # Input indices: [global_head_idx, 1, kt_global_offsets]
            input_indices = global_head_idx * total_kt_tokens + kt_global_offsets

            values = tl.load(input_ptr + input_indices, mask=kt_mask, other=0.0)
            sum_vals += values

        # Compute mean
        mean_vals = sum_vals / num_heads_per_kv

        # Store output: [kv_head_idx, kt_global_offsets]
        output_indices = kv_head_idx * total_kt_tokens + kt_global_offsets
        tl.store(output_ptr + output_indices, mean_vals, mask=kt_mask)


def triton_rocket_reduce_scores(
    scores: torch.Tensor,
    cum_lens: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    num_heads_per_kv: int,
) -> torch.Tensor:
    """
    Reduce scores for generation phase with batch-aware processing.

    Args:
        scores: Input scores [num_kv_heads * num_heads_per_kv, 1, total_kt_tokens]
        cum_lens: Cumulative scores lengths [batch_size + 1]
        batch_size: Number of batches
        num_kv_heads: Number of KV heads
        num_heads_per_kv: Number of Q heads per KV head

    Returns:
        output: Reduced scores [num_kv_heads, total_kt_tokens]
    """
    total_kt_tokens = scores.shape[-1]

    output = torch.empty((num_kv_heads, total_kt_tokens), dtype=torch.float32, device=scores.device)

    BLOCK_SIZE = 256
    grid = (batch_size, num_kv_heads)

    rocket_reduce_scores_kernel[grid](
        scores,
        output,
        cum_lens,
        batch_size,
        num_kv_heads,
        num_heads_per_kv,
        total_kt_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


########################################################
# Convert request-local indices to global KV cache pool indices
########################################################
