import math

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like

########################################################
# Index gather kernel
########################################################


@triton.jit
def _index_gather_kernel(output_ptr, input_ptr, index_ptr, in_row_stride,
                         in_token_stride, in_head_stride, idx_row_stride,
                         idx_token_stride, idx_head_stride, dim_size,
                         BLOCK_SIZE: tl.constexpr):
    # get program id and block offset
    row_pid = tl.program_id(0)
    token_pid = tl.program_id(1)
    head_pid = tl.program_id(2)
    token_block_num = tl.num_programs(1)
    head_num = tl.num_programs(2)

    # get index
    indices_idx = row_pid * idx_row_stride + token_pid * idx_token_stride + head_pid * idx_head_stride
    token_idx = tl.load(index_ptr + indices_idx)

    # get input and output base address
    input_base = (row_pid * in_row_stride + token_idx * in_token_stride +
                  head_pid * in_head_stride)
    output_base = (row_pid * token_block_num * head_num * dim_size +
                   token_pid * head_num * dim_size + head_pid * dim_size)

    # process elements in blocks
    for dim_offset in tl.range(0, dim_size, BLOCK_SIZE):
        # get offsets
        offsets = tl.arange(0, BLOCK_SIZE)
        dim_indices = dim_offset + offsets
        mask = dim_indices < dim_size

        # load input and store output
        input_val = tl.load(input_ptr + input_base + dim_indices,
                            mask=mask,
                            other=0.0)
        tl.store(output_ptr + output_base + dim_indices, input_val, mask=mask)


def triton_index_gather(input, indices):
    assert input.ndim == 4, "Input must be a 4D tensor, [row, token, head, dim]"
    assert indices.ndim == 3, "Indices must be a 3D tensor, [row, token, head]"

    # shape of input and indices
    row_size = input.shape[0]
    head_num = input.shape[2]
    dim_size = input.shape[3]
    num_tokens = indices.shape[1]

    # create output tensor
    output = torch.empty((row_size, num_tokens, head_num, dim_size),
                         device='cuda',
                         dtype=input.dtype)

    # launch kernel
    grid = (row_size, num_tokens, head_num)
    _index_gather_kernel[grid](output,
                               input,
                               indices,
                               input.stride(0),
                               input.stride(1),
                               input.stride(2),
                               indices.stride(0),
                               indices.stride(1),
                               indices.stride(2),
                               dim_size,
                               BLOCK_SIZE=1024)
    return output


########################################################
# QK split kernel
########################################################


@triton.jit
def rocket_qk_split_kernel(input_ptr, q_output_ptr, k_output_ptr,
                           k_output_offsets_ptr, context_lens_ptr,
                           context_cumsum_ptr, valid_seq_indices_ptr,
                           window_size, num_heads, num_kv_heads, head_dim,
                           q_total_output_tokens, k_total_output_tokens,
                           valid_batch_size, BLOCK_SIZE: tl.constexpr,
                           BLOCK_SIZE_M: tl.constexpr):
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
        src_indices = src_token_pos[:, None] * (
            num_heads * head_dim +
            2 * num_kv_heads * head_dim) + input_dim_indices[None, :]

        # Calculate output indices
        dst_indices = (actual_head_idx * total_output_tokens +
                       dst_token_pos[:, None]) * head_dim + dim_indices[None, :]

        full_mask = token_mask[:, None] & dim_mask[
            None, :]  # [BLOCK_SIZE_M, BLOCK_SIZE]

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

    q_window = torch.empty((num_heads, q_total_output_tokens, head_dim),
                           device=input_tensor.device,
                           dtype=input_tensor.dtype)
    k_context = torch.empty((num_kv_heads, k_total_output_tokens, head_dim),
                            device=input_tensor.device,
                            dtype=input_tensor.dtype)

    BLOCK_SIZE = 128  # Dimension block size
    BLOCK_SIZE_M = 128  # Token block size for parallel processing

    # Grid: (valid_batch_size, num_heads + num_kv_heads, triton.cdiv(head_dim, BLOCK_SIZE))
    total_heads = num_heads + num_kv_heads
    grid = (valid_batch_size, total_heads, triton.cdiv(head_dim, BLOCK_SIZE))

    rocket_qk_split_kernel[grid](input_tensor,
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
                                 BLOCK_SIZE_M=BLOCK_SIZE_M)

    return q_window, k_context


########################################################
# BMM kernel
########################################################


@triton.jit
def bmm_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    q_cu_seqlens_ptr,
    k_cu_seqlens_ptr,
    total_q_tokens,
    total_k_tokens,
    head_dim,
    batch_size,
    num_q_heads,
    num_k_heads,
    q_len_per_seq,
    sm_scale,
    causal,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if batch_idx >= batch_size or head_idx >= num_q_heads:
        return

    # Continuous mapping of query heads to key heads
    k_head_idx = head_idx // (num_q_heads // num_k_heads)

    q_seq_start = tl.load(q_cu_seqlens_ptr + batch_idx)
    q_seq_end = tl.load(q_cu_seqlens_ptr + batch_idx + 1)
    k_seq_start = tl.load(k_cu_seqlens_ptr + batch_idx)
    k_seq_end = tl.load(k_cu_seqlens_ptr + batch_idx + 1)

    q_seqlen = q_seq_end - q_seq_start
    k_seqlen = k_seq_end - k_seq_start

    if q_seqlen <= 0 or k_seqlen <= 0:
        return

    # Process queries in this batch with BLOCK_M parallelization
    for q_block_start in tl.range(0, q_seqlen, BLOCK_M):
        q_offsets = q_block_start + tl.arange(0, BLOCK_M)
        q_mask = q_offsets < q_seqlen
        q_global_offsets = q_seq_start + q_offsets

        for k_block_start in tl.range(0, k_seqlen, BLOCK_N):
            k_offsets = k_block_start + tl.arange(0, BLOCK_N)
            k_mask = k_offsets < k_seqlen
            k_global_offsets = k_seq_start + k_offsets

            # Initialize QK^T accumulator for this (M, N) block
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            # Tiled matrix multiplication following mm_demo.py pattern
            for k_dim_start in tl.range(0, head_dim, BLOCK_K):
                k_dim_offsets = k_dim_start + tl.arange(0, BLOCK_K)
                k_dim_mask = k_dim_offsets < head_dim

                # Load query chunk [BLOCK_M, BLOCK_K]
                q_indices = head_idx * total_q_tokens * head_dim + q_global_offsets[:, None] * head_dim + k_dim_offsets[
                    None, :]
                q_chunk = tl.load(q_ptr + q_indices,
                                  mask=q_mask[:, None] & k_dim_mask[None, :],
                                  other=0.0)

                # Load key chunk [BLOCK_N, BLOCK_K]
                k_indices = k_head_idx * total_k_tokens * head_dim + k_global_offsets[:, None] * head_dim + k_dim_offsets[
                    None, :]
                k_chunk = tl.load(k_ptr + k_indices,
                                  mask=k_mask[:, None] & k_dim_mask[None, :],
                                  other=0.0)

                # Accumulate QK^T using tl.dot for better performance
                qk += tl.dot(q_chunk, tl.trans(k_chunk))

            # Scale the accumulated QK^T
            qk = qk * sm_scale

            # Apply masking
            valid_mask = q_mask[:, None] & k_mask[None, :]
            if causal:
                # Create causal mask based on positions within this batch's sequence
                q_pos_in_seq = q_offsets[:, None]  # [BLOCK_M, 1]
                k_pos_in_seq = k_offsets[None, :]  # [1, BLOCK_N]
                causal_mask = q_pos_in_seq >= k_pos_in_seq
                qk = tl.where(causal_mask & valid_mask, qk, float("-inf"))
            else:
                qk = tl.where(valid_mask, qk, float("-inf"))

            # Store results - note that we store in the global k position space
            # This matches the original bmm_softmax kernel behavior
            output_indices = (head_idx * q_len_per_seq * total_k_tokens +
                              q_offsets[:, None] * total_k_tokens +
                              k_global_offsets[None, :])  # [BLOCK_M, BLOCK_N]

            tl.store(scores_ptr + output_indices, qk, mask=valid_mask)


def triton_bmm(q: torch.Tensor,
               k: torch.Tensor,
               q_cu_seqlens: torch.Tensor,
               k_cu_seqlens: torch.Tensor,
               batch_size: int,
               sm_scale: float = None,
               causal: bool = False) -> torch.Tensor:
    """
    Compute softmax(QK^T) with flattened input and output.
    In this kernel we assume that each sequence in the batch has the same Q length.

    Args:
        q: Query tensor [num_q_heads, total_q_tokens, head_dim]
        k: Key tensor [num_kv_heads, total_k_tokens, head_dim]
        q_cu_seqlens: Query cumulative sequence lengths [batch_size + 1]
        k_cu_seqlens: Key cumulative sequence lengths [batch_size + 1]
        batch_size: Number of batches
        sm_scale: Scaling factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking

    Returns:
        scores: Attention scores [num_q_heads, q_len_per_seq, total_k_tokens]
    """
    num_q_heads, total_q_tokens, head_dim = q.shape
    num_k_heads, total_k_tokens, _ = k.shape

    assert total_q_tokens % batch_size == 0, "total_q_tokens must be divisible by batch_size"
    q_len_per_seq = total_q_tokens // batch_size

    if total_k_tokens == 0:
        return torch.zeros((num_q_heads, q_len_per_seq, total_k_tokens),
                           dtype=torch.float32,
                           device=q.device)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    bmm_results = torch.empty((num_q_heads, q_len_per_seq, total_k_tokens),
                              dtype=torch.float32,
                              device=q.device)

    # BMM kernel configuration
    BLOCK_M = 32
    BLOCK_N = 256
    BLOCK_K = 64

    grid_bmm = lambda meta: (batch_size, num_q_heads)

    bmm_kernel[grid_bmm](
        q,
        k,
        bmm_results,
        q_cu_seqlens,
        k_cu_seqlens,
        total_q_tokens,
        total_k_tokens,
        head_dim,
        batch_size,
        num_q_heads,
        num_k_heads,
        q_len_per_seq,
        sm_scale,
        causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return bmm_results


########################################################
# Softmax kernel
########################################################


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    cu_seq_lens_ptr,
    batch_size,
    num_heads,
    q_len_per_seq,
    total_k_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    head_idx = tl.program_id(0)
    q_global_idx = tl.program_id(1)

    if head_idx >= num_heads or q_global_idx >= (batch_size * q_len_per_seq):
        return

    # Determine which batch this q belongs to
    batch_idx = q_global_idx // q_len_per_seq
    q_local_idx = q_global_idx % q_len_per_seq

    if batch_idx >= batch_size:
        return

    # Get k sequence boundaries for this batch
    k_seq_start = tl.load(cu_seq_lens_ptr + batch_idx)
    k_seq_end = tl.load(cu_seq_lens_ptr + batch_idx + 1)
    k_seqlen = k_seq_end - k_seq_start

    if k_seqlen <= 0:
        return

    # Calculate input/output row start
    input_row_start = head_idx * q_len_per_seq * total_k_tokens + q_local_idx * total_k_tokens
    output_row_start = input_row_start

    # Find max value only within the valid k range for this batch
    row_max = float('-inf')
    for k_block_start in tl.range(0, k_seqlen, BLOCK_SIZE):
        k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < k_seqlen
        k_global_offsets = k_seq_start + k_offsets

        input_indices = input_row_start + k_global_offsets
        values = tl.load(input_ptr + input_indices,
                         mask=k_mask,
                         other=float('-inf'))
        block_max = tl.max(values, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # Compute sum of exp(x - max) only within valid k range
    row_sum = 0.0
    for k_block_start in tl.range(0, k_seqlen, BLOCK_SIZE):
        k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < k_seqlen
        k_global_offsets = k_seq_start + k_offsets

        input_indices = input_row_start + k_global_offsets
        values = tl.load(input_ptr + input_indices,
                         mask=k_mask,
                         other=float('-inf'))
        exp_values = tl.exp(values - row_max)
        masked_exp = tl.where(k_mask, exp_values, 0.0)
        row_sum += tl.sum(masked_exp, axis=0)

    # Apply softmax and store results
    for k_block_start in tl.range(0, k_seqlen, BLOCK_SIZE):
        k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < k_seqlen
        k_global_offsets = k_seq_start + k_offsets

        input_indices = input_row_start + k_global_offsets
        output_indices = output_row_start + k_global_offsets

        values = tl.load(input_ptr + input_indices,
                         mask=k_mask,
                         other=float('-inf'))
        exp_values = tl.exp(values - row_max)
        masked_exp = tl.where(k_mask, exp_values, 0.0)
        softmax_values = masked_exp / row_sum

        tl.store(output_ptr + output_indices, softmax_values, mask=k_mask)


def triton_softmax(
    input_tensor: torch.Tensor,
    cum_lens: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Apply softmax to flattened input tensor.

    Args:
        input_tensor: Input tensor [num_heads, len_per_seq, total_k_tokens] or [num_heads, total_k_tokens]
        cum_lens: Cumulative lengths [batch_size + 1]
        batch_size: Number of batches

    Returns:
        output: Softmax results, shape is like input_tensor
    """
    if input_tensor.ndim == 2:
        num_heads, total_k_tokens = input_tensor.shape
        len_per_seq = 1
    else:
        num_heads, len_per_seq, total_k_tokens = input_tensor.shape

    output = torch.empty_like(input_tensor,
                              dtype=input_tensor.dtype,
                              device=input_tensor.device)

    BLOCK_SIZE = 512

    grid = (num_heads, batch_size * len_per_seq)

    softmax_kernel[grid](
        input_tensor,
        output,
        cum_lens,
        batch_size,
        num_heads,
        len_per_seq,
        total_k_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


########################################################
# Reshape flatten to batched kernel
########################################################


@triton.jit
def flatten_to_batch_kernel(
    input_ptr,
    output_ptr,
    input_offsets,
    num_heads,
    total_tokens,
    padding_to_size,
    padding_value,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    k_offset = tl.load(input_offsets + batch_idx)
    context_len = tl.load(input_offsets + batch_idx + 1) - k_offset

    # Process in blocks
    for block_start in tl.range(0, padding_to_size, BLOCK_SIZE):
        pos_offsets = block_start + tl.arange(0, BLOCK_SIZE)

        pos_mask = pos_offsets < padding_to_size
        valid_mask = pos_offsets < context_len
        combined_mask = pos_mask & valid_mask

        input_indices = head_idx * total_tokens + k_offset + pos_offsets

        output_indices = batch_idx * num_heads * padding_to_size + head_idx * padding_to_size + pos_offsets

        values = tl.where(
            combined_mask,
            tl.load(input_ptr + input_indices,
                    mask=combined_mask,
                    other=padding_value), padding_value)

        tl.store(output_ptr + output_indices, values, mask=pos_mask)


def triton_flatten_to_batch(input_tensor: torch.Tensor,
                            input_offsets: torch.Tensor,
                            batch_size: int,
                            padding_to_size: int,
                            padding_value=-1e10) -> torch.Tensor:
    """
    Reshape input_tensor from [num_heads, total_tokens] to [batch_size, num_heads, padding_to_size]

    Args:
        input_tensor: Input tensor tensor [num_heads, total_tokens]
        input_offsets: Offset for each valid sequence [batch_size + 1]
        batch_size: Number of batches
        padding_to_size: Target padding size
        padding_value: Value to fill for padding

    Returns:
        batched_tensor: Output tensor [batch_size, num_heads, padding_to_size]
    """
    num_heads, total_tokens = input_tensor.shape

    batched_tensor = torch.empty((batch_size, num_heads, padding_to_size),
                                 device=input_tensor.device,
                                 dtype=input_tensor.dtype)

    grid = lambda meta: (batch_size, num_heads)
    flatten_to_batch_kernel[grid](input_tensor,
                                  batched_tensor,
                                  input_offsets,
                                  num_heads,
                                  total_tokens,
                                  padding_to_size,
                                  padding_value,
                                  BLOCK_SIZE=1024)

    return batched_tensor


########################################################
# RocketKV sparse indices batch to flatten flattening kernel
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
            prefix_values = tl.load(prefix_indices_ptr + prefix_indices,
                                    mask=token_mask,
                                    other=0)

            # Store to output
            output_indices = head_idx * total_sparse_tokens + output_offset + token_offsets
            tl.store(output_indices_ptr + output_indices,
                     prefix_values,
                     mask=token_mask)

        # Process window tokens
        for token_block_start in tl.range(0, window_size, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < window_size

            # Compute window indices: [context_len - window_size, context_len - window_size + 1, ...]
            window_values = context_len - window_size + token_offsets

            # Store to output at prefix_budget offset
            output_indices = head_idx * total_sparse_tokens + output_offset + prefix_budget + token_offsets
            tl.store(output_indices_ptr + output_indices,
                     window_values,
                     mask=token_mask)
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
            tl.store(output_indices_ptr + output_indices,
                     sequential_indices,
                     mask=token_mask)


def triton_rocket_batch_to_flatten(
        prefix_indices: torch.Tensor, input_lens: torch.Tensor,
        valid_seq_indices: torch.Tensor, output_offsets: torch.Tensor,
        batch_size: int, total_output_tokens: int, window_size: int,
        prompt_budget: int,
        num_kv_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
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
    sparse_indices = torch.empty((num_kv_heads, total_output_tokens),
                                 dtype=prefix_indices.dtype,
                                 device=prefix_indices.device)

    # Launch kernel
    BLOCK_SIZE = 512
    grid = (batch_size, num_kv_heads)

    rocket_batch_to_flatten_kernel[grid](prefix_indices,
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
                                         BLOCK_SIZE=BLOCK_SIZE)

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

    block_idx = tl.load(kt_cache_block_offsets_ptr +
                        batch_idx * max_kt_blocks_per_seq + block_offset_in_seq)
    token_idx_in_block = last_kt_token_idx % tokens_per_block

    cache_base = ((block_idx * tokens_per_block + token_idx_in_block) *
                  num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim)

    cache_min_indices = cache_base + dim_indices
    cache_max_indices = cache_base + head_dim + dim_indices

    kt_mask = dim_mask & (kt_token_idx_in_page > 0)

    k_min_existing = tl.load(kt_cache_tensor_ptr + cache_min_indices,
                             mask=kt_mask,
                             other=float('inf'))
    k_max_existing = tl.load(kt_cache_tensor_ptr + cache_max_indices,
                             mask=kt_mask,
                             other=float('-inf'))

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

    rocket_update_kt_cache_gen_kernel[grid](k,
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
                                            DIM_BLOCK_SIZE=DIM_BLOCK_SIZE)


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

        kv_token_indices = tl.load(sparse_kv_indices_ptr + kv_indices,
                                   mask=kv_mask,
                                   other=0)
        # Calculate indices for loading keys [BLOCK_SIZE_DIM, BLOCK_SIZE_KV]
        k_base_indices = (kv_token_indices[None, :] + context_start) * (
            q_hidden_size + 2 * kv_hidden_size) + k_dim_base
        k_indices = k_base_indices + dim_indices[:, None]

        combined_mask = kv_mask[None, :] & dim_mask[:, None]

        # Load key values [BLOCK_SIZE_DIM, BLOCK_SIZE_KV]
        k_values = tl.load(k_ptr + k_indices, mask=combined_mask, other=0.0)

        k_values = tl.reshape(k_values,
                              (BLOCK_SIZE_DIM, BLOCK_SIZE_KT, kt_page_size))

        k_min = tl.min(k_values,
                       axis=-1).to(kt_cache_tensor_ptr.dtype.element_ty)
        k_max = tl.max(k_values,
                       axis=-1).to(kt_cache_tensor_ptr.dtype.element_ty)

        # Calculate cache locations [BLOCK_SIZE_KT]
        block_offsets_in_seq = kt_offsets // tokens_per_block
        valid_block_mask = (block_offsets_in_seq
                            < max_kt_blocks_per_seq) & kt_mask

        # Load block indices [BLOCK_SIZE_KT]
        block_offset_addrs = batch_idx * max_kt_blocks_per_seq + block_offsets_in_seq
        block_indices = tl.load(kt_cache_block_offsets_ptr + block_offset_addrs,
                                mask=valid_block_mask,
                                other=0)

        tokens_in_block = kt_offsets % tokens_per_block

        # Calculate cache base addresses [BLOCK_SIZE_KT]
        cache_bases = ((block_indices * tokens_per_block + tokens_in_block) *
                       num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim)

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
        qkv_input: QKV input tensor [total_sparse_tokens, num_heads*head_dim + num_kv_heads*head_dim + num_kv_heads*head_dim]
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

    grid = (batch_size, num_kv_heads,
            (total_kt_tokens + BLOCK_SIZE_KT - 1) // BLOCK_SIZE_KT)

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

    q_indices = q_base + q_head_offsets[:, None] * q_stride_2 + dim_indices[
        None, :] * q_stride_3
    q_values = tl.load(q_ptr + q_indices,
                       mask=q_head_mask[:, None] & dim_mask[None, :])

    dim_pos_values = tl.sum(q_values, axis=0) > 0
    dim_pos_values = tl.broadcast_to(dim_pos_values[None, :],
                                     (KT_BLOCK_SIZE, DIM_BLOCK_SIZE))

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

        block_indices = tl.load(kt_cache_block_offsets_ptr + block_offsets,
                                mask=kt_token_mask,
                                other=0)
        token_indices_in_block = kt_token_indices % tokens_per_block

        cache_bases = (
            (block_indices * tokens_per_block + token_indices_in_block) *
            num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim)

        combined_mask = dim_mask[
            None, :] & kt_token_mask[:,
                                     None]  # Shape: [KT_BLOCK_SIZE, DIM_BLOCK_SIZE]

        kt_cache_indices_min = cache_bases[:, None] + dim_indices[None, :]
        kt_cache_indices_max = kt_cache_indices_min + head_dim
        kt_cache_values_min = tl.load(kt_cache_tensor_ptr +
                                      kt_cache_indices_min,
                                      mask=combined_mask,
                                      other=0.0)
        kt_cache_values_max = tl.load(kt_cache_tensor_ptr +
                                      kt_cache_indices_max,
                                      mask=combined_mask,
                                      other=0.0)

        kt_cache_values = tl.where(dim_pos_values > 0, kt_cache_values_max,
                                   kt_cache_values_min)

        results = tl.dot(q_values,
                         kt_cache_values.T)  # [Q_BLOCK_SIZE, KT_BLOCK_SIZE]

        output_mask = q_head_mask[:, None] & kt_token_mask[None, :]
        output_indices = (kv_head_idx * num_heads_per_kv * total_kt_tokens +
                          q_head_offsets[:, None] * total_kt_tokens +
                          output_offset + kt_token_indices[None, :])

        tl.store(output_ptr + output_indices,
                 results * sm_scale,
                 mask=output_mask)


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
    output = torch.empty((num_kv_heads, num_heads_per_kv, total_kt_tokens),
                         dtype=torch.float32,
                         device=q.device)

    grid = lambda meta: (num_gen_tokens, num_kv_heads)

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
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    # idx
    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape)
    right_idx = core.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = tl.int32
    num_bits = x.dtype.primitive_bitwidth
    if num_bits == 16:
        idtype = tl.int16

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip

    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, order: core.constexpr,
                   n_dims: core.constexpr):
    '''
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    '''
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        # Create boolean flip pattern instead of integer
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape),
            x.shape) != 0
    else:
        # Ensure flip is boolean for XOR operations
        flip = order != 0
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x,
            ids,
            dim: core.constexpr = None,
            descending: core.constexpr = core.CONSTEXPR_0):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1,
                       "only minor dimension is currently supported")
    # iteratively run bitonic merge-sort steps
    n_dims: core.constexpr = _log2(x.shape[_dim])

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending,
                                n_dims)
    return x, ids


@triton.jit
def convert_to_sortable_uint16(values):
    values_fp16 = values.to(tl.float16)
    values_u16 = values_fp16.to(tl.uint16, bitcast=True)
    # For negative numbers, flip all bits; for positive, flip sign bit
    values_u16 = tl.where(values < 0.0, ~values_u16 & 0xffff,
                          values_u16 | 0x8000)
    return values_u16


@triton.jit
def extract_bin_idx(values_u16, step: tl.constexpr):
    if step == 0:
        return 255 - (values_u16 >> 8)
    else:
        return 255 - (values_u16 & 0xff)


@triton.jit
def is_partial_match(values_u16, pattern, step: tl.constexpr,
                     BLOCK_SIZE: tl.constexpr):
    if step == 0:
        return tl.full((BLOCK_SIZE, ), True, dtype=tl.int1)
    else:
        values_high = values_u16 >> 8
        pattern_high = tl.full((BLOCK_SIZE, ), pattern >> 8, dtype=tl.uint16)
        return values_high == pattern_high


@triton.jit
def process_histogram_step(
    input_ptr,
    input_base,
    input_len,
    output_indices_ptr,
    temp_values_ptr,
    temp_indices_ptr,
    output_base,
    temp_base,
    topk,
    step: tl.constexpr,
    pattern,
    found_topk_values,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    FINAL_SIZE: tl.constexpr,
):
    """
    Process one histogram step.
    Returns: (continue_next_step, new_pattern, threshold_bin_size, found_topk_values)
    """
    if step == 0:
        pass
    else:  # step == 1
        pass

    # Build histogram
    histogram = tl.zeros((NUM_BINS, ), dtype=tl.int32)

    for i in tl.range(0, input_len, BLOCK_SIZE):
        block_offsets = i + tl.arange(0, BLOCK_SIZE)
        block_mask = block_offsets < input_len
        values = tl.load(input_ptr + input_base + block_offsets,
                         mask=block_mask,
                         other=-1e10)

        values_u16 = convert_to_sortable_uint16(values)

        # Only count values matching the pattern
        pattern_match = is_partial_match(values_u16, pattern, step, BLOCK_SIZE)

        bin_indices = extract_bin_idx(values_u16, step).to(tl.int32)

        # Apply pattern mask
        bin_indices_masked = tl.where(pattern_match & block_mask, bin_indices,
                                      NUM_BINS - 1)

        histogram += tl.histogram(bin_indices_masked, NUM_BINS)

    # Find threshold bin via cumsum
    cum_hist = tl.cumsum(histogram)

    bin_range = tl.arange(0, NUM_BINS)
    crosses_threshold = cum_hist >= topk - found_topk_values

    threshold_indices = tl.where(crosses_threshold, bin_range, NUM_BINS)
    threshold_bin_idx = tl.min(threshold_indices)

    threshold_bin_size = tl.sum(
        tl.where(bin_range == threshold_bin_idx, histogram, 0))

    if step == 0:
        new_pattern = (255 - threshold_bin_idx) << 8
    else:
        new_pattern = pattern | (255 - threshold_bin_idx)

    # Collect elements
    base_offset = found_topk_values
    final_offset = 0

    for i in tl.range(0, input_len, BLOCK_SIZE):
        block_offsets = i + tl.arange(0, BLOCK_SIZE)
        block_mask = block_offsets < input_len
        values = tl.load(input_ptr + input_base + block_offsets,
                         mask=block_mask,
                         other=-1e10)

        values_u16 = convert_to_sortable_uint16(values)

        pattern_match = is_partial_match(values_u16, pattern, step, BLOCK_SIZE)

        bin_indices = extract_bin_idx(values_u16, step).to(tl.int32)

        # Elements smaller than threshold bin are guaranteed top-k
        guaranteed_mask = (bin_indices
                           < threshold_bin_idx) & pattern_match & block_mask

        guaranteed_int = guaranteed_mask.to(tl.int32)
        write_positions = tl.cumsum(guaranteed_int, axis=0) - guaranteed_int
        num_guaranteed = tl.sum(guaranteed_int)

        guaranteed_write_offsets = base_offset + output_base + write_positions
        tl.store(output_indices_ptr + guaranteed_write_offsets,
                 block_offsets,
                 mask=guaranteed_mask)

        base_offset += num_guaranteed

        # Write candidate elements to temp storage (only if threshold bin is small)
        # Then we can directly sort the temp storage
        if threshold_bin_size <= FINAL_SIZE:
            candidate_mask = (bin_indices
                              == threshold_bin_idx) & pattern_match & block_mask
            candidate_int = candidate_mask.to(tl.int32)
            candidate_positions = tl.cumsum(candidate_int,
                                            axis=0) - candidate_int
            num_candidates = tl.sum(candidate_int)

            candidate_write_offsets = final_offset + temp_base + candidate_positions
            candidate_write_mask = (final_offset + candidate_positions
                                    < FINAL_SIZE) & candidate_mask

            tl.store(temp_values_ptr + candidate_write_offsets,
                     values,
                     mask=candidate_write_mask)
            tl.store(temp_indices_ptr + candidate_write_offsets,
                     block_offsets,
                     mask=candidate_write_mask)

            final_offset += num_candidates

    if step == 1 and threshold_bin_size > FINAL_SIZE:
        # Elements in the bin have the same logits
        for i in tl.range(0, input_len, BLOCK_SIZE):
            block_offsets = i + tl.arange(0, BLOCK_SIZE)
            block_mask = block_offsets < input_len
            values = tl.load(input_ptr + input_base + block_offsets,
                             mask=block_mask,
                             other=-1e10)

            values_u16 = convert_to_sortable_uint16(values)

            pattern_match = is_partial_match(values_u16, pattern, step,
                                             BLOCK_SIZE)

            bin_indices = extract_bin_idx(values_u16, step).to(tl.int32)

            candidate_mask = (bin_indices
                              == threshold_bin_idx) & pattern_match & block_mask

            candidate_int = candidate_mask.to(tl.int32)
            candidate_positions = tl.cumsum(candidate_int,
                                            axis=0) - candidate_int
            num_candidates = tl.sum(candidate_int)

            candidate_write_offsets = base_offset + output_base + candidate_positions
            candidate_write_mask = (base_offset + candidate_positions
                                    < topk) & candidate_mask

            tl.store(output_indices_ptr + candidate_write_offsets,
                     block_offsets,
                     mask=candidate_write_mask)

            base_offset += num_candidates

    # Decide if we need next step
    continue_next_step = threshold_bin_size > FINAL_SIZE

    return continue_next_step, new_pattern, threshold_bin_size, base_offset, final_offset


@triton.jit
def topk_kernel(
    input_ptr,
    output_indices_ptr,
    temp_values_ptr,
    temp_indices_ptr,
    input_offsets_ptr,
    output_offsets_ptr,
    batch_size,
    num_kv_heads,
    topk,
    total_input_tokens,
    total_sparse_indices,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    FINAL_SIZE: tl.constexpr,
):
    """
    Two-stage histogram-based top-k kernel.
    Stage 0: Process high 8 bits (256 bins)
    Stage 1: Process low 8 bits (256 bins)
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if batch_idx >= batch_size or head_idx >= num_kv_heads:
        return

    input_start = tl.load(input_offsets_ptr + batch_idx)
    input_end = tl.load(input_offsets_ptr + batch_idx + 1)
    input_len = input_end - input_start

    output_start = tl.load(output_offsets_ptr + batch_idx)
    output_end = tl.load(output_offsets_ptr + batch_idx + 1)
    output_len = output_end - output_start

    if input_len <= 0 or output_len <= 0:
        return

    input_base = head_idx * total_input_tokens + input_start
    temp_base = head_idx * batch_size * FINAL_SIZE + batch_idx * FINAL_SIZE
    output_base = head_idx * total_sparse_indices + output_start

    if input_len <= topk:
        for i in tl.range(0, input_len, BLOCK_SIZE):
            block_offsets = i + tl.arange(0, BLOCK_SIZE)
            block_mask = block_offsets < input_len
            tl.store(output_indices_ptr + output_base + block_offsets,
                     block_offsets,
                     mask=block_mask)
        return

    pattern = 0
    found_topk_values = 0

    continue_step, pattern, threshold_bin_size, found_topk_values, final_offset = \
        process_histogram_step(
            input_ptr, input_base, input_len,
            output_indices_ptr, temp_values_ptr, temp_indices_ptr,
            output_base, temp_base, topk,
            step=0,
            pattern=pattern,
            found_topk_values=found_topk_values,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_BINS=NUM_BINS,
            FINAL_SIZE=FINAL_SIZE,
        )

    if continue_step:
        continue_step, pattern, threshold_bin_size, found_topk_values, final_offset = \
            process_histogram_step(
                input_ptr, input_base, input_len,
                output_indices_ptr, temp_values_ptr, temp_indices_ptr,
                output_base, temp_base, topk,
                step=1,
                pattern=pattern,
                found_topk_values=found_topk_values,
                BLOCK_SIZE=BLOCK_SIZE,
                NUM_BINS=NUM_BINS,
                FINAL_SIZE=FINAL_SIZE,
            )

    # Sort the final candidates
    if not continue_step and final_offset > 0:
        final_offsets = tl.arange(0, FINAL_SIZE)
        final_mask = final_offsets < final_offset
        final_values = tl.load(temp_values_ptr + temp_base + final_offsets,
                               mask=final_mask,
                               other=-1e10)
        final_indices = tl.load(temp_indices_ptr + temp_base + final_offsets,
                                mask=final_mask,
                                other=-1)

        _, final_sorted_indices = argsort(final_values,
                                          final_indices,
                                          dim=0,
                                          descending=True)

        remain_num = topk - found_topk_values

        write_offsets = tl.arange(0, FINAL_SIZE)
        write_mask = write_offsets < remain_num

        tl.store(output_indices_ptr + output_base + found_topk_values +
                 write_offsets,
                 final_sorted_indices,
                 mask=write_mask)


def triton_topk(input_tensor: torch.Tensor, input_offsets: torch.Tensor,
                output_offsets: torch.Tensor, total_output_tokens: int,
                topk: int) -> torch.Tensor:
    """
    Perform topk operation on input tensor using two-stage histogram algorithm.
    Args:
        input_tensor: Input scores [num_kv_heads, total_tokens]
        input_offsets: Input offsets [batch_size + 1]
        output_offsets: Sparse offsets [batch_size + 1]
        total_output_tokens: Total number of output tokens
        topk: TopK parameter

    Returns:
        output_indices: Selected indices [num_kv_heads, total_output_tokens]
    """

    num_kv_heads, total_input_tokens = input_tensor.shape
    batch_size = output_offsets.shape[0] - 1
    device = input_tensor.device

    # Create output tensor
    output_indices = torch.empty((num_kv_heads, total_output_tokens),
                                 dtype=torch.int32,
                                 device=device)

    grid = (batch_size, num_kv_heads)

    BLOCK_SIZE = 256
    NUM_BINS = 256
    FINAL_SIZE = 256

    temp_values = torch.empty((num_kv_heads, batch_size, FINAL_SIZE),
                              dtype=torch.float32,
                              device=device)
    temp_indices = torch.empty((num_kv_heads, batch_size, FINAL_SIZE),
                               dtype=torch.int32,
                               device=device)

    topk_kernel[grid](
        input_tensor,
        output_indices,
        temp_values,
        temp_indices,
        input_offsets,
        output_offsets,
        batch_size,
        num_kv_heads,
        topk,
        total_input_tokens,
        total_output_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_BINS=NUM_BINS,
        FINAL_SIZE=FINAL_SIZE,
    )

    return output_indices


########################################################
# Reduce scores generation kernel
########################################################


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

    output = torch.empty((num_kv_heads, total_kt_tokens),
                         dtype=torch.float32,
                         device=scores.device)

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


@triton.jit
def _convert_req_index_to_global_index_kernel_with_stride_factor(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    # shapes (compile-time where possible)
    max_num_blocks_per_req: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns # strides (in elements)
    stride_factor: tl.constexpr,  # for strided memory layout adjustment
    layer_id: tl.constexpr,  # for layer interleaving layout
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    """
    Triton kernel for converting request-local token indices to global KV cache pool indices.
    Derived from vllm's flashmla_sparse.py, with stride_factor fused in the kernel.
    """
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    is_invalid_tok = tok < 0

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE + layer_id * BLOCK_SIZE

    # Guard block_table access
    valid_block = block_id < max_num_blocks_per_req
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # If token == -1 OR block_id OOB, output -1
    # Otherwise: base * stride_factor + inblock_off
    # (stride_factor accounts for layer interleaving in strided KV cache pools)
    out_val = tl.where(is_invalid_tok | (~valid_block), -1,
                       base * stride_factor + inblock_off)

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def triton_convert_req_index_to_global_index(
        req_id: torch.Tensor,  # int32 [num_tokens]
        block_table: torch.
    Tensor,  # int32 [num_requests, max_num_blocks_per_req]
        token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
        BLOCK_SIZE: int,
        NUM_TOPK_TOKENS: int = 2048,
        BLOCK_N: int = 128,  # tile width along columns
        stride_factor:
    int = None,  # for strided memory layout (with layer interleaving), defaults to BLOCK_SIZE
        layer_id: int = 0,  # for layer interleaving layout
):
    """
    Convert request-local token indices to global KV cache pool indices.

    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * stride_factor
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Args:
        stride_factor: Memory stride between consecutive blocks (default: BLOCK_SIZE).
                        For non-contiguous pools with layer interleaving, use
                        (num_layers * BLOCK_SIZE) to account for memory gaps.

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    if stride_factor is None:
        stride_factor = BLOCK_SIZE
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert NUM_TOPK_TOKENS % BLOCK_N == 0, \
        f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible by" \
        f"BLOCK_N ({BLOCK_N})"

    num_tokens = req_id.shape[0]
    num_requests, max_num_blocks_per_req = block_table.shape
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_index_kernel_with_stride_factor[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        # shapes / constexprs
        max_num_blocks_per_req,
        BLOCK_SIZE,
        BLOCK_N,
        stride_factor,
        # strides
        layer_id,
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out
