import math

import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like

########################################################
# Argsort utilities for topk operations
# Adapted from https://github.com/triton-lang/triton/issues/3698#issuecomment-2067681396
########################################################


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
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth,
                                signed=True)
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
                           context_cumsum_ptr, valid_seq_indices_ptr,
                           q_extract_start_offsets_ptr, q_extract_lengths_ptr,
                           q_output_offsets_ptr, k_extract_start_offsets_ptr,
                           k_extract_lengths_ptr, k_output_offsets_ptr,
                           num_heads, num_kv_heads, head_dim, total_tokens,
                           q_total_output_tokens, k_total_output_tokens,
                           valid_batch_size, q_start_offset, k_start_offset,
                           BLOCK_SIZE: tl.constexpr,
                           BLOCK_SIZE_M: tl.constexpr):
    valid_seq_idx = tl.program_id(0)  # Which valid sequence
    head_idx = tl.program_id(1)  # Which head
    dim_offset = tl.program_id(2) * BLOCK_SIZE  # Which dimension block

    if valid_seq_idx >= valid_batch_size:
        return

    # Determine if this is a Q head or K head based on head_idx
    is_q_head = head_idx < num_heads
    is_k_head = head_idx >= num_heads and head_idx < (num_heads + num_kv_heads)

    if not (is_q_head or is_k_head):
        return

    # Get original sequence index
    orig_seq_idx = tl.load(valid_seq_indices_ptr + valid_seq_idx)

    # Get sequence start offset in original tensor
    seq_start_offset = tl.load(context_cumsum_ptr + orig_seq_idx)

    if is_q_head:
        # Process Q head
        actual_head_idx = head_idx
        extract_start_offset = tl.load(q_extract_start_offsets_ptr +
                                       valid_seq_idx)
        extract_length = tl.load(q_extract_lengths_ptr + valid_seq_idx)
        output_offset = tl.load(q_output_offsets_ptr + valid_seq_idx)
        output_ptr = q_output_ptr
        total_output_tokens = q_total_output_tokens
        input_dim_offset = q_start_offset
    else:
        # Process K head
        actual_head_idx = head_idx - num_heads
        extract_start_offset = tl.load(k_extract_start_offsets_ptr +
                                       valid_seq_idx)
        extract_length = tl.load(k_extract_lengths_ptr + valid_seq_idx)
        output_offset = tl.load(k_output_offsets_ptr + valid_seq_idx)
        output_ptr = k_output_ptr
        total_output_tokens = k_total_output_tokens
        input_dim_offset = k_start_offset

    # Calculate dimension indices and mask
    dim_indices = dim_offset + tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_indices < head_dim

    # Use tl.range for dynamic loop over token blocks
    for token_block_start in tl.range(0, extract_length, BLOCK_SIZE_M):
        # Generate token indices for current block
        token_indices = token_block_start + tl.arange(0, BLOCK_SIZE_M)
        token_mask = token_indices < extract_length

        # Calculate source and destination positions for current token block
        src_token_pos = seq_start_offset + extract_start_offset + token_indices  # [BLOCK_SIZE_M]
        dst_token_pos = output_offset + token_indices  # [BLOCK_SIZE_M]

        # Calculate input indices with proper dimension offset for Q/K separation
        # Input tensor: [total_tokens, num_heads*head_dim + num_kv_heads*head_dim + num_kv_heads*head_dim]
        input_dim_indices = input_dim_offset + actual_head_idx * head_dim + dim_indices
        src_indices = src_token_pos[:, None] * (
            num_heads * head_dim +
            2 * num_kv_heads * head_dim) + input_dim_indices[None, :]

        # Calculate output indices
        # Output tensor: [num_heads/num_kv_heads, total_output_tokens, head_dim]
        dst_indices = (actual_head_idx * total_output_tokens +
                       dst_token_pos[:, None]) * head_dim + dim_indices[None, :]

        # Create 2D mask combining token and dimension masks
        full_mask = token_mask[:, None] & dim_mask[
            None, :]  # [BLOCK_SIZE_M, BLOCK_SIZE]

        # Parallel load and store for current token block
        data = tl.load(input_ptr + src_indices, mask=full_mask, other=0.0)
        tl.store(output_ptr + dst_indices, data, mask=full_mask)


def triton_rocket_qk_split(
        input_tensor: torch.Tensor, num_heads: int, num_kv_heads: int,
        head_dim: int, window_size: int, prompt_budget: int,
        context_cumsum: torch.Tensor, valid_batch_size: int,
        valid_seq_indices: torch.Tensor, q_extract_start_offsets: torch.Tensor,
        q_extract_lengths: torch.Tensor, q_output_offsets: torch.Tensor,
        k_extract_start_offsets: torch.Tensor, k_extract_lengths: torch.Tensor,
        k_output_offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RocketKV QK split with unified kernel parallelism for optimal performance.
    This kernel is used to split the input tensor into window queries and context keys.

    Returns:
        q_window: Window queries [num_heads, window_size * valid_batch_size, head_dim]
        k_context: Context keys [num_kv_heads, sum(valid_context_lens), head_dim]
    """
    total_tokens = input_tensor.shape[0]
    total_k_context_tokens = k_extract_lengths[:valid_batch_size].sum().item()

    q_window = torch.empty(
        (num_heads, window_size * valid_batch_size, head_dim),
        device=input_tensor.device,
        dtype=input_tensor.dtype)
    k_context = torch.empty((num_kv_heads, total_k_context_tokens, head_dim),
                            device=input_tensor.device,
                            dtype=input_tensor.dtype)

    q_start_offset = 0
    k_start_offset = num_heads * head_dim

    BLOCK_SIZE = 128  # Dimension block size
    BLOCK_SIZE_M = 128  # Token block size for parallel processing

    # Grid: (valid_batch_size, num_heads + num_kv_heads, triton.cdiv(head_dim, BLOCK_SIZE))
    total_heads = num_heads + num_kv_heads
    grid = (valid_batch_size, total_heads, triton.cdiv(head_dim, BLOCK_SIZE))

    rocket_qk_split_kernel[grid](input_tensor,
                                 q_window,
                                 k_context,
                                 context_cumsum,
                                 valid_seq_indices,
                                 q_extract_start_offsets,
                                 q_extract_lengths,
                                 q_output_offsets,
                                 k_extract_start_offsets,
                                 k_extract_lengths,
                                 k_output_offsets,
                                 num_heads,
                                 num_kv_heads,
                                 head_dim,
                                 total_tokens,
                                 window_size * valid_batch_size,
                                 total_k_context_tokens,
                                 valid_batch_size,
                                 q_start_offset,
                                 k_start_offset,
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
        softmax_values = tl.exp(values - row_max) / row_sum

        tl.store(output_ptr + output_indices, softmax_values, mask=k_mask)


def triton_softmax(
    input_tensor: torch.Tensor,
    cum_lens: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """
    Apply softmax to KT token scores with batch-aware sequence boundaries.

    Args:
        input_tensor: Input tensor [total_num_heads, 1, total_kt_tokens]
        cum_lens: Cumulative lengths [batch_size + 1]
        batch_size: Number of generation batches

    Returns:
        output: Softmax results [total_num_heads, 1, total_kt_tokens]
    """
    total_num_heads, q_len_per_seq, total_kt_tokens = input_tensor.shape

    output = torch.empty_like(input_tensor,
                              dtype=torch.float32,
                              device=input_tensor.device)

    BLOCK_SIZE = 1024

    grid = (total_num_heads, batch_size * q_len_per_seq)

    softmax_kernel[grid](
        input_tensor,
        output,
        cum_lens,
        batch_size,
        total_num_heads,
        q_len_per_seq,
        total_kt_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


########################################################
# Reshape flatten to batched kernel
########################################################


@triton.jit
def flatten_to_batched_kernel(
    input_ptr,
    output_ptr,
    context_lens,
    cu_context_lens,
    num_heads,
    total_tokens,
    padding_size,
    padding_value,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    context_len = tl.load(context_lens + batch_idx)
    k_offset = tl.load(cu_context_lens + batch_idx)

    # Process in blocks
    for block_start in tl.range(0, padding_size, BLOCK_SIZE):
        pos_offsets = block_start + tl.arange(0, BLOCK_SIZE)

        pos_mask = pos_offsets < padding_size
        valid_mask = pos_offsets < context_len
        combined_mask = pos_mask & valid_mask

        input_indices = head_idx * total_tokens + k_offset + pos_offsets

        output_indices = batch_idx * num_heads * padding_size + head_idx * padding_size + pos_offsets

        values = tl.where(
            combined_mask,
            tl.load(input_ptr + input_indices,
                    mask=combined_mask,
                    other=padding_value), padding_value)

        tl.store(output_ptr + output_indices, values, mask=pos_mask)


def triton_flatten_to_batched(input_tensor: torch.Tensor,
                              context_lens: torch.Tensor,
                              cu_context_lens: torch.Tensor,
                              batch_size: int,
                              padding_size: int,
                              padding_value=-1e10) -> torch.Tensor:
    """
    Reshape input_tensor from [num_heads, total_tokens] to [batch_size, num_heads, padding_size]

    Args:
        input_tensor: Input tensor tensor [num_heads, total_tokens]
        context_lens: List of context lengths for each batch
        cu_context_lens: Cumulative sum of context lengths [batch_size + 1]
        batch_size: Number of batches
        padding_size: Target padding size
        padding_value: Value to fill for padding

    Returns:
        batched_tensor: Output tensor [batch_size, num_heads, padding_size]
    """
    num_heads, total_tokens = input_tensor.shape

    batched_tensor = torch.empty((batch_size, num_heads, padding_size),
                                 device=input_tensor.device,
                                 dtype=input_tensor.dtype)

    grid = lambda meta: (batch_size, num_heads)
    flatten_to_batched_kernel[grid](input_tensor,
                                    batched_tensor,
                                    context_lens,
                                    cu_context_lens,
                                    num_heads,
                                    total_tokens,
                                    padding_size,
                                    padding_value,
                                    BLOCK_SIZE=1024)

    return batched_tensor


########################################################
# Sparse indices flattening kernel
########################################################


@triton.jit
def flatten_sparse_indices_kernel(
    prefix_indices_ptr,
    context_lens_ptr,
    valid_seq_indices_ptr,
    k_context_lens_ptr,
    sparse_indices_ptr,
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
        # Get the context length for this valid sequence
        k_context_len = tl.load(k_context_lens_ptr + valid_idx_in_selected)

        # Process prefix tokens
        for token_block_start in tl.range(0, prefix_budget, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < prefix_budget

            # Load from prefix_indices
            prefix_indices = valid_idx_in_selected * num_kv_heads * prefix_budget + head_idx * prefix_budget + token_offsets
            prefix_values = tl.load(prefix_indices_ptr + prefix_indices,
                                    mask=token_mask,
                                    other=0)

            # Store to output
            output_indices = head_idx * total_sparse_tokens + output_offset + token_offsets
            tl.store(sparse_indices_ptr + output_indices,
                     prefix_values,
                     mask=token_mask)

        # Process window tokens
        for token_block_start in tl.range(0, window_size, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < window_size

            # Compute window indices: [context_len - window_size, context_len - window_size + 1, ...]
            window_values = k_context_len + token_offsets

            # Store to output at prefix_budget offset
            output_indices = head_idx * total_sparse_tokens + output_offset + prefix_budget + token_offsets
            tl.store(sparse_indices_ptr + output_indices,
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
            tl.store(sparse_indices_ptr + output_indices,
                     sequential_indices,
                     mask=token_mask)


def triton_flatten_sparse_indices(
        prefix_indices: torch.Tensor, context_lens: torch.Tensor,
        valid_seq_indices: torch.Tensor, k_context_lens: torch.Tensor,
        sparse_offsets: torch.Tensor, batch_size: int, total_sparse_tokens: int,
        window_size: int,
        prompt_budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten sparse indices considering both valid and invalid batches.
    For valid sequences, combines prefix_indices with dynamically computed window indices.

    Args:
        prefix_indices: Selected prefix indices [valid_batch_size, num_kv_heads, prefix_budget]
        context_lens: Context lengths for all sequences [batch_size]
        valid_seq_indices: Valid sequence indices [valid_batch_size]
        k_context_lens: Context lengths for valid sequences [valid_batch_size]
        sparse_offsets: Offset for each batch [batch_size + 1]
        batch_size: Number of batches
        total_sparse_tokens: Total number of sparse tokens
        window_size: Size of sliding window at the end
        prompt_budget: Total number of tokens for valid sequences (prefix_budget + window_size)

    Returns:
        sparse_indices: Flattened sparse indices [num_kv_heads, total_sparse_tokens]
    """
    valid_batch_size, num_kv_heads, prefix_budget = prefix_indices.shape

    # Create output tensor
    sparse_indices = torch.empty((num_kv_heads, total_sparse_tokens),
                                 dtype=prefix_indices.dtype,
                                 device=prefix_indices.device)

    # Launch kernel
    BLOCK_SIZE = 512
    grid = (batch_size, num_kv_heads)

    flatten_sparse_indices_kernel[grid](prefix_indices,
                                        context_lens,
                                        valid_seq_indices,
                                        k_context_lens,
                                        sparse_indices,
                                        sparse_offsets,
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
def kt_cache_update_kernel(
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

    k_base = batch_idx * num_kv_heads * head_dim + kv_head_idx * head_dim
    k_indices = k_base + dim_indices
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


@triton.jit
def kt_cache_update_ctx_kernel(
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
    kt_page_size,
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

    if batch_idx >= batch_size or kv_head_idx >= num_kv_heads:
        return

    context_start = tl.load(context_cumsum_ptr + batch_idx)

    sparse_start = tl.load(sparse_kv_offsets_ptr + batch_idx)
    sparse_end = tl.load(sparse_kv_offsets_ptr + batch_idx + 1)
    num_sparse_tokens = sparse_end - sparse_start

    if num_sparse_tokens <= 0:
        return

    # Calculate number of kt_tokens for this batch
    num_kt_tokens = (num_sparse_tokens + kt_page_size - 1) // kt_page_size

    q_hidden_size = num_heads * head_dim
    kv_hidden_size = num_kv_heads * head_dim
    k_dim_base = q_hidden_size + kv_head_idx * head_dim

    # Process kt_tokens and dimensions in blocks
    for kt_block_start in tl.range(0, num_kt_tokens, BLOCK_SIZE_KT):
        # Get kt_token indices for this block [BLOCK_SIZE_KT]
        kt_offsets = kt_block_start + tl.arange(0, BLOCK_SIZE_KT)
        kt_mask = kt_offsets < num_kt_tokens

        # Calculate page boundaries for all kt_tokens in this block
        page_starts = sparse_start + kt_offsets * kt_page_size
        page_ends = tl.minimum(page_starts + kt_page_size, sparse_end)

        for dim_block_start in tl.range(0, head_dim, BLOCK_SIZE_DIM):
            dim_offsets = tl.arange(0, BLOCK_SIZE_DIM)
            dim_indices = dim_block_start + dim_offsets
            dim_mask = dim_indices < head_dim

            k_min = tl.full((BLOCK_SIZE_KT, BLOCK_SIZE_DIM),
                            float('inf'),
                            dtype=tl.float32)
            k_max = tl.full((BLOCK_SIZE_KT, BLOCK_SIZE_DIM),
                            float('-inf'),
                            dtype=tl.float32)

            # Iterate through all tokens in the page
            for page_offset in range(kt_page_size):
                # Calculate token indices within sparse range [BLOCK_SIZE_KT]
                token_indices = page_starts + page_offset
                token_mask = (token_indices < page_ends) & kt_mask

                # Load sparse indices for all valid tokens [BLOCK_SIZE_KT]
                sparse_idx_offsets = kv_head_idx * total_sparse_tokens + token_indices
                kv_token_indices = tl.load(sparse_kv_indices_ptr +
                                           sparse_idx_offsets,
                                           mask=token_mask,
                                           other=0)

                # Broadcast for 2D operations [BLOCK_SIZE_KT, BLOCK_SIZE_DIM]
                valid_mask_2d = token_mask[:, None] & dim_mask[None, :]

                # Calculate indices for loading keys [BLOCK_SIZE_KT, BLOCK_SIZE_DIM]
                k_base_indices = (kv_token_indices[:, None] + context_start) * (
                    q_hidden_size + 2 * kv_hidden_size) + k_dim_base
                k_indices = k_base_indices + dim_indices[None, :]

                # Load key values [BLOCK_SIZE_KT, BLOCK_SIZE_DIM]
                k_values = tl.load(k_ptr + k_indices,
                                   mask=valid_mask_2d,
                                   other=0.0)

                k_min = tl.where(valid_mask_2d, tl.minimum(k_min, k_values),
                                 k_min)
                k_max = tl.where(valid_mask_2d, tl.maximum(k_max, k_values),
                                 k_max)

            k_min = k_min.to(kt_cache_tensor_ptr.dtype.element_ty)
            k_max = k_max.to(kt_cache_tensor_ptr.dtype.element_ty)

            # Calculate cache locations [BLOCK_SIZE_KT]
            block_offsets_in_seq = kt_offsets // tokens_per_block
            valid_block_mask = (block_offsets_in_seq
                                < max_kt_blocks_per_seq) & kt_mask

            # Load block indices [BLOCK_SIZE_KT]
            block_offset_addrs = batch_idx * max_kt_blocks_per_seq + block_offsets_in_seq
            block_indices = tl.load(kt_cache_block_offsets_ptr +
                                    block_offset_addrs,
                                    mask=valid_block_mask,
                                    other=0)

            tokens_in_block = kt_offsets % tokens_per_block

            # Calculate cache base addresses [BLOCK_SIZE_KT]
            cache_bases = (
                (block_indices * tokens_per_block + tokens_in_block) *
                num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim)

            cache_min_addrs = cache_bases[:, None] + dim_indices[None, :]
            cache_max_addrs = cache_bases[:, None] + head_dim + dim_indices[
                None, :]

            store_mask = valid_block_mask[:, None] & dim_mask[None, :]

            tl.store(kt_cache_tensor_ptr + cache_min_addrs,
                     k_min,
                     mask=store_mask)
            tl.store(kt_cache_tensor_ptr + cache_max_addrs,
                     k_max,
                     mask=store_mask)


def triton_update_kt_cache_ctx(
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
        tokens_per_block: Tokens per cache block
        max_kt_blocks_per_seq: Maximum KT blocks per sequence
    """
    batch_size = sparse_kv_offsets.size(0) - 1
    total_sparse_tokens = sparse_kv_indices.size(1)

    BLOCK_SIZE_KT = 64
    BLOCK_SIZE_DIM = 128

    grid = (batch_size, num_kv_heads)

    kt_cache_update_ctx_kernel[grid](
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


# TODO: Bottleneck of inference here. Need to optimize.
@triton.autotune(
    configs=[
        triton.Config({
            'KT_BLOCK_SIZE': 32,
            'DIM_BLOCK_SIZE': 16
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 32,
            'DIM_BLOCK_SIZE': 32
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 32,
            'DIM_BLOCK_SIZE': 64
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 64,
            'DIM_BLOCK_SIZE': 16
        },
                      num_warps=2,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 64,
            'DIM_BLOCK_SIZE': 32
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 64,
            'DIM_BLOCK_SIZE': 64
        },
                      num_warps=4,
                      num_stages=3),
        triton.Config({
            'KT_BLOCK_SIZE': 64,
            'DIM_BLOCK_SIZE': 64
        },
                      num_warps=8,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 128,
            'DIM_BLOCK_SIZE': 16
        },
                      num_warps=4,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 128,
            'DIM_BLOCK_SIZE': 32
        },
                      num_warps=4,
                      num_stages=3),
        triton.Config({
            'KT_BLOCK_SIZE': 128,
            'DIM_BLOCK_SIZE': 32
        },
                      num_warps=8,
                      num_stages=2),
        triton.Config({
            'KT_BLOCK_SIZE': 128,
            'DIM_BLOCK_SIZE': 64
        },
                      num_warps=8,
                      num_stages=3),
        triton.Config({
            'KT_BLOCK_SIZE': 128,
            'DIM_BLOCK_SIZE': 64
        },
                      num_warps=8,
                      num_stages=4),
    ],
    key=['max_num_kt_tokens', 'head_dim'],
    use_cuda_graph=True,
)
@triton.jit
def paged_kt_cache_bmm_kernel(
    q_ptr,
    kt_cache_tensor_ptr,
    kt_cache_block_offsets_ptr,
    dim_pos_ptr,
    kv_lens_ptr,
    output_ptr,
    output_offsets_ptr,
    num_gen_tokens,
    num_kv_heads,
    num_heads_per_kv,
    head_dim,
    kt_page_size,
    tokens_per_block,
    max_kt_blocks_per_seq,
    max_num_kt_tokens,
    total_kt_tokens,
    sm_scale,
    KT_BLOCK_SIZE: tl.constexpr,
    DIM_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kt_block_idx = tl.program_id(1)
    global_head_idx = tl.program_id(2)

    total_num_heads = num_kv_heads * num_heads_per_kv

    if batch_idx >= num_gen_tokens or global_head_idx >= total_num_heads:
        return

    kv_head_idx = global_head_idx // num_heads_per_kv
    q_head_idx = global_head_idx % num_heads_per_kv

    kv_len = tl.load(kv_lens_ptr + batch_idx)
    num_kt_tokens = (kv_len + kt_page_size - 1) // kt_page_size

    if num_kt_tokens <= 0:
        return

    kt_token_start = kt_block_idx * KT_BLOCK_SIZE

    kt_token_offsets = tl.arange(0, KT_BLOCK_SIZE)
    kt_token_indices = kt_token_start + kt_token_offsets
    kt_token_mask = kt_token_indices < num_kt_tokens

    q_base = (batch_idx * num_kv_heads * num_heads_per_kv * head_dim +
              kv_head_idx * num_heads_per_kv * head_dim + q_head_idx * head_dim)
    dim_pos_base = (batch_idx * num_kv_heads * head_dim +
                    kv_head_idx * head_dim)

    block_offsets = batch_idx * max_kt_blocks_per_seq + kt_token_indices // tokens_per_block
    block_indices = tl.load(kt_cache_block_offsets_ptr + block_offsets,
                            mask=kt_token_mask,
                            other=0)
    token_indices_in_block = kt_token_indices % tokens_per_block

    cache_bases = ((block_indices * tokens_per_block + token_indices_in_block) *
                   num_kv_heads * 2 * head_dim + kv_head_idx * 2 * head_dim)

    results = tl.zeros([KT_BLOCK_SIZE], dtype=tl.float32)

    for dim_block_start in tl.range(0, head_dim, DIM_BLOCK_SIZE):
        dim_offsets = tl.arange(0, DIM_BLOCK_SIZE)
        dim_indices = dim_block_start + dim_offsets
        dim_mask = dim_indices < head_dim

        q_indices = q_base + dim_indices
        q_values = tl.load(q_ptr + q_indices, mask=dim_mask, other=0.0)

        dim_pos_indices = dim_pos_base + dim_indices
        kt_cache_offsets = tl.load(dim_pos_ptr + dim_pos_indices,
                                   mask=dim_mask,
                                   other=0)

        dim_indices_expanded = dim_indices[:,
                                           None]  # Shape: [DIM_BLOCK_SIZE, 1]
        cache_bases_expanded = cache_bases[None, :]  # Shape: [1, KT_BLOCK_SIZE]
        dim_mask_expanded = dim_mask[:, None]  # Shape: [DIM_BLOCK_SIZE, 1]

        kt_cache_offsets_expanded = kt_cache_offsets[:,
                                                     None]  # Shape: [DIM_BLOCK_SIZE, 1]
        kt_cache_indices = cache_bases_expanded + kt_cache_offsets_expanded + dim_indices_expanded  # Shape: [DIM_BLOCK_SIZE, KT_BLOCK_SIZE]

        kt_token_mask_expanded = kt_token_mask[
            None, :]  # Shape: [1, KT_BLOCK_SIZE]
        combined_mask = dim_mask_expanded & kt_token_mask_expanded  # Shape: [DIM_BLOCK_SIZE, KT_BLOCK_SIZE]

        kt_cache_flat = tl.reshape(kt_cache_indices,
                                   [DIM_BLOCK_SIZE * KT_BLOCK_SIZE])
        mask_flat = tl.reshape(combined_mask, [DIM_BLOCK_SIZE * KT_BLOCK_SIZE])

        kt_values_flat = tl.load(kt_cache_tensor_ptr + kt_cache_flat,
                                 mask=mask_flat,
                                 other=0.0)
        kt_values = tl.reshape(kt_values_flat, [DIM_BLOCK_SIZE, KT_BLOCK_SIZE])

        q_values_expanded = q_values[:, None]  # Shape: [DIM_BLOCK_SIZE, 1]
        products = q_values_expanded * kt_values  # Shape: [DIM_BLOCK_SIZE, KT_BLOCK_SIZE]
        masked_products = tl.where(combined_mask, products, 0.0)

        results += tl.sum(masked_products, axis=0)  # Shape: [KT_BLOCK_SIZE]

    output_offset = tl.load(output_offsets_ptr + batch_idx)
    output_indices = global_head_idx * total_kt_tokens + output_offset + kt_token_indices
    tl.store(output_ptr + output_indices,
             results * sm_scale,
             mask=kt_token_mask)


def triton_kt_cache_update_and_bmm(
    q: torch.Tensor,
    k: torch.Tensor,
    dim_pos: torch.Tensor,
    kv_lens: torch.Tensor,
    kt_page_size: int,
    tokens_per_block: int,
    max_kt_blocks_per_seq: int,
    num_kt_tokens: torch.Tensor,
    total_kt_tokens: int,
    kt_cache_tensor: torch.Tensor,
    kt_cache_block_offsets: torch.Tensor,
    output_offsets: torch.Tensor,
    max_real_kt_tokens: int,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Separated KT cache update and BMM computation for generation phase.

    This function first updates the KT cache with new key values, then performs
    the matrix multiplication with the cached values.

    Args:
        q: Query tensor [num_gen_tokens, num_kv_heads, num_heads_per_kv, head_dim]
        k: Key tensor [num_gen_tokens, num_kv_heads * head_dim]
        dim_pos: Dimension offsets [num_gen_tokens, num_kv_heads, 1, head_dim] (0 or head_dim for each dim)
        kv_lens: Sequence lengths [num_gen_tokens]
        kt_page_size: Page size for KT tokens
        tokens_per_block: Tokens per cache block
        max_kt_blocks_per_seq: Maximum KT blocks per sequence
        num_kt_tokens: Number of KT tokens for each batch
        total_kt_tokens: Total number of KT tokens
        kt_cache_tensor: KT cache tensor
        kt_cache_block_offsets: Block offsets [num_gen_tokens, max_kt_blocks_per_seq]
        output_offsets: Output offsets [num_gen_tokens + 1]
        max_real_kt_tokens: Maximum real KT tokens
        sm_scale: Scale factor for softmax
    Returns:
        output: BMM results [num_heads, 1, total_kt_tokens]
    """
    num_gen_tokens, num_kv_heads, num_heads_per_kv, head_dim = q.shape
    total_num_heads = num_kv_heads * num_heads_per_kv

    max_num_kt_tokens = triton.next_power_of_2(max_real_kt_tokens)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Step 1: Update KT cache with new key values
    grid = (num_gen_tokens, num_kv_heads, 1)

    kt_cache_update_kernel[grid](k,
                                 kt_cache_tensor,
                                 kt_cache_block_offsets,
                                 kv_lens,
                                 num_gen_tokens,
                                 num_kv_heads,
                                 head_dim,
                                 kt_page_size,
                                 tokens_per_block,
                                 max_kt_blocks_per_seq,
                                 DIM_BLOCK_SIZE=128)

    # Step 2: Perform BMM with updated cache
    # Create output tensor with shape [num_heads, 1, total_kt_tokens]
    output = torch.empty((total_num_heads, 1, total_kt_tokens),
                         dtype=torch.float32,
                         device=q.device)

    # Grid: (num_gen_tokens, ceil(max_num_kt_tokens / KT_BLOCK_SIZE), total_num_heads)
    def grid(meta):
        return (num_gen_tokens,
                (max_num_kt_tokens + meta['KT_BLOCK_SIZE'] - 1) //
                meta['KT_BLOCK_SIZE'], total_num_heads)

    paged_kt_cache_bmm_kernel[grid](
        q,
        kt_cache_tensor,
        kt_cache_block_offsets,
        dim_pos,
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
        max_num_kt_tokens,
        total_kt_tokens,
        sm_scale,
    )

    return output


########################################################
# Triton TopK kernel with optional interleave
########################################################


@triton.jit
def triton_interleave_kernel(
    input_ptr,
    output_ptr,
    kt_offsets_ptr,
    kv_lens_ptr,
    kv_offsets_ptr,
    batch_size,
    num_kv_heads,
    kt_page_size,
    total_kt_tokens,
    total_kv_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Interleave kt tokens to kv tokens by repeating each kt token kt_page_size times.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if batch_idx >= batch_size or head_idx >= num_kv_heads:
        return

    # Get batch kt and kv ranges
    kt_start = tl.load(kt_offsets_ptr + batch_idx)
    kt_end = tl.load(kt_offsets_ptr + batch_idx + 1)
    kt_len = kt_end - kt_start

    kv_len = tl.load(kv_lens_ptr + batch_idx)
    kv_start = tl.load(kv_offsets_ptr + batch_idx)

    if kt_len <= 0 or kv_len <= 0:
        return

    # Process in blocks
    for block_start in tl.range(0, kv_len, BLOCK_SIZE):
        block_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        block_mask = block_offsets < kv_len

        # Calculate which kt_token each kv position corresponds to
        kt_indices = block_offsets // kt_page_size
        kt_valid_mask = kt_indices < kt_len
        combined_mask = block_mask & kt_valid_mask

        # Load from input kt tokens
        input_indices = head_idx * total_kt_tokens + kt_start + kt_indices

        values = tl.load(input_ptr + input_indices,
                         mask=combined_mask,
                         other=0.0)

        # Store to output kv positions
        output_indices = head_idx * total_kv_tokens + kv_start + block_offsets
        tl.store(output_ptr + output_indices, values, mask=block_mask)


@triton.jit
def topk_kernel(
    input_ptr,
    output_indices_ptr,
    temp_values_ptr,
    temp_indices_ptr,
    input_offsets_ptr,
    sparse_offsets_ptr,
    batch_size,
    num_kv_heads,
    topk,
    total_input_tokens,
    total_sparse_indices,
    max_seq_len,
    max_real_tokens,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Perform topk operation on each batch independently using efficient argsort implementation.
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if batch_idx >= batch_size or head_idx >= num_kv_heads:
        return

    input_start = tl.load(input_offsets_ptr + batch_idx)
    input_end = tl.load(input_offsets_ptr + batch_idx + 1)
    input_len = input_end - input_start

    sparse_start = tl.load(sparse_offsets_ptr + batch_idx)
    sparse_end = tl.load(sparse_offsets_ptr + batch_idx + 1)
    sparse_len = sparse_end - sparse_start

    if input_len <= 0 or sparse_len <= 0:
        return

    actual_topk = tl.minimum(topk, input_len)
    actual_topk = tl.minimum(actual_topk, sparse_len)

    # Base addresses
    input_base = head_idx * total_input_tokens + input_start
    temp_base = batch_idx * num_kv_heads * max_seq_len * 2 + head_idx * max_seq_len * 2
    output_base = head_idx * total_sparse_indices + sparse_start

    # Process sequence in chunks to handle variable lengths efficiently
    max_process_len = tl.cdiv(input_len, BLOCK_SIZE) * BLOCK_SIZE

    for block_start in tl.range(0, max_process_len, BLOCK_SIZE):
        block_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        block_mask = block_offsets < input_len

        values = tl.load(input_ptr + input_base + block_offsets,
                         mask=block_mask,
                         other=0.0)

        # Store values to temporary storage
        tl.store(temp_values_ptr + temp_base + block_offsets,
                 values,
                 mask=block_mask)
        # Store original indices
        tl.store(temp_indices_ptr + temp_base + block_offsets,
                 block_offsets,
                 mask=block_mask)

    # Multi-round iterative argsort approach
    # This works for both short and long sequences uniformly
    current_len = input_len.to(tl.int32)
    current_base = temp_base
    round_num = 0

    while current_len > BLOCK_SIZE:
        round_num += 1

        num_chunks = tl.cdiv(current_len, BLOCK_SIZE)

        # Alternate between two halves of temp storage to avoid conflicts
        if round_num % 2 == 1:
            next_base = temp_base + max_seq_len
        else:
            next_base = temp_base

        next_len = 0

        # Process each chunk in this round
        for chunk_id in tl.range(0, num_chunks):
            chunk_start = chunk_id * BLOCK_SIZE
            chunk_end = tl.minimum(chunk_start + BLOCK_SIZE, current_len)
            chunk_len = chunk_end - chunk_start

            if chunk_len > 0:
                # Load chunk data from current round's storage
                chunk_offsets = tl.arange(0, BLOCK_SIZE)
                chunk_mask = chunk_offsets < chunk_len

                chunk_values = tl.load(temp_values_ptr + current_base +
                                       chunk_start + chunk_offsets,
                                       mask=chunk_mask,
                                       other=0.0)
                chunk_indices = tl.load(temp_indices_ptr + current_base +
                                        chunk_start + chunk_offsets,
                                        mask=chunk_mask,
                                        other=0.0).to(tl.int32)

                # Sort this chunk using argsort
                chunk_sorted_values, chunk_sorted_indices = argsort(
                    chunk_values, chunk_indices, dim=0, descending=True)

                # Extract top-k candidates from this chunk
                chunk_topk = tl.minimum(actual_topk, chunk_len).to(tl.int32)
                chunk_topk_mask = chunk_offsets < chunk_topk

                # Store top-k candidates to next round's storage
                next_offsets = next_len + chunk_offsets
                next_store_mask = chunk_topk_mask & (next_offsets < max_seq_len)

                tl.store(temp_values_ptr + next_base + next_offsets,
                         chunk_sorted_values,
                         mask=next_store_mask)
                tl.store(temp_indices_ptr + next_base + next_offsets,
                         chunk_sorted_indices,
                         mask=next_store_mask)

                next_len += chunk_topk

        # Update parameters for next round
        current_len = next_len
        current_base = next_base

    final_offsets = tl.arange(0, BLOCK_SIZE)
    final_mask = final_offsets < current_len

    final_values = tl.load(temp_values_ptr + current_base + final_offsets,
                           mask=final_mask,
                           other=-1e10)
    final_indices = tl.load(temp_indices_ptr + current_base + final_offsets,
                            mask=final_mask,
                            other=-1).to(tl.int32)

    final_sorted_values, final_sorted_indices = argsort(final_values,
                                                        final_indices,
                                                        dim=0,
                                                        descending=True)

    result_offsets = tl.arange(0, BLOCK_SIZE)
    result_mask = result_offsets < actual_topk

    selected_indices = tl.where(result_mask, final_sorted_indices,
                                tl.zeros_like(final_sorted_indices))
    tl.store(output_indices_ptr + output_base + result_offsets,
             selected_indices,
             mask=result_mask)


def triton_topk(
        input_tensor: torch.Tensor,
        kt_offsets: torch.Tensor,
        kv_lens: torch.Tensor,
        kv_cu_lens: torch.Tensor,
        sparse_offsets: torch.Tensor,
        total_sparse_attn_indices: int,
        total_kv_tokens: int,
        max_attn_seq_len: int,
        max_real_kt_tokens: int,
        max_real_kv_tokens: int,
        topk: int,
        kt_page_size: int,
        use_interleave: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform topk operation with optional interleaving.

    Args:
        input_tensor: Input scores [num_kv_heads, sum(kt_lens)]
        kt_offsets: KT offsets [batch_size + 1]
        kv_lens: KV lengths [batch_size]
        kv_cu_lens: KV cumulative lengths [batch_size + 1]
        sparse_offsets: Sparse offsets [batch_size + 1]
        total_sparse_attn_indices: Total number of sparse attention indices
        total_kv_tokens: Total number of KV tokens
        max_attn_seq_len: Maximum sequence length
        max_real_kt_tokens: Maximum real KT tokens
        max_real_kv_tokens: Maximum real KV tokens
        topk: scalar of TopK parameter
        kt_page_size: Page size for interleaving
        use_interleave: Whether to perform interleaving

    Returns:
        output_indices: Selected indices [num_kv_heads, num_total_sparse_indices]
        sparse_offsets: Sparse offsets [batch_size + 1]
    """

    num_kv_heads = input_tensor.shape[0]
    batch_size = len(kv_lens)
    device = input_tensor.device

    if use_interleave:
        total_kt_tokens = input_tensor.shape[1]

        # Create interleaved tensor
        interleaved_tensor = torch.empty((num_kv_heads, total_kv_tokens),
                                         dtype=input_tensor.dtype,
                                         device=device)

        # Launch interleave kernel
        grid = (batch_size, num_kv_heads)
        triton_interleave_kernel[grid](input_tensor,
                                       interleaved_tensor,
                                       kt_offsets,
                                       kv_lens,
                                       kv_cu_lens,
                                       batch_size,
                                       num_kv_heads,
                                       kt_page_size,
                                       total_kt_tokens,
                                       total_kv_tokens,
                                       BLOCK_SIZE=1024)

        # Use interleaved tensor and kv_cu_lens for topk
        working_tensor = interleaved_tensor
        working_offsets = kv_cu_lens
        max_real_tokens = triton.next_power_of_2(max_real_kv_tokens)
    else:
        # Use original tensor and kt_offsets for topk
        working_tensor = input_tensor
        working_offsets = kt_offsets
        max_real_tokens = triton.next_power_of_2(max_real_kt_tokens)

    total_working_tokens = working_tensor.shape[1]

    # Create output tensor
    output_indices = torch.empty((num_kv_heads, total_sparse_attn_indices),
                                 dtype=torch.int32,
                                 device=device)

    # Create temporary storage for topk algorithm (double size for dual-buffer design)
    temp_values = torch.empty((batch_size, num_kv_heads, max_attn_seq_len * 2),
                              dtype=working_tensor.dtype,
                              device=device)
    temp_indices = torch.empty((batch_size, num_kv_heads, max_attn_seq_len * 2),
                               dtype=torch.int32,
                               device=device)

    grid = (batch_size, num_kv_heads)

    assert topk & (topk - 1) == 0, "Topk must be a power of 2"
    BLOCK_SIZE = max(512, 2 * topk)

    topk_kernel[grid](
        working_tensor,
        output_indices,
        temp_values,
        temp_indices,
        working_offsets,
        sparse_offsets,
        batch_size,
        num_kv_heads,
        topk,
        total_working_tokens,
        total_sparse_attn_indices,
        max_attn_seq_len,
        max_real_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_indices


########################################################
# Reduce scores generation kernel
########################################################


@triton.jit
def reduce_scores_kernel(
    input_ptr,
    output_ptr,
    cum_kt_lens_ptr,
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
    kt_start = tl.load(cum_kt_lens_ptr + batch_idx)
    kt_end = tl.load(cum_kt_lens_ptr + batch_idx + 1)
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


def triton_reduce_scores(
    scores: torch.Tensor,
    cum_kt_lens: torch.Tensor,
    batch_size: int,
    num_kv_heads: int,
    num_heads_per_kv: int,
) -> torch.Tensor:
    """
    Reduce scores for generation phase with batch-aware processing.

    Args:
        scores: Input scores [num_kv_heads * num_heads_per_kv, 1, total_kt_tokens]
        cum_kt_lens: Cumulative KT lengths [batch_size + 1]
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

    reduce_scores_kernel[grid](
        scores,
        output,
        cum_kt_lens,
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
