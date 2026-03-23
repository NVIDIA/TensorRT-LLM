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


