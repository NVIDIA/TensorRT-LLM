import torch
import triton
import triton.language as tl


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


@triton.jit
def _update_kt_cache_ctx_kernel(k_ptr, cache_ptr, block_offsets_ptr,
                                cum_seq_lens_ptr, cum_kt_seq_lens_ptr,
                                token_to_batch_map_ptr, num_kv_heads, dim_size,
                                kt_page_size, tokens_per_block,
                                max_kt_blocks_per_seq,
                                BLOCK_SIZE: tl.constexpr):
    # get program id
    kt_token_idx = tl.program_id(0)

    # get params
    batch_idx = tl.load(token_to_batch_map_ptr + kt_token_idx)
    kv_start_idx = tl.load(cum_seq_lens_ptr + batch_idx)
    kv_end_idx = tl.load(cum_seq_lens_ptr + batch_idx + 1)
    kt_start_idx = tl.load(cum_kt_seq_lens_ptr + batch_idx)
    local_kt_token_idx = kt_token_idx - kt_start_idx
    global_kv_token_idx = kv_start_idx + local_kt_token_idx * kt_page_size

    # get offsets
    hidden_size = num_kv_heads * dim_size
    k_base = k_ptr + global_kv_token_idx * hidden_size
    block_offset = batch_idx * max_kt_blocks_per_seq + local_kt_token_idx // tokens_per_block
    block_idx = tl.load(block_offsets_ptr + block_offset)
    token_idx_in_block = local_kt_token_idx % tokens_per_block
    cache_base = cache_ptr + (block_idx * tokens_per_block +
                              token_idx_in_block) * hidden_size * 2

    # compute min/max and store kt
    for hidden_start in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_indices = hidden_start + tl.arange(0, BLOCK_SIZE)
        head_idx = hidden_indices // dim_size
        dim_idx = hidden_indices % dim_size
        dim_mask = hidden_indices < hidden_size

        # get k_min and k_max
        k_min = tl.full((BLOCK_SIZE, ), float('inf'), dtype=tl.float32)
        k_max = tl.full((BLOCK_SIZE, ), float('-inf'), dtype=tl.float32)
        for i in range(kt_page_size):
            if global_kv_token_idx + i < kv_end_idx:
                k = tl.load(k_base + i * hidden_size + hidden_indices,
                            mask=dim_mask,
                            other=0.0)
                k_min = tl.minimum(k_min, k)
                k_max = tl.maximum(k_max, k)
        k_min = k_min.to(cache_ptr.dtype.element_ty)
        k_max = k_max.to(cache_ptr.dtype.element_ty)

        # store k_min and k_max to cache
        k_min_offset = cache_base + head_idx * dim_size * 2 + dim_idx
        k_max_offset = k_min_offset + dim_size
        tl.store(k_min_offset, k_min, mask=dim_mask)
        tl.store(k_max_offset, k_max, mask=dim_mask)


@triton.jit
def _update_kt_cache_gen_kernel(k_ptr, cache_ptr, block_offsets_ptr,
                                seq_lens_ptr, num_kv_heads, dim_size,
                                kt_page_size, tokens_per_block,
                                max_kt_blocks_per_seq,
                                BLOCK_SIZE: tl.constexpr):
    # get program id
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # get params
    past_key_value_length = tl.load(seq_lens_ptr + batch_idx) - 1
    kt_token_idx = past_key_value_length // kt_page_size
    kt_token_idx_in_page = past_key_value_length % kt_page_size

    # get offsets
    hidden_size = num_kv_heads * dim_size
    k_base = k_ptr + batch_idx * hidden_size + head_idx * dim_size
    block_offset = batch_idx * max_kt_blocks_per_seq + kt_token_idx // tokens_per_block
    block_idx = tl.load(block_offsets_ptr + block_offset)
    kt_token_idx_in_block = kt_token_idx % tokens_per_block
    cache_base = cache_ptr + (block_idx * tokens_per_block +
                              kt_token_idx_in_block) * hidden_size * 2
    cache_base += head_idx * dim_size * 2

    # update kt cache
    for hidden_start in tl.range(0, dim_size, BLOCK_SIZE):
        hidden_indices = hidden_start + tl.arange(0, BLOCK_SIZE)
        dim_mask = hidden_indices < dim_size

        # load k
        k = tl.load(k_base + hidden_indices, mask=dim_mask, other=0.0)

        # load kt cache
        kt_mask = dim_mask & (kt_token_idx_in_page > 0)
        k_min = tl.load(cache_base + hidden_indices,
                        mask=kt_mask,
                        other=float('inf'))
        k_max = tl.load(cache_base + hidden_indices + dim_size,
                        mask=kt_mask,
                        other=float('-inf'))
        k_min = tl.minimum(k_min, k)
        k_max = tl.maximum(k_max, k)
        k_min = k_min.to(cache_ptr.dtype.element_ty)
        k_max = k_max.to(cache_ptr.dtype.element_ty)

        # store k_min and k_max to cache
        tl.store(cache_base + hidden_indices, k_min, mask=dim_mask)
        tl.store(cache_base + hidden_indices + dim_size, k_max, mask=dim_mask)


@triton.jit
def _load_kt_cache_kernel(kt_states_ptr, cache_ptr, block_offsets_ptr,
                          cum_kt_seq_lens_ptr, token_to_batch_map_ptr,
                          num_kv_heads, dim_size, tokens_per_block,
                          max_kt_blocks_per_seq, BLOCK_SIZE: tl.constexpr):
    # get program id
    kt_token_idx = tl.program_id(0)

    # get params
    batch_idx = tl.load(token_to_batch_map_ptr + kt_token_idx)
    kt_start_idx = tl.load(cum_kt_seq_lens_ptr + batch_idx)
    local_kt_token_idx = kt_token_idx - kt_start_idx

    # get offsets
    hidden_size = num_kv_heads * dim_size * 2
    kt_states_base = kt_states_ptr + kt_token_idx * hidden_size
    block_offset = batch_idx * max_kt_blocks_per_seq + local_kt_token_idx // tokens_per_block
    block_idx = tl.load(block_offsets_ptr + block_offset)
    token_idx_in_block = local_kt_token_idx % tokens_per_block
    cache_base = cache_ptr + (block_idx * tokens_per_block +
                              token_idx_in_block) * hidden_size

    # load kt cache
    for hidden_start in tl.range(0, hidden_size, BLOCK_SIZE):
        hidden_indices = hidden_start + tl.arange(0, BLOCK_SIZE)
        mask = hidden_indices < hidden_size
        # load kt cache
        kt = tl.load(cache_base + hidden_indices, mask=mask, other=0.0)
        # store kt to kt_states
        tl.store(kt_states_base + hidden_indices, kt, mask=mask)


def triton_update_kt_cache(k,
                           kt_cache_tensor,
                           kt_cache_block_offsets,
                           seq_lens,
                           kt_page_size,
                           tokens_per_block,
                           max_kt_blocks_per_seq,
                           update=True):
    # inputs:
    # k: (total_seq_len, num_kv_heads, head_dim)
    # kt_cache_tensor: (num_blocks, tokens_per_block, num_kv_heads, 2 * head_dim)
    # kt_cache_block_offsets: (max_batch_size, max_kt_blocks_per_seq)
    # seq_lens: (batch_size)
    # kt_page_size: int
    # update: bool

    # outputs:
    # kt_states: (total_kt_tokens, num_kv_heads, 2 * head_dim)

    # params
    batch_size = seq_lens.size(0)
    num_kv_heads = k.size(1)
    head_dim = k.size(2)
    tokens_per_block = kt_cache_tensor.size(1)
    num_kt_tokens = (seq_lens + kt_page_size - 1) // kt_page_size

    # context
    if not update:
        total_num_kt_tokens = num_kt_tokens.sum().item()
        cum_seq_lens = torch.cumsum(torch.cat([
            torch.zeros(1, device='cuda', dtype=torch.long),
            seq_lens.to(torch.long)
        ]),
                                    dim=0)
        cum_kt_seq_lens = torch.cumsum(torch.cat([
            torch.zeros(1, device='cuda', dtype=torch.long),
            num_kt_tokens.to(torch.long)
        ]),
                                       dim=0)

        token_to_batch_map = torch.repeat_interleave(
            torch.arange(batch_size,
                         device='cuda'), repeats=num_kt_tokens).to(torch.long)
        grid = (total_num_kt_tokens, )
        _update_kt_cache_ctx_kernel[grid](k,
                                          kt_cache_tensor,
                                          kt_cache_block_offsets,
                                          cum_seq_lens,
                                          cum_kt_seq_lens,
                                          token_to_batch_map,
                                          num_kv_heads,
                                          head_dim,
                                          kt_page_size,
                                          tokens_per_block,
                                          max_kt_blocks_per_seq,
                                          BLOCK_SIZE=1024)
        return
    else:
        # generation
        # update kt cache
        grid = (batch_size, num_kv_heads)
        _update_kt_cache_gen_kernel[grid](k,
                                          kt_cache_tensor,
                                          kt_cache_block_offsets,
                                          seq_lens,
                                          num_kv_heads,
                                          head_dim,
                                          kt_page_size,
                                          tokens_per_block,
                                          max_kt_blocks_per_seq,
                                          BLOCK_SIZE=1024)

        # load kt cache
        total_num_kt_tokens = num_kt_tokens.sum().item()
        kt_states = torch.empty(
            (total_num_kt_tokens, num_kv_heads, 2 * head_dim),
            device='cuda',
            dtype=k.dtype)
        token_to_batch_map = torch.repeat_interleave(
            torch.arange(batch_size,
                         device='cuda'), repeats=num_kt_tokens).to(torch.long)
        cum_kt_seq_lens = torch.cumsum(torch.cat([
            torch.zeros(1, device='cuda', dtype=torch.long),
            num_kt_tokens.to(torch.long)
        ]),
                                       dim=0)
        grid = (total_num_kt_tokens, )
        _load_kt_cache_kernel[grid](kt_states,
                                    kt_cache_tensor,
                                    kt_cache_block_offsets,
                                    cum_kt_seq_lens,
                                    token_to_batch_map,
                                    num_kv_heads,
                                    head_dim,
                                    tokens_per_block,
                                    max_kt_blocks_per_seq,
                                    BLOCK_SIZE=1024)

        return kt_states
