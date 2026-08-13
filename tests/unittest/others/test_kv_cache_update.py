import random

import pytest
import torch

import tensorrt_llm  # noqa


def is_integer_type(torch_dtype):
    integer_types = {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        torch.short, torch.int, torch.long
    }
    return torch_dtype in integer_types


def is_float_type(torch_dtype):
    float_types = {
        torch.float16, torch.float32, torch.float64, torch.float, torch.double,
        torch.half
    }
    return torch_dtype in float_types


def _make_random_cache_data(shape, dtype, device='cuda'):
    """Create random cache data of the given shape and dtype."""
    if is_integer_type(dtype):
        return torch.randint(0, 100, shape, dtype=dtype, device=device)
    elif is_float_type(dtype):
        return torch.rand(shape, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 67, 128])
@pytest.mark.parametrize("layer_count", [1, 32, 45])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("kv_cache_type", [torch.int8, torch.float16])
@pytest.mark.parametrize("rewind_draft_token_count", [5, 63])
@pytest.mark.parametrize("separate_draft_count", [False, True])
@pytest.mark.parametrize("max_kv_cache_length", [100, 200])
def test_linear_kvcache_update(num_kv_heads: int, head_dim: int,
                               layer_count: int, batch_size: int,
                               kv_cache_type: torch.dtype,
                               rewind_draft_token_count: int,
                               separate_draft_count: bool,
                               max_kv_cache_length: int):
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(1234)
    cache_shape = (
        batch_size,
        2,
        num_kv_heads,
        max_kv_cache_length,
        head_dim,
    )
    elt_size = torch.zeros(1, dtype=kv_cache_type).element_size()
    if is_integer_type(kv_cache_type):
        past_key_values = [
            torch.randint(0,
                          100,
                          cache_shape,
                          dtype=kv_cache_type,
                          device='cuda') for i in range(layer_count)
        ]
    elif is_float_type(kv_cache_type):
        past_key_values = [
            torch.rand(cache_shape, dtype=kv_cache_type, device='cuda')
            for i in range(layer_count)
        ]
    else:
        raise ValueError("dtype is neither float or integer.")
    rewind_draft_token_count_tensor_cuda = None
    if separate_draft_count:
        rewind_draft_token_count_tensor_cpu = torch.randint(
            1, rewind_draft_token_count, (batch_size, ), dtype=torch.int32)
        rewind_draft_token_count_tensor_cuda = rewind_draft_token_count_tensor_cpu.cuda(
        )
    else:
        rewind_draft_token_count_tensor_cpu = torch.full(
            (batch_size, ), rewind_draft_token_count, dtype=torch.int32)
    rewind_draft_token_tensor_list = rewind_draft_token_count_tensor_cpu.tolist(
    )
    accepted_draft_token_counts_list = [
        random.randint(0, rewind_draft_token_tensor_list_value) for
        rewind_draft_token_tensor_list_value in rewind_draft_token_tensor_list
    ]
    accepted_draft_token_counts = torch.tensor(accepted_draft_token_counts_list,
                                               dtype=torch.int32).cuda()
    accepted_draft_token_offsets = torch.zeros(batch_size + 1,
                                               dtype=torch.int32,
                                               device='cuda')
    accepted_draft_token_offsets[1:] = torch.cumsum(accepted_draft_token_counts,
                                                    dim=0)
    accepted_draft_token_offsets_cpu = accepted_draft_token_offsets.to('cpu')

    packed_accepted_draft_tokens_indices_cpu = torch.empty(
        accepted_draft_token_offsets_cpu[batch_size], dtype=torch.int32)

    for seq_idx in range(batch_size):
        rand_perm = torch.randperm(rewind_draft_token_tensor_list[seq_idx],
                                   dtype=torch.int32)
        seq_start = accepted_draft_token_offsets_cpu[seq_idx]
        seq_end = accepted_draft_token_offsets_cpu[seq_idx + 1]
        packed_accepted_draft_tokens_indices_cpu[
            seq_start:seq_end] = rand_perm[:seq_end - seq_start]

    packed_accepted_draft_tokens_indices = packed_accepted_draft_tokens_indices_cpu.to(
        'cuda')
    past_key_value_lengths = torch.randint(rewind_draft_token_count,
                                           max_kv_cache_length, (batch_size, ),
                                           dtype=torch.int32,
                                           device='cuda')
    past_key_value_lengths_cpu = past_key_value_lengths.to('cpu')

    # compute ground truth first
    ground_truth_past_key_values = []
    for i in range(layer_count):
        layer_past_key_value = past_key_values[i]
        new_layer_past_key_value = layer_past_key_value.clone()
        for seq_idx in range(batch_size):
            token_start = accepted_draft_token_offsets_cpu[seq_idx]
            token_end = accepted_draft_token_offsets_cpu[seq_idx + 1]
            for relative_target_idx in range(token_end - token_start):
                relative_draft_idx = packed_accepted_draft_tokens_indices_cpu[
                    token_start + relative_target_idx]
                past_key_value_len = past_key_value_lengths_cpu[seq_idx]
                rewind_key_value_len = past_key_value_len - rewind_draft_token_tensor_list[
                    seq_idx]
                new_layer_past_key_value[
                    seq_idx, :, :, rewind_key_value_len +
                    relative_target_idx] = layer_past_key_value[
                        seq_idx, :, :,
                        rewind_key_value_len + relative_draft_idx]
        ground_truth_past_key_values.append(new_layer_past_key_value)

    torch.cuda.synchronize()

    torch.ops.tensorrt_llm.update_kv_cache_draft_token_location(
        accepted_draft_token_offsets,
        packed_accepted_draft_tokens_indices,
        past_key_value_lengths,
        False,
        layer_count,
        num_kv_heads,
        head_dim * elt_size,
        0 if separate_draft_count else rewind_draft_token_count,
        max_kv_cache_length,
        rewind_draft_token_count_tensor_cuda if separate_draft_count else None,
        past_key_values,
        None,
        None,
        None,
        None,
        None,
    )
    torch.cuda.synchronize()

    for i in range(layer_count):
        layer_past_key_value = past_key_values[i]
        ground_truth_layer_past_key_value = ground_truth_past_key_values[i]
        assert torch.allclose(layer_past_key_value,
                              ground_truth_layer_past_key_value)


@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("layer_count", [1, 32])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("kv_cache_type", [torch.int8, torch.float16])
@pytest.mark.parametrize("rewind_draft_token_count", [5, 63])
@pytest.mark.parametrize("max_kv_cache_length", [128, 256])
def test_paged_kvcache_update_2d(num_kv_heads: int, head_dim: int,
                                 layer_count: int, batch_size: int,
                                 kv_cache_type: torch.dtype,
                                 rewind_draft_token_count: int,
                                 max_kv_cache_length: int):
    """Test the 2D variant of update_kv_cache_draft_token_location (paged KV cache only).

    The 2D kernel takes a [batch_size, max_draft_len] indices tensor instead of
    packed 1D indices with offsets.  It reads KV entries from scattered draft
    positions and compacts them into sequential positions.
    """
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(1234)

    elt_size = torch.zeros(1, dtype=kv_cache_type).element_size()
    head_size_in_bytes = head_dim * elt_size
    bytes_per_token = num_kv_heads * head_size_in_bytes

    # tokens_per_block must be power of 2
    tokens_per_block = 64
    bytes_per_block = tokens_per_block * bytes_per_token
    max_blocks_per_seq = (max_kv_cache_length + tokens_per_block -
                          1) // tokens_per_block + 1  # +1 for oneMoreBlock
    total_blocks = batch_size * max_blocks_per_seq

    # Allocate pool: each pool block stores all layers, each layer has K+V
    # Layout per pool block: [layer_count * 2 * bytes_per_block] bytes
    pool_bytes_per_block = layer_count * 2 * bytes_per_block
    pool = torch.zeros(total_blocks * pool_bytes_per_block,
                       dtype=torch.uint8,
                       device='cuda')

    # Fill pool with random data via a view
    # We'll use a structured view for ground truth computation
    # Pool block layout per layer: [K: num_kv_heads, tokens_per_block, head_dim]
    #                               [V: num_kv_heads, tokens_per_block, head_dim]
    pool_int8 = pool.view(-1)
    pool_int8[:] = torch.randint(0,
                                 256, (pool_int8.shape[0], ),
                                 dtype=torch.uint8,
                                 device='cuda')

    # pointer array: [primary_pool_ptr, secondary_pool_ptr] as int64
    pointer_array = torch.tensor(
        [pool.data_ptr(), pool.data_ptr()], dtype=torch.int64, device='cuda')

    # offset array: [batch_size, 2, max_blocks_per_seq] as int32
    # Assign blocks sequentially: seq i gets blocks [i*max_blocks_per_seq, ...]
    # Both K and V rows point to the same blocks (primary pool, high bit = 0)
    offset_array = torch.zeros(batch_size * 2 * max_blocks_per_seq,
                               dtype=torch.int32,
                               device='cuda')
    for seq_idx in range(batch_size):
        for blk in range(max_blocks_per_seq):
            block_id = seq_idx * max_blocks_per_seq + blk
            # K row
            offset_array[seq_idx * max_blocks_per_seq * 2 + blk] = block_id
            # V row
            offset_array[seq_idx * max_blocks_per_seq * 2 + max_blocks_per_seq +
                         blk] = block_id

    # Generate accepted draft token data
    max_draft_len = rewind_draft_token_count
    # num_accepted_tokens: the kernel uses max(num_accepted - 1, 0) as draft count
    num_accepted_list = [
        random.randint(1, rewind_draft_token_count) for _ in range(batch_size)
    ]
    num_accepted_tokens = torch.tensor(num_accepted_list,
                                       dtype=torch.int32,
                                       device='cuda')

    # accepted_draft_tokens_indices_2d: [batch_size, max_draft_len]
    # Each row has indices in [0, rewind_draft_token_count) for accepted tokens
    accepted_indices_2d_cpu = torch.zeros(batch_size,
                                          max_draft_len,
                                          dtype=torch.int32)
    for seq_idx in range(batch_size):
        n_accepted = num_accepted_list[seq_idx]
        draft_count = max(n_accepted - 1, 0)
        if draft_count > 0:
            perm = torch.randperm(rewind_draft_token_count,
                                  dtype=torch.int32)[:draft_count]
            accepted_indices_2d_cpu[seq_idx, :draft_count] = perm
    accepted_indices_2d = accepted_indices_2d_cpu.cuda()

    past_key_value_lengths = torch.randint(rewind_draft_token_count,
                                           max_kv_cache_length, (batch_size, ),
                                           dtype=torch.int32,
                                           device='cuda')
    past_key_value_lengths_cpu = past_key_value_lengths.to('cpu')

    # Helper to read/write a KV element from the pool (CPU-side for ground truth)
    pool_cpu = pool.clone().cpu()

    def _get_block_offset(seq_idx, token_pos, kv_idx):
        """Return (pool_block_id, local_token_idx) for a given seq/token/kv."""
        blk_idx = token_pos // tokens_per_block
        local_tok = token_pos % tokens_per_block
        row_base = seq_idx * max_blocks_per_seq * 2 + kv_idx * max_blocks_per_seq
        block_id = offset_array[row_base + blk_idx].item()
        return block_id, local_tok

    def _pool_byte_offset(block_id, layer_idx, kv_idx, head_idx, local_tok,
                          chan):
        """Compute byte offset into the pool for a specific element."""
        block_base = block_id * pool_bytes_per_block
        layer_base = layer_idx * 2 * bytes_per_block
        kv_base = kv_idx * bytes_per_block
        # Within a KV block: [num_kv_heads, tokens_per_block, head_dim] in element units
        elem_offset = (head_idx * tokens_per_block * head_dim +
                       local_tok * head_dim + chan)
        return block_base + layer_base + kv_base + elem_offset * elt_size

    def _read_kv_token(pool_data, seq_idx, token_pos, layer_idx, kv_idx):
        """Read [num_kv_heads, head_dim] for one token from pool."""
        block_id, local_tok = _get_block_offset(seq_idx, token_pos, kv_idx)
        result = torch.zeros(num_kv_heads, head_dim, dtype=kv_cache_type)
        for h in range(num_kv_heads):
            for d in range(head_dim):
                off = _pool_byte_offset(block_id, layer_idx, kv_idx, h,
                                        local_tok, d)
                raw = pool_data[off:off + elt_size]
                result[h, d] = torch.frombuffer(raw.numpy().tobytes(),
                                                dtype=kv_cache_type)[0]
        return result

    def _write_kv_token(pool_data, seq_idx, token_pos, layer_idx, kv_idx,
                        values):
        """Write [num_kv_heads, head_dim] for one token to pool."""
        block_id, local_tok = _get_block_offset(seq_idx, token_pos, kv_idx)
        val_bytes = values.contiguous().numpy().view('uint8')
        for h in range(num_kv_heads):
            for d in range(head_dim):
                off = _pool_byte_offset(block_id, layer_idx, kv_idx, h,
                                        local_tok, d)
                src_off = (h * head_dim + d) * elt_size
                pool_data[off:off + elt_size] = torch.from_numpy(
                    val_bytes[src_off:src_off + elt_size].copy())

    # Compute ground truth on CPU
    gt_pool = pool_cpu.clone()
    for layer_idx in range(layer_count):
        for seq_idx in range(batch_size):
            draft_count = max(num_accepted_list[seq_idx] - 1, 0)
            if draft_count == 0:
                continue
            past_len = past_key_value_lengths_cpu[seq_idx].item()
            token_start = past_len - rewind_draft_token_count
            for target_idx in range(draft_count):
                src_pos = accepted_indices_2d_cpu[seq_idx, target_idx].item()
                for kv_idx in range(2):  # K=0, V=1
                    val = _read_kv_token(pool_cpu, seq_idx,
                                         token_start + src_pos, layer_idx,
                                         kv_idx)
                    _write_kv_token(gt_pool, seq_idx, token_start + target_idx,
                                    layer_idx, kv_idx, val)

    torch.cuda.synchronize()

    # Run the 2D kernel
    torch.ops.tensorrt_llm.update_kv_cache_draft_token_location_2d(
        accepted_indices_2d,
        num_accepted_tokens,
        past_key_value_lengths,
        True,
        layer_count,
        num_kv_heads,
        head_size_in_bytes,
        rewind_draft_token_count,
        max_kv_cache_length,
        pointer_array,
        offset_array,
        max_blocks_per_seq,
        tokens_per_block,
        None,
    )
    torch.cuda.synchronize()

    # Compare
    result_pool = pool.cpu()
    assert torch.equal(result_pool, gt_pool), \
        "Paged KV cache 2D update mismatch"
