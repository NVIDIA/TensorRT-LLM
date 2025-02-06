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
