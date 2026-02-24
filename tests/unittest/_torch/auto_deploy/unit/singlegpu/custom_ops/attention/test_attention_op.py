import pytest
import torch
from torch_attention_reference import TorchAttentionReference

import tensorrt_llm._torch.auto_deploy  # noqa: F401


@pytest.mark.parametrize("num_generate_ratio", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("max_seq_len", [0, 1, 16])
@pytest.mark.parametrize("group_size", [1, 4])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("device", ["cuda"])
def test_flat_gqa_op(
    device, dtype, batch_size, n_heads, group_size, max_seq_len, num_generate_ratio
):
    n_heads = n_heads
    n_kv_heads = n_heads // group_size
    D_HEAD = 16
    dtype = getattr(torch, dtype)
    int_kwargs = {"device": device, "dtype": torch.int32}
    dtype_kwargs = {"device": device, "dtype": dtype}

    # setup caches with 2*batch_size, 2*max_seq_len since we also randomize input_pos
    cache_max_seq_len = 2 * (max_seq_len + 1)
    cache_max_batch_size = 2 * batch_size
    cache_size = (cache_max_batch_size, cache_max_seq_len, n_kv_heads, D_HEAD)
    cache_loc = torch.randperm(cache_max_batch_size, **int_kwargs)[:batch_size]

    k_cache = torch.randn(cache_size, **dtype_kwargs)
    v_cache = torch.randn(cache_size, **dtype_kwargs)

    # randomize num_context vs num_generate; NOTE: we can use context kernel for generate as well
    num_generate = torch.tensor(num_generate_ratio * batch_size, **int_kwargs)
    num_context = batch_size - num_generate

    # construct random input_positions
    input_positions = torch.randint(0, max_seq_len + 1, (batch_size,), **int_kwargs)

    # construct seq_len, seq_start;
    seq_len = torch.cat(
        [
            torch.randint(0, max_seq_len + 1, (num_context,), **int_kwargs),  # context
            torch.zeros(num_generate, **int_kwargs) + (max_seq_len > 0),  # generate
        ]
    )
    seq_start = seq_len.cumsum(0) - seq_len

    # get fake input
    q = torch.randn(1, seq_len.sum(), n_heads * D_HEAD, **dtype_kwargs)
    k = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)
    v = torch.randn(1, seq_len.sum(), n_kv_heads * D_HEAD, **dtype_kwargs)

    # create batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    num_prefill_tokens = seq_len[:num_context].sum()
    batch_info_host = torch.tensor([num_context, num_prefill_tokens, num_generate], **int_kwargs)

    # run op
    output = torch.ops.auto_deploy.triton_attention_flattened_mha_with_cache(
        # Q, K, V
        q,
        k,
        v,
        # STANDARD METADATA
        batch_info_host,
        seq_len,
        input_positions,
        cache_loc,
        seq_start,  # cu_seqlen
        # CACHES
        k_cache,
        v_cache,
        # CONSTANTS
        scale=None,
    )

    # Use torch backend as clean reference
    ref_flat = TorchAttentionReference.flattened_mha_with_cache(
        q, k, v, batch_info_host, seq_len, input_positions, cache_loc, seq_start, k_cache, v_cache
    )

    assert torch.allclose(
        ref_flat.cpu().to(torch.float32),
        output.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )
