"""Unit tests for Triton-backed cached causal conv1d custom ops.

Covers:
- Generate-only path comparing Triton vs CUDA backend
- Context (flattened) path comparing Triton vs CUDA backend
- Ensures numerical consistency between backends
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def _random_params_depthwise(device, dtype, batch, seq, channels, k):
    x = torch.randn(batch, seq, channels, device=device, dtype=dtype)
    # Depthwise: out_channels == in_channels, groups == channels, weight [C, 1, K]
    weight = torch.randn(channels, 1, k, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)
    stride = 1
    padding = k - 1
    dilation = 1
    groups = channels
    padding_mode = "zeros"
    return x, weight, bias, stride, padding, dilation, groups, padding_mode


@pytest.fixture
def conv_env():
    device = "cuda"
    dtype = torch.float16
    atol = 5e-2
    rtol = 5e-2
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    return {"device": device, "dtype": dtype, "atol": atol, "rtol": rtol}


def test_generate_only_triton_vs_cuda(conv_env):
    """Test that Triton backend matches CUDA backend for generate-only (decode) path."""
    device = conv_env["device"]
    dtype = conv_env["dtype"]

    batch, seq = 1, 1
    c, k = 2, 3
    x, w, b, s, p, d, g, pm = _random_params_depthwise(device, dtype, batch, seq, c, k)

    # Slot mapping with arbitrary order within max_batch_size
    max_batch_size = 2
    slot_idx = torch.tensor([0], device=device, dtype=torch.int32)
    # Cache holds K-1 entries (TRT update kernel contract)
    conv_state_cache_cuda = torch.randn(
        max_batch_size,
        c,
        k - 1,
        device=device,
        dtype=dtype,
    )
    # Clone for Triton backend
    conv_state_cache_triton = conv_state_cache_cuda.clone()

    # Metadata
    cu_seqlen = torch.zeros(batch, device=device, dtype=torch.int32)
    seq_len = torch.zeros(batch, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(batch, device=device, dtype=torch.bool)
    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    # For generate-only: num_decode = batch, num_prefill = 0
    batch_info_host = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)

    # Clone inputs for each backend
    x_cuda = x.clone()
    x_triton = x.clone()

    # Run CUDA backend
    torch.ops.auto_deploy.cuda_cached_causal_conv1d(
        x_cuda,
        w,
        b,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_cuda,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Run Triton backend (includes seq_len parameter)
    torch.ops.auto_deploy.triton_cached_causal_conv1d(
        x_triton,
        w,
        b,
        batch_info_host,
        seq_len,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_triton,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Compare outputs
    assert x_cuda.shape == x_triton.shape
    assert torch.allclose(x_cuda, x_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]), (
        f"Output mismatch: max diff = {(x_cuda - x_triton).abs().max()}"
    )

    # Compare cache states
    assert torch.allclose(
        conv_state_cache_cuda, conv_state_cache_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]
    ), (
        f"Cache state mismatch: max diff = {(conv_state_cache_cuda - conv_state_cache_triton).abs().max()}"
    )


def test_context_flattened_triton_vs_cuda(conv_env):
    """Test that Triton backend matches CUDA backend for context (prefill) path."""
    device = conv_env["device"]
    dtype = conv_env["dtype"]

    # Two short sequences with lengths 2 and 1, flattened to [1,3]
    lens = [2, 1]
    total = sum(lens)
    batch, seq = 1, total
    c, k = 2, 3
    x, w, b, s, p, d, g, pm = _random_params_depthwise(device, dtype, batch, seq, c, k)

    max_batch_size = 2
    slot_idx = torch.tensor([1, 0], device=device, dtype=torch.int32)
    conv_state_cache_cuda = torch.randn(
        max_batch_size,
        c,
        k - 1,
        device=device,
        dtype=dtype,
    )
    # Clone for Triton backend
    conv_state_cache_triton = conv_state_cache_cuda.clone()

    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    num_prefill = len(lens)
    batch_info_host = torch.tensor([num_prefill, total, 0], device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, lens[0], total], device=device, dtype=torch.int32)
    seq_len = torch.tensor(lens, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_prefill, device=device, dtype=torch.bool)

    # Clone inputs for each backend
    x_cuda = x.clone()
    x_triton = x.clone()

    # Run CUDA backend
    torch.ops.auto_deploy.cuda_cached_causal_conv1d(
        x_cuda,
        w,
        b,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_cuda,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Run Triton backend (includes seq_len parameter)
    torch.ops.auto_deploy.triton_cached_causal_conv1d(
        x_triton,
        w,
        b,
        batch_info_host,
        seq_len,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_triton,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Compare outputs
    assert x_cuda.shape == x_triton.shape
    assert torch.allclose(x_cuda, x_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]), (
        f"Output mismatch: max diff = {(x_cuda - x_triton).abs().max()}"
    )

    # Compare cache states
    assert torch.allclose(
        conv_state_cache_cuda, conv_state_cache_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]
    ), (
        f"Cache state mismatch: max diff = {(conv_state_cache_cuda - conv_state_cache_triton).abs().max()}"
    )


def test_mixed_prefill_decode_triton_vs_cuda(conv_env):
    """Test that Triton backend matches CUDA backend for mixed prefill + decode batch."""
    device = conv_env["device"]
    dtype = conv_env["dtype"]

    # Mixed batch: 1 prefill sequence (len=3) + 2 decode tokens
    prefill_lens = [3]
    num_prefill = len(prefill_lens)
    num_prefill_tokens = sum(prefill_lens)
    num_decode = 2

    total_tokens = num_prefill_tokens + num_decode
    batch, seq = 1, total_tokens
    c, k = 4, 3
    x, w, b, s, p, d, g, pm = _random_params_depthwise(device, dtype, batch, seq, c, k)

    max_batch_size = 4
    # Slot indices: first for prefill, then for decode
    slot_idx = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    conv_state_cache_cuda = torch.randn(
        max_batch_size,
        c,
        k - 1,
        device=device,
        dtype=dtype,
    )
    conv_state_cache_triton = conv_state_cache_cuda.clone()

    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    batch_info_host = torch.tensor(
        [num_prefill, num_prefill_tokens, num_decode], device=device, dtype=torch.int32
    )
    cu_seqlen = torch.tensor([0, prefill_lens[0]], device=device, dtype=torch.int32)
    seq_len = torch.tensor(prefill_lens, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_prefill, device=device, dtype=torch.bool)

    # Clone inputs
    x_cuda = x.clone()
    x_triton = x.clone()

    # Run CUDA backend
    torch.ops.auto_deploy.cuda_cached_causal_conv1d(
        x_cuda,
        w,
        b,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_cuda,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Run Triton backend
    torch.ops.auto_deploy.triton_cached_causal_conv1d(
        x_triton,
        w,
        b,
        batch_info_host,
        seq_len,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_triton,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Compare outputs
    assert x_cuda.shape == x_triton.shape
    assert torch.allclose(x_cuda, x_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]), (
        f"Output mismatch: max diff = {(x_cuda - x_triton).abs().max()}"
    )

    # Compare cache states
    assert torch.allclose(
        conv_state_cache_cuda, conv_state_cache_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]
    ), (
        f"Cache state mismatch: max diff = {(conv_state_cache_cuda - conv_state_cache_triton).abs().max()}"
    )


def test_larger_batch_triton_vs_cuda(conv_env):
    """Test with larger batch and longer sequences."""
    device = conv_env["device"]
    dtype = conv_env["dtype"]

    # Multiple prefill sequences
    lens = [5, 8, 3, 10]
    total = sum(lens)
    batch, seq = 1, total
    c, k = 16, 4
    x, w, b, s, p, d, g, pm = _random_params_depthwise(device, dtype, batch, seq, c, k)

    max_batch_size = 8
    slot_idx = torch.tensor([3, 1, 5, 0], device=device, dtype=torch.int32)
    conv_state_cache_cuda = torch.randn(
        max_batch_size,
        c,
        k - 1,
        device=device,
        dtype=dtype,
    )
    conv_state_cache_triton = conv_state_cache_cuda.clone()

    num_prefill = len(lens)
    batch_info_host = torch.tensor([num_prefill, total, 0], device=device, dtype=torch.int32)

    # Build cumulative sequence lengths
    cu_seqlen_list = [0]
    for ln in lens:
        cu_seqlen_list.append(cu_seqlen_list[-1] + ln)
    cu_seqlen = torch.tensor(cu_seqlen_list, device=device, dtype=torch.int32)
    seq_len = torch.tensor(lens, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_prefill, device=device, dtype=torch.bool)

    x_cuda = x.clone()
    x_triton = x.clone()

    # Run CUDA backend
    torch.ops.auto_deploy.cuda_cached_causal_conv1d(
        x_cuda,
        w,
        b,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_cuda,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Run Triton backend
    torch.ops.auto_deploy.triton_cached_causal_conv1d(
        x_triton,
        w,
        b,
        batch_info_host,
        seq_len,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        conv_state_cache_triton,
        s,
        p,
        d,
        g,
        pm,
        None,
    )

    # Compare outputs
    assert torch.allclose(x_cuda, x_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]), (
        f"Output mismatch: max diff = {(x_cuda - x_triton).abs().max()}"
    )

    # Compare cache states
    assert torch.allclose(
        conv_state_cache_cuda, conv_state_cache_triton, atol=conv_env["atol"], rtol=conv_env["rtol"]
    ), (
        f"Cache state mismatch: max diff = {(conv_state_cache_cuda - conv_state_cache_triton).abs().max()}"
    )
