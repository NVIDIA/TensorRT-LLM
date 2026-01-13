"""Unit tests for CUDA-backed cached causal conv1d custom ops.

Covers:
- Generate-only path with slot-indexed cache mapping
- Context (flattened) path and state write-back per slot
- Metadata preparation
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


def test_generate_only_with_slot_mapping_cuda(conv_env):
    device = conv_env["device"]
    dtype = conv_env["dtype"]

    batch, seq = 1, 1
    c, k = 2, 3
    x, w, b, s, p, d, g, pm = _random_params_depthwise(device, dtype, batch, seq, c, k)

    # Slot mapping with arbitrary order within max_batch_size
    max_batch_size = 2
    slot_idx = torch.tensor([0], device=device, dtype=torch.int32)
    # Cache holds K-1 entries (TRT update kernel contract)
    conv_state_cache = torch.randn(
        max_batch_size,
        c,
        k - 1,
        device=device,
        dtype=dtype,
    )

    # Metadata (not used in generate-only op entry, but required by the interface)
    cu_seqlen = torch.zeros(batch, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(batch, device=device, dtype=torch.bool)
    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    # For generate-only: num_decode = batch, num_prefill = 0
    batch_info_host = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)
    # Snapshot caches for reference before running op (op mutates caches)
    gathered_before = conv_state_cache.clone().index_select(0, slot_idx)
    x_ref = x.clone()
    # Run CUDA cached op (modifies x in-place and returns None)
    torch.ops.auto_deploy.cuda_cached_causal_conv1d(
        # INPUTS
        x,
        w,
        b,
        # STANDARD METADATA
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        # CACHES
        conv_state_cache,
        # CONSTANTS
        s,
        p,
        d,
        g,
        pm,
        None,
    )
    y = x  # The op modifies x in-place

    assert y.shape == (batch, seq, c)
    assert torch.isfinite(y).all()

    # Reference via torch uncached conv on window [state(K-1) | x]
    window_bt_c = torch.cat([gathered_before.transpose(1, 2), x_ref], dim=-2)
    y_ref = torch.ops.auto_deploy.torch_causal_conv1d(window_bt_c, w, b, 1, 0, 1, g, pm)
    assert torch.allclose(y, y_ref, atol=conv_env["atol"], rtol=conv_env["rtol"])
    after = conv_state_cache.index_select(0, slot_idx)
    expected_after = torch.cat([gathered_before[..., 1:], x_ref.transpose(1, 2)[..., :1]], dim=-1)
    assert torch.allclose(
        after, expected_after.to(after.dtype), atol=conv_env["atol"], rtol=conv_env["rtol"]
    )


def test_context_flattened_and_state_writeback_cuda(conv_env):
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
    conv_state_cache = torch.randn(
        max_batch_size,
        c,
        k - 1,
        device=device,
        dtype=dtype,
    )

    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    num_prefill = len(lens)
    batch_info_host = torch.tensor([num_prefill, total, 0], device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, lens[0], total], device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_prefill, device=device, dtype=torch.bool)

    # Snapshot input for reference before running op (op mutates x)
    x_ref = x.clone()

    # Run CUDA cached op (modifies x in-place and returns None)
    torch.ops.auto_deploy.cuda_cached_causal_conv1d(
        # INPUTS
        x,
        w,
        b,
        # STANDARD METADATA
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        # CACHES
        conv_state_cache,
        # CONSTANTS
        s,
        p,
        d,
        g,
        pm,
        None,
    )
    y = x  # The op modifies x in-place

    assert y.shape == (batch, seq, c)
    assert torch.isfinite(y).all()

    # Reference by per-sequence prefill output and expected conv state (K-1 window)
    y_ref = torch.empty_like(y)
    for i, ln in enumerate(lens):
        st = 0 if i == 0 else lens[0]
        x_i = x_ref[:, st : st + ln]
        y_i, _ = (
            tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_backend_causal_conv._torch_causal_conv1d_prefill(  # type: ignore  # noqa: E501
                x_i, w, b, s, p, d, g, pm
            )
        )
        y_ref[:, st : st + ln].copy_(y_i)
        # Cache should hold K-1 latest inputs
        x_b_c_t = x_i.transpose(1, 2)
        if ln >= (k - 1):
            expected_state = x_b_c_t[..., -(k - 1) :]
        else:
            pad = (k - 1) - ln
            expected_state = torch.nn.functional.pad(x_b_c_t, (pad, 0))
        assert torch.allclose(
            conv_state_cache[slot_idx[i]].to(expected_state.dtype),
            expected_state,
            atol=conv_env["atol"],
            rtol=conv_env["rtol"],
        )

    assert torch.allclose(y, y_ref.to(y.dtype), atol=conv_env["atol"], rtol=conv_env["rtol"])
