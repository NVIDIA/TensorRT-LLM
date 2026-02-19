"""Unit tests for cached causal conv1d custom ops.

Covers:
- Generate-only path with slot-indexed cache mapping
- Context (flattened) path and state write-back per slot
- Metadata preparation
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def _random_params(device, dtype, batch, seq, c_in, c_out, k, groups=1):
    x = torch.randn(batch, seq, c_in, device=device, dtype=dtype)
    weight = torch.randn(c_out, c_in // groups, k, device=device, dtype=dtype)
    bias = torch.randn(c_out, device=device, dtype=dtype)
    stride = 1
    padding = k - 1
    dilation = 1
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


def test_generate_only_with_slot_mapping(conv_env):
    device = conv_env["device"]
    dtype = conv_env["dtype"]

    batch, seq = 3, 1
    c_in, c_out, k = 8, 6, 3
    x, w, b, s, p, d, g, pm = _random_params(device, dtype, batch, seq, c_in, c_out, k)

    # Slot mapping with arbitrary order within max_batch_size
    max_batch_size = 6
    slot_idx = torch.tensor([4, 1, 3], device=device, dtype=torch.int32)
    conv_state_cache = torch.randn(
        max_batch_size,
        c_in,
        k,
        device=device,
        dtype=dtype,
    )

    # Metadata (not used in generate-only op entry, but required by the interface)
    seq_len = torch.ones(batch, device=device, dtype=torch.int32)
    cu_seqlen = torch.zeros(batch, device=device, dtype=torch.int32)
    # Snapshot caches for reference before running op (op mutates caches)
    gathered_before = conv_state_cache.clone().index_select(0, slot_idx)
    use_initial_states = torch.zeros(batch, device=device, dtype=torch.bool)
    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    # For generate-only: num_decode = batch, num_prefill = 0
    batch_info_host = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)
    # Run cached op
    y = torch.ops.auto_deploy.torch_cached_causal_conv1d(
        # INPUTS
        x,
        w,
        b,
        # STANDARD METADATA
        batch_info_host,
        seq_len,
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
    )

    assert y.shape == (batch, seq, c_out)
    assert torch.isfinite(y).all()

    # Reference: use pre-op gathered states, run decode helper directly, compare
    y_ref, updated = (
        tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_backend_causal_conv._torch_causal_conv1d_decode(  # type: ignore  # noqa: E501
            x, w, b, s, p, d, g, pm, gathered_before
        )
    )
    assert torch.allclose(y, y_ref, atol=conv_env["atol"], rtol=conv_env["rtol"])
    after = conv_state_cache.index_select(0, slot_idx)
    assert torch.allclose(
        after, updated.to(after.dtype), atol=conv_env["atol"], rtol=conv_env["rtol"]
    )


def test_context_flattened_and_state_writeback(conv_env):
    device = conv_env["device"]
    dtype = conv_env["dtype"]

    # Two sequences with lengths 3 and 2, flattened to [1,5]
    lens = [3, 2]
    total = sum(lens)
    batch, seq = 1, total
    c_in, c_out, k = 8, 6, 3
    x, w, b, s, p, d, g, pm = _random_params(device, dtype, batch, seq, c_in, c_out, k)

    max_batch_size = 4
    slot_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    conv_state_cache = torch.randn(
        max_batch_size,
        c_in,
        k,
        device=device,
        dtype=dtype,
    )

    seq_len = torch.tensor(lens, device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, lens[0]], device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(batch, device=device, dtype=torch.bool)
    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    # For context/prefill phase: num_prefill = len(lens), num_decode = 0
    num_seqs = len(lens)
    num_prefill_tokens = sum(lens)
    batch_info_host = torch.tensor(
        [num_seqs, num_prefill_tokens, 0], device=device, dtype=torch.int32
    )
    y = torch.ops.auto_deploy.torch_cached_causal_conv1d(
        # INPUTS
        x,
        w,
        b,
        # STANDARD METADATA
        batch_info_host,
        seq_len,
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
    )

    assert y.shape == (batch, seq, c_out)
    assert torch.isfinite(y).all()

    # Reference by per-sequence prefill
    y_ref = torch.empty_like(y)
    for i, ln in enumerate(lens):
        st = 0 if i == 0 else lens[0]
        x_i = x[:, st : st + ln]
        y_i, s_i = (
            tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_backend_causal_conv._torch_causal_conv1d_prefill(  # type: ignore  # noqa: E501
                x_i, w, b, s, p, d, g, pm
            )
        )
        y_ref[:, st : st + ln].copy_(y_i)
        # Cache should hold final state at slot
        assert torch.allclose(
            conv_state_cache[slot_idx[i]].to(s_i.dtype),
            s_i,
            atol=conv_env["atol"],
            rtol=conv_env["rtol"],
        )

    assert torch.allclose(y, y_ref.to(y.dtype), atol=conv_env["atol"], rtol=conv_env["rtol"])
