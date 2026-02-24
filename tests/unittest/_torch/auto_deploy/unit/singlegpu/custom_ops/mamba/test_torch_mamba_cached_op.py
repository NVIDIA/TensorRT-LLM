"""Unit tests for cached Mamba (SSM) custom ops.

Covers:
- Generate-only path with slot-indexed cache mapping
- Context (flattened) path and state write-back per slot
- Metadata preparation
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def _random_params(device, dtype, batch, seq, num_heads, head_dim, n_groups, ssm_state_size):
    hidden_states = torch.randn(batch, seq, num_heads, head_dim, device=device, dtype=dtype)
    A = torch.randn(num_heads, device=device, dtype=torch.float32)
    # B, C live in fp16 typically, but decode/prefill handle casting
    B = torch.randn(batch, seq, n_groups, ssm_state_size, device=device, dtype=dtype)
    C = torch.randn(batch, seq, n_groups, ssm_state_size, device=device, dtype=dtype)
    D = torch.randn(num_heads, device=device, dtype=dtype)
    dt = torch.randn(batch, seq, num_heads, device=device, dtype=dtype)
    dt_bias = torch.randn(num_heads, device=device, dtype=dtype)
    time_step_limit = [1e-6, 1.0]
    chunk_size = 4
    return hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size


@pytest.fixture
def mamba_env():
    device = "cuda"
    dtype = torch.float16
    atol = 5e-2
    rtol = 5e-2
    torch.manual_seed(123)
    torch.cuda.empty_cache()
    return {"device": device, "dtype": dtype, "atol": atol, "rtol": rtol}


def test_generate_only_with_slot_mapping(mamba_env):
    device = mamba_env["device"]
    dtype = mamba_env["dtype"]
    atol = mamba_env["atol"]
    rtol = mamba_env["rtol"]

    batch, seq = 3, 1
    num_heads, head_dim = 4, 8
    n_groups, ssm_state_size = 2, 4
    (hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size) = _random_params(
        device, dtype, batch, seq, num_heads, head_dim, n_groups, ssm_state_size
    )

    # Slot mapping with arbitrary order within max_batch_size
    max_batch_size = 6
    slot_idx = torch.tensor([4, 1, 3], device=device, dtype=torch.int32)
    ssm_state_cache = torch.randn(
        max_batch_size,
        num_heads,
        head_dim,
        ssm_state_size,
        device=device,
        dtype=dtype,
    )

    # Metadata
    seq_len = torch.ones(batch, device=device, dtype=torch.int32)
    cu_seqlen = torch.zeros(batch, device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(batch, device=device, dtype=torch.bool)
    # batch_info_host: [num_prefill, num_prefill_tokens, num_decode]
    # For generate-only: num_decode = batch, num_prefill = 0
    batch_info_host = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)
    # Snapshot caches for reference before running op (op mutates caches)
    gathered_before = ssm_state_cache.clone().index_select(0, slot_idx)

    # Run cached op
    y = torch.ops.auto_deploy.torch_cached_ssm(
        # INPUTS
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        # STANDARD METADATA
        batch_info_host,
        seq_len,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        # CACHES
        ssm_state_cache,
        # CONSTANTS
        time_step_limit,
        chunk_size,
    )

    assert y.shape == hidden_states.shape
    assert torch.isfinite(y).all()

    # Reference: use pre-op gathered states, run decode helper directly, compare
    y_ref, updated = (
        tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_backend_mamba._torch_cached_ssm_decode(  # type: ignore  # noqa: E501
            hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size, gathered_before
        )
    )
    # y close to y_ref
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol)
    # Updated states scattered correctly
    after = ssm_state_cache.index_select(0, slot_idx)
    assert torch.allclose(after, updated.to(after.dtype), atol=atol, rtol=rtol)


def test_context_flattened_and_state_writeback(mamba_env):
    device = mamba_env["device"]
    dtype = mamba_env["dtype"]
    atol = mamba_env["atol"]
    rtol = mamba_env["rtol"]

    # Two sequences with lengths 3 and 2, flattened to [1,5]
    lens = [3, 2]
    total = sum(lens)
    batch, seq = 1, total
    num_heads, head_dim = 4, 8
    n_groups, ssm_state_size = 2, 4
    (hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size) = _random_params(
        device, dtype, batch, seq, num_heads, head_dim, n_groups, ssm_state_size
    )

    max_batch_size = 4
    slot_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    ssm_state_cache = torch.randn(
        max_batch_size,
        num_heads,
        head_dim,
        ssm_state_size,
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
    y = torch.ops.auto_deploy.torch_cached_ssm(
        # INPUTS
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        # STANDARD METADATA
        batch_info_host,
        seq_len,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        # CACHES
        ssm_state_cache,
        # CONSTANTS
        time_step_limit,
        chunk_size,
    )

    assert y.shape == hidden_states.shape
    assert torch.isfinite(y).all()

    # Reference by per-sequence prefill
    y_ref = torch.empty_like(hidden_states)
    for i, ln in enumerate(lens):
        st = 0 if i == 0 else lens[0]
        hs = hidden_states[:, st : st + ln]
        Bb = B[:, st : st + ln]
        Cb = C[:, st : st + ln]
        dtb = dt[:, st : st + ln]
        y_i, s_i = tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_mamba._torch_ssm_prefill(  # type: ignore  # noqa: E501
            hs, A, Bb, Cb, D, dtb, dt_bias, time_step_limit, chunk_size
        )
        y_ref[:, st : st + ln].copy_(y_i)
        # Cache should hold final state at slot
        assert torch.allclose(ssm_state_cache[slot_idx[i]].to(s_i.dtype), s_i, atol=atol, rtol=rtol)

    assert torch.allclose(y, y_ref.to(y.dtype), atol=atol, rtol=rtol)
