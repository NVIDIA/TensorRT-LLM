import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def _random_params(device, dtype, batch, seq, num_heads, head_dim, n_groups, ssm_state_size):
    hidden_states = torch.randn(batch, seq, num_heads, head_dim, device=device, dtype=dtype)
    A = torch.randn(num_heads, device=device, dtype=torch.float32)
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


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5548861")
def test_triton_generate_only_with_slot_mapping(mamba_env):
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

    max_batch_size = 6
    slot_idx = torch.tensor([4, 1, 3], device=device, dtype=torch.int32)
    ssm_state_cache_torch = torch.randn(
        max_batch_size, num_heads, head_dim, ssm_state_size, device=device, dtype=dtype
    )
    ssm_state_cache_triton = ssm_state_cache_torch.clone()

    seq_len = torch.ones(batch, device=device, dtype=torch.int32)
    seq_start = torch.zeros(batch, device=device, dtype=torch.int32)

    # Torch reference
    y_torch = torch.ops.auto_deploy.torch_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        seq_len,
        seq_start,
        slot_idx,
        ssm_state_cache_torch,
        time_step_limit,
        chunk_size,
    )

    # Triton under test
    y_triton = torch.ops.auto_deploy.triton_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        seq_len,
        seq_start,
        slot_idx,
        ssm_state_cache_triton,
        time_step_limit,
        chunk_size,
    )

    assert y_triton.shape == hidden_states.shape
    assert torch.isfinite(y_triton).all()

    # Compare outputs
    assert torch.allclose(y_triton, y_torch.to(y_triton.dtype), atol=atol, rtol=rtol)

    # Compare cache updates at slots
    after_torch = ssm_state_cache_torch.index_select(0, slot_idx)
    after_triton = ssm_state_cache_triton.index_select(0, slot_idx)
    assert torch.allclose(after_triton.to(after_torch.dtype), after_torch, atol=atol, rtol=rtol)


def test_triton_context_flattened_and_state_writeback(mamba_env):
    device = mamba_env["device"]
    dtype = mamba_env["dtype"]
    atol = mamba_env["atol"]
    rtol = mamba_env["rtol"]

    lens = [2]
    total = sum(lens)
    batch, seq = 1, total
    num_heads, head_dim = 1, 4
    n_groups, ssm_state_size = 1, 1
    (hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size) = _random_params(
        device, dtype, batch, seq, num_heads, head_dim, n_groups, ssm_state_size
    )

    max_batch_size = 2
    slot_idx = torch.tensor([1], device=device, dtype=torch.int32)
    ssm_state_cache_torch = torch.randn(
        max_batch_size, num_heads, head_dim, ssm_state_size, device=device, dtype=dtype
    )
    ssm_state_cache_triton = ssm_state_cache_torch.clone()

    seq_len = torch.tensor(lens, device=device, dtype=torch.int32)
    seq_start = torch.tensor([0, lens[0]], device=device, dtype=torch.int32)
    use_initial_states = torch.tensor([0] * batch, device=device).to(torch.bool)
    # Torch reference
    y_torch = torch.ops.auto_deploy.torch_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        seq_len,
        seq_start,
        slot_idx,
        use_initial_states,
        ssm_state_cache_torch,
        time_step_limit,
        chunk_size,
    )

    # Triton under test
    y_triton = torch.ops.auto_deploy.triton_cached_ssm(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        seq_len,
        seq_start,
        slot_idx,
        use_initial_states,
        ssm_state_cache_triton,
        time_step_limit,
        chunk_size,
    )

    assert y_triton.shape == hidden_states.shape
    assert torch.isfinite(y_triton).all()
    # Compare outputs
    assert torch.allclose(y_triton, y_torch.to(y_triton.dtype), atol=1e-1, rtol=1e-1)

    # Cache should hold final state at slots
    for i, ln in enumerate(lens):
        slot = slot_idx[i]
        state_torch = ssm_state_cache_torch[slot]
        state_triton = ssm_state_cache_triton[slot]
        assert torch.allclose(state_triton.to(state_torch.dtype), state_torch, atol=atol, rtol=rtol)
