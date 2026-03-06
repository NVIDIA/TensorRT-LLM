"""Unit tests for the torch_cached_gated_delta_rule custom op.

Covers:
- Decode-only path: batch of single tokens, verify output and final state
  match ``_torch_gated_delta_step`` called directly.
- Prefill-only path: batch of multi-token sequences, verify output and final
  state match ``_torch_gated_delta_prefill``.
- Prefill with initial state: same as prefill but with ``use_initial_states=True``
  and non-zero initial cache, verifying the cache history is correctly loaded.
"""

import pytest
import torch

# Register all auto_deploy custom ops
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.fla.torch_backend_gated_delta import (
    _torch_gated_delta_prefill,
    _torch_gated_delta_step,
)


@pytest.fixture
def gated_delta_env():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    return {"device": device, "dtype": dtype}


def _random_inputs(device, dtype, batch, seq, num_heads, key_dim, value_dim):
    """Generate random gated delta rule inputs."""
    q = torch.randn(batch, seq, num_heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, num_heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, num_heads, value_dim, device=device, dtype=dtype)
    g = -torch.rand(batch, seq, num_heads, device=device, dtype=dtype)  # negative (decay)
    beta = torch.sigmoid(torch.randn(batch, seq, num_heads, device=device, dtype=dtype))

    # L2 normalize Q and K as the patched forward does
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    return q, k, v, g, beta


def test_decode_only(gated_delta_env):
    """Decode-only: batch of single tokens through the cached op.

    Verifies output and cache state match _torch_gated_delta_step.
    """
    device = gated_delta_env["device"]
    dtype = gated_delta_env["dtype"]

    batch = 4
    seq = 1
    num_heads = 2
    key_dim = 8
    value_dim = 8
    max_batch_size = 6
    scale = key_dim**-0.5

    q, k, v, g, beta = _random_inputs(device, dtype, batch, seq, num_heads, key_dim, value_dim)

    # Slot mapping with arbitrary order
    slot_idx = torch.tensor([5, 1, 3, 0], device=device, dtype=torch.int32)

    # Initialize cache with random state (simulating existing history)
    delta_cache = torch.randn(
        max_batch_size,
        num_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )

    # Metadata for decode-only: no prefill
    batch_info_host = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)
    cu_seqlen = torch.zeros(1, device=device, dtype=torch.int32)
    use_initial_states = torch.ones(batch, device=device, dtype=torch.bool)

    # Snapshot cache before mutation for reference
    gathered_before = delta_cache.clone().index_select(0, slot_idx.long())

    # Run cached op
    y = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        delta_cache,
        scale,
    )

    assert y.shape == (batch, seq, num_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: call _torch_gated_delta_step directly per sequence
    y_ref_list = []
    final_state_ref_list = []
    for i in range(batch):
        o_ref, s_ref = _torch_gated_delta_step(
            q[i, 0].unsqueeze(0),  # [1, H, K]
            k[i, 0].unsqueeze(0),  # [1, H, K]
            v[i, 0].unsqueeze(0),  # [1, H, V]
            g[i, 0].unsqueeze(0),  # [1, H]
            beta[i, 0].unsqueeze(0),  # [1, H]
            gathered_before[i].unsqueeze(0),  # [1, H, K, V]
            scale,
        )
        y_ref_list.append(o_ref.squeeze(0))  # [H, V]
        final_state_ref_list.append(s_ref.squeeze(0))  # [H, K, V]

    y_ref = torch.stack(y_ref_list, dim=0).unsqueeze(1).to(dtype)  # [B, 1, H, V]
    final_state_ref = torch.stack(final_state_ref_list, dim=0)  # [B, H, K, V]

    # Compare outputs
    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)

    # Compare updated cache states
    after = delta_cache.index_select(0, slot_idx.long())
    torch.testing.assert_close(
        after,
        final_state_ref.to(after.dtype),
        atol=1e-3,
        rtol=1e-3,
    )


def test_prefill_only(gated_delta_env):
    """Prefill-only: two sequences of different lengths, flattened.

    Verifies output and final state match _torch_gated_delta_prefill.
    """
    device = gated_delta_env["device"]
    dtype = gated_delta_env["dtype"]

    seq_lens = [3, 5]
    total_tokens = sum(seq_lens)
    num_heads = 2
    key_dim = 8
    value_dim = 8
    max_batch_size = 4
    scale = key_dim**-0.5

    # Create flattened inputs: [1, total_tokens, H, D]
    q, k, v, g, beta = _random_inputs(
        device,
        dtype,
        1,
        total_tokens,
        num_heads,
        key_dim,
        value_dim,
    )

    slot_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    delta_cache = torch.zeros(
        max_batch_size,
        num_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )

    # Metadata for prefill-only
    num_prefill = len(seq_lens)
    batch_info_host = torch.tensor(
        [num_prefill, total_tokens, 0],
        device=device,
        dtype=torch.int32,
    )
    cu_seqlen = torch.tensor([0, seq_lens[0], total_tokens], device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_prefill, device=device, dtype=torch.bool)

    # Run cached op
    y = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        delta_cache,
        scale,
    )

    assert y.shape == (1, total_tokens, num_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: run _torch_gated_delta_prefill per sequence
    y_ref = torch.empty_like(y)
    for i, sl in enumerate(seq_lens):
        start = sum(seq_lens[:i])
        end = start + sl

        q_seq = q[:, start:end]
        k_seq = k[:, start:end]
        v_seq = v[:, start:end]
        g_seq = g[:, start:end]
        beta_seq = beta[:, start:end]

        init_state = torch.zeros(
            1,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )

        y_seq, final_state = _torch_gated_delta_prefill(
            q_seq,
            k_seq,
            v_seq,
            g_seq,
            beta_seq,
            scale,
            init_state,
        )

        y_ref[:, start:end] = y_seq.to(dtype)

        # Verify cache was updated for this slot
        torch.testing.assert_close(
            delta_cache[slot_idx[i].long()],
            final_state.squeeze(0).to(delta_cache.dtype),
            atol=1e-3,
            rtol=1e-3,
        )

    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)


def test_prefill_with_initial_state(gated_delta_env):
    """Prefill with initial state: verifies cache history is correctly loaded.

    Sets use_initial_states=True and a non-zero initial cache, then checks that
    the result differs from prefill without initial state.
    """
    device = gated_delta_env["device"]
    dtype = gated_delta_env["dtype"]

    seq_len = 4
    num_heads = 2
    key_dim = 8
    value_dim = 8
    max_batch_size = 2
    scale = key_dim**-0.5

    q, k, v, g, beta = _random_inputs(device, dtype, 1, seq_len, num_heads, key_dim, value_dim)

    slot_idx = torch.tensor([1], device=device, dtype=torch.int32)

    # Non-zero initial state in cache
    delta_cache = torch.randn(
        max_batch_size,
        num_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )
    initial_state = delta_cache[1].clone()  # snapshot

    # Metadata: one prefill sequence with initial state
    batch_info_host = torch.tensor([1, seq_len, 0], device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    use_initial_states = torch.tensor([True], device=device, dtype=torch.bool)

    # Run cached op WITH initial state
    y_with_init = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        delta_cache,
        scale,
    )

    # Reference: _torch_gated_delta_prefill with the same initial state
    y_ref, final_ref = _torch_gated_delta_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state.unsqueeze(0),
    )

    torch.testing.assert_close(y_with_init, y_ref.to(dtype), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        delta_cache[1],
        final_ref.squeeze(0).to(delta_cache.dtype),
        atol=1e-3,
        rtol=1e-3,
    )

    # Also verify it's different from running WITHOUT initial state (zero state)
    delta_cache_zero = torch.zeros_like(delta_cache)
    use_initial_states_false = torch.tensor([False], device=device, dtype=torch.bool)

    y_without_init = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states_false,
        delta_cache_zero,
        scale,
    )

    # The results should differ when there's a non-zero initial state
    assert not torch.allclose(y_with_init, y_without_init, atol=1e-3, rtol=1e-3), (
        "Output with initial state should differ from output without initial state"
    )
