"""Unit tests for the fla_cached_gated_delta_rule custom op.

Covers:
- Decode-only path: batch of single tokens, verify output and final state
  match ``fused_recurrent_gated_delta_rule_fwd`` called directly with gathered
  initial states.
- Prefill-only path: batch of multi-token sequences (variable length), verify
  output and final state match per-sequence ``chunk_gated_delta_rule``.
- Prefill with initial state: same as prefill but with ``use_initial_states=True``
  and non-zero initial cache, verifying the cache history is correctly loaded and
  passed to the kernel.
- GVA (Grouped Value Attention): q/k have fewer heads than v/g/beta.
"""

import pytest
import torch

# Register all auto_deploy custom ops
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_fwd


@pytest.fixture
def gdr_env():
    device = "cuda"
    dtype = torch.bfloat16
    # FLA Triton kernels do not support float32
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    return {"device": device, "dtype": dtype}


def _random_inputs(device, dtype, batch, seq, num_k_heads, num_v_heads, key_dim, value_dim):
    """Generate random gated delta rule inputs.

    q/k have num_k_heads, v/g/beta have num_v_heads (GVA when num_v_heads > num_k_heads).
    q/k are NOT L2-normalized here; normalization is handled inside the op.
    """
    q = torch.randn(batch, seq, num_k_heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, num_k_heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, num_v_heads, value_dim, device=device, dtype=dtype)
    g = -torch.rand(batch, seq, num_v_heads, device=device, dtype=dtype)  # negative (decay)
    beta = torch.sigmoid(torch.randn(batch, seq, num_v_heads, device=device, dtype=dtype))
    return q, k, v, g, beta


@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 2), (2, 4)])
def test_decode_only(gdr_env, num_k_heads, num_v_heads):
    """Decode-only: batch of single tokens through the cached op.

    Verifies output and cache state match fused_recurrent_gated_delta_rule_fwd
    called directly with gathered initial states.
    """
    device = gdr_env["device"]
    dtype = gdr_env["dtype"]
    atol = 5e-3
    rtol = 5e-3

    batch = 3
    seq = 1
    key_dim = 8
    value_dim = 8
    max_batch_size = 6
    scale = key_dim**-0.5

    q, k, v, g, beta = _random_inputs(
        device, dtype, batch, seq, num_k_heads, num_v_heads, key_dim, value_dim
    )

    # Slot mapping with arbitrary order
    slot_idx = torch.tensor([4, 1, 3], device=device, dtype=torch.int32)

    # Initialize cache with random state (simulating existing history)
    # Cache shape uses num_v_heads (HV), not num_k_heads
    delta_cache = torch.randn(
        max_batch_size,
        num_v_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=dtype,
    )

    # Metadata for decode-only: no prefill
    batch_info_host = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)
    cu_seqlen = torch.zeros(1, device=device, dtype=torch.int32)
    use_initial_states = torch.ones(batch, device=device, dtype=torch.bool)

    # Snapshot cache before mutation for reference
    gathered_before = delta_cache.clone().index_select(0, slot_idx.long())

    # Run cached op
    y = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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

    assert y.shape == (batch, seq, num_v_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: call fused_recurrent_gated_delta_rule_fwd directly
    # q/k have num_k_heads, v/g/beta have num_v_heads
    q_flat = q.view(batch, num_k_heads, -1)  # [B, Hg, K]
    k_flat = k.view(batch, num_k_heads, -1)  # [B, Hg, K]
    v_flat = v.view(batch, num_v_heads, -1)  # [B, HV, V]
    g_flat = g.view(batch, num_v_heads)  # [B, HV]
    beta_flat = beta.view(batch, num_v_heads)  # [B, HV]

    y_ref, final_state_ref = fused_recurrent_gated_delta_rule_fwd(
        q=q_flat[:, None],  # [B, 1, Hg, K]
        k=k_flat[:, None],  # [B, 1, Hg, K]
        v=v_flat[:, None],  # [B, 1, HV, V]
        g=g_flat[:, None],  # [B, 1, HV]
        beta=beta_flat[:, None],  # [B, 1, HV]
        scale=scale,
        initial_state=gathered_before.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    # y_ref shape: [B, 1, HV, V] -> compare against y: [B, 1, HV, V]
    y_ref_reshaped = y_ref.to(dtype)
    torch.testing.assert_close(y, y_ref_reshaped, atol=atol, rtol=rtol)

    # Compare updated cache states
    after = delta_cache.index_select(0, slot_idx.long())
    torch.testing.assert_close(
        after,
        final_state_ref.to(after.dtype),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 2), (2, 4)])
def test_prefill_only(gdr_env, num_k_heads, num_v_heads):
    """Prefill-only: two sequences of different lengths, flattened.

    Verifies output and final state match per-sequence chunk_gated_delta_rule.
    """
    device = gdr_env["device"]
    dtype = gdr_env["dtype"]
    atol = 5e-3
    rtol = 5e-3

    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
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
        num_k_heads,
        num_v_heads,
        key_dim,
        value_dim,
    )

    slot_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    # Cache shape uses num_v_heads
    delta_cache = torch.zeros(
        max_batch_size,
        num_v_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=dtype,
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
    y = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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

    assert y.shape == (1, total_tokens, num_v_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: call chunk_gated_delta_rule per sequence
    y_ref = torch.empty_like(y)
    for i, sl in enumerate(seq_lens):
        start = sum(seq_lens[:i])
        end = start + sl

        # chunk_gated_delta_rule expects [B, T, H, D] layout
        y_seq, final_state = chunk_gated_delta_rule(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            scale=scale,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        y_ref[:, start:end] = y_seq.to(dtype)

        # Verify cache was updated for this slot
        torch.testing.assert_close(
            delta_cache[slot_idx[i].long()],
            final_state.squeeze(0).to(delta_cache.dtype),
            atol=atol,
            rtol=rtol,
        )

    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 2), (2, 4)])
def test_prefill_with_initial_state(gdr_env, num_k_heads, num_v_heads):
    """Prefill with initial state: verifies cache history is correctly loaded.

    Sets use_initial_states=True and a non-zero initial cache, then checks that
    the result matches chunk_gated_delta_rule called with the same initial state
    and differs from running without initial state.
    """
    device = gdr_env["device"]
    dtype = gdr_env["dtype"]
    atol = 5e-3
    rtol = 5e-3

    seq_len = 4
    key_dim = 8
    value_dim = 8
    max_batch_size = 2
    scale = key_dim**-0.5

    q, k, v, g, beta = _random_inputs(
        device, dtype, 1, seq_len, num_k_heads, num_v_heads, key_dim, value_dim
    )

    slot_idx = torch.tensor([1], device=device, dtype=torch.int32)

    # Non-zero initial state in cache (uses num_v_heads)
    delta_cache = torch.randn(
        max_batch_size,
        num_v_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=dtype,
    )
    initial_state = delta_cache[1].clone()  # snapshot

    # Metadata: one prefill sequence with initial state
    batch_info_host = torch.tensor([1, seq_len, 0], device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    use_initial_states = torch.tensor([True], device=device, dtype=torch.bool)

    # Run cached op WITH initial state
    y_with_init = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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

    # Reference: chunk_gated_delta_rule with the same initial state
    y_ref, final_ref = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.unsqueeze(0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(y_with_init, y_ref.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        delta_cache[1],
        final_ref.squeeze(0).to(delta_cache.dtype),
        atol=atol,
        rtol=rtol,
    )

    # Also verify it's different from running WITHOUT initial state (zero state)
    delta_cache_zero = torch.zeros_like(delta_cache)
    use_initial_states_false = torch.tensor([False], device=device, dtype=torch.bool)

    y_without_init = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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
