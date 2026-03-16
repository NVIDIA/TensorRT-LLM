"""Unit tests for the torch_cached_gated_delta_rule custom op.

Covers:
- Decode-only path: batch of single tokens, verify output and final state
  match ``_torch_gated_delta_step`` called directly with preprocessed inputs.
- Prefill-only path: batch of multi-token sequences, verify output and final
  state match ``_torch_gated_delta_prefill`` with preprocessed inputs.
- Prefill with initial state: same as prefill but with ``use_initial_states=True``
  and non-zero initial cache, verifying the cache history is correctly loaded.

The cached op accepts raw (un-normalized, un-expanded) q/k with raw gating
projections (a, b, A_log, dt_bias). L2 normalization, GQA expansion, and
gating are performed internally.
"""

import pytest
import torch
import torch.nn.functional as F

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
    """Generate random gated delta rule inputs (raw, un-preprocessed)."""
    q = torch.randn(batch, seq, num_heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, num_heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, num_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(batch, seq, num_heads, device=device, dtype=dtype)
    b = torch.randn(batch, seq, num_heads, device=device, dtype=dtype)
    A_log = torch.zeros(num_heads, device=device, dtype=dtype)
    dt_bias = torch.zeros(num_heads, device=device, dtype=dtype)
    return q, k, v, a, b, A_log, dt_bias


def _preprocess_for_reference(q, k, a, b_proj, A_log, dt_bias):
    """Manually preprocess raw inputs for use with reference helpers."""
    q_norm = F.normalize(q.float(), dim=-1).to(q.dtype)
    k_norm = F.normalize(k.float(), dim=-1).to(k.dtype)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta = b_proj.float().sigmoid()
    return q_norm, k_norm, g, beta


def test_decode_only(gated_delta_env):
    """Decode-only: batch of single tokens through the cached op.

    Verifies output and cache state match _torch_gated_delta_step with
    manually preprocessed inputs.
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

    q, k, v, a, b, A_log, dt_bias = _random_inputs(
        device,
        dtype,
        batch,
        seq,
        num_heads,
        key_dim,
        value_dim,
    )

    slot_idx = torch.tensor([5, 1, 3, 0], device=device, dtype=torch.int32)

    delta_cache = torch.randn(
        max_batch_size,
        num_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )

    batch_info_host = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)
    cu_seqlen = torch.zeros(1, device=device, dtype=torch.int32)
    use_initial_states = torch.ones(batch, device=device, dtype=torch.bool)

    gathered_before = delta_cache.clone().index_select(0, slot_idx.long())

    y = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        delta_cache,
        scale,
    )

    assert y.shape == (batch, seq, num_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: preprocess and call _torch_gated_delta_step directly
    q_norm, k_norm, g, beta = _preprocess_for_reference(q, k, a, b, A_log, dt_bias)

    y_ref_list = []
    final_state_ref_list = []
    for i in range(batch):
        o_ref, s_ref = _torch_gated_delta_step(
            q_norm[i, 0].unsqueeze(0),
            k_norm[i, 0].unsqueeze(0),
            v[i, 0].unsqueeze(0),
            g[i, 0].unsqueeze(0),
            beta[i, 0].unsqueeze(0),
            gathered_before[i].unsqueeze(0),
            scale,
        )
        y_ref_list.append(o_ref.squeeze(0))
        final_state_ref_list.append(s_ref.squeeze(0))

    y_ref = torch.stack(y_ref_list, dim=0).unsqueeze(1).to(dtype)
    final_state_ref = torch.stack(final_state_ref_list, dim=0)

    torch.testing.assert_close(y, y_ref, atol=1e-3, rtol=1e-3)

    after = delta_cache.index_select(0, slot_idx.long())
    torch.testing.assert_close(
        after,
        final_state_ref.to(after.dtype),
        atol=1e-3,
        rtol=1e-3,
    )


def test_prefill_only(gated_delta_env):
    """Prefill-only: two sequences of different lengths, flattened.

    Verifies output and final state match _torch_gated_delta_prefill with
    manually preprocessed inputs.
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

    q, k, v, a, b, A_log, dt_bias = _random_inputs(
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

    num_prefill = len(seq_lens)
    batch_info_host = torch.tensor(
        [num_prefill, total_tokens, 0],
        device=device,
        dtype=torch.int32,
    )
    cu_seqlen = torch.tensor([0, seq_lens[0], total_tokens], device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_prefill, device=device, dtype=torch.bool)

    y = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        delta_cache,
        scale,
    )

    assert y.shape == (1, total_tokens, num_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: preprocess and call _torch_gated_delta_prefill per sequence
    q_norm, k_norm, g, beta = _preprocess_for_reference(q, k, a, b, A_log, dt_bias)

    y_ref = torch.empty_like(y)
    for i, sl in enumerate(seq_lens):
        start = sum(seq_lens[:i])
        end = start + sl

        init_state = torch.zeros(
            1,
            num_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )

        y_seq, final_state = _torch_gated_delta_prefill(
            q_norm[:, start:end],
            k_norm[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale,
            init_state,
        )

        y_ref[:, start:end] = y_seq.to(dtype)

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

    q, k, v, a, b, A_log, dt_bias = _random_inputs(
        device,
        dtype,
        1,
        seq_len,
        num_heads,
        key_dim,
        value_dim,
    )

    slot_idx = torch.tensor([1], device=device, dtype=torch.int32)

    delta_cache = torch.randn(
        max_batch_size,
        num_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )
    initial_state = delta_cache[1].clone()

    batch_info_host = torch.tensor([1, seq_len, 0], device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    use_initial_states = torch.tensor([True], device=device, dtype=torch.bool)

    y_with_init = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        delta_cache,
        scale,
    )

    # Reference: preprocess and call _torch_gated_delta_prefill
    q_norm, k_norm, g, beta = _preprocess_for_reference(q, k, a, b, A_log, dt_bias)

    y_ref, final_ref = _torch_gated_delta_prefill(
        q_norm,
        k_norm,
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

    # Verify it differs from running WITHOUT initial state
    delta_cache_zero = torch.zeros_like(delta_cache)
    use_initial_states_false = torch.tensor([False], device=device, dtype=torch.bool)

    y_without_init = torch.ops.auto_deploy.torch_cached_gated_delta_rule(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states_false,
        delta_cache_zero,
        scale,
    )

    assert not torch.allclose(y_with_init, y_without_init, atol=1e-3, rtol=1e-3), (
        "Output with initial state should differ from output without initial state"
    )
