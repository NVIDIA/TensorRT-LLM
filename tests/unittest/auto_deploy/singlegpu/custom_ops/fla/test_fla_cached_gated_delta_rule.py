# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the fla_cached_gated_delta_rule custom op.

Covers:
- Decode-only path: batch of single tokens, verify output and final state
  match ``fused_recurrent_gated_delta_rule_fwd`` called directly with gathered
  initial states and manually preprocessed inputs.
- Prefill-only path: batch of multi-token sequences (variable length), verify
  output and final state match per-sequence ``chunk_gated_delta_rule``.
- Prefill with initial state: same as prefill but with ``use_initial_states=True``
  and non-zero initial cache, verifying the cache history is correctly loaded and
  passed to the kernel.
- GVA (Grouped Value Attention): q/k have fewer heads than v (a/b/A_log/dt_bias).

The cached op accepts raw (un-normalized, un-expanded) q/k with raw gating
projections (a, b, A_log, dt_bias). L2 normalization, GQA expansion, and
gating are performed internally.
"""

import pytest
import torch
import torch.nn.functional as F

# Register all auto_deploy custom ops
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
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
    """Generate random gated delta rule inputs (raw, un-preprocessed).

    q/k have num_k_heads, v/a/b have num_v_heads (GVA when num_v_heads > num_k_heads).
    A_log, dt_bias have shape [num_v_heads].
    """
    q = torch.randn(batch, seq, num_k_heads, key_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, num_k_heads, key_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, num_v_heads, value_dim, device=device, dtype=dtype)
    a = torch.randn(batch, seq, num_v_heads, device=device, dtype=dtype)
    b = torch.randn(batch, seq, num_v_heads, device=device, dtype=dtype)
    A_log = torch.zeros(num_v_heads, device=device, dtype=dtype)
    dt_bias = torch.zeros(num_v_heads, device=device, dtype=dtype)
    return q, k, v, a, b, A_log, dt_bias


def _preprocess_for_reference(q, k, a, b_proj, A_log, dt_bias, num_v_heads=None):
    """Manually preprocess raw inputs for use with reference kernels.

    If num_v_heads is given and greater than q.shape[2], GQA-expands q and k
    to num_v_heads. Returns q_norm, k_norm, g, beta ready for chunk/fused_recurrent.
    """
    q_norm = F.normalize(q.float(), dim=-1).to(q.dtype)
    k_norm = F.normalize(k.float(), dim=-1).to(k.dtype)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta = b_proj.float().sigmoid()
    if num_v_heads is not None and num_v_heads > q.shape[2]:
        interleave = num_v_heads // q.shape[2]
        q_norm = q_norm.repeat_interleave(interleave, dim=2)
        k_norm = k_norm.repeat_interleave(interleave, dim=2)
    return q_norm, k_norm, g, beta


def make_extend_kernel_inputs(device, dtype, num_k_heads=2, num_v_heads=2):
    """Build a minimal spec-dec extend request for the cached gated-delta op.

    The request writes per-prefix states into ``intermediate_delta_cache`` instead of
    updating ``delta_cache`` directly, matching how extend requests are used during
    speculative decoding.
    """
    num_extend = 2
    tokens_per_extend = 3
    total_tokens = num_extend * tokens_per_extend
    key_dim = 8
    value_dim = 8
    max_batch_size = 4
    scale = key_dim**-0.5

    q, k, v, a, b, A_log, dt_bias = _random_inputs(
        device,
        dtype,
        1,
        total_tokens,
        num_k_heads,
        num_v_heads,
        key_dim,
        value_dim,
    )

    slot_idx = torch.tensor([1, 3], device=device, dtype=torch.int32)
    delta_cache = torch.zeros(
        max_batch_size,
        num_v_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=dtype,
    )

    batch_info = BatchInfo()
    batch_info.update([0, 0, num_extend, total_tokens, 0, 0])
    batch_info.update_max_draft_len(tokens_per_extend - 1)
    batch_info_host = batch_info.serialize()
    cu_seqlen = torch.tensor([0], device=device, dtype=torch.int32)
    use_initial_states = torch.ones(num_extend, device=device, dtype=torch.bool)
    any_prefill_use_initial_states_host = torch.tensor([False], device=device, dtype=torch.bool)
    intermediate_delta_cache = torch.zeros(
        max_batch_size,
        tokens_per_extend,
        num_v_heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )

    args = (
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
        any_prefill_use_initial_states_host,
        delta_cache,
        intermediate_delta_cache,
        scale,
    )
    return args, v


@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 2), (2, 4)])
def test_decode_only(gdr_env, num_k_heads, num_v_heads):
    """Decode-only: batch of single tokens through the cached op.

    Verifies output and cache state match fused_recurrent_gated_delta_rule_fwd
    called directly with gathered initial states and preprocessed inputs.
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

    q, k, v, a, b, A_log, dt_bias = _random_inputs(
        device, dtype, batch, seq, num_k_heads, num_v_heads, key_dim, value_dim
    )

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
    _bi = BatchInfo()
    _bi.update([0, 0, 0, 0, batch, batch])
    batch_info_host = _bi.serialize()
    cu_seqlen = torch.zeros(1, device=device, dtype=torch.int32)
    use_initial_states = torch.ones(batch, device=device, dtype=torch.bool)
    any_prefill_use_initial_states_host = torch.tensor(
        [False], device=device, dtype=torch.bool
    )  # decode-only, no prefill

    gathered_before = delta_cache.clone().index_select(0, slot_idx.long())

    y = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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
        any_prefill_use_initial_states_host,
        delta_cache,
        None,
        scale,
    )

    assert y.shape == (batch, seq, num_v_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: preprocess (L2 + GQA expand) and call fused_recurrent directly
    q_norm, k_norm, g, beta = _preprocess_for_reference(q, k, a, b, A_log, dt_bias, num_v_heads)
    q_flat = q_norm.view(batch, num_v_heads, -1)
    k_flat = k_norm.view(batch, num_v_heads, -1)
    v_flat = v.view(batch, num_v_heads, -1)
    g_flat = g.view(batch, num_v_heads)
    beta_flat = beta.view(batch, num_v_heads)

    y_ref, final_state_ref = fused_recurrent_gated_delta_rule_fwd(
        q=q_flat[:, None],
        k=k_flat[:, None],
        v=v_flat[:, None],
        g=g_flat[:, None],
        beta=beta_flat[:, None],
        scale=scale,
        initial_state=gathered_before.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,  # already normalized in _preprocess_for_reference
    )

    y_ref_reshaped = y_ref.to(dtype)
    torch.testing.assert_close(y, y_ref_reshaped, atol=atol, rtol=rtol)

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

    Verifies output and final state match per-sequence chunk_gated_delta_rule
    with use_qk_l2norm_in_kernel=True and manually computed g/beta.
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

    q, k, v, a, b, A_log, dt_bias = _random_inputs(
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

    num_prefill = len(seq_lens)
    _bi = BatchInfo()
    _bi.update([num_prefill, total_tokens, 0, 0, 0, 0])
    batch_info_host = _bi.serialize()
    cu_seqlen = torch.tensor([0, seq_lens[0], total_tokens], device=device, dtype=torch.int32)
    use_initial_states = torch.zeros(num_prefill, device=device, dtype=torch.bool)
    any_prefill_use_initial_states_host = torch.tensor(
        [False], device=device, dtype=torch.bool
    )  # no prefill seq uses initial state

    y = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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
        any_prefill_use_initial_states_host,
        delta_cache,
        None,
        scale,
    )

    assert y.shape == (1, total_tokens, num_v_heads, value_dim)
    assert torch.isfinite(y).all()

    # Reference: compute g/beta and call chunk per sequence with l2norm
    q_norm, k_norm, g, beta = _preprocess_for_reference(q, k, a, b, A_log, dt_bias, num_v_heads)

    y_ref = torch.empty_like(y)
    for i, sl in enumerate(seq_lens):
        start = sum(seq_lens[:i])
        end = start + sl

        y_seq, final_state = chunk_gated_delta_rule(
            q=q_norm[:, start:end],
            k=k_norm[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            scale=scale,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,  # already normalized
        )

        y_ref[:, start:end] = y_seq.to(dtype)

        torch.testing.assert_close(
            delta_cache[slot_idx[i].long()],
            final_state.squeeze(0).to(delta_cache.dtype),
            atol=atol,
            rtol=rtol,
        )

    torch.testing.assert_close(y, y_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 2), (2, 4)])
def test_intermediate_delta_cache(gdr_env, num_k_heads, num_v_heads):
    """Extend writes each prefix state to ``intermediate_delta_cache``.

    For an extend request with N tokens, the intermediate state at each prefix length X
    should match the ``delta_cache`` state produced by running a cached prefill over
    the first X tokens from the same initial cache. The full-prefix output should also
    match the extend output, while the original ``delta_cache`` remains unchanged.
    """
    device = gdr_env["device"]
    dtype = gdr_env["dtype"]
    atol = 5e-3
    rtol = 5e-3
    state_atol = 1e-2
    state_rtol = 2e-2

    args, _ = make_extend_kernel_inputs(device, dtype, num_k_heads, num_v_heads)
    (
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
        any_prefill_use_initial_states_host,
        delta_cache,
        intermediate_delta_cache,
        scale,
    ) = args
    del batch_info_host, cu_seqlen, use_initial_states, any_prefill_use_initial_states_host

    delta_cache.copy_(torch.randn_like(delta_cache))
    initial_cache = delta_cache.clone()

    y_extend = torch.ops.auto_deploy.fla_cached_gated_delta_rule(*args)

    num_extend = slot_idx.numel()
    tokens_per_extend = q.shape[1] // num_extend
    for request_idx, slot in enumerate(slot_idx):
        request_start = request_idx * tokens_per_extend
        request_end = request_start + tokens_per_extend
        for prefix_len in range(1, tokens_per_extend + 1):
            prefix_delta_cache = initial_cache.clone()
            prefix_batch_info = BatchInfo()
            prefix_batch_info.update([1, prefix_len, 0, 0, 0, 0])
            prefix_cu_seqlen = torch.tensor([0, prefix_len], device=device, dtype=torch.int32)
            prefix_slot_idx = slot_idx[request_idx : request_idx + 1]
            prefix_use_initial_states = torch.ones(1, device=device, dtype=torch.bool)
            prefix_any_use_initial_states_host = torch.tensor(
                [True], device=device, dtype=torch.bool
            )

            prefix_slice = slice(request_start, request_start + prefix_len)
            y_prefill = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
                q[:, prefix_slice],
                k[:, prefix_slice],
                v[:, prefix_slice],
                a[:, prefix_slice],
                b[:, prefix_slice],
                A_log,
                dt_bias,
                prefix_batch_info.serialize(),
                prefix_cu_seqlen,
                prefix_slot_idx,
                prefix_use_initial_states,
                prefix_any_use_initial_states_host,
                prefix_delta_cache,
                None,
                scale,
            )

            torch.testing.assert_close(
                intermediate_delta_cache[request_idx, prefix_len - 1],
                prefix_delta_cache[slot.long()].to(intermediate_delta_cache.dtype),
                atol=state_atol,
                rtol=state_rtol,
            )
            if prefix_len == tokens_per_extend:
                torch.testing.assert_close(
                    y_extend[:, request_start:request_end], y_prefill, atol=atol, rtol=rtol
                )

    torch.testing.assert_close(delta_cache, initial_cache, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_extend_cuda_graph_capture():
    """Spec-dec extend FLA cached gated-delta op should be CUDA-graph capturable."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    warmup_args, _ = make_extend_kernel_inputs(device, dtype)
    for _ in range(3):
        torch.ops.auto_deploy.fla_cached_gated_delta_rule(*warmup_args)

    capture_args, capture_v = make_extend_kernel_inputs(device, dtype)
    static_out = torch.empty_like(capture_v)

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        static_out.copy_(torch.ops.auto_deploy.fla_cached_gated_delta_rule(*capture_args))

    cuda_graph.replay()
    assert torch.isfinite(static_out).all()


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

    q, k, v, a, b, A_log, dt_bias = _random_inputs(
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
    initial_state = delta_cache[1].clone()

    # Metadata: one prefill sequence with initial state
    _bi = BatchInfo()
    _bi.update([1, seq_len, 0, 0, 0, 0])
    batch_info_host = _bi.serialize()
    cu_seqlen = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    use_initial_states = torch.tensor([True], device=device, dtype=torch.bool)
    any_prefill_use_initial_states_host = torch.tensor(
        [True], device=device, dtype=torch.bool
    )  # one prefill seq with initial state

    y_with_init = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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
        any_prefill_use_initial_states_host,
        delta_cache,
        None,
        scale,
    )

    # Reference: compute g/beta and call chunk with l2norm and initial state
    q_norm, k_norm, g, beta = _preprocess_for_reference(q, k, a, b, A_log, dt_bias, num_v_heads)

    y_ref, final_ref = chunk_gated_delta_rule(
        q=q_norm,
        k=k_norm,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state.unsqueeze(0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,  # already normalized in _preprocess_for_reference
    )

    torch.testing.assert_close(y_with_init, y_ref.to(dtype), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        delta_cache[1],
        final_ref.squeeze(0).to(delta_cache.dtype),
        atol=atol,
        rtol=rtol,
    )

    # Verify it differs from running WITHOUT initial state
    delta_cache_zero = torch.zeros_like(delta_cache)
    use_initial_states_false = torch.tensor([False], device=device, dtype=torch.bool)
    any_prefill_use_initial_states_host_false = torch.tensor(
        [False], device=device, dtype=torch.bool
    )

    y_without_init = torch.ops.auto_deploy.fla_cached_gated_delta_rule(
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
        any_prefill_use_initial_states_host_false,
        delta_cache_zero,
        None,
        scale,
    )

    assert not torch.allclose(y_with_init, y_without_init, atol=1e-3, rtol=1e-3), (
        "Output with initial state should differ from output without initial state"
    )
