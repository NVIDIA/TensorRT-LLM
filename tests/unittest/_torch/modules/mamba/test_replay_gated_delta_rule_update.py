# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the GDN MTP *replay* state update.

The replay path reconstructs the committed recurrent state from a compact
double-buffered history (old_k/old_v/old_g/old_beta) instead of caching a full
intermediate state per draft position. These tests assert that:

1. A single verify step (PNAT == 0) matches the legacy
   ``fused_recurrent_gated_delta_rule_update`` reference exactly (fp32).
2. A multi-step MTP protocol (with buffer flips / checkpoints) keeps producing
   outputs that match a golden recurrence tracked over the committed tokens.
3. The kernel is CUDA-graph capturable and replays deterministically.
"""

import pytest
import torch

skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                  reason="Requires CUDA")


def _gating(A_log, a, dt_bias, N, T, HV):
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import fused_gdn_gating
    return fused_gdn_gating(A_log, a.reshape(N * T, HV), dt_bias).reshape(
        N, T, HV)


def _rand_step(N, T, H, HV, K, V, dev, dtype, scale_amp=0.1):
    q = (torch.randn(N, T, H, K, device=dev) * scale_amp).to(dtype)
    k = (torch.randn(N, T, H, K, device=dev) * scale_amp).to(dtype)
    v = (torch.randn(N, T, HV, V, device=dev) * scale_amp).to(dtype)
    a = torch.randn(N, T, HV, device=dev) * scale_amp
    b = torch.randn(N, T, HV, device=dev) * scale_amp
    return q, k, v, a, b


@skip_no_cuda
@pytest.mark.parametrize("draft_token_num", [1, 2, 4, 5])
@pytest.mark.parametrize("num_decodes", [1, 3])
@pytest.mark.parametrize("H,HV", [(4, 8), (2, 2)])
def test_replay_single_step_matches_recurrent(draft_token_num, num_decodes, H,
                                              HV):
    """PNAT == 0: replay output equals the fused_recurrent reference (fp32)."""
    from tensorrt_llm._torch.modules.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update, )
    from tensorrt_llm._torch.modules.fla.replay_gated_delta_rule_update import (
        replay_gated_delta_rule_update, )

    torch.manual_seed(0)
    dev = "cuda"
    N, T, K, V = num_decodes, draft_token_num, 128, 128
    HISTORY = max(16, T)
    scale = K**-0.5
    dtype = torch.float32

    q, k, v, a, b = _rand_step(N, T, H, HV, K, V, dev, dtype)
    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    g = _gating(A_log, a, dt_bias, N, T, HV)
    beta = b.sigmoid()

    state_pool = torch.randn(N, HV, V, K, device=dev, dtype=dtype) * 0.1
    idx = torch.arange(N, device=dev, dtype=torch.int32)

    # Reference from the committed (initial) state.
    ref = fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state_source=state_pool.clone(),
        initial_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
    )

    old_k = torch.zeros(N, 2, HISTORY, H, K, device=dev, dtype=dtype)
    old_v = torch.zeros(N, 2, HISTORY, HV, V, device=dev, dtype=dtype)
    old_g = torch.zeros(N, 2, HISTORY, HV, device=dev, dtype=torch.float32)
    old_beta = torch.zeros(N, 2, HISTORY, HV, device=dev, dtype=torch.float32)
    cache_buf_idx = torch.zeros(N, device=dev, dtype=torch.int32)
    pnat = torch.zeros(N, device=dev, dtype=torch.int32)
    state_replay = state_pool.clone()

    out = replay_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        ssm_states=state_replay,
        old_k=old_k,
        old_v=old_v,
        old_g=old_g,
        old_beta=old_beta,
        cache_buf_idx=cache_buf_idx,
        prev_num_accepted_tokens=pnat,
        state_batch_indices=idx,
        replay_step_width=T,
        replay_history_size=HISTORY,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out.view(N, T, HV, V),
                               ref.view(N, T, HV, V),
                               rtol=1e-4,
                               atol=1e-4)
    # PNAT == 0 leaves the checkpoint untouched (replaying zero history tokens).
    torch.testing.assert_close(state_replay, state_pool, rtol=1e-4, atol=1e-4)


def _replay_protocol(dtype, rtol, atol, accepted_pattern, T, HISTORY, H=4, HV=8):
    """Run the full MTP replay protocol and compare per-step outputs to a golden
    recurrence tracked over the committed tokens."""
    from tensorrt_llm._torch.modules.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update, )
    from tensorrt_llm._torch.modules.fla.replay_gated_delta_rule_update import (
        replay_gated_delta_rule_update, )

    torch.manual_seed(1)
    dev = "cuda"
    N, K, V = 3, 128, 128
    scale = K**-0.5

    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    idx = torch.arange(N, device=dev, dtype=torch.int32)

    # Golden committed state (advanced by accepted tokens each step).
    golden_state = torch.randn(N, HV, V, K, device=dev, dtype=dtype) * 0.1
    # Replay checkpoint starts equal to the committed state.
    state_replay = golden_state.clone()

    old_k = torch.zeros(N, 2, HISTORY, H, K, device=dev, dtype=dtype)
    old_v = torch.zeros(N, 2, HISTORY, HV, V, device=dev, dtype=dtype)
    old_g = torch.zeros(N, 2, HISTORY, HV, device=dev, dtype=torch.float32)
    old_beta = torch.zeros(N, 2, HISTORY, HV, device=dev, dtype=torch.float32)
    cache_buf_idx = torch.zeros(N, device=dev, dtype=torch.int32)
    pnat = torch.zeros(N, device=dev, dtype=torch.int32)

    for accepted in accepted_pattern:
        q, k, v, a, b = _rand_step(N, T, H, HV, K, V, dev, dtype)
        g = _gating(A_log, a, dt_bias, N, T, HV)
        beta = b.sigmoid()

        # Golden: outputs for all T tokens from the committed state.
        ref = fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state_source=golden_state.clone(),
            initial_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
        )

        out = replay_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=state_replay,
            old_k=old_k,
            old_v=old_v,
            old_g=old_g,
            old_beta=old_beta,
            cache_buf_idx=cache_buf_idx,
            prev_num_accepted_tokens=pnat,
            state_batch_indices=idx,
            replay_step_width=T,
            replay_history_size=HISTORY,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(out.view(N, T, HV, V),
                                   ref.view(N, T, HV, V),
                                   rtol=rtol,
                                   atol=atol)

        # Advance the golden committed state by the accepted prefix.
        fused_recurrent_gated_delta_rule_update(
            q=q[:, :accepted].contiguous(),
            k=k[:, :accepted].contiguous(),
            v=v[:, :accepted].contiguous(),
            g=g[:, :accepted].contiguous(),
            beta=beta[:, :accepted].contiguous(),
            scale=scale,
            initial_state_source=golden_state,
            initial_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=False,
            disable_output_calculation=True,
        )

        # Emulate update_mamba_states PNAT / buffer bookkeeping.
        acc = torch.full((N, ), accepted, device=dev, dtype=torch.int32)
        wrote = (pnat + T) > HISTORY
        pnat = torch.where(wrote, acc, pnat + acc)
        cache_buf_idx = torch.where(wrote, 1 - cache_buf_idx, cache_buf_idx)


@skip_no_cuda
def test_replay_protocol_nowrite_only_fp32():
    """Large history => never checkpoints; pure nowrite accumulation."""
    _replay_protocol(torch.float32,
                     rtol=2e-4,
                     atol=2e-4,
                     accepted_pattern=[2, 3, 1, 2],
                     T=4,
                     HISTORY=16)


@skip_no_cuda
def test_replay_protocol_with_checkpoints_fp32():
    """Small history forces buffer flips / checkpoint writes."""
    _replay_protocol(torch.float32,
                     rtol=2e-4,
                     atol=2e-4,
                     accepted_pattern=[3, 3, 2, 3, 1],
                     T=4,
                     HISTORY=8)


@skip_no_cuda
def test_replay_protocol_bf16_loose():
    """bf16 checkpoint/history: same protocol with bf16-appropriate tolerance."""
    _replay_protocol(torch.bfloat16,
                     rtol=3e-2,
                     atol=3e-2,
                     accepted_pattern=[3, 2, 3, 2],
                     T=4,
                     HISTORY=8)


@skip_no_cuda
def test_replay_cuda_graph_capturable():
    """The replay kernel captures into a CUDA graph and replays deterministically."""
    from tensorrt_llm._torch.modules.fla.replay_gated_delta_rule_update import (
        replay_gated_delta_rule_update, )

    torch.manual_seed(2)
    dev = "cuda"
    N, T, H, HV, K, V = 2, 4, 4, 8, 128, 128
    HISTORY = 16
    dtype = torch.float32
    scale = K**-0.5

    q, k, v, a, b = _rand_step(N, T, H, HV, K, V, dev, dtype)
    A_log = torch.empty(HV, device=dev).uniform_(1.0, 16.0).log()
    dt_bias = torch.randn(HV, device=dev) * 0.1
    g = _gating(A_log, a, dt_bias, N, T, HV)
    beta = b.sigmoid()

    state_pool = torch.randn(N, HV, V, K, device=dev, dtype=dtype) * 0.1
    idx = torch.arange(N, device=dev, dtype=torch.int32)
    old_k = torch.zeros(N, 2, HISTORY, H, K, device=dev, dtype=dtype)
    old_v = torch.zeros(N, 2, HISTORY, HV, V, device=dev, dtype=dtype)
    old_g = torch.zeros(N, 2, HISTORY, HV, device=dev, dtype=torch.float32)
    old_beta = torch.zeros(N, 2, HISTORY, HV, device=dev, dtype=torch.float32)
    cache_buf_idx = torch.zeros(N, device=dev, dtype=torch.int32)
    pnat = torch.zeros(N, device=dev, dtype=torch.int32)
    out = torch.zeros(N, T, HV, V, device=dev, dtype=dtype)

    def run():
        replay_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=state_pool,
            old_k=old_k,
            old_v=old_v,
            old_g=old_g,
            old_beta=old_beta,
            cache_buf_idx=cache_buf_idx,
            prev_num_accepted_tokens=pnat,
            state_batch_indices=idx,
            replay_step_width=T,
            replay_history_size=HISTORY,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            output=out,
        )

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Reset inputs/state and replay: PNAT==0 so state stays; output determinstic.
    state_pool_ref = state_pool.clone()
    graph.replay()
    torch.cuda.synchronize()
    out_first = out.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, out_first, rtol=0, atol=0)
    # Idempotent for PNAT==0: checkpoint unchanged.
    torch.testing.assert_close(state_pool, state_pool_ref, rtol=1e-4, atol=1e-4)
