# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Equivalence test for the GDN MTP replay verify kernel.

Simulates several target-verify iterations with randomized acceptance and
checks cached replay plus PNAT/double-buffer bookkeeping mirroring
update_mamba_states against:
  1. the legacy path (fused_recurrent_gated_delta_rule_update with
     intermediate_states_buffer + accepted-state copy), and
  2. a sequential fp32 reference.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fla.cached_replay import (
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update


def _seq_ref_step(S, q, k, v, g, beta, scale):
    """Sequential fp32 reference over T steps for one request.

    S: [HV, V, K] fp32 (modified out-of-place). Returns (o [T, HV, V],
    per-step states list).
    """
    T = q.shape[0]
    HV = S.shape[0]
    H = k.shape[1]
    outs, states = [], []
    S = S.clone()
    for t in range(T):
        o_t = torch.empty(*S.shape[:1], S.shape[1], dtype=torch.float32, device=S.device)
        for hv in range(HV):
            h = hv // (HV // H)
            qt = q[t, h].float()
            kt = k[t, h].float()
            vt = v[t, hv].float()
            qt = qt / (qt.norm() + 1e-6) * scale
            kt = kt / (kt.norm() + 1e-6)
            St = S[hv] * torch.exp(g[t, hv].float())
            vt = (vt - (St * kt[None, :]).sum(-1)) * beta[t, hv].float()
            St = St + kt[None, :] * vt[:, None]
            o_t[hv] = (St * qt[None, :]).sum(-1)
            S[hv] = St
        outs.append(o_t)
        states.append(S.clone())
    return torch.stack(outs), states


@pytest.mark.parametrize("fused_gating", [False, True], ids=["pre_gated", "fused_gating"])
@pytest.mark.parametrize(
    "pool_dtype", [torch.bfloat16, torch.float32], ids=["bf16_pool", "fp32_pool"]
)
@pytest.mark.parametrize(
    "T,HIST,iters",
    [(2, 14, 16), (4, 12, 10), (5, 11, 12), (4, 16, 10)],
    ids=["T2_H14", "T4_H12", "T5_H11", "T4_H16"],
)
@pytest.mark.parametrize(
    "H,HV,K,V",
    [(4, 8, 128, 128), (2, 4, 64, 64), (4, 16, 128, 128)],
    ids=["qwen3_like", "small", "qwen3_5_like"],
)
def test_gdn_replay_vs_legacy_and_ref(H, HV, K, V, T, HIST, iters, pool_dtype, fused_gating):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(1234)
    device = "cuda"
    dtype = torch.bfloat16
    N = 4
    scale = K**-0.5
    slots = N + 2  # exercise non-trivial slot indices

    # Fused-gating mode: the replay kernel receives RAW a/b plus
    # A_log/dt_bias and applies g = -exp(A_log)*softplus(a + dt_bias),
    # beta = sigmoid(b) in-kernel; legacy + reference use the same gating
    # computed on the host.
    A_log_t = (torch.randn(HV, device=device, dtype=torch.float32) * 0.5) if fused_gating else None
    dt_bias_t = (
        (torch.randn(HV, device=device, dtype=torch.float32) * 0.5) if fused_gating else None
    )

    pool_init = torch.randn(slots, HV, V, K, device=device, dtype=torch.float32) * 0.5
    # Exercise a pool-backed (non-contiguous, padded slot stride) layout:
    # the kernel must commit through the real strides — a contiguous copy
    # would silently discard checkpoint commits (Qwen3.5 looping root
    # cause). Odd T uses the strided layout, even T stays dense.
    strided_pool = T % 2 == 1
    # .clone() everywhere: for fp32 pools .to(pool_dtype)/.float() are no-ops
    # returning the same tensor, which would alias pool/reference storage.
    if strided_pool:
        _pad = 1024
        _backing = torch.zeros(slots, HV * V * K + _pad, device=device, dtype=pool_dtype)
        pool_replay = _backing[:, : HV * V * K].view(slots, HV, V, K)
        pool_replay.copy_(pool_init.to(pool_dtype))
        assert not pool_replay.is_contiguous()
    else:
        pool_replay = pool_init.to(pool_dtype).clone()
    pool_legacy = pool_init.to(pool_dtype).clone()
    ref_S = pool_init.to(pool_dtype).float().clone()  # follow quantized init

    # Replay buffers + bookkeeping (per-layer slice shapes)
    old_v = torch.zeros(slots, 2, HIST, HV, V, device=device, dtype=dtype)
    old_k = torch.zeros(slots, 2, HIST, H, K, device=device, dtype=dtype)
    old_g = torch.zeros(slots, 2, HV, HIST, device=device, dtype=torch.float32)
    old_beta = torch.zeros(slots, 2, HV, HIST, device=device, dtype=torch.float32)
    buf_idx = torch.zeros(slots, dtype=torch.int32, device=device)
    pnat = torch.zeros(slots, dtype=torch.int32, device=device)
    # Non-identity slot mapping
    state_indices = torch.arange(1, N + 1, dtype=torch.int32, device=device)

    # Legacy intermediate buffer, indexed by [0..N)
    intermediate_ssm = torch.zeros(N, T, HV, V, K, device=device, dtype=pool_dtype)
    arange_n = torch.arange(N, dtype=torch.int32, device=device)

    n_checkpoints = 0
    prev_ref_S = ref_S.clone()
    for it in range(iters):
        q = torch.randn(N, T, H, K, device=device, dtype=dtype)
        k = torch.randn(N, T, H, K, device=device, dtype=dtype)
        v = torch.randn(N, T, HV, V, device=device, dtype=dtype) * 0.5
        packed_qkv = None
        if T == 4 and HIST == 16:
            packed_qkv = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1).reshape(
                N * T, -1
            )
            packed_qkv_3d = packed_qkv.view(N, T, -1)
            key_width = H * K
            q = packed_qkv_3d[..., :key_width].view(N, T, H, K)
            k = packed_qkv_3d[..., key_width : 2 * key_width].view(N, T, H, K)
            v = packed_qkv_3d[..., 2 * key_width :].view(N, T, HV, V)
            assert packed_qkv.is_contiguous()
            assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
        if fused_gating:
            packed_ba = torch.randn(N * T, 2 * HV, device=device, dtype=dtype)
            packed_ba[:, HV:].mul_(0.5)
            b_raw = packed_ba[:, :HV].view(N, T, HV)
            a_raw = packed_ba[:, HV:].view(N, T, HV)
            assert not a_raw.is_contiguous() and not b_raw.is_contiguous()
            # Host reference of the in-kernel gating (fp32, same inputs).
            g = -(A_log_t.exp() * torch.nn.functional.softplus(a_raw.float() + dt_bias_t))
            beta = torch.sigmoid(b_raw.float())
        else:
            g = -torch.rand(N, T, HV, device=device, dtype=torch.float32) * 2.0
            beta = torch.rand(N, T, HV, device=device, dtype=torch.float32)
        num_accepted = torch.randint(1, T + 1, (N,))

        # --- replay path ---
        replay_kwargs = {}
        # Piggyback PDL coverage on the fused-gating arm: a PDL launch
        # without a producer signal is legal (waits resolve at predecessor
        # completion), so this exercises the gdc_wait code path.
        replay_kwargs["launch_with_pdl"] = fused_gating
        if packed_qkv is not None:
            replay_kwargs["packed_qkv"] = packed_qkv
        replay_output = torch.empty((N, T, HV, V), device=device, dtype=dtype)
        replay_kwargs["output"] = replay_output
        o_replay = fused_recurrent_gated_delta_rule_cached_replay_update(
            q,
            k,
            v,
            a_raw if fused_gating else g,
            b_raw if fused_gating else beta,
            pool_replay,
            state_indices,
            old_v,
            old_k,
            old_g,
            old_beta,
            buf_idx,
            pnat,
            history_size=HIST,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            A_log=A_log_t,
            dt_bias=dt_bias_t,
            **replay_kwargs,
        )
        assert o_replay is replay_output

        # --- legacy path ---
        src = pool_legacy[state_indices]
        o_legacy = fused_recurrent_gated_delta_rule_update(
            q=q,
            k=k,
            v=v,
            g=g.to(dtype),
            beta=beta.to(dtype),
            scale=scale,
            initial_state_source=src,
            initial_state_indices=arange_n,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_ssm,
            cache_steps=T,
        )

        # --- reference outputs + acceptance ---
        for n in range(N):
            o_ref, states = _seq_ref_step(
                ref_S[state_indices[n]], q[n], k[n], v[n], g[n], beta[n], scale
            )
            torch.testing.assert_close(o_replay[n].float(), o_ref, atol=6e-3, rtol=2e-2)
            torch.testing.assert_close(o_legacy[n].float(), o_ref, atol=6e-3, rtol=2e-2)
            ref_S[state_indices[n]] = states[num_accepted[n] - 1]

        # replay vs legacy outputs directly (both bf16 outputs, tighter)
        torch.testing.assert_close(o_replay.float(), o_legacy.float(), atol=4e-3, rtol=2e-2)

        # --- legacy acceptance: copy accepted state ---
        pool_legacy[state_indices] = intermediate_ssm[
            arange_n.long(), (num_accepted - 1).to(device).long()
        ]

        # --- replay bookkeeping (mirrors update_mamba_states) ---
        pnat_h = pnat.cpu()
        buf_h = buf_idx.cpu()
        for n in range(N):
            slot = state_indices[n].item()
            if (pnat_h[slot] + T) > HIST:
                # Kernel committed the pre-iteration state this launch
                # (prev_ref_S = reference state before this iteration's
                # acceptance advance).
                err = (pool_replay[slot].float() - prev_ref_S[slot]).abs().max().item()
                assert err < 6e-3, f"checkpoint state err {err}"
                n_checkpoints += 1
                pnat_h[slot] = num_accepted[n]
                buf_h[slot] = 1 - buf_h[slot]
            else:
                pnat_h[slot] = pnat_h[slot] + num_accepted[n]
        pnat.copy_(pnat_h.to(device))
        buf_idx.copy_(buf_h.to(device))
        prev_ref_S = ref_S.clone()

    # Ensure the test exercised checkpoint commits and buffer flips.
    assert n_checkpoints >= N, f"only {n_checkpoints} checkpoints hit; increase iters"


def test_gdn_replay_negative_slot_skipped():
    """Padding requests (slot < 0) must not touch pool or history."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    N, T, H, HV, K, V, HIST = 2, 4, 2, 4, 64, 64, 16

    pool = torch.randn(N, HV, V, K, device=device, dtype=dtype)
    pool_orig = pool.clone()
    old_v = torch.zeros(N, 2, HIST, HV, V, device=device, dtype=dtype)
    old_k = torch.zeros(N, 2, HIST, H, K, device=device, dtype=dtype)
    old_g = torch.zeros(N, 2, HV, HIST, device=device, dtype=torch.float32)
    old_beta = torch.zeros(N, 2, HV, HIST, device=device, dtype=torch.float32)
    buf_idx = torch.zeros(N, dtype=torch.int32, device=device)
    pnat = torch.full((N,), HIST, dtype=torch.int32, device=device)  # force write
    state_indices = torch.tensor([-1, -1], dtype=torch.int32, device=device)

    q = torch.randn(N, T, H, K, device=device, dtype=dtype)
    k = torch.randn(N, T, H, K, device=device, dtype=dtype)
    v = torch.randn(N, T, HV, V, device=device, dtype=dtype)
    g = -torch.rand(N, T, HV, device=device, dtype=torch.float32)
    beta = torch.rand(N, T, HV, device=device, dtype=torch.float32)

    fused_recurrent_gated_delta_rule_cached_replay_update(
        q,
        k,
        v,
        g,
        beta,
        pool,
        state_indices,
        old_v,
        old_k,
        old_g,
        old_beta,
        buf_idx,
        pnat,
        history_size=HIST,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(pool, pool_orig)
    assert old_v.abs().max().item() == 0
    assert old_k.abs().max().item() == 0


def _cached_replay_commit_reference(initial_states, old_u, old_k, old_G, work_items, n_writes):
    reference = initial_states.clone()
    num_layers, _, HV, _, _ = reference.shape
    H = old_k.shape[-2]
    rows = work_items[: int(n_writes.item())].cpu().tolist()
    for _, slot, pnat, active_buffer in rows:
        for layer in range(num_layers):
            for hv in range(HV):
                h = hv // (HV // H)
                history_k = old_k[layer, slot, active_buffer, :pnat, h]
                history_u = old_u[layer, slot, active_buffer, :pnat, hv]
                history_G = old_G[layer, slot, active_buffer, hv, :pnat]
                g_start = history_G[-1]
                commit_decay = torch.exp(g_start - history_G)
                scaled_u = history_u.float() * commit_decay[:, None]
                scaled_u_hi = scaled_u.to(history_u.dtype)
                scaled_u_lo = (scaled_u - scaled_u_hi.float()).to(history_u.dtype)
                delta = history_k.float().T @ scaled_u_hi.float()
                delta += history_k.float().T @ scaled_u_lo.float()
                committed = reference[layer, slot, hv].float() * torch.exp(g_start)
                reference[layer, slot, hv] = (committed + delta.T).to(reference.dtype)
    return reference


def test_gdn_cached_replay_all_layer_commit_matches_reference():
    """The all-layer commit must match its direct mathematical reference."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from tensorrt_llm._torch.modules.fla.cached_replay import (
        commit_gdn_cached_replay_history_layers,
    )

    torch.manual_seed(1234)
    device = "cuda"
    dtype = torch.bfloat16
    num_layers, N, num_slots = 3, 16, 18
    H, HV, K, V, HIST = 2, 8, 128, 128, 16
    initial_states = torch.randn(num_layers, num_slots, HV, V, K, device=device, dtype=dtype)
    old_u = torch.randn(num_layers, num_slots, 2, HIST, HV, V, device=device, dtype=dtype)
    old_k = torch.randn(num_layers, num_slots, 2, HIST, H, K, device=device, dtype=dtype)
    old_G = -torch.cumsum(
        torch.rand(
            num_layers,
            num_slots,
            2,
            HV,
            HIST,
            device=device,
            dtype=torch.float32,
        )
        * 0.1,
        dim=-1,
    )

    positions = torch.arange(N, device=device, dtype=torch.int32)
    cache_slots = positions + 1
    pnat = positions.remainder(HIST - 1) + 1
    active_buffers = positions.remainder(2)
    work_items = torch.stack((positions, cache_slots, pnat, active_buffers), dim=1)
    n_writes = torch.tensor([N // 2], device=device, dtype=torch.int32)

    expected_states = _cached_replay_commit_reference(
        initial_states,
        old_u,
        old_k,
        old_G,
        work_items,
        n_writes,
    )
    actual_states = initial_states.clone()
    commit_gdn_cached_replay_history_layers(
        ssm_states=actual_states,
        old_u=old_u,
        old_k=old_k,
        old_G=old_G,
        replay_work_items=work_items,
        n_writes=n_writes,
        history_size=HIST,
    )

    torch.testing.assert_close(actual_states.float(), expected_states.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("batch_size", [8, 16], ids=["small_fused", "large_all_layer"])
def test_gdn_cached_replay_dispatch_cuda_graph_matches_eager(batch_size):
    """Both sides of the BS16 dispatch must be safe under CUDA graphs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from tensorrt_llm._torch.modules.fla.cached_replay import (
        CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE,
        commit_gdn_cached_replay_history_layers,
    )

    torch.manual_seed(1234)
    device = "cuda"
    dtype = torch.bfloat16
    num_layers, T, HIST = 2, 4, 16
    H, HV, K, V = 2, 4, 64, 64
    num_slots = batch_size + 1
    use_all_layer_commit = batch_size >= CACHED_REPLAY_PARTITION_MIN_BATCH_SIZE

    initial_states = torch.randn(num_layers, num_slots, HV, V, K, device=device, dtype=dtype)
    initial_u = torch.randn(num_layers, num_slots, 2, HIST, HV, V, device=device, dtype=dtype)
    initial_k = torch.randn(num_layers, num_slots, 2, HIST, H, K, device=device, dtype=dtype)
    initial_G = -torch.cumsum(
        torch.rand(
            num_layers,
            num_slots,
            2,
            HV,
            HIST,
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    )
    old_beta = torch.empty_like(initial_G)
    state_indices = torch.arange(1, batch_size + 1, device=device, dtype=torch.int32)
    cache_buf_idx = torch.zeros(num_slots, device=device, dtype=torch.int32)
    pnat = torch.full((num_slots,), HIST, device=device, dtype=torch.int32)
    q = torch.randn(num_layers, batch_size, T, H, K, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(num_layers, batch_size, T, HV, V, device=device, dtype=dtype)
    g = -torch.rand(num_layers, batch_size, T, HV, device=device, dtype=torch.float32)
    beta = torch.rand_like(g)

    replay_work_items = None
    n_writes = None
    if use_all_layer_commit:
        positions = torch.arange(batch_size, device=device, dtype=torch.int32)
        replay_work_items = torch.stack(
            (
                positions,
                state_indices,
                torch.full_like(positions, HIST),
                torch.zeros_like(positions),
            ),
            dim=1,
        )
        n_writes = torch.tensor([batch_size], device=device, dtype=torch.int32)

    def run_replay(states, old_u, old_k, old_G):
        outputs = []
        for layer in range(num_layers):
            outputs.append(
                fused_recurrent_gated_delta_rule_cached_replay_update(
                    q[layer],
                    k[layer],
                    v[layer],
                    g[layer],
                    beta[layer],
                    states[layer],
                    state_indices,
                    old_u[layer],
                    old_k[layer],
                    old_G[layer],
                    old_beta[layer],
                    cache_buf_idx,
                    pnat,
                    history_size=HIST,
                    use_qk_l2norm_in_kernel=True,
                    replay_work_items=replay_work_items,
                    n_writes=n_writes,
                    use_all_layer_commit=use_all_layer_commit,
                )
            )
        return outputs

    def run_all_layer_commit(states, old_u, old_k, old_G):
        if use_all_layer_commit:
            commit_gdn_cached_replay_history_layers(
                ssm_states=states,
                old_u=old_u,
                old_k=old_k,
                old_G=old_G,
                replay_work_items=replay_work_items,
                n_writes=n_writes,
                history_size=HIST,
            )

    warmup_states = initial_states.clone()
    warmup_u = initial_u.clone()
    warmup_k = initial_k.clone()
    warmup_G = initial_G.clone()
    run_replay(warmup_states, warmup_u, warmup_k, warmup_G)
    run_all_layer_commit(warmup_states, warmup_u, warmup_k, warmup_G)
    torch.cuda.synchronize()

    eager_states = initial_states.clone()
    eager_u = initial_u.clone()
    eager_k = initial_k.clone()
    eager_G = initial_G.clone()
    eager_outputs = run_replay(eager_states, eager_u, eager_k, eager_G)
    run_all_layer_commit(eager_states, eager_u, eager_k, eager_G)

    graph_states = initial_states.clone()
    graph_u = initial_u.clone()
    graph_k = initial_k.clone()
    graph_G = initial_G.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = run_replay(graph_states, graph_u, graph_k, graph_G)
    graph_states.copy_(initial_states)
    graph_u.copy_(initial_u)
    graph_k.copy_(initial_k)
    graph_G.copy_(initial_G)
    torch.cuda.synchronize()
    graph.replay()
    run_all_layer_commit(graph_states, graph_u, graph_k, graph_G)
    torch.cuda.synchronize()

    for eager_output, graph_output in zip(eager_outputs, graph_outputs):
        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(graph_states, eager_states, rtol=0, atol=0)
    torch.testing.assert_close(graph_u, eager_u, rtol=0, atol=0)
    torch.testing.assert_close(graph_k, eager_k, rtol=0, atol=0)
    torch.testing.assert_close(graph_G, eager_G, rtol=0, atol=0)
