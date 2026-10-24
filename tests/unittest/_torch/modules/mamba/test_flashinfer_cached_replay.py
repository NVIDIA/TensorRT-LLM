# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for GDN replay through the FlashInfer ring-buffer kernel.

* Parity with the Triton double-buffer kernel: both run side by side over many
  speculative-decoding steps with random acceptance, advancing each side's
  history bookkeeping after every step. Outputs and checkpoint states must
  match while the history fills, flushes into the checkpoint, and wraps around
  the 32-row ring, and an unused slot must never be written.
* Padding rows (slot -1) are skipped and write nothing.
* Misaligned slot indices and a/b give the same results as aligned copies.
* A CUDA-graph captured call matches an eager call.
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fla.cached_replay import (
    fused_recurrent_gated_delta_rule_cached_replay_update,
)
from tensorrt_llm._torch.modules.fla.flashinfer_cached_replay import (
    FLASHINFER_REPLAY_RING_SIZE,
    flashinfer_cached_replay_update,
    is_flashinfer_cached_replay_available,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    ReplayStateUpdateMetadata,
    _advance_replay_state,
    _advance_ring_start,
)

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flashinfer_cached_replay_available(),
    reason="Requires SM90/SM100/SM103 and FlashInfer gated_delta_rule_mtp_ucache_flush",
)


def _make_step_inputs(N, T, H, HV, K, V, packed, device, dtype):
    """Random q/k/v and raw a/b laid out the way the GDN layer passes them.

    a/b are column slices of one in_proj_ba-like buffer (b first, then a). With
    ``packed``, q/k/v are column slices of one buffer, as on the decode-only path.
    """
    q = torch.randn(N, T, H, K, device=device, dtype=dtype)
    k = torch.randn(N, T, H, K, device=device, dtype=dtype)
    v = torch.randn(N, T, HV, V, device=device, dtype=dtype) * 0.5
    packed_qkv = None
    if packed:
        packed_qkv = torch.cat((q.flatten(2), k.flatten(2), v.flatten(2)), dim=-1)
        packed_qkv = packed_qkv.reshape(N * T, -1)
        rows = packed_qkv.view(N, T, -1)
        key_width = H * K
        q = rows[..., :key_width].view(N, T, H, K)
        k = rows[..., key_width : 2 * key_width].view(N, T, H, K)
        v = rows[..., 2 * key_width :].view(N, T, HV, V)
    ba = torch.randn(N * T, 2 * HV, device=device, dtype=dtype)
    b = ba[:, :HV].view(N, T, HV)
    a = ba[:, HV:].view(N, T, HV)
    return q, k, v, a, b, packed_qkv


def _random_ring_replay_state(slots, H, HV, K, V, device, dtype):
    """Random state pool, ring contents and legal cursors for single-call tests.

    The kernel only needs legal cursors (length in [0, 16], start in [0, 32)),
    so a random mid-flight state replaces running many steps first. Lengths up
    to 16 include requests that flush on this call.
    """
    ring_size = FLASHINFER_REPLAY_RING_SIZE
    return {
        "ssm_states": (torch.randn(slots, HV, V, K, device=device) * 0.5).to(dtype),
        "k_ring": torch.randn(slots, H, ring_size, K, device=device, dtype=dtype) * 0.1,
        "u_ring": torch.randn(slots, HV, ring_size, V, device=device, dtype=dtype) * 0.1,
        "g_ring": -torch.rand(slots, HV, ring_size, device=device),
        "hist_len": torch.randint(0, 17, (slots,), device=device, dtype=torch.int32),
        "cache_base": torch.randint(0, ring_size, (slots,), device=device, dtype=torch.int32),
    }


@skip_unsupported
@pytest.mark.parametrize("packed", [False, True], ids=["separate_qkv", "packed_qkv_strided_pool"])
@pytest.mark.parametrize("H,HV", [(2, 4), (2, 8), (4, 16)], ids=["hv4", "hv8", "hv16"])
@pytest.mark.parametrize("T", [4, 8])
@pytest.mark.parametrize("N", [1, 6, 33])
def test_flashinfer_ring_replay_matches_triton(N, T, H, HV, packed):
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    K = V = 128
    steps, history_size = 24, 16
    slots = N + 3
    unused_slot = slots - 1
    # Rounded to bf16 so FlashInfer's bf16 cast of A_log/dt_bias adds no error
    # on top of the kernel difference under test.
    A_log = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    dt_bias = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    init_states = (torch.randn(slots, HV, V, K, device=device) * 0.5).to(dtype)

    def make_state_pool():
        if not packed:
            return init_states.clone()
        # Block-strided pool, as V2 lays out one layer's states when a GPU
        # holds several GDN layers.
        backing = torch.zeros(slots, 3, HV, V, K, device=device, dtype=dtype)
        pool = backing[:, 0]
        pool.copy_(init_states)
        return pool

    # Triton: two history buffers per slot plus a live-buffer flag.
    triton_states = make_state_pool()
    old_u = torch.zeros(slots, 2, history_size, HV, V, device=device, dtype=dtype)
    old_k = torch.zeros(slots, 2, history_size, H, K, device=device, dtype=dtype)
    old_G = torch.zeros(slots, 2, HV, history_size, device=device)
    old_beta = torch.zeros_like(old_G)
    triton_len = torch.zeros(slots, dtype=torch.int32, device=device)
    cache_buf_idx = torch.zeros_like(triton_len)
    replay_metadata = ReplayStateUpdateMetadata(triton_len, cache_buf_idx, T, history_size)

    # FlashInfer: one 32-row ring per slot plus start/length cursors.
    ring_size = FLASHINFER_REPLAY_RING_SIZE
    fi_states = make_state_pool()
    k_ring = torch.zeros(slots, H, ring_size, K, device=device, dtype=dtype)
    u_ring = torch.zeros(slots, HV, ring_size, V, device=device, dtype=dtype)
    g_ring = torch.zeros(slots, HV, ring_size, device=device)
    fi_len = torch.zeros_like(triton_len)
    cache_base = torch.zeros_like(triton_len)
    fi_metadata = ReplayStateUpdateMetadata(fi_len, torch.zeros_like(triton_len), T, history_size)

    # Shuffled slots; the last slot is never used.
    state_indices = torch.randperm(slots - 1, device=device)[:N].int()
    rows = state_indices.long()

    num_flushes = 0
    wrapped = False
    for _ in range(steps):
        q, k, v, a, b, packed_qkv = _make_step_inputs(N, T, H, HV, K, V, packed, device, dtype)
        triton_out = fused_recurrent_gated_delta_rule_cached_replay_update(
            q,
            k,
            v,
            a,
            b,
            triton_states,
            state_indices,
            old_u,
            old_k,
            old_G,
            old_beta,
            cache_buf_idx,
            triton_len,
            history_size=history_size,
            use_qk_l2norm_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            packed_qkv=packed_qkv,
            output=torch.empty(N, T, HV, V, device=device, dtype=dtype),
        )
        # This step's new rows wrap past the end of the ring.
        wrapped |= bool((cache_base[rows] + fi_len[rows] + T > ring_size).any())
        fi_out = flashinfer_cached_replay_update(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            fi_states,
            state_indices,
            k_ring,
            u_ring,
            g_ring,
            fi_len[rows],
            cache_base[rows],
            history_size=history_size,
            output=torch.empty(N, T, HV, V, device=device, dtype=dtype),
        )
        torch.testing.assert_close(fi_out, triton_out, atol=2e-3, rtol=1.6e-2)
        torch.testing.assert_close(fi_states[rows], triton_states[rows], atol=1e-2, rtol=1.6e-2)

        accepted = torch.randint(1, T + 1, (N,), device=device, dtype=torch.int32)
        _advance_replay_state(replay_metadata, state_indices, accepted)
        num_flushes += int((fi_len[rows] + T > history_size).sum())  # before the advance
        _advance_ring_start(cache_base, fi_metadata, state_indices)
        _advance_replay_state(fi_metadata, state_indices, accepted)
        assert torch.equal(fi_len, triton_len)

    assert num_flushes >= N, f"only {num_flushes} flushes; increase steps"
    assert wrapped, "ring never wrapped; increase steps"
    assert torch.equal(fi_states[unused_slot], init_states[unused_slot])
    assert not k_ring[unused_slot].any()
    assert not u_ring[unused_slot].any()
    assert not g_ring[unused_slot].any()


@skip_unsupported
def test_flashinfer_ring_replay_skips_padding_rows():
    """Rows with slot -1 are skipped: the real rows' results match a batch
    without the padding, and padding writes no state or ring anywhere."""
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    T, H, HV, K, V = 4, 2, 8, 128, 128
    slots, history_size = 8, 16
    A_log = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    dt_bias = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    state = _random_ring_replay_state(slots, H, HV, K, V, device, dtype)
    state["hist_len"][5] = history_size  # one real row flushes

    padded_slots = torch.tensor([5, -1, 1, 6, -1], dtype=torch.int32, device=device)
    real_slots = torch.tensor([5, 1, 6], dtype=torch.int32, device=device)
    real_rows = (padded_slots >= 0).nonzero().squeeze(1)

    def run(state_indices, q, k, v, a, b):
        s = {name: t.clone() for name, t in state.items()}
        # Padding rows gather the last slot's cursors (index -1); the kernel
        # skips those rows, so the values are never used.
        rows = state_indices.long()
        out = flashinfer_cached_replay_update(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            s["ssm_states"],
            state_indices,
            s["k_ring"],
            s["u_ring"],
            s["g_ring"],
            s["hist_len"][rows],
            s["cache_base"][rows],
            history_size=history_size,
        )
        return out, s

    q, k, v, a, b, _ = _make_step_inputs(len(padded_slots), T, H, HV, K, V, False, device, dtype)
    padded_out, padded_state = run(padded_slots, q, k, v, a, b)
    real_out, real_state = run(
        real_slots, q[real_rows], k[real_rows], v[real_rows], a[real_rows], b[real_rows]
    )

    torch.testing.assert_close(padded_out[real_rows], real_out)
    for name in ("ssm_states", "k_ring", "u_ring", "g_ring"):
        torch.testing.assert_close(padded_state[name], real_state[name])


@skip_unsupported
def test_flashinfer_ring_replay_misaligned_inputs():
    """Misaligned slot indices and a/b give the same results as aligned copies."""
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    N, T, H, HV, K, V = 3, 4, 2, 8, 128, 128
    slots, history_size = 8, 16
    A_log = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    dt_bias = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    state = _random_ring_replay_state(slots, H, HV, K, V, device, dtype)

    # Slot indices sliced from a larger buffer, like state_indices[num_prefills:].
    idx_buf = torch.tensor([0, 5, 1, 6], dtype=torch.int32, device=device)
    misaligned_idx = idx_buf[1:]
    # in_proj_ba-like buffer shifted 8 bytes into its allocation, so both b and a
    # start misaligned (with HV=8 they would otherwise be aligned).
    flat = torch.randn(N * T * 2 * HV + 4, device=device, dtype=dtype)
    ba = flat[4:].view(N * T, 2 * HV)
    misaligned_b = ba[:, :HV].view(N, T, HV)
    misaligned_a = ba[:, HV:].view(N, T, HV)
    for t in (misaligned_idx, misaligned_a, misaligned_b):
        assert t.data_ptr() % 16 != 0
    q, k, v, _, _, _ = _make_step_inputs(N, T, H, HV, K, V, False, device, dtype)

    def run(state_indices, a, b):
        s = {name: t.clone() for name, t in state.items()}
        rows = state_indices.long()
        out = flashinfer_cached_replay_update(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            s["ssm_states"],
            state_indices,
            s["k_ring"],
            s["u_ring"],
            s["g_ring"],
            s["hist_len"][rows],
            s["cache_base"][rows],
            history_size=history_size,
        )
        return out, s

    misaligned_out, misaligned_state = run(misaligned_idx, misaligned_a, misaligned_b)
    # clone() copies into fresh storage, which starts at an aligned address.
    aligned_out, aligned_state = run(
        misaligned_idx.clone(), misaligned_a.clone(), misaligned_b.clone()
    )

    torch.testing.assert_close(misaligned_out, aligned_out)
    for name in ("ssm_states", "k_ring", "u_ring", "g_ring"):
        torch.testing.assert_close(misaligned_state[name], aligned_state[name])


@skip_unsupported
def test_flashinfer_ring_replay_cuda_graph_matches_eager():
    """A captured and replayed call matches an eager call on the same inputs.

    Uses the decode-only layout (packed q/k/v, block-strided pool). The first
    call compiles the kernel and allocates FlashInfer's buffers, so it runs as
    warmup before capture, as the CUDA graph runner does.
    """
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    N, T, H, HV, K, V = 4, 4, 2, 8, 128, 128
    slots, history_size = 8, 16
    A_log = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    dt_bias = (torch.randn(HV, device=device) * 0.5).to(dtype).float()
    start = _random_ring_replay_state(slots, H, HV, K, V, device, dtype)
    start["hist_len"][5] = history_size  # one row flushes inside the graph
    state_indices = torch.tensor([5, 1, 6, 3], dtype=torch.int32, device=device)
    rows = state_indices.long()
    q, k, v, a, b, _ = _make_step_inputs(N, T, H, HV, K, V, True, device, dtype)

    def fresh_state():
        s = {name: t.clone() for name, t in start.items()}
        # Block-strided pool, as on V2 with several GDN layers per GPU.
        backing = torch.zeros(slots, 3, HV, V, K, device=device, dtype=dtype)
        s["ssm_states"] = backing[:, 0]
        s["ssm_states"].copy_(start["ssm_states"])
        s["hist_len"] = s["hist_len"][rows]
        s["cache_base"] = s["cache_base"][rows]
        s["output"] = torch.empty(N, T, HV, V, device=device, dtype=dtype)
        return s

    def run(s):
        flashinfer_cached_replay_update(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            s["ssm_states"],
            state_indices,
            s["k_ring"],
            s["u_ring"],
            s["g_ring"],
            s["hist_len"],
            s["cache_base"],
            history_size=history_size,
            output=s["output"],
        )

    eager = fresh_state()
    run(eager)

    graphed = fresh_state()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(graphed)  # warmup: compile and first-call allocations
    torch.cuda.current_stream().wait_stream(stream)
    # Undo the warmup's writes in place so the replay starts from the same state.
    for name, t in fresh_state().items():
        graphed[name].copy_(t)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(graphed)
    graph.replay()
    torch.cuda.synchronize()

    for name in ("output", "ssm_states", "k_ring", "u_ring", "g_ring"):
        torch.testing.assert_close(graphed[name], eager[name])
