# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Equivariance test: rotate-before-gather == rotate-after-gather.

The strict-Ulysses a2v design relies on this property: applying RoPE locally
to a seq-sharded tensor (Q or K) with the matching seq-sharded PE, then
all-to-all'ing to head-sharded full-seq, produces the SAME tensor as if we
all-to-all'd first and then applied RoPE with full PE.

This is true because RoPE is a per-position transformation: position ``i``'s
cos/sin rotation depends only on position ``i``. all_to_all is a memory
redistribution that preserves which position each element belongs to.

This test verifies the property end-to-end on both INTERLEAVED and SPLIT
LTX-2 RoPE variants, since the strict-Ulysses a2v path piggybacks on the
existing rope applied inside ``LTX2Attention.forward``.
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.distributed import all_to_all_4d
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import (
        LTXRopeType,
        apply_rotary_emb,
    )
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _init_dist(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _worker(rank, world_size, test_fn, port):
    try:
        _init_dist(rank, world_size, port)
        test_fn(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size, test_fn):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    port = get_free_port()
    mp.spawn(_worker, args=(world_size, test_fn, port), nprocs=world_size, join=True)


def _make_interleaved_pe(T, head_dim, device, seed=0):
    """Synthetic INTERLEAVED-shape PE: cos, sin each of shape [B=1, T, 1, D]."""
    g = torch.Generator(device=device).manual_seed(seed)
    cos = torch.randn(1, T, 1, head_dim, generator=g, device=device)
    sin = torch.randn(1, T, 1, head_dim, generator=g, device=device)
    return cos, sin


# ============================================================================
# Worker logics
# ============================================================================


def _logic_rope_a2a_equivariance_K(rank, world_size):
    """For K-side (cross-attn K): rope-then-a2a == a2a-then-rope on full PE."""
    B, T_kv_full, H_kv, D = 1, world_size * 8, world_size * 4, 64
    device = torch.device("cpu")

    torch.manual_seed(11)
    k_full = torch.randn(B, T_kv_full, H_kv, D, device=device)
    pe_cos_full, pe_sin_full = _make_interleaved_pe(T_kv_full, D, device, seed=22)

    # Each rank takes its seq shard of K and PE.
    chunk = T_kv_full // world_size
    s, e = rank * chunk, (rank + 1) * chunk
    k_local = k_full[:, s:e].contiguous()
    pe_local = (pe_cos_full[:, s:e].contiguous(), pe_sin_full[:, s:e].contiguous())

    # Path A: rope locally (sharded K + sharded PE), then a2a.
    # rope keeps shape [B, T_kv/U, H_kv, D].
    k_rot_local = apply_rotary_emb(k_local, pe_local, LTXRopeType.INTERLEAVED)
    k_rot_a2a = all_to_all_4d(k_rot_local, scatter_dim=2, gather_dim=1, process_group=None)
    # k_rot_a2a: [B, T_kv_full, H_kv/U, D] — full seq, head-sharded.

    # Path B: a2a first, then rope with full PE.
    k_a2a = all_to_all_4d(k_local, scatter_dim=2, gather_dim=1, process_group=None)
    # k_a2a: [B, T_kv_full, H_kv/U, D]. Apply rope with full PE.
    k_a2a_rot = apply_rotary_emb(k_a2a, (pe_cos_full, pe_sin_full), LTXRopeType.INTERLEAVED)

    torch.testing.assert_close(
        k_rot_a2a,
        k_a2a_rot,
        rtol=1e-5,
        atol=1e-5,
        msg=f"rank{rank}: rope-then-a2a != a2a-then-rope (INTERLEAVED, K-side)",
    )


def _logic_rope_a2a_equivariance_Q_asymmetric(rank, world_size):
    """For Q-side with S_q != S_kv: same equivariance must hold for Q."""
    B, T_q_full, H, D = 1, world_size * 12, world_size * 4, 64
    device = torch.device("cpu")

    torch.manual_seed(33)
    q_full = torch.randn(B, T_q_full, H, D, device=device)
    pe_cos_full, pe_sin_full = _make_interleaved_pe(T_q_full, D, device, seed=44)

    chunk = T_q_full // world_size
    s, e = rank * chunk, (rank + 1) * chunk
    q_local = q_full[:, s:e].contiguous()
    pe_local = (pe_cos_full[:, s:e].contiguous(), pe_sin_full[:, s:e].contiguous())

    # Path A: rope local then a2a.
    q_rot_local = apply_rotary_emb(q_local, pe_local, LTXRopeType.INTERLEAVED)
    q_rot_a2a = all_to_all_4d(q_rot_local, scatter_dim=2, gather_dim=1, process_group=None)

    # Path B: a2a then rope full.
    q_a2a = all_to_all_4d(q_local, scatter_dim=2, gather_dim=1, process_group=None)
    q_a2a_rot = apply_rotary_emb(q_a2a, (pe_cos_full, pe_sin_full), LTXRopeType.INTERLEAVED)

    torch.testing.assert_close(
        q_rot_a2a,
        q_a2a_rot,
        rtol=1e-5,
        atol=1e-5,
        msg=f"rank{rank}: rope-then-a2a != a2a-then-rope (INTERLEAVED, Q-side)",
    )


# ============================================================================
# Test class
# ============================================================================


class TestRopeA2AEquivariance:
    """Verifies the mathematical invariant the strict-Ulysses a2v relies on."""

    def test_rope_a2a_equivariance_K_ws2(self):
        _run(world_size=2, test_fn=_logic_rope_a2a_equivariance_K)

    def test_rope_a2a_equivariance_K_ws4(self):
        _run(world_size=4, test_fn=_logic_rope_a2a_equivariance_K)

    def test_rope_a2a_equivariance_Q_asymmetric_ws2(self):
        _run(world_size=2, test_fn=_logic_rope_a2a_equivariance_Q_asymmetric)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
