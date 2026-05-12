# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU tests for UlyssesCrossAttention.

Cross-attention case: S_q != S_kv, K/V pre-projected. Three-collective
pattern (Q a2a + fused K|V 5D a2a + output a2a).

Uses torch.multiprocessing.spawn; CPU+gloo when CUDA not available.
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import math

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from tensorrt_llm._torch.visual_gen.attention_backend import VanillaAttention
    from tensorrt_llm._torch.visual_gen.attention_backend.parallel import UlyssesCrossAttention
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _init_dist(rank, world_size, backend, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _worker(rank, world_size, backend, test_fn, port):
    try:
        _init_dist(rank, world_size, backend, port)
        test_fn(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size, test_fn, use_cuda=False):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs")
    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()
    mp.spawn(_worker, args=(world_size, backend, test_fn, port), nprocs=world_size, join=True)


# ============================================================================
# Worker logics (module-level for pickling)
# ============================================================================


def _logic_init(rank, world_size):
    """Wrapper instantiation: full head counts exposed."""
    H, H_kv, d_h = world_size * 8, world_size * 8, 64
    inner = VanillaAttention(
        num_heads=H // world_size, head_dim=d_h, num_kv_heads=H_kv // world_size
    )
    wrap = UlyssesCrossAttention(inner_backend=inner, process_group=None)
    assert wrap.num_heads == H
    assert wrap.num_kv_heads == H_kv
    assert wrap.head_dim == d_h
    assert wrap.world_size == world_size
    assert wrap.support_fused_qkv() is False
    assert rank >= 0


def _logic_forward_shape(rank, world_size):
    """Forward returns Q-side seq-sharded, full-head shape."""
    B, S_q_full, S_kv_full = 2, world_size * 12, world_size * 4
    H, H_kv, d_h = world_size * 8, world_size * 8, 64
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    inner = VanillaAttention(
        num_heads=H // world_size, head_dim=d_h, num_kv_heads=H_kv // world_size
    )
    wrap = UlyssesCrossAttention(inner_backend=inner, process_group=None)

    q = torch.randn(B, S_q_full // world_size, H, d_h, device=device)
    k = torch.randn(B, S_kv_full // world_size, H_kv, d_h, device=device)
    v = torch.randn(B, S_kv_full // world_size, H_kv, d_h, device=device)

    out = wrap.forward(q=q, k=k, v=v)
    assert out.shape == (B, S_q_full // world_size, H, d_h), (
        f"rank{rank}: expected {(B, S_q_full // world_size, H, d_h)}, got {tuple(out.shape)}"
    )


def _logic_parity_vs_full_sdpa(rank, world_size):
    """Output (gathered across ranks) == plain SDPA on full Q/K/V."""
    B, S_q_full, S_kv_full = 2, world_size * 12, world_size * 4
    H, H_kv, d_h = world_size * 8, world_size * 8, 64
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Same Q/K/V on every rank via shared seed.
    torch.manual_seed(123)
    q_full = torch.randn(B, S_q_full, H, d_h, device=device)
    k_full = torch.randn(B, S_kv_full, H_kv, d_h, device=device)
    v_full = torch.randn(B, S_kv_full, H_kv, d_h, device=device)

    # Each rank takes its shard.
    q_chunk = S_q_full // world_size
    kv_chunk = S_kv_full // world_size
    q_shard = q_full[:, rank * q_chunk : (rank + 1) * q_chunk].contiguous()
    k_shard = k_full[:, rank * kv_chunk : (rank + 1) * kv_chunk].contiguous()
    v_shard = v_full[:, rank * kv_chunk : (rank + 1) * kv_chunk].contiguous()

    inner = VanillaAttention(
        num_heads=H // world_size, head_dim=d_h, num_kv_heads=H_kv // world_size
    )
    wrap = UlyssesCrossAttention(inner_backend=inner, process_group=None)
    uly_out = wrap.forward(q=q_shard, k=k_shard, v=v_shard)  # [B, S_q/U, H, D]

    # Reference: plain SDPA on full tensors.
    q_hnd = q_full.transpose(1, 2)  # [B, H, S_q, D]
    k_hnd = k_full.transpose(1, 2)
    v_hnd = v_full.transpose(1, 2)
    ref_out = (
        F.scaled_dot_product_attention(
            q_hnd, k_hnd, v_hnd, scale=1.0 / math.sqrt(d_h), dropout_p=0.0
        )
        .transpose(1, 2)
        .contiguous()
    )  # [B, S_q, H, D]

    expected_shard = ref_out[:, rank * q_chunk : (rank + 1) * q_chunk]
    torch.testing.assert_close(
        uly_out,
        expected_shard,
        rtol=1e-4,
        atol=1e-4,
        msg=f"rank{rank}: UlyssesCrossAttention output diverges from plain SDPA",
    )


def _logic_world_size_1_fast_path(rank, world_size):
    """world_size==1: bit-exact with direct inner.forward (no collectives)."""
    B, S_q, S_kv, H, H_kv, d_h = 2, 12, 4, 8, 8, 64
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    inner = VanillaAttention(num_heads=H, head_dim=d_h, num_kv_heads=H_kv)
    wrap = UlyssesCrossAttention(inner_backend=inner, process_group=None)
    assert wrap.world_size == 1

    torch.manual_seed(7)
    q = torch.randn(B, S_q, H, d_h, device=device)
    k = torch.randn(B, S_kv, H_kv, d_h, device=device)
    v = torch.randn(B, S_kv, H_kv, d_h, device=device)

    uly_out = wrap.forward(q=q, k=k, v=v)
    # Direct call to inner (which expects HND) for reference.
    ref = inner.forward(q=q.transpose(1, 2), k=k.transpose(1, 2), v=v.transpose(1, 2))
    ref = ref.transpose(1, 2).contiguous()
    torch.testing.assert_close(
        uly_out, ref, rtol=0, atol=0, msg="world_size==1 fast path is not bit-exact"
    )


# ============================================================================
# Test classes
# ============================================================================


class TestUlyssesCrossAttention:
    def test_init_world_size_1(self):
        _run(world_size=1, test_fn=_logic_init, use_cuda=False)

    def test_init_world_size_2(self):
        _run(world_size=2, test_fn=_logic_init, use_cuda=False)

    def test_forward_shape_ws2(self):
        _run(world_size=2, test_fn=_logic_forward_shape, use_cuda=False)

    def test_forward_shape_ws4(self):
        _run(world_size=4, test_fn=_logic_forward_shape, use_cuda=False)

    def test_world_size_1_fast_path(self):
        _run(world_size=1, test_fn=_logic_world_size_1_fast_path, use_cuda=False)

    def test_parity_vs_full_sdpa_ws2(self):
        _run(world_size=2, test_fn=_logic_parity_vs_full_sdpa, use_cuda=False)

    def test_parity_vs_full_sdpa_ws4(self):
        _run(world_size=4, test_fn=_logic_parity_vs_full_sdpa, use_cuda=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
