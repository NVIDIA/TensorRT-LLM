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
"""Multi-GPU tests for RingAttention and Ring + Ulysses composition.

Uses the same spawn pattern as test_attn2d_attention.py.  Process groups for
ring (``cp`` dim) and Ulysses (``ulysses`` dim) match ``VisualGenMapping``:
mesh layout ``cfg-tp-cp-ulysses`` with shape ``(1, 1, ring_size, ulysses_size)``.

Ring-only: ``ulysses_size=1``, ``ring_size=world_size``.

Composition matches ``attention.py``: ``UlyssesAttention(RingAttention(inner))``.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_ring_attention.py -v
"""

import math
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable, Optional, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

try:
    from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
    from tensorrt_llm._torch.visual_gen.attention_backend import RingAttention, UlyssesAttention
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import FlashAttn4Attention
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
        _flash_attn_fwd as _fa4_fwd,
    )
    from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    _fa4_fwd = None

_flash_attn4_available = _fa4_fwd is not None


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Test-only inner backend: VanillaAttention with LSE output (RingAttention)
# =============================================================================


class _LSEVanillaAttention(nn.Module):
    """Inner backend with ``forward_with_lse`` for RingAttention tests."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.NHD

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_lse(cls) -> bool:
        return True

    def forward(self, q, k, v, batch_size=None, seq_len=None, **kwargs):
        q_t = q.transpose(1, 2).float()
        k_t = k.transpose(1, 2).float()
        v_t = v.transpose(1, 2).float()
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale)
        return out.to(q.dtype).transpose(1, 2).contiguous()

    def forward_with_lse(self, q, k, v, batch_size=None, seq_len=None, **kwargs):
        q_t = q.transpose(1, 2).float()
        k_t = k.transpose(1, 2).float()
        v_t = v.transpose(1, 2).float()
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale
        lse = torch.logsumexp(scores, dim=-1)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t)
        return out.to(q.dtype).transpose(1, 2).contiguous(), lse


# =============================================================================
# Distributed helpers
# =============================================================================


def _init_worker(rank: int, world_size: int, backend: str, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port):
    try:
        _init_worker(rank, world_size, backend, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        _cleanup()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")

    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, backend, test_fn, port),
        nprocs=world_size,
        join=True,
    )


def _cp_ulysses_mesh_groups(
    ring_size: int, ulysses_size: int
) -> Tuple[dist.ProcessGroup, Optional[dist.ProcessGroup]]:
    """DeviceMesh (1,1,ring_size,ulysses_size) matching VisualGenMapping axis order.

    Returns ``(ring_group, ulysses_group)``.  ``ulysses_group`` is ``None`` when
    ``ulysses_size == 1``.
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = init_device_mesh(
        device_type,
        (1, 1, ring_size, ulysses_size),
        mesh_dim_names=("cfg", "tp", "cp", "ulysses"),
    )
    ring_pg = mesh["cp"].get_group()
    uly_pg = mesh["ulysses"].get_group() if ulysses_size > 1 else None
    return ring_pg, uly_pg


def _seq_shard_bounds(rank: int, world_size: int, seq_full: int) -> Tuple[int, int]:
    """Contiguous global-token shard for rank (same linear order as cp-major, uly-minor)."""
    assert seq_full % world_size == 0
    chunk = seq_full // world_size
    return rank * chunk, (rank + 1) * chunk


# =============================================================================
# Ring only (1D cp group)
# =============================================================================


def _logic_ring_forward(rank, world_size):
    """Ring CP size ``world_size``: output shape [B, S/P, H, D], finite values."""
    ring_size, ulysses_size = world_size, 1
    assert dist.get_world_size() == ring_size * ulysses_size

    batch, seq_per_rank, num_heads, head_dim = 2, 8, 8, 64
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    ring_pg, _ = _cp_ulysses_mesh_groups(ring_size, ulysses_size)
    inner = _LSEVanillaAttention(num_heads=num_heads, head_dim=head_dim)
    attn = RingAttention(inner, ring_pg)

    q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)

    output = attn(q, k, v, batch_size=batch)
    assert output.shape == q.shape, f"Rank {rank}: expected {q.shape}, got {output.shape}"
    assert torch.isfinite(output).all(), f"Rank {rank}: non-finite output"


def _logic_ring_vs_standard(rank, world_size):
    """Ring output on each rank matches the corresponding SDPA sequence shard."""
    ring_size, ulysses_size = world_size, 1
    batch, num_heads, head_dim = 2, 8, 64
    seq_per_rank = 8
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ring_pg, _ = _cp_ulysses_mesh_groups(ring_size, ulysses_size)

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)

    lo, hi = _seq_shard_bounds(rank, world_size, seq_full)
    q_shard = q_full[:, lo:hi].contiguous()
    k_shard = k_full[:, lo:hi].contiguous()
    v_shard = v_full[:, lo:hi].contiguous()

    inner = _LSEVanillaAttention(num_heads=num_heads, head_dim=head_dim)
    attn = RingAttention(inner, ring_pg)
    ring_out = attn(q_shard, k_shard, v_shard, batch_size=batch)

    scale = 1.0 / math.sqrt(head_dim)
    q_std = q_full.transpose(1, 2).float()
    k_std = k_full.transpose(1, 2).float()
    v_std = v_full.transpose(1, 2).float()
    std_output = F.scaled_dot_product_attention(q_std, k_std, v_std, scale=scale)
    std_output = std_output.transpose(1, 2).to(ring_out.dtype)
    expected = std_output[:, lo:hi]

    torch.testing.assert_close(
        ring_out,
        expected,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: RingAttention differs from SDPA reference",
    )


def _logic_ring_invalid_mask(rank, world_size):
    """RingAttention rejects non-FULL masks for self-attention."""
    ring_size, ulysses_size = world_size, 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ring_pg, _ = _cp_ulysses_mesh_groups(ring_size, ulysses_size)
    inner = _LSEVanillaAttention(num_heads=8, head_dim=64)
    attn = RingAttention(inner, ring_pg)

    q = torch.randn(2, 8, 8, 64, device=device)
    k = torch.randn(2, 8, 8, 64, device=device)
    v = torch.randn(2, 8, 8, 64, device=device)

    with pytest.raises(NotImplementedError, match="only supports FULL attention mask"):
        attn(q, k, v, batch_size=2, attention_mask=PredefinedAttentionMask.CAUSAL)


def _logic_ring_fa4_vs_standard(rank, world_size):
    """Ring with FlashAttn4 inner matches SDPA reference shards."""
    if not _flash_attn4_available:
        pytest.skip("FlashAttn4 JIT kernels not available")

    ring_size, ulysses_size = world_size, 1
    batch, num_heads, head_dim = 1, 8, 128
    seq_per_rank = 16
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ring_pg, _ = _cp_ulysses_mesh_groups(ring_size, ulysses_size)

    inner = FlashAttn4Attention(num_heads=num_heads, head_dim=head_dim)
    attn = RingAttention(inner, ring_pg)

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    lo, hi = _seq_shard_bounds(rank, world_size, seq_full)
    q_shard = q_full[:, lo:hi].contiguous()
    k_shard = k_full[:, lo:hi].contiguous()
    v_shard = v_full[:, lo:hi].contiguous()

    ring_out = attn(q_shard, k_shard, v_shard, batch_size=batch)

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(ring_out.dtype)
    )
    expected = ref[:, lo:hi]
    torch.testing.assert_close(
        ring_out,
        expected,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: Ring FA4 differs from SDPA reference",
    )


# =============================================================================
# Ring + Ulysses: UlyssesAttention(RingAttention(inner))
# =============================================================================


def _logic_ring_ulysses_forward(rank, world_size):
    """ring_size=2, ulysses_size=2 on 4 GPUs: finite output, correct shape."""
    ring_size, ulysses_size = 2, 2
    assert world_size == ring_size * ulysses_size

    batch, seq_per_rank, num_heads, head_dim = 2, 8, 8, 64
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    ring_pg, uly_pg = _cp_ulysses_mesh_groups(ring_size, ulysses_size)
    assert uly_pg is not None

    inner = _LSEVanillaAttention(num_heads=num_heads // ulysses_size, head_dim=head_dim)
    attn = UlyssesAttention(RingAttention(inner, ring_pg), uly_pg)

    q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)

    output = attn(q, k, v, batch_size=batch)
    assert output.shape == q.shape, f"Rank {rank}: expected {q.shape}, got {output.shape}"
    assert torch.isfinite(output).all(), f"Rank {rank}: non-finite output"


def _logic_ring_ulysses_vs_standard(rank, world_size):
    """Combined parallelism matches SDPA on full sequence; compare local shard."""
    ring_size, ulysses_size = 2, 2
    assert world_size == ring_size * ulysses_size

    batch, num_heads, head_dim = 2, 8, 64
    seq_per_rank = 8
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ring_pg, uly_pg = _cp_ulysses_mesh_groups(ring_size, ulysses_size)

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)

    lo, hi = _seq_shard_bounds(rank, world_size, seq_full)
    q_shard = q_full[:, lo:hi].contiguous()
    k_shard = k_full[:, lo:hi].contiguous()
    v_shard = v_full[:, lo:hi].contiguous()

    inner = _LSEVanillaAttention(num_heads=num_heads // ulysses_size, head_dim=head_dim)
    attn = UlyssesAttention(RingAttention(inner, ring_pg), uly_pg)
    combo_out = attn(q_shard, k_shard, v_shard, batch_size=batch)

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(combo_out.dtype)
    )
    expected = ref[:, lo:hi]

    torch.testing.assert_close(
        combo_out,
        expected,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: Ring+Ulysses differs from SDPA reference",
    )


def _logic_ring_ulysses_fa4_vs_standard(rank, world_size):
    """Ring + Ulysses with FA4 inner matches SDPA shards."""
    if not _flash_attn4_available:
        pytest.skip("FlashAttn4 JIT kernels not available")

    ring_size, ulysses_size = 2, 2
    batch, num_heads, head_dim = 1, 8, 128
    seq_per_rank = 16
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ring_pg, uly_pg = _cp_ulysses_mesh_groups(ring_size, ulysses_size)

    inner = FlashAttn4Attention(num_heads=num_heads // ulysses_size, head_dim=head_dim)
    attn = UlyssesAttention(RingAttention(inner, ring_pg), uly_pg)

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    lo, hi = _seq_shard_bounds(rank, world_size, seq_full)
    q_shard = q_full[:, lo:hi].contiguous()
    k_shard = k_full[:, lo:hi].contiguous()
    v_shard = v_full[:, lo:hi].contiguous()

    combo_out = attn(q_shard, k_shard, v_shard, batch_size=batch)

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(combo_out.dtype)
    )
    expected = ref[:, lo:hi]

    torch.testing.assert_close(
        combo_out,
        expected,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: Ring+Ulysses FA4 differs from SDPA reference",
    )


# =============================================================================
# Init guards (RingAttention)
# =============================================================================


def _logic_ring_init_guard_no_lse(rank, world_size):
    class _NoLSEBackend(nn.Module):
        num_heads = 8
        head_dim = 64
        _preferred_layout = AttentionTensorLayout.NHD

        @property
        def preferred_layout(self):
            return self._preferred_layout

        @classmethod
        def support_lse(cls):
            return False

        @classmethod
        def support_fused_qkv(cls):
            return False

    pg = dist.new_group(list(range(world_size)), use_local_synchronization=True)
    with pytest.raises(ValueError, match="LSE"):
        RingAttention(_NoLSEBackend(), pg)


# =============================================================================
# Test classes
# =============================================================================


class TestRingAttention:
    """Ring CP only (``ulysses_size=1``), mesh ``cp = world_size``."""

    def test_ring_forward(self):
        run_test_in_distributed(world_size=4, test_fn=_logic_ring_forward, use_cuda=True)

    def test_ring_vs_standard_attention(self):
        run_test_in_distributed(world_size=4, test_fn=_logic_ring_vs_standard, use_cuda=True)

    def test_ring_invalid_mask_raises(self):
        run_test_in_distributed(world_size=4, test_fn=_logic_ring_invalid_mask, use_cuda=True)


class TestRingAttentionFlashAttn4:
    """Ring with FA4 inner (production-style)."""

    def test_ring_fa4_vs_standard(self):
        run_test_in_distributed(world_size=4, test_fn=_logic_ring_fa4_vs_standard, use_cuda=True)


class TestRingUlyssesAttention:
    """``UlyssesAttention(RingAttention(inner))`` on 4 GPUs (2×2 cp × ulysses)."""

    def test_ring_ulysses_forward(self):
        run_test_in_distributed(world_size=4, test_fn=_logic_ring_ulysses_forward, use_cuda=True)

    def test_ring_ulysses_vs_standard_attention(self):
        run_test_in_distributed(
            world_size=4, test_fn=_logic_ring_ulysses_vs_standard, use_cuda=True
        )


class TestRingUlyssesAttentionFlashAttn4:
    """Ring + Ulysses with FA4 inner."""

    def test_ring_ulysses_fa4_vs_standard(self):
        run_test_in_distributed(
            world_size=4, test_fn=_logic_ring_ulysses_fa4_vs_standard, use_cuda=True
        )


class TestRingAttentionInitGuards:
    def test_inner_without_lse_raises(self):
        run_test_in_distributed(world_size=4, test_fn=_logic_ring_init_guard_no_lse, use_cuda=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
