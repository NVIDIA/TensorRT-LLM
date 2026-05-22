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
"""Multi-GPU tests for Attention2DAttention.

These tests use torch.multiprocessing.spawn to launch multiple processes internally.
Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_attn2d_attention.py -v
"""

import math
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

try:
    from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
    from tensorrt_llm._torch.visual_gen.attention_backend import Attention2DAttention
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

# FA4 kernel availability (separate from the combine kernel used by Attention2DAttention)
_flash_attn4_available = _fa4_fwd is not None


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Test-only inner backend: VanillaAttention with LSE output
# =============================================================================


class _LSEVanillaAttention(nn.Module):
    """VanillaAttention extended with LSE output for Attention2DAttention tests.

    Computes attention manually so that both the output and the log-sum-exp
    values are available, as required by Attention2DAttention.
    """

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
        """Return (output [B, S, H, D], lse [B, H, S])."""
        q_t = q.transpose(1, 2).float()  # [B, H, S_q, D]
        k_t = k.transpose(1, 2).float()  # [B, H, S_k, D]
        v_t = v.transpose(1, 2).float()  # [B, H, S_k, D]
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale  # [B, H, S_q, S_k]
        lse = torch.logsumexp(scores, dim=-1)  # [B, H, S_q]
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t)  # [B, H, S_q, D]
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


def _make_process_groups(rank, world_size, row_size, col_size):
    """Create row and col process groups for a single-CFG-group Attention2D mesh.

    Returns (row_pg, col_pg) for the calling rank.
    """
    mesh_size = row_size * col_size
    assert world_size == mesh_size, f"world_size ({world_size}) must equal row_size * col_size"

    row_pg = None
    col_pg = None

    for r in range(row_size):
        ranks = [r * col_size + c for c in range(col_size)]
        pg = dist.new_group(ranks, use_local_synchronization=True)
        if rank in ranks:
            row_pg = pg

    for c in range(col_size):
        ranks = [r * col_size + c for r in range(row_size)]
        pg = dist.new_group(ranks, use_local_synchronization=True)
        if rank in ranks:
            col_pg = pg

    return row_pg, col_pg


# =============================================================================
# Test logic (module-level for mp.spawn pickling)
# =============================================================================


def _logic_attn2d_forward(rank, world_size):
    """Basic forward pass: output shape is [B, S/P, H, D] and values are finite."""
    row_size, col_size = 2, 2
    batch, seq_per_rank, num_heads, head_dim = 2, 8, 8, 64
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    inner = _LSEVanillaAttention(num_heads=num_heads, head_dim=head_dim)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)

    # No mask (None) and explicit FULL mask are both valid
    output = attn(q, k, v, batch_size=batch)
    assert output.shape == q.shape, f"Rank {rank}: expected {q.shape}, got {output.shape}"
    assert torch.isfinite(output).all(), f"Rank {rank}: output contains non-finite values"

    output_full = attn(q, k, v, batch_size=batch, attention_mask=PredefinedAttentionMask.FULL)
    assert output_full.shape == q.shape


def _logic_attn2d_vs_standard(rank, world_size):
    """Attention2DAttention output matches standard full-sequence SDPA on every shard."""
    row_size, col_size = 2, 2
    batch, num_heads, head_dim = 2, 8, 64
    seq_per_rank = 8
    seq_full = seq_per_rank * world_size  # 32

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    # Each rank takes its shard; all ranks build from the same full tensors.
    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)

    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    inner = _LSEVanillaAttention(num_heads=num_heads, head_dim=head_dim)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    attn2d_output = attn(q_shard, k_shard, v_shard, batch_size=batch)

    # Reference: standard full-sequence SDPA
    scale = 1.0 / math.sqrt(head_dim)
    q_std = q_full.transpose(1, 2).float()  # [B, H, S, D]
    k_std = k_full.transpose(1, 2).float()
    v_std = v_full.transpose(1, 2).float()
    std_output = F.scaled_dot_product_attention(q_std, k_std, v_std, scale=scale)
    std_output = std_output.transpose(1, 2).to(attn2d_output.dtype)  # [B, S, H, D]

    expected_shard = std_output[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]
    torch.testing.assert_close(
        attn2d_output,
        expected_shard,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: Attention2D output differs from standard attention",
    )


def _logic_attn2d_invalid_mask(rank, world_size):
    """Attention2DAttention raises ValueError for non-FULL attention mask."""
    row_size, col_size = 2, 2
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    inner = _LSEVanillaAttention(num_heads=8, head_dim=64)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    q = torch.randn(2, 8, 8, 64, device=device)
    k = torch.randn(2, 8, 8, 64, device=device)
    v = torch.randn(2, 8, 8, 64, device=device)

    with pytest.raises(ValueError, match="FULL"):
        attn(q, k, v, batch_size=2, attention_mask=PredefinedAttentionMask.CAUSAL)


def _logic_attn2d_asymmetric_mesh_1x4(rank, world_size):
    """1x4 mesh: no row all-gather (row_size=1), all K/V gathered via col_size=4."""
    row_size, col_size = 1, 4
    batch, num_heads, head_dim = 2, 8, 64
    seq_per_rank = 8
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    inner = _LSEVanillaAttention(num_heads=num_heads, head_dim=head_dim)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)

    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    output = attn(q_shard, k_shard, v_shard, batch_size=batch)

    assert output.shape == q_shard.shape, (
        f"Rank {rank}: expected {q_shard.shape}, got {output.shape}"
    )

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(output.dtype)
    )
    expected_shard = ref[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]
    torch.testing.assert_close(
        output,
        expected_shard,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: 1x4 mesh output differs from standard attention",
    )


def _logic_attn2d_asymmetric_mesh_4x1(rank, world_size):
    """4x1 mesh: all Q gathered via row_size=4, no col all-gather (col_size=1)."""
    row_size, col_size = 4, 1
    batch, num_heads, head_dim = 2, 8, 64
    seq_per_rank = 8
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    inner = _LSEVanillaAttention(num_heads=num_heads, head_dim=head_dim)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)

    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    output = attn(q_shard, k_shard, v_shard, batch_size=batch)

    assert output.shape == q_shard.shape, (
        f"Rank {rank}: expected {q_shard.shape}, got {output.shape}"
    )

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(output.dtype)
    )
    expected_shard = ref[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]
    torch.testing.assert_close(
        output,
        expected_shard,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Rank {rank}: 4x1 mesh output differs from standard attention",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestAttn2DAttention:
    """Tests for Attention2DAttention module."""

    def test_attn2d_forward(self):
        """Forward pass returns correct shape and finite values (2x2 mesh)."""
        run_test_in_distributed(world_size=4, test_fn=_logic_attn2d_forward, use_cuda=True)

    def test_attn2d_vs_standard_attention(self):
        """Attention2DAttention output matches standard full-sequence SDPA (2x2 mesh)."""
        run_test_in_distributed(world_size=4, test_fn=_logic_attn2d_vs_standard, use_cuda=True)

    def test_attn2d_invalid_mask_raises(self):
        """CAUSAL mask raises ValueError (only FULL is supported)."""
        run_test_in_distributed(world_size=4, test_fn=_logic_attn2d_invalid_mask, use_cuda=True)


class TestAttn2DAttentionMeshVariants:
    """Attention2DAttention with asymmetric mesh configurations."""

    def test_attn2d_1x4_mesh(self):
        """1x4 mesh: row_size=1 (no row all-gather), col_size=4."""
        run_test_in_distributed(
            world_size=4, test_fn=_logic_attn2d_asymmetric_mesh_1x4, use_cuda=True
        )

    def test_attn2d_4x1_mesh(self):
        """4x1 mesh: row_size=4, col_size=1 (no col all-gather)."""
        run_test_in_distributed(
            world_size=4, test_fn=_logic_attn2d_asymmetric_mesh_4x1, use_cuda=True
        )


def _logic_attn2d_fa4_vs_standard(rank, world_size):
    """Attention2DAttention with FlashAttn4 inner backend matches standard SDPA (2x2 mesh).

    2x2 mesh: both row_size and col_size are 2.  After gathering, Q and K/V have the
    same full sequence length, so FA4 runs with equal Q/KV lengths and _combine is
    called with N=2 partial results.
    """
    if not _flash_attn4_available:
        pytest.skip("FlashAttn4 JIT kernels not available")

    row_size, col_size = 2, 2
    batch, num_heads, head_dim = 1, 8, 128
    seq_per_rank = 16
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    inner = FlashAttn4Attention(num_heads=num_heads, head_dim=head_dim)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    output = attn(q_shard, k_shard, v_shard, batch_size=batch)

    assert output.shape == q_shard.shape, (
        f"Rank {rank}: expected {q_shard.shape}, got {output.shape}"
    )

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(output.dtype)
    )
    expected_shard = ref[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]
    torch.testing.assert_close(
        output,
        expected_shard,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FA4 Attention2D output differs from standard attention",
    )


def _logic_attn2d_fa4_asymmetric_1x4(rank, world_size):
    """Attention2DAttention with FlashAttn4 inner backend matches standard SDPA (1x4 mesh).

    1x4 mesh: row_size=1, col_size=4.  Q is NOT gathered (row_size=1), so FA4 runs with
    asymmetric Q/KV sequence lengths — Q has shard_seq tokens while K/V have the full
    4*shard_seq tokens.  The output reduce-scatter (_combine) is also skipped (row_size=1),
    so this exercises FA4 with non-square Q/KV without the combine kernel.
    """
    if not _flash_attn4_available:
        pytest.skip("FlashAttn4 JIT kernels not available")

    row_size, col_size = 1, 4
    batch, num_heads, head_dim = 1, 8, 128
    seq_per_rank = 16
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    inner = FlashAttn4Attention(num_heads=num_heads, head_dim=head_dim)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    output = attn(q_shard, k_shard, v_shard, batch_size=batch)

    assert output.shape == q_shard.shape, (
        f"Rank {rank}: expected {q_shard.shape}, got {output.shape}"
    )

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(output.dtype)
    )
    expected_shard = ref[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]
    torch.testing.assert_close(
        output,
        expected_shard,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FA4 1x4 mesh output differs from standard attention",
    )


def _logic_attn2d_fa4_asymmetric_4x1(rank, world_size):
    """Attention2DAttention with FlashAttn4 inner backend matches standard SDPA (4x1 mesh).

    4x1 mesh: row_size=4, col_size=1.  Q is gathered across all 4 ranks so FA4 sees
    equal Q/KV lengths.  K/V are NOT gathered (col_size=1).  The output reduce-scatter
    calls _combine with N=4 partial results, exercising flash_attn_combine with a larger
    fan-in than the symmetric 2x2 case (N=2).
    """
    if not _flash_attn4_available:
        pytest.skip("FlashAttn4 JIT kernels not available")

    row_size, col_size = 4, 1
    batch, num_heads, head_dim = 1, 8, 128
    seq_per_rank = 16
    seq_full = seq_per_rank * world_size

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    row_pg, col_pg = _make_process_groups(rank, world_size, row_size, col_size)

    inner = FlashAttn4Attention(num_heads=num_heads, head_dim=head_dim)
    try:
        attn = Attention2DAttention(inner, row_pg, col_pg)
    except ImportError:
        pytest.skip("flash_attn_combine JIT kernels not available")

    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    output = attn(q_shard, k_shard, v_shard, batch_size=batch)

    assert output.shape == q_shard.shape, (
        f"Rank {rank}: expected {q_shard.shape}, got {output.shape}"
    )

    scale = 1.0 / math.sqrt(head_dim)
    ref = (
        F.scaled_dot_product_attention(
            q_full.transpose(1, 2).float(),
            k_full.transpose(1, 2).float(),
            v_full.transpose(1, 2).float(),
            scale=scale,
        )
        .transpose(1, 2)
        .to(output.dtype)
    )
    expected_shard = ref[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]
    torch.testing.assert_close(
        output,
        expected_shard,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FA4 4x1 mesh output differs from standard attention",
    )


class TestFlashAttn4Forward:
    """Smoke tests for FlashAttn4Attention.forward directly (single GPU, no wrapping)."""

    def test_fa4_forward_returns_correct_shape(self):
        """FlashAttn4Attention.forward runs end-to-end and returns the correct shape."""
        if not _flash_attn4_available:
            pytest.skip("FlashAttn4 JIT kernels not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        batch, seq, num_heads, head_dim = 1, 16, 8, 128
        device = torch.device("cuda:0")

        inner = FlashAttn4Attention(num_heads=num_heads, head_dim=head_dim)
        q = torch.randn(batch, seq, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, seq, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, seq, num_heads, head_dim, device=device, dtype=torch.bfloat16)

        out = inner.forward(q=q, k=k, v=v)
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"


class TestAttn2DAttentionFlashAttn4:
    """Attention2DAttention using FlashAttn4 as the inner backend (production path)."""

    def test_attn2d_fa4_vs_standard(self):
        """FlashAttn4 inner backend: output matches standard full-sequence SDPA (2x2 mesh)."""
        run_test_in_distributed(world_size=4, test_fn=_logic_attn2d_fa4_vs_standard, use_cuda=True)

    def test_attn2d_fa4_1x4_mesh(self):
        """FA4 with 1x4 mesh: asymmetric Q/KV lengths, no _combine call."""
        run_test_in_distributed(
            world_size=4, test_fn=_logic_attn2d_fa4_asymmetric_1x4, use_cuda=True
        )

    def test_attn2d_fa4_4x1_mesh(self):
        """FA4 with 4x1 mesh: _combine called with N=4 partial results."""
        run_test_in_distributed(
            world_size=4, test_fn=_logic_attn2d_fa4_asymmetric_4x1, use_cuda=True
        )


def _logic_init_guard_no_lse(rank, world_size):
    """Inner backend with support_lse()=False raises RuntimeError."""

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

    pg = dist.new_group([0], use_local_synchronization=True)
    try:
        Attention2DAttention(_NoLSEBackend(), pg, pg)
        raise AssertionError("Expected RuntimeError or ImportError")
    except ImportError:
        pass  # flash_attn_combine not built; guard not reachable — acceptable
    except RuntimeError as e:
        assert "support_lse" in str(e), f"Unexpected RuntimeError: {e}"


def _logic_init_guard_missing_head_attrs(rank, world_size):
    """Inner backend missing head_dim/num_heads raises RuntimeError."""

    class _NoHeadAttrsBackend(nn.Module):
        _preferred_layout = AttentionTensorLayout.NHD

        @property
        def preferred_layout(self):
            return self._preferred_layout

        @classmethod
        def support_lse(cls):
            return True

        @classmethod
        def support_fused_qkv(cls):
            return False

    pg = dist.new_group([0], use_local_synchronization=True)
    try:
        Attention2DAttention(_NoHeadAttrsBackend(), pg, pg)
        raise AssertionError("Expected RuntimeError or ImportError")
    except ImportError:
        pass  # flash_attn_combine not built; guards not reachable — acceptable
    except RuntimeError as e:
        assert "head_dim" in str(e) or "num_heads" in str(e), f"Unexpected RuntimeError: {e}"


class TestAttn2DAttentionInitGuards:
    """Attention2DAttention.__init__ rejects invalid inner backends."""

    def test_missing_support_lse_raises(self):
        """Inner backend with support_lse()=False raises RuntimeError."""
        run_test_in_distributed(world_size=1, test_fn=_logic_init_guard_no_lse, use_cuda=False)

    def test_missing_head_attrs_raises(self):
        """Inner backend missing head_dim or num_heads raises RuntimeError."""
        run_test_in_distributed(
            world_size=1, test_fn=_logic_init_guard_missing_head_attrs, use_cuda=False
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
