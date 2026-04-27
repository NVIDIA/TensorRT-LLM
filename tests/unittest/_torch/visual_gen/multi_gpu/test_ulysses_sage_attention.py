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
"""
Multi-GPU tests for SageAttention + Ulysses sequence parallelism.

Verifies that SageAttention TRTLLM kernels produce correct results when
combined with Ulysses sequence parallelism (sharding the sequence across GPUs).

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_sage_ulysses_attention.py -v
"""

import functools
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from tensorrt_llm._torch.visual_gen.attention_backend import UlyssesAttention
    from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


def _cuda_cc():
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return -1, -1


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Clean up TLLM_DISABLE_MPI env var after tests complete."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_ulysses_attention.py)
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port):
    """Worker function that runs in each spawned process.  Module-level for pickling."""
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for SageAttention")
    if _cuda_cc()[0] != 10:
        pytest.skip("SageAttention requires CUDA compute capability major version 10")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")

    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, "nccl", test_fn, port),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# Test logic functions (module-level so mp.spawn can pickle them)
# =============================================================================


def _logic_sage_ulysses_forward(rank, world_size, *, sage_attn_qk_int8: bool):
    """SageAttention + Ulysses forward pass: output shape is correct and contains no Inf/NaN."""
    batch = 1
    seq_per_rank = 256
    num_heads = world_size * 4  # 8 total, 4 per rank
    head_dim = 128

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    blk_k = 16 if sage_attn_qk_int8 else 1
    inner = TrtllmAttention(
        layer_idx=0,
        num_heads=num_heads // world_size,
        head_dim=head_dim,
        sage_attn_num_elts_per_blk_q=1,
        sage_attn_num_elts_per_blk_k=blk_k,
        sage_attn_num_elts_per_blk_v=1,
        sage_attn_qk_int8=sage_attn_qk_int8,
    )
    attention = UlyssesAttention(inner_backend=inner, process_group=None)

    q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    output = attention(q, k, v)

    assert output.shape[0] == batch, f"Rank {rank}: batch dim mismatch"
    assert output.shape[1] == seq_per_rank, f"Rank {rank}: seq dim mismatch"
    assert torch.isfinite(output).all(), f"Rank {rank}: Inf/NaN in output"


def _logic_sage_ulysses_vs_reference(
    rank,
    world_size,
    *,
    sage_attn_qk_int8: bool,
    sage_attn_num_elts_per_blk_k: int,
):
    """
    Ulysses+SageAttention output is close to full-sequence SDPA reference.

    All ranks use the same random seed to build identical full-sequence tensors,
    then each rank slices its portion.  The Ulysses output on each shard should
    match the corresponding slice of the reference SDPA output within the same
    tolerances used by the single-GPU SageAttention test.
    """
    batch = 1
    seq_per_rank = 256
    seq_full = seq_per_rank * world_size
    num_heads = world_size * 4
    head_dim = 128

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    inner = TrtllmAttention(
        layer_idx=0,
        num_heads=num_heads // world_size,
        head_dim=head_dim,
        sage_attn_num_elts_per_blk_q=1,
        sage_attn_num_elts_per_blk_k=sage_attn_num_elts_per_blk_k,
        sage_attn_num_elts_per_blk_v=1,
        sage_attn_qk_int8=sage_attn_qk_int8,
    )
    attention = UlyssesAttention(inner_backend=inner, process_group=None)

    # Output shape: [B, S/P, H, D]
    ulysses_out = attention(q_shard, k_shard, v_shard)

    # Reference: standard SDPA on the full sequence.
    ref_out = F.scaled_dot_product_attention(
        q_full.transpose(1, 2),  # [B, H, S, D]
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
    )
    ref_out = ref_out.transpose(1, 2).contiguous()  # [B, S, H, D]
    ref_shard = ref_out[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]  # [B, S/P, H, D]

    ulysses_out = ulysses_out.view(batch, seq_per_rank, num_heads, head_dim).to(torch.bfloat16)
    ref_shard = ref_shard.to(torch.bfloat16)

    cos_sim = F.cosine_similarity(
        ulysses_out.reshape(-1).float(),
        ref_shard.reshape(-1).float(),
        dim=0,
    ).item()
    assert cos_sim > 0.990, f"Rank {rank}: cosine similarity {cos_sim:.6f} is below threshold 0.990"
    torch.testing.assert_close(ulysses_out, ref_shard, atol=5e-1, rtol=5e-1)


# =============================================================================
# Test class
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    _cuda_cc()[0] != 10,
    reason="SageAttention requires CUDA compute capability major version 10",
)
class TestSageUlyssesAttention:
    """SageAttention TRTLLM kernels combined with Ulysses sequence parallelism."""

    @pytest.mark.parametrize("sage_attn_qk_int8", [True, False])
    def test_sage_ulysses_forward(self, sage_attn_qk_int8: bool):
        """Forward pass runs without error and produces finite output."""
        if sage_attn_qk_int8 and _cuda_cc()[1] == 3:
            pytest.skip("SM103 does not have Int8 Tensor Cores.")
        run_test_in_distributed(
            world_size=2,
            test_fn=functools.partial(
                _logic_sage_ulysses_forward, sage_attn_qk_int8=sage_attn_qk_int8
            ),
        )

    @pytest.mark.parametrize(
        "sage_attn_qk_int8,sage_attn_num_elts_per_blk_k",
        [
            (True, 4),  # Int8, small block
            (True, 16),  # Int8, large block
            (False, 1),  # FP8 / no Int8 quantization
        ],
    )
    def test_sage_ulysses_vs_reference(
        self, sage_attn_qk_int8: bool, sage_attn_num_elts_per_blk_k: int
    ):
        """Ulysses+SageAttention output matches full-sequence SDPA within approximation tolerance."""
        if sage_attn_qk_int8 and _cuda_cc()[1] == 3:
            pytest.skip("SM103 does not have Int8 Tensor Cores.")
        run_test_in_distributed(
            world_size=2,
            test_fn=functools.partial(
                _logic_sage_ulysses_vs_reference,
                sage_attn_qk_int8=sage_attn_qk_int8,
                sage_attn_num_elts_per_blk_k=sage_attn_num_elts_per_blk_k,
            ),
        )
