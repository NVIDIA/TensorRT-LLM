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
"""Multi-GPU tests for VSA + Ulysses sequence parallelism."""

import functools
import math
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from tensorrt_llm._torch.visual_gen.attention_backend import (
        CuTeDSLAttention,
        UlyssesAttention,
        VSAMetadataBuilder,
        set_vsa_forward_context,
    )
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.visual_gen.args import VideoSparseAttentionConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_ulysses_sage_attention.py)
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
        pytest.skip("CUDA required for VSA")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")

    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, "nccl", test_fn, port),
        nprocs=world_size,
        join=True,
    )


# (8,8,8) latent -> 512 tokens (256/rank at P=2); the (4,4,4) tile gives 8 cubes
# (even, as the paired-block kernel needs) of 64 tokens (the kernel block_size).
_DIT_SEQ_SHAPE = (8, 8, 8)
_VSA_PATCH_SIZE = (1, 1, 1)
_HEAD_DIM = 128  # CuTe VSA fine-stage kernel requires head_dim == 128
_HEADS_PER_RANK = 4


def _make_vsa_backend(num_heads: int, vsa_sparsity: float) -> "CuTeDSLAttention":
    """CUTEDSL backend on the VSA path; effective sparsity comes from the forward context."""
    return CuTeDSLAttention(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=vsa_sparsity),
    )


def _build_full_seq_vsa_metadata(vsa_sparsity: float, device: torch.device):
    """VSAMetadata for the full sequence — identical on every rank after Ulysses all-to-all."""
    builder = VSAMetadataBuilder()
    return builder.build(
        current_timestep=0,
        raw_latent_shape=_DIT_SEQ_SHAPE,
        patch_size=_VSA_PATCH_SIZE,
        vsa_sparsity=vsa_sparsity,
        device=device,
    )


# =============================================================================
# Test logic functions (module-level so mp.spawn can pickle them)
# =============================================================================


def _logic_vsa_ulysses_forward(rank, world_size, *, vsa_sparsity: float):
    """Forward pass: output shape correct and finite."""
    batch = 1
    seq_full = math.prod(_DIT_SEQ_SHAPE)
    assert seq_full % world_size == 0
    seq_per_rank = seq_full // world_size
    num_heads = world_size * _HEADS_PER_RANK

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    inner = _make_vsa_backend(num_heads // world_size, vsa_sparsity)
    attention = UlyssesAttention(inner_backend=inner, process_group=None)

    # Ulysses input: sequence-sharded, head-full [B, S/P, H, D].
    shape = (batch, seq_per_rank, num_heads, _HEAD_DIM)
    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)
    gate_compress = torch.randn(shape, device=device, dtype=torch.bfloat16)
    gate_fine = torch.randn(shape, device=device, dtype=torch.bfloat16)

    metadata = _build_full_seq_vsa_metadata(vsa_sparsity, device)
    with set_vsa_forward_context(metadata):
        output = attention(q, k, v, gate_compress=gate_compress, gate_fine=gate_fine)

    assert output.shape == (batch, seq_per_rank, num_heads, _HEAD_DIM), (
        f"Rank {rank}: expected {(batch, seq_per_rank, num_heads, _HEAD_DIM)}, got {output.shape}"
    )
    assert torch.isfinite(output).all(), f"Rank {rank}: Inf/NaN in output"


def _logic_vsa_ulysses_vs_reference(rank, world_size, *, vsa_sparsity: float):
    """Each rank's Ulysses+VSA output matches the single-GPU VSA reference's sequence slice."""
    batch = 1
    seq_full = math.prod(_DIT_SEQ_SHAPE)
    assert seq_full % world_size == 0
    seq_per_rank = seq_full // world_size
    num_heads = world_size * _HEADS_PER_RANK

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    full_shape = (batch, seq_full, num_heads, _HEAD_DIM)
    q_full = torch.randn(full_shape, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(full_shape, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(full_shape, device=device, dtype=torch.bfloat16)
    gate_c_full = torch.randn(full_shape, device=device, dtype=torch.bfloat16)
    gate_f_full = torch.randn(full_shape, device=device, dtype=torch.bfloat16)

    sl = slice(rank * seq_per_rank, (rank + 1) * seq_per_rank)
    q_shard = q_full[:, sl].contiguous()
    k_shard = k_full[:, sl].contiguous()
    v_shard = v_full[:, sl].contiguous()
    gate_c_shard = gate_c_full[:, sl].contiguous()
    gate_f_shard = gate_f_full[:, sl].contiguous()

    metadata = _build_full_seq_vsa_metadata(vsa_sparsity, device)

    # Ulysses path: sharded input, head-sharded inner backend.
    inner = _make_vsa_backend(num_heads // world_size, vsa_sparsity)
    attention = UlyssesAttention(inner_backend=inner, process_group=None)
    with set_vsa_forward_context(metadata):
        ulysses_out = attention(
            q_shard, k_shard, v_shard, gate_compress=gate_c_shard, gate_fine=gate_f_shard
        )

    # Single-GPU VSA reference over the full sequence.
    ref_attn = _make_vsa_backend(num_heads, vsa_sparsity)
    with set_vsa_forward_context(metadata):
        ref_out = ref_attn.forward(
            q_full, k_full, v_full, gate_compress=gate_c_full, gate_fine=gate_f_full
        )
    ref_shard = ref_out[:, sl]

    ulysses_out = ulysses_out.view(batch, seq_per_rank, num_heads, _HEAD_DIM).to(torch.bfloat16)
    ref_shard = ref_shard.to(torch.bfloat16)

    cos_sim = F.cosine_similarity(
        ulysses_out.reshape(-1).float(),
        ref_shard.reshape(-1).float(),
        dim=0,
    ).item()
    assert cos_sim > 0.990, f"Rank {rank}: cosine similarity {cos_sim:.6f} is below threshold 0.990"
    torch.testing.assert_close(ulysses_out, ref_shard, atol=2e-2, rtol=2e-2)


# =============================================================================
# Test class
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestWanVsaUlysses:
    """VSA (CUTEDSL) attention backend combined with Ulysses sequence parallelism."""

    @pytest.mark.parametrize("vsa_sparsity", [0.0, 0.5])
    def test_vsa_ulysses_forward(self, vsa_sparsity: float):
        run_test_in_distributed(
            world_size=2,
            test_fn=functools.partial(_logic_vsa_ulysses_forward, vsa_sparsity=vsa_sparsity),
        )

    @pytest.mark.parametrize("vsa_sparsity", [0.0, 0.5, 0.75])
    def test_vsa_ulysses_vs_reference(self, vsa_sparsity: float):
        run_test_in_distributed(
            world_size=2,
            test_fn=functools.partial(_logic_vsa_ulysses_vs_reference, vsa_sparsity=vsa_sparsity),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
