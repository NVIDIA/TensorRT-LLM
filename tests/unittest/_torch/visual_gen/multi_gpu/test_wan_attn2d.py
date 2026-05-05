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
"""Multi-GPU tests for WAN Attention2D context parallelism.

Tests that WanTransformer3DModel produces correct outputs when using
Attention2D context parallelism (2D mesh for sequence sharding across GPUs).

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_attn2d.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TeaCacheConfig,
        TorchCompileConfig,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.models.modeling_utils import QuantConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

try:
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
        _flash_attn_fwd as _fa4_fwd,
    )

    _flash_attn4_available = _fa4_fwd is not None
except (ImportError, OSError):
    _flash_attn4_available = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Clean up TLLM_DISABLE_MPI env var after tests complete."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_flux_ulysses.py)
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
    """Worker function run in each spawned process. Module-level for pickling."""
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()
    mp.spawn(
        _distributed_worker, args=(world_size, backend, test_fn, port), nprocs=world_size, join=True
    )


# =============================================================================
# Model config helpers
# =============================================================================

# Small WAN config for testing.
# hidden_size = num_attention_heads * attention_head_dim = 4 * 64 = 256.
# Attention2D shards sequence (not heads), so no head-count divisibility is required.
_WAN_TEST_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=64,
    num_layers=2,
    in_channels=4,
    out_channels=4,
    patch_size=[1, 2, 2],
    text_dim=64,
    freq_dim=32,
)

# Video input: [B=1, C=4, T=2, H=4, W=4]
# After patchify with patch_size=[1,2,2]: seq_len = T * (H/2) * (W/2) = 2*2*2 = 8
# With attn2d_size=4 (2x2 mesh): chunk_size = 8/4 = 2 tokens per rank ✓
_VIDEO_SHAPE = (1, 4, 2, 4, 4)
_TEXT_SEQ = 4


def _make_model_config(pretrained_dict, attn2d_row_size=1, attn2d_col_size=1, backend="FA4"):
    """Create DiffusionModelConfig for testing.

    With default attn2d_row_size=attn2d_col_size=1, creates a single-GPU config.
    With attn2d_row_size/col_size > 1, creates an Attention2D distributed config
    using the current process rank and world size.
    """
    pretrained_config = SimpleNamespace(**pretrained_dict)
    mesh_size = attn2d_row_size * attn2d_col_size
    if mesh_size > 1 and dist.is_initialized():
        ws = dist.get_world_size()
        rk = dist.get_rank()
    else:
        ws = 1
        rk = 0
    vgm = VisualGenMapping(
        world_size=ws,
        rank=rk,
        attn2d_row_size=attn2d_row_size,
        attn2d_col_size=attn2d_col_size,
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        teacache=TeaCacheConfig(),
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _stabilize_model_weights(model):
    """Reinitialize weights to prevent BF16 overflow through multiple transformer blocks."""
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 2:
                fan_in = p.shape[1]
                std = 0.02 / max(1.0, fan_in**0.5)
                p.data.uniform_(-std, std)
            else:
                p.data.uniform_(-0.01, 0.01)


# =============================================================================
# Test logic (module-level for mp.spawn pickling)
# =============================================================================


def _logic_wan_attn2d_forward(rank, world_size):
    """WAN forward with Attention2D (2x2 mesh): verify output shape and no NaN/Inf."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    torch.manual_seed(42)
    config = _make_model_config(_WAN_TEST_CONFIG, attn2d_row_size=2, attn2d_col_size=2)
    try:
        model = WanTransformer3DModel(config).to(device).to(dtype)
    except ImportError as e:
        pytest.skip(f"Attention2D JIT kernels not available: {e}")
    _stabilize_model_weights(model)

    B, C, T, H, W = _VIDEO_SHAPE
    text_dim = _WAN_TEST_CONFIG["text_dim"]

    torch.manual_seed(100)
    hidden_states = torch.randn(_VIDEO_SHAPE, device=device, dtype=dtype) * 0.1
    encoder_hidden_states = torch.randn(B, _TEXT_SEQ, text_dim, device=device, dtype=dtype) * 0.1
    timestep = torch.tensor([0.5], device=device, dtype=dtype)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    assert output.shape == (B, C, T, H, W), (
        f"Rank {rank}: expected shape {(B, C, T, H, W)}, got {output.shape}"
    )
    assert not torch.isnan(output).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(output).any(), f"Rank {rank}: Inf in output"


def _logic_wan_attn2d_vs_single_gpu(rank, world_size):
    """WAN Attention2D (2x2 mesh) output matches single-GPU FA4 reference.

    Both models use FA4 as the inner attention backend, so the only difference
    is the Attention2D gather + LSE-combine step vs a single full-sequence pass.
    The WAN forward already gathers the output across ranks (all_gather at the
    end of forward), so each rank receives the same full-sequence result and
    can be compared directly to the single-GPU reference.
    """
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    B, C, T, H, W = _VIDEO_SHAPE
    text_dim = _WAN_TEST_CONFIG["text_dim"]

    # Single-GPU reference (each rank computes independently — no distributed ops).
    torch.manual_seed(42)
    ref_config = _make_model_config(_WAN_TEST_CONFIG)
    ref_model = WanTransformer3DModel(ref_config).to(device).to(dtype)
    _stabilize_model_weights(ref_model)
    ref_state = ref_model.state_dict()

    # Attention2D model with identical weights.
    torch.manual_seed(42)
    attn2d_config = _make_model_config(_WAN_TEST_CONFIG, attn2d_row_size=2, attn2d_col_size=2)
    try:
        attn2d_model = WanTransformer3DModel(attn2d_config).to(device).to(dtype)
    except ImportError as e:
        pytest.skip(f"Attention2D JIT kernels not available: {e}")
    attn2d_model.load_state_dict(ref_state)

    torch.manual_seed(100)
    hidden_states = torch.randn(_VIDEO_SHAPE, device=device, dtype=dtype) * 0.1
    encoder_hidden_states = torch.randn(B, _TEXT_SEQ, text_dim, device=device, dtype=dtype) * 0.1
    timestep = torch.tensor([0.5], device=device, dtype=dtype)

    with torch.no_grad():
        ref_output = ref_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        attn2d_output = attn2d_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    torch.testing.assert_close(
        attn2d_output,
        ref_output,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: WAN Attention2D output differs from single-GPU FA4 reference",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestWanAttn2D:
    """Attention2D context parallelism tests for WAN transformer (2x2 mesh, 4 GPUs)."""

    def test_wan_attn2d_forward(self):
        """WAN Attention2D forward: correct output shape and no NaN/Inf."""
        if not _flash_attn4_available:
            pytest.skip("FlashAttn4 JIT kernels not available")
        run_test_in_distributed(world_size=4, test_fn=_logic_wan_attn2d_forward)

    def test_wan_attn2d_vs_single_gpu(self):
        """WAN Attention2D (2x2 mesh) output matches single-GPU FA4 reference."""
        if not _flash_attn4_available:
            pytest.skip("FlashAttn4 JIT kernels not available")
        run_test_in_distributed(world_size=4, test_fn=_logic_wan_attn2d_vs_single_gpu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
