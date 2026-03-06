# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-GPU tests for FLUX Ulysses sequence parallelism.

Tests that FLUX.1 and FLUX.2 transformers produce correct outputs when using
Ulysses sequence parallelism (sharding sequence across GPUs).

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_flux_ulysses.py -v
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
        ParallelConfig,
        TeaCacheConfig,
        TorchCompileConfig,
    )
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.models.modeling_utils import QuantConfig

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Clean up TLLM_DISABLE_MPI env var after tests complete."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_ulysses_attention.py)
# =============================================================================


def init_distributed_worker(rank: int, world_size: int, backend: str = "nccl", port: int = 29500):
    """Initialize distributed environment for a worker process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port):
    """Worker function that runs in each process. Module-level for pickling."""
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True):
    """Run a test function in a distributed environment."""
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

# Small FLUX.1 config for testing (reduced layers, 8 heads, 64 head_dim = 512 inner_dim)
_FLUX1_TEST_CONFIG = dict(
    num_attention_heads=8,
    attention_head_dim=64,
    in_channels=64,
    out_channels=64,
    num_layers=2,
    num_single_layers=4,
    joint_attention_dim=256,
    pooled_projection_dim=128,
    guidance_embeds=False,
    patch_size=1,
    axes_dims_rope=[16, 24, 24],
    theta=10000,
)

# Small FLUX.2 config for testing
_FLUX2_TEST_CONFIG = dict(
    num_attention_heads=8,
    attention_head_dim=64,
    in_channels=128,
    out_channels=128,
    num_layers=2,
    num_single_layers=4,
    joint_attention_dim=256,
    pooled_projection_dim=128,
    guidance_embeds=False,
    patch_size=1,
    mlp_ratio=3.0,
    axes_dims_rope=[16, 16, 16, 16],
    rope_theta=2000.0,
    eps=1e-6,
    timestep_guidance_channels=256,
)


def _make_model_config(pretrained_dict, ulysses_size=1):
    """Create DiffusionModelConfig for testing."""
    pretrained_config = SimpleNamespace(**pretrained_dict)
    parallel = ParallelConfig(dit_ulysses_size=ulysses_size)

    return DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
        attention=AttentionConfig(backend="VANILLA"),
        parallel=parallel,
        teacache=TeaCacheConfig(),
        skip_create_weights_in_init=False,
    )


def _stabilize_model_weights(model):
    """Reinitialize model weights for stable BF16 forward pass.

    Random default init (std~1.0) causes BF16 overflow through multiple
    transformer blocks. Use small uniform init that keeps activations bounded.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                # Xavier-like: scale by 1/sqrt(fan_in)
                fan_in = p.shape[1] if p.ndim >= 2 else p.shape[0]
                std = 0.02 / max(1.0, fan_in**0.5)
                p.data.uniform_(-std, std)
            else:
                # Bias/1D params: small values
                p.data.uniform_(-0.01, 0.01)


# =============================================================================
# FLUX.1 test logic functions (module-level for pickling)
# =============================================================================


def _logic_flux1_ulysses_forward(rank, world_size):
    """FLUX.1 transformer forward with Ulysses — verify shape and no NaN/Inf.

    Uses BF16 (required by flashinfer RMSNorm) with scaled-down weights to
    prevent overflow through multiple transformer blocks.
    """
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import FluxTransformer2DModel

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)  # Same seed on all ranks for identical model weights

    model_config = _make_model_config(_FLUX1_TEST_CONFIG, ulysses_size=world_size)
    model = FluxTransformer2DModel(model_config).to(device).to(torch.bfloat16)
    _stabilize_model_weights(model)

    batch = 1
    img_seq = 16  # Must be divisible by world_size
    txt_seq = 8  # Must be divisible by world_size

    # Same inputs on all ranks (required for Ulysses — each rank shards the same input)
    torch.manual_seed(100)
    hidden_states = torch.randn(batch, img_seq, 64, device=device, dtype=torch.bfloat16) * 0.1
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 256, device=device, dtype=torch.bfloat16) * 0.1
    )
    pooled_projections = torch.randn(batch, 128, device=device, dtype=torch.bfloat16) * 0.1
    timestep = torch.tensor([0.5], device=device, dtype=torch.bfloat16)
    img_ids = torch.randn(img_seq, 3, device=device)
    txt_ids = torch.randn(txt_seq, 3, device=device)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    sample = output["sample"]
    # Output should have full image sequence (gathered)
    assert sample.shape == (batch, img_seq, 64), (
        f"Rank {rank}: Expected shape {(batch, img_seq, 64)}, got {sample.shape}"
    )
    assert not torch.isnan(sample).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(sample).any(), f"Rank {rank}: Inf in output"


def _logic_flux1_ulysses_vs_single_gpu(rank, world_size):
    """FLUX.1: Ulysses 2-GPU output matches single-GPU reference.

    Uses BF16 (required by flashinfer RMSNorm) with scaled-down weights.
    Ulysses all-to-all is a pure data shuffle so results should be nearly
    identical — the only drift is from BF16 rounding in attention.
    """
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import FluxTransformer2DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16

    batch = 1
    img_seq = 16
    txt_seq = 8

    # Create single-GPU reference model with shared seed
    torch.manual_seed(123)
    ref_config = _make_model_config(_FLUX1_TEST_CONFIG, ulysses_size=1)
    ref_model = FluxTransformer2DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)
    ref_state = ref_model.state_dict()

    # Create Ulysses model with same weights
    torch.manual_seed(123)
    ulysses_config = _make_model_config(_FLUX1_TEST_CONFIG, ulysses_size=world_size)
    ulysses_model = FluxTransformer2DModel(ulysses_config).to(device).to(compute_dtype)
    ulysses_model.load_state_dict(ref_state)

    # Same inputs on all ranks
    torch.manual_seed(456)
    hidden_states = torch.randn(batch, img_seq, 64, device=device, dtype=compute_dtype) * 0.1
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 256, device=device, dtype=compute_dtype) * 0.1
    )
    pooled_projections = torch.randn(batch, 128, device=device, dtype=compute_dtype) * 0.1
    timestep = torch.tensor([0.5], device=device, dtype=compute_dtype)
    img_ids = torch.randn(img_seq, 3, device=device)
    txt_ids = torch.randn(txt_seq, 3, device=device)

    with torch.no_grad():
        ref_output = ref_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )
        ulysses_output = ulysses_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    torch.testing.assert_close(
        ulysses_output["sample"],
        ref_output["sample"],
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FLUX.1 Ulysses output differs from single-GPU reference",
    )


# =============================================================================
# FLUX.2 test logic functions (module-level for pickling)
# =============================================================================


def _logic_flux2_ulysses_forward(rank, world_size):
    """FLUX.2 transformer forward with Ulysses — verify shape and no NaN/Inf.

    Uses BF16 (required by flashinfer RMSNorm) with scaled-down weights.
    """
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import Flux2Transformer2DModel

    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)  # Same seed on all ranks for identical model weights

    model_config = _make_model_config(_FLUX2_TEST_CONFIG, ulysses_size=world_size)
    model = Flux2Transformer2DModel(model_config).to(device).to(torch.bfloat16)
    _stabilize_model_weights(model)

    batch = 1
    img_seq = 16
    txt_seq = 8

    # Same inputs on all ranks
    torch.manual_seed(100)
    hidden_states = torch.randn(batch, img_seq, 128, device=device, dtype=torch.bfloat16) * 0.1
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 256, device=device, dtype=torch.bfloat16) * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=torch.bfloat16)
    img_ids = torch.randn(img_seq, 4, device=device)
    txt_ids = torch.randn(txt_seq, 4, device=device)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    sample = output["sample"]
    # Output should have full image sequence (gathered), out_channels=128
    assert sample.shape == (batch, img_seq, 128), (
        f"Rank {rank}: Expected shape {(batch, img_seq, 128)}, got {sample.shape}"
    )
    assert not torch.isnan(sample).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(sample).any(), f"Rank {rank}: Inf in output"


def _logic_flux2_ulysses_vs_single_gpu(rank, world_size):
    """FLUX.2: Ulysses 2-GPU output matches single-GPU reference.

    Uses BF16 (required by flashinfer RMSNorm) with scaled-down weights.
    """
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import Flux2Transformer2DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16

    batch = 1
    img_seq = 16
    txt_seq = 8

    # Create single-GPU reference model with shared seed
    torch.manual_seed(123)
    ref_config = _make_model_config(_FLUX2_TEST_CONFIG, ulysses_size=1)
    ref_model = Flux2Transformer2DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)
    ref_state = ref_model.state_dict()

    # Create Ulysses model with same weights
    torch.manual_seed(123)
    ulysses_config = _make_model_config(_FLUX2_TEST_CONFIG, ulysses_size=world_size)
    ulysses_model = Flux2Transformer2DModel(ulysses_config).to(device).to(compute_dtype)
    ulysses_model.load_state_dict(ref_state)

    # Same inputs on all ranks
    torch.manual_seed(456)
    hidden_states = torch.randn(batch, img_seq, 128, device=device, dtype=compute_dtype) * 0.1
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 256, device=device, dtype=compute_dtype) * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=compute_dtype)
    img_ids = torch.randn(img_seq, 4, device=device)
    txt_ids = torch.randn(txt_seq, 4, device=device)

    with torch.no_grad():
        ref_output = ref_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )
        ulysses_output = ulysses_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    torch.testing.assert_close(
        ulysses_output["sample"],
        ref_output["sample"],
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FLUX.2 Ulysses output differs from single-GPU reference",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestFlux1Ulysses:
    """Ulysses sequence parallelism tests for FLUX.1 transformer."""

    def test_flux1_ulysses_forward(self):
        """FLUX.1 Ulysses forward: correct output shape and no NaN/Inf."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux1_ulysses_forward)

    def test_flux1_ulysses_vs_single_gpu(self):
        """FLUX.1 Ulysses 2-GPU output matches single-GPU reference."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux1_ulysses_vs_single_gpu)


class TestFlux2Ulysses:
    """Ulysses sequence parallelism tests for FLUX.2 transformer."""

    def test_flux2_ulysses_forward(self):
        """FLUX.2 Ulysses forward: correct output shape and no NaN/Inf."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux2_ulysses_forward)

    def test_flux2_ulysses_vs_single_gpu(self):
        """FLUX.2 Ulysses 2-GPU output matches single-GPU reference."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux2_ulysses_vs_single_gpu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
