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

"""Multi-GPU tests for FLUX Tensor Parallelism (TP).

Tests that FLUX.1 and FLUX.2 transformers produce correct outputs when using
TP (sharding weights across GPUs), and combined TP + Ulysses.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_flux_tp.py -v
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
    import sys
    from pathlib import Path

    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TorchCompileConfig,
        create_attention_metadata_state,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.models.flux.joint_proj import (
        FluxJointAttnMLPProj,
        FluxJointQKVMLPProj,
    )

    # Reuse the CI-aware free-port allocator from tests/integration so that
    # sequentially spawned distributed workers get disjoint MASTER_PORTs and
    # don't collide with ports still in TIME_WAIT (EADDRINUSE).
    _integration_dir = Path(__file__).resolve().parents[4] / "integration"
    if str(_integration_dir) not in sys.path:
        sys.path.insert(0, str(_integration_dir))
    from defs.common import get_free_port_in_ci

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
# Distributed helpers (same pattern as test_flux_ulysses.py)
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
    port = get_free_port_in_ci()

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


def _make_model_config(pretrained_dict, tp_size=1, ulysses_size=1, backend="VANILLA"):
    """Create DiffusionModelConfig for testing with TP and/or Ulysses."""
    pretrained_config = SimpleNamespace(**pretrained_dict)
    ws = tp_size * ulysses_size
    if ws > 1 and dist.is_initialized():
        ws = dist.get_world_size()
        rk = dist.get_rank()
    else:
        rk = 0
    vgm = VisualGenMapping(world_size=ws, rank=rk, tp_size=tp_size, ulysses_size=ulysses_size)

    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        cache=None,
        attention_metadata_state=(
            create_attention_metadata_state() if backend.upper() == "TRTLLM" else None
        ),
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _stabilize_model_weights(model):
    """Reinitialize model weights for stable BF16 forward pass.

    Random default init (std~1.0) causes BF16 overflow through multiple
    transformer blocks. Use small uniform init that keeps activations bounded.
    """
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                fan_in = p.shape[1] if p.ndim >= 2 else p.shape[0]
                std = 0.02 / max(1.0, fan_in**0.5)
                p.data.uniform_(-std, std)
            else:
                p.data.uniform_(-0.01, 0.01)


# =============================================================================
# TP weight sharding helpers
# =============================================================================


def _shard_dim0(tensor, tp_rank, tp_size):
    """Shard a tensor along dim 0."""
    chunk = tensor.shape[0] // tp_size
    return tensor[tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def _shard_dim1(tensor, tp_rank, tp_size):
    """Shard a tensor along dim 1."""
    chunk = tensor.shape[1] // tp_size
    return tensor[:, tp_rank * chunk : (tp_rank + 1) * chunk].contiguous()


def _shard_fused_qkv(tensor, tp_rank, tp_size, q_dim, kv_dim):
    """Shard a fused QKV weight [q_dim + 2*kv_dim, ...] preserving Q/K/V structure."""
    q, k, v = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
    return torch.cat(
        [
            _shard_dim0(q, tp_rank, tp_size),
            _shard_dim0(k, tp_rank, tp_size),
            _shard_dim0(v, tp_rank, tp_size),
        ],
        dim=0,
    )


def _shard_fused_gate_up(tensor, tp_rank, tp_size):
    """Shard a fused gate_up weight [2*intermediate, ...] preserving gate/up structure."""
    half = tensor.shape[0] // 2
    gate, up = tensor.split([half, half], dim=0)
    return torch.cat(
        [
            _shard_dim0(gate, tp_rank, tp_size),
            _shard_dim0(up, tp_rank, tp_size),
        ],
        dim=0,
    )


def _copy_ref_weights_to_tp(ref_model, tp_model, tp_rank, tp_size):
    """Copy weights from a TP=1 reference model into a TP model with correct sharding.

    Handles column-parallel (QKV, MLP up/gate), row-parallel (output projs),
    fused QKV/gate_up weights, and wrapper projectors (FluxJointAttnMLPProj,
    FluxJointQKVMLPProj) that have different sub-module structure at TP>1.
    """
    ref_params = dict(ref_model.named_parameters())

    # First handle wrapper projectors whose sub-module names differ between TP=1 and TP>1.
    # At TP=1: single .proj Linear. At TP>1: split into sub-Linears.
    handled_tp_params = set()

    for tp_name, tp_module in tp_model.named_modules():
        if isinstance(tp_module, FluxJointAttnMLPProj) and tp_module.tp_size > 1:
            # TP model has .attn_proj + .mlp_proj; ref has .proj
            ref_w = ref_params[f"{tp_name}.proj.weight"]  # [out, attn_dim + mlp_dim]
            w_attn = ref_w[:, : tp_module.attn_dim]
            w_mlp = ref_w[:, tp_module.attn_dim :]
            tp_model.get_parameter(f"{tp_name}.attn_proj.weight").data.copy_(
                _shard_dim1(w_attn, tp_rank, tp_size)
            )
            tp_model.get_parameter(f"{tp_name}.mlp_proj.weight").data.copy_(
                _shard_dim1(w_mlp, tp_rank, tp_size)
            )
            handled_tp_params.update(
                [
                    f"{tp_name}.attn_proj.weight",
                    f"{tp_name}.mlp_proj.weight",
                ]
            )
            if tp_module.has_bias:
                ref_b = ref_params[f"{tp_name}.proj.bias"]
                tp_model.get_parameter(f"{tp_name}.bias").data.copy_(ref_b)
                handled_tp_params.add(f"{tp_name}.bias")

        elif isinstance(tp_module, FluxJointQKVMLPProj) and tp_module.tp_size > 1:
            # TP model has .qkv_proj + .mlp_proj; ref has .proj
            ref_w = ref_params[f"{tp_name}.proj.weight"]  # [qkv+mlp, hidden]
            w_qkv = ref_w[: tp_module.full_qkv_dim]
            w_mlp = ref_w[tp_module.full_qkv_dim :]

            # QKV: split into Q/K/V, shard each, re-fuse
            tp_model.get_parameter(f"{tp_name}.qkv_proj.weight").data.copy_(
                _shard_fused_qkv(
                    w_qkv, tp_rank, tp_size, tp_module.full_q_dim, tp_module.full_kv_dim
                )
            )
            # MLP: split gate/up, shard each, re-fuse
            tp_model.get_parameter(f"{tp_name}.mlp_proj.weight").data.copy_(
                _shard_fused_gate_up(w_mlp, tp_rank, tp_size)
            )
            handled_tp_params.update(
                [
                    f"{tp_name}.qkv_proj.weight",
                    f"{tp_name}.mlp_proj.weight",
                ]
            )
            # Handle bias if present
            if f"{tp_name}.proj.bias" in ref_params:
                ref_b = ref_params[f"{tp_name}.proj.bias"]
                b_qkv = ref_b[: tp_module.full_qkv_dim]
                b_mlp = ref_b[tp_module.full_qkv_dim :]
                tp_model.get_parameter(f"{tp_name}.qkv_proj.bias").data.copy_(
                    _shard_fused_qkv(
                        b_qkv, tp_rank, tp_size, tp_module.full_q_dim, tp_module.full_kv_dim
                    )
                )
                tp_model.get_parameter(f"{tp_name}.mlp_proj.bias").data.copy_(
                    _shard_fused_gate_up(b_mlp, tp_rank, tp_size)
                )
                handled_tp_params.update(
                    [
                        f"{tp_name}.qkv_proj.bias",
                        f"{tp_name}.mlp_proj.bias",
                    ]
                )

    # Now handle all remaining parameters by shape comparison.
    with torch.no_grad():
        for tp_name, tp_param in tp_model.named_parameters():
            if tp_name in handled_tp_params:
                continue
            if tp_name not in ref_params:
                continue

            ref_param = ref_params[tp_name]

            if tp_param.shape == ref_param.shape:
                # Replicated parameter (norms, embeddings, etc.)
                tp_param.data.copy_(ref_param.data)
            elif tp_param.ndim >= 2 and tp_param.shape[1] == ref_param.shape[1]:
                # Column parallel: dim 0 is smaller (output dim sharded)
                if "qkv_proj" in tp_name or "add_qkv_proj" in tp_name:
                    # Fused QKV: figure out q_dim from total (q=k=v for FLUX)
                    q_dim = ref_param.shape[0] // 3
                    tp_param.data.copy_(
                        _shard_fused_qkv(ref_param.data, tp_rank, tp_size, q_dim, q_dim)
                    )
                elif "gate_up_proj" in tp_name:
                    tp_param.data.copy_(_shard_fused_gate_up(ref_param.data, tp_rank, tp_size))
                else:
                    tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
            elif tp_param.ndim >= 2 and tp_param.shape[0] == ref_param.shape[0]:
                # Row parallel: dim 1 is smaller (input dim sharded)
                tp_param.data.copy_(_shard_dim1(ref_param.data, tp_rank, tp_size))
            elif tp_param.ndim == 1 and tp_param.shape[0] < ref_param.shape[0]:
                # 1D bias for column parallel
                if "qkv_proj" in tp_name or "add_qkv_proj" in tp_name:
                    q_dim = ref_param.shape[0] // 3
                    tp_param.data.copy_(
                        _shard_fused_qkv(ref_param.data, tp_rank, tp_size, q_dim, q_dim)
                    )
                elif "gate_up_proj" in tp_name:
                    tp_param.data.copy_(_shard_fused_gate_up(ref_param.data, tp_rank, tp_size))
                else:
                    tp_param.data.copy_(_shard_dim0(ref_param.data, tp_rank, tp_size))
            else:
                raise ValueError(
                    f"Cannot shard {tp_name}: ref={ref_param.shape}, tp={tp_param.shape}"
                )


# =============================================================================
# FLUX.1 test logic functions (module-level for pickling)
# =============================================================================


def _logic_flux1_tp_forward(rank, world_size):
    """FLUX.1 transformer forward with TP — verify shape and no NaN/Inf."""
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import FluxTransformer2DModel

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    model_config = _make_model_config(_FLUX1_TEST_CONFIG, tp_size=world_size)
    model = FluxTransformer2DModel(model_config).to(device).to(torch.bfloat16)
    _stabilize_model_weights(model)

    batch = 1
    img_seq = 16
    txt_seq = 8

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
    assert sample.shape == (batch, img_seq, 64), (
        f"Rank {rank}: Expected shape {(batch, img_seq, 64)}, got {sample.shape}"
    )
    assert not torch.isnan(sample).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(sample).any(), f"Rank {rank}: Inf in output"


def _logic_flux1_tp_vs_single_gpu(rank, world_size):
    """FLUX.1: TP 2-GPU output matches single-GPU reference."""
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux import FluxTransformer2DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16

    batch = 1
    img_seq = 16
    txt_seq = 8

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(_FLUX1_TEST_CONFIG, tp_size=1)
    ref_model = FluxTransformer2DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create TP model and copy sharded weights from ref
    torch.manual_seed(123)
    tp_config = _make_model_config(_FLUX1_TEST_CONFIG, tp_size=world_size)
    tp_model = FluxTransformer2DModel(tp_config).to(device).to(compute_dtype)
    _copy_ref_weights_to_tp(ref_model, tp_model, rank, world_size)

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
        tp_output = tp_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    torch.testing.assert_close(
        tp_output["sample"],
        ref_output["sample"],
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FLUX.1 TP output differs from single-GPU reference",
    )


# =============================================================================
# FLUX.2 test logic functions (module-level for pickling)
# =============================================================================


def _logic_flux2_tp_forward(rank, world_size):
    """FLUX.2 transformer forward with TP — verify shape and no NaN/Inf."""
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import Flux2Transformer2DModel

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    model_config = _make_model_config(_FLUX2_TEST_CONFIG, tp_size=world_size)
    model = Flux2Transformer2DModel(model_config).to(device).to(torch.bfloat16)
    _stabilize_model_weights(model)

    batch = 1
    img_seq = 16
    txt_seq = 8

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
    assert sample.shape == (batch, img_seq, 128), (
        f"Rank {rank}: Expected shape {(batch, img_seq, 128)}, got {sample.shape}"
    )
    assert not torch.isnan(sample).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(sample).any(), f"Rank {rank}: Inf in output"


def _logic_flux2_tp_vs_single_gpu(rank, world_size):
    """FLUX.2: TP 2-GPU output matches single-GPU reference."""
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import Flux2Transformer2DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16

    batch = 1
    img_seq = 16
    txt_seq = 8

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(_FLUX2_TEST_CONFIG, tp_size=1)
    ref_model = Flux2Transformer2DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create TP model and copy sharded weights from ref
    torch.manual_seed(123)
    tp_config = _make_model_config(_FLUX2_TEST_CONFIG, tp_size=world_size)
    tp_model = Flux2Transformer2DModel(tp_config).to(device).to(compute_dtype)
    _copy_ref_weights_to_tp(ref_model, tp_model, rank, world_size)

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
        tp_output = tp_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    torch.testing.assert_close(
        tp_output["sample"],
        ref_output["sample"],
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FLUX.2 TP output differs from single-GPU reference",
    )


# =============================================================================
# Combined TP + Ulysses test logic (module-level for pickling)
# =============================================================================


def _logic_flux2_tp_ulysses_vs_single_gpu(rank, world_size):
    """FLUX.2: TP=2 + Ulysses=2 (4 GPUs) output matches single-GPU reference.

    Each rank has a shard of both the weights (TP) and sequence (Ulysses).
    The output should match the single-GPU model after gathering.
    """
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import Flux2Transformer2DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16
    tp_size = 2
    ulysses_size = 2

    batch = 1
    img_seq = 16  # Must be divisible by ulysses_size
    txt_seq = 8

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(_FLUX2_TEST_CONFIG, tp_size=1, ulysses_size=1)
    ref_model = Flux2Transformer2DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create combined TP + Ulysses model and copy sharded weights
    torch.manual_seed(123)
    combined_config = _make_model_config(
        _FLUX2_TEST_CONFIG, tp_size=tp_size, ulysses_size=ulysses_size
    )
    combined_model = Flux2Transformer2DModel(combined_config).to(device).to(compute_dtype)
    vgm = combined_config.visual_gen_mapping
    _copy_ref_weights_to_tp(ref_model, combined_model, vgm.tp_rank, tp_size)

    # Same inputs on all ranks (Ulysses shards at runtime)
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
        combined_output = combined_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

    torch.testing.assert_close(
        combined_output["sample"],
        ref_output["sample"],
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: FLUX.2 TP+Ulysses output differs from single-GPU reference",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestFlux1TP:
    """Tensor parallelism tests for FLUX.1 transformer."""

    def test_flux1_tp_forward(self):
        """FLUX.1 TP forward: correct output shape and no NaN/Inf."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux1_tp_forward)

    def test_flux1_tp_vs_single_gpu(self):
        """FLUX.1 TP 2-GPU output matches single-GPU reference."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux1_tp_vs_single_gpu)


class TestFlux2TP:
    """Tensor parallelism tests for FLUX.2 transformer."""

    def test_flux2_tp_forward(self):
        """FLUX.2 TP forward: correct output shape and no NaN/Inf."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux2_tp_forward)

    def test_flux2_tp_vs_single_gpu(self):
        """FLUX.2 TP 2-GPU output matches single-GPU reference."""
        run_test_in_distributed(world_size=2, test_fn=_logic_flux2_tp_vs_single_gpu)


class TestFlux2TPUlyssesCombined:
    """Combined TP + Ulysses tests for FLUX.2 transformer."""

    def test_flux2_tp_ulysses_vs_single_gpu(self):
        """FLUX.2 TP=2 + Ulysses=2 (4 GPUs) matches single-GPU reference."""
        run_test_in_distributed(world_size=4, test_fn=_logic_flux2_tp_ulysses_vs_single_gpu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
