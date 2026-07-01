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

"""Multi-GPU tests for WAN Tensor Parallelism (TP).

Tests that WAN T2V and I2V transformers produce correct outputs when using
TP (sharding weights across GPUs), and combined TP + Ulysses.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_tp.py -v
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
        TorchCompileConfig,
        create_attention_metadata_state,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.models.modeling_utils import QuantConfig

    from .tp_shard_utils import copy_tp_parameter

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Clean up TLLM_DISABLE_MPI env var after tests complete."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_flux_tp.py)
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
    # Reset the DeviceMesh singleton before destroying the process group
    # to prevent NCCL process group destructor segfaults during interpreter exit.
    try:
        from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

        DeviceMeshTopologyImpl.device_mesh = None
        DeviceMeshTopologyImpl.tp_mesh = None
    except ImportError:
        pass
    if dist.is_initialized():
        dist.barrier()
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
# Model configs
# =============================================================================

# Small WAN T2V config (4 heads, 64 head_dim = 256 hidden)
_WAN_T2V_TEST_CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=64,
    num_layers=2,
    in_channels=16,
    out_channels=16,
    text_dim=128,
    freq_dim=64,
    patch_size=[1, 2, 2],
    ffn_dim=512,
    eps=1e-6,
    cross_attn_norm=True,
)

# Small WAN I2V config (same base + image embedding and added KV projections)
_WAN_I2V_TEST_CONFIG = dict(
    **_WAN_T2V_TEST_CONFIG,
    image_dim=64,  # image embedder: image_dim -> hidden_size
    added_kv_proj_dim=256,  # add_k_proj input dim = hidden_size (image embeds projected to hidden_size before blocks)
)

# Existing WAN configs are already uneven with TP=3:
# 4 attention heads split as 2+1+1, and ffn_dim=512 is not divisible by 3.
_WAN_UNEVEN_TP3_CONFIG = dict(_WAN_T2V_TEST_CONFIG)
_WAN_I2V_UNEVEN_TP3_CONFIG = dict(_WAN_I2V_TEST_CONFIG)


# =============================================================================
# Model config + weight helpers
# =============================================================================


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
# TP weight sharding helpers (see tp_shard_utils.py)
# =============================================================================


def _copy_ref_weights_to_tp(ref_model, tp_model, tp_rank, tp_size, config_dict):
    """Copy weights from a TP=1 reference model into a TP model with correct sharding."""
    ref_params = dict(ref_model.named_parameters())
    num_heads = config_dict["num_attention_heads"]
    head_dim = config_dict["attention_head_dim"]
    vgm = getattr(tp_model.model_config, "visual_gen_mapping", None)
    ulysses_size = vgm.ulysses_size if vgm is not None else 1

    with torch.no_grad():
        for tp_name, tp_param in tp_model.named_parameters():
            if tp_name not in ref_params:
                continue
            copy_tp_parameter(
                tp_name,
                ref_params[tp_name],
                tp_param,
                tp_rank,
                tp_size,
                num_heads,
                head_dim,
                ulysses_size=ulysses_size,
            )


# =============================================================================
# WAN T2V test logic functions (module-level for pickling)
# =============================================================================


def _logic_wan_t2v_tp_forward(rank, world_size):
    """WAN T2V transformer forward with TP — verify shape and no NaN/Inf."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    model_config = _make_model_config(_WAN_T2V_TEST_CONFIG, tp_size=world_size)
    model = WanTransformer3DModel(model_config).to(device).to(torch.bfloat16)
    _stabilize_model_weights(model)

    batch = 1
    T, H, W = 2, 4, 4  # Small spatial dims (must be divisible by patch_size [1,2,2])
    in_channels = 16
    txt_seq = 8

    torch.manual_seed(100)
    hidden_states = (
        torch.randn(batch, in_channels, T, H, W, device=device, dtype=torch.bfloat16) * 0.1
    )
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 128, device=device, dtype=torch.bfloat16) * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    assert output.shape == (batch, in_channels, T, H, W), (
        f"Rank {rank}: Expected shape {(batch, in_channels, T, H, W)}, got {output.shape}"
    )
    assert not torch.isnan(output).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(output).any(), f"Rank {rank}: Inf in output"


def _logic_wan_t2v_tp_vs_single_gpu(rank, world_size):
    """WAN T2V: TP 2-GPU output matches single-GPU reference."""
    _logic_wan_t2v_tp_vs_single_gpu_with_config(rank, world_size, _WAN_T2V_TEST_CONFIG)


def _logic_wan_t2v_tp3_uneven_vs_single_gpu(rank, world_size):
    """WAN T2V: TP=3 with uneven head/FFN dims matches single-GPU reference."""
    _logic_wan_t2v_tp_vs_single_gpu_with_config(rank, world_size, _WAN_UNEVEN_TP3_CONFIG)


def _logic_wan_t2v_tp_vs_single_gpu_with_config(rank, world_size, config_dict):
    """WAN T2V: TP output matches single-GPU reference."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16

    batch = 1
    T, H, W = 2, 4, 4
    in_channels = 16
    txt_seq = 8

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(config_dict, tp_size=1)
    ref_model = WanTransformer3DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create TP model and copy sharded weights from ref
    torch.manual_seed(123)
    tp_config = _make_model_config(config_dict, tp_size=world_size)
    tp_model = WanTransformer3DModel(tp_config).to(device).to(compute_dtype)
    _copy_ref_weights_to_tp(ref_model, tp_model, rank, world_size, config_dict)

    # Same inputs on all ranks
    torch.manual_seed(456)
    hidden_states = (
        torch.randn(batch, in_channels, T, H, W, device=device, dtype=compute_dtype) * 0.1
    )
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 128, device=device, dtype=compute_dtype) * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=compute_dtype)

    with torch.no_grad():
        ref_output = ref_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        tp_output = tp_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    torch.testing.assert_close(
        tp_output,
        ref_output,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: WAN T2V TP output differs from single-GPU reference",
    )


def _logic_wan_t2v_tp_ulysses_vs_single_gpu(rank, world_size):
    """WAN T2V: TP=2 + Ulysses=2 (4 GPUs) output matches single-GPU reference."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16
    tp_size = 2
    ulysses_size = 2

    batch = 1
    T, H, W = 2, 4, 4  # seq_len = T * (H/2) * (W/2) = 2*2*2 = 8, divisible by ulysses_size=2
    in_channels = 16
    txt_seq = 8

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(_WAN_T2V_TEST_CONFIG, tp_size=1, ulysses_size=1)
    ref_model = WanTransformer3DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create combined TP + Ulysses model and copy sharded weights
    torch.manual_seed(123)
    combined_config = _make_model_config(
        _WAN_T2V_TEST_CONFIG, tp_size=tp_size, ulysses_size=ulysses_size
    )
    combined_model = WanTransformer3DModel(combined_config).to(device).to(compute_dtype)
    vgm = combined_config.visual_gen_mapping
    _copy_ref_weights_to_tp(ref_model, combined_model, vgm.tp_rank, tp_size, _WAN_T2V_TEST_CONFIG)

    # Same inputs on all ranks (Ulysses shards at runtime)
    torch.manual_seed(456)
    hidden_states = (
        torch.randn(batch, in_channels, T, H, W, device=device, dtype=compute_dtype) * 0.1
    )
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 128, device=device, dtype=compute_dtype) * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=compute_dtype)

    with torch.no_grad():
        ref_output = ref_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        combined_output = combined_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    torch.testing.assert_close(
        combined_output,
        ref_output,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: WAN T2V TP+Ulysses output differs from single-GPU reference",
    )


# =============================================================================
# WAN I2V test logic functions (module-level for pickling)
# =============================================================================


def _logic_wan_i2v_tp_forward(rank, world_size):
    """WAN I2V transformer forward with TP — verify shape and no NaN/Inf."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)

    model_config = _make_model_config(_WAN_I2V_TEST_CONFIG, tp_size=world_size)
    model = WanTransformer3DModel(model_config).to(device).to(torch.bfloat16)
    _stabilize_model_weights(model)

    batch = 1
    T, H, W = 2, 4, 4
    in_channels = 16
    txt_seq = 8
    img_seq = 4

    torch.manual_seed(100)
    hidden_states = (
        torch.randn(batch, in_channels, T, H, W, device=device, dtype=torch.bfloat16) * 0.1
    )
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 128, device=device, dtype=torch.bfloat16) * 0.1
    )
    encoder_hidden_states_image = (
        torch.randn(batch, img_seq, 64, device=device, dtype=torch.bfloat16) * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

    assert output.shape == (batch, in_channels, T, H, W), (
        f"Rank {rank}: Expected shape {(batch, in_channels, T, H, W)}, got {output.shape}"
    )
    assert not torch.isnan(output).any(), f"Rank {rank}: NaN in output"
    assert not torch.isinf(output).any(), f"Rank {rank}: Inf in output"


def _logic_wan_i2v_tp_vs_single_gpu(rank, world_size):
    """WAN I2V: TP 2-GPU output matches single-GPU reference."""
    _logic_wan_i2v_tp_vs_single_gpu_with_config(rank, world_size, _WAN_I2V_TEST_CONFIG)


def _logic_wan_i2v_tp3_uneven_vs_single_gpu(rank, world_size):
    """WAN I2V: TP=3 with uneven head/FFN dims matches single-GPU reference."""
    _logic_wan_i2v_tp_vs_single_gpu_with_config(rank, world_size, _WAN_I2V_UNEVEN_TP3_CONFIG)


def _logic_wan_i2v_tp_vs_single_gpu_with_config(rank, world_size, config_dict):
    """WAN I2V: TP output matches single-GPU reference."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    compute_dtype = torch.bfloat16

    batch = 1
    T, H, W = 2, 4, 4
    in_channels = 16
    txt_seq = 8
    img_seq = 4

    # Create single-GPU reference model
    torch.manual_seed(123)
    ref_config = _make_model_config(config_dict, tp_size=1)
    ref_model = WanTransformer3DModel(ref_config).to(device).to(compute_dtype)
    _stabilize_model_weights(ref_model)

    # Create TP model and copy sharded weights from ref
    torch.manual_seed(123)
    tp_config = _make_model_config(config_dict, tp_size=world_size)
    tp_model = WanTransformer3DModel(tp_config).to(device).to(compute_dtype)
    _copy_ref_weights_to_tp(ref_model, tp_model, rank, world_size, config_dict)

    # Same inputs on all ranks
    torch.manual_seed(456)
    hidden_states = (
        torch.randn(batch, in_channels, T, H, W, device=device, dtype=compute_dtype) * 0.1
    )
    encoder_hidden_states = (
        torch.randn(batch, txt_seq, 128, device=device, dtype=compute_dtype) * 0.1
    )
    encoder_hidden_states_image = (
        torch.randn(batch, img_seq, config_dict["image_dim"], device=device, dtype=compute_dtype)
        * 0.1
    )
    timestep = torch.tensor([0.5], device=device, dtype=compute_dtype)

    with torch.no_grad():
        ref_output = ref_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )
        tp_output = tp_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

    torch.testing.assert_close(
        tp_output,
        ref_output,
        rtol=1e-2,
        atol=1e-2,
        msg=f"Rank {rank}: WAN I2V TP output differs from single-GPU reference",
    )


# =============================================================================
# Test classes
# =============================================================================


class TestWanT2VTP:
    """Tensor parallelism tests for WAN T2V transformer."""

    def test_wan_t2v_tp_forward(self):
        """WAN T2V TP forward: correct output shape and no NaN/Inf."""
        run_test_in_distributed(world_size=2, test_fn=_logic_wan_t2v_tp_forward)

    def test_wan_t2v_tp_vs_single_gpu(self):
        """WAN T2V TP 2-GPU output matches single-GPU reference."""
        run_test_in_distributed(world_size=2, test_fn=_logic_wan_t2v_tp_vs_single_gpu)


class TestWanT2VTPUlyssesCombined:
    """Combined TP + Ulysses tests for WAN T2V transformer."""

    def test_wan_t2v_tp_ulysses_vs_single_gpu(self):
        """WAN T2V TP=2 + Ulysses=2 (4 GPUs) matches single-GPU reference."""
        run_test_in_distributed(world_size=4, test_fn=_logic_wan_t2v_tp_ulysses_vs_single_gpu)


class TestWanI2VTP:
    """Tensor parallelism tests for WAN I2V transformer."""

    def test_wan_i2v_tp_forward(self):
        """WAN I2V TP forward: correct output shape and no NaN/Inf."""
        run_test_in_distributed(world_size=2, test_fn=_logic_wan_i2v_tp_forward)

    def test_wan_i2v_tp_vs_single_gpu(self):
        """WAN I2V TP 2-GPU output matches single-GPU reference."""
        run_test_in_distributed(world_size=2, test_fn=_logic_wan_i2v_tp_vs_single_gpu)


class TestWanUnevenTP3:
    """TP=3 tests where head count and FFN dims are not divisible by tp_size."""

    def test_wan_t2v_tp3_uneven_vs_single_gpu(self):
        """WAN T2V TP=3 (4 heads, uneven FFN) matches single-GPU reference."""
        run_test_in_distributed(world_size=3, test_fn=_logic_wan_t2v_tp3_uneven_vs_single_gpu)

    def test_wan_i2v_tp3_uneven_vs_single_gpu(self):
        """WAN I2V TP=3 (4 heads, uneven FFN) matches single-GPU reference."""
        run_test_in_distributed(world_size=3, test_fn=_logic_wan_i2v_tp3_uneven_vs_single_gpu)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
