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
"""Transformer-only WAN parallel correctness harness.

Compares distributed WanTransformer3DModel forward passes against a single-GPU
reference using the same input and randomly-initialized (stabilized) weights —
no real checkpoint is loaded. Also includes a forward-only distributed sanity
test.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_transformer_parallel.py -v -s
"""

import gc
import os
from types import SimpleNamespace
from typing import Callable

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TorchCompileConfig,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

try:
    from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (
        _flash_attn_fwd as _fa4_fwd,
    )
    from tensorrt_llm._torch.visual_gen.attention_backend.parallel import (
        _flash_attn_combine as _fa_combine,
    )

    _flash_attn4_available = _fa4_fwd is not None
    _attn2d_available = _fa4_fwd is not None and _fa_combine is not None
except (ImportError, OSError):
    _flash_attn4_available = False
    _attn2d_available = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers
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


def _distributed_worker(rank, world_size, backend, test_fn, port, kwargs):
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size, **kwargs)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True, **kwargs):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    backend = "nccl" if use_cuda else "gloo"
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, backend, test_fn, port, kwargs),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# WAN transformer test config / data
# =============================================================================


# Small in-code WAN transformer config (reduced layers/dims) so tests run
# without loading a real checkpoint. head_dim=128 keeps the FA4 backend happy;
# num_attention_heads=8 is divisible by every ulysses_size exercised below.
_WAN_TEST_CONFIG = dict(
    num_attention_heads=8,
    attention_head_dim=128,
    num_layers=2,
    in_channels=16,
    out_channels=16,
    text_dim=256,
    freq_dim=256,
    ffn_dim=512,
    patch_size=[1, 2, 2],
    eps=1e-6,
    cross_attn_norm=True,
)

# Latent video dims chosen so the patchified sequence length
# (T/p_t * H/p_h * W/p_w = 4 * 8 * 8 = 256) is divisible by every world_size.
_VIDEO_T = 4
_VIDEO_H = 16
_VIDEO_W = 16
_TEXT_SEQ = 32
_TIMESTEP = 0.5

SEED_WEIGHTS = 42
SEED_INPUT = 100

ATOL = 1e-2
RTOL = 1e-3

# All valid 8-GPU combinations of (ulysses, ring, attn2d):
# world_size = (ring or attn2d_row*attn2d_col or 1) * ulysses = 8
_WAN_8GPU_PARALLEL_COMBINATIONS = [
    # Ulysses-only family (no ring / no attn2d)
    ("ulysses_only_ul8", dict(dit_ulysses_size=8)),
    # Ring/Ulysses family
    ("ring4_ul2", dict(dit_ring_size=4, dit_ulysses_size=2)),
    # Attention2D/Ulysses family
    ("attn2d_1x4_ul2", dict(dit_attn2d_row_size=1, dit_attn2d_col_size=4, dit_ulysses_size=2)),
    # Attention2D/Ulysses family
    ("attn2d_1x8_ul1", dict(dit_attn2d_row_size=1, dit_attn2d_col_size=8, dit_ulysses_size=1)),
    ("attn2d_2x4_ul1", dict(dit_attn2d_row_size=2, dit_attn2d_col_size=4, dit_ulysses_size=1)),
]


def _stabilize_model_weights(model):
    """Reinitialize model weights for a stable BF16 forward pass.

    Random default init (std~1.0) overflows BF16 through multiple transformer
    blocks. Use a small uniform init that keeps activations bounded so the
    distributed-vs-single-GPU comparison is meaningful.
    """
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim >= 2:
                fan_in = p.shape[1]
                std = 0.02 / max(1.0, fan_in**0.5)
                p.data.uniform_(-std, std)
            else:
                p.data.uniform_(-0.01, 0.01)


def _make_model_config(
    pretrained_dict,
    *,
    cfg_size=1,
    ulysses_size=1,
    ring_size=1,
    attn2d_row_size=1,
    attn2d_col_size=1,
    backend="FA4",
    **parallel_kwargs,
):
    # Accept both shorthand names (cfg_size, ...) and VisualGen-style names
    # (dit_cfg_size, ...), since tests pass the latter.
    cfg_size = parallel_kwargs.pop("dit_cfg_size", cfg_size)
    ulysses_size = parallel_kwargs.pop("dit_ulysses_size", ulysses_size)
    ring_size = parallel_kwargs.pop("dit_ring_size", ring_size)
    attn2d_row_size = parallel_kwargs.pop("dit_attn2d_row_size", attn2d_row_size)
    attn2d_col_size = parallel_kwargs.pop("dit_attn2d_col_size", attn2d_col_size)
    if parallel_kwargs:
        raise TypeError(f"Unexpected parallel config args: {sorted(parallel_kwargs.keys())}")

    pretrained_config = SimpleNamespace(**pretrained_dict)
    use_dist = (
        cfg_size > 1 or ulysses_size > 1 or ring_size > 1 or attn2d_row_size * attn2d_col_size > 1
    ) and dist.is_initialized()
    if use_dist:
        ws = dist.get_world_size()
        rk = dist.get_rank()
    else:
        ws = 1
        rk = 0
    vgm = VisualGenMapping(
        world_size=ws,
        rank=rk,
        cfg_size=cfg_size,
        ulysses_size=ulysses_size,
        ring_size=ring_size,
        attn2d_row_size=attn2d_row_size,
        attn2d_col_size=attn2d_col_size,
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Core logic
# =============================================================================


def _logic_wan_transformer_parallel_vs_single_gpu(
    rank: int,
    world_size: int,
    *,
    parallel_cfg_kwargs: dict,
    label: str,
):
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    pretrained_cfg = _WAN_TEST_CONFIG
    B, C, T, H, W = (1, int(pretrained_cfg["in_channels"]), _VIDEO_T, _VIDEO_H, _VIDEO_W)
    text_dim = int(pretrained_cfg["text_dim"])

    torch.manual_seed(SEED_WEIGHTS)
    ref_config = _make_model_config(pretrained_cfg, backend="FA4")
    ref_model = WanTransformer3DModel(ref_config).to(device).to(dtype)
    _stabilize_model_weights(ref_model)
    ref_state = ref_model.state_dict()

    torch.manual_seed(SEED_WEIGHTS)
    dist_config = _make_model_config(pretrained_cfg, backend="FA4", **parallel_cfg_kwargs)
    try:
        dist_model = WanTransformer3DModel(dist_config).to(device).to(dtype)
    except (ImportError, ValueError, NotImplementedError) as e:
        pytest.skip(f"[{label}] Parallel backend unavailable: {e}")

    dist_model.load_state_dict(ref_state)

    torch.manual_seed(SEED_INPUT)
    hidden_states = torch.randn((B, C, T, H, W), device=device, dtype=dtype) * 0.1
    encoder_hidden_states = torch.randn(B, _TEXT_SEQ, text_dim, device=device, dtype=dtype) * 0.1
    timestep = torch.tensor([_TIMESTEP], device=device, dtype=dtype)

    with torch.no_grad():
        ref_output = ref_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        dist_output = dist_model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    abs_diff = (dist_output.float() - ref_output.float()).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    if rank == 0:
        print(
            f"[{label}] comparison stats: "
            f"max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}"
        )

    torch.testing.assert_close(
        dist_output,
        ref_output,
        rtol=RTOL,
        atol=ATOL,
        msg=f"Rank {rank}: [{label}] WanTransformer3DModel output differs from single-GPU FA4 reference",
    )

    _free(ref_model, dist_model, ref_output, dist_output)


def _logic_wan_transformer_parallel_forward_sanity(
    rank: int,
    world_size: int,
    *,
    parallel_cfg_kwargs: dict,
    label: str,
):
    """Distributed forward sanity: shape + finite checks only."""
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    pretrained_cfg = _WAN_TEST_CONFIG
    B, C, T, H, W = (1, int(pretrained_cfg["in_channels"]), _VIDEO_T, _VIDEO_H, _VIDEO_W)
    text_dim = int(pretrained_cfg["text_dim"])

    torch.manual_seed(SEED_WEIGHTS)
    config = _make_model_config(pretrained_cfg, backend="FA4", **parallel_cfg_kwargs)
    try:
        model = WanTransformer3DModel(config).to(device).to(dtype)
    except (ImportError, ValueError, NotImplementedError) as e:
        pytest.skip(f"[{label}] Parallel backend unavailable: {e}")
    _stabilize_model_weights(model)

    torch.manual_seed(SEED_INPUT)
    hidden_states = torch.randn((B, C, T, H, W), device=device, dtype=dtype) * 0.1
    encoder_hidden_states = torch.randn(B, _TEXT_SEQ, text_dim, device=device, dtype=dtype) * 0.1
    timestep = torch.tensor([_TIMESTEP], device=device, dtype=dtype)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    assert output.shape == (B, C, T, H, W), (
        f"[{label}] Rank {rank}: expected shape {(B, C, T, H, W)}, got {tuple(output.shape)}"
    )
    assert not torch.isnan(output).any(), f"[{label}] Rank {rank}: NaN in output"
    assert not torch.isinf(output).any(), f"[{label}] Rank {rank}: Inf in output"

    _free(model, output)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanTransformerParallel:
    """Transformer-only WAN correctness across parallel topologies."""

    def _skip_if_unavailable(self):
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")
        if not _flash_attn4_available:
            pytest.skip("FlashAttn4 JIT kernels not available")

    @pytest.mark.parametrize(
        "label,parallel_cfg_kwargs",
        _WAN_8GPU_PARALLEL_COMBINATIONS,
        ids=[name for name, _ in _WAN_8GPU_PARALLEL_COMBINATIONS],
    )
    def test_parallel_all_combinations_vs_single_gpu_8gpu(self, label, parallel_cfg_kwargs):
        self._skip_if_unavailable()
        run_test_in_distributed(
            world_size=8,
            test_fn=_logic_wan_transformer_parallel_vs_single_gpu,
            parallel_cfg_kwargs=parallel_cfg_kwargs,
            label=label,
        )

    def test_parallel_attn2d_2x2_forward_sanity_4gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(
            world_size=4,
            test_fn=_logic_wan_transformer_parallel_forward_sanity,
            parallel_cfg_kwargs=dict(dit_attn2d_row_size=2, dit_attn2d_col_size=2),
            label="attn2d(2x2)-forward-sanity",
        )

    def test_parallel_attn2d_2x2_vs_single_gpu_4gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(
            world_size=4,
            test_fn=_logic_wan_transformer_parallel_vs_single_gpu,
            parallel_cfg_kwargs=dict(dit_attn2d_row_size=2, dit_attn2d_col_size=2),
            label="attn2d(2x2)-4gpu",
        )

    def test_parallel_attn2d_2x2_ulysses2_vs_single_gpu_8gpu(self):
        """world=8, attn2d=2×2, ulysses=2 vs single-GPU FA4 reference."""
        self._skip_if_unavailable()
        if not _attn2d_available:
            pytest.skip("FA4 / flash_attn_combine JIT kernels not available")
        run_test_in_distributed(
            world_size=8,
            test_fn=_logic_wan_transformer_parallel_vs_single_gpu,
            parallel_cfg_kwargs=dict(
                dit_attn2d_row_size=2,
                dit_attn2d_col_size=2,
                dit_ulysses_size=2,
            ),
            label="attn2d(2x2),ul=2-8gpu",
        )

    def test_parallel_ring4_vs_single_gpu_4gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(
            world_size=4,
            test_fn=_logic_wan_transformer_parallel_vs_single_gpu,
            parallel_cfg_kwargs=dict(dit_ring_size=4),
            label="ring=4-4gpu",
        )

    def test_parallel_ring2_ul2_vs_single_gpu_4gpu(self):
        self._skip_if_unavailable()
        run_test_in_distributed(
            world_size=4,
            test_fn=_logic_wan_transformer_parallel_vs_single_gpu,
            parallel_cfg_kwargs=dict(dit_ring_size=2, dit_ulysses_size=2),
            label="ring=2,ul=2-4gpu",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
