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
"""Transformer-only FLUX.2 parallel correctness harness.

Compares distributed Flux2Transformer2DModel forward passes against a
single-GPU reference using randomly-initialized (stabilized) weights — no real
checkpoint is loaded.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_flux2_transformer_parallel.py -v -s
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
    import sys
    from pathlib import Path

    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TorchCompileConfig,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping

    # Reuse the CI-aware free-port allocator from tests/integration so that
    # sequentially spawned distributed workers get disjoint MASTER_PORTs and
    # don't collide with ports still in TIME_WAIT (EADDRINUSE).
    _integration_dir = Path(__file__).resolve().parents[4] / "integration"
    if str(_integration_dir) not in sys.path:
        sys.path.insert(0, str(_integration_dir))
    from defs.common import get_free_port_in_ci

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
    port = get_free_port_in_ci()
    mp.spawn(
        _distributed_worker,
        args=(world_size, backend, test_fn, port, kwargs),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# FLUX.2 checkpoint + test input setup
# =============================================================================


# Small in-code FLUX.2 transformer config (reduced layers/dims) so tests run
# without loading a real checkpoint. head_dim=128 keeps the FA4 backend happy;
# num_attention_heads=8 is divisible by every ulysses_size exercised below.
# axes_dims_rope must sum to attention_head_dim (32*4 = 128).
_FLUX2_TEST_CONFIG = dict(
    num_attention_heads=8,
    attention_head_dim=128,
    num_layers=2,
    num_single_layers=2,
    in_channels=64,
    out_channels=64,
    joint_attention_dim=256,
    pooled_projection_dim=128,
    mlp_ratio=3.0,
    patch_size=1,
    guidance_embeds=True,
    axes_dims_rope=[32, 32, 32, 32],
    rope_theta=2000.0,
    eps=1e-6,
    timestep_guidance_channels=256,
)

# Latent token grid + text seq chosen so both sequence dims are divisible by
# every world_size (img_seq = 8 * 8 = 64; txt_seq = 16).
_IMG_H = 8
_IMG_W = 8
_TXT_SEQ = 16

_TIMESTEP = 0.5
_GUIDANCE = 3.5
SEED_WEIGHTS = 42
SEED_INPUT = 100

ATOL = 1e-2
RTOL = 1e-3

# All valid 8-GPU combinations of (ulysses, ring, attn2d), with cfg_size fixed at 1:
# world_size = (ring or attn2d_row*attn2d_col or 1) * ulysses = 8
_FLUX2_8GPU_PARALLEL_COMBINATIONS = [
    # Ulysses-only
    ("ulysses_only_ul8", dict(dit_ulysses_size=8)),
    # Ring/Ulysses family
    ("ring4_ul2", dict(dit_ring_size=4, dit_ulysses_size=2)),
    # Attention2D/Ulysses family
    ("attn2d_4x1_ul2", dict(dit_attn2d_row_size=4, dit_attn2d_col_size=1, dit_ulysses_size=2)),
    ("attn2d_2x4_ul1", dict(dit_attn2d_row_size=2, dit_attn2d_col_size=4, dit_ulysses_size=1)),
    ("attn2d_1x8_ul1", dict(dit_attn2d_row_size=1, dit_attn2d_col_size=8, dit_ulysses_size=1)),
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
    # Support VisualGen-style dit_* kwargs for convenience.
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
        cache=None,
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def _logic_flux2_transformer_parallel_vs_single_gpu(
    rank: int,
    world_size: int,
    *,
    parallel_cfg_kwargs: dict,
    label: str,
):
    from tensorrt_llm._torch.visual_gen.models.flux.transformer_flux2 import Flux2Transformer2DModel

    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    pretrained_cfg = _FLUX2_TEST_CONFIG

    in_channels = int(pretrained_cfg["in_channels"])
    joint_dim = int(pretrained_cfg["joint_attention_dim"])
    rope_id_dims = len(pretrained_cfg["axes_dims_rope"])

    batch = 1
    img_seq = _IMG_H * _IMG_W

    torch.manual_seed(SEED_WEIGHTS)
    ref_config = _make_model_config(pretrained_cfg, backend="FA4")
    ref_model = Flux2Transformer2DModel(ref_config).to(device).to(dtype)
    _stabilize_model_weights(ref_model)
    ref_state = ref_model.state_dict()

    torch.manual_seed(SEED_WEIGHTS)
    dist_config = _make_model_config(pretrained_cfg, backend="FA4", **parallel_cfg_kwargs)
    try:
        dist_model = Flux2Transformer2DModel(dist_config).to(device).to(dtype)
    except (ImportError, ValueError, NotImplementedError) as e:
        pytest.skip(f"[{label}] Parallel backend unavailable: {e}")
    dist_model.load_state_dict(ref_state)

    torch.manual_seed(SEED_INPUT)
    hidden_states = torch.randn(batch, img_seq, in_channels, device=device, dtype=dtype) * 0.1
    encoder_hidden_states = (
        torch.randn(batch, _TXT_SEQ, joint_dim, device=device, dtype=dtype) * 0.1
    )
    timestep = torch.tensor([_TIMESTEP], device=device, dtype=dtype)
    guidance = torch.tensor([_GUIDANCE], device=device, dtype=dtype)
    img_ids = torch.zeros(img_seq, rope_id_dims, device=device)
    txt_ids = torch.zeros(_TXT_SEQ, rope_id_dims, device=device)

    with torch.no_grad():
        ref_out = ref_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )["sample"]
        dist_out = dist_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )["sample"]

    abs_diff = (dist_out.float() - ref_out.float()).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    if rank == 0:
        print(
            f"[{label}] comparison stats: "
            f"max_abs_diff={max_abs_diff:.6e}, mean_abs_diff={mean_abs_diff:.6e}"
        )

    torch.testing.assert_close(
        dist_out,
        ref_out,
        rtol=RTOL,
        atol=ATOL,
        msg=f"Rank {rank}: [{label}] Flux2Transformer2DModel output differs from single-GPU FA4 reference",
    )

    _free(ref_model, dist_model, ref_out, dist_out)


@pytest.mark.integration
@pytest.mark.flux2
class TestFlux2TransformerParallel:
    def _skip_if_unavailable(self):
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")
        if not _flash_attn4_available:
            pytest.skip("FlashAttn4 JIT kernels not available")

    @pytest.mark.parametrize(
        "label,parallel_cfg_kwargs",
        _FLUX2_8GPU_PARALLEL_COMBINATIONS,
        ids=[name for name, _ in _FLUX2_8GPU_PARALLEL_COMBINATIONS],
    )
    def test_parallel_all_combinations_vs_single_gpu_8gpu(self, label, parallel_cfg_kwargs):
        self._skip_if_unavailable()
        run_test_in_distributed(
            world_size=8,
            test_fn=_logic_flux2_transformer_parallel_vs_single_gpu,
            parallel_cfg_kwargs=parallel_cfg_kwargs,
            label=label,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
