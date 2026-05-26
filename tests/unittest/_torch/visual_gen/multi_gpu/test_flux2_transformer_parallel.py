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
single-GPU reference using real checkpoint config + transformer weights from:

    $LLM_MODELS_ROOT/<model_subdir>/transformer

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_flux2_transformer_parallel.py -v -s
"""

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.checkpoints.weight_loader import WeightLoader
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
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, backend, test_fn, port, kwargs),
        nprocs=world_size,
        join=True,
    )


# =============================================================================
# FLUX.2 checkpoint + test input setup
# =============================================================================


_PRETRAINED_CONFIG_CACHE: dict | None = None

# Use production-like minimum pipeline size by default (512x512 image),
# converted to latent token grid (VAE downsample x8 -> 64x64; seq_len=4096).
_IMG_H = 64
_IMG_W = 64
_TXT_SEQ = 77

_TIMESTEP = 0.5
_GUIDANCE = 3.5
SEED_WEIGHTS = 42
SEED_INPUT = 100

DEFAULT_FLUX2_MODEL_SUBDIR = "FLUX.2-dev"
FLUX2_TRANSFORMER_MODEL_SUBDIR = os.environ.get(
    "FLUX2_TRANSFORMER_MODEL_SUBDIR", DEFAULT_FLUX2_MODEL_SUBDIR
)

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


def _llm_models_root() -> Path:
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    if not root.exists():
        pytest.skip("LLM model root not found. Set LLM_MODELS_ROOT.")
    return root


def _transformer_checkpoint_dir() -> Path:
    ckpt_dir = _llm_models_root() / FLUX2_TRANSFORMER_MODEL_SUBDIR / "transformer"
    if not ckpt_dir.exists():
        pytest.skip(f"Transformer checkpoint dir not found: {ckpt_dir}")
    return ckpt_dir


def _transformer_pretrained_config(checkpoint_dir: Path) -> dict:
    global _PRETRAINED_CONFIG_CACHE
    if _PRETRAINED_CONFIG_CACHE is not None:
        return _PRETRAINED_CONFIG_CACHE

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        pytest.skip(f"Transformer config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        pytest.skip(f"Invalid transformer config format in {config_path}")

    required = [
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "num_single_layers",
        "in_channels",
        "out_channels",
        "joint_attention_dim",
        "patch_size",
    ]
    missing = [k for k in required if k not in loaded]
    if missing:
        pytest.skip(f"Transformer config missing required keys {missing} in {config_path}")

    _PRETRAINED_CONFIG_CACHE = loaded
    return loaded


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
        torch_compile=TorchCompileConfig(enable_torch_compile=False),
        attention=AttentionConfig(backend=backend),
        visual_gen_mapping=vgm,
        cache=None,
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _load_transformer_weights(checkpoint_dir: Path, mapping) -> dict:
    loader = WeightLoader(components="transformer")
    return loader.load_weights(str(checkpoint_dir), mapping)


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def _tolerance_for_config(parallel_cfg_kwargs: dict) -> tuple[float, float]:
    """Return (rtol, atol) calibrated from empirical FLUX.2 BF16 worst-case measurements.

    Unlike WAN, FLUX.2 Ulysses is not lossless: joint image+text attention
    always scatters tokens across ranks, so all configs incur collective rounding.
    Two classes are still distinguishable:

    Ulysses / col=1 — no cross-rank KV exchange:
        empirical max_abs_diff ~2.64e-02, mean ~4.20e-03
        → atol=5e-2 (1.9× headroom), rtol=1e-3

    KV-exchange — ring CP or attn2d col_size > 1:
        empirical max_abs_diff up to ~4.64e-02, mean up to ~7.80e-03
        → atol=1e-1 (2.2× headroom), rtol=1e-2
    """
    ring_size = parallel_cfg_kwargs.get("dit_ring_size", 1)
    col_size = parallel_cfg_kwargs.get("dit_attn2d_col_size", 1)
    if ring_size > 1 or col_size > 1:
        return 1e-2, 1e-1
    return 1e-3, 5e-2


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

    checkpoint_dir = _transformer_checkpoint_dir()
    pretrained_cfg = _transformer_pretrained_config(checkpoint_dir)

    in_channels = int(pretrained_cfg["in_channels"])
    joint_dim = int(pretrained_cfg["joint_attention_dim"])
    rope_id_dims = len(
        pretrained_cfg.get("axes_dims_rope", pretrained_cfg.get("axes_dim", [0, 0, 0, 0]))
    )
    if rope_id_dims <= 0:
        rope_id_dims = 4

    batch = 1
    img_seq = _IMG_H * _IMG_W

    torch.manual_seed(SEED_WEIGHTS)
    ref_config = _make_model_config(pretrained_cfg, backend="FA4")
    ref_model = Flux2Transformer2DModel(ref_config).to(device).to(dtype)
    ref_weights = _load_transformer_weights(checkpoint_dir, ref_config.mapping)
    ref_model.load_weights(ref_weights)
    ref_model.post_load_weights()

    torch.manual_seed(SEED_WEIGHTS)
    dist_config = _make_model_config(pretrained_cfg, backend="FA4", **parallel_cfg_kwargs)
    try:
        dist_model = Flux2Transformer2DModel(dist_config).to(device).to(dtype)
    except (ImportError, ValueError, NotImplementedError) as e:
        pytest.skip(f"[{label}] Parallel backend unavailable: {e}")
    dist_model.load_weights(ref_weights)
    dist_model.post_load_weights()

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

    rtol, atol = _tolerance_for_config(parallel_cfg_kwargs)
    torch.testing.assert_close(
        dist_out,
        ref_out,
        rtol=rtol,
        atol=atol,
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
