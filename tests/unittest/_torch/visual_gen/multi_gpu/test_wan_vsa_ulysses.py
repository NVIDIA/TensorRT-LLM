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
"""Multi-GPU correctness tests for WAN VSA + Ulysses parallelism.

Loads the Wan2.1-VSA-T2V-14B-720P checkpoint and compares a CFG + Ulysses
transformer forward against a single-GPU VSA reference on small video input.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_wan_vsa_ulysses.py -v -s
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
    from tensorrt_llm._torch.visual_gen.attention_backend.cute_dsl import (
        VSAMetadataBuilder,
        _cute_dsl_import_error,
        set_vsa_forward_context,
    )
    from tensorrt_llm._torch.visual_gen.config import (
        AttentionConfig,
        DiffusionModelConfig,
        TorchCompileConfig,
    )
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.visual_gen.sparse_attention import VideoSparseAttentionConfig

    MODULES_AVAILABLE = True
    _cute_dsl_available = _cute_dsl_import_error is None
except ImportError:
    MODULES_AVAILABLE = False
    _cute_dsl_available = False
    _cute_dsl_import_error = None


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
    # Disable NVLS before init_process_group; resolves B200 timeout errors
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
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


def run_test_in_distributed(world_size: int, test_fn: Callable, **kwargs):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")
    port = get_free_port()
    mp.spawn(
        _distributed_worker,
        args=(world_size, "nccl", test_fn, port, kwargs),
        nprocs=world_size,
        join=True,
    )


_VIDEO_T = 4
_VIDEO_H = 16
_VIDEO_W = 16
_TEXT_SEQ = 32
_TIMESTEP = 0.5

VSA_SPARSITY = 0.9

SEED_INPUT = 100

ATOL = 1e-2
RTOL = 1e-3


def _make_vsa_model_config(
    pretrained_dict: dict,
    *,
    ulysses_size: int = 1,
    cfg_size: int = 1,
) -> "DiffusionModelConfig":
    """Build a DiffusionModelConfig with CUTEDSL+VSA attention."""
    pretrained_config = SimpleNamespace(**pretrained_dict)
    use_dist = (ulysses_size > 1 or cfg_size > 1) and dist.is_initialized()
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
    )
    config = DiffusionModelConfig(
        pretrained_config=pretrained_config,
        torch_compile=TorchCompileConfig(enable=False),
        attention=AttentionConfig(
            backend="CUTEDSL",
            sparse_attention_config=VideoSparseAttentionConfig(vsa_sparsity=VSA_SPARSITY),
        ),
        visual_gen_mapping=vgm,
        skip_create_weights_in_init=False,
    )
    config.mapping = vgm.to_llm_mapping()
    return config


def _vsa_forward(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    patch_size,
) -> torch.Tensor:
    """Single transformer forward wrapped in a VSA context."""
    vsa_builder = VSAMetadataBuilder()
    raw_latent_shape = (hidden_states.shape[2], hidden_states.shape[3], hidden_states.shape[4])
    vsa_metadata = vsa_builder.build(
        current_timestep=0,
        raw_latent_shape=raw_latent_shape,
        patch_size=tuple(patch_size),
        vsa_sparsity=VSA_SPARSITY,
        device=hidden_states.device,
    )
    with set_vsa_forward_context(vsa_metadata):
        return model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Worker logic
# =============================================================================


def _log(rank: int, msg: str) -> None:
    import os as _os

    _os.write(2, f"[rank {rank}] {msg}\n".encode())


# =============================================================================
# Checkpoint path helpers
# =============================================================================


def _llm_models_root() -> str | None:
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    return str(root) if root.exists() else None


def _checkpoint(env_var: str, default_name: str) -> str | None:
    if env_var in os.environ:
        return os.environ[env_var]
    models_root = _llm_models_root()
    return os.path.join(models_root, default_name) if models_root is not None else None


WAN21_VSA_14B_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_WAN21_VSA_T2V_14B_720P",
    "Wan2.1-VSA-T2V-14B-720P-Diffusers",
)


# =============================================================================
# Real-model worker logic
# =============================================================================


def _logic_vsa_ulysses_real_model(
    rank: int,
    world_size: int,
    *,
    ulysses_size: int,
    cfg_size: int = 1,
    checkpoint_path: str,
    label: str,
) -> None:
    from tensorrt_llm._torch.visual_gen.checkpoints.weight_loader import WeightLoader
    from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel

    _log(rank, f"[{label}] imports done")
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    config_json_path = os.path.join(checkpoint_path, "transformer", "config.json")
    with open(config_json_path) as f:
        pretrained_dict = json.load(f)

    _log(rank, f"[{label}] building ref_model (world_size=1)")
    ref_config = _make_vsa_model_config(pretrained_dict)
    try:
        ref_model = WanTransformer3DModel(ref_config).to(device).to(dtype)
    except (ImportError, ValueError, NotImplementedError) as e:
        pytest.skip(f"[{label}] VSA ref model unavailable: {e}")
    _log(rank, f"[{label}] ref_model created")

    _log(rank, f"[{label}] building dist_model (ul={ulysses_size} cfg={cfg_size})")
    dist_config = _make_vsa_model_config(
        pretrained_dict, ulysses_size=ulysses_size, cfg_size=cfg_size
    )
    try:
        dist_model = WanTransformer3DModel(dist_config).to(device).to(dtype)
    except (ImportError, ValueError, NotImplementedError) as e:
        pytest.skip(f"[{label}] VSA parallel model unavailable: {e}")
    _log(rank, f"[{label}] dist_model created")

    _log(rank, f"[{label}] loading checkpoint weights")
    weight_loader = WeightLoader(components="transformer")
    weights = weight_loader.load_weights(checkpoint_path, ref_config.mapping)
    ref_model.load_weights(weights)
    ref_model.post_load_weights()
    ref_state = ref_model.state_dict()
    dist_model.load_state_dict(ref_state)
    dist_model.post_load_weights()
    _log(rank, f"[{label}] weights loaded")

    in_channels = getattr(ref_model.config, "in_channels", 16)
    text_dim = getattr(ref_model.config, "text_dim", 4096)
    patch_size = getattr(ref_model.config, "patch_size", [1, 2, 2])

    B = 1
    T, H, W = _VIDEO_T, _VIDEO_H, _VIDEO_W

    torch.manual_seed(SEED_INPUT)
    hidden_states = torch.randn((B, in_channels, T, H, W), device=device, dtype=dtype) * 0.1
    encoder_hidden_states = torch.randn(B, _TEXT_SEQ, text_dim, device=device, dtype=dtype) * 0.1
    timestep = torch.tensor([_TIMESTEP], device=device, dtype=dtype)

    with torch.no_grad():
        _log(rank, f"[{label}] dist_model forward start")
        dist_output = _vsa_forward(
            dist_model, hidden_states, encoder_hidden_states, timestep, patch_size
        )
        _log(rank, f"[{label}] dist_model forward done")

    _free(dist_model)
    dist.barrier()

    if rank != 0:
        return

    with torch.no_grad():
        _log(rank, f"[{label}] ref_model forward start")
        ref_output = _vsa_forward(
            ref_model, hidden_states, encoder_hidden_states, timestep, patch_size
        )
        _log(rank, f"[{label}] ref_model forward done")

    abs_diff = (dist_output.float() - ref_output.float()).abs()
    print(
        f"\n  [{label}] real model "
        f"max_abs_diff={abs_diff.max().item():.6e}, "
        f"mean_abs_diff={abs_diff.mean().item():.6e}"
    )
    torch.testing.assert_close(
        dist_output,
        ref_output,
        rtol=RTOL,
        atol=ATOL,
        msg=f"[{label}] real-model VSA+Ulysses output differs from single-GPU VSA reference",
    )
    _free(ref_model, ref_output, dist_output)


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanVsaUlyssesRealModel:
    """Multi-GPU VSA correctness using the Wan2.1-VSA-T2V-14B-720P checkpoint.

    Loads the VSA transformer weights and compares a CFG=2, Ulysses=4 (8-GPU)
    forward against a single-GPU reference on small video input (T=4, H=16, W=16).
    """

    def _skip_if_unavailable(self):
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")
        if not _cute_dsl_available:
            pytest.skip(f"CUTEDSL not available (requires Blackwell GPU): {_cute_dsl_import_error}")
        if WAN21_VSA_14B_PATH is None or not os.path.isdir(WAN21_VSA_14B_PATH):
            pytest.skip(
                "Wan2.1-VSA-T2V-14B-720P checkpoint not found; "
                "set DIFFUSION_MODEL_PATH_WAN21_VSA_T2V_14B_720P or LLM_MODELS_ROOT"
            )

    def test_real_model_vsa_cfg2_ulysses4_vs_single_gpu(self):
        """world=8, cfg=2, ulysses=4, real VSA 14B weights, VSA sparsity=0.9."""
        self._skip_if_unavailable()
        run_test_in_distributed(
            world_size=8,
            test_fn=_logic_vsa_ulysses_real_model,
            ulysses_size=4,
            cfg_size=2,
            checkpoint_path=WAN21_VSA_14B_PATH,
            label="real,cfg=2,ul=4",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
