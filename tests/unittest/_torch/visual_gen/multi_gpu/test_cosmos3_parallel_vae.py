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

"""Multi-GPU parallel-VAE tests for the Cosmos3 decode/encode path.

``test_parallel_vae.py`` covers the parallel wrappers in isolation. These cover
what Cosmos3 additionally depends on:

* the wrapped VAE still exposes ``config.latents_mean`` / ``latents_std`` /
  ``scale_factor_*`` and ``dtype`` through ``ParallelVAEBase.__getattr__`` --
  ``_decode_latents`` and ``_encode_video_tensor`` read those *after* the
  pipeline has been wrapped, and a broken delegation would silently take the
  ``scaling_factor`` fallback and mis-normalise every latent;
* the Cosmos3 normalisation round trip (denormalise -> decode) matches
  single-GPU;
* ``temporal_chunk_size`` (which only the native parallel wrapper sets) does
  not change results.

Uses small randomly-initialised VAEs -- no checkpoint required.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_cosmos3_parallel_vae.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import sys
    from pathlib import Path

    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

    from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import (
        ParallelVAE_TrtllmWan,
        ParallelVAE_Wan,
    )
    from tensorrt_llm._torch.visual_gen.models.wan.wan_vae import WanVAE, WanVAEConfig
    from tensorrt_llm._torch.visual_gen.modules.vae.parallel_vae_interface import ParallelVAEFactory

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _visual_gen_dist_utils import spawn_with_retry

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _distributed_worker(rank, world_size, test_fn, port):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world_size: int, test_fn: Callable):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker, args=(world_size, test_fn, port), nprocs=world_size, join=True
        )
    )


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

# Cosmos3 ships the Wan2.2 VAE. Shrunk here (base_dim/z_dim/levels) but keeping
# the structural features Cosmos3 depends on: is_residual, patch_size=2 and
# explicit per-channel latents_mean/std.
#
# ``dim_mult`` must END IN A REPEAT (here [1, 1]; the real config is
# [1, 2, 4, 4]). ``WanResidualDownBlock`` builds an ``AvgDown3D`` for every
# level including the last, where ``down_flag=False`` makes factor 1 and the
# grouping check collapses to ``in_dim % out_dim == 0``. An ascending tail like
# [1, 2] gives 32 % 64 and raises "AvgDown3D channel grouping must divide
# evenly" at construction.
_Z_DIM = 4


def _cosmos3_like_native_vae(device):
    vae = (
        WanVAE(
            WanVAEConfig(
                base_dim=32,
                decoder_base_dim=32,
                z_dim=_Z_DIM,
                dim_mult=[1, 1],
                num_res_blocks=1,
                attn_scales=[],
                temperal_downsample=[False],
                is_residual=True,
                in_channels=12,
                out_channels=12,
                patch_size=2,
                latents_mean=[0.1 * i for i in range(_Z_DIM)],
                latents_std=[0.5 + 0.1 * i for i in range(_Z_DIM)],
            )
        )
        .to(device)
        .float()
    )
    vae.eval()
    return vae


def _cosmos3_like_diffusers_vae(device):
    vae = (
        AutoencoderKLWan(
            base_dim=32,
            z_dim=_Z_DIM,
            dim_mult=[1, 2],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False],
        )
        .to(device)
        .float()
    )
    vae.eval()
    return vae


def _broadcast_params(module):
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


def _wrap(vae, world_size, split_dim="width"):
    ranks = list(range(world_size))
    pg = dist.new_group(ranks, use_local_synchronization=False)
    adj = [
        dist.new_group([ranks[i], ranks[i + 1]], use_local_synchronization=False)
        for i in range(world_size - 1)
    ]
    return ParallelVAEFactory.from_vae(vae, split_dim, pg, adj)


def _cosmos3_denormalise(latents, config, device, dtype):
    """The scaling half of ``Cosmos3OmniMoTPipeline._decode_latents``."""
    mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
    std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1).to(device, dtype)
    return latents * std + mean


# ===========================================================================
# Test logic
# ===========================================================================


def _logic_config_delegation(rank, world_size):
    """A wrapped VAE must still answer the attributes ``_decode_latents`` reads.

    Cosmos3 branches on ``hasattr(vae.config, "latents_mean")``. If delegation
    through ``ParallelVAEBase.__getattr__`` were broken the branch would fall
    through to ``scaling_factor`` and every latent would be mis-normalised --
    silently, and only on multi-GPU.
    """
    device = f"cuda:{rank}"
    vae = _cosmos3_like_native_vae(device)
    _broadcast_params(vae)
    ref_mean = list(vae.config.latents_mean)
    ref_dtype = vae.dtype

    wrapped = _wrap(vae, world_size)

    assert hasattr(wrapped.config, "latents_mean"), "latents_mean lost through the wrapper"
    assert hasattr(wrapped.config, "latents_std"), "latents_std lost through the wrapper"
    assert list(wrapped.config.latents_mean) == ref_mean
    assert wrapped.config.scale_factor_spatial == vae.config.scale_factor_spatial
    assert wrapped.config.scale_factor_temporal == vae.config.scale_factor_temporal
    assert wrapped.dtype == ref_dtype, "dtype lost through the wrapper"


def _logic_native_decode(rank, world_size):
    """Sharded native decode matches single-GPU, through Cosmos3's normalisation."""
    device = f"cuda:{rank}"
    vae = _cosmos3_like_native_vae(device)
    _broadcast_params(vae)

    latents = torch.randn(1, _Z_DIM, 3, 16, 16, dtype=torch.float32, device=device)
    dist.broadcast(latents, src=0)
    denorm = _cosmos3_denormalise(latents, vae.config, device, torch.float32)

    with torch.no_grad():
        ref = vae.decode(denorm, return_dict=False)[0].detach().clone()

    wrapped = _wrap(vae, world_size)
    assert isinstance(wrapped, ParallelVAE_TrtllmWan)
    with torch.no_grad():
        par = wrapped.decode(denorm, return_dict=False)[0]

    max_diff = (par - ref).abs().max().item()
    assert max_diff < 0.01, f"Rank {rank}: native parallel decode max_diff={max_diff:.6f}"


def _logic_native_encode(rank, world_size):
    """Sharded native encode matches single-GPU (the I2V/V2V conditioning path)."""
    device = f"cuda:{rank}"
    vae = _cosmos3_like_native_vae(device)
    _broadcast_params(vae)

    video = torch.randn(1, 3, 5, 64, 64, dtype=torch.float32, device=device)
    dist.broadcast(video, src=0)

    with torch.no_grad():
        ref = vae.encode(video).latent_dist.mode().detach().clone()

    wrapped = _wrap(vae, world_size)
    with torch.no_grad():
        par = wrapped.encode(video).latent_dist.mode()

    max_diff = (par - ref).abs().max().item()
    assert max_diff < 0.01, f"Rank {rank}: native parallel encode max_diff={max_diff:.6f}"


def _logic_temporal_chunk_invariance(rank, world_size):
    """``temporal_chunk_size`` batches latent frames; it must not change results.

    Only the native parallel wrapper sets it, so single-GPU never exercises
    the multi-frame ``feat_cache`` path.
    """
    device = f"cuda:{rank}"
    vae = _cosmos3_like_native_vae(device)
    _broadcast_params(vae)

    latents = torch.randn(1, _Z_DIM, 5, 16, 16, dtype=torch.float32, device=device)
    dist.broadcast(latents, src=0)

    with torch.no_grad():
        ref = vae.decode(latents, return_dict=False, temporal_chunk_size=1)[0].detach().clone()
        for chunk in (2, 4):
            out = vae.decode(latents, return_dict=False, temporal_chunk_size=chunk)[0]
            assert out.shape == ref.shape, (
                f"temporal_chunk_size={chunk} changed shape: {out.shape} vs {ref.shape}"
            )
            max_diff = (out - ref).abs().max().item()
            assert max_diff < 0.01, (
                f"temporal_chunk_size={chunk} changed decode output: max_diff={max_diff:.6f}"
            )


def _logic_diffusers_decode(rank, world_size):
    """The diffusers wrapper keeps working while TRTLLM_USE_DIFFUSER_VAE exists."""
    device = f"cuda:{rank}"
    vae = _cosmos3_like_diffusers_vae(device)
    _broadcast_params(vae)

    latents = torch.randn(1, _Z_DIM, 3, 16, 16, dtype=torch.float32, device=device)
    dist.broadcast(latents, src=0)

    with torch.no_grad():
        ref = vae.decode(latents, return_dict=False)[0].detach().clone()

    wrapped = _wrap(vae, world_size)
    assert isinstance(wrapped, ParallelVAE_Wan)
    with torch.no_grad():
        par = wrapped.decode(latents, return_dict=False)[0]

    max_diff = (par - ref).abs().max().item()
    assert max_diff < 0.01, f"Rank {rank}: diffusers parallel decode max_diff={max_diff:.6f}"


# ===========================================================================
# Tests
# ===========================================================================


class TestCosmos3ParallelVAEContract:
    def test_config_delegation_2gpu(self):
        _run(2, _logic_config_delegation)

    def test_temporal_chunk_invariance_2gpu(self):
        _run(2, _logic_temporal_chunk_invariance)


class TestCosmos3ParallelVAENative:
    def test_decode_2gpu(self):
        _run(2, _logic_native_decode)

    def test_encode_2gpu(self):
        _run(2, _logic_native_encode)


class TestCosmos3ParallelVAEDiffusers:
    def test_decode_2gpu(self):
        _run(2, _logic_diffusers_decode)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
