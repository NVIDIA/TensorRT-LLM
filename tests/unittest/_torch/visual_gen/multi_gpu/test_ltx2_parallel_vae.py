"""Multi-GPU tests for LTX-2 tile-parallel VAE decode.

Validates that ``tile_parallel_decode`` (tiles distributed across VAE ranks +
all_reduce of the blend buffer) is numerically equivalent to single-GPU serial
``VideoDecoder.tiled_decode``. Equivalence is up to bf16 summation re-association
in the tile-overlap regions (a few ulp), so the check is ``allclose``, not
``torch.equal``.

Uses a small randomly-initialised VideoDecoder (no pretrained weights). The
decoder is bf16 and decode is deterministic (``timestep_conditioning`` and
per-block ``inject_noise`` are False), so the threaded ``generator`` is unused.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_ltx2_parallel_vae.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import sys
from pathlib import Path
from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae import TilingConfig
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.video_vae.model_configurator import (
    VideoDecoderConfigurator,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.parallel_vae import tile_parallel_decode

# Spawn distributed workers via a helper that retries with a fresh master
# port when the c10d rendezvous TCPStore loses the bind race (EADDRINUSE).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _visual_gen_dist_utils import spawn_with_retry  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _init_worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, test_fn, port):
    try:
        _init_worker(rank, world_size, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        _cleanup()


def _run(world_size: int, test_fn: Callable):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker, args=(world_size, test_fn, port), nprocs=world_size, join=True
        )
    )


# ---------------------------------------------------------------------------
# Model + data helpers
# ---------------------------------------------------------------------------
# Small VideoDecoder: keep patch_size=4 + three compress_all(x2) so the fixed
# 8x temporal / 32x spatial upscale holds, but shrink latent_channels and
# res-block depth so the model + decode stay light. feature_channels derive
# from latent_channels (x8 here -> 256).
_SMALL_VAE_CONFIG = {
    "vae": {
        "dims": 3,
        "latent_channels": 32,
        "out_channels": 3,
        "patch_size": 4,
        "norm_layer": "pixel_norm",
        "causal_decoder": False,
        "timestep_conditioning": False,
        "decoder_blocks": [
            ["res_x", {"num_layers": 1, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 1, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 1, "inject_noise": False}],
            ["compress_all", {"residual": True, "multiplier": 2}],
            ["res_x", {"num_layers": 1, "inject_noise": False}],
        ],
    }
}
# Latent sized so TilingConfig.default() splits into multiple tiles along both
# spatial (2x2) and temporal (3 groups) axes -> exercises overlap blend + the
# cross-temporal-group stitch that tile_parallel_decode must reproduce.
_LATENT_SHAPE = (1, 32, 16, 18, 18)


def _create_small_video_decoder(device):
    dec = (
        VideoDecoderConfigurator.from_config(_SMALL_VAE_CONFIG)
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )
    # Config-initialised per-channel stats are uncalibrated; bypass un_normalize
    # so the (otherwise NaN) decode is well-defined. It is a per-element op applied
    # identically in serial and parallel, so it does not affect the parity check.
    dec.per_channel_statistics.un_normalize = lambda x, *a, **k: x
    return dec


def _broadcast_params(module):
    for p in module.parameters():
        dist.broadcast(p.data, src=0)
    for b in module.buffers():
        dist.broadcast(b.data, src=0)


# ===========================================================================
# Test-logic functions
# ===========================================================================
def _logic_tile_parallel_decode_parity(rank, world_size):
    """tile_parallel_decode matches single-GPU serial tiled_decode (within bf16 ulp)."""
    device = f"cuda:{rank}"

    torch.manual_seed(0)
    dec = _create_small_video_decoder(device)
    _broadcast_params(dec)  # ensure every rank shares identical weights

    latent = torch.randn(*_LATENT_SHAPE, device=device, dtype=torch.bfloat16)
    dist.broadcast(latent, src=0)

    cfg = TilingConfig.default()
    pg = dist.new_group(list(range(world_size)), use_local_synchronization=False)

    with torch.no_grad():
        # serial reference (deterministic -> identical on every rank)
        ref = torch.cat(list(dec.tiled_decode(latent, cfg)), dim=2)
        # distributed tile-parallel decode
        par = tile_parallel_decode(dec, latent, cfg, pg)

    assert par.shape == ref.shape, f"Rank {rank}: shape {tuple(par.shape)} != {tuple(ref.shape)}"
    diff = (par.float() - ref.float()).abs()
    max_abs = diff.max().item()
    scale = ref.float().abs().max().item()
    rel = max_abs / max(scale, 1e-6)
    # bf16 re-association in the overlap regions -> a few ulp (~2^-7 relative).
    assert rel < 0.02, (
        f"Rank {rank}: tile-parallel parity rel={rel:.4f} (max_abs={max_abs:.4e}, scale={scale:.4e})"
    )


# ===========================================================================
# Pytest test classes
# ===========================================================================
class TestLTX2ParallelVAEDecode:
    def test_tile_parallel_decode_parity_2gpu(self):
        _run(2, _logic_tile_parallel_decode_parity)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
