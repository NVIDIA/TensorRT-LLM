"""Multi-GPU tests for parallel VAE (WanParallelVAEAdapter).

Validates that the parallel VAE adapter produces numerically equivalent
decode/encode output compared to the original single-GPU AutoencoderKLWan.

Uses a small randomly-initialised model (no pretrained weights required).

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_parallel_vae.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

    from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import WanParallelVAEAdapter
    from tensorrt_llm._utils import get_free_port

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
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    port = get_free_port()
    mp.spawn(_distributed_worker, args=(world_size, test_fn, port), nprocs=world_size, join=True)


# ---------------------------------------------------------------------------
# Model + data helpers
# ---------------------------------------------------------------------------


def _make_adj_groups(world_size: int):
    return [dist.new_group([i, i + 1]) for i in range(world_size - 1)]


def _broadcast_params(module):
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


def _create_small_vae(device):
    """Create a small AutoencoderKLWan with random weights for testing.

    Config: base_dim=32, z_dim=4, 2 resolution levels, 1 res block,
    no attention, no temporal downsampling.
    Spatial compression = 2x (one downsample/upsample).
    """
    vae = (
        AutoencoderKLWan(
            base_dim=32,
            z_dim=4,
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


# ===========================================================================
# Test-logic functions
# ===========================================================================


def _logic_decode_width(rank, world_size):
    """Parallel decode with width split matches single-GPU decode."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    vae = _create_small_vae(device)
    _broadcast_params(vae)

    # z_dim=4, spatial 16x16 (divisible by world_size), 3 frames
    latent = torch.randn(1, 4, 3, 16, 16, dtype=torch.float32, device=device)
    dist.broadcast(latent, src=0)

    with torch.no_grad():
        ref = vae.decode(latent, return_dict=False)[0].detach().clone()

    WanParallelVAEAdapter(vae, "width", rank, world_size, adj)

    with torch.no_grad():
        par = vae.decode(latent, return_dict=False)[0]

    max_diff = torch.max(torch.abs(par - ref)).item()
    assert max_diff < 0.01, f"Rank {rank}: decode width-split max_diff={max_diff:.6f}"


def _logic_decode_height(rank, world_size):
    """Parallel decode with height split matches single-GPU decode."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    vae = _create_small_vae(device)
    _broadcast_params(vae)

    latent = torch.randn(1, 4, 3, 16, 16, dtype=torch.float32, device=device)
    dist.broadcast(latent, src=0)

    with torch.no_grad():
        ref = vae.decode(latent, return_dict=False)[0].detach().clone()

    WanParallelVAEAdapter(vae, "height", rank, world_size, adj)

    with torch.no_grad():
        par = vae.decode(latent, return_dict=False)[0]

    max_diff = torch.max(torch.abs(par - ref)).item()
    assert max_diff < 0.01, f"Rank {rank}: decode height-split max_diff={max_diff:.6f}"


def _logic_encode_width(rank, world_size):
    """Parallel encode with width split matches single-GPU encode."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    vae = _create_small_vae(device)
    _broadcast_params(vae)

    # Input video: (B, C, T, H, W) — H,W must be divisible by spatial_factor * world_size
    # With spatial_factor=2, world_size=2: min W divisible by 4 → use 32
    video = torch.randn(1, 3, 3, 32, 32, dtype=torch.float32, device=device)
    dist.broadcast(video, src=0)

    with torch.no_grad():
        ref = vae.encode(video).latent_dist.mode().detach().clone()

    WanParallelVAEAdapter(vae, "width", rank, world_size, adj)

    with torch.no_grad():
        par = vae.encode(video).latent_dist.mode()

    max_diff = torch.max(torch.abs(par - ref)).item()
    assert max_diff < 0.01, f"Rank {rank}: encode width-split max_diff={max_diff:.6f}"


# ===========================================================================
# Pytest test classes
# ===========================================================================


class TestParallelVAEDecode:
    def test_decode_width_2gpu(self):
        _run(2, _logic_decode_width)


class TestParallelVAEEncode:
    def test_encode_width_2gpu(self):
        _run(2, _logic_encode_width)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
