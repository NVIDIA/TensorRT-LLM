"""Multi-GPU tests for parallel convolution wrappers.

Tests HaloExchangeConv (stride-1) and HaloExchangeConv2dStride2 (stride-2)
against single-GPU reference computations.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_parallel_conv.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

try:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d

    from tensorrt_llm._torch.visual_gen.models.wan.parallel_vae import WanCausalConvHalo
    from tensorrt_llm._torch.visual_gen.modules.vae import (
        HaloExchangeConv,
        HaloExchangeConv2dStride2,
    )
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# ---------------------------------------------------------------------------
# Distributed helpers (same pattern as test_ulysses_attention.py)
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
# Shared utilities used inside worker processes
# ---------------------------------------------------------------------------


def _make_adj_groups(world_size: int):
    return [dist.new_group([i, i + 1]) for i in range(world_size - 1)]


def _broadcast_params(module: nn.Module):
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


def _prepare(rank, world_size, chunk_dim, shape, device):
    x = torch.randn(shape, dtype=torch.float32, device=device)
    dist.broadcast(x, src=0)
    local_x = x.chunk(world_size, dim=chunk_dim)[rank]
    return x, local_x


def _gather_and_check(local_out, ref_out, chunk_dim, world_size, rank, atol=0.01):
    local_out = local_out.contiguous()
    gathered = [torch.empty_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out)
    out = torch.cat(gathered, dim=chunk_dim)
    max_diff = torch.max(torch.abs(out - ref_out)).item()
    assert max_diff < atol, f"Rank {rank}: max_diff={max_diff:.6f} (>= {atol})"


# ===========================================================================
# Test-logic functions (module-level for mp.spawn pickling)
# ===========================================================================


def _logic_halo_conv3d(rank, world_size):
    """WanCausalConvHalo wrapping WanCausalConv3d (kernel=3, with cache_x)."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    conv = WanCausalConv3d(96, 96, kernel_size=(3, 3, 3), stride=1, padding=1).to(device).float()
    _broadcast_params(conv)

    for chunk_dim in [3, 4]:
        x, local_x = _prepare(rank, world_size, chunk_dim, (1, 96, 4, 64, 48), device)

        cache_x = torch.randn(1, 96, 2, 64, 48, dtype=torch.float32, device=device)
        dist.broadcast(cache_x, src=0)
        local_cache = cache_x.chunk(world_size, dim=chunk_dim)[rank]

        ref = conv(x, cache_x).detach()

        par = WanCausalConvHalo(conv, chunk_dim, adj, rank, world_size)
        local_out = par(local_x, local_cache)

        _gather_and_check(local_out, ref, chunk_dim, world_size, rank)


def _logic_halo_conv2d(rank, world_size):
    """HaloExchangeConv wrapping nn.Conv2d (kernel=3, stride=1)."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    conv = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1).to(device).float()
    _broadcast_params(conv)

    for chunk_dim in [2, 3]:
        x, local_x = _prepare(rank, world_size, chunk_dim, (1, 96, 64, 48), device)
        ref = conv(x).detach()

        par = HaloExchangeConv(conv, chunk_dim, adj, rank, world_size)
        local_out = par(local_x)

        _gather_and_check(local_out, ref, chunk_dim, world_size, rank)


def _logic_halo_conv2d_stride2(rank, world_size):
    """HaloExchangeConv2dStride2 wrapping nn.Conv2d (kernel=3, stride=2)."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    conv = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0).to(device).float()
    _broadcast_params(conv)

    pad = nn.ZeroPad2d((0, 1, 0, 1))

    for chunk_dim in [2, 3]:
        x, local_x = _prepare(rank, world_size, chunk_dim, (4, 96, 64, 48), device)
        ref = conv(pad(x)).detach()

        par = HaloExchangeConv2dStride2(
            conv,
            chunk_dim,
            adj,
            rank,
            world_size,
            pad_before_conv=(0, 1, 0, 1),
        )
        local_out = par(local_x)

        _gather_and_check(local_out, ref, chunk_dim, world_size, rank)


class TestHaloExchangeConv:
    def test_wan_conv3d_with_cache_2gpu(self):
        _run(2, _logic_halo_conv3d)

    def test_conv2d_2gpu(self):
        _run(2, _logic_halo_conv2d)


class TestHaloExchangeConv2dStride2:
    def test_conv2d_stride2_2gpu(self):
        _run(2, _logic_halo_conv2d_stride2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
