"""Multi-GPU tests for GroupNormParallel.

Validates that GroupNormParallel (which all-reduces mean/var across spatial
splits) matches standard nn.GroupNorm on the full tensor.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_parallel_group_norm.py -v
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
    from tensorrt_llm._torch.visual_gen.modules.vae import GroupNormParallel
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


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


# ===========================================================================
# Test-logic functions
# ===========================================================================


def _logic_groupnorm_4d(rank, world_size):
    """GroupNormParallel on 4D tensor (B, C, H, W), split along height and width."""
    device = f"cuda:{rank}"

    gn = nn.GroupNorm(num_groups=32, num_channels=256, eps=1e-6, affine=True).to(device).float()
    for p in gn.parameters():
        dist.broadcast(p.data, src=0)

    parallel_gn = GroupNormParallel(gn, world_size=world_size)

    for chunk_dim in [2, 3]:
        x = torch.randn(2, 256, 64, 48, dtype=torch.float32, device=device)
        dist.broadcast(x, src=0)
        local_x = x.chunk(world_size, dim=chunk_dim)[rank]

        ref = gn(x).detach()

        local_out = parallel_gn(local_x).contiguous()
        gathered = [torch.empty_like(local_out) for _ in range(world_size)]
        dist.all_gather(gathered, local_out)
        out = torch.cat(gathered, dim=chunk_dim)

        max_diff = torch.max(torch.abs(out - ref)).item()
        assert max_diff < 0.01, f"Rank {rank}, chunk_dim={chunk_dim}: max_diff={max_diff:.6f}"


def _logic_groupnorm_5d(rank, world_size):
    """GroupNormParallel on 5D tensor (B, C, T, H, W), split along height and width."""
    device = f"cuda:{rank}"

    gn = nn.GroupNorm(num_groups=16, num_channels=128, eps=1e-6, affine=True).to(device).float()
    for p in gn.parameters():
        dist.broadcast(p.data, src=0)

    parallel_gn = GroupNormParallel(gn, world_size=world_size)

    for chunk_dim in [3, 4]:
        x = torch.randn(1, 128, 4, 64, 48, dtype=torch.float32, device=device)
        dist.broadcast(x, src=0)
        local_x = x.chunk(world_size, dim=chunk_dim)[rank]

        ref = gn(x).detach()

        local_out = parallel_gn(local_x).contiguous()
        gathered = [torch.empty_like(local_out) for _ in range(world_size)]
        dist.all_gather(gathered, local_out)
        out = torch.cat(gathered, dim=chunk_dim)

        max_diff = torch.max(torch.abs(out - ref)).item()
        assert max_diff < 0.01, f"Rank {rank}, chunk_dim={chunk_dim}: max_diff={max_diff:.6f}"


# ===========================================================================
# Pytest test classes
# ===========================================================================


class TestGroupNormParallel:
    def test_4d_2gpu(self):
        _run(2, _logic_groupnorm_4d)

    def test_5d_2gpu(self):
        _run(2, _logic_groupnorm_5d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
