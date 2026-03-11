"""Multi-GPU tests for ParallelVaeAttentionBlock.

Validates that the gather-attend-slice wrapper produces the same output as
running WanAttentionBlock on the full (unsplit) tensor.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_parallel_attention.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

from typing import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock

    from tensorrt_llm._torch.visual_gen.modules.vae import ParallelVaeAttentionBlock
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


def _broadcast_params(module):
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
    assert max_diff < atol, f"Rank {rank}, chunk_dim={chunk_dim}: max_diff={max_diff:.6f}"


def _logic_wan_attention_multi_frame(rank, world_size):
    """ParallelVaeAttentionBlock with multiple temporal frames."""
    device = f"cuda:{rank}"

    attn = WanAttentionBlock(dim=256).to(device).float()
    _broadcast_params(attn)

    for chunk_dim in [3, 4]:
        x, local_x = _prepare(rank, world_size, chunk_dim, (1, 256, 4, 64, 48), device)
        ref = attn(x).detach()

        par_attn = ParallelVaeAttentionBlock(attn, chunk_dim, rank, world_size)
        local_out = par_attn(local_x)

        _gather_and_check(local_out, ref, chunk_dim, world_size, rank)


class TestParallelVaeAttention:
    def test_wan_attention_multi_frame_2gpu(self):
        _run(2, _logic_wan_attention_multi_frame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
