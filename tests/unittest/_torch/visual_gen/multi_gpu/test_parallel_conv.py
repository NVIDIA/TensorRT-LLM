# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    from tensorrt_llm._torch.visual_gen.modules.vae.conv import (
        _cat_spatial_halos,
        _halo_exchange_buffer,
        _physical_to_logical_channels_last,
        _spatial_channels_last_format,
    )

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
    # Spawn distributed workers via a helper that retries with a fresh master
    # port when the c10d rendezvous TCPStore loses the bind race (EADDRINUSE).
    from ._visual_gen_dist_utils import spawn_with_retry

    spawn_with_retry(
        lambda port: mp.spawn(
            _distributed_worker,
            args=(world_size, test_fn, port),
            nprocs=world_size,
            join=True,
        )
    )


# ---------------------------------------------------------------------------
# Shared utilities used inside worker processes
# ---------------------------------------------------------------------------


def _make_adj_groups(world_size: int):
    return [dist.new_group([i, i + 1]) for i in range(world_size - 1)]


def _broadcast_params(module: nn.Module):
    for p in module.parameters():
        dist.broadcast(p.data, src=0)


def _prepare(
    rank: int,
    world_size: int,
    chunk_dim: int,
    shape: tuple[int, ...],
    device: str,
    memory_format: torch.memory_format | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(shape, dtype=torch.float32, device=device)
    dist.broadcast(x, src=0)
    local_x = x.chunk(world_size, dim=chunk_dim)[rank]
    if memory_format is not None:
        local_x = local_x.contiguous(memory_format=memory_format)
    return x, local_x


def _gather_and_check(
    local_out,
    ref_out,
    chunk_dim,
    world_size,
    rank,
    atol=0.01,
):
    local_out = local_out.contiguous()
    gathered = [torch.empty_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out)
    out = torch.cat(gathered, dim=chunk_dim)
    max_diff = torch.max(torch.abs(out - ref_out)).item()
    assert max_diff < atol, f"Rank {rank}: max_diff={max_diff:.6f} (>= {atol})"


# ===========================================================================
# Test-logic functions (module-level for mp.spawn pickling)
# ===========================================================================


def _logic_halo_conv3d(rank: int, world_size: int) -> None:
    """WanCausalConvHalo wrapping WanCausalConv3d (kernel=3, with cache_x)."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    conv = WanCausalConv3d(96, 96, kernel_size=(3, 3, 3), stride=1, padding=1).to(device).float()
    _broadcast_params(conv)

    for chunk_dim in [3, 4]:
        for memory_format in [torch.contiguous_format, torch.channels_last_3d]:
            x, local_x = _prepare(
                rank,
                world_size,
                chunk_dim,
                (1, 96, 4, 64, 48),
                device,
                memory_format,
            )

            cache_x = torch.randn(1, 96, 2, 64, 48, dtype=torch.float32, device=device)
            dist.broadcast(cache_x, src=0)
            local_cache = cache_x.chunk(world_size, dim=chunk_dim)[rank].contiguous(
                memory_format=memory_format
            )

            ref = conv(x, cache_x).detach()

            par = WanCausalConvHalo(conv, chunk_dim, adj, rank, world_size)
            local_out = par(local_x, local_cache)

            _gather_and_check(
                local_out,
                ref,
                chunk_dim,
                world_size,
                rank,
            )


def _logic_halo_conv3d_even_kernel(rank: int, world_size: int) -> None:
    """Channels-last exchange trims an asymmetric halo on its physical axis."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)
    chunk_dim = 4
    conv = nn.Conv3d(4, 4, kernel_size=(3, 3, 4), padding=1).to(device).float()
    x, local_x = _prepare(
        rank,
        world_size,
        chunk_dim,
        (1, 4, 3, 8, 8),
        device,
        torch.channels_last_3d,
    )

    parallel = HaloExchangeConv(conv, chunk_dim, adj, rank, world_size)
    exchanged = parallel._exchange_halos(local_x)

    padded = torch.nn.functional.pad(x, (parallel.halo_left, parallel.halo_right))
    local_width = local_x.shape[chunk_dim]
    expected = torch.narrow(
        padded,
        chunk_dim,
        rank * local_width,
        local_width + parallel.halo_left + parallel.halo_right,
    )
    torch.testing.assert_close(exchanged, expected, rtol=0, atol=0)
    assert exchanged.is_contiguous(memory_format=torch.channels_last_3d)


def _logic_halo_conv2d(rank: int, world_size: int) -> None:
    """HaloExchangeConv wrapping nn.Conv2d (kernel=3, stride=1)."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    conv = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1).to(device).float()
    _broadcast_params(conv)

    for chunk_dim in [2, 3]:
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            x, local_x = _prepare(
                rank,
                world_size,
                chunk_dim,
                (1, 96, 64, 48),
                device,
                memory_format,
            )
            ref = conv(x).detach()

            par = HaloExchangeConv(conv, chunk_dim, adj, rank, world_size)
            local_out = par(local_x)

            _gather_and_check(
                local_out,
                ref,
                chunk_dim,
                world_size,
                rank,
            )


def _logic_halo_conv2d_stride2(rank: int, world_size: int) -> None:
    """HaloExchangeConv2dStride2 wrapping nn.Conv2d (kernel=3, stride=2)."""
    device = f"cuda:{rank}"
    adj = _make_adj_groups(world_size)

    conv = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0).to(device).float()
    _broadcast_params(conv)

    pad = nn.ZeroPad2d((0, 1, 0, 1))

    for chunk_dim in [2, 3]:
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            x, local_x = _prepare(
                rank,
                world_size,
                chunk_dim,
                (4, 96, 64, 48),
                device,
                memory_format,
            )
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

            _gather_and_check(
                local_out,
                ref,
                chunk_dim,
                world_size,
                rank,
            )


def _logic_halo_conv2d_stride2_offset_group(rank: int, world_size: int) -> None:
    """Stride-2 halo works when VAE-local ranks differ from global ranks."""
    assert world_size == 4
    vae_ranks = [2, 3]
    vae_group = dist.new_group(vae_ranks)
    adjacent_group = dist.new_group(vae_ranks)
    if rank not in vae_ranks:
        dist.barrier()
        return

    local_rank = vae_ranks.index(rank)
    device = f"cuda:{rank}"
    conv = nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0).to(device).float()
    for parameter in conv.parameters():
        dist.broadcast(parameter.data, src=vae_ranks[0], group=vae_group)

    x = torch.randn((4, 96, 64, 48), dtype=torch.float32, device=device)
    dist.broadcast(x, src=vae_ranks[0], group=vae_group)
    reference = conv(nn.ZeroPad2d((0, 1, 0, 1))(x)).detach()
    local_x = x.chunk(len(vae_ranks), dim=3)[local_rank]
    parallel = HaloExchangeConv2dStride2(
        conv,
        chunk_dim=3,
        adj_groups=[adjacent_group],
        rank=local_rank,
        world_size=len(vae_ranks),
        pad_before_conv=(0, 1, 0, 1),
    )
    local_output = parallel(local_x).contiguous()
    gathered = [torch.empty_like(local_output) for _ in vae_ranks]
    dist.all_gather(gathered, local_output, group=vae_group)
    output = torch.cat(gathered, dim=3)
    assert torch.max(torch.abs(output - reference)).item() < 0.01
    dist.barrier()


class TestHaloExchangeConv:
    def test_wan_conv3d_with_cache_2gpu(self):
        _run(2, _logic_halo_conv3d)

    def test_conv3d_even_kernel_channels_last_2gpu(self) -> None:
        _run(2, _logic_halo_conv3d_even_kernel)

    def test_conv2d_2gpu(self):
        _run(2, _logic_halo_conv2d)


class TestHaloExchangeConv2dStride2:
    def test_conv2d_stride2_2gpu(self):
        _run(2, _logic_halo_conv2d_stride2)

    def test_conv2d_stride2_offset_group_4gpu(self) -> None:
        _run(4, _logic_halo_conv2d_stride2_offset_group)


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
class TestHaloExchangeValidation:
    def test_missing_required_adjacent_group_fails_at_construction(self) -> None:
        conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        with pytest.raises(ValueError, match="missing VAE adjacent process group 0"):
            HaloExchangeConv(conv, chunk_dim=3, adj_groups=[None], rank=0, world_size=2)

    def test_kernel_one_does_not_require_adjacent_group(self) -> None:
        conv = nn.Conv2d(4, 4, kernel_size=(3, 1))

        HaloExchangeConv(conv, chunk_dim=3, adj_groups=[None], rank=0, world_size=2)


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Required modules not available")
class TestSpatialChannelsLastFormat:
    """CPU unit tests for the halo channels-last layout helper.

    ``_spatial_channels_last_format`` is what lets the halo ``cat`` preserve
    channels-last (so downstream convs skip a full-tensor re-conversion). It runs
    on CPU tensors, so no GPU/distributed setup is needed here.
    """

    def test_channels_last_3d_detected(self):
        x = torch.randn(1, 4, 3, 8, 8).contiguous(memory_format=torch.channels_last_3d)
        assert _spatial_channels_last_format(x) is torch.channels_last_3d

    def test_channels_last_2d_detected(self):
        x = torch.randn(2, 4, 8, 8).contiguous(memory_format=torch.channels_last)
        assert _spatial_channels_last_format(x) is torch.channels_last

    def test_row_major_returns_none(self):
        assert _spatial_channels_last_format(torch.randn(1, 4, 3, 8, 8)) is None
        assert _spatial_channels_last_format(torch.randn(2, 4, 8, 8)) is None

    def test_cat_preserves_channels_last_when_halos_match(self):
        # The core invariant: cat of same-format channels-last slices stays
        # channels-last (a mixed-format cat would fall back to row-major).
        x = torch.randn(1, 4, 3, 8, 8).contiguous(memory_format=torch.channels_last_3d)
        halo = torch.zeros(1, 4, 3, 1, 8).contiguous(memory_format=torch.channels_last_3d)
        out = torch.cat([halo, x, halo], dim=3)
        assert out.is_contiguous(memory_format=torch.channels_last_3d)

    @pytest.mark.parametrize("dim", [3, 4])
    def test_physical_layout_cat_preserves_channels_last_3d(self, dim: int) -> None:
        x = torch.randn(1, 4, 3, 8, 8).contiguous(memory_format=torch.channels_last_3d)
        halo_shape = list(x.shape)
        halo_shape[dim] = 1
        halo = torch.randn(halo_shape).contiguous(memory_format=torch.channels_last_3d)

        actual = _cat_spatial_halos([halo, x, halo], dim, torch.channels_last_3d)
        expected = torch.cat([halo, x, halo], dim=dim)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.is_contiguous(memory_format=torch.channels_last_3d)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_physical_layout_cat_preserves_channels_last_2d(self, dim: int) -> None:
        x = torch.randn(1, 4, 8, 8).contiguous(memory_format=torch.channels_last)
        halo_shape = list(x.shape)
        halo_shape[dim] = 1
        halo = torch.randn(halo_shape).contiguous(memory_format=torch.channels_last)

        actual = _cat_spatial_halos([halo, x, halo], dim, torch.channels_last)
        expected = torch.cat([halo, x, halo], dim=dim)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.is_contiguous(memory_format=torch.channels_last)

    @pytest.mark.parametrize(
        ("shape", "dim"),
        [
            ((2, 4, 8, 9), 3),
            ((1, 4, 3, 8, 9), 4),
        ],
    )
    def test_channels_last_halo_buffer_round_trip(
        self,
        shape: tuple[int, ...],
        dim: int,
    ) -> None:
        x = torch.randn(shape)
        expected = torch.narrow(x, dim, shape[dim] - 1, 1)

        buffer = _halo_exchange_buffer(
            x,
            dim,
            shape[dim] - 1,
            1,
            memory_format=(torch.channels_last_3d if len(shape) == 5 else torch.channels_last),
        )
        actual = _physical_to_logical_channels_last(buffer)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert buffer.is_contiguous()
        expected_format = torch.channels_last_3d if len(shape) == 5 else torch.channels_last
        assert actual.is_contiguous(memory_format=expected_format)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
