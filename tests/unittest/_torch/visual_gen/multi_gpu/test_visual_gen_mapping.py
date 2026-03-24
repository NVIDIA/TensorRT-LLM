"""Tests for VisualGenMapping — the unified multi-dimensional communicator mesh.

Single-GPU tests run without dist. Multi-GPU tests use mp.spawn with NCCL.
"""

import itertools
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from tensorrt_llm._torch.visual_gen.mapping import _VALID_DIM_NAMES, VisualGenMapping
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Helpers for multi-GPU tests
# =============================================================================


def _init_dist(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _worker(rank, world_size, test_fn, port):
    try:
        _init_dist(rank, world_size, port)
        test_fn(rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_multi_gpu(world_size, test_fn):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs, have {torch.cuda.device_count()}")
    port = get_free_port()
    mp.spawn(_worker, args=(world_size, test_fn, port), nprocs=world_size, join=True)


# =============================================================================
# Single-GPU tests (no dist required)
# =============================================================================


class TestConstruction:
    def test_single_gpu_defaults(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.world_size == 1
        assert vgm.cfg_size == 1
        assert vgm.tp_size == 1
        assert vgm.ring_size == 1
        assert vgm.ulysses_size == 1

    def test_stores_sizes(self):
        vgm = VisualGenMapping(
            world_size=8,
            rank=0,
            cfg_size=2,
            tp_size=2,
            ulysses_size=2,
        )
        assert vgm.cfg_size == 2
        assert vgm.tp_size == 2
        assert vgm.ring_size == 1
        assert vgm.ulysses_size == 2
        assert vgm.world_size == 8

    def test_product_mismatch_raises(self):
        with pytest.raises(ValueError, match="!= world_size"):
            VisualGenMapping(world_size=4, rank=0, cfg_size=2, ulysses_size=3)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            VisualGenMapping(world_size=1, rank=0, order="cfg-tp-ulysses")

    def test_duplicate_dim_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            VisualGenMapping(world_size=1, rank=0, order="cfg-cfg-tp-ulysses")

    def test_custom_order_stored(self):
        vgm = VisualGenMapping(world_size=1, rank=0, order="ulysses-ring-tp-cfg")
        assert vgm._dim_names == ("ulysses", "ring", "tp", "cfg")

    def test_all_valid_orders(self):
        for perm in itertools.permutations(sorted(_VALID_DIM_NAMES)):
            order = "-".join(perm)
            vgm = VisualGenMapping(world_size=1, rank=0, order=order)
            assert vgm._dim_names == perm


class TestSingleGPURanksAndGroups:
    def test_ranks_are_zero(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.cfg_rank == 0
        assert vgm.tp_rank == 0
        assert vgm.ring_rank == 0
        assert vgm.ulysses_rank == 0

    def test_is_cfg_conditional(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.is_cfg_conditional is True

    def test_groups_return_single_process_group(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.ulysses_group is not None
        assert vgm.ring_group is not None
        assert vgm.tp_group_pg is not None
        assert vgm.cfg_group is not None


class TestToLlmMapping:
    def test_single_gpu(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        m = vgm.to_llm_mapping()
        assert m.tp_size == 1
        assert m.world_size == 1

    def test_tp_size_propagated(self):
        vgm = VisualGenMapping(world_size=4, rank=0, tp_size=4)
        m = vgm.to_llm_mapping()
        assert m.tp_size == 4

    def test_mixed_parallelism(self):
        vgm = VisualGenMapping(
            world_size=8,
            rank=0,
            cfg_size=2,
            tp_size=2,
            ulysses_size=2,
        )
        m = vgm.to_llm_mapping()
        assert m.tp_size == 2
        assert m.world_size == 2


# =============================================================================
# Multi-GPU tests — validate actual DeviceMesh groups and ranks
# =============================================================================


def _logic_default_order_cfg2_ulysses2(rank, world_size):
    """Default order cfg-tp-ring-ulysses with cfg=2, ulysses=2 on 4 GPUs.

    Expected rank layout (outermost=cfg, innermost=ulysses):
        Rank 0: cfg=0, ulysses=0  (conditional, ulysses group 0)
        Rank 1: cfg=0, ulysses=1  (conditional, ulysses group 1)
        Rank 2: cfg=1, ulysses=0  (unconditional, ulysses group 0)
        Rank 3: cfg=1, ulysses=1  (unconditional, ulysses group 1)
    """
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    DeviceMeshTopologyImpl.device_mesh = None

    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        cfg_size=2,
        ulysses_size=2,
    )

    assert vgm.cfg_rank == rank // 2
    assert vgm.ulysses_rank == rank % 2
    assert vgm.tp_rank == 0
    assert vgm.ring_rank == 0
    assert vgm.is_cfg_conditional == (rank < 2)

    assert vgm.cfg_group is not None
    assert vgm.ulysses_group is not None

    cfg_pg_size = dist.get_world_size(vgm.cfg_group)
    ulysses_pg_size = dist.get_world_size(vgm.ulysses_group)
    assert cfg_pg_size == 2
    assert ulysses_pg_size == 2

    m = vgm.to_llm_mapping()
    assert m.tp_size == 1


def _logic_custom_order_ulysses_outermost(rank, world_size):
    """Custom order ulysses-ring-tp-cfg with cfg=2, ulysses=2 on 4 GPUs.

    Expected rank layout (outermost=ulysses, innermost=cfg):
        Rank 0: ulysses=0, cfg=0
        Rank 1: ulysses=0, cfg=1
        Rank 2: ulysses=1, cfg=0
        Rank 3: ulysses=1, cfg=1
    """
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    DeviceMeshTopologyImpl.device_mesh = None

    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        cfg_size=2,
        ulysses_size=2,
        order="ulysses-ring-tp-cfg",
    )

    assert vgm.ulysses_rank == rank // 2
    assert vgm.cfg_rank == rank % 2
    assert vgm.is_cfg_conditional == (rank % 2 == 0)

    cfg_pg_size = dist.get_world_size(vgm.cfg_group)
    ulysses_pg_size = dist.get_world_size(vgm.ulysses_group)
    assert cfg_pg_size == 2
    assert ulysses_pg_size == 2


def _logic_allreduce_over_tp_group(rank, world_size):
    """Verify TP group works for collective ops (tp=2, ulysses=2 on 4 GPUs).

    Default order cfg-tp-ring-ulysses with cfg=1, tp=2, ring=1, ulysses=2:
        Rank 0: tp=0, ulysses=0
        Rank 1: tp=0, ulysses=1
        Rank 2: tp=1, ulysses=0
        Rank 3: tp=1, ulysses=1
    TP groups: {0, 2} and {1, 3}.
    """
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    DeviceMeshTopologyImpl.device_mesh = None

    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        tp_size=2,
        ulysses_size=2,
    )

    device = torch.device(f"cuda:{rank}")

    # Each rank contributes 1.0; after all_reduce(sum) over tp_size=2, expect 2.0
    tensor = torch.ones(1, device=device)
    dist.all_reduce(tensor, group=vgm.tp_group_pg)
    assert tensor.item() == float(vgm.tp_size), (
        f"Rank {rank}: expected {vgm.tp_size}, got {tensor.item()}"
    )

    # Also verify ulysses group with the same pattern
    tensor2 = torch.ones(1, device=device)
    dist.all_reduce(tensor2, group=vgm.ulysses_group)
    assert tensor2.item() == float(vgm.ulysses_size), (
        f"Rank {rank}: expected {vgm.ulysses_size}, got {tensor2.item()}"
    )


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Modules not available")
class TestMultiGPU:
    def test_default_order_cfg2_ulysses2(self):
        _run_multi_gpu(4, _logic_default_order_cfg2_ulysses2)

    def test_custom_order_ulysses_outermost(self):
        _run_multi_gpu(4, _logic_custom_order_ulysses_outermost)

    def test_allreduce_over_tp_group(self):
        _run_multi_gpu(4, _logic_allreduce_over_tp_group)
