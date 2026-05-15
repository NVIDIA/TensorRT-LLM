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
        assert vgm.cp_size == 1
        assert vgm.ulysses_size == 1
        assert vgm.parallel_vae_size == 1
        assert vgm.vae_ranks == [0]
        assert vgm.vae_group is None
        assert vgm.vae_adj_groups == []

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
        assert vgm.cp_size == 1
        assert vgm.ulysses_size == 2
        assert vgm.world_size == 8

    def test_stores_parallel_vae_ranks(self):
        vgm = VisualGenMapping(
            world_size=4,
            rank=0,
            ulysses_size=4,
            parallel_vae_size=2,
        )
        assert vgm.parallel_vae_size == 2
        assert vgm.vae_ranks == [0, 1]
        assert vgm.vae_group is None
        assert vgm.vae_adj_groups == []

    def test_stores_attn2d_sizes(self):
        vgm = VisualGenMapping(
            world_size=4,
            rank=0,
            attn2d_row_size=2,
            attn2d_col_size=2,
        )
        assert vgm.attn2d_row_size == 2
        assert vgm.attn2d_col_size == 2
        assert vgm.cp_size == 4

    def test_product_mismatch_raises(self):
        with pytest.raises(ValueError, match="!= world_size"):
            VisualGenMapping(world_size=4, rank=0, cfg_size=2, ulysses_size=3)

    def test_parallel_vae_size_cannot_exceed_world_size(self):
        with pytest.raises(ValueError, match="cannot exceed world_size"):
            VisualGenMapping(world_size=1, rank=0, parallel_vae_size=2)

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            VisualGenMapping(world_size=1, rank=0, order="cfg-tp-ulysses")

    def test_duplicate_dim_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            VisualGenMapping(world_size=1, rank=0, order="cfg-cfg-tp-ulysses")

    def test_custom_order_stored(self):
        vgm = VisualGenMapping(world_size=1, rank=0, order="ulysses-cp-tp-cfg")
        assert vgm._dim_names == ("ulysses", "cp", "tp", "cfg")

    def test_all_valid_orders(self):
        for perm in itertools.permutations(sorted(_VALID_DIM_NAMES)):
            order = "-".join(perm)
            vgm = VisualGenMapping(world_size=1, rank=0, order=order)
            assert vgm._dim_names == perm

    def test_ulysses_and_attn2d_raises(self):
        """Combining Attention2D and Ulysses raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet supported"):
            VisualGenMapping(
                world_size=4,
                rank=0,
                ulysses_size=2,
                attn2d_row_size=2,
                attn2d_col_size=1,
            )

    def test_ring_and_attn2d_raises(self):
        """Combining ring and Attention2D raises ValueError (both shard the sequence axis)."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            VisualGenMapping(
                world_size=8,
                rank=0,
                ring_size=2,
                attn2d_row_size=2,
                attn2d_col_size=2,
            )


class TestSingleGPURanksAndGroups:
    def test_ranks_are_zero(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.cfg_rank == 0
        assert vgm.tp_rank == 0
        assert vgm.cp_rank == 0
        assert vgm.ulysses_rank == 0
        assert vgm.attn2d_mesh_rank == 0

    def test_attn2d_mesh_rank_aliases_cp_rank(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.attn2d_mesh_rank == vgm.cp_rank

    def test_is_cfg_conditional(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.is_cfg_conditional is True

    def test_groups_return_single_process_group(self):
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.ulysses_group is not None
        assert vgm.cp_group is not None
        assert vgm.tp_group_pg is not None
        assert vgm.cfg_group is not None

    def test_attn2d_row_col_groups_none_without_attn2d(self):
        """Row/col groups are None when Attention2D is not active."""
        vgm = VisualGenMapping(world_size=1, rank=0)
        assert vgm.attn2d_row_group is None
        assert vgm.attn2d_col_group is None


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
    """Default order cfg-tp-cp-ulysses with cfg=2, ulysses=2 on 4 GPUs.

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
    assert vgm.cp_rank == 0
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
    """Custom order ulysses-cp-tp-cfg with cfg=2, ulysses=2 on 4 GPUs.

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
        order="ulysses-cp-tp-cfg",
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

    Default order cfg-tp-cp-ulysses with cfg=1, tp=2, cp=1, ulysses=2:
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


def _logic_attn2d_mesh_rank_and_group(rank, world_size):
    """attn2d_mesh_rank aliases cp_rank and attn2d_mesh_group works for collectives (2x2 mesh).

    Default order cfg-tp-cp-ulysses with attn2d 2x2 (cp_size=4), cfg=1, tp=1, ulysses=1:
        Rank 0: cp=0  (attn2d_mesh_rank=0)
        Rank 1: cp=1  (attn2d_mesh_rank=1)
        Rank 2: cp=2  (attn2d_mesh_rank=2)
        Rank 3: cp=3  (attn2d_mesh_rank=3)
    All 4 ranks are in a single CP group covering the full attn2d mesh.
    """
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    DeviceMeshTopologyImpl.device_mesh = None

    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        attn2d_row_size=2,
        attn2d_col_size=2,
    )

    # attn2d_mesh_rank must equal cp_rank on every rank
    assert vgm.attn2d_mesh_rank == vgm.cp_rank, (
        f"Rank {rank}: attn2d_mesh_rank={vgm.attn2d_mesh_rank} != cp_rank={vgm.cp_rank}"
    )
    assert vgm.cp_rank == rank, f"Rank {rank}: expected cp_rank={rank}, got {vgm.cp_rank}"

    # attn2d_mesh_group must span all 4 ranks (full CP group)
    assert vgm.attn2d_mesh_group is not None
    cp_pg_size = dist.get_world_size(vgm.attn2d_mesh_group)
    assert cp_pg_size == 4, f"Rank {rank}: expected cp group size 4, got {cp_pg_size}"

    # Verify attn2d_mesh_group works for actual collective ops
    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(1, device=device)
    dist.all_reduce(tensor, group=vgm.attn2d_mesh_group)
    assert tensor.item() == float(world_size), (
        f"Rank {rank}: expected all_reduce sum {world_size}, got {tensor.item()}"
    )


def _logic_vae_group_and_adj_groups(rank, world_size):
    """VAE group uses the first N ranks and stores adjacent groups by pair index."""
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    DeviceMeshTopologyImpl.device_mesh = None

    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        ulysses_size=world_size,
        parallel_vae_size=3,
    )

    assert vgm.vae_ranks == [0, 1, 2]
    if rank < 3:
        assert vgm.vae_group is not None
        assert dist.get_world_size(vgm.vae_group) == 3
        assert len(vgm.vae_adj_groups) == 2
        for i, adj_group in enumerate(vgm.vae_adj_groups):
            if rank in (i, i + 1):
                assert adj_group is not None
                assert dist.get_world_size(adj_group) == 2
            else:
                assert adj_group is None
    else:
        assert vgm.vae_group is None
        assert vgm.vae_adj_groups == []


def _logic_vae_group_full_world_cfg2_ulysses2(rank, world_size):
    """Regression: full-world VAE group works for cfg=2, ulysses=2."""
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl

    DeviceMeshTopologyImpl.device_mesh = None

    vgm = VisualGenMapping(
        world_size=world_size,
        rank=rank,
        cfg_size=2,
        ulysses_size=2,
        parallel_vae_size=4,
    )

    assert vgm.vae_ranks == [0, 1, 2, 3]
    assert vgm.vae_group is not None
    assert dist.get_world_size(vgm.vae_group) == 4
    assert len(vgm.vae_adj_groups) == 3

    for i, adj_group in enumerate(vgm.vae_adj_groups):
        if rank in (i, i + 1):
            assert adj_group is not None
            assert dist.get_world_size(adj_group) == 2
        else:
            assert adj_group is None

    device = torch.device(f"cuda:{rank}")
    tensor = torch.ones(1, device=device)
    dist.all_reduce(tensor, group=vgm.vae_group)
    assert tensor.item() == float(world_size), (
        f"Rank {rank}: expected all_reduce sum {world_size}, got {tensor.item()}"
    )


@pytest.mark.skipif(not MODULES_AVAILABLE, reason="Modules not available")
class TestMultiGPU:
    def test_default_order_cfg2_ulysses2(self):
        _run_multi_gpu(4, _logic_default_order_cfg2_ulysses2)

    def test_custom_order_ulysses_outermost(self):
        _run_multi_gpu(4, _logic_custom_order_ulysses_outermost)

    def test_allreduce_over_tp_group(self):
        _run_multi_gpu(4, _logic_allreduce_over_tp_group)

    def test_attn2d_mesh_rank_and_group(self):
        """attn2d_mesh_rank aliases cp_rank and attn2d_mesh_group supports collectives."""
        _run_multi_gpu(4, _logic_attn2d_mesh_rank_and_group)

    def test_vae_group_and_adj_groups(self):
        _run_multi_gpu(4, _logic_vae_group_and_adj_groups)

    def test_vae_group_full_world_cfg2_ulysses2(self):
        _run_multi_gpu(4, _logic_vae_group_full_world_cfg2_ulysses2)
