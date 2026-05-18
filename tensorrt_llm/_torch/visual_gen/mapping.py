"""Unified multi-dimensional communicator mesh for visual generation models.

VisualGenMapping subclasses DeviceMeshTopologyImpl and overrides build_mesh()
to create a single PyTorch DeviceMesh covering all parallelism axes
(CFG, TP, CP, Ulysses).  The resulting mesh is stored in the shared
DeviceMeshTopologyImpl.device_mesh class variable so that any Mapping object
constructed afterward (e.g. via to_llm_mapping()) can reuse the same
process groups.
"""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import init_device_mesh

from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl, SingleProcessGroup
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_VALID_DIM_NAMES = frozenset({"cfg", "tp", "cp", "ulysses"})
DEFAULT_DIM_ORDER = "cfg-tp-cp-ulysses"


class VisualGenMapping(DeviceMeshTopologyImpl):
    """Multi-dimensional communicator mesh for visual generation models.

    Parallelism Strategy:
    - CFG Parallelism: Distributes positive/negative prompts across GPUs
    - CP (Context Parallelism): Ring attention or Attention2D (sequence sharding)
    - Ulysses Parallelism: Head sharding within each CFG group

    Parallelism Hierarchy:
        total_workers = cfg × sp
        sp (sequence parallelism) = cp × ulysses  [mutually exclusive today; TODO to combine]
        cp (context parallelism)  = ring           [ring attention, not yet implemented]
                                  | attn2d         [Attention2D 2D mesh, row_size × col_size]

        cfg:     Splits positive/negative CFG prompts across GPUs (independent streams).
        tp:      Tensor parallelism all-reduce within tp groups.
        sp:      Total sequence-axis parallel degree = cp × ulysses.
          cp:    Shards the sequence dimension across ranks.
            ring:   Passes KV blocks in a ring; ranks form a 1D cp group.
            attn2d: 2D mesh; Q all-gathered within row group, K/V within col group.
          ulysses: Shards heads via all-to-all (head-sharding, not sequence-sharding).

    Ordering rationale (default ``"cfg-tp-cp-ulysses"``):
    - Ulysses innermost: all-to-all is latency-sensitive, contiguous ranks
    - CP next: KV streaming (ring) or sequence shard communication (Attention2D)
    - TP next: all-reduce for Linear
    - CFG outermost: independent until final all-gather

    The *order* string maps directly to ``init_device_mesh``'s
    ``mesh_shape`` tuple (first = outermost / slowest-varying, last =
    innermost / most contiguous).
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        cfg_size: int = 1,
        tp_size: int = 1,
        ring_size: int = 1,
        ulysses_size: int = 1,
        attn2d_row_size: int = 1,
        attn2d_col_size: int = 1,
        parallel_vae_size: int = 1,
        order: str = DEFAULT_DIM_ORDER,
    ):
        # cp_size unifies ring and Attention2D under one context-parallelism mesh dimension.
        # Ring and Attention2D are mutually exclusive: both shard the sequence axis.
        attn2d_size = attn2d_row_size * attn2d_col_size
        if ring_size > 1 and attn2d_size > 1:
            raise ValueError(
                "Ring and Attention2D are mutually exclusive: both shard the sequence "
                f"dimension. Got ring_size={ring_size}, attn2d={attn2d_row_size}x{attn2d_col_size}."
            )
        cp_size = attn2d_size if attn2d_size > 1 else ring_size
        if cp_size > 1 and ulysses_size > 1:
            raise NotImplementedError(
                "Combining CP and Ulysses is not yet supported. "
                "They are orthogonal (CP shards sequence; Ulysses shards heads) "
                "but the combined wrapper is not implemented."
            )
        if attn2d_size > 1 and tp_size > 1:
            raise NotImplementedError(
                "Combining Attention2D and TP is not yet supported. "
                "The row/col group construction in _build_attn2d_groups does not account "
                "for TP ranks."
            )
        product = cfg_size * tp_size * cp_size * ulysses_size
        if product != world_size:
            raise ValueError(
                f"cfg({cfg_size}) * tp({tp_size}) * cp({cp_size}) * "
                f"ulysses({ulysses_size}) = {product} != world_size({world_size})"
            )
        if parallel_vae_size < 1:
            raise ValueError(f"parallel_vae_size ({parallel_vae_size}) must be >= 1")
        if parallel_vae_size > world_size:
            raise ValueError(
                f"parallel_vae_size ({parallel_vae_size}) cannot exceed world_size ({world_size})"
            )

        dims = order.split("-")
        if set(dims) != _VALID_DIM_NAMES or len(dims) != len(_VALID_DIM_NAMES):
            raise ValueError(
                f"order must be a '-'-separated permutation of "
                f"{sorted(_VALID_DIM_NAMES)}, got '{order}'"
            )

        self.world_size = world_size
        self._rank = rank
        self.cfg_size = cfg_size
        self.tp_size = tp_size
        self.ring_size = ring_size
        self.cp_size = cp_size
        self.ulysses_size = ulysses_size
        self.attn2d_row_size = attn2d_row_size
        self.attn2d_col_size = attn2d_col_size
        self.parallel_vae_size = parallel_vae_size
        self._vae_ranks = list(range(self.parallel_vae_size))
        self._vae_group: Optional[ProcessGroup] = None
        self._vae_adj_groups: list[Optional[ProcessGroup]] = []
        self._attn2d_row_group: Optional[ProcessGroup] = None
        self._attn2d_col_group: Optional[ProcessGroup] = None
        self._order = order
        self._dim_names = tuple(dims)
        # cp_size covers both ring (1D) and Attention2D (2D, row_size * col_size).
        # For Attention2D, _build_attn2d_groups() creates row/col sub-groups;
        # the full CP group is already provided by the "cp" mesh dimension.
        self._dim_sizes = {
            "cfg": cfg_size,
            "tp": tp_size,
            "cp": cp_size,
            "ulysses": ulysses_size,
        }

        if dist.is_initialized() and world_size > 1:
            self.build_mesh()
            self._build_vae_group()

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------
    def build_mesh(self):
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is not None:
            return

        shape = tuple(self._dim_sizes[d] for d in self._dim_names)
        cls.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=shape,
            mesh_dim_names=self._dim_names,
        )
        logger.debug(
            f"VisualGenMapping.build_mesh: dims={self._dim_names}, "
            f"shape={shape}, mesh={cls.device_mesh}"
        )

        if self.attn2d_row_size * self.attn2d_col_size > 1:
            self._build_attn2d_groups()

    def _build_attn2d_groups(self) -> None:
        """Create row and col process groups for Attention2D parallelism.

        Within each CFG group, rank (r, c) has local index r * col_size + c:
          - row_process_group: all ranks with same r (Q all-gather + output reduce-scatter)
          - col_process_group: all ranks with same c (K/V all-gather)

        The full CP group (all row_size * col_size ranks per CFG group) is
        provided by the ``"cp"`` mesh dimension and does not need a separate
        new_group() call — use ``cp_group`` / ``attn2d_mesh_group`` for it.

        Stores the row/col groups for this rank on self._attn2d_{row,col}_group.
        """
        row_size = self.attn2d_row_size
        col_size = self.attn2d_col_size
        mesh_size = row_size * col_size
        rank = self._rank

        row_pg: Optional[ProcessGroup] = None
        col_pg: Optional[ProcessGroup] = None

        for cfg_id in range(self.cfg_size):
            base = cfg_id * mesh_size

            # Row groups: same row-index r, varying col-index c
            for r in range(row_size):
                ranks = [base + r * col_size + c for c in range(col_size)]
                pg = dist.new_group(ranks, use_local_synchronization=True)
                if rank in ranks:
                    row_pg = pg

            # Col groups: varying row-index r, same col-index c
            for c in range(col_size):
                ranks = [base + r * col_size + c for r in range(row_size)]
                pg = dist.new_group(ranks, use_local_synchronization=True)
                if rank in ranks:
                    col_pg = pg

        self._attn2d_row_group = row_pg
        self._attn2d_col_group = col_pg
        logger.debug(
            f"VisualGenMapping._build_attn2d_groups: row_size={row_size}, col_size={col_size}"
        )

    def _validate_vae_ranks_share_cfg_group(self) -> None:
        """Ensure all ``vae_ranks`` share one cfg coordinate (or span the full world).

        Today ``_vae_ranks = list(range(parallel_vae_size))`` only sits inside
        one CFG group when ``cfg`` is the outermost axis of ``dit_dim_order``.
        """
        if self.parallel_vae_size <= 1 or self.cfg_size <= 1:
            return
        if self.parallel_vae_size == self.world_size:
            return

        # Row-major stride along the cfg axis from the mesh order.
        stride, cfg_stride = 1, None
        for dim in reversed(self._dim_names):
            if dim == "cfg":
                cfg_stride = stride
                break
            stride *= self._dim_sizes[dim]
        assert cfg_stride is not None  # 'cfg' is always present in _dim_names

        cfg_coords = {(r // cfg_stride) % self.cfg_size for r in self._vae_ranks}
        if len(cfg_coords) > 1:
            raise NotImplementedError(
                f"vae_ranks={self._vae_ranks} straddle CFG groups "
                f"(cfg coordinates={sorted(cfg_coords)}) under order='{self._order}'. "
                "VAE ranks must share a single CFG group, or span the full world. "
                "_vae_ranks is currently hardcoded to list(range(parallel_vae_size)), "
                "which only lands in one CFG group when 'cfg' is the outermost axis "
                "of dit_dim_order. Either pick parallel_vae_size <= ranks_per_cfg_group "
                "or derive _vae_ranks from the mesh."
            )

    def _build_vae_group(self) -> None:
        """Create the process group used by parallel VAE."""
        if self.parallel_vae_size <= 1:
            return

        self._validate_vae_ranks_share_cfg_group()

        # use_local_synchronization=False since new_group is world-collective, so every
        # rank (including non-VAE ranks) must participate or the next world-wide
        # collective deadlocks.
        pg = dist.new_group(self._vae_ranks, use_local_synchronization=False)
        if self._rank in self._vae_ranks:
            self._vae_group = pg

        adj_groups: list[Optional[ProcessGroup]] = [None] * (self.parallel_vae_size - 1)
        for i in range(self.parallel_vae_size - 1):
            ranks = [self._vae_ranks[i], self._vae_ranks[i + 1]]
            pg = dist.new_group(ranks, use_local_synchronization=False)
            if self._rank in ranks:
                adj_groups[i] = pg
        if self._rank in self._vae_ranks:
            self._vae_adj_groups = adj_groups

    # ------------------------------------------------------------------
    # Rank decomposition
    # ------------------------------------------------------------------
    def _local_rank(self, dim: str) -> int:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            return 0
        return cls.device_mesh[dim].get_local_rank()

    @property
    def cfg_rank(self) -> int:
        return self._local_rank("cfg")

    @property
    def tp_rank(self) -> int:
        return self._local_rank("tp")

    @property
    def cp_rank(self) -> int:
        return self._local_rank("cp")

    @property
    def ulysses_rank(self) -> int:
        return self._local_rank("ulysses")

    @property
    def attn2d_mesh_rank(self) -> int:
        """Rank within the Attention2D CP group (same as cp_rank)."""
        return self._local_rank("cp")

    @property
    def is_cfg_conditional(self) -> bool:
        return self.cfg_rank == 0

    # ------------------------------------------------------------------
    # Process groups (None when size == 1 and mesh was not built)
    # ------------------------------------------------------------------
    def _group(self, dim: str) -> Optional[ProcessGroup]:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            if self.world_size == 1:
                return SingleProcessGroup.get_group()
            return None
        return cls.device_mesh[dim].get_group()

    @property
    def ulysses_group(self) -> Optional[ProcessGroup]:
        return self._group("ulysses")

    @property
    def cp_group(self) -> Optional[ProcessGroup]:
        return self._group("cp")

    @property
    def tp_group_pg(self) -> Optional[ProcessGroup]:
        return self._group("tp")

    @property
    def cfg_group(self) -> Optional[ProcessGroup]:
        return self._group("cfg")

    @property
    def attn2d_row_group(self) -> Optional[ProcessGroup]:
        return self._attn2d_row_group

    @property
    def attn2d_col_group(self) -> Optional[ProcessGroup]:
        return self._attn2d_col_group

    @property
    def attn2d_mesh_group(self) -> Optional[ProcessGroup]:
        """Full CP group for Attention2D (same as cp_group)."""
        return self._group("cp")

    @property
    def vae_ranks(self) -> list[int]:
        return self._vae_ranks

    @property
    def vae_group(self) -> Optional[ProcessGroup]:
        return self._vae_group

    @property
    def vae_adj_groups(self) -> list[Optional[ProcessGroup]]:
        return self._vae_adj_groups

    # ------------------------------------------------------------------
    # Bridge to LLM Mapping (for Linear layers)
    # ------------------------------------------------------------------
    def to_llm_mapping(self) -> Mapping:
        """Return a ``Mapping`` whose TP group is backed by this mesh's TP dim.

        ``build_mesh()`` has already populated
        ``DeviceMeshTopologyImpl.device_mesh``, so the returned ``Mapping``'s
        ``build_mesh()`` is a no-op and ``tp_group_pg`` reads from the shared
        mega-mesh.
        """
        return Mapping(
            world_size=self.tp_size,
            rank=self.tp_rank,
            tp_size=self.tp_size,
        )
