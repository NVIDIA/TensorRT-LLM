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
        cp_size: int = 1,
        ulysses_size: int = 1,
        attn2d_row_size: int = 1,
        attn2d_col_size: int = 1,
        order: str = DEFAULT_DIM_ORDER,
    ):
        attn2d_size = attn2d_row_size * attn2d_col_size
        if attn2d_size > 1 and cp_size != attn2d_size:
            raise ValueError(
                f"cp_size ({cp_size}) must equal attn2d_row_size * attn2d_col_size "
                f"({attn2d_row_size} * {attn2d_col_size} = {attn2d_size}) "
                "when Attention2D is enabled."
            )
        if attn2d_size > 1 and ulysses_size > 1:
            raise NotImplementedError(
                "Combining Attention2D and Ulysses is not yet supported. "
                "They are orthogonal (Attention2D shards sequence; Ulysses shards heads) "
                "but the combined wrapper is not implemented."
            )
        product = cfg_size * tp_size * cp_size * ulysses_size
        if product != world_size:
            raise ValueError(
                f"cfg({cfg_size}) * tp({tp_size}) * cp({cp_size}) * "
                f"ulysses({ulysses_size}) = {product} != world_size({world_size})"
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
        self.cp_size = cp_size
        self.ulysses_size = ulysses_size
        self.attn2d_row_size = attn2d_row_size
        self.attn2d_col_size = attn2d_col_size
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
