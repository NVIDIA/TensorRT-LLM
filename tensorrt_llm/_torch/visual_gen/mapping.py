"""Unified multi-dimensional communicator mesh for visual generation models.

VisualGenMapping subclasses DeviceMeshTopologyImpl and overrides build_mesh()
to create a single PyTorch DeviceMesh covering all parallelism axes
(CFG, TP, Ring, Ulysses).  The resulting mesh is stored in the shared
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

_VALID_DIM_NAMES = frozenset({"cfg", "tp", "ring", "ulysses"})
DEFAULT_DIM_ORDER = "cfg-tp-ring-ulysses"


class VisualGenMapping(DeviceMeshTopologyImpl):
    """Multi-dimensional communicator mesh for visual generation models.

    Parallelism Strategy:
    - CFG Parallelism: Distributes positive/negative prompts across GPUs
    - Ulysses Parallelism: Distributes sequence within each CFG group

    Ordering rationale (default ``"cfg-tp-ring-ulysses"``):
    - Ulysses innermost: all-to-all is latency-sensitive, contiguous ranks
    - Ring next: KV streaming between adjacent ranks
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
        order: str = DEFAULT_DIM_ORDER,
    ):
        product = cfg_size * tp_size * ring_size * ulysses_size
        if product != world_size:
            raise ValueError(
                f"cfg({cfg_size}) * tp({tp_size}) * ring({ring_size}) * "
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
        self.ring_size = ring_size
        self.ulysses_size = ulysses_size
        self._order = order
        self._dim_names = tuple(dims)
        self._dim_sizes = {
            "cfg": cfg_size,
            "tp": tp_size,
            "ring": ring_size,
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
    def ring_rank(self) -> int:
        return self._local_rank("ring")

    @property
    def ulysses_rank(self) -> int:
        return self._local_rank("ulysses")

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
    def ring_group(self) -> Optional[ProcessGroup]:
        return self._group("ring")

    @property
    def tp_group_pg(self) -> Optional[ProcessGroup]:
        return self._group("tp")

    @property
    def cfg_group(self) -> Optional[ProcessGroup]:
        return self._group("cfg")

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
