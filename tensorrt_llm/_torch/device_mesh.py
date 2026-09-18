from functools import wraps
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, get_process_group_ranks
from torch.distributed.device_mesh import init_device_mesh

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.mapping import MappingBase as _MappingBaseForTypeCheck
else:
    _MappingBaseForTypeCheck = object


def require_device_mesh(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if DeviceMeshTopologyImpl.device_mesh is None:
            self.build_mesh()
        return func(self, *args, **kwargs)

    return wrapper


class SingleProcessGroup:

    @staticmethod
    def get_group():
        return dist.group.WORLD if dist.is_initialized(
        ) else SingleProcessGroup()

    @staticmethod
    def rank():
        return 0

    @staticmethod
    def size():
        return 1


class DeviceMeshTopologyImpl(_MappingBaseForTypeCheck):
    device_mesh = None
    tp_mesh = None

    _MESH_DIM_CACHE_PREFIX = '_mesh_dim_cache_'

    # Group membership is fixed once the device mesh is built (build_mesh is
    # build-once and c10d ProcessGroups are immutable), so each mesh dim is
    # resolved once and memoized as a plain attribute. Later reads --
    # including inside a torch.compile'd forward, where the DeviceMesh
    # resolver is compiler-disabled and must never run -- are then constant
    # attribute loads, matching MpiTopology's precomputed rank lists.
    @torch.compiler.disable
    def _resolve_mesh_dim(self, name: str) -> tuple:
        pg = self._get_mesh_dim_by_name(name).get_group()
        group_name = pg.group_name if hasattr(pg, 'group_name') else ''
        resolved = (pg, pg.rank(), self._get_group_ranks(pg), group_name)
        # Before torch.distributed is initialized (single-process runs) the
        # group is a placeholder stub -- don't memoize it.
        if DeviceMeshTopologyImpl.device_mesh is not None or dist.is_initialized(
        ):
            setattr(self, f'{self._MESH_DIM_CACHE_PREFIX}{name}', resolved)
        return resolved

    def _mesh_dim(self, name: str) -> tuple:
        resolved = getattr(self, f'{self._MESH_DIM_CACHE_PREFIX}{name}', None)
        if resolved is None:
            resolved = self._resolve_mesh_dim(name)
        return resolved

    def __getstate__(self):
        # ProcessGroups are not picklable (and Ray pickles Mapping objects);
        # the caches are re-resolved lazily on the other side.
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith(self._MESH_DIM_CACHE_PREFIX)
        }

    # Access Torch ProcessGroup
    @property
    def tp_group_pg(self) -> ProcessGroup:
        return self._mesh_dim('tp')[0]

    @property
    def tp_group_name(self) -> str:
        return self._mesh_dim('tp')[3]

    @property
    def pp_group_pg(self) -> ProcessGroup:
        return self._mesh_dim('pp')[0]

    @property
    def cp_group_pg(self) -> ProcessGroup:
        return self._mesh_dim('cp')[0]

    @property
    def cp_group_name(self) -> str:
        return self._mesh_dim('cp')[3]

    @property
    def moe_tp_group_pg(self) -> ProcessGroup:
        return self._mesh_dim('moe_tp')[0]

    @property
    def moe_ep_group_pg(self) -> ProcessGroup:
        return self._mesh_dim('moe_ep')[0]

    # Access rank
    @property
    def tp_rank(self) -> int:
        return self._mesh_dim('tp')[1]

    @property
    def pp_rank(self) -> int:
        return self._mesh_dim('pp')[1]

    @property
    def cp_rank(self) -> int:
        # TODO: WIP
        return self._mesh_dim('cp')[1]

    # Access group ranks
    @property
    def tp_group(self) -> List[int]:
        return self._mesh_dim('tp')[2]

    @property
    def pp_group(self) -> List[int]:
        return self._mesh_dim('pp')[2]

    @property
    def cp_group(self) -> List[int]:
        return self._mesh_dim('cp')[2]

    @property
    def moe_tp_group(self) -> List[int]:
        return self._mesh_dim('moe_tp')[2]

    @property
    def moe_ep_group(self) -> List[int]:
        return self._mesh_dim('moe_ep')[2]

    def build_mesh(self):
        cls = DeviceMeshTopologyImpl

        if self.world_size == 1 or cls.device_mesh is not None:
            # only build mesh once
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "DeviceMesh creation requested but torch.distributed process group "
                "has not been initialised.")

        # Dimensions go from slowest-varying (outermost) to fastest-varying (innermost).
        # Layout: pp is outermost, then tp, then cp is innermost (consecutive).
        dims = ["pp"]
        shape = [self.pp_size]

        if self.moe_ep_size > 1:
            dims += ["moe_tp", "moe_ep"]
            shape += [self.moe_tp_size, self.moe_ep_size]
        else:
            dims += ["tp"]
            shape += [self.tp_size]

        dims += ["cp"]
        shape += [self.cp_size]

        cls.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=tuple(shape),
            mesh_dim_names=tuple(dims),
        )

        if self.moe_ep_size > 1:
            cls.tp_mesh = cls.device_mesh["moe_tp",
                                          "moe_ep"]._flatten(mesh_dim_name="tp")
        logger.debug(f"DeviceMeshTopology.device_mesh: {cls.device_mesh}")
        logger.debug(f"DeviceMeshTopology.tp_mesh: {cls.tp_mesh}")

    @require_device_mesh
    @torch.compiler.disable
    def _get_mesh_dim_by_name(self, name: str) -> dist.DeviceMesh:
        cls = DeviceMeshTopologyImpl

        if cls.device_mesh is None and self.world_size == 1:
            return SingleProcessGroup()

        if name == 'tp':
            if 'tp' in cls.device_mesh.mesh_dim_names:
                return cls.device_mesh['tp']
            else:
                return cls.tp_mesh
        else:
            assert name in cls.device_mesh.mesh_dim_names, f"Dimension name {name} not found in device mesh."
            return cls.device_mesh[name]

    def _get_group_ranks(self, pg) -> List[int]:
        if self.world_size == 1:
            return [0]
        return get_process_group_ranks(pg)
