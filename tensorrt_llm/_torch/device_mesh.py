from functools import wraps
from typing import TYPE_CHECKING, Dict, List, Optional

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
    # Mesh-dim name -> ProcessGroup, filled by build_mesh() and validated
    # against the mesh it was built from (tests reset ``device_mesh``).  The
    # ``*_group_pg`` properties are then plain dict reads, which torch.compile
    # folds to constants instead of breaking the graph on a DeviceMesh slice.
    _group_cache: Dict[str, ProcessGroup] = {}
    _group_cache_mesh = None

    # Access Torch ProcessGroup
    @property
    def tp_group_pg(self) -> ProcessGroup:
        return self._get_group_by_name('tp')

    @property
    def pp_group_pg(self) -> ProcessGroup:
        return self._get_group_by_name('pp')

    @property
    def cp_group_pg(self) -> ProcessGroup:
        return self._get_group_by_name('cp')

    @property
    def moe_tp_group_pg(self) -> ProcessGroup:
        return self._get_group_by_name('moe_tp')

    @property
    def moe_ep_group_pg(self) -> ProcessGroup:
        return self._get_group_by_name('moe_ep')

    # Access rank
    @property
    def tp_rank(self) -> int:
        return self.tp_group_pg.rank()

    @property
    def pp_rank(self) -> int:
        return self.pp_group_pg.rank()

    @property
    def cp_rank(self) -> int:
        # TODO: WIP
        return self.cp_group_pg.rank()

    # Access group ranks
    @property
    def tp_group(self) -> List[int]:
        return self._get_group_ranks(self.tp_group_pg)

    @property
    def pp_group(self) -> List[int]:
        return self._get_group_ranks(self.pp_group_pg)

    @property
    def cp_group(self) -> List[int]:
        return self._get_group_ranks(self.cp_group_pg)

    @property
    def moe_tp_group(self) -> List[int]:
        return self._get_group_ranks(self.moe_tp_group_pg)

    @property
    def moe_ep_group(self) -> List[int]:
        return self._get_group_ranks(self.moe_ep_group_pg)

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
        cls._populate_group_cache()

    @classmethod
    def _populate_group_cache(cls) -> None:
        """(Re)build the mesh-dim name -> ProcessGroup cache from the class mesh.

        ``DeviceMesh`` already created one ProcessGroup per dimension at
        ``init_device_mesh`` time, so this only looks them up (no collective).
        """
        impl = DeviceMeshTopologyImpl
        impl._group_cache = {}
        impl._group_cache_mesh = impl.device_mesh
        if impl.device_mesh is None:
            return
        for name in impl.device_mesh.mesh_dim_names or ():
            impl._group_cache[name] = impl.device_mesh[name].get_group()
        if impl.tp_mesh is not None:
            # MoE layout: 'tp' is the flattened (moe_tp, moe_ep) mesh.
            impl._group_cache['tp'] = impl.tp_mesh.get_group()

    @classmethod
    def _cached_group(cls, name: str) -> Optional[ProcessGroup]:
        impl = DeviceMeshTopologyImpl
        if impl._group_cache_mesh is not impl.device_mesh:
            impl._populate_group_cache()
        return impl._group_cache.get(name)

    @require_device_mesh
    def _get_group_by_name(self, name: str) -> ProcessGroup:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None and self.world_size == 1:
            return SingleProcessGroup.get_group()
        pg = cls._cached_group(name)
        if pg is None:
            pg = self._get_mesh_dim_by_name(name).get_group()
            cls._group_cache[name] = pg
        return pg

    @require_device_mesh
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
