from functools import wraps
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist
from torch.distributed import get_process_group_ranks
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

    # Access Torch ProcessGroup
    @property
    @require_device_mesh
    def tp_group_pg(self):
        return self._get_mesh_dim_by_name('tp').get_group()

    @property
    @require_device_mesh
    def pp_group_pg(self):
        return self._get_mesh_dim_by_name('pp').get_group()

    @property
    @require_device_mesh
    def cp_group_pg(self):
        return self._get_mesh_dim_by_name('cp').get_group()

    @property
    @require_device_mesh
    def moe_tp_group_pg(self):
        return self._get_mesh_dim_by_name('moe_tp').get_group()

    @property
    @require_device_mesh
    def moe_ep_group_pg(self):
        return self._get_mesh_dim_by_name('moe_ep').get_group()

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

        dims = ["cp", "pp"]
        shape = [self.cp_size, self.pp_size]

        if self.moe_ep_size > 1:
            dims += ["moe_tp", "moe_ep"]
            shape += [self.moe_tp_size, self.moe_ep_size]
        else:
            dims += ["tp"]
            shape += [self.tp_size]

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
