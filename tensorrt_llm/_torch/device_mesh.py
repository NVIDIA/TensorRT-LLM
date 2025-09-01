from functools import wraps
from typing import TYPE_CHECKING, List

import torch
from torch.distributed import get_process_group_ranks
from torch.distributed.device_mesh import init_device_mesh

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.mapping import MappingBase


def require_device_mesh(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if DeviceMeshTopology.device_mesh is None:
            self.build_mesh()
        return func(self, *args, **kwargs)

    return wrapper


class DeviceMeshTopology:
    '''PyTorch DeviceMesh-based mapping implementation'''

    # Static variables to store the device mesh
    device_mesh = None
    tp_mesh = None

    def __init__(self, mapping: 'MappingBase'):
        self.mapping = mapping

    def __getattr__(self, name):
        mapping = object.__getattribute__(self, "mapping")
        return getattr(mapping, name)

    def build_mesh(self):
        if self.world_size == 1 or DeviceMeshTopology.device_mesh is not None:
            # only build mesh once
            return
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "DeviceMesh creation requested but torch.distributed process group "
                "has not been initialised")

        dims = ["cp", "pp"]
        shape = [self.cp_size, self.pp_size]

        if self.moe_ep_size > 1:
            dims += ["moe_tp", "moe_ep"]
            shape += [self.moe_tp_size, self.moe_ep_size]
        else:
            dims += ["tp"]
            shape += [self.tp_size]

        DeviceMeshTopology.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=tuple(shape),
            mesh_dim_names=tuple(dims),
        )

        if self.moe_ep_size > 1:
            DeviceMeshTopology.tp_mesh = DeviceMeshTopology.device_mesh[
                "moe_tp", "moe_ep"]._flatten(mesh_dim_name="tp")
        logger.debug(
            f"DeviceMeshTopology.device_mesh: {DeviceMeshTopology.device_mesh}")
        logger.debug(
            f"DeviceMeshTopology.tp_mesh: {DeviceMeshTopology.tp_mesh}")

    @require_device_mesh
    def _get_mesh_dim_by_name(self, name: str):
        if name == 'tp':
            if 'tp' in DeviceMeshTopology.device_mesh.mesh_dim_names:
                return DeviceMeshTopology.device_mesh['tp']
            else:
                return DeviceMeshTopology.tp_mesh
        else:
            assert name in DeviceMeshTopology.device_mesh.mesh_dim_names, f"DeviceMeshTopology.device_mesh.mesh_dim_names: {DeviceMeshTopology.device_mesh.mesh_dim_names}, {name=}"
            return DeviceMeshTopology.device_mesh[name]

    # Access Torch ProcessGroup
    @property
    def tp_group_pg(self):
        return self._get_mesh_dim_by_name('tp').get_group()

    @property
    def pp_group_pg(self):
        return self._get_mesh_dim_by_name('pp').get_group()

    @property
    def cp_group_pg(self):
        return self._get_mesh_dim_by_name('cp').get_group()

    @property
    def moe_tp_group_pg(self):
        return self._get_mesh_dim_by_name('moe_tp').get_group()

    @property
    def moe_ep_group_pg(self):
        return self._get_mesh_dim_by_name('moe_ep').get_group()

    # Access rank
    @property
    @require_device_mesh
    def tp_rank(self) -> int:
        assert self.auto_parallel == False, "Auto parallel not yet implemented in DeviceMesh path."
        return self.tp_group_pg.rank()

    @property
    @require_device_mesh
    def pp_rank(self) -> int:
        assert self.auto_parallel == False, "Auto parallel not yet implemented in Ray path."
        return self.pp_group_pg.rank()

    @property
    @require_device_mesh
    def cp_rank(self) -> int:
        # TODO: WIP
        assert self.auto_parallel == False, "Auto parallel not yet implemented in Ray path."
        return self.cp_group_pg.rank()

    # Access group ranks
    @property
    @require_device_mesh
    def tp_group(self) -> List[int]:
        return get_process_group_ranks(self.tp_group_pg)

    @property
    @require_device_mesh
    def pp_group(self) -> List[int]:
        return get_process_group_ranks(self.pp_group_pg)

    @property
    @require_device_mesh
    def cp_group(self) -> List[int]:
        return get_process_group_ranks(self.cp_group_pg)

    @property
    @require_device_mesh
    def moe_tp_group(self) -> List[int]:
        return get_process_group_ranks(self.moe_tp_group_pg)

    @property
    @require_device_mesh
    def moe_ep_group(self) -> List[int]:
        return get_process_group_ranks(self.moe_ep_group_pg)
