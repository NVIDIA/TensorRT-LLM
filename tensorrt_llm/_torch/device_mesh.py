from functools import wraps
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist
from torch.distributed import get_process_group_ranks
from torch.distributed.device_mesh import init_device_mesh

from tensorrt_llm._utils import mpi_disabled
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

    device_mesh = None
    tp_mesh = None

    def __init__(self, mapping: 'MappingBase'):
        assert mpi_disabled(
        ), "DeviceMeshTopology is only available in Ray orchestrator mode."
        self.world_size = mapping.world_size
        self.cp_size = mapping.cp_size
        self.pp_size = mapping.pp_size
        self.moe_ep_size = mapping.moe_ep_size
        self.moe_tp_size = mapping.moe_tp_size
        self.tp_size = mapping.tp_size
        self.auto_parallel = mapping.auto_parallel

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
        assert self.auto_parallel == False, "Auto parallel is not currently supported in Ray mode."
        return self.tp_group_pg.rank()

    @property
    def pp_rank(self) -> int:
        assert self.auto_parallel == False, "Auto parallel is not currently supported in Ray mode."
        return self.pp_group_pg.rank()

    @property
    def cp_rank(self) -> int:
        # TODO: WIP
        assert self.auto_parallel == False, "Auto parallel is not currently supported in Ray mode."
        return self.cp_group_pg.rank()

    # Access group ranks
    @property
    def tp_group(self) -> List[int]:
        return get_process_group_ranks(self.tp_group_pg)

    @property
    def pp_group(self) -> List[int]:
        return get_process_group_ranks(self.pp_group_pg)

    @property
    def cp_group(self) -> List[int]:
        return get_process_group_ranks(self.cp_group_pg)

    @property
    def moe_tp_group(self) -> List[int]:
        return get_process_group_ranks(self.moe_tp_group_pg)

    @property
    def moe_ep_group(self) -> List[int]:
        return get_process_group_ranks(self.moe_ep_group_pg)

    def build_mesh(self):
        if self.world_size == 1 or DeviceMeshTopology.device_mesh is not None:
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
        if DeviceMeshTopology.device_mesh is None and self.world_size == 1:

            class SingleProcessGroup:

                def get_group(self):
                    return dist.group.WORLD if dist.is_initialized() else None

            return SingleProcessGroup()

        if name == 'tp':
            if 'tp' in DeviceMeshTopology.device_mesh.mesh_dim_names:
                return DeviceMeshTopology.device_mesh['tp']
            else:
                return DeviceMeshTopology.tp_mesh
        else:
            assert name in DeviceMeshTopology.device_mesh.mesh_dim_names, f"Dimension name {name} not found in device mesh."
            return DeviceMeshTopology.device_mesh[name]
