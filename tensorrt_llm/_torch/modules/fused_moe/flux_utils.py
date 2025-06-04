import os

import flux
import torch
import torch.distributed as dist

from tensorrt_llm.mapping import Mapping


class DistEnv:

    def __init__(self, mapping: Mapping) -> None:
        self.mapping = mapping
        self.init_global_group()

    def init_global_group(self) -> None:
        if not dist.is_initialized():
            master_ip = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "6000")
            init_method = f"tcp://{master_ip}:{master_port}"
            dist.init_process_group(backend="nccl",
                                    init_method=init_method,
                                    world_size=self.mapping.world_size,
                                    rank=self.mapping.rank)

    def get_world(self):
        return torch.distributed.group.WORLD

    def new_group(self, ranks):
        return torch.distributed.new_group(ranks=ranks, backend="nccl")

    @property
    def rank(self):
        return self.mapping.rank

    @property
    def world_size(self):
        return self.mapping.world_size


DIST_ENV = None
EP_GROUP_DICT = None


def get_dist_env(mapping: Mapping):
    global DIST_ENV

    if DIST_ENV is None:
        DIST_ENV = DistEnv(mapping=mapping)
        flux.init_flux_shm(DIST_ENV.get_world())
    return DIST_ENV


def get_ep_group(mapping: Mapping):
    global EP_GROUP_DICT
    if EP_GROUP_DICT is None:
        EP_GROUP_DICT = {}
    if mapping not in EP_GROUP_DICT:
        assert DIST_ENV is not None, f"DIST_ENV should be initialized first. Current rank is {mapping.rank}"
        new_ep_group = DIST_ENV.new_group(mapping.moe_ep_group)
        EP_GROUP_DICT[mapping] = new_ep_group
    return EP_GROUP_DICT[mapping]
