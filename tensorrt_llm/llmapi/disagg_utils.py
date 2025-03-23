import logging
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

import yaml
from mpi4py.MPI import COMM_WORLD, Comm

from .._utils import global_mpi_rank, global_mpi_size

__all__ = [
    'ServerConfig',
    'parse_disagg_config_file',
    'extract_server_configs',
    'split_world_comm',
]


@dataclass
class CtxGenServerConfig():
    type: Literal['ctx', 'gen']
    hostname: Optional[str] = None
    port: Optional[int] = None
    instance_num_ranks: int = 1
    other_args: dict = field(default_factory=dict)


@dataclass
class DisaggServerConfig():
    server_configs: List[CtxGenServerConfig]
    hostname: str = "localhost"
    port: int = 8000


def parse_disagg_config_file(yaml_config_file: str):

    with open(yaml_config_file, 'r') as file:

        config = yaml.safe_load(file)

        disagg_server_config = extract_disagg_cfg(**config)

        return disagg_server_config


def extract_disagg_cfg(hostname: str = 'localhost',
                       port: int = 8000,
                       context_servers: dict = {},
                       generation_servers: dict = {},
                       **kwargs: Any) -> DisaggServerConfig:

    # If parameters are specified outside the context_severs and generation_servers sections,
    # make sure they match
    # Also inherit the values from the top-level
    for key, value in kwargs.items():
        for server_type, servers in [("context_servers", context_servers),
                                     ("generation_servers", generation_servers)
                                     ]:
            if key in servers:
                if servers[key] != value:
                    raise ValueError(
                        f"Parameter {key} is specified both in the top-level and in the {server_type} section, but with different values"
                    )
            else:
                # Inherit the value from the top-level
                servers[key] = value

    server_configs = extract_ctx_gen_cfgs(
        type="ctx", **context_servers) + extract_ctx_gen_cfgs(
            type="gen", **generation_servers)

    return DisaggServerConfig(server_configs, hostname, port)


def extract_ctx_gen_cfgs(type: Literal['ctx', 'gen'],
                         num_instances: int = 1,
                         urls: Optional[List[str]] = None,
                         **kwargs: Any) -> List[CtxGenServerConfig]:

    hostnames = []
    ports = []
    if urls:
        for url in urls:
            hostname, port_str = url.split(':')
            port = int(port_str)
            hostnames.append(hostname)
            ports.append(port)

        if len(hostnames) != num_instances:
            raise ValueError(
                f"Number of hostnames ({len(hostnames)}) should be equal to the number of instances ({num_instances})"
            )

        if len(ports) != num_instances:
            raise ValueError(
                f"Number of ports ({len(ports)}) should be equal to the number of instances ({num_instances})"
            )

    else:
        hostnames = [None] * num_instances
        ports = [None] * num_instances

    # Compute the number of ranks per instance
    instance_num_ranks = kwargs.get('tensor_parallel_size', 1) * kwargs.get(
        'pipeline_parallel_size', 1)

    cfgs = []
    for hostname, port in zip(hostnames, ports):
        cfgs.append(
            CtxGenServerConfig(type=type,
                               hostname=hostname,
                               port=port,
                               instance_num_ranks=instance_num_ranks,
                               other_args=kwargs))
    return cfgs


def split_world_comm(
        server_configs: List[CtxGenServerConfig]) -> Tuple[bool, int, Comm]:

    # Check that MPI_COMM_WORLD size is compatible with the number of workers
    global_size = global_mpi_size()
    global_rank = global_mpi_rank()
    num_workers = 0

    # check for duplicate server configs
    server_dict = {}
    for cfg in server_configs:
        url = (cfg.hostname, cfg.port)
        if url in server_dict:
            cfg_prev = server_dict[url]
            if cfg_prev.type == cfg.type:
                raise ValueError(
                    f"Duplicated {cfg.type} server config for {url}")
            # mixed server, config should be the same
            if cfg_prev.other_args != cfg.other_args:
                raise ValueError(
                    f"Server config for {url} has different args:\n{cfg_prev.other_args}\n{cfg.other_args}"
                )
        else:
            server_dict[url] = cfg
            num_workers += cfg.instance_num_ranks

    assert global_size == num_workers, f"global_size ({global_size}) should be equal to the number of distinct workers ({num_workers})"

    # Identify the leader ranks and the instance idx for each rank
    is_leader = False
    offset = 0
    instance_idx = 0
    instance_sub_rank = 0
    for idx, cfg in enumerate(server_configs):
        if (cfg.hostname, cfg.port) not in server_dict:
            continue
        server_dict.pop((cfg.hostname, cfg.port))
        if global_rank >= offset and global_rank < offset + cfg.instance_num_ranks:
            instance_idx = idx
            instance_sub_rank = global_rank - offset
            # The first rank in each instance is the leader
            if global_rank == offset:
                is_leader = True
        offset += cfg.instance_num_ranks

    # Split MPI_COMM_WORLD into sub-communicators based on rank_instance_idx
    sub_comm = COMM_WORLD.Split(color=instance_idx, key=instance_sub_rank)
    sub_rank = sub_comm.Get_rank()
    if sub_rank != instance_sub_rank:
        raise RuntimeError(
            f"Expected sub_rank {sub_rank} to be equal to instance_sub_rank {instance_sub_rank}"
        )

    sub_comm.Barrier()

    logging.info(
        f"global_rank: {global_rank}, instance_idx: {instance_idx}, sub_rank: {sub_rank}, is_leader: {is_leader}"
    )

    return is_leader, instance_idx, sub_comm
