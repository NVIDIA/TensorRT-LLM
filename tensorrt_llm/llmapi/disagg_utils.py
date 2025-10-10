import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from mpi4py.MPI import COMM_WORLD, Comm

from .._utils import global_mpi_rank, global_mpi_size

__all__ = [
    'ServerConfig',
    'parse_disagg_config_file',
    'extract_server_configs',
    'split_world_comm',
]


class ServerRole(IntEnum):
    CONTEXT = 0
    GENERATION = 1
    MM_ENCODER = 2


@dataclass
class CtxGenServerConfig():
    type: Literal['ctx', 'gen']
    hostname: Optional[str] = None
    port: Optional[int] = None
    instance_num_ranks: int = 1
    other_args: dict = field(default_factory=dict)


@dataclass
class RouterConfig():
    type: str = "round_robin"
    args: dict = field(default_factory=dict)
    server_role: ServerRole = None


@dataclass
class ConditionalDisaggConfig():
    max_local_prefill_length: int = 0


@dataclass
class MinimalInstances:
    context_servers: int = 1  # the minimal number of context servers
    generation_servers: int = 1  # the minimal number of generation servers


@dataclass
class DisaggClusterConfig:
    cluster_uri: str  # the uri of the cluster storage
    cluster_name: str = ""  # the name of the cluster, used like a namespace
    minimal_instances: Optional[MinimalInstances] = None
    heartbeat_interval_sec: int = 5  # the worker will send heartbeat to the cluster storage every heartbeat_interval_sec seconds
    inactive_timeout_sec: int = 10  # the worker will be considered inactive if it doesn't send heartbeat for inactive_timeout_sec seconds


@dataclass
class DisaggServerConfig():
    server_configs: List[CtxGenServerConfig]
    hostname: str = "localhost"
    port: int = 8000
    ctx_router_config: Optional[RouterConfig] = None
    gen_router_config: Optional[RouterConfig] = None
    conditional_disagg_config: Optional[ConditionalDisaggConfig] = None
    max_retries: int = 1
    perf_metrics_max_requests: int = 0
    disagg_cluster_config: Optional[DisaggClusterConfig] = None


@dataclass
class MetadataServerConfig():
    server_type: Literal['etcd']
    hostname: str = "localhost"
    port: int = 2379
    health_check_timeout: float = 5.0
    refresh_interval: float = 10.0


def get_ctx_gen_server_urls(
        server_configs: list[CtxGenServerConfig]
) -> tuple[list[str], list[str]]:
    ctx_server_urls = []
    gen_server_urls = []
    for cfg in server_configs:
        if cfg.type == "ctx":
            ctx_server_urls.append(f"http://{cfg.hostname}:{cfg.port}")
        else:
            gen_server_urls.append(f"http://{cfg.hostname}:{cfg.port}")

    return ctx_server_urls, gen_server_urls


def parse_disagg_config_file(yaml_config_file: str):

    with open(yaml_config_file, 'r') as file:

        config = yaml.safe_load(file)

        disagg_server_config = extract_disagg_cfg(**config)

        return disagg_server_config


def extract_disagg_cfg(hostname: str = 'localhost',
                       port: int = 8000,
                       max_retries: int = 1,
                       perf_metrics_max_requests: int = 0,
                       context_servers: Optional[dict] = None,
                       generation_servers: Optional[dict] = None,
                       conditional_disagg_config: Optional[dict] = None,
                       disagg_cluster: Optional[dict] = None,
                       **kwargs: Any) -> DisaggServerConfig:
    context_servers = context_servers or {}
    generation_servers = generation_servers or {}

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

    server_configs = []
    disagg_cluster_config = None
    ctx_router_config = extract_router_config(context_servers)
    gen_router_config = extract_router_config(generation_servers)
    ctx_router_config.server_role = ServerRole.CONTEXT
    gen_router_config.server_role = ServerRole.GENERATION
    if disagg_cluster:
        disagg_cluster_config = extract_disagg_cluster_config(disagg_cluster)
    else:
        server_configs = extract_ctx_gen_cfgs(
            type="ctx", **context_servers) + extract_ctx_gen_cfgs(
                type="gen", **generation_servers)

    conditional_disagg_config = ConditionalDisaggConfig(
        **conditional_disagg_config) if conditional_disagg_config else None

    config = DisaggServerConfig(server_configs, hostname, port,
                                ctx_router_config, gen_router_config,
                                conditional_disagg_config, max_retries,
                                perf_metrics_max_requests,
                                disagg_cluster_config)

    return config


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


def extract_router_config(server_cfg: dict) -> RouterConfig:

    args = server_cfg.pop("router", {})
    router_type = args.pop("type", "round_robin")

    # add fields that are not specific to router
    extract_keys = ["max_batch_size", "max_num_tokens"]
    for key in extract_keys:
        if key in server_cfg:
            args[key] = server_cfg[key]

    return RouterConfig(type=router_type, args=args)


def get_server_configs_dict(
        server_configs: List[CtxGenServerConfig]) -> Tuple[int, dict]:

    num_workers = 0
    server_dict = {}

    # check for duplicate server configs
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

    return num_workers, server_dict


def extract_disagg_cluster_config(
        cluster_config_dict: Dict[str, Any],
        cluster_uri: Optional[str] = None) -> DisaggClusterConfig:
    """
    Build the DisaggClusterConfig from the cluster_config_dict.
    Use the default value of DisaggClusterConfig and MinimalInstances if the corresponding fields are not provided.
    If cluster_uri is provided, it will override the cluster_uri in the cluster_config_dict.
    """

    def update_dataclass(obj, data_dict: Dict[str, Any]):
        for key, value in data_dict.items():
            if key not in obj.__dataclass_fields__:
                raise KeyError(
                    f"Key {key} not found in {obj.__class__.__name__}")
            if value is not None:
                setattr(obj, key, value)
        return obj

    cluster_config_dict["minimal_instances"] = update_dataclass(
        MinimalInstances(), cluster_config_dict.get("minimal_instances", {}))
    cluster_config = update_dataclass(
        DisaggClusterConfig(cluster_uri or cluster_config_dict["cluster_uri"]),
        cluster_config_dict,
    )
    return cluster_config


def split_world_comm(
        server_configs: List[CtxGenServerConfig]) -> Tuple[bool, int, Comm]:

    # Check that MPI_COMM_WORLD size is compatible with the number of workers
    global_size = global_mpi_size()
    global_rank = global_mpi_rank()

    [num_workers, server_dict] = get_server_configs_dict(server_configs)
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


def parse_metadata_server_config_file(
    metadata_server_config_file: Optional[str]
) -> Optional[MetadataServerConfig]:
    if metadata_server_config_file is None:
        return None

    with open(metadata_server_config_file, 'r') as file:
        config = yaml.safe_load(file)
        return MetadataServerConfig(**config)
