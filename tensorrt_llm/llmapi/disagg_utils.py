import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from mpi4py.MPI import COMM_WORLD, Comm
from mpi4py.util import pkl5

from .._utils import global_mpi_rank, global_mpi_size

__all__ = [
    'ServerConfig',
    'parse_disagg_config_file',
    'extract_server_configs',
    'split_world_comm',
    'get_usage_tokens_from_ctx',
    'rewrite_usage_info_from_ctx',
    'rewrite_usage_response_from_ctx',
]


def validate_config_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(
        f"{field_name} must be a boolean, got {type(value).__name__}")


class ServerRole(IntEnum):
    CONTEXT = 0
    GENERATION = 1
    MM_ENCODER = 2
    VISUAL_GEN = 3
    EMBEDDING = 4


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
class OtlpConfig():
    otlp_traces_endpoint: Optional[
        str] = None  # Target URL to which OpenTelemetry traces will be sent


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
    otlp_config: Optional[OtlpConfig] = None
    max_retries: int = 1
    perf_metrics_max_requests: int = 0
    disagg_cluster_config: Optional[DisaggClusterConfig] = None
    node_id: int = uuid.getnode(
    ) % 256  # Assuming only one disagg-server is running on a machine, modulo 256.
    # If this causes collisions, users can set node_id manually within range [0, 255] in config
    schedule_style: Literal['context_first',
                            'generation_first'] = 'context_first'
    allow_request_chat_template: bool = False
    # Drop conversation history from generation_only requests to shrink the gen
    # worker's request body / JSON-parse GIL cost at high concurrency. Enable
    # ONLY for text-only, non-harmony deployments (see _get_gen_request).
    gen_strip_message_history: bool = False
    # Ask context workers to return prompt_token_ids as a base64 int32 buffer so
    # the orchestrator relays a string instead of materializing the token-id list
    # on its event loop. Text-only, non-harmony deployments (see _get_ctx_request).
    gen_tokids_ctxbytes: bool = False
    # Number of uvicorn disagg-server worker processes to fork on the public port.
    # >1 means a fleet of delegating servers behind one coordinator. Replaces the
    # WEB_CONCURRENCY env var (explicit config over implicit env).
    num_workers: int = 1
    # URL of an already-running coordinator (e.g. "http://host:8332"). When set the
    # fleet delegates to it; when absent, num_workers>1 starts an implicit in-process
    # coordinator and num_workers==1 runs a single self-contained server.
    disagg_coordinator_url: Optional[str] = None


@dataclass
class MetadataServerConfig():
    server_type: Literal['etcd']
    hostname: str = "localhost"
    port: int = 2379
    health_check_timeout: float = 5.0
    refresh_interval: float = 10.0


def get_usage_tokens_from_ctx(
        ctx_usage: Optional[Any]) -> tuple[Optional[int], int]:
    if ctx_usage is None:
        return None, 0

    prompt_tokens = ctx_usage.prompt_tokens
    cached_tokens = 0
    prompt_tokens_details = ctx_usage.prompt_tokens_details
    if prompt_tokens_details is not None:
        cached_tokens = prompt_tokens_details.cached_tokens
    return prompt_tokens, cached_tokens


def rewrite_usage_info_from_ctx(usage: Optional[Any],
                                ctx_usage: Optional[Any]) -> Optional[Any]:
    prompt_tokens, cached_tokens = get_usage_tokens_from_ctx(ctx_usage)
    if prompt_tokens is None or usage is None:
        return usage

    from tensorrt_llm.serve.openai_protocol import PromptTokensDetails

    usage.prompt_tokens = prompt_tokens
    usage.total_tokens = prompt_tokens + (usage.completion_tokens or 0)
    usage.prompt_tokens_details = PromptTokensDetails(
        cached_tokens=cached_tokens)
    return usage


def rewrite_usage_response_from_ctx(response: Any,
                                    ctx_usage: Optional[Any]) -> Any:
    rewrite_usage_info_from_ctx(response.usage, ctx_usage)
    return response


def get_ctx_gen_server_addrs(
        server_configs: list[CtxGenServerConfig]
) -> tuple[list[str], list[str]]:
    ctx_server_urls = []
    gen_server_urls = []
    for cfg in server_configs:
        if cfg.type == "ctx":
            ctx_server_urls.append(f"{cfg.hostname}:{cfg.port}")
        else:
            gen_server_urls.append(f"{cfg.hostname}:{cfg.port}")

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
                       otlp_config: Optional[dict] = None,
                       disagg_cluster: Optional[dict] = None,
                       node_id: Optional[int] = None,
                       schedule_style: Literal[
                           'context_first',
                           'generation_first'] = 'context_first',
                       allow_request_chat_template: bool = False,
                       gen_strip_message_history: bool = False,
                       gen_tokids_ctxbytes: bool = False,
                       num_workers: int = 1,
                       disagg_coordinator_url: Optional[str] = None,
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

    otlp_config = OtlpConfig(**otlp_config) if otlp_config else None

    config = DisaggServerConfig(server_configs, hostname, port,
                                ctx_router_config, gen_router_config,
                                conditional_disagg_config, otlp_config,
                                max_retries, perf_metrics_max_requests,
                                disagg_cluster_config)
    if node_id is not None:
        node_id_space = 1 << DISAGG_NODE_ID_BITS
        if not 0 <= node_id < node_id_space:
            raise ValueError(
                f"node_id must be in range [0, {node_id_space}), got {node_id}")
        config.node_id = node_id
    if schedule_style:
        config.schedule_style = schedule_style
    config.allow_request_chat_template = validate_config_bool(
        allow_request_chat_template, "allow_request_chat_template")
    config.gen_strip_message_history = gen_strip_message_history
    config.gen_tokids_ctxbytes = gen_tokids_ctxbytes
    config.num_workers = num_workers
    config.disagg_coordinator_url = disagg_coordinator_url
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
        'pipeline_parallel_size', 1) * kwargs.get('context_parallel_size', 1)

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

    if router_type == "kv_cache_aware" and "model_path" not in args:
        model_path = server_cfg.get("model")
        if model_path is not None:
            args["model_path"] = model_path

    # add fields that are not specific to router
    extract_keys = ["max_batch_size", "max_num_tokens"]
    for key in extract_keys:
        if key in server_cfg:
            args[key] = server_cfg[key]

    # tokens_per_block lives under kv_cache_config; the cache-aware router must
    # use the same block size as the worker or block hashes never match. Carry
    # the explicit server value over unless the router block already set it.
    kv_cache_config = server_cfg.get("kv_cache_config") or {}
    if "tokens_per_block" not in args and "tokens_per_block" in kv_cache_config:
        args["tokens_per_block"] = kv_cache_config["tokens_per_block"]

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

    return is_leader, instance_idx, pkl5.Intracomm(sub_comm)


def parse_metadata_server_config_file(
    metadata_server_config_file: Optional[str]
) -> Optional[MetadataServerConfig]:
    if metadata_server_config_file is None:
        return None

    with open(metadata_server_config_file, 'r') as file:
        config = yaml.safe_load(file)
        return MetadataServerConfig(**config)


# Snowflake global disagg request id, 64-bit / positive int64 (MSB reserved 0):
#   [ 0 (1) | timestamp_ms (39) | node_id (8) | process_id (6) | counter (10) ]
# The (node_id, process_id) pair identifies a fleet worker process, so co-located
# workers never emit the same id in the same millisecond. See docs/source/
# advanced/disaggregated-service.md for the full disagg-request-id design.
DISAGG_TIMESTAMP_BITS = 39
DISAGG_NODE_ID_BITS = 8
DISAGG_PROCESS_ID_BITS = 6
DISAGG_COUNTER_BITS = 10

# Local ids [0, MIN_GLOBAL_ID) and global disagg ids [MIN_GLOBAL_ID, 2^63) are
# disjoint by construction so they never collide. Power of two (masked in
# get_local_request_id).
MIN_GLOBAL_ID = 1 << 40

# Consider GIL being removed in the future, use a lock to protect the counter
_global_disagg_request_id_lock = threading.Lock()
_global_disagg_request_id_counter = 0


def get_global_disagg_request_id(node_id: int, process_id: int = 0) -> int:
    """A snowflake global disagg request id (does not guarantee monotonicity).

    Layout: 0(1) | timestamp_ms(39) | node_id(8) | process_id(6) | counter(10).
    node_id identifies the node, process_id the fleet worker process on it -- the
    pair makes the id unique across co-located workers without any coordination.
    """
    global _global_disagg_request_id_lock
    global _global_disagg_request_id_counter

    NODE_ID_SPACE = 1 << DISAGG_NODE_ID_BITS
    PROCESS_ID_SPACE = 1 << DISAGG_PROCESS_ID_BITS
    COUNTER_MASK = (1 << DISAGG_COUNTER_BITS) - 1
    TIMESTAMP_MASK = (1 << DISAGG_TIMESTAMP_BITS) - 1
    MAX_INT64 = (1 << 63) - 1

    if node_id not in range(0, NODE_ID_SPACE):
        raise ValueError(f"node_id must be in range [0, {NODE_ID_SPACE})")
    if process_id not in range(0, PROCESS_ID_SPACE):
        raise ValueError(f"process_id must be in range [0, {PROCESS_ID_SPACE})")

    timestamp_ms = int(time.monotonic() * 1000) & TIMESTAMP_MASK
    with _global_disagg_request_id_lock:
        counter = _global_disagg_request_id_counter & COUNTER_MASK
        _global_disagg_request_id_counter += 1

    global_id = (
        (timestamp_ms <<
         (DISAGG_NODE_ID_BITS + DISAGG_PROCESS_ID_BITS + DISAGG_COUNTER_BITS))
        | (node_id << (DISAGG_PROCESS_ID_BITS + DISAGG_COUNTER_BITS))
        | (process_id << DISAGG_COUNTER_BITS)
        | counter)
    # Rotate into [MIN_GLOBAL_ID, MAX_INT64); [0, MIN_GLOBAL_ID) is local-id space.
    global_id_int64 = global_id % (MAX_INT64 - MIN_GLOBAL_ID) + MIN_GLOBAL_ID
    return global_id_int64


def get_local_request_id(last_id: int) -> int:
    """ increment the last_id by 1 and mod by MIN_GLOBAL_ID """
    return (last_id + 1) & (MIN_GLOBAL_ID - 1)


def disagg_process_id_space() -> int:
    """Number of distinct process_id slots in the snowflake id (2^bits)."""
    return 1 << DISAGG_PROCESS_ID_BITS


def worker_local_process_id() -> int:
    """Return this fleet worker's process index.

    The fleet launcher sets ``TRTLLM_DISAGG_WORKER_PROCESS_ID`` to a distinct
    value per process. A standalone disaggregated server defaults to 0.
    """
    process_id = int(os.environ.get("TRTLLM_DISAGG_WORKER_PROCESS_ID", "0"))
    process_id_space = disagg_process_id_space()
    if not 0 <= process_id < process_id_space:
        raise ValueError(
            "TRTLLM_DISAGG_WORKER_PROCESS_ID must be between 0 and "
            f"{process_id_space - 1}, got {process_id}")
    return process_id
