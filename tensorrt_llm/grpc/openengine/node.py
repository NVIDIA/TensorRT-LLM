# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup-only node discovery, independent of inference and routing policy."""

import socket
from concurrent.futures import ThreadPoolExecutor

import grpc
from openengine.v1 import kv_pb2, openengine_pb2_grpc, server_pb2

from tensorrt_llm._torch.pyexecutor.kv_cache_events import (
    HEARTBEAT_INTERVAL_SECONDS,
    StreamingKVCacheEventManager,
)
from tensorrt_llm.version import __version__

from .kv_events import _format_tcp_endpoint, _split_tcp_endpoint

__all__ = ["NodeMetadataServer", "initialize_node_discovery"]

NODE_METADATA_KEY = "trtllm_node"


def _source(descriptor: dict, config, host: str) -> dict:
    endpoint_host, port = _split_tcp_endpoint(descriptor["endpoint"])
    if endpoint_host in ("", "*", "0.0.0.0", "::"):
        endpoint_host = host
    replay = descriptor["replay_endpoint"]
    if replay:
        replay_host, replay_port = _split_tcp_endpoint(replay)
        if replay_host in ("", "*", "0.0.0.0", "::"):
            replay_host = host
        replay = _format_tcp_endpoint(replay_host, replay_port)
    return {
        "transport": "zmq",
        "endpoint_addr": {"host": endpoint_host, "port": port, "protocol": "tcp"},
        "topic": config.topic,
        "replay_endpoint": replay,
        "data_parallel_rank": descriptor["rank"],
        "encoding": "msgpack",
        "schema_version": 1,
        "buffer_steps": config.buffer_steps,
        "hwm": config.hwm,
        "max_queue_size": config.max_queue_size,
    }


class _NodeControl(openengine_pb2_grpc.ControlServicer):
    def __init__(self, info: server_pb2.ServerInfo, sources: list[dict]) -> None:
        self._info = info
        self._sources = [kv_pb2.KvEventSource(**source) for source in sources]

    def GetServerInfo(self, request, context):
        return self._info

    def GetKvEventSources(self, request, context):
        ranks = set(request.data_parallel_ranks)
        available = {source.data_parallel_rank for source in self._sources}
        if ranks - available:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Requested rank has no local KV source")
        return kv_pb2.GetKvEventSourcesResponse(
            sources=[s for s in self._sources if not ranks or s.data_parallel_rank in ranks]
        )


class NodeMetadataServer:
    """A follower's bounded metadata-only listener, owned by its executor."""

    def __init__(self, host: str, port: int, info: server_pb2.ServerInfo, sources: list[dict]):
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="openengine-metadata")
        self._server = grpc.server(
            self._pool, maximum_concurrent_rpcs=8, options=(("grpc.so_reuseport", 0),)
        )
        openengine_pb2_grpc.add_ControlServicer_to_server(_NodeControl(info, sources), self._server)
        try:
            address = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
            self.port = self._server.add_insecure_port(address)
            if not self.port:
                raise RuntimeError(f"Cannot bind OpenEngine node discovery to {address}")
            self._server.start()
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Stop discovery before its engine's publishers are torn down."""
        self._server.stop(0).wait()
        self._pool.shutdown(wait=True)


def initialize_node_discovery(executor) -> None:
    """Publish final per-node placement once, before any scheduler thread runs.

    Two startup collectives exchange actual publishers and propagate bind errors.
    No metadata RPC touches GPU state or performs a distributed collective.
    """
    args = executor.llm_args
    bootstrap = args._openengine_discovery
    dist = executor.dist
    host = socket.getfqdn()
    manager = getattr(executor.kv_cache_manager, "event_manager", None)
    local_error = None
    local_source = None
    try:
        descriptor = (
            manager.get_source_descriptor()
            if isinstance(manager, StreamingKVCacheEventManager)
            else None
        )
        config = args.kv_cache_config.kv_events_config
        local_source = _source(descriptor, config, host) if descriptor else None
    except (RuntimeError, ValueError, OSError) as exc:
        local_error = str(exc)
    gathered = dist.allgather((dist.rank, host, local_source, local_error))
    errors = [row[3] for row in gathered if row[3] is not None]
    if errors:
        raise RuntimeError(f"OpenEngine publisher discovery failed: {errors}")
    placements = [row[:3] for row in gathered]
    placements.sort(key=lambda row: row[0])
    leader_host = placements[0][1]
    node_hosts = list(dict.fromkeys(row[1] for row in placements))
    local_ranks = [rank for rank, node, _ in placements if node == host]
    sources = [source for _, _, source in placements if source is not None]
    source_owners = {
        str(source["data_parallel_rank"]): node for _, node, source in placements if source
    }
    if len(source_owners) != len(sources):
        raise ValueError("Duplicate attention-DP event publisher ownership")
    dp_size = dist.tp_size if executor.enable_attention_dp else 1
    metadata = {
        "version": 1,
        # Unique for this entire engine incarnation, not a reusable address.
        "engine_id": bootstrap["engine_id"],
        "node_id": host,
        "leader": host == leader_host,
        "node_count": len(node_hosts),
        "kv_block_size": int(getattr(executor.kv_cache_manager, "tokens_per_block", 0)),
        "local_dp_ranks": sorted(
            int(rank) for rank, owner in source_owners.items() if owner == host
        ),
        "source_owners": source_owners,
    }
    capacity = executor._kv_cache_capacity
    capacity["openEngineNode"] = metadata
    capacity["kvEventSources"] = sources
    capacity["kvEventRankHosts"] = {int(rank): node for rank, node in source_owners.items()}
    info = server_pb2.ServerInfo(
        engine_name="tensorrt_llm",
        engine_version=__version__,
        instance_id=bootstrap["engine_id"],
        schema_revision=1,
        minimum_client_revision=1,
    )
    info.parallelism.data_parallel_size = dp_size
    info.parallelism.data_parallel_start_rank = 0
    info.capacity.kv_block_size = metadata["kv_block_size"]
    info.extra.update({NODE_METADATA_KEY: metadata})
    if sources:
        info.extra.update(
            {"kv_event_heartbeat_interval_ms": int(HEARTBEAT_INTERVAL_SECONDS * 1000)}
        )
    server = None
    error = None
    try:
        if host != leader_host and dist.rank == min(local_ranks):
            if int(bootstrap["port"]) == 0:
                raise ValueError("Multi-node OpenEngine discovery requires a fixed listener port")
            server = NodeMetadataServer(
                str(bootstrap["host"]),
                int(bootstrap["port"]),
                info,
                [
                    source
                    for source in sources
                    if source_owners[str(source["data_parallel_rank"])] == host
                ],
            )
    except (RuntimeError, ValueError, OSError) as exc:
        error = str(exc)
    errors = dist.allgather(error)
    if any(errors):
        if server is not None:
            server.close()
        raise RuntimeError(f"OpenEngine node discovery startup failed: {errors}")
    executor._openengine_node_server = server
