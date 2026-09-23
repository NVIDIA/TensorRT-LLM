# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV cache event source discovery for the OpenEngine Control service.

TensorRT-LLM publishes KV cache events from the engine itself: every attention-DP
rank binds its own ZeroMQ ``PUB`` socket and sends msgpack batches from a
background thread (``tensorrt_llm._torch.pyexecutor.kv_cache_events``). This
module exposes that publisher through OpenEngine discovery:

``GetKvEventSources`` advertises the sockets so a client can subscribe to the
engine directly, which keeps events off the gRPC server's event loop.
Event delivery is exclusively direct ZMQ; SubscribeKvEvents is deferred.

Discovery reads one source of truth -- ``kv_cache_config.kv_events_config`` -- and
reuses the publisher's own ``base_port + rank`` convention rather than
reimplementing it, so the advertisement cannot drift from the binds.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Optional

from openengine.v1 import kv_pb2

from tensorrt_llm._torch.pyexecutor.kv_cache_events import ZmqEventPublisher

__all__ = [
    "KvEventsUnavailableError",
    "ResolvedKvEventSource",
    "resolve_advertise_host",
    "resolve_sources",
    "to_proto_source",
]

# A bind wildcard is not a connectable address, and the protocol requires
# `endpoint_addr` to carry one.
_WILDCARD_HOSTS = frozenset({"*", "0.0.0.0", "::", ""})


class KvEventsUnavailableError(RuntimeError):
    """Raised when KV events exist but cannot be advertised or subscribed to."""


@dataclass(frozen=True)
class ResolvedKvEventSource:
    """One attention-DP rank's publisher, addressed so a client can connect."""

    data_parallel_rank: int
    host: str
    port: int
    replay_endpoint: Optional[str]

    @property
    def endpoint(self) -> str:
        """Connectable ZeroMQ endpoint for this rank's PUB socket."""
        return _format_tcp_endpoint(self.host, self.port)


def resolve_advertise_host(bind_host: str) -> str:
    """Return a routable host for a server bound to `bind_host`.

    A wildcard bind means "every interface", which is not an address a client can
    connect to, so name this host explicitly instead.
    """
    cleaned = (bind_host or "").strip().strip("[]")
    if cleaned.lower() not in _WILDCARD_HOSTS and cleaned.lower() != "::":
        return cleaned
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except OSError:
        # No resolvable A record; the name is still better than a wildcard.
        return hostname


def events_config(llm: Any) -> Optional[Any]:
    """Return the active streaming `KVEventsConfig`, or None when it is off.

    A `null` publisher is the documented way to build the event path without
    publishing, so it reads the same as disabled here.
    """
    cache_config = getattr(getattr(llm, "args", None), "kv_cache_config", None)
    config = getattr(cache_config, "kv_events_config", None)
    if config is None or not getattr(config, "enable_kv_cache_events", False):
        return None
    if getattr(config, "publisher", None) != "zmq":
        return None
    return config


def data_parallel_size(llm: Any) -> int:
    """Number of ranks that bind a publisher.

    Mirrors `Mapping.dp_size`, which is what actually decides how many sockets
    exist: without attention DP only rank 0 publishes.
    """
    args = getattr(llm, "args", None)
    if not getattr(args, "enable_attention_dp", False):
        return 1
    return max(1, int(getattr(args, "tensor_parallel_size", 1) or 1))


def _split_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    """Split a `tcp://host:port` endpoint, rejecting anything else."""
    if not endpoint.startswith("tcp://"):
        raise KvEventsUnavailableError(
            f"KV cache events are published on {endpoint!r}; only tcp:// endpoints can be "
            "advertised because the protocol addresses a source by host and port"
        )
    remainder = endpoint[len("tcp://") :]
    host, _, port_text = remainder.rpartition(":")
    if not port_text.isdigit():
        raise KvEventsUnavailableError(f"KV cache event endpoint has no port: {endpoint!r}")
    return host.strip("[]"), int(port_text)


def _format_tcp_endpoint(host: str, port: int) -> str:
    """Format a connectable TCP endpoint, including IPv6 brackets."""
    formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"tcp://{formatted_host}:{port}"


def resolve_sources(
    llm: Any, advertise_host: str, rank_hosts: dict[int, str] | None = None
) -> list[ResolvedKvEventSource]:
    """Address every rank's publisher, or return [] when events are disabled.

    Raises:
        KvEventsUnavailableError: Events are on but cannot be addressed.
    """
    config = events_config(llm)
    if config is None:
        return []

    dp_size = data_parallel_size(llm)
    if rank_hosts is not None:
        # Executor transports may encode dictionary keys as JSON strings.
        rank_hosts = {int(rank): host for rank, host in rank_hosts.items()}
        if set(rank_hosts) != set(range(dp_size)) or any(
            not host or host in _WILDCARD_HOSTS for host in rank_hosts.values()
        ):
            raise KvEventsUnavailableError("KV event rank placement is incomplete or invalid")
    if (
        rank_hosts is None
        and dp_size > 1
        and getattr(getattr(llm, "args", None), "orchestrator_type", None) == "ray"
    ):
        raise KvEventsUnavailableError(
            "multi-rank KV event discovery cannot infer per-rank hosts for Ray placement; "
            "use a single-node RPC/MPI deployment"
        )
    gpus_per_node = getattr(getattr(llm, "args", None), "gpus_per_node", None) or dp_size
    if rank_hosts is None and dp_size > gpus_per_node:
        # Ranks bind by *global* rank, so ranks beyond this node bind on a host
        # this process cannot name. Advertising only the local ones would hand a
        # router a partial view of the cache, which is worse than none.
        raise KvEventsUnavailableError(
            f"KV cache events span {dp_size} attention-DP ranks across more than one node "
            f"({gpus_per_node} GPUs per node); this server can only address the ranks on "
            "its own host"
        )

    sources = []
    for rank in range(dp_size):
        rank_host = rank_hosts[rank] if rank_hosts is not None else advertise_host
        endpoint = ZmqEventPublisher.offset_endpoint_port(config.endpoint, rank)
        host, port = _split_tcp_endpoint(endpoint)
        if host.lower() in _WILDCARD_HOSTS:
            host = rank_host
        replay = ZmqEventPublisher.offset_endpoint_port(config.replay_endpoint, rank)
        if replay:
            replay_host, replay_port = _split_tcp_endpoint(replay)
            if replay_host.lower() in _WILDCARD_HOSTS:
                replay_host = rank_host
            replay = _format_tcp_endpoint(replay_host, replay_port)
        sources.append(
            ResolvedKvEventSource(
                data_parallel_rank=rank,
                host=host,
                port=port,
                replay_endpoint=replay or None,
            )
        )
    return sources


def to_proto_source(
    source: ResolvedKvEventSource,
    config: Any,
    schema_version: int,
) -> kv_pb2.KvEventSource:
    """Describe one resolved publisher to a client."""
    return kv_pb2.KvEventSource(
        transport="zmq",
        endpoint_addr=kv_pb2.KvEndpoint(
            host=source.host,
            port=source.port,
            protocol="tcp",
        ),
        topic=config.topic,
        replay_endpoint=source.replay_endpoint or "",
        data_parallel_rank=source.data_parallel_rank,
        encoding="msgpack",
        schema_version=schema_version,
        buffer_steps=config.buffer_steps,
        hwm=config.hwm,
        max_queue_size=config.max_queue_size,
    )
