# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration and endpoint helpers for OpenEngine KV event discovery."""

from typing import Any

_WILDCARD_HOSTS = frozenset({"*", "0.0.0.0", "::", ""})  # nosec B104: compared, never bound


class KvEventsUnavailableError(RuntimeError):
    """The configured event publisher cannot be advertised."""


def events_config(llm: Any) -> Any | None:
    """Return the active ZMQ event configuration, if any."""
    cache_config = getattr(getattr(llm, "args", None), "kv_cache_config", None)
    config = getattr(cache_config, "kv_events_config", None)
    if config is None or not getattr(config, "enable_kv_cache_events", False):
        return None
    return config if getattr(config, "publisher", None) == "zmq" else None


def data_parallel_size(llm: Any) -> int:
    """Return the number of attention-DP KV event publishers."""
    args = getattr(llm, "args", None)
    if not getattr(args, "enable_attention_dp", False):
        return 1
    return max(1, int(getattr(args, "tensor_parallel_size", 1) or 1))


def _split_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    if not endpoint.startswith("tcp://"):
        raise KvEventsUnavailableError(
            f"KV cache events are published on {endpoint!r}; only tcp:// endpoints can be advertised"
        )
    host, _, port_text = endpoint[len("tcp://") :].rpartition(":")
    if not port_text.isdigit():
        raise KvEventsUnavailableError(f"KV cache event endpoint has no port: {endpoint!r}")
    return host.strip("[]"), int(port_text)


def _format_tcp_endpoint(host: str, port: int) -> str:
    formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"tcp://{formatted_host}:{port}"
