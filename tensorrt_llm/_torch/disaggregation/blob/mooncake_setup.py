# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opening Mooncake from a store config: the deployment glue around the driver.

A store config has two sections: ``blob``, handed back untouched for the backend to read, and
``mooncake``, read here. Mooncake reads its environment once, when the process builds its first
client, so every Mooncake store of a process shares those values; this module never sets them.

Over RDMA, a process that used ibverbs before its first Mooncake client (importing
``tensorrt_llm`` does) logs ``RDMA context setup failed: fork compatibility: Invalid argument``
once: ``ibv_fork_init`` only works before any other verbs call. It is harmless on Linux 5.13 and
later, which copy pinned pages on fork, and the devices come up normally.
"""

from __future__ import annotations

import importlib.metadata
import os
import socket
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from tensorrt_llm.logger import logger

from .mooncake import MooncakeDriver

__all__ = ["MooncakeConfig", "load_store_config", "open_mooncake_store", "open_sdk"]

SDK_DISTRIBUTION = "mooncake-transfer-engine-cuda13"
SDK_VERSION = "0.3.13"
IB_SYSFS_ROOT = "/sys/class/infiniband"
MASTER_WAIT_S = 60.0
MASTER_RETRY_S = 0.5
SECTIONS = ("blob", "mooncake")

# Logged before setup, as Mooncake will read them.
_LOGGED_ENVIRONMENT = (
    "MC_STORE_MEMCPY",
    "MOONCAKE_STORE_CHECKSUM",
    "WITH_NVIDIA_PEERMEM",
    "MC_TCP_SLICE_SIZE",
    "MC_TCP_PROTO",
    "MC_STORE_LOCAL_HOT_CACHE_SIZE",
)
# Reserved for documentation (RFC 5737): a datagram socket connected to it reveals the route off
# this host without sending anything.
_ROUTE_PROBE_ENDPOINT = ("192.0.2.1", 9)


@dataclass(frozen=True)
class MooncakeConfig:
    """The ``mooncake`` section of a store config.

    Attributes:
        master_server_address: The master's ``host:port``.
        global_segment_size: Bytes this process lends the pool. Required: the objects in a
            process's segment go when the process does; 0 leaves holding objects to others.
        protocol: ``"rdma"`` or ``"tcp"``.
        device_name: Comma-separated RDMA devices; empty picks every active InfiniBand device at
            the highest rate.
        local_hostname: The address peers reach this process at; empty takes the address the
            route to the master leaves from.
        metadata_server: Where segments are described; ``P2PHANDSHAKE`` needs no server.
        local_buffer_size: Bytes for the SDK's copying calls, which the driver never makes; 0 is
            allowed.
    """

    master_server_address: str
    global_segment_size: int
    protocol: str = "rdma"
    device_name: str = ""
    local_hostname: str = ""
    metadata_server: str = "P2PHANDSHAKE"
    local_buffer_size: int = 16 << 20

    def __post_init__(self) -> None:
        for name in ("master_server_address", "device_name", "local_hostname", "metadata_server"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a str, got {getattr(self, name)!r}")
        if self.protocol not in ("rdma", "tcp"):
            raise ValueError(f"protocol must be rdma or tcp, got {self.protocol!r}")
        for name, least in (("global_segment_size", 0), ("local_buffer_size", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"{name} must be an int of at least {least}, got {value!r}")


def load_store_config(
    config: str | os.PathLike | Mapping,
) -> tuple[Mapping[str, object], MooncakeConfig]:
    """Reads a store config: a YAML file, or the mapping parsed from one.

    Returns:
        The ``blob`` section as given (empty when absent) and the ``mooncake`` section.

    Raises:
        ValueError: An unknown section or key, a missing ``mooncake`` section or required key, or
            a value of the wrong type.
    """
    document = config
    if not isinstance(document, Mapping):
        with open(config) as file:
            document = yaml.safe_load(file)
    if not isinstance(document, Mapping):
        raise ValueError(f"store config {config} is not a mapping")
    unknown = sorted(set(document) - set(SECTIONS))
    if unknown:
        raise ValueError(f"store config: unknown sections {unknown}; it has {list(SECTIONS)}")
    blob, section = document.get("blob") or {}, document.get("mooncake")
    if not isinstance(blob, Mapping) or not isinstance(section, Mapping):
        raise ValueError("store config: 'mooncake' is a required mapping, 'blob' an optional one")
    try:
        return blob, MooncakeConfig(**section)
    except (TypeError, ValueError) as error:  # an unknown or missing key, or a bad value
        raise ValueError(f"store config section 'mooncake': {error}") from error


def open_sdk(config: MooncakeConfig) -> Any:
    """Opens a set-up ``MooncakeDistributedStore``, on the builder thread: ``setup`` holds the GIL
    while it mounts this process's segment.

    Raises:
        ValueError: TCP would use its legacy framing.
        ImportError: ``mooncake.store`` does not import.
        TimeoutError: The master does not accept connections.
        RuntimeError: ``setup`` failed; the half-open client is closed.
    """
    _refuse_legacy_tcp_framing(config.protocol)
    _log_environment()
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as error:
        raise ImportError(_sdk_missing(error)) from error
    _warn_unless_pinned_version()
    _wait_for_master(config.master_server_address)
    store = MooncakeDistributedStore()
    code = store.setup(
        config.local_hostname or _local_address(config.master_server_address),
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        _devices(config),
        config.master_server_address,
    )
    if code != 0:
        store.close()  # no registration and no call in flight yet, unlike a client in service
        raise RuntimeError(
            f"Mooncake setup failed with {code} (master {config.master_server_address}, "
            f"protocol {config.protocol})"
        )
    return store


def open_mooncake_store(config: object) -> tuple[MooncakeDriver, Mapping[str, object]]:
    """``BlobAssembly``'s ``open_driver`` for Mooncake: the driver, and the ``blob`` section."""
    blob_section, mooncake_config = load_store_config(config)
    store = open_sdk(mooncake_config)
    return MooncakeDriver(store), blob_section


def _refuse_legacy_tcp_framing(protocol: str) -> None:
    """Under ``MC_TCP_PROTO=1`` a TCP write completes once its last chunk is in the local socket
    buffer, so a put could be READY before its bytes landed. The SDK reads the variable on every
    transfer: setting it after this check goes unnoticed."""
    if protocol == "tcp" and os.environ.get("MC_TCP_PROTO") == "1":
        raise ValueError("MC_TCP_PROTO=1 selects TCP framing that reports writes before they land")


def _log_environment() -> None:
    values = ", ".join(f"{name}={os.environ.get(name, '<unset>')}" for name in _LOGGED_ENVIRONMENT)
    logger.info(f"mooncake: frozen at the process's first client: {values}")
    hot_cache = os.environ.get("MC_STORE_LOCAL_HOT_CACHE_SIZE", "0")
    if hot_cache.isdigit() and int(hot_cache) > 0:
        logger.warning("mooncake: the local hot cache is on: every get also copies into the cache")


def _sdk_missing(error: ImportError) -> str:
    hint = f"Install {SDK_DISTRIBUTION}=={SDK_VERSION} and put it before NIXL on the path."
    shadow = getattr(sys.modules.get("mooncake"), "__file__", None)
    if shadow:
        hint = f"The 'mooncake' found is {shadow}; NIXL ships one that shadows the SDK's. {hint}"
    return f"cannot import mooncake.store ({error}). {hint}"


def _warn_unless_pinned_version() -> None:
    try:
        version = importlib.metadata.version(SDK_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = "not an installed distribution"
    if version != SDK_VERSION:
        logger.warning(
            f"mooncake: {SDK_DISTRIBUTION} is {version}; the driver's code mapping and access-end "
            f"rule follow {SDK_VERSION}"
        )


# Three deployment helpers, with the same rules as the Mooncake connector's
# (pyexecutor/connectors/mooncake_store: master.py and config.py), which this package cannot import.


def _devices(config: MooncakeConfig) -> str:
    """``device_name``, or for RDMA left empty every active InfiniBand device at the highest rate:
    the compute fabric, without the Ethernet ports that share an HCA."""
    if config.device_name or config.protocol != "rdma":
        return config.device_name
    detected = _fastest_active_ib_devices(IB_SYSFS_ROOT)
    if not detected:
        logger.warning(f"mooncake: no active InfiniBand device under {IB_SYSFS_ROOT}")
    return ",".join(detected)


def _fastest_active_ib_devices(sysfs_root: str) -> list[str]:
    try:
        devices = sorted(os.listdir(sysfs_root))
    except OSError:
        return []
    rates = {}
    for device in devices:
        port = os.path.join(sysfs_root, device, "ports", "1")
        link, state = _read_attribute(port, "link_layer"), _read_attribute(port, "state")
        rate = _read_attribute(port, "rate").split()  # "400 Gb/sec (4X NDR)"
        if link == "InfiniBand" and "ACTIVE" in state and rate and rate[0].isdigit():
            rates[device] = int(rate[0])
    fastest = max(rates.values(), default=None)
    return [device for device, rate in rates.items() if rate == fastest]


def _read_attribute(directory: str, name: str) -> str:
    try:
        with open(os.path.join(directory, name)) as file:
            return file.read().strip()
    except OSError:
        return ""


def _local_address(master: str) -> str:
    """The address the route to the master leaves from. Not ``gethostbyname(gethostname())``,
    which answers 127.0.1.1 where /etc/hosts maps the host's name to loopback."""
    for endpoint in (_split_address(master), _ROUTE_PROBE_ENDPOINT):
        if endpoint is None:
            continue
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(endpoint)
                address = probe.getsockname()[0]
        except OSError:
            continue
        if not address.startswith("127."):
            return address
    return socket.gethostbyname(socket.gethostname())


def _wait_for_master(address: str) -> None:
    """Turns an unreachable master into one line naming it, instead of a bare code from setup."""
    endpoint = _split_address(address)
    if endpoint is None:
        logger.warning(f"mooncake: cannot check master_server_address {address!r}: not host:port")
        return
    deadline = time.monotonic() + MASTER_WAIT_S
    while True:
        try:
            with socket.create_connection(endpoint, timeout=1.0):
                return
        except OSError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"the Mooncake master at {address} did not accept connections within "
                    f"{MASTER_WAIT_S:g} s: {error}"
                ) from error
        time.sleep(MASTER_RETRY_S)


def _split_address(address: str) -> tuple[str, int] | None:
    host, _, port = address.rpartition(":")
    return (host.strip("[]"), int(port)) if host and port.isdigit() else None
