# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opening Mooncake from a store config: the deployment glue around the driver.

A store config has two sections: ``blob``, read by the caller's reader, and ``mooncake``, read
here; both are checked before the SDK is opened. Mooncake reads its own environment variables, some
once per process (``MOONCAKE_STORE_CHECKSUM``), some per client (``MC_STORE_MEMCPY``,
``MC_USE_TENT``) and ``MC_TCP_PROTO`` on every transfer; this module logs them before setup and
never sets them. ``open_sdk`` refuses the settings the driver's rules do not cover: TCP's legacy
framing (``MC_TCP_PROTO=1``), the local hot cache (``MC_STORE_LOCAL_HOT_CACHE_SIZE``), the newer
transfer engine (``MC_USE_TENT``, ``MC_USE_TEV1``) and any SDK build other than the pinned ones.

Over RDMA, a process that used ibverbs before its first Mooncake client (importing
``tensorrt_llm`` does) logs ``RDMA context setup failed: fork compatibility: Invalid argument``
once: ``ibv_fork_init`` only works before any other verbs call. It is harmless on Linux 5.13 and
later, which copy pinned pages on fork, and the devices come up normally.
"""

from __future__ import annotations

import importlib.metadata
import os
import pathlib
import re
import socket
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import yaml

from tensorrt_llm.logger import logger

from .mooncake import DEFAULT_BYTES_PER_CALL, DEFAULT_ROWS_PER_CALL, DEFAULT_WORKERS, MooncakeDriver

__all__ = [
    "MooncakeConfig",
    "load_store_config",
    "open_mooncake_driver",
    "open_sdk",
]

RECOMMENDED_SDK_DISTRIBUTION = "mooncake-transfer-engine-cuda13"  # the one TensorRT-LLM installs
# The builds of the SDK with the transports the driver's rules were read for (RDMA and TCP).
PINNED_SDK_DISTRIBUTIONS = (
    RECOMMENDED_SDK_DISTRIBUTION,
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-non-cuda",
)
PINNED_SDK_VERSION = "0.3.13"
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
    "MC_USE_TENT",
    "MC_USE_TEV1",
)
BlobSectionT = TypeVar("BlobSectionT")
# Reserved for documentation (RFC 5737): a datagram socket connected to it reveals the route off
# this host without sending anything.
_ROUTE_PROBE_ENDPOINT = ("192.0.2.1", 9)


@dataclass(frozen=True)
class MooncakeConfig:
    """The ``mooncake`` section of a store config.

    Attributes:
        master_server_address: The master's ``host:port``.
        global_segment_size: Bytes of this process's memory that hold pool objects (its
            segment). Required: the objects in it go when the process does; 0 leaves holding
            objects to other processes.
        protocol: ``"rdma"`` or ``"tcp"``.
        device_name: Comma-separated RDMA devices; empty picks every active InfiniBand device at
            the highest rate.
        local_hostname: The address peers reach this process at; empty takes the address the
            route to the master leaves from.
        metadata_server: Where segments are described; ``P2PHANDSHAKE`` needs no server.
        local_buffer_size: Bytes for the SDK's copying calls, which the driver never makes; 0 is
            allowed, and the default is the SDK's.
        workers: The most threads the driver runs SDK calls on. One put call does not fill the
            links, and each running get busy-polls a core; with two workers or more, puts leave
            one to gets.
        rows_per_call: The most rows of one SDK call; the driver cuts larger operations into
            several. The SDK converts a call's Python lists while holding the GIL, so this bounds
            how long one call holds it.
        bytes_per_call: The most payload bytes of one SDK call that moves rows. Larger calls return
            into Python fewer times per byte, and next to a busy executor thread each return waits
            for the GIL; but a call must end well inside the read lease (10 s), and an operation
            holds all its slots until its last call returned, or for good once a get answers
            TRANSFER_FAIL.

    The defaults of the last three were measured on GB200 over RDMA.
    """

    master_server_address: str
    global_segment_size: int
    protocol: Literal["rdma", "tcp"] = "rdma"
    device_name: str = ""
    local_hostname: str = ""
    metadata_server: str = "P2PHANDSHAKE"
    local_buffer_size: int = 16 << 20
    workers: int = DEFAULT_WORKERS
    rows_per_call: int = DEFAULT_ROWS_PER_CALL
    bytes_per_call: int = DEFAULT_BYTES_PER_CALL

    def __post_init__(self) -> None:
        for name in ("master_server_address", "device_name", "local_hostname", "metadata_server"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a str, got {getattr(self, name)!r}")
        if "://" in self.master_server_address:  # the waits below need one host:port
            raise ValueError(
                "master_server_address must be one master's host:port; an HA master list "
                f"is not supported, got {self.master_server_address!r}"
            )
        if self.protocol not in ("rdma", "tcp"):
            raise ValueError(f"protocol must be rdma or tcp, got {self.protocol!r}")
        for name, least in (
            ("global_segment_size", 0),
            ("local_buffer_size", 0),
            ("workers", 1),
            ("rows_per_call", 1),
            ("bytes_per_call", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"{name} must be an int of at least {least}, got {value!r}")


def load_store_config(
    config: str | os.PathLike | Mapping,
) -> tuple[Mapping[str, object], MooncakeConfig]:
    """Reads a store config: a YAML file, or the mapping parsed from one.

    Returns:
        The ``blob`` section (empty when absent), and the ``mooncake`` section.

    Raises:
        ValueError: The file cannot be read or is not YAML, an unknown section or key, a missing
            ``mooncake`` section or required key, or a value of the wrong type.
    """
    document = config
    if not isinstance(document, Mapping):
        try:
            with open(config) as file:
                document = yaml.safe_load(file)
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"store config {config}: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"store config {config} is not a mapping")
    unknown = sorted(set(document) - set(SECTIONS), key=str)
    if unknown:
        raise ValueError(f"store config: unknown sections {unknown}; it has {list(SECTIONS)}")
    blob = document.get("blob")
    blob = {} if blob is None else blob  # ``blob:`` with nothing under it is an empty section
    section = document.get("mooncake")
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
        ValueError: TCP would use its legacy framing, the local hot cache is on, or the newer
            transfer engine is selected.
        ImportError: ``mooncake.store`` does not import, or is not SDK 0.3.13.
        TimeoutError: The master does not accept connections.
        RuntimeError: ``setup`` failed; the half-open client is closed.
    """
    _refuse_legacy_tcp_framing(config.protocol)
    _refuse_hot_cache()
    _refuse_tent()
    _log_environment()
    try:
        import mooncake.store as sdk
    except ImportError as error:
        raise ImportError(_sdk_import_error_message(error)) from error
    _require_pinned_builds(sdk)
    _wait_for_master(config.master_server_address)
    store = sdk.MooncakeDistributedStore()
    code = store.setup(
        local_hostname=config.local_hostname or _local_address(config.master_server_address),
        metadata_server=config.metadata_server,
        global_segment_size=config.global_segment_size,
        local_buffer_size=config.local_buffer_size,
        protocol=config.protocol,
        rdma_devices=_rdma_devices(config),
        master_server_addr=config.master_server_address,
    )
    if code != 0:
        store.close()  # no registration and no call in flight yet, unlike a client in service
        raise RuntimeError(
            f"Mooncake setup failed with {code} (master {config.master_server_address}, "
            f"protocol {config.protocol})"
        )
    return store


def open_mooncake_driver(
    config: str | os.PathLike | Mapping,
    read_blob_section: Callable[[Mapping[str, object]], BlobSectionT],
) -> tuple[MooncakeDriver, BlobSectionT]:
    """Opens the Mooncake driver a store config describes, once both its sections are read: opening
    waits for the master and mounts this process's segment, so a bad config fails before that.

    The caller owns the driver and closes it last, with ``MooncakeDriver.close``, which also closes
    the SDK client.

    Args:
        config: A store config, as ``load_store_config`` takes it.
        read_blob_section: Reads the ``blob`` section (for the blob backend, into a
            ``BlobConfig``); what it returns comes back with the driver.

    Raises:
        ValueError: A section is wrong, as ``load_store_config`` or ``read_blob_section`` found.
        As ``open_sdk`` otherwise.
    """
    blob_section, mooncake_config = load_store_config(config)
    try:
        blob = read_blob_section(blob_section)
    except (TypeError, ValueError) as error:  # an unknown key, or a bad value
        raise ValueError(f"store config section 'blob': {error}") from error
    store = open_sdk(mooncake_config)
    try:
        driver = MooncakeDriver(
            store,
            workers=mooncake_config.workers,
            rows_per_call=mooncake_config.rows_per_call,
            bytes_per_call=mooncake_config.bytes_per_call,
        )
    except BaseException:
        store.close()  # no region was registered and no call made: the client is not in service
        raise
    return driver, blob


def _refuse_legacy_tcp_framing(protocol: str) -> None:
    """Under ``MC_TCP_PROTO=1`` a TCP write completes once its last chunk is in the local socket
    buffer, so a put could be READY before its bytes landed. The SDK reads the variable on every
    transfer: setting it after this check goes unnoticed. The same legacy framing is used toward a
    segment whose process runs an SDK older than 0.3.12, which this process cannot check, so every
    process of the pool must run 0.3.12 or later (``tcp_transport.cpp:720``,
    ``transfer_metadata.cpp:644-646``)."""
    if protocol == "tcp" and os.environ.get("MC_TCP_PROTO") == "1":
        raise ValueError("MC_TCP_PROTO=1 selects TCP framing that reports writes before they land")


def _refuse_hot_cache() -> None:
    """The SDK copies the slot a get read into its hot cache before it checks the read lease, and
    never drops the copy when the master evicts the object. So a read answered LEASE_EXPIRED, torn
    by a reuse of the object's memory, stays cached; once the key is put again, every get of it
    here is served those bytes as a whole object. Every value but empty or "0" is refused: the SDK
    reads it with ``stoull``, which also turns the cache on for values like "64MB" or " 64"."""
    if os.environ.get("MC_STORE_LOCAL_HOT_CACHE_SIZE", "0").strip() not in ("", "0"):
        raise ValueError("MC_STORE_LOCAL_HOT_CACHE_SIZE turns on a cache that can serve torn reads")


def _refuse_tent() -> None:
    """Either variable, set to any value, makes the SDK build its transfer engine on TENT, the newer
    engine, whose completion rules the driver was not read against (``mooncake`` module
    docstring). The SDK reads them when a client is set up."""
    selected = [name for name in ("MC_USE_TENT", "MC_USE_TEV1") if name in os.environ]
    if selected:
        raise ValueError(
            f"{', '.join(selected)} selects a transfer engine the driver does not cover"
        )


def _log_environment() -> None:
    values = ", ".join(f"{name}={os.environ.get(name, '<unset>')}" for name in _LOGGED_ENVIRONMENT)
    logger.info(f"mooncake: environment at setup: {values}")


def _sdk_import_error_message(error: ImportError) -> str:
    install = f"{RECOMMENDED_SDK_DISTRIBUTION}=={PINNED_SDK_VERSION}"
    hint = f"Install {install} and put it before NIXL on the path."
    shadow = getattr(sys.modules.get("mooncake"), "__file__", None)
    if shadow:
        hint = f"The 'mooncake' found is {shadow}; NIXL ships one that shadows the SDK's. {hint}"
    return f"cannot import mooncake.store ({error}). {hint}"


def _require_pinned_builds(sdk: Any) -> None:
    """The driver's code mapping and access-end rules were read from 0.3.13 with the RDMA and TCP
    transports; on another version or build a put could return early, and a revoke then free
    space a transfer still writes. So the imported ``mooncake.store`` must be shipped by one of
    ``PINNED_SDK_DISTRIBUTIONS`` at 0.3.13, not by another build, a source tree or a shadowing
    package. Builds installed side by side share the module's path and the file is the last one's,
    so every build that lists it must be pinned."""
    builds = _builds_listing_file(sdk)
    pinned = {(name, PINNED_SDK_VERSION) for name in PINNED_SDK_DISTRIBUTIONS}
    if not builds or not set(builds) <= pinned:
        found = ", ".join(f"{name}=={version}" for name, version in builds)
        raise ImportError(
            f"mooncake.store ({getattr(sdk, '__file__', 'no file')}) comes from "
            f"{found or 'no installed distribution'}; the driver follows {PINNED_SDK_VERSION} of "
            f"{', '.join(PINNED_SDK_DISTRIBUTIONS)} only. Install "
            f"{RECOMMENDED_SDK_DISTRIBUTION}=={PINNED_SDK_VERSION}."
        )


def _builds_listing_file(module: Any) -> list[tuple[str, str]]:
    """Every installed distribution that lists ``module``'s file, by normalized name and version,
    including two of one name, as a reinstall over another version can leave."""
    file = getattr(module, "__file__", None)
    if file is None:
        return []
    path = pathlib.Path(file).resolve()
    builds = []
    for build in importlib.metadata.distributions():
        files = [f for f in build.files or () if f.parts[:1] == ("mooncake",)]
        if any(pathlib.Path(build.locate_file(f)).resolve() == path for f in files):
            name = build.metadata.get("Name") or "a distribution without a name"
            builds.append((re.sub(r"[-_.]+", "-", name).lower(), build.version))
    return builds


# Three deployment helpers, with the same rules as the Mooncake connector's
# (pyexecutor/connectors/mooncake_store: master.py and config.py), which this package cannot import.


def _rdma_devices(config: MooncakeConfig) -> str:
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
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


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
