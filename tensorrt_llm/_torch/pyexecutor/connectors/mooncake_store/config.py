# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for the Mooncake store KV cache connector.

Read from the JSON file named by `MOONCAKE_CONFIG_PATH`, the same file and
environment variable the vLLM Mooncake store connector uses, so one deployment
can point both engines at the same pool.

Pool-wide settings come from the manifest its master publishes; what stays per
server is the traffic it drives and the memory it lends. `master.provision_pool`
merges the two into the file this reads back.
"""

import json
import os
import re
import socket
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MooncakeStoreConfig, TorchLlmArgs

__all__ = [
    "CLIENT_CONFIG_NAME",
    "CONFIG_PATH_ENV",
    "MooncakeStoreConnectorConfig",
    "SEGMENTS_DIR_NAME",
    "StoreRole",
    "local_address",
    "parse_size",
    "pool_config",
    "provisioned_config_path",
    "split_address",
]

#: Mooncake's own variable for the client config path, read by the vLLM
#: connector too. `master.provision_pool` exports it for the ranks the LLM
#: constructor spawns.
CONFIG_PATH_ENV = "MOONCAKE_CONFIG_PATH"
#: Name the rendered client config takes in the run directory.
CLIENT_CONFIG_NAME = "mooncake.json"
#: Records which server rendered the client config in a run directory; see
#: `master.claim_run_dir`.
RUN_DIR_OWNER_NAME = "owner.json"
#: Subdirectory of the run directory where each rank records the segment it
#: mounted; see `ledger.py`.
SEGMENTS_DIR_NAME = "segments"

DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200
#: Per-process Mooncake transfer buffer, not pool capacity. Written into the
#: rendered config so a vLLM reader of the same file sees the same value.
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824
DEFAULT_NAMESPACE = "trtllm"
#: Mooncake's peer-to-peer handshake, which keeps a separate metadata process
#: out of the deployment. An empty connstring is not a form `store.setup`
#: accepts, so this is the only sensible fallback.
DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"

#: Suffixes this parser reads as powers of 1000. vLLM's Mooncake config parser
#: reads the same spellings as powers of 1024, so `strict_units` rejects them
#: in a file both engines may open.
_DECIMAL_UNITS = {
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "t": 1000**4,
    "tb": 1000**4,
}
_BINARY_UNITS = {
    "": 1,
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}
_SIZE_UNITS = {**_BINARY_UNITS, **_DECIMAL_UNITS}
#: The binary spelling to recommend for each rejected decimal one.
_BINARY_SPELLING = {
    "k": "KiB",
    "kb": "KiB",
    "m": "MiB",
    "mb": "MiB",
    "g": "GiB",
    "gb": "GiB",
    "t": "TiB",
    "tb": "TiB",
}
_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")

#: Stands in for "somewhere off this host" when there is no peer to route to.
#: RFC 5737 reserves it for documentation, so it names no real destination.
#: That costs nothing: connecting a datagram socket consults the routing table
#: without sending a packet.
_ROUTE_PROBE_ENDPOINT = ("192.0.2.1", 9)


def split_address(address: str) -> Optional[Tuple[str, int]]:
    """Split `host:port`, or return `None` if it is not in that form."""
    host, separator, port = address.rpartition(":")
    if not separator or not port.isdigit():
        return None
    return host.strip("[]"), int(port)


def local_address(peer: Optional[str] = None) -> str:
    """This host's address as something else on the pool's network reaches it.

    Every segment is registered under this address and the pool manifest names
    the master by it, so peers dial whatever comes back here. That rules out
    `gethostbyname(gethostname())`, which answers `127.0.1.1` on a host whose
    `/etc/hosts` maps its own name to loopback, as Ubuntu's default does. Such
    an answer leaves a multi-node pool unreachable while a single-node run
    keeps working.

    Args:
        peer: `host:port` of something this address has to be reachable from,
            usually the master. The route to it names the interface to report.
            The master itself has no peer to measure against and passes
            nothing, which falls back to the route off this host.

    Returns:
        A dotted-quad address, or the hostname lookup when there is no route
        at all. A deployment whose address cannot be derived this way pins it
        with `mooncake_store.local_hostname`.
    """
    endpoints = []
    if peer:
        endpoint = split_address(peer)
        if endpoint is not None:
            endpoints.append(endpoint)
    endpoints.append(_ROUTE_PROBE_ENDPOINT)
    for endpoint in endpoints:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(endpoint)
                address = probe.getsockname()[0]
        except OSError:
            # No route to it, or it is not an IPv4 address: try the next.
            continue
        if address and not address.startswith("127."):
            return address
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


class StoreRole(Enum):
    """Which directions of traffic this engine is allowed to drive.

    Traffic is separate from capacity: every role mounts the segment its config
    asks for, and the role says only what the engine then does with the pool.

    A disaggregated deployment typically runs context servers as `both` and
    generation servers as `capacity`: generated tokens are rarely a reused
    prefix, and prompt KV reaches decode over the cache transceiver rather than
    through the pool, so decode-side traffic would cost bandwidth for no hits.
    """

    PRODUCER = "producer"
    CONSUMER = "consumer"
    BOTH = "both"
    #: Mounts a segment and transfers nothing.
    CAPACITY = "capacity"

    @property
    def loads(self) -> bool:
        """Whether this role reads previously stored KV back onto the GPU."""
        return self in (StoreRole.CONSUMER, StoreRole.BOTH)

    @property
    def saves(self) -> bool:
        """Whether this role writes newly computed KV into the store."""
        return self in (StoreRole.PRODUCER, StoreRole.BOTH)

    @property
    def transfers(self) -> bool:
        """Whether this role moves KV at all.

        False only for `CAPACITY`, which short-circuits every transfer path
        including KV registration.
        """
        return self.loads or self.saves


def parse_size(value: Any, *, strict_units: bool = False) -> int:
    """Accept either a byte count or a suffixed string such as `"4GiB"`.

    Args:
        value: A byte count, or a magnitude with a unit suffix.
        strict_units: Reject `GB`/`MB`/`KB`/`TB` and their one-letter forms.
            Set this wherever the value comes from, or goes into, a file the
            vLLM connector may also read, since it scales those suffixes by
            1024 rather than 1000.

    Returns:
        The size in bytes.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a size, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = _SIZE_RE.match(str(value))
    if match is None:
        raise ValueError(f"cannot parse size {value!r}")
    magnitude, unit = match.groups()
    normalized = unit.lower()
    if strict_units and normalized in _DECIMAL_UNITS:
        raise ValueError(
            f"size {value!r} uses the ambiguous unit {unit!r}, which this "
            f"parser reads as a power of 1000 and vLLM's Mooncake parser reads "
            f"as a power of 1024. Write "
            f"{magnitude}{_BINARY_SPELLING[normalized]} for the binary size, "
            f"or a plain byte count; both mean the same thing to either engine."
        )
    scale = _SIZE_UNITS.get(normalized)
    if scale is None:
        raise ValueError(f"unknown size unit {unit!r} in {value!r}")
    return int(float(magnitude) * scale)


def pool_config(llm_args: "TorchLlmArgs") -> Optional["MooncakeStoreConfig"]:
    """The `mooncake_store` block of `llm_args`, if the deployment set one.

    Every rank parses the same worker config, so this is how a rank reaches
    settings the process that rendered the client config could not pass it.
    """
    connector = llm_args.kv_connector_config
    return connector.mooncake_store if connector is not None else None


def provisioned_config_path(run_dir: Optional[str]) -> Optional[str]:
    """The client config a server rendered into `run_dir`, if there is one.

    Ranks an external launcher started were already running when
    `provision_pool` exported `MOONCAKE_CONFIG_PATH`, so they read the config
    back from the run directory instead. That needs the config to have named
    the directory, since it otherwise defaults to a per-process temporary one.
    """
    if not run_dir:
        return None
    path = os.path.join(run_dir, CLIENT_CONFIG_NAME)
    return path if os.path.exists(path) else None


@dataclass(frozen=True)
class MooncakeStoreConnectorConfig:
    """Everything needed to open a store handle and name keys in it."""

    master_server_address: str
    metadata_server: str = DEFAULT_METADATA_SERVER
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    local_hostname: Optional[str] = None
    tenant_id: Optional[str] = None
    role: StoreRole = StoreRole.BOTH
    #: Key namespace for the pool. Two deployments that should not share cache
    #: set different values; bump it after any change to page layout or
    #: contents.
    namespace: str = DEFAULT_NAMESPACE
    #: Checkpoint identity the keys are namespaced by; see
    #: :meth:`resolve_model_key`.
    model_key: Optional[str] = None
    #: How many page keys go into one store call. Bounds the size of a single
    #: RPC without bounding how much a request may transfer.
    transfer_batch_size: int = 64
    #: Pass pages through a pinned host buffer instead of registering the KV
    #: pools with Mooncake. The only transfer path supported today, so a False
    #: here is raised back to True.
    stage_through_host: bool = True

    def __post_init__(self) -> None:
        """Reject settings that would fail later, inside a transfer."""
        if not self.stage_through_host:
            if self.role.transfers:
                logger.warning(
                    "mooncake-store: ignoring stage_through_host=False. Pages "
                    "pass through pinned host memory; registering the KV pools "
                    "takes one registration per pool mapping, which this "
                    "connector does not do yet."
                )
            object.__setattr__(self, "stage_through_host", True)
        if not self.master_server_address:
            raise ValueError("master_server_address is required")
        if self.local_buffer_size <= 0:
            raise ValueError("local_buffer_size must be > 0")
        if self.global_segment_size < 0:
            raise ValueError("global_segment_size must be >= 0")
        if self.transfer_batch_size <= 0:
            raise ValueError("transfer_batch_size must be > 0")

    @property
    def capacity_only(self) -> bool:
        """Whether this rank lends memory without transferring any KV."""
        return not self.role.transfers

    @staticmethod
    def from_file(path: str) -> "MooncakeStoreConnectorConfig":
        """Read the topology from a vLLM-compatible Mooncake JSON config."""
        with open(path) as handle:
            raw = json.load(handle)
        role = str(raw.get("role", StoreRole.BOTH.value)).strip().lower()
        try:
            parsed_role = StoreRole(role)
        except ValueError as exc:
            known = ", ".join(member.value for member in StoreRole)
            raise ValueError(f"role={role!r} in {path} is not one of: {known}") from exc
        return MooncakeStoreConnectorConfig(
            master_server_address=raw.get("master_server_address", ""),
            metadata_server=raw.get("metadata_server") or DEFAULT_METADATA_SERVER,
            protocol=raw.get("protocol", "rdma"),
            device_name=raw.get("device_name", ""),
            global_segment_size=parse_size(
                raw.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE), strict_units=True
            ),
            local_buffer_size=parse_size(
                raw.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE), strict_units=True
            ),
            local_hostname=raw.get("local_hostname") or None,
            tenant_id=raw.get("tenant_id") or None,
            role=parsed_role,
            namespace=str(raw.get("namespace", DEFAULT_NAMESPACE)),
            model_key=raw.get("model_key") or None,
            transfer_batch_size=int(raw.get("transfer_batch_size", 64)),
            stage_through_host=bool(raw.get("stage_through_host", True)),
        )

    @staticmethod
    def resolve(llm_args: "TorchLlmArgs") -> "MooncakeStoreConnectorConfig":
        """The client config this rank should open, read from wherever it is.

        An inherited `MOONCAKE_CONFIG_PATH` wins, since it names a pool the
        deployment provisioned itself. Otherwise the config is the one this
        server rendered into `mooncake_store.run_dir`.
        """
        pool = pool_config(llm_args)
        run_dir = pool.run_dir if pool is not None else None
        path = os.getenv(CONFIG_PATH_ENV) or provisioned_config_path(run_dir)
        if not path:
            raise ValueError(
                f"The mooncake-store connector needs {CONFIG_PATH_ENV} set to a "
                "Mooncake JSON config (metadata_server, master_server_address, "
                "protocol, device_name, global_segment_size, local_buffer_size), "
                "or kv_connector_config.mooncake_store set so the server renders "
                "one. Set mooncake_store.run_dir as well if this rank was "
                "started by the launcher rather than spawned by the server, "
                "since it cannot inherit the path."
            )
        return MooncakeStoreConnectorConfig.from_file(path)

    def resolve_model_key(self, model: Any) -> str:
        """The model identity to namespace keys by.

        Deliberately has no default. Deriving one from the model path would let
        two checkpoints that share a directory name, such as `org-a/model` and
        `org-b/model`, agree on a namespace while disagreeing on what the pages
        mean, and each would read the other's KV as its own.
        """
        if self.model_key:
            return self.model_key
        raise ValueError(
            f"The mooncake-store connector needs a model key to namespace its "
            f"pool keys by, and there is no safe default: set the model_key "
            f"field of the Mooncake JSON config. Give it a value that "
            f"separates this checkpoint from any other an engine sharing the "
            f"pool might load, rather than one derived from {model!r}."
        )
