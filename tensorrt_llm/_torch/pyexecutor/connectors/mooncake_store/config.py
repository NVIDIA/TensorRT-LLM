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

Topology settings are read from the JSON file named by `MOONCAKE_CONFIG_PATH`,
the same file and environment variable the vLLM Mooncake store connector uses,
so one deployment can point both engines at the same pool.

Settings that are the pool's rather than any engine's, such as which master
owns it and how it is reached, come from the manifest that master publishes, so
they cannot drift between participants. What stays per server is the traffic it
drives and the memory it lends. See `master.provision_pool`, which merges the
two into the file this reads back.
"""

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

__all__ = [
    "CLIENT_CONFIG_NAME",
    "CONFIG_PATH_ENV",
    "MooncakeStoreConnectorConfig",
    "ROLE_ENV",
    "RUN_DIR_ENV",
    "SEGMENTS_DIR_NAME",
    "STAGE_THROUGH_HOST_ENV",
    "StoreRole",
    "parse_size",
    "provisioned_config_path",
]

CONFIG_PATH_ENV = "MOONCAKE_CONFIG_PATH"
#: Where a server keeps the client config it renders and the master's log. Set
#: it to keep them after shutdown; otherwise they live in a temporary directory.
RUN_DIR_ENV = "TRTLLM_MOONCAKE_RUN_DIR"
#: Name the rendered client config takes in the run directory.
CLIENT_CONFIG_NAME = "mooncake.json"
#: Subdirectory of the run directory where each rank records the segment it
#: mounted. Telemetry reads it instead of scraping logs; see `segment_ledger`.
SEGMENTS_DIR_NAME = "segments"
ROLE_ENV = "TRTLLM_MOONCAKE_STORE_ROLE"
STAGE_THROUGH_HOST_ENV = "TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST"

DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200
#: Per-process Mooncake transfer buffer. Not pool capacity, and not worth a
#: knob: it bounds one client's scratch space, which no deployment has had
#: reason to tune. Still written into the rendered config so a vLLM reader of
#: the same file sees the same value rather than falling back to its own.
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824
DEFAULT_NAMESPACE = "trtllm"
#: Mooncake's own peer-to-peer handshake, which keeps a separate metadata
#: process out of the deployment. Nothing else is a sensible fallback: an empty
#: connstring is not one of the forms `store.setup` accepts, so a config that
#: leaves the field out means this rather than meaning no metadata service.
DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

#: Suffixes this parser reads as powers of 1000. vLLM's Mooncake config parser
#: reads the same spellings as powers of 1024, so a file both engines may open
#: must not use them; `strict_units` rejects them for that reason.
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


class StoreRole(Enum):
    """Which directions of traffic this engine is allowed to drive.

    Traffic is a separate concern from capacity: every role mounts the segment
    its config asks for, and the role says only what the engine then does with
    the pool. `CAPACITY` is the role that does nothing with it, which is how a
    generation server lends its memory without reading or writing.

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

        False only for `CAPACITY`. The connector short-circuits every transfer
        path on this, including KV registration.
        """
        return self.loads or self.saves


def parse_size(value: Any, *, strict_units: bool = False) -> int:
    """Accept either a byte count or a suffixed string such as `"4GiB"`.

    Args:
        value: A byte count, or a magnitude with a unit suffix.
        strict_units: Reject `GB`/`MB`/`KB`/`TB` and their one-letter forms.
            Set this wherever the value comes from, or goes into, a file the
            vLLM connector may also read: it scales those suffixes by 1024
            rather than 1000, so `"80GB"` would name two different sizes.
            Binary suffixes and plain byte counts mean the same thing to both.

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


def provisioned_config_path() -> Optional[str]:
    """The client config a server on this node rendered, if there is one.

    `provision_pool` writes one and exports `MOONCAKE_CONFIG_PATH`, which the
    ranks the LLM constructor spawns inherit. Ranks an external launcher
    started, one task per rank, were already running by then and never see it,
    so they read the config back from the run directory instead.

    Only possible when the deployment named that directory, since it otherwise
    defaults to a per-process temporary one that no other rank could read.
    """
    run_dir = os.getenv(RUN_DIR_ENV)
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
    #: Identity the keys are namespaced by. Two engines only share cache when
    #: they agree on this, so it defaults to the model directory's basename
    #: rather than its full path: the same checkpoint is routinely mounted
    #: somewhere else on another host, which is exactly the case sharing is for.
    model_key: Optional[str] = None
    #: How many page keys go into one store call. Bounds the size of a single
    #: RPC without bounding how much a request may transfer.
    transfer_batch_size: int = 64
    #: Pass pages through a pinned host buffer instead of registering the KV
    #: pools with Mooncake. Costs a copy each way, but works without GPUDirect
    #: RDMA, which registering device memory requires. The pinned allocation is
    #: sized from the layout rather than configured; see `staging.py`.
    stage_through_host: bool = False

    def __post_init__(self) -> None:
        """Reject settings that would fail later, inside a transfer."""
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
        """Read the topology from a vLLM-compatible Mooncake JSON config.

        Sizes are parsed with `strict_units`, since this file is the one both
        engines may open and the two parsers disagree about `GB`.
        """
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
            stage_through_host=bool(raw.get("stage_through_host", False)),
        )

    @staticmethod
    def from_env() -> "MooncakeStoreConnectorConfig":
        """Load the JSON config, then apply the TensorRT-LLM env overrides."""
        path = os.getenv(CONFIG_PATH_ENV) or provisioned_config_path()
        if not path:
            raise ValueError(
                f"The mooncake-store connector needs {CONFIG_PATH_ENV} set to a "
                "Mooncake JSON config (metadata_server, master_server_address, "
                "protocol, device_name, global_segment_size, local_buffer_size), "
                "or kv_connector_config.mooncake_store set so the server renders "
                f"one, into ${RUN_DIR_ENV} if this rank was started by the "
                "launcher rather than spawned by the server."
            )
        config = MooncakeStoreConnectorConfig.from_file(path)
        return config.with_env_overrides()

    def with_env_overrides(self) -> "MooncakeStoreConnectorConfig":
        """Apply `TRTLLM_MOONCAKE_STORE_*` on top of the file's settings."""
        import dataclasses

        updates: dict[str, Any] = {}
        role = os.getenv(ROLE_ENV)
        if role:
            try:
                updates["role"] = StoreRole(role.strip().lower())
            except ValueError as exc:
                known = ", ".join(member.value for member in StoreRole)
                raise ValueError(f"{ROLE_ENV}={role!r} is not one of: {known}") from exc
        staging = os.getenv(STAGE_THROUGH_HOST_ENV)
        if staging:
            normalized = staging.strip().lower()
            if normalized in _TRUE:
                updates["stage_through_host"] = True
            elif normalized in _FALSE:
                updates["stage_through_host"] = False
            else:
                known = ", ".join(sorted(_TRUE | _FALSE))
                raise ValueError(
                    f"{STAGE_THROUGH_HOST_ENV}={staging!r} is not a boolean; use one of: {known}"
                )
        return dataclasses.replace(self, **updates) if updates else self

    def resolve_model_key(self, model: Any) -> str:
        """The model identity to namespace keys by, given the configured model."""
        if self.model_key:
            return self.model_key
        return os.path.basename(str(model).rstrip("/")) or str(model)
