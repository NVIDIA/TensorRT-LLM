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

Topology settings are read from the JSON file named by ``MOONCAKE_CONFIG_PATH``,
the same file and environment variable the vLLM Mooncake store connector uses,
so one deployment can point both engines at the same pool.

``KvCacheConnectorConfig`` carries no free-form dictionary, so the two settings
that are TensorRT-LLM's rather than Mooncake's -- the read/write role and the
key prefix -- are also taken from the environment.
"""

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

__all__ = [
    "CONFIG_PATH_ENV",
    "MooncakeStoreConnectorConfig",
    "ROLE_ENV",
    "STAGE_THROUGH_HOST_ENV",
    "StoreRole",
]

CONFIG_PATH_ENV = "MOONCAKE_CONFIG_PATH"
ROLE_ENV = "TRTLLM_MOONCAKE_STORE_ROLE"
CACHE_PREFIX_ENV = "TRTLLM_MOONCAKE_STORE_PREFIX"
MODEL_KEY_ENV = "TRTLLM_MOONCAKE_STORE_MODEL_KEY"
STAGE_THROUGH_HOST_ENV = "TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST"

DEFAULT_GLOBAL_SEGMENT_SIZE = 3355443200
DEFAULT_LOCAL_BUFFER_SIZE = 1073741824
DEFAULT_CACHE_PREFIX = "trtllm"
DEFAULT_STAGING_BUFFER_SIZE = 536870912

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "t": 1000**4,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}
_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")


class StoreRole(Enum):
    """Which directions of traffic this engine is allowed to drive.

    A disaggregated deployment typically runs context servers as ``both`` and
    leaves generation servers unconfigured: generated tokens are rarely a reused
    prefix, so writing them costs bandwidth for no hit rate.
    """

    PRODUCER = "producer"
    CONSUMER = "consumer"
    BOTH = "both"

    @property
    def loads(self) -> bool:
        """Whether this role reads previously stored KV back onto the GPU."""
        return self is not StoreRole.PRODUCER

    @property
    def saves(self) -> bool:
        """Whether this role writes newly computed KV into the store."""
        return self is not StoreRole.CONSUMER


def parse_size(value: Any) -> int:
    """Accept either a byte count or a suffixed string such as ``"4GiB"``."""
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
    scale = _SIZE_UNITS.get(unit.lower())
    if scale is None:
        raise ValueError(f"unknown size unit {unit!r} in {value!r}")
    return int(float(magnitude) * scale)


@dataclass(frozen=True)
class MooncakeStoreConnectorConfig:
    """Everything needed to open a store handle and name keys in it."""

    metadata_server: str
    master_server_address: str
    protocol: str = "rdma"
    device_name: str = ""
    global_segment_size: int = DEFAULT_GLOBAL_SEGMENT_SIZE
    local_buffer_size: int = DEFAULT_LOCAL_BUFFER_SIZE
    local_hostname: Optional[str] = None
    tenant_id: Optional[str] = None
    role: StoreRole = StoreRole.BOTH
    cache_prefix: str = DEFAULT_CACHE_PREFIX
    #: Identity the keys are namespaced by. Two engines only share cache when
    #: they agree on this, so it defaults to the model directory's basename
    #: rather than its full path: the same checkpoint is routinely mounted
    #: somewhere else on another host, which is exactly the case sharing is for.
    model_key: Optional[str] = None
    #: How many page keys go into one store call. Bounds the size of a single
    #: RPC without bounding how much a request may transfer.
    transfer_batch_size: int = 64
    #: Pass pages through a pinned host buffer instead of registering the KV
    #: pools with Mooncake. Costs a copy in each direction and buys independence
    #: from GPUDirect RDMA, without which registering device memory fails
    #: outright. Leave off wherever the pool can reach GPU memory.
    stage_through_host: bool = False
    #: Ceiling on the pinned allocation per direction when staging. The pool is
    #: sized from the layout's largest page, so this caps how many pages may be
    #: in flight rather than how large one may be.
    staging_buffer_bytes: int = DEFAULT_STAGING_BUFFER_SIZE

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
        if self.stage_through_host and self.staging_buffer_bytes <= 0:
            raise ValueError("staging_buffer_bytes must be > 0 when staging is on")

    @staticmethod
    def from_file(path: str) -> "MooncakeStoreConnectorConfig":
        """Read the topology from a vLLM-compatible Mooncake JSON config."""
        with open(path) as handle:
            raw = json.load(handle)
        return MooncakeStoreConnectorConfig(
            metadata_server=raw.get("metadata_server", ""),
            master_server_address=raw.get("master_server_address", ""),
            protocol=raw.get("protocol", "rdma"),
            device_name=raw.get("device_name", ""),
            global_segment_size=parse_size(
                raw.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE)
            ),
            local_buffer_size=parse_size(raw.get("local_buffer_size", DEFAULT_LOCAL_BUFFER_SIZE)),
            local_hostname=raw.get("local_hostname") or None,
            tenant_id=raw.get("tenant_id") or None,
            role=StoreRole(str(raw.get("role", StoreRole.BOTH.value)).strip().lower()),
            cache_prefix=str(raw.get("cache_prefix", DEFAULT_CACHE_PREFIX)),
            model_key=raw.get("model_key") or None,
            transfer_batch_size=int(raw.get("transfer_batch_size", 64)),
            stage_through_host=bool(raw.get("stage_through_host", False)),
            staging_buffer_bytes=parse_size(
                raw.get("staging_buffer_bytes", DEFAULT_STAGING_BUFFER_SIZE)
            ),
        )

    @staticmethod
    def from_env() -> "MooncakeStoreConnectorConfig":
        """Load the JSON config, then apply the TensorRT-LLM env overrides."""
        path = os.getenv(CONFIG_PATH_ENV)
        if not path:
            raise ValueError(
                f"The mooncake-store connector needs {CONFIG_PATH_ENV} set to a "
                "Mooncake JSON config (metadata_server, master_server_address, "
                "protocol, device_name, global_segment_size, local_buffer_size)."
            )
        config = MooncakeStoreConnectorConfig.from_file(path)
        return config.with_env_overrides()

    def with_env_overrides(self) -> "MooncakeStoreConnectorConfig":
        """Apply ``TRTLLM_MOONCAKE_STORE_*`` on top of the file's settings."""
        import dataclasses

        updates: dict[str, Any] = {}
        role = os.getenv(ROLE_ENV)
        if role:
            try:
                updates["role"] = StoreRole(role.strip().lower())
            except ValueError as exc:
                known = ", ".join(member.value for member in StoreRole)
                raise ValueError(f"{ROLE_ENV}={role!r} is not one of: {known}") from exc
        prefix = os.getenv(CACHE_PREFIX_ENV)
        if prefix:
            updates["cache_prefix"] = prefix
        model_key = os.getenv(MODEL_KEY_ENV)
        if model_key:
            updates["model_key"] = model_key
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
