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
"""The settings a mooncake-store deployment runs with, not the ones it asked for.

Two diverge from `kv_connector_config`. An inherited `MOONCAKE_CONFIG_PATH`
supersedes the `mooncake_store` block, since it names a pool the deployment
provisioned itself. The pool is also the deployment's offload tier, which takes
the engine's own host and disk tiers out along with its partial reuse.

`apply_effective_settings` settles both on the args, so that a reader which is
not a worker sees what the workers will use. The usage report the LLM
constructor sends reads `role` and `segment_size` out of the block, and a
server whose ranks lend 32 GiB as `capacity` would otherwise be reported as
lending 16 as `both`.
"""

import os
from typing import TYPE_CHECKING, Optional

from tensorrt_llm.logger import logger

from ..registry import uses_connector
from .config import (
    CONFIG_PATH_ENV,
    MooncakeStoreConnectorConfig,
    StoreRole,
    parse_size,
    pool_config,
    provisioned_config_path,
)

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MooncakeStoreConfig, TorchLlmArgs

__all__ = [
    "apply_effective_settings",
    "apply_transferring_role_overrides",
    "disable_native_kv_offload",
    "disable_partial_reuse",
]


def apply_effective_settings(llm_args: "TorchLlmArgs") -> None:
    """Settle the effective settings on `llm_args`, before anything reads them.

    A no-op for every other connector. Each override says nothing once it has
    been applied, so a caller that cannot tell whether an earlier one ran is
    free to call this again.
    """
    if not uses_connector(llm_args.kv_connector_config, "mooncake-store"):
        return

    pool = pool_config(llm_args)
    path = os.getenv(CONFIG_PATH_ENV) or provisioned_config_path(
        pool.run_dir if pool is not None else None
    )
    effective = _read_client_config(path) if path else None
    if pool is not None and effective is not None:
        _restate_pool_block(pool, effective, path)

    kv_cache_config = llm_args.kv_cache_config
    if kv_cache_config is None:
        return
    # The role comes from the client config where there is one to read, and
    # from the block otherwise; a pool `trtllm-serve` has yet to provision
    # offers neither, and is settled in `create_py_executor` against the
    # worker it built.
    role = effective.role if effective is not None else None
    if role is None and pool is not None:
        role = StoreRole(pool.role)
    if role is not None and role.transfers:
        apply_transferring_role_overrides(kv_cache_config)


def apply_transferring_role_overrides(kv_cache_config: "KvCacheConfig") -> None:
    """Every setting the pool supersedes for a role that moves KV.

    Call this behind whichever answer about the role the caller has: the
    resolved config before any worker exists, and `capacity_only` off the
    worker once one does.
    """
    disable_native_kv_offload(kv_cache_config)
    disable_partial_reuse(kv_cache_config)


def disable_native_kv_offload(kv_cache_config: "KvCacheConfig") -> None:
    """Turn off this engine's own host and disk cache tiers.

    For a role that transfers KV. Such a rank registers page addresses with the
    pool, so migrating a page off GPU reassigns a slot the pool still names, and
    pool pages already hold host memory the ranks lent. A capacity-only rank has
    neither problem and keeps the tiers it was configured with, including the
    host tier `KVCacheManagerV2` auto-provisions from an unset `host_cache_size`.

    Both fields are pinned to 0 rather than left unset, since a host_cache_size
    of None asks KVCacheManagerV2 to size a host tier automatically. Call this
    before the KV cache manager is built, which reads both to decide its tiers.
    """
    if kv_cache_config.host_cache_size == 0 and kv_cache_config.disk_cache_size == 0:
        return
    explicit = {
        name: value
        for name, value in (
            ("host_cache_size", kv_cache_config.host_cache_size),
            ("disk_cache_size", kv_cache_config.disk_cache_size),
        )
        if value
    }
    if explicit:
        requested = ", ".join(f"{name}={value}" for name, value in explicit.items())
        logger.warning(
            f"Ignoring kv_cache_config {requested}: the mooncake-store "
            "connector is this deployment's offload tier. Put the memory into "
            "mooncake_store.segment_size instead, where every server on the "
            "node can reuse what any of them stored."
        )
    else:
        logger.info(
            "Native KV cache offloading is off: the mooncake-store connector "
            "provides the offload tier."
        )
    kv_cache_config.host_cache_size = 0
    kv_cache_config.disk_cache_size = 0


def disable_partial_reuse(kv_cache_config: "KvCacheConfig") -> None:
    """Take partial reuse off a role that looks pages up in the pool."""
    if not kv_cache_config.enable_partial_reuse:
        return
    logger.warning(
        "Disabling partial reuse: the mooncake-store connector addresses "
        "whole blocks, so a partial match leaves the matched length off a "
        "block boundary and the connector declines the lookup, trading part "
        "of one block for every stored block of the remaining prefix."
    )
    kv_cache_config.enable_partial_reuse = False


def _read_client_config(path: str) -> Optional[MooncakeStoreConnectorConfig]:
    """The config at `path` as the workers will read it, or `None` if it will not.

    An unreadable config is left to the workers, which fail on it with the path
    and the parse error in hand.
    """
    try:
        return MooncakeStoreConnectorConfig.from_file(path)
    except (OSError, ValueError) as exc:
        logger.warning(
            f"mooncake-store: {path} could not be read ({exc}), so "
            "kv_connector_config.mooncake_store still describes what this "
            "server asked for rather than what its ranks will use."
        )
        return None


def _restate_pool_block(
    pool: "MooncakeStoreConfig", effective: MooncakeStoreConnectorConfig, path: str
) -> None:
    """Restate `pool` as the client config at `path` leaves it.

    `pool`, `local_hostname`, `run_dir` and `master_timeout` describe how to
    reach the pool and where this run's files go, not what the server joins it
    as, so they stay as they are.
    """
    # Each asked-for value is compared in the config's own terms, a role as its
    # string and a size as a byte count, so resolving a segment_size of "16GiB"
    # does not read as a change to what it already meant.
    superseded = (
        ("role", effective.role.value, StoreRole(pool.role).value),
        (
            "segment_size",
            effective.global_segment_size,
            parse_size(pool.segment_size, strict_units=True),
        ),
        ("namespace", effective.namespace, pool.namespace),
        ("model_key", effective.model_key, pool.model_key),
        ("transfer_batch_size", effective.transfer_batch_size, pool.transfer_batch_size),
        ("stage_through_host", effective.stage_through_host, pool.stage_through_host),
    )
    # Assignment adds to `model_fields_set`, so what the deployment stated for
    # itself has to be read before any of it is replaced. Only a value it chose
    # is worth reporting; the rest are defaults being resolved.
    stated = set(pool.model_fields_set)
    overridden = {}
    for setting, value, asked in superseded:
        if value is None or value == asked:
            continue
        setattr(pool, setting, value)
        if setting in stated:
            overridden[setting] = value
    if overridden:
        logger.warning(
            f"mooncake-store: {path} states "
            f"{', '.join(f'{k}={v!r}' for k, v in overridden.items())}, which is "
            "what this server joins the pool as."
        )
