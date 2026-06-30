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
"""Prometheus-independent serialization of committed EP membership.

Workers serialize the recovery coordinator's committed data-plane membership
for a dedicated serving RPC without importing the client before its
multiprocess directory is configured. This module is a passive observer: it
does not expose detected/suspected physical liveness and must not drive
recovery.
"""

from typing import Any

from .ep_group_health import (
    EP_GROUP_HEALTH_EXTRA_ATTR,
    EP_GROUP_HEALTH_LEGACY_EXTRA_ATTR,
    EPGroupHealth,
)

_EP_HEALTH_METRICS_STATUS_KEY = "status"
_EP_HEALTH_METRICS_PENDING_STATUS = "pending"


def validate_ep_health_metrics_topology(model_config: Any) -> None:
    """Reject topologies whose rank-0 endpoint cannot expose every EP group."""
    mapping = getattr(model_config, "mapping", None)
    if any(
        getattr(mapping, axis, 1) > 1 for axis in ("moe_tp_size", "pp_size", "moe_cluster_size")
    ):
        raise NotImplementedError("EP health telemetry does not support multi-group MoE topologies")


def get_ep_group_health(model_config: Any) -> EPGroupHealth | None:
    """Return an already-configured coordinator-owned membership tracker.

    ``ModelConfig.extra_attrs`` shares the committed data-plane membership
    tracker with communication. It does not carry raw detector observations or
    suspected physical liveness. This telemetry consumer never creates,
    registers, enables, or mutates it. The legacy prototype key remains
    readable during rollout; if both keys are populated they must reference the
    same object.
    """
    extra_attrs = getattr(model_config, "extra_attrs", None)
    if extra_attrs is None:
        return None

    health = extra_attrs.get(EP_GROUP_HEALTH_EXTRA_ATTR)
    legacy_health = extra_attrs.get(EP_GROUP_HEALTH_LEGACY_EXTRA_ATTR)
    if health is not None and legacy_health is not None and health is not legacy_health:
        raise ValueError("configured EP group health keys reference different trackers")
    health = health if health is not None else legacy_health
    if health is None:
        return None
    if not isinstance(health, EPGroupHealth):
        raise TypeError("configured EP group health must be an EPGroupHealth")

    mapping = getattr(model_config, "mapping", None)
    moe_world_size = getattr(mapping, "moe_ep_size", None)
    if (
        type(moe_world_size) is int
        and moe_world_size > 0
        and health.moe_world_size != moe_world_size
    ):
        raise ValueError("configured EP group health world size does not match model mapping")
    return health


def is_ep_group_health_registration_pending(model_config: Any) -> bool:
    """Return whether a producer reserved a tracker slot for late attachment."""
    extra_attrs = getattr(model_config, "extra_attrs", None)
    if extra_attrs is None:
        return False
    return any(
        key in extra_attrs and extra_attrs[key] is None
        for key in (EP_GROUP_HEALTH_EXTRA_ATTR, EP_GROUP_HEALTH_LEGACY_EXTRA_ATTR)
    )


def pending_ep_health_metrics() -> dict[str, str]:
    """Return the JSON-safe marker for an explicitly pending registration."""
    return {_EP_HEALTH_METRICS_STATUS_KEY: _EP_HEALTH_METRICS_PENDING_STATUS}


def is_pending_ep_health_metrics(ep_health_stats: dict[str, Any]) -> bool:
    """Return whether a worker reported an explicitly pending registration."""
    return ep_health_stats == pending_ep_health_metrics()


def serialize_ep_health_metrics(ep_group_health: EPGroupHealth) -> dict[str, Any]:
    """Serialize one coherent committed-membership snapshot.

    ``failedRanks`` contains ranks excluded from the committed data-plane mask;
    it is not a physical-liveness or suspicion signal. The per-rank gauge is
    derived from ``worldSize`` and ``failedRanks`` without mutating recovery
    state.
    """
    snapshot = ep_group_health.snapshot()
    return {
        "sourceEpoch": ep_group_health.source_epoch,
        "worldSize": ep_group_health.moe_world_size,
        "activeCount": snapshot.active_count,
        "failedRanks": sorted(snapshot.failed_ranks),
        "generation": snapshot.generation,
    }
