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
"""Unit tests for passive committed WideEP membership serialization."""

import copy
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import (
    EP_GROUP_HEALTH_EXTRA_ATTR,
    EP_GROUP_HEALTH_LEGACY_EXTRA_ATTR,
    EPGroupHealth,
)
from tensorrt_llm._torch.modules.fused_moe.ep_metrics import (
    get_ep_group_health,
    is_ep_group_health_registration_pending,
    is_pending_ep_health_metrics,
    pending_ep_health_metrics,
    serialize_ep_health_metrics,
    validate_ep_health_metrics_topology,
)


def test_serialize_initial_health() -> None:
    health = EPGroupHealth(4)

    assert serialize_ep_health_metrics(health) == {
        "sourceEpoch": health.source_epoch,
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }


def test_deepcopy_preserves_shared_producer_identity() -> None:
    health = EPGroupHealth(4)

    assert copy.deepcopy(health) is health


def test_serialization_reads_one_coherent_snapshot() -> None:
    health: Any = SimpleNamespace(
        moe_world_size=4,
        source_epoch="source-a",
        snapshot=MagicMock(
            return_value=SimpleNamespace(
                active_count=3,
                failed_ranks=frozenset({2}),
                generation=1,
            )
        ),
    )

    assert serialize_ep_health_metrics(health) == {
        "sourceEpoch": "source-a",
        "worldSize": 4,
        "activeCount": 3,
        "failedRanks": [2],
        "generation": 1,
    }
    health.snapshot.assert_called_once_with()


def test_serialize_committed_exclusion_and_reinclusion() -> None:
    health = EPGroupHealth(72)
    health.mark_failed(70)
    health.mark_failed(3)
    health.mark_failed(3)

    assert serialize_ep_health_metrics(health) == {
        "sourceEpoch": health.source_epoch,
        "worldSize": 72,
        "activeCount": 70,
        "failedRanks": [3, 70],
        "generation": 2,
    }

    health.mark_active(3)
    assert serialize_ep_health_metrics(health) == {
        "sourceEpoch": health.source_epoch,
        "worldSize": 72,
        "activeCount": 71,
        "failedRanks": [70],
        "generation": 3,
    }


@pytest.mark.parametrize(
    "key",
    [EP_GROUP_HEALTH_EXTRA_ATTR, EP_GROUP_HEALTH_LEGACY_EXTRA_ATTR],
)
def test_get_ep_group_health_passively_reads_registered_tracker(key: str) -> None:
    health = EPGroupHealth(4)
    extra_attrs = {key: health}
    model_config = SimpleNamespace(
        extra_attrs=extra_attrs,
        mapping=SimpleNamespace(moe_ep_size=4),
    )

    assert get_ep_group_health(model_config) is health
    assert model_config.extra_attrs == extra_attrs


def test_get_ep_group_health_does_not_create_or_enable_tracker(monkeypatch) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_WIDE_EP_FT", "1")
    model_config = SimpleNamespace(
        extra_attrs={},
        mapping=SimpleNamespace(moe_ep_size=8),
    )

    assert get_ep_group_health(model_config) is None
    assert not is_ep_group_health_registration_pending(model_config)
    assert model_config.extra_attrs == {}


def test_explicit_empty_registration_is_pending_without_mutation() -> None:
    model_config = SimpleNamespace(extra_attrs={EP_GROUP_HEALTH_EXTRA_ATTR: None})

    assert get_ep_group_health(model_config) is None
    assert is_ep_group_health_registration_pending(model_config)
    pending = pending_ep_health_metrics()
    assert pending == {"status": "pending"}
    assert is_pending_ep_health_metrics(pending)
    assert not is_pending_ep_health_metrics({"status": "unsupported"})
    assert model_config.extra_attrs == {EP_GROUP_HEALTH_EXTRA_ATTR: None}


def test_get_ep_group_health_rejects_split_trackers() -> None:
    model_config = SimpleNamespace(
        extra_attrs={
            EP_GROUP_HEALTH_EXTRA_ATTR: EPGroupHealth(4),
            EP_GROUP_HEALTH_LEGACY_EXTRA_ATTR: EPGroupHealth(4),
        },
        mapping=SimpleNamespace(moe_ep_size=4),
    )

    with pytest.raises(ValueError, match="different trackers"):
        get_ep_group_health(model_config)


def test_get_ep_group_health_rejects_invalid_shared_object() -> None:
    wrong_type = SimpleNamespace(
        extra_attrs={EP_GROUP_HEALTH_EXTRA_ATTR: object()},
        mapping=SimpleNamespace(moe_ep_size=8),
    )
    with pytest.raises(TypeError, match="must be an EPGroupHealth"):
        get_ep_group_health(wrong_type)

    wrong_size = SimpleNamespace(
        extra_attrs={EP_GROUP_HEALTH_EXTRA_ATTR: EPGroupHealth(4)},
        mapping=SimpleNamespace(moe_ep_size=8),
    )
    with pytest.raises(ValueError, match="world size does not match"):
        get_ep_group_health(wrong_size)


def test_topology_validation_is_deferred_to_metric_read() -> None:
    health = EPGroupHealth(4)
    model_config = SimpleNamespace(
        extra_attrs={EP_GROUP_HEALTH_EXTRA_ATTR: health},
        mapping=SimpleNamespace(
            moe_ep_size=4,
            moe_tp_size=1,
            pp_size=2,
            moe_cluster_size=1,
        ),
    )

    assert get_ep_group_health(model_config) is health
    with pytest.raises(NotImplementedError, match="multi-group MoE topologies"):
        validate_ep_health_metrics_topology(model_config)


def test_concurrent_serialization_is_coherent() -> None:
    health = EPGroupHealth(8)
    start = threading.Barrier(2)
    worker_errors: list[BaseException] = []

    def toggle_rank() -> None:
        try:
            start.wait(timeout=10.0)
            for _ in range(1_000):
                health.mark_failed(3)
                health.mark_active(3)
        except BaseException as error:
            worker_errors.append(error)

    thread = threading.Thread(target=toggle_rank)
    thread.start()
    start.wait(timeout=10.0)

    for _ in range(1_000):
        stats = serialize_ep_health_metrics(health)
        assert stats["activeCount"] == stats["worldSize"] - len(stats["failedRanks"])
        assert stats["failedRanks"] in ([], [3])
        assert bool(stats["generation"] % 2) is bool(stats["failedRanks"])

    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert not worker_errors
    assert health.generation == 2_000
    assert health.get_failed_ranks() == frozenset()
