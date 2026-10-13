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
"""The KV cache v2 manager must report its per-pool-group slot counts at startup.

Without this, a worker whose admission is limited by one small pool group is
indistinguishable from a healthy one: the per-iteration `kv_cache_util` field
sums every pool group, so it reads low -- 13% in the case this was written for
-- while the binding pool group sits at its ceiling and refuses to activate
requests. Nothing else in the manager prints per-group sizes, so the operator
has no way to see which group is the constraint, or to compare a group's slot
count against `max_batch_size` before committing a run.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2

pytestmark = pytest.mark.cpu_only

_MODULE = "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2"


def _manager() -> KVCacheManagerV2:
    """A manager with only the surface `_format_kv_cache_pool_group_sizes` reads."""
    manager = object.__new__(KVCacheManagerV2)
    manager.impl = Mock()
    return manager


def test_reports_one_line_per_pool_group() -> None:
    """Both groups must appear, so the small one cannot hide behind the large one."""
    stats = [
        SimpleNamespace(total=207735, slot_sizes=[512]),
        SimpleNamespace(total=136, slot_sizes=[31457280, 7864320]),
    ]
    with patch(f"{_MODULE}._introspection.storage_statistics", return_value=stats):
        entries = _manager()._format_kv_cache_pool_group_sizes()

    assert len(entries) == 2
    assert "pool_group_id=0" in entries[0]
    assert "num_slots=207735" in entries[0]
    # The auxiliary group is the one worth finding: a slot count this close to
    # a plausible max_batch_size is the shape that cannot absorb per-request
    # variation.
    assert "pool_group_id=1" in entries[1]
    assert "num_slots=136" in entries[1]
    assert "slot_size=[31457280, 7864320]" in entries[1]
    assert f"bytes={136 * (31457280 + 7864320)}" in entries[1]


def test_accepts_scalar_slot_size() -> None:
    """Statistics expose `slot_size` or `slot_sizes` depending on the backend."""
    stats = [SimpleNamespace(total=64, slot_size=[4096])]
    with patch(f"{_MODULE}._introspection.storage_statistics", return_value=stats):
        entries = _manager()._format_kv_cache_pool_group_sizes()

    assert len(entries) == 1
    assert "num_slots=64" in entries[0]
    assert f"bytes={64 * 4096}" in entries[0]


def test_introspection_failure_does_not_break_startup() -> None:
    """A debugging aid must never be able to stop a worker from starting."""
    with patch(
        f"{_MODULE}._introspection.storage_statistics",
        side_effect=RuntimeError("no storage"),
    ):
        assert _manager()._format_kv_cache_pool_group_sizes() == []


def test_mapping_log_includes_the_sizes() -> None:
    """The sizes ride along with the existing role-to-pool mapping log.

    They are emitted from the same call site so that a log which shows the
    mapping always shows the sizes too -- one without the other is what made
    the original investigation slow.
    """
    manager = _manager()
    manager.kv_cache_manager_py_config = SimpleNamespace(
        layers=[SimpleNamespace(layer_id=0, buffers=[SimpleNamespace(role="KEY")])]
    )
    manager._format_kv_cache_pool_lifecycle_entry = Mock(return_value="role=KEY, pool_group_id=0")

    with (
        patch.object(
            KVCacheManagerV2,
            "_format_kv_cache_pool_group_sizes",
            return_value=["pool_group_id=0, num_slots=7"],
        ),
        patch(f"{_MODULE}.logger") as mock_logger,
    ):
        manager._log_kv_cache_pool_lifecycle_mapping()

    logged = [call.args[0] for call in mock_logger.info.call_args_list]
    assert any("role=KEY" in line for line in logged)
    assert any("num_slots=7" in line for line in logged)
