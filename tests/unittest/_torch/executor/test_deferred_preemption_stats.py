# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Deferred connector preemptions contribute to iteration statistics once."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("dummy", [False, True])
def test_deferred_preemption_counts_once_in_next_batch(dummy):
    executor = object.__new__(PyExecutor)
    executor._is_kv_manager_v2 = True
    executor.enable_attention_dp = True
    executor.kv_cache_manager = Mock()
    executor.kv_cache_manager.try_complete_preemption.side_effect = [True, False]
    executor._free_request_resources = Mock()
    executor._pause_recompute_request = Mock()
    request = SimpleNamespace(is_dummy=dummy)

    assert executor._resume_preempted_request(request)
    assert not executor._resume_preempted_request(request)
    executor._free_request_resources.assert_called_once_with(request)
    executor._pause_recompute_request.assert_called_once_with(request)

    batch = SimpleNamespace(
        context_requests=[],
        generation_requests=[],
        paused_requests=[SimpleNamespace(is_dummy=False)],
        recompute_paused_requests=[SimpleNamespace(is_dummy=False)],
    )
    assert executor._collect_scheduled_batch_stats(batch).num_paused_requests == 2 + (not dummy)
    assert executor._collect_scheduled_batch_stats(batch).num_paused_requests == 2
    assert executor._take_deferred_recompute_pauses() == 0


def test_deferred_preemptions_can_be_consumed_without_a_forward_batch():
    executor = object.__new__(PyExecutor)
    executor._completed_deferred_recompute_pauses = 2

    assert executor._take_deferred_recompute_pauses() == 2
    assert executor._take_deferred_recompute_pauses() == 0
