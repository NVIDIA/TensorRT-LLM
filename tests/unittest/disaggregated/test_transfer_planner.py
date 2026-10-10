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
"""Contract tests for the model-owned transfer planner seam."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.planner import (
    PeerIncompatibleError,
    PeerTargets,
    TransferPlan,
    TransferPlanner,
)


def test_transfer_plan_empty_is_three_int64_arrays():
    plan = TransferPlan.empty()
    for arr in (plan.src_ptrs, plan.dst_ptrs, plan.sizes):
        assert arr.dtype == np.int64 and arr.size == 0


def test_transfer_planner_is_abstract():
    with pytest.raises(TypeError):
        TransferPlanner()  # type: ignore[abstract]


def test_peer_incompatible_error_is_value_error():
    assert issubclass(PeerIncompatibleError, ValueError)
    assert PeerTargets(ranks=[0], uniform_writers=True).ranks == [0]
