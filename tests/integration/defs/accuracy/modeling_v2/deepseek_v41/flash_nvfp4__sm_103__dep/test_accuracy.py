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
"""Whole-model gate for the deepseek_v41 modeling_v2 target flash_nvfp4__sm_103__dep.

The harness is in ``../_common.py``.
"""

import pytest

from .._common import _AttentionDpGenerationGate, skip_not_sm103

# (world size, DSpark draft length, batch) of each deployment the target serves.
_WORLDS = [
    pytest.param(
        (8, 3, 16),
        id="dep8",
        marks=[pytest.mark.skip_less_mpi_world_size(8), pytest.mark.timeout(7200)],
    ),
    pytest.param(
        (32, 5, 25),
        id="dep32",
        marks=[pytest.mark.skip_less_mpi_world_size(32), pytest.mark.timeout(10800)],
    ),
]


class TestModelingV2DeepseekV41FlashNvfp4Sm103Dep(_AttentionDpGenerationGate):
    """deepseek-v4.1-flash-nvfp4 / sm_103 / dep: TP = EP = world (8 or 32) + attention DP, DSpark, lm_head TP in ADP.

    dep8 runs on two GB300 trays (8 ranks), dep32 on eight (32 ranks; not a CI shape yet, selected by name in the
    multi-node QA list). With and without speculation, like tep8: DSpark is lossless, so both rows share one floor;
    the speculative row proves the target's verification path (drafts accepted or rejected, greedy output unchanged).
    Whether the draft reads the right hidden states shows in the acceptance length, not the accuracy; it is gated
    against the stock implementation in the disaggregated measurement, not here. The deployed batches are 320 (dep8,
    DSpark 3/3) and 25 (dep32, DSpark 5/5); the dep8 gate runs the small batch the other gates use, the dep32 gate
    its deployed 25.
    """

    @skip_not_sm103
    @pytest.mark.parametrize("dspark", [False, True], ids=["no_spec", "dspark"])
    @pytest.mark.parametrize("deployment", _WORLDS)
    def test_gsm8k(self, mocker, deployment, dspark):
        self.WORLD, self.DRAFT_LEN, self.BATCH = deployment
        self._run(mocker, dspark)
