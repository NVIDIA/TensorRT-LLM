# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Full-GSM8K gate for the MXFP4 Kimi K3 target kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 on SM100."""

import pytest

from ....accuracy_core import LlmapiAccuracyTestHarness
from .._common import _GB200_ONLY, _MODEL_NAME, _MODEL_PATH, _run_gsm8k


@_GB200_ONLY
@pytest.mark.skip_less_mpi_world_size(16)
@pytest.mark.timeout(10800)
class TestModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4(LlmapiAccuracyTestHarness):
    """Attention TP16 with routed experts split over TP4 and EP4."""

    MODEL_NAME = _MODEL_NAME
    MODEL_PATH = _MODEL_PATH

    def test_gsm8k(self) -> None:
        _run_gsm8k(self.MODEL_PATH, moe_tp_size=4, moe_ep_size=4)
