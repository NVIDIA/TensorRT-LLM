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
"""Whole-model gate for nemotron3_ultra_bf16__sm_100__tep8: GSM8K on two GB200 trays."""

import pytest

from .._common import NemotronHBf16Target, skip_not_gb200


@skip_not_gb200
@pytest.mark.skip_less_mpi_world_size(8)
class TestModelingV2Nemotron3UltraBf16Sm100Tep8(NemotronHBf16Target):
    """Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / tep8.

    The routed experts sit in eight 64-expert windows, one per rank.
    """

    MOE_EP = 8

    def test_gsm8k(self):
        self._gsm8k(max_batch_size=256)

    def test_gsm8k_megamoe_cutedsl(self):
        # The routed experts on the engine's CuTe DSL MegaMoE BF16 leaf (Relu2, top-22). Built-in
        # implementation, same layout: 94.85 with MegaMoE, 95.11 with CUTLASS.
        self._gsm8k(max_batch_size=256, moe_backend="MEGAMOE_CUTEDSL")

    def test_gsm8k_trtllm_moe(self):
        # The routed experts on the engine's TRTLLM-Gen BF16 MoE (FlashInfer, Relu2, top-22).
        self._gsm8k(max_batch_size=256, moe_backend="TRTLLM")
