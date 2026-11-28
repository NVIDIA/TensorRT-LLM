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
"""Full GSM8K for the Kimi K3 Rubin Helix16 ULL generation target."""

import pytest

from .._sm107_gsm8k import Sm107Gsm8kHarness, requires_sm107, run_disaggregated_gsm8k

_CASE = (
    "disagg-e2e-vr200_kimi-k3-nvfp4-dspark_gsm8k_ctx12_dwdp12_gen1_helix16_moetp16ep1_bs4_dspark7"
)


@requires_sm107
@pytest.mark.timeout(7200)
class TestModelingV2KimiK3Nvfp4Sm107Helix16Moetp16ep1(Sm107Gsm8kHarness):
    @pytest.mark.parametrize("test_case", [_CASE])
    def test_gsm8k(self, output_dir: str, test_case: str) -> None:
        run_disaggregated_gsm8k(output_dir, test_case, max_draft_len=7)
