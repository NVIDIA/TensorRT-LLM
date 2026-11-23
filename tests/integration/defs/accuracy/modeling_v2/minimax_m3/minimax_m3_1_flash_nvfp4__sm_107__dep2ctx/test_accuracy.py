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
"""Whole-model gate for the MiniMax-M3.1-Flash modeling_v2 target minimax_m3_1_flash_nvfp4__sm_107__dep2ctx.

GSM8K without speculation and with the standalone DSpark drafter; the harness is in ``../_m3_1_common.py``.
"""

import pytest

from .._m3_1_common import _MiniMaxM31Target, skip_not_sm107


class TestModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx(_MiniMaxM31Target):
    """MiniMax-M3.1-Flash NVFP4 / sm_107 / dep2ctx: two Rubin GPUs, attention DP, EP2 experts."""

    DEPLOYMENT = dict(tensor_parallel_size=2, moe_expert_parallel_size=2, enable_attention_dp=True)

    @skip_not_sm107
    @pytest.mark.skip_less_device(2)
    def test_gsm8k(self):
        self._gsm8k(dspark=False)

    @skip_not_sm107
    @pytest.mark.skip_less_device(2)
    def test_dspark_gsm8k(self):
        self._gsm8k(dspark=True)
