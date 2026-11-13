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
"""Whole-model gate for the MiniMax-M3.1-Flash modeling_v2 target minimax_m3_1_flash_nvfp4__sm_103__tp8.

GSM8K without speculation and with the standalone DSpark drafter; the harness is in ``../_m3_1_common.py``.
"""

import pytest

from .._m3_1_common import _MiniMaxM31Target, skip_not_sm103


class TestModelingV2MiniMaxM31FlashNvfp4Sm103Tp8(_MiniMaxM31Target):
    """MiniMax-M3.1-Flash NVFP4 / sm_103 / tp8: eight GB300s, tensor-parallel attention and experts."""

    DEPLOYMENT = dict(tensor_parallel_size=8, moe_expert_parallel_size=1)

    @skip_not_sm103
    @pytest.mark.skip_less_device(8)
    def test_gsm8k(self):
        self._gsm8k(dspark=False)

    @skip_not_sm103
    @pytest.mark.skip_less_device(8)
    def test_dspark_gsm8k(self):
        self._gsm8k(dspark=True)
