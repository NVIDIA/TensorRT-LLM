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
"""Whole-model gate for the MiniMax-M3 modeling_v2 target minimax_m3_nvfp4__sm_107__tp8.

GSM8K without speculation and with one-model Eagle3 plus its acceptance probe; the harness is in ``../_common.py``.
"""

import pytest

from .._common import _MiniMaxM3Target, skip_not_sm107


class TestModelingV2MiniMaxM3Nvfp4Sm107Tp8(_MiniMaxM3Target):
    """MiniMax-M3 NVFP4 / sm_107 / tp8: eight Rubin GPUs over two trays (an 8-rank MPI world across the two nodes)."""

    TP = 8

    @skip_not_sm107
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_gsm8k(self):
        self._gsm8k(eagle3=False)

    @skip_not_sm107
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_eagle3_gsm8k_and_acceptance(self):
        self._gsm8k(eagle3=True)
