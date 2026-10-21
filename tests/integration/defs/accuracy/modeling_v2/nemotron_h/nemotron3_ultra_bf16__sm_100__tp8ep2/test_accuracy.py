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
"""Whole-model gate for nemotron3_ultra_bf16__sm_100__tp8ep2: GSM8K on two GB200 trays."""

import pytest

from .._common import NemotronHBf16Target, skip_not_gb200


@skip_not_gb200
@pytest.mark.skip_less_mpi_world_size(8)
class TestModelingV2Nemotron3UltraBf16Sm100Tp8ep2(NemotronHBf16Target):
    """Nemotron-3 Ultra 550B-A55B BF16 / sm_100 / tp8ep2.

    tep8's layout with the routed experts in two expert-parallel windows of four
    tensor-parallel ranks (moe_expert_parallel_size 2 leaves moe_tp 4).
    """

    MOE_EP = 2

    def test_gsm8k(self):
        self._gsm8k(max_batch_size=256)

    def test_gsm8k_fp8_kv(self):
        # The fp8-e4m3 KV pool at unit kv scales (94.731 before the rebase; the built-in
        # implementation over the same pool 94.655, the reference).
        self._gsm8k(max_batch_size=256, kv_cache_dtype="fp8")
