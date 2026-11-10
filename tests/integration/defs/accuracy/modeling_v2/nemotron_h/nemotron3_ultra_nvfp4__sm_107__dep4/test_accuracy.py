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
"""Whole-model gate for nemotron3_ultra_nvfp4__sm_107__dep4: GSM8K on 4 Rubin GPUs.

Validated only before the rebase onto the FallbackFmha attention path: GSM8K
94.769 (reference 94.655).
"""

import pytest

from .._common import NemotronHNvfp4Target, skip_not_sm107


@skip_not_sm107
@pytest.mark.skip_less_device(4)
class TestModelingV2Nemotron3UltraNvfp4Sm107Dep4(NemotronHNvfp4Target):
    """Nemotron-3 Ultra 550B-A55B NVFP4 / sm_107 / dep4.

    Attention data parallel, everything but the routed experts whole on every rank;
    the routed experts expert-parallel over the same 4 ranks.
    """

    ATTENTION_DP = True

    def test_gsm8k(self):
        self._gsm8k(max_batch_size=32)
