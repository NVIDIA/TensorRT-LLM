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
"""Whole-model gate for the kimi_k31 modeling_v2 target kimi_k31_pro_mxfp4__sm_103__dep.

The tests run the implementations in ``../_common.py``, which also holds the layouts.
"""

import pytest

from .._common import (
    _K31Target,
    pytest_generate_tests,  # noqa: F401 (the hook parametrizes the deployments)
    skip_not_sm103,
)


class TestModelingV2KimiK31ProMxfp4Sm103Dep(_K31Target):
    """kimi-k3.1-pro-mxfp4 / sm_103 / dep, the language model."""

    pytestmark = skip_not_sm103
    DEPLOYMENTS = {
        "test_gsm8k": ("dep16",),
        "test_gsm8k_dflash2": ("dep16",),
        "test_gpqa_diamond": ("dep16",),
    }

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k(self, deployment):
        self._test_gsm8k(deployment)

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_dflash2(self, deployment):
        """GSM8K with DFlash2 w5."""
        self._test_gsm8k_dflash2(deployment, max_draft_len=4)

    @pytest.mark.timeout(10800)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gpqa_diamond(self, deployment):
        self._test_gpqa_diamond(deployment)
