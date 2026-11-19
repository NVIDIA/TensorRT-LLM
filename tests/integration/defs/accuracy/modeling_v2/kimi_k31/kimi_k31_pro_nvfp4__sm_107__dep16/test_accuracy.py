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
"""Whole-model gate for the kimi_k31 modeling_v2 target kimi_k31_pro_nvfp4__sm_107__dep16.

The recipe of the sm_107 targets is in ``../_common.py``.
"""

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV

from .....conftest import skip_no_rubin
from ....accuracy_core import GSM8K, LlmapiAccuracyTestHarness
from ..._env import scoped_env_overrides
from .._common import NVFP4_MODEL_NAME, NVFP4_PATH, nvfp4_aggregated


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Dep16(LlmapiAccuracyTestHarness):
    """``kimi_k31_pro_nvfp4__sm_107__dep16``: attention DP and MoE EP over 16 ranks."""

    MODEL_NAME = NVFP4_MODEL_NAME
    MODEL_PATH = NVFP4_PATH

    @pytest.mark.timeout(5400)
    @pytest.mark.skip_less_mpi_world_size(16)
    @pytest.mark.skip_less_device_memory(250000)
    def test_gsm8k_dep16(self):
        """GSM8K on DEP16 with DFlash2 (draft length 7), MegaMoE at batch 12."""
        kwargs = nvfp4_aggregated(16, True, "MEGAMOE_CUTEDSL", max_batch_size=12)
        with (
            scoped_env_overrides({MODELING_V2_VALIDATE_ENV: "1"}) as rank_env,
            LLM(self.MODEL_PATH, env_overrides=rank_env, **kwargs) as llm,
        ):
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)
