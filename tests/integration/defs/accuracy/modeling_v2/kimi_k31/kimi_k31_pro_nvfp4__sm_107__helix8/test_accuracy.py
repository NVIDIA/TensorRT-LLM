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
"""GSM8K on a 16-rank DEP context worker and an 8-rank Helix generation worker on Rubin."""

import os

import pytest

from tensorrt_llm.llmapi.llm_args import LlmArgs
from tensorrt_llm.quantization import QuantAlgo

from .....conftest import skip_no_rubin
from ....accuracy_core import LlmapiAccuracyTestHarness
from ..._disaggregated_gsm8k import run_disaggregated_gsm8k
from .._common import NVFP4_MODEL_NAME, NVFP4_PATH, nvfp4_dflash2


@skip_no_rubin
class TestModelingV2KimiK31ProNvfp4Sm107Helix8(LlmapiAccuracyTestHarness):
    MODEL_NAME = NVFP4_MODEL_NAME
    MODEL_PATH = NVFP4_PATH

    @pytest.mark.skipif(
        not os.environ.get("DISAGG_SERVING_TYPE"),
        reason="Run Helix8 GSM8K through the multi-node disaggregated launcher",
    )
    @pytest.mark.timeout(7200)
    @pytest.mark.parametrize(
        "test_case", ["disagg-e2e-vr200_kimi-k31-nvfp4-dflash2_gsm8k_ctx1_dep16_gen1_helix8_bs16"]
    )
    def test_gsm8k_helix8(self, output_dir: str, test_case: str) -> None:
        args = LlmArgs(model=self.MODEL_PATH)
        args.quant_config.quant_algo = QuantAlgo.MIXED_PRECISION
        args.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
        args.speculative_config = nvfp4_dflash2(5)
        run_disaggregated_gsm8k(output_dir, test_case, self.MODEL_NAME, args)
