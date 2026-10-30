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
"""First-100 GSM8K regression for the aggregated NVFP4 serving recipe.

QA/LR selects this test on one four-GB300 node with a 45-minute timeout.
The pinned public protocol requires at least 95 correct answers and no execution,
parsing or truncation failures. Full GSM8K is not selected by any CI list.

Stage the target, DFlash2 drafter and FP8 KV scale table below LLM_MODELS_ROOT;
set TRTLLM_TEST_TARGET_RELATIVE_PATH, TRTLLM_TEST_DRAFTER_RELATIVE_PATH and
TRTLLM_TEST_KV_SCALES_RELATIVE_PATH to their relative paths. The pinned GSM8K
dataset belongs at datasets/openai/gsm8k. Missing inputs fail a selected run.
Private checkpoint locations belong in the CI environment, not this repository.
Results and resumable records go below <--output-dir>/nvfp4_serving_gsm8k_first100.
"""

import os

import pytest

from ..conftest import get_device_count, get_llm_root, llm_models_root
from .gsm8k_first100 import runner


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_device_not_contain(["GB300"])
def test_gsm8k_first100(request, timeout_from_marker, timeout_from_command_line):
    if int(get_device_count()) != 4:
        pytest.skip("first-100 qualification requires exactly four visible GB300 GPUs")
    environ = dict(os.environ)
    environ["TRTLLM_RUN_NVFP4_SERVING_GSM8K100"] = "1"
    environ["TRTLLM_NVFP4_SERVING_GSM8K100_MODE"] = "regression"
    runner.require_hardware_when_enabled(environ)
    runner.run_first100(
        environ=environ,
        models_root=llm_models_root,
        repo_root=get_llm_root,
        timeout_s=(
            timeout_from_marker if timeout_from_marker is not None else timeout_from_command_line
        ),
        output_dir=request.config.getoption("--output-dir"),
    )
