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

QA/LR selects the first 100 questions on four GB300s with a 45-minute timeout
and a minimum score of 95/100. Stage the optimized inputs under
LLM_MODELS_ROOT/beam/nvfp4-dflash2-fp8kv or configure the relative input paths
documented in the evaluator README. An explicit opt-out wins. Once enabled,
missing or invalid inputs fail the regression.
Results and resumable records go below <--output-dir>/nvfp4_serving_gsm8k_first100.
"""

import os
from pathlib import Path

import pytest

from ..conftest import get_device_count, get_llm_root, llm_models_root
from .gsm8k_first100 import deployment, runner


@pytest.mark.skip_less_device(4)
@pytest.mark.skip_device_not_contain(["GB300"])
def test_gsm8k_first100(request, timeout_from_marker, timeout_from_command_line):
    if int(get_device_count()) != 4:
        pytest.skip("first-100 qualification requires exactly four visible GB300 GPUs")
    environ = dict(os.environ)
    if deployment.GUARD_ENV not in environ:
        # The target can be staged for the independent BF16 baseline alone.
        if not (environ.get(deployment.DRAFTER_ENV) or environ.get(deployment.KV_SCALES_ENV)):
            try:
                root = Path(llm_models_root())
            except AssertionError as exc:
                pytest.skip(f"no QA model mount for optimized Beam: {exc}")
            if not (root / deployment.STAGING_DIR).is_dir():
                pytest.skip("optimized Beam inputs are not staged or configured on this worker")
        environ[deployment.GUARD_ENV] = "1"
    environ[deployment.MODE_ENV] = "regression"
    runner.run_first100(
        environ=environ,
        models_root=llm_models_root,
        repo_root=get_llm_root,
        timeout_s=(
            timeout_from_marker if timeout_from_marker is not None else timeout_from_command_line
        ),
        output_dir=request.config.getoption("--output-dir"),
    )
