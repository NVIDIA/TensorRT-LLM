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
"""Fixtures shared by VisualGen example tests."""

import shutil

import pytest
from defs.examples.visual_gen.visual_gen_test_utils import _prepare_vbench_repo
from defs.trt_test_alternative import check_call


@pytest.fixture(scope="session")
def _visual_gen_deps(llm_venv):
    """Install the Python and system media dependencies once per session."""
    llm_venv.run_cmd(["-m", "pip", "install", "av"])
    llm_venv.run_cmd(["-m", "pip", "install", "diffusers>=0.37.0"])
    if shutil.which("ffmpeg") is None:
        check_call(["apt-get", "update", "-y"], shell=False)
        check_call(["apt-get", "install", "-y", "ffmpeg"], shell=False)


@pytest.fixture(scope="session")
def _visual_gen_lpips_scorer():
    """Reuse one lazily initialized AlexNet LPIPS model for media tests."""
    from defs.examples.visual_gen.visual_gen_test_utils import ReusableLPIPSScorer

    scorer = ReusableLPIPSScorer()
    try:
        yield scorer
    finally:
        scorer.close()


@pytest.fixture(scope="session")
def vbench_repo_root(llm_venv):
    """Prepare the pinned VBench checkout and return its repository root."""
    return _prepare_vbench_repo(llm_venv)
