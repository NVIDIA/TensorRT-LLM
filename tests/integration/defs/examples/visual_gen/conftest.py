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

import importlib.util
import os
import shutil

import pytest
from defs.trt_test_alternative import check_call

# Fixtures shared by VisualGen example tests.


@pytest.fixture(scope="session")
def _visual_gen_deps(llm_venv, _auto_install_media_deps):
    """Ensure PyAV and ffmpeg are available for VisualGen tests.

    Uses packages already on the system when present. Otherwise set
    ``TRTLLM_AUTO_INSTALL_MEDIA_DEPS=1`` to install them. OpenCV is handled by
    ``_auto_install_media_deps``.
    """
    av_available = importlib.util.find_spec("av") is not None
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if av_available and ffmpeg_available:
        return
    if os.environ.get("TRTLLM_AUTO_INSTALL_MEDIA_DEPS", "0") != "1":
        pytest.fail(
            "PyAV and/or ffmpeg are not installed. Install them manually, or set "
            "TRTLLM_AUTO_INSTALL_MEDIA_DEPS=1 to auto-install."
        )
    if not av_available:
        llm_venv.run_cmd(["-m", "pip", "install", "av"])
    if not ffmpeg_available:
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
