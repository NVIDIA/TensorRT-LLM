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
import subprocess
import time

import pytest
from defs.trt_test_alternative import check_call, print_warning

# Fixtures shared by VisualGen example tests.


def _install_ffmpeg_via_apt():
    """Install ffmpeg via apt with a bounded timeout and bounded retries.

    Installing at test time depends on external mirrors, which can stall or
    briefly serve inconsistent metadata. Each attempt is guarded by a timeout so
    a stalled mirror fails fast instead of hanging the whole session, and the
    install is retried a few times with a backoff to ride out transient errors.
    ``apt-get update`` is best-effort so an unrelated repo serving mid-sync
    metadata (e.g. the NVIDIA CUDA repo right after a CUDA release) cannot block
    the install; the ffmpeg install from Ubuntu is the real gate.
    """
    max_attempts = 3
    # Bound apt's per-connection timeout so a stalled mirror fails within seconds;
    # retries are owned by the loop below. `update` is best-effort, `install` gates.
    install_cmd = (
        "apt-get -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 "
        "update || true; "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg"
    )

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            # 240s per attempt: generous headroom over a normal install.
            check_call(["bash", "-c", install_cmd], shell=False, timeout=240)
            return
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as err:
            last_err = err
            print_warning(f"ffmpeg apt install attempt {attempt}/{max_attempts} failed: {err}")
            if attempt < max_attempts:
                time.sleep(15)  # backoff between attempts
    pytest.fail(f"Failed to install ffmpeg via apt after {max_attempts} attempts: {last_err}")


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
        _install_ffmpeg_via_apt()


@pytest.fixture(scope="session")
def _visual_gen_lpips_scorer():
    """Reuse one lazily initialized AlexNet LPIPS model for media tests."""
    from defs.examples.visual_gen.visual_gen_test_utils import ReusableLPIPSScorer

    scorer = ReusableLPIPSScorer()
    try:
        yield scorer
    finally:
        scorer.close()
