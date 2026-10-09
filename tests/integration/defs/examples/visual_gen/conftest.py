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


def install_ffmpeg_via_apt():
    """Install ffmpeg via apt with a bounded timeout and bounded retries.

    Installing at test time depends on external mirrors, which can stall or
    briefly serve inconsistent metadata. Each attempt is guarded by a timeout so
    a stalled mirror fails fast instead of hanging the whole session, and the
    install is retried a few times with a backoff to ride out transient errors.
    ``apt-get update`` is required because the CI image ships without apt lists;
    its exit code is ignored (``|| true``) so an unrelated repo that is
    unreachable or serving mid-sync metadata (e.g. the NVIDIA CUDA repo right
    after a CUDA release) cannot fail the whole step, while the Ubuntu indexes it
    fetches let the ffmpeg install proceed.
    """
    max_attempts = 3
    # `apt-get update` is required (the CI image ships without apt lists), but its
    # exit code is tolerated (`|| true`) so a broken/mid-sync repo (e.g. CUDA)
    # can't fail the step. Bound both apt calls so a stalled mirror fails fast on
    # either the index or the .deb download. DEBIAN_FRONTEND=noninteractive plus a
    # closed stdin keep `dpkg --configure -a` (repairing a killed prior attempt's
    # interrupted unpack) from blocking on a debconf prompt.
    install_cmd = (
        "export DEBIAN_FRONTEND=noninteractive; "
        "dpkg --configure -a < /dev/null || true; "
        'opts="-o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"; '
        "apt-get $opts update || true; "
        "apt-get $opts install -y ffmpeg"
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
        install_ffmpeg_via_apt()


@pytest.fixture(scope="session")
def _visual_gen_lpips_scorer():
    """Reuse one lazily initialized AlexNet LPIPS model for media tests."""
    from defs.examples.visual_gen.visual_gen_test_utils import ReusableLPIPSScorer

    scorer = ReusableLPIPSScorer()
    try:
        yield scorer
    finally:
        scorer.close()
