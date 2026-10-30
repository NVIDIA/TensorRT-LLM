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
"""TEMPORARY, for the draft pull request only: a test that fails once and passes on its rerun.

A stage that runs it reruns it, so the stage publishes a per-stage rerun report and contributes to
the build-level one. The marker file is specific to the stage's workspace, so the rerun, which
runs in the same place, finds it while another stage or build does not.
"""

import hashlib
import os
from pathlib import Path


def _marker() -> Path:
    place = "|".join(os.environ.get(name, "") for name in ("BUILD_TAG", "NODE_NAME", "WORKSPACE"))
    digest = hashlib.sha256(place.encode()).hexdigest()[:16]
    return Path("/tmp") / f"rerun-once-probe-{digest}"


def test_fails_once_then_passes_on_rerun():
    marker = _marker()
    if not marker.exists():
        marker.touch()
        raise AssertionError(f"first run: created {marker}; the rerun passes")
