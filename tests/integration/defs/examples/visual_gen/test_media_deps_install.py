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
"""CPU coverage for the VisualGen ffmpeg install helper.

This runs the real apt install (no mock) so it verifies the deps install
actually works end-to-end — the bounded/retried command really makes the
``ffmpeg`` CLI available. It needs network + apt but no GPU or model.
"""

import shutil

from defs.examples.visual_gen.conftest import install_ffmpeg_via_apt


def test_install_ffmpeg_via_apt_installs_ffmpeg():
    """Running the helper leaves the ffmpeg CLI available on PATH."""
    install_ffmpeg_via_apt()
    assert shutil.which("ffmpeg") is not None
