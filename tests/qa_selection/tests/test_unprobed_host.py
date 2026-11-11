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
"""A host that cannot be probed is a usage error, raised before any test module is imported.

    pytest --collect-only -p qa_selection.plugin --selection-out-dir=out/
    -> usage error: cannot probe this host ...: nvidia-smi not found; ... --machine=NAME

Runs only where `nvidia-smi` does not exist, such as a `docker --runtime=runc`
container.
"""

import shutil

import pytest
from mocks import DeviceCases


# A host that cannot be probed is a usage error / No `nvidia-smi`
@pytest.mark.skipif(shutil.which("nvidia-smi") is not None, reason="this host has nvidia-smi")
def test_no_nvidia_smi(selection):
    """No `--machine` and no `nvidia-smi`: a usage error naming both, and nothing written."""
    run = selection.refuse(DeviceCases.MODULE, "--selection-out-dir={out}")

    run.result.stderr.fnmatch_lines(
        [
            "*cannot probe this host for its machine profile: nvidia-smi not found; "
            "name a catalogue machine with --machine=NAME*"
        ]
    )
    assert not run.out_dir.exists()
    assert not (run.path / DeviceCases.IMPORT_SENTINEL).exists()
