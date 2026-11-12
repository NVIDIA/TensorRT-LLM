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
"""Without `--machine`, the plugin selects for this host, probed with `nvidia-smi`.

    pytest --collect-only -p qa_selection.plugin --selection-out-dir=out/
    -> out/local.json, out/local-<GPUs>gpu.ids

The QA pipeline's single-node jobs collect this way on their GPU node. Runs
only where `nvidia-smi` does, such as a `docker run --gpus all` container.
"""

import platform
import shutil

import pytest
from mocks import DeviceCases

pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None, reason="needs nvidia-smi to probe this host"
)


# Loading the plugin selects for this host by default / The default target is this host
def test_default_target_is_this_host(selection):
    """No `--machine`: the record is `local`, for the probed profile on a ladder of its GPUs."""
    run = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")

    profile = run.record["profile"]
    gpus = profile["max_gpu_per_node"]
    assert run.record["machine"] == "local"
    assert sorted(profile) == [
        "cpu_arch",
        "device_memory_mib",
        "device_name",
        "max_gpu_per_node",
        "sm",
    ]
    assert profile["cpu_arch"] == platform.machine()
    assert run.record["ladder"] == [gpus]
    assert run.written == [f"local-{gpus}gpu.ids", "local.json"]
    assert DeviceCases.UNMARKED in run.selected
