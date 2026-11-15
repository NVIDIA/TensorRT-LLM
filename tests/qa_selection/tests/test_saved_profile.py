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
"""A saved local profile is read before the host is probed.

    out/local.json  {"profile": {...}}
    pytest --collect-only -p qa_selection.plugin --selection-out-dir=out/
    -> the saved profile, nothing probed; out/local.json is then the run's record

Run without a GPU, where a probe would fail: a selection, or an error naming
`local.json`, shows nothing was probed.
"""

import json

from mocks import DeviceCases


class Saved:
    """The `local.json` a caller saves: H20's measured profile on a 4-GPU node."""

    NAME = "local.json"
    H20 = {
        "sm": 90,
        "device_name": "H20",
        "device_memory_mib": 97871,
        "max_gpu_per_node": 4,
        "cpu_arch": "x86_64",
    }

    @classmethod
    def of(cls, profile: dict) -> str:
        """`local.json` as a caller writes it: the profile object alone."""
        return json.dumps({"profile": profile})


# A saved local profile is read before the host is probed / A saved profile is used instead of probing
def test_saved_profile_is_used_instead_of_probing(selection):
    """The saved H20 profile decides, on a ladder of its four GPUs, and becomes the record."""
    selection.seed(Saved.NAME, Saved.of(Saved.H20))

    run = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")

    assert run.record["machine"] == "local"
    assert run.record["profile"] == Saved.H20
    assert run.record["ladder"] == [4]
    assert run.selected == [DeviceCases.UNMARKED, DeviceCases.TWO_GPUS]
    assert run.written == ["local-4gpu.ids", "local.json"]


# A saved local profile is read before the host is probed / A bad saved profile is a usage error
def test_bad_saved_profile_is_a_usage_error(selection):
    """The error names `local.json` and its fault, never a probe, and leaves the file alone."""
    content = json.dumps({"machine": "local"})
    seeded = selection.seed(Saved.NAME, content)

    run = selection.refuse(DeviceCases.MODULE, "--selection-out-dir={out}")

    run.result.stderr.fnmatch_lines([f'*{seeded}: no "profile" object*'])
    assert "nvidia-smi" not in run.result.stderr.str()
    assert seeded.read_text(encoding="utf-8") == content
    assert not (run.path / DeviceCases.IMPORT_SENTINEL).exists()
