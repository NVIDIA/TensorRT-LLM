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

The evidence is a run in a `docker --runtime=runc` container, where a probe
would fail: a selection, or an error naming `local.json`, shows nothing was
probed. The rerun of a probed directory needs `--probed-out-dir` from a run
where `nvidia-smi` exists.
"""

import json
import shutil

import pytest
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
    B200 = {
        "sm": 100,
        "device_name": "B200",
        "device_memory_mib": 183359,
        "max_gpu_per_node": 8,
        "cpu_arch": "x86_64",
    }

    #: A seed with no valid profile, and the fault its usage error names.
    FAULTS = {
        "invalid_json": ('{\n  "profile": {\n    "sm": 120,,\n  }\n}\n', "invalid JSON at line 3"),
        "no_profile": (json.dumps({"machine": "local"}), 'no "profile" object'),
        "no_cpu_arch": (
            json.dumps({"profile": {k: v for k, v in H20.items() if k != "cpu_arch"}}),
            "profile is missing cpu_arch",
        ),
        "unknown_field": (
            json.dumps({"profile": dict(H20, rack="r7")}),
            "profile has unexpected field rack",
        ),
        "rung_above_node": (
            json.dumps({"profile": dict(H20, ladder=[1, 8])}),
            "profile ladder 1,8 exceeds max_gpu_per_node 4",
        ),
        "sm_text": (
            json.dumps({"profile": dict(H20, sm="120")}),
            'profile.sm must be a positive integer, got "120"',
        ),
        "no_gpus": (
            json.dumps({"profile": dict(H20, max_gpu_per_node=0)}),
            "profile.max_gpu_per_node must be a positive integer, got 0",
        ),
    }
    REMEDY = "fix or delete it to probe this host, or name a catalogue machine with --machine=NAME"

    @classmethod
    def of(cls, profile: dict) -> str:
        """`local.json` as a caller writes it: the profile object alone."""
        return json.dumps({"profile": profile})


# A saved local profile is read before the host is probed / A saved profile is used instead of probing
def test_saved_profile_is_used_instead_of_probing(selection):
    """The saved H20 profile decides, on a ladder of its four GPUs, and becomes the record."""
    selection.seed(Saved.NAME, Saved.of(Saved.H20))

    run = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")

    assert run.record["profile"] == Saved.H20
    assert run.record["ladder"] == [4]
    assert run.selected == [DeviceCases.UNMARKED, DeviceCases.TWO_GPUS]
    assert run.written == ["local-4gpu.ids", "local.json"]
    assert len(run.record["tests"]) == 4


# A saved local profile is read before the host is probed / A probed profile is saved for the next run
def test_probed_profile_is_saved_for_the_next_run(selection, probed_out_dir):
    """A probed run's directory, collected again, gives the same profile and the same ids."""
    if probed_out_dir is None or not (probed_out_dir / Saved.NAME).is_file():
        pytest.skip("needs --probed-out-dir=DIR, written by a run where nvidia-smi exists")
    probed = json.loads((probed_out_dir / Saved.NAME).read_text())
    shutil.copytree(probed_out_dir, selection.out_dir, dirs_exist_ok=True)

    run = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")

    assert run.record["profile"] == probed["profile"]
    assert run.ids(1) == (probed_out_dir / "local-1gpu.ids").read_text().splitlines()


# A saved local profile is read before the host is probed / A bad saved profile is a usage error
@pytest.mark.parametrize("case", sorted(Saved.FAULTS))
def test_bad_saved_profile_is_a_usage_error(selection, case):
    """The error names `local.json` and its fault, never a probe, and leaves the file alone."""
    content, fault = Saved.FAULTS[case]
    seeded = selection.seed(Saved.NAME, content)

    run = selection.refuse(DeviceCases.MODULE, "--selection-out-dir={out}")

    run.result.stderr.fnmatch_lines([f"*{seeded}: {fault}; {Saved.REMEDY}*"])
    assert "nvidia-smi" not in run.result.stderr.str()
    assert seeded.read_text(encoding="utf-8") == content
    assert not (run.path / DeviceCases.IMPORT_SENTINEL).exists()


# A saved local profile is read before the host is probed / A saved ladder is the default
def test_saved_ladder_is_the_default(selection):
    """`ladder` in `local.json` is the run's ladder when `--ladder` is absent."""
    selection.seed(Saved.NAME, Saved.of(dict(Saved.H20, ladder=[1, 4])))

    run = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")

    assert run.record["ladder"] == [1, 4]
    assert run.written == ["local-1gpu.ids", "local-4gpu.ids", "local.json"]


# A saved local profile is read before the host is probed /
# An explicit ladder replaces the saved one
def test_explicit_ladder_replaces_the_saved_one(selection):
    """`--ladder=1` is written into `local.json`, so the next run without it keeps `1`."""
    selection.seed(Saved.NAME, Saved.of(dict(Saved.H20, ladder=[1, 4])))

    explicit = selection.run(DeviceCases.MODULE, "--ladder=1", "--selection-out-dir={out}")
    following = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")

    assert explicit.record["ladder"] == following.record["ladder"] == [1]
    assert following.record["profile"] == dict(Saved.H20, ladder=[1])


# A saved local profile is read before the host is probed / A named machine ignores the saved profile
def test_named_machine_ignores_the_saved_profile(selection):
    """`--machine=B200` selects for B200 and leaves `local.json` byte for byte as it was."""
    content = Saved.of(Saved.H20)
    seeded = selection.seed(Saved.NAME, content)

    run = selection.run(DeviceCases.MODULE, "--machine=B200", "--selection-out-dir={out}")

    assert run.record["profile"] == dict(Saved.B200, ladder=[1, 4, 8])
    assert seeded.read_text(encoding="utf-8") == content
