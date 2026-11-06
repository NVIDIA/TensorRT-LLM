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
"""Loading the plugin selects for this host by default: with no `--machine`, it is probed.

    pytest --collect-only -p qa_selection.plugin --selection-out-dir=out/
    -> out/local.json, out/local-1gpu.ids

Runs only where `nvidia-smi` does: the dev box's GPU container
(`docker run --gpus all`). The expected profile is that box's, one
`NVIDIA RTX PRO 6000 Blackwell Workstation Edition`.
"""

import json
import shutil

import pytest
from mocks import DeviceCases
from qa_selection.core.machine import MachineCatalog

pytestmark = pytest.mark.skipif(
    shutil.which("nvidia-smi") is None, reason="needs nvidia-smi to probe this host"
)


class DevBox:
    """What the dev box's GPU container probes as."""

    PROFILE = {
        "sm": 120,
        "device_name": "RTX PRO 6000 Blackwell Workstation Edition",
        "device_memory_mib": 97887,
        "max_gpu_per_node": 1,
        "cpu_arch": "x86_64",
    }
    HEADER = (
        "qa selection: local (RTX PRO 6000 Blackwell Workstation Edition, sm 120, "
        "1 GPUs/node), ladder 1"
    )


class Catalogue:
    """GB200's `profiles.json` entry, without its `ladder`."""

    GB200 = {
        "sm": 100,
        "device_name": "GB200",
        "device_memory_mib": 189471,
        "max_gpu_per_node": 4,
        "cpu_arch": "aarch64",
    }


# Loading the plugin selects for this host by default / The default target is this host
def test_default_target_is_this_host(selection, probed_out_dir):
    """No `--machine`: the record is `local`, for the probed profile on a ladder of its GPUs."""
    run = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")

    assert run.record["machine"] == "local"
    assert run.record["profile"] == DevBox.PROFILE
    assert run.record["ladder"] == [1]
    assert run.written == ["local-1gpu.ids", "local.json"]

    assert run.selected == [DeviceCases.UNMARKED]
    assert run.record["deselected_by_reason"] == {
        "skip_less_device(2): needs 2 GPUs, largest rung is 1": [DeviceCases.TWO_GPUS],
        "skip_less_device(8): needs 8 GPUs, largest rung is 1": [DeviceCases.EIGHT_GPUS],
        "skip_less_mpi_world_size(8): needs 8 GPUs, largest rung is 1": [DeviceCases.EIGHT_RANKS],
    }

    # Loading the plugin selects for this host by default / The header names the probed device
    assert DevBox.HEADER in run.result.stdout.lines

    if probed_out_dir is not None:
        shutil.copytree(run.out_dir, probed_out_dir, dirs_exist_ok=True)


# Loading the plugin selects for this host by default / An assigned machine overrides the host
def test_assigned_machine_overrides_the_host(selection):
    """`--machine=GB200` on this host selects for GB200's catalogue profile and ladder."""
    run = selection.run(DeviceCases.MODULE, "--machine=GB200", "--selection-out-dir={out}")

    assert run.record["machine"] == "GB200"
    assert run.record["profile"] == Catalogue.GB200
    assert run.record["ladder"] == [4]
    assert run.written == ["GB200-4gpu.ids", "GB200.json"]


# The record carries the profile it decided for / A probed profile becomes a catalogue entry
def test_probed_profile_becomes_a_catalogue_entry(selection, tmp_path):
    """`local.json`'s `profile`, saved under a name, loads as that machine's profile."""
    run = selection.run(DeviceCases.MODULE, "--selection-out-dir={out}")
    catalogue = tmp_path / "profiles.json"
    catalogue.write_text(json.dumps({"RTX_PRO_6000_WS": run.record["profile"]}))

    profile = MachineCatalog.load(catalogue).profile_for("RTX_PRO_6000_WS")

    assert {field: getattr(profile, field) for field in DevBox.PROFILE} == DevBox.PROFILE
