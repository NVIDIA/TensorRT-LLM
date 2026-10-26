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
"""A named machine is decided from its catalogue profile alone, without touching hardware.

    pytest --collect-only -p qa_selection.plugin --machine=B200 --selection-out-dir=out/
    -> out/B200.json "profile": B200's catalogue entry

The evidence is a run in a `docker --runtime=runc` container, which has no GPU
and no `nvidia-smi`. The frozen mark conditions are pinned in `test_arch.py`.
"""

import json
from pathlib import Path

from mocks import DeviceCases


class Catalogue:
    """`profiles.json`, and B200's entry in it, without its `ladder`, spelled out."""

    PROFILES = Path(__file__).parents[1] / "core" / "profiles.json"
    B200 = {
        "sm": 100,
        "device_name": "B200",
        "device_memory_mib": 183359,
        "max_gpu_per_node": 8,
        "cpu_arch": "x86_64",
    }


# A named machine is decided without touching hardware / Selection runs where no GPU is present
def test_selection_runs_where_no_gpu_is_present(selection):
    """B200 keeps every device case, and records the profile it decided for."""
    run = selection.run(DeviceCases.MODULE, "--machine=B200", "--selection-out-dir={out}")

    assert run.selected == [
        DeviceCases.UNMARKED,
        DeviceCases.TWO_GPUS,
        DeviceCases.EIGHT_GPUS,
        DeviceCases.EIGHT_RANKS,
    ]
    assert run.record["profile"] == Catalogue.B200


# The record carries the profile it decided for / A catalogue run records its entry
def test_catalogue_run_records_its_entry(selection):
    """`B200.json`'s `profile` is B200's `profiles.json` entry, less its `ladder`."""
    run = selection.run(DeviceCases.MODULE, "--machine=B200", "--selection-out-dir={out}")

    entry = json.loads(Catalogue.PROFILES.read_text())["B200"]
    del entry["ladder"]
    assert run.record["profile"] == entry
