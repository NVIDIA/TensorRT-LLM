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
"""The machine catalogue: each machine's measured facts, and the ladder it runs by default.

    pytest --collect-only -p qa_selection.plugin --machine=GB200 --selection-out-dir=out/
    -> out/GB200.json "ladder": the profile's `ladder`, or [max_gpu_per_node] without one

A profile's `ladder` is the allocation policy its fleet schedules by: the bucket
clusters get 1,4,8, and a machine always held whole has none. `--ladder` still
overrides it for one run.
"""

import json
from pathlib import Path

import pytest
from mocks import DeviceCases
from qa_selection.core.machine import MachineCatalog, ProfileConfigError

PROFILES = Path(__file__).parents[1] / "core" / "profiles.json"

#: The ladder each machine runs when `--ladder` is absent.
DEFAULT_LADDERS = {
    "H100": [1, 4, 8],
    "B200": [1, 4, 8],
    "B300": [1, 4, 8],
    "GB200": [4],
    "GB300": [4],
    "VR200": [4],
}


def test_every_machine_has_a_default_ladder():
    """The table below covers the catalogue exactly, so a new profile states its ladder."""
    assert sorted(json.loads(PROFILES.read_text())) == sorted(DEFAULT_LADDERS)


@pytest.mark.parametrize("machine", sorted(DEFAULT_LADDERS))
def test_default_ladder(selection, machine):
    """Without `--ladder`, a run uses the profile's ladder, or one rung of the whole node."""
    run = selection.run(DeviceCases.MODULE, f"--machine={machine}", "--selection-out-dir={out}")

    ladder = DEFAULT_LADDERS[machine]
    assert run.record["ladder"] == ladder
    assert run.written == sorted(
        [f"{machine}-{rung}gpu.ids" for rung in ladder] + [f"{machine}.json"]
    )


def test_option_overrides_profile_ladder(selection):
    """`--ladder` replaces the profile's ladder for one run."""
    run = selection.run(
        DeviceCases.MODULE, "--machine=B200", "--ladder=8", "--selection-out-dir={out}"
    )

    assert run.record["ladder"] == [8]
    assert run.written == ["B200-8gpu.ids", "B200.json"]


@pytest.mark.parametrize(
    "ladder, message",
    [
        ([1, 4], "ladder 1,4 must end at max_gpu_per_node 8"),
        ([1, 4, 8, 16], "ladder 1,4,8,16 must end at max_gpu_per_node 8"),
        ([4, 1, 8], "rungs must ascend without repeating"),
    ],
)
def test_profile_ladder_ends_at_the_node(tmp_path, ladder, message):
    """A profile's ladder ascends to exactly its node, or the catalogue is refused.

    Read through the loader: no option points the plugin at another catalogue.
    """
    profile = dict(json.loads(PROFILES.read_text())["B200"], ladder=ladder)
    source = tmp_path / "profiles.json"
    source.write_text(json.dumps({"B200": profile}))

    with pytest.raises(ProfileConfigError, match=message):
        MachineCatalog.load(source)
