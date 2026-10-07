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
"""Select the tests a target machine can run, before Slurm allocates it.

    plugin.py        the options, the markers, the item adapter, the six hooks;
                     the `-p qa_selection.plugin` entry point, and the only
                     module here that imports pytest

    core/            the decisions -- stdlib only, never pytest
      ladder.py      what a legal ladder is                    -> Ladder
      machines.py    the machines selection can target         -> MachineProfile
      rules.py       the curated skip rules, and what holds    -> SkipRuleTable
      markers.py     the marks whose first argument is a need  -> ResourceMarkers
      selector.py    can this machine run this test            -> Decision
      allocation.py  how much it wants, which rung takes it    -> GpuDemand, Assignment
      artifacts.py   what the output files are called          -> ArtifactNames
      request.py     what one run was asked for                -> SelectionRequest
      selection.py   what it decided about every test          -> Selection
      report.py      the .ids lists and the JSON record

Three of those read a JSON file beside them: `machines.py` reads
`profiles.json`, `rules.py` reads `rules.json`, `markers.py` reads
`markers.json`. Every `core/` module imports only the ones above it, and
nothing imports `plugin.py`.

No executable statements: pytest imports this package before any conftest, to
load the plugin.
"""
