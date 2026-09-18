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
"""The selection decision layer: stdlib only, and never pytest.

    machines.py    machine facts, read from profiles.json          (C1)
    rules.py       skip rules, read from rules.json                (C2)
    selector.py    feasibility: can this machine run this test     (C2)
    allocation.py  demand and rung: which allocation it belongs to (C3)

Read in that order; each module imports only the ones above it. Nothing here
imports pytest, torch, or any hardware probe, which is what lets a login node
with no GPU decide a whole test list. The package above this one adapts pytest
to it, so the invariant is checkable:

    grep -lE '^(import|from) pytest' tests/qa_selection/core/*.py  # prints nothing
"""
