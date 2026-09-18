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

The package splits on what the code is allowed to depend on:

    core/          the decision layer -- stdlib only, never pytest
    collection.py  options, markers, the item adapter, the decisions
    report.py      the .ids files, the JSON record, the terminal summary
    plugin.py      the pytest hooks; the `-p qa_selection.plugin` entry point

Code outside `core/` may import pytest; code inside it may not. That is what
lets `core/` be exercised with no pytest session, no hardware and no wheel --
see `core/__init__.py` for its own reading order.

No executable statements: pytest imports this package before any conftest in
order to load the plugin, so it must stay cheap.
"""
