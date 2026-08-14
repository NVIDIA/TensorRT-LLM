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
"""Make optional reuse deps importable + register markers.

The harness optionally imports ``bench_moe.timing.cupti`` (launch_count) and
``defs.perf.gpu_clock_lock`` (clock lock). Both live elsewhere in the tree;
adding their roots here lets those optional imports succeed when present, while
the harness still degrades gracefully if they are not.
"""

import sys
from pathlib import Path

_MICROBENCH = Path(__file__).resolve().parents[1]  # tests/microbenchmarks
_INTEGRATION = _MICROBENCH.parents[0] / "integration"  # tests/integration

for p in (_MICROBENCH, _INTEGRATION):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "discrete: zero-threshold structural assert, pre-merge gate")
    config.addinivalue_line("markers", "continuous: gpu_time vs baseline, post-merge detector")
