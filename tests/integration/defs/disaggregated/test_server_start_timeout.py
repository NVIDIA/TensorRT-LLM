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
"""GPU-free unit test for the disaggregated readiness budget.

pytest -sv disaggregated/test_server_start_timeout.py
"""

from disagg_test_utils import resolve_server_start_timeout


def _config(ctx_pp: int, gen_pp: int) -> dict:
    return {
        "context_servers": {"pipeline_parallel_size": ctx_pp},
        "generation_servers": {"pipeline_parallel_size": gen_pp},
    }


def test_resolve_server_start_timeout():
    """900s for PP on an SM100-family GPU, 300s otherwise; an explicit value wins.

    Budgets are written out as numbers so that a change to either constant fails here.
    """
    top_level_pp2 = {"pipeline_parallel_size": 2, "context_servers": {}, "generation_servers": {}}
    server_pp1_overrides = {
        "pipeline_parallel_size": 2,
        "context_servers": {"pipeline_parallel_size": 1},
        "generation_servers": {"pipeline_parallel_size": 1},
    }
    # (case, config, explicit budget, SM version, expected budget)
    cases = [
        ("ctx PP2 on SM100", _config(2, 1), None, 100, 900),
        ("gen PP2 on SM103", _config(1, 2), None, 103, 900),
        ("PP4 on SM107", _config(4, 4), None, 107, 900),
        ("PP4 on SM90", _config(4, 4), None, 90, 300),
        ("PP4 on SM120", _config(4, 4), None, 120, 300),
        ("no PP on SM100", _config(1, 1), None, 100, 300),
        ("no server sections", {}, None, 100, 300),
        ("top-level PP inherited", top_level_pp2, None, 100, 900),
        ("server PP1 overrides top-level", server_pp1_overrides, None, 100, 300),
        ("explicit below PP budget", _config(4, 4), 600, 100, 600),
        ("explicit above default", _config(1, 1), 1200, 100, 1200),
    ]
    failures = [
        f"{case}: got {actual}s, expected {expected}s"
        for case, config, explicit, sm_version, expected in cases
        if (actual := resolve_server_start_timeout(config, explicit, sm_version)) != expected
    ]
    assert not failures, "\n".join(failures)
