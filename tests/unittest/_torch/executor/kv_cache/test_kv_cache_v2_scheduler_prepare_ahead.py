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
"""``TRTLLM_SCHED_PREPARE_AHEAD_PASSES`` of ``KVCacheV2Scheduler``, on the host only.

The real ``scheduler_v2.py`` runs in a child process (``_prepare_ahead_sim.py``) against a fake
KV cache manager with a block radix tree, under a closed-loop multi-turn workload with a
background load on the GPU pool, over a grid of seeds, batch sizes, pool sizes and pool spikes.
The tests pin that, with prepare-ahead on:

* the scheduler admits the same requests in the same passes, with the same claim depths, as the
  same file with the variable unset (the trace and claims digests are equal);
* the pages held ahead never exceed the ceiling when a claim is taken, when a deferred pass ends
  and when context admission starts, and no running request is evicted for them;
* it engages (claims are taken ahead, fewer at admission), and with the variable unset it takes
  no claim ahead, logs nothing and reads no pool state;
* the ceiling knob parses as documented and refuses values outside (0, 0.95];

and that the simulation notices each of the three release paths missing (before an eviction in
phase 1, before admission in phase 2, after a claim that crossed the ceiling).
"""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[5]
SCHEDULER = _ROOT / "tensorrt_llm/_torch/pyexecutor/scheduler/scheduler_v2.py"
SIM = Path(__file__).with_name("_prepare_ahead_sim.py")
AHEAD = "TRTLLM_SCHED_PREPARE_AHEAD_PASSES"
MAX_UTIL = "TRTLLM_SCHED_PREPARE_AHEAD_MAX_UTIL"
INTERVAL = {
    "TRTLLM_SCHED_PREFILL_DECODE_INTERVAL": "8",
    "TRTLLM_SCHED_SHORTEST_PREFILL_FIRST": "1",
    "TRTLLM_SCHED_SPF_MAX_WAIT_PASSES": "24",
}
PASSES = 1500
#: (seed, max batch size, pool pages, spike probability per pass)
SCENARIOS = [(1, 24, 5000, "0"), (2, 6, 5000, "0.05"), (3, 24, 8000, "0.05"), (1, 6, 8000, "0")]
#: name -> (env, ceiling)
ARMS = {
    "passes1": ({AHEAD: "1"}, 0.475),
    "passes4": ({AHEAD: "4"}, 0.475),
    "passes8": ({AHEAD: "8"}, 0.475),
    "passes8_util70": ({AHEAD: "8", MAX_UTIL: "0.70"}, 0.70),
}


def _sim(scheduler, scenario, env):
    seed, batch, pages, spike = scenario
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("TRTLLM_SCHED_")}
    child_env.update(
        SIM_SEED=str(seed),
        SIM_MAX_BATCH=str(batch),
        SIM_TOTAL_PAGES=str(pages),
        SIM_SPIKE_P=spike,
        SIM_PASSES=str(PASSES),
    )
    args = [sys.executable, str(SIM), str(scheduler)]
    args += [f"{k}={v}" for k, v in {**INTERVAL, **env}.items()]
    run = subprocess.run(args, env=child_env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads(run.stdout)


@pytest.fixture(scope="module")
def runs():
    jobs = {(s, "off"): (s, {}) for s in SCENARIOS}
    jobs.update({(s, arm): (s, ARMS[arm][0]) for s in SCENARIOS for arm in ARMS})
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        futures = {key: pool.submit(_sim, SCHEDULER, s, env) for key, (s, env) in jobs.items()}
        return {key: future.result() for key, future in futures.items()}


_GRID = [
    pytest.param(s, arm, id=f"{s[0]}-b{s[1]}-p{s[2]}-spike{s[3]}-{arm}")
    for s in SCENARIOS
    for arm in ARMS
]


@pytest.mark.parametrize("scenario, arm", _GRID)
def test_the_scheduler_admits_what_it_admits_with_the_gate_off(runs, scenario, arm):
    off, on = runs[scenario, "off"], runs[scenario, arm]
    assert on["digest"] == off["digest"]
    assert on["claims_digest"] == off["claims_digest"]
    assert on["ctx_admitted"] == off["ctx_admitted"]


@pytest.mark.parametrize("scenario, arm", _GRID)
def test_the_pages_held_ahead_stay_under_the_ceiling(runs, scenario, arm):
    on, ceiling = runs[scenario, arm], ARMS[arm][1]
    assert (on["I1"], on["I2"], on["I3"], on["I4"]) == (0, 0, 0, 0)
    assert on["ceiling"] == ceiling
    assert on["max_util_with_held"] <= ceiling + 1e-9


@pytest.mark.parametrize("scenario, arm", _GRID)
def test_claims_are_taken_ahead_of_admission(runs, scenario, arm):
    off, on = runs[scenario, "off"], runs[scenario, arm]
    assert on["ahead_claims"] > 0
    assert on["fresh_claims_at_admission"] < off["fresh_claims_at_admission"]
    assert on["init_line"] is not None and f"max_util={ARMS[arm][1]}" in on["init_line"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_the_gate_off_takes_no_claim_ahead_and_reads_no_pool(runs, scenario):
    off = runs[scenario, "off"]
    assert off["ahead_claims"] == 0
    assert off["pool_reads"] == 0
    assert off["init_line"] is None and off["last_stats_line"] is None
    assert runs[scenario, "passes8"]["pool_reads"] > 0


def test_the_ceiling_knob():
    run = subprocess.run(
        [sys.executable, str(SIM), str(SCHEDULER), "knobs"],
        env={k: v for k, v in os.environ.items() if not k.startswith("TRTLLM_SCHED_")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    knobs = json.loads(run.stdout)
    assert knobs["ceiling"] == {
        "off": 0.0,
        "default": 0.475,  # half of max_util_for_resume
        "empty": 0.475,
        "0.7": 0.7,
        "0.95": 0.95,
        "ceiling without passes": 0.0,
    }
    assert knobs["rejected"] == ["0", "-0.1", "0.96", "1.5", "abc"]


_PHASE_1 = """\
        if not success and self._prepared_ahead:
            # Caches prepared ahead of admission give their pages back before
            # any running request is evicted for them.
            self._release_prepared_ahead()
            success = self._try_allocate_generation(req)
"""
_PHASE_2 = """\
        if self._prepared_ahead and not self._ahead_pool_has_headroom():
            # The pool rose past the prepare-ahead ceiling since these claims
            # were taken: give them back before context admission reads it.
            self._release_prepared_ahead()
"""
_ROLLBACK = """\
            if not self._ahead_pool_has_headroom():
                # This claim took a pool group past the ceiling: undo it and
                # leave the rest to the admitting pass.
                type(self).release_context_admission(self, req)
                stats["rolled_back"] += 1
                stats["ceiling_passes"] += 1
                self._ahead_stopped = True
                break
"""


@pytest.mark.parametrize(
    "block, noticed",
    [
        pytest.param(
            _PHASE_1, lambda r, off: r["I4"] > 0 or r["digest"] != off["digest"], id="phase1"
        ),
        pytest.param(_PHASE_2, lambda r, off: r["I3"] > 0, id="phase2"),
        pytest.param(_ROLLBACK, lambda r, off: r["I1"] + r["I2"] > 0, id="rollback"),
    ],
)
def test_the_simulation_notices_a_missing_release_path(runs, tmp_path, block, noticed):
    source = SCHEDULER.read_text()
    assert source.count(block) == 1
    mutant = tmp_path / "scheduler_v2.py"
    mutant.write_text(source.replace(block, ""))
    scenario = (2, 6, 5000, "0.05")
    assert noticed(_sim(mutant, scenario, ARMS["passes8"][0]), runs[scenario, "off"])
