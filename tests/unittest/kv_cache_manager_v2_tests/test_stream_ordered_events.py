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
"""Stream-ordered KV-cache events (TLLM_KV_CACHE_MANAGER_V2_STREAM_ORDERED_EVENTS).

The switch is read once per process, so every case runs in a child process with the variable set
or unset. Each case holds a producer stream open with a host gate and then checks two things:
the ordering the wait / merge helpers establish (a consumer marker must not complete while the
gate is held, and must complete after it is released), and, with the switch on, which waits and
merges the helpers skipped (the process-wide counters).
"""

import json
import os
import subprocess
import sys

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV = "TLLM_KV_CACHE_MANAGER_V2_STREAM_ORDERED_EVENTS"


def _introspection():
    try:
        from bindings.internal.batch_manager.kv_cache_manager_v2 import _introspection
    except ImportError:
        from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2 import _introspection
    return _introspection


def _counters_delta(intro, before):
    after = intro.stream_ordered_event_counters()
    return {k: after[k] - before[k] for k in after}


def _worker() -> dict:
    """Run every case in this process and return their observations."""
    sys.path.insert(0, _TEST_DIR)
    from cuda_test_utils import CachedCudaStream, init_cuda_once
    from kernels import HostGate

    init_cuda_once()
    intro = _introspection()
    event = intro.CachedCudaEvent
    out = {"enabled": intro.stream_ordered_events_enabled()}
    a_holder, b_holder, c_holder = CachedCudaStream(), CachedCudaStream(), CachedCudaStream()
    a, b, c = int(a_holder.handle), int(b_holder.handle), int(c_holder.handle)

    def sync_all():
        for h in (a_holder, b_holder, c_holder):
            h.synchronize()

    # 1. A stream waiting for its own earlier event.
    before = intro.stream_ordered_event_counters()
    e = event(a)
    intro.stream_wait_events(a, [e, e])
    e.wait_in_stream(a)
    sync_all()
    out["same_stream"] = _counters_delta(intro, before)

    # 2. Two events of one gated stream: the consumer must follow the later one.
    gate = HostGate()
    try:
        early = event(a)
        gate.block_stream(a)
        late = event(a)
        before = intro.stream_ordered_event_counters()
        intro.stream_wait_events(b, [early, late])
        marker = event(b)
        out["latest_blocked"] = not marker.query_complete()
        gate.release()
        marker.synchronize()
        out["latest_done_after_release"] = marker.is_closed()
        out["latest"] = _counters_delta(intro, before)
    finally:
        gate.close()
    sync_all()

    # 3. Merge of two events of one gated stream.
    gate = HostGate()
    try:
        early = event(a)
        gate.block_stream(a)
        late = event(a)
        before = intro.stream_ordered_event_counters()
        merged = intro.merge_events([early, late])
        merged.wait_in_stream(b)
        marker = event(b)
        out["merge_same_blocked"] = not marker.query_complete()
        gate.release()
        marker.synchronize()
        out["merge_same"] = _counters_delta(intro, before)
    finally:
        gate.close()
    sync_all()

    # 4. Merge of pending events of two gated streams: needs both.
    gate_a, gate_b = HostGate(), HostGate()
    try:
        gate_a.block_stream(a)
        gate_b.block_stream(b)
        ea, eb = event(a), event(b)
        before = intro.stream_ordered_event_counters()
        merged = intro.merge_events([ea, eb])
        merged.wait_in_stream(c)
        marker = event(c)
        gate_a.release()
        a_holder.synchronize()
        out["merge_cross_blocked_on_b"] = not marker.query_complete()
        gate_b.release()
        marker.synchronize()
        out["merge_cross"] = _counters_delta(intro, before)
    finally:
        gate_a.close()
        gate_b.close()
    sync_all()

    # 5. Merge of a pending event with a completed (not yet closed) event of another stream.
    gate = HostGate()
    try:
        eb = event(b)
        b_holder.synchronize()
        gate.block_stream(a)
        ea = event(a)
        before = intro.stream_ordered_event_counters()
        merged = intro.merge_events([eb, ea])
        merged.wait_in_stream(c)
        marker = event(c)
        out["merge_done_blocked"] = not marker.query_complete()
        gate.release()
        marker.synchronize()
        out["merge_done"] = _counters_delta(intro, before)
    finally:
        gate.close()
    sync_all()
    return out


def _run(enabled: bool) -> dict:
    env = dict(os.environ)
    if enabled:
        env[_ENV] = "1"
    else:
        env.pop(_ENV, None)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT ") :])


@pytest.mark.parametrize("enabled", [False, True], ids=["off", "on"])
def test_orderings_hold(enabled):
    r = _run(enabled)
    assert r["enabled"] is enabled
    assert r["latest_blocked"] and r["latest_done_after_release"]
    assert r["merge_same_blocked"]
    assert r["merge_cross_blocked_on_b"]
    assert r["merge_done_blocked"]
    if not enabled:
        for case in ("same_stream", "latest", "merge_same", "merge_cross", "merge_done"):
            assert all(v == 0 for v in r[case].values()), (case, r[case])


def test_redundant_waits_and_merges_are_skipped():
    r = _run(True)
    # Own-stream waits: two copies through the list helper, one through wait_in_stream.
    assert r["same_stream"]["waits_issued"] == 0
    assert r["same_stream"]["waits_elided"] == 3
    # Two events of one other stream: only the later one is waited for.
    assert r["latest"]["waits_issued"] == 1
    assert r["latest"]["waits_elided"] == 1
    # One stream: the later event is the merge, no temporary stream.
    assert r["merge_same"]["merges_without_stream"] == 1
    assert r["merge_same"]["merges_with_stream"] == 0
    # Two pending streams: a real merge.
    assert r["merge_cross"]["merges_with_stream"] == 1
    # A completed event is dropped, so the pending one is the merge.
    assert r["merge_done"]["merges_without_stream"] == 1
    assert r["merge_done"]["merges_with_stream"] == 0


if __name__ == "__main__" and "--worker" in sys.argv:
    print("RESULT " + json.dumps(_worker()), flush=True)
