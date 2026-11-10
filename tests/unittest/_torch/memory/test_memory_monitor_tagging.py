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
"""Tag-based attribution of torch-visible memory.

Each trap pinned here is silent: the ring buffer drops the earliest tags
(which are the weights), a non-native allocator accepts a tag and discards
it, an unsupported setup produces a breakdown of zeros that reads exactly
like a process holding nothing, and a block allocated after the map is
frozen can sit at the address of a tagged block that was freed.

Needs a GPU: the attribution is read back out of a real allocator snapshot.
"""

import pytest
import torch
from memory_monitor_test_utils import MB

from tensorrt_llm._torch.memory.monitor.checks import check_tag_coverage
from tensorrt_llm._torch.memory.monitor.tagging import (
    UNTAGGED,
    TagSnapshot,
    freeze_tag_map,
    live_bytes_by_tag,
    start_tagging_window,
    stop_tagging_window,
    tag,
    tagging_supported,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="reads a real allocator snapshot"
)


@pytest.fixture
def window(monkeypatch):
    """A tagging window, with the monitor on, that is always closed again."""
    from tensorrt_llm._torch.memory.monitor.tagging import clear_claims, set_frozen_map

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    clear_claims()
    started = start_tagging_window()
    try:
        yield started
    finally:
        stop_tagging_window()
        clear_claims()
        set_frozen_map(TagSnapshot(supported=False))


def test_tags_land_on_the_right_bytes_and_survive_the_window(window):
    if not window:
        pytest.skip("tagging unsupported here")

    with tag("WEIGHTS"):
        weights = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
    with tag("SAMPLER"):
        sampler = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    with tag("OUTER"):
        a = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        with tag("INNER"):
            b = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        c = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    plain = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")

    try:
        frozen = freeze_tag_map()
        by_tag = live_bytes_by_tag(frozen)
        assert by_tag["WEIGHTS"] == 64 * MB
        assert by_tag["SAMPLER"] == 32 * MB
        assert by_tag[UNTAGGED] >= 16 * MB
        # Reporting only the tags would read as a complete breakdown.
        assert UNTAGGED in by_tag
        assert by_tag.get("INNER", 0) == 16 * MB
        # Clearing instead of restoring would leave this at 16 MiB.
        assert by_tag.get("OUTER", 0) == 32 * MB
        # Check B holds on a real reading.
        assert all(r.passed for r in check_tag_coverage(by_tag, torch.cuda.memory_allocated()))

        # Why the map is frozen rather than replayed on demand.
        stop_tagging_window()
        # Steady state: plenty of churn, no recording. A design that
        # replayed the trace at read time would have lost WEIGHTS by now.
        for _ in range(3000):
            scratch = torch.empty(4096, dtype=torch.uint8, device="cuda")
            del scratch
        activation = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
        by_tag = live_bytes_by_tag(frozen)
        assert by_tag["WEIGHTS"] == 64 * MB, (
            "the frozen map must keep attributing weights once the trace has moved on"
        )
        assert by_tag[UNTAGGED] >= 64 * MB
        del activation
    finally:
        del weights, sampler, a, b, c, plain


def test_unsupported_is_not_zero(monkeypatch):
    """An unsupported setup must not look like an empty one."""
    import tensorrt_llm._torch.memory.monitor.tagging as mod

    empty = TagSnapshot(supported=False)
    assert live_bytes_by_tag(empty) == {}, "a dict of zeros would read as a measurement"
    assert not empty

    with monkeypatch.context() as m:
        m.setattr(mod, "tagging_supported", lambda: False)
        # Call sites must not need a guard of their own.
        with mod.tag("WEIGHTS"):
            pass
        assert mod.freeze_tag_map().supported is False

    with monkeypatch.context() as m:
        # The torch docs are explicit that cudaMallocAsync and pluggable
        # allocators accept the call and drop the tag. Detecting that up
        # front is the difference between "unsupported" and a small,
        # plausible, wrong number.
        m.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
        assert tagging_supported() is False, "cudaMallocAsync"

    def boom():
        raise RuntimeError("pluggable allocator installed")

    with monkeypatch.context() as m:
        m.setattr(torch.cuda.memory, "_memory_metadata_supported", boom)
        assert tagging_supported() is False, "pluggable allocator"


def test_check_b():
    by_tag = {"WEIGHTS": 64 * MB, UNTAGGED: 16 * MB}
    assert check_tag_coverage(by_tag, torch_live_bytes=80 * MB)[0].passed, "tags fit inside live"

    # What a stale frozen tag looks like: an address reused by an
    # untagged allocation whose reuse event the ring buffer dropped.
    by_tag = {"WEIGHTS": 512 * MB, UNTAGGED: 0}
    results = check_tag_coverage(by_tag, torch_live_bytes=64 * MB)
    assert not results[0].passed, "tags exceed live"
    assert "tagged=536870912" in results[0].detail

    results = check_tag_coverage({}, torch_live_bytes=64 * MB)
    assert results[0].passed, "unsupported is not a failure"
    assert "not reported as zero" in results[0].detail

    # Activations are never tagged, so a large untagged share is normal
    # and must not be raised as a problem.
    by_tag = {"WEIGHTS": 1 * MB, UNTAGGED: 99 * MB}
    results = check_tag_coverage(by_tag, torch_live_bytes=100 * MB)
    assert all(r.passed for r in results)
    assert "99.0%" in results[1].detail


def _claimed():
    return live_bytes_by_tag(TagSnapshot(supported=False)).get("WEIGHTS", 0)


def test_claimed_module_memory(window):
    """Weights are attributed by walking the live module at report time."""
    import gc

    from tensorrt_llm._torch.memory.monitor.tagging import (
        claim_module_memory,
        clear_claims,
        set_frozen_map,
    )

    module = torch.nn.Linear(512, 512).cuda()
    module.register_buffer("scratch", torch.empty(1024, 1024, device="cuda"))
    assert claim_module_memory(module, "WEIGHTS") is True
    # The buffer alone is 4 MiB; weight and bias add to that.
    assert _claimed() >= 4 * MB, "parameters and buffers"
    # The closing freeze rebuilds the trace-derived map wholesale; claims
    # live elsewhere so they are not discarded by it.
    set_frozen_map(freeze_tag_map())
    assert live_bytes_by_tag(freeze_tag_map()).get("WEIGHTS", 0) > 0, "survives the freeze"
    set_frozen_map(TagSnapshot(supported=False))

    # A claim must not keep the model alive, nor survive it.
    del module
    gc.collect()
    assert "WEIGHTS" not in live_bytes_by_tag(TagSnapshot(supported=False)), "released module"
    clear_claims()

    # Tied embeddings: two parameters, one storage. Summing their
    # logical sizes double counts; claiming by block counts it once.
    shared = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")

    class Tied(torch.nn.Module):
        def __init__(self, storage):
            super().__init__()
            self.embed = torch.nn.Parameter(storage, requires_grad=False)
            self.head = torch.nn.Parameter(storage, requires_grad=False)

    tied = Tied(shared)
    claim_module_memory(tied, "WEIGHTS")
    assert _claimed() == 8 * MB, "shared storage counted once"
    del tied, shared
    clear_claims()

    cpu_module = torch.nn.Linear(8, 8)
    claim_module_memory(cpu_module, "WEIGHTS")
    assert "WEIGHTS" not in live_bytes_by_tag(TagSnapshot(supported=False)), "CPU module"


def test_freed_weights_do_not_label_their_successor(window):
    # An address recorded at load time keeps its label after the block
    # is freed and reused. Walking the module sees only what it holds
    # now, so a replaced weight's old block, reused by an activation, is
    # not counted as weights.
    from tensorrt_llm._torch.memory.monitor.tagging import claim_module_memory

    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(
        torch.empty(16 * MB, dtype=torch.uint8, device="cuda"), requires_grad=False
    )
    claim_module_memory(module, "WEIGHTS")
    assert _claimed() == 16 * MB

    # Replace the weight (a smaller one, as quantisation would), free the
    # old block, and let an unrelated allocation take its place.
    module.weight = torch.nn.Parameter(
        torch.empty(4 * MB, dtype=torch.uint8, device="cuda"), requires_grad=False
    )
    torch.cuda.synchronize()
    activation = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    try:
        after = _claimed()
        assert after == 4 * MB, f"expected only the new weight, got {after / MB} MiB"
    finally:
        del activation, module


def test_frozen_label_does_not_follow_a_reused_address(window):
    if not window:
        pytest.skip("tagging unsupported here")
    with tag("SAMPLER"):
        freed = torch.empty(64 * MB, dtype=torch.uint8, device="cuda")
    address = freed.untyped_storage().data_ptr()
    frozen = freeze_tag_map()
    assert live_bytes_by_tag(frozen).get("SAMPLER", 0) >= 64 * MB, "labelled while it lives"
    del freed
    torch.cuda.synchronize()
    successor = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    try:
        if successor.untyped_storage().data_ptr() != address:
            pytest.skip("the allocator placed the new block elsewhere")
        totals = live_bytes_by_tag(frozen)
        assert totals.get("SAMPLER", 0) == 0, "a block at a freed owner's address is not its owner"
        assert totals[UNTAGGED] >= 32 * MB
    finally:
        del successor


def test_a_recording_someone_else_started_is_left_alone(monkeypatch):
    from tensorrt_llm._torch.memory.monitor.tagging import history_recording_active

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    if not tagging_supported():
        pytest.skip("tagging unsupported here")
    torch.cuda.memory._record_memory_history(context=None, max_entries=10_000)
    try:
        theirs = torch.empty(MB, dtype=torch.uint8, device="cuda")
        assert start_tagging_window() is False, "must not replace another recording"
        assert history_recording_active(), "their recording is still on"
        traces = torch.cuda.memory._snapshot()["device_traces"]
        address = theirs.untyped_storage().data_ptr()
        assert any(e.get("addr") == address for t in traces for e in t), "their trace is intact"
        del theirs
    finally:
        torch.cuda.memory._record_memory_history(None)


def test_nothing_happens_when_off(monkeypatch):
    """Loading a model with the monitor off must cost nothing."""
    from tensorrt_llm._torch.memory.monitor import tagging

    monkeypatch.delenv("TLLM_MEM_MONITOR", raising=False)
    # Claims are recorded even when off, so an on-demand report names the
    # weights; but they must cost nothing and must not keep the model alive.
    tagging.clear_claims()
    snapshots = []
    monkeypatch.setattr(
        torch.cuda.memory, "_snapshot", lambda *a, **k: snapshots.append(1) or {"segments": []}
    )
    model = torch.nn.Linear(4, 4)
    assert tagging.claim_module_memory(model, "WEIGHTS") is True
    assert snapshots == [], "a disabled monitor must not take an allocator snapshot"
    assert tagging.claimed_modules() == [model]
    del model
    assert tagging.claimed_modules() == []
    tagging.clear_claims()

    calls = []
    monkeypatch.setattr(
        torch.cuda.memory, "_set_memory_metadata", lambda *a: calls.append(a), raising=False
    )
    with tagging.tag("SAMPLER"):
        pass
    assert calls == [], "a tag must not touch allocator metadata"


def test_capture_builds_tag_entries(window, monkeypatch):
    """``capture()`` with tags present, which is the shipped path.

    Earlier tests all passed ``entries`` explicitly and so never ran
    ``_tag_entries``. That left a reference to a non-existent enum member
    unexercised until a real serving run hit it, so this goes through
    ``capture()`` itself.
    """
    if not window:
        pytest.skip("tagging unsupported here")
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")

    from tensorrt_llm._torch.memory.monitor import capture, render_capture
    from tensorrt_llm._torch.memory.monitor.capture import _tag_entries
    from tensorrt_llm._torch.memory.monitor.schema import Placement
    from tensorrt_llm._torch.memory.monitor.tagging import set_frozen_map

    with tag("WEIGHTS"):
        weights = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    try:
        set_frozen_map(freeze_tag_map())
        result = capture(tag="test", rank=0)
        assert result is not None
        # Rendering exercises every field each entry declares.
        assert "weights" in render_capture(result)
        entries = _tag_entries()
        assert entries, "a tagged allocation must produce an entry"
        for entry in entries:
            assert isinstance(entry.placement, Placement)
    finally:
        set_frozen_map(TagSnapshot(supported=False))
        del weights


def test_startup_window(monkeypatch):
    from tensorrt_llm._torch.memory.monitor import tagging, triggers

    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")

    # On cudaMallocAsync tags cannot work, but the KV cache and every
    # other component still belong in the startup breakdown.
    with monkeypatch.context() as m:
        m.setattr(tagging, "tagging_supported", lambda: False)
        reports = []
        m.setattr(triggers, "report_now", lambda **kw: reports.append(kw) or "r")
        with tagging.startup_window() as recording:
            assert recording is False
        assert len(reports) == 1 and reports[0]["tag"] == "startup_complete"

    # Stacks make each snapshot cost seconds; the tag survives without them.
    with monkeypatch.context() as m:
        calls = []
        m.setattr(tagging, "tagging_supported", lambda: True)
        m.setattr(torch.cuda.memory, "_record_memory_history", lambda *a, **k: calls.append((a, k)))
        assert tagging.start_tagging_window()
        assert calls[0][1].get("context", "all") is None, "recorded without stacks"

    # Recording stops before the report is built.
    with monkeypatch.context() as m:
        order = []
        m.setattr(tagging, "start_tagging_window", lambda: True)
        m.setattr(tagging, "freeze_tag_map", lambda: TagSnapshot(supported=True))
        m.setattr(tagging, "stop_tagging_window", lambda: order.append("stop"))
        m.setattr(tagging, "live_bytes_by_tag", lambda s: order.append("read") or {})
        m.setattr(triggers, "report_now", lambda **kw: order.append("report"))
        with tagging.startup_window():
            pass
        assert order.index("stop") < order.index("read") < order.index("report")
