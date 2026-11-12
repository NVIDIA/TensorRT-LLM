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
"""Splitting PyTorch's memory by pool.

Free space in the default pool can serve the next allocation; free space in
a CUDA graph's pool is held for replay. These use real graph capture and the
real sleep-mode memory pool, since the classification rests on how torch
numbers and reports those pools.
"""

import pytest
import torch
from memory_monitor_test_utils import MB, capture_graph

from tensorrt_llm._torch.memory.monitor import pools as pool_mod
from tensorrt_llm._torch.memory.monitor import tagging
from tensorrt_llm._torch.memory.monitor.summary import Group, group_totals

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def _monitor_on(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    yield
    tagging.set_frozen_map(tagging.TagSnapshot(supported=False))


def _capture():
    from tensorrt_llm._torch.memory.monitor import capture

    return capture(tag="unit")


def _segments():
    return torch.cuda.memory._snapshot()["segments"]


def test_pools_are_classified_by_kind():
    pool_mod.clear_registered_pools()
    mem = torch.cuda.MemPool()
    pool_mod.register_pool(mem.id, "test pool")
    assert pool_mod.classify((0, 0))[0] == "default"
    assert pool_mod.classify(mem.id) == ("mempool", "test pool")
    assert pool_mod.classify(torch.cuda.graph_pool_handle())[0] == "graph"

    # A MemPool created by code that never registers, e.g. GMS weight sharing.
    pool_mod.clear_registered_pools()
    unregistered = torch.cuda.MemPool()  # deliberately not registered
    with torch.cuda.use_mem_pool(unregistered):
        held = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
    try:
        usage = {p.pool_id: p for p in pool_mod.pool_usage(_segments())}
        assert usage[tuple(unregistered.id)].kind == "mempool", "not a graph pool"
        assert usage[tuple(unregistered.id)].label == "unnamed MemPool"
        by_tag = tagging.live_bytes_by_tag(tagging.TagSnapshot(supported=True))
        assert by_tag.get(tagging.CUDA_GRAPHS, 0) == 0, "its blocks are not CUDA graphs"
    finally:
        del held

    graph, pool, kept = capture_graph(32 * MB, scratch=True)
    try:
        kinds = pool_mod.classify_all([tuple(unregistered.id), tuple(pool)])
        assert kinds[tuple(pool)][0] == "graph", "graph pools are still graph pools"
        assert kinds[tuple(unregistered.id)][0] == "mempool"
        usage = pool_mod.pool_usage(_segments())
        assert sum(p.reserved for p in usage) == torch.cuda.memory_reserved(), (
            "per-pool reserved adds up to memory_reserved"
        )
    finally:
        del graph, kept


def test_sleep_mode_pools_are_named_and_released_memory_leaves_the_footprint():
    from tensorrt_llm._torch.memory.monitor.summary import render_summary
    from tensorrt_llm._torch.virtual_memory import (
        RestoreMode,
        materialize_with_tag,
        release_with_tag,
    )
    from tensorrt_llm._torch.virtual_memory import scope as vm_scope

    def labels():
        return {
            pool_mod.classify(seg["segment_pool_id"])
            for seg in _segments()
            if seg["allocated_size"]
        }

    pool_mod.clear_registered_pools()
    with vm_scope("unit_scope", RestoreMode.NONE) as proxy:
        held = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    # Without registration the sleep pool would be reported as a graph.
    assert ("mempool", "sleep: unit_scope") in labels()
    del held, proxy

    pool_mod.clear_registered_pools()
    with vm_scope("unit_outer", RestoreMode.NONE) as proxy:
        with vm_scope("unit_inner", RestoreMode.NONE):
            inner = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")
        after = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")
    names = {label for _, label in labels()}
    # Memory allocated after the inner scope still belongs to the outer
    # tag and is released with it.
    assert "sleep: unit_outer" in names, "a resumed parent pool keeps the parent tag"
    assert "sleep: resumed parent scope" not in names
    del inner, after, proxy

    # Released sleep-mode memory: torch keeps counting it, the GPU does not.
    pool_mod.clear_registered_pools()
    with vm_scope("unit_sleep", RestoreMode.NONE) as proxy:
        held = torch.empty(256 * MB, dtype=torch.uint8, device="cuda")
    # Claimed like model weights: asleep, they must not be counted.
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(held, requires_grad=False)
    tagging.claim_module_memory(module, "WEIGHTS")
    try:
        torch.cuda.synchronize()
        awake = _capture().frame
        release_with_tag("unit_sleep")
        torch.cuda.synchronize()
        asleep = _capture().frame
        # torch's own counter does not move ...
        assert torch.cuda.memory_reserved() >= awake.torch.reserved
        # ... but the reading leaves the released pool out, so the
        # residual stays put instead of dropping by the released size.
        assert sum(p.reserved for p in asleep.torch.pools if p.released) >= 256 * MB
        assert asleep.torch.reserved <= awake.torch.reserved - 256 * MB
        assert abs(asleep.nontorch_residual - awake.nontorch_residual) < 64 * MB
        assert group_totals(awake).sizes.get(Group.WEIGHTS, 0) >= 256 * MB
        totals = group_totals(asleep)
        assert totals.sizes.get(Group.WEIGHTS, 0) < 256 * MB
        # Rows show what is resident: none of them is negative, and they
        # still add up to the process.
        assert all(
            s >= 0 for g, s in totals.sizes.items() if s is not None and g is not Group.OVERHEAD
        )
        assert sum(s for s in totals.sizes.values() if s is not None) == totals.process_total
        note = "\n".join(render_summary(asleep, totals))
        assert "Released by sleep mode" in note and "Unit sleep 0.25 GiB" in note
        materialize_with_tag("unit_sleep")
        torch.cuda.synchronize()
        assert not any(p.released for p in _capture().frame.torch.pools)
    finally:
        tagging.clear_claims()
        del module, held, proxy


def test_graph_memory_is_cuda_graphs_not_reusable_cache():
    from tensorrt_llm._torch.memory.monitor.capture import _tag_entries
    from tensorrt_llm._torch.memory.monitor.schema import Category

    # Capture runs inside an executor creation step, whose tag would
    # otherwise claim the graph's memory as that step's.
    if not tagging.start_tagging_window():
        pytest.skip("tagging unsupported here")
    try:
        with tagging.tag("executor_extra"):
            graph, pool, kept = capture_graph(32 * MB, scratch=True)
        frozen = tagging.freeze_tag_map()
    finally:
        tagging.stop_tagging_window()
    by_tag = tagging.live_bytes_by_tag(frozen)
    assert by_tag.get(tagging.CUDA_GRAPHS, 0) >= 32 * MB, "graph blocks inside a tagged step"
    del graph, kept

    graph, pool, kept = capture_graph(64 * MB, scratch=True)
    try:
        totals = group_totals(_capture().frame)
        graph_pool = max(
            (p for p in pool_mod.pool_usage(_segments()) if p.kind == "graph"),
            key=lambda p: p.free,
        )
        assert graph_pool.free >= 64 * MB, "the freed scratch stays in the graph pool"
        # Counting it as reusable overstated headroom by every captured graph.
        assert totals.sizes[Group.CUDA_GRAPHS] >= graph_pool.reserved
        assert totals.sizes.get(Group.TORCH_CACHE, 0) < graph_pool.free
        assert sum(s for s in totals.sizes.values() if s is not None) == totals.process_total, (
            "groups still add up with a graph alive"
        )
        rows = [
            e
            for e in _tag_entries(tagging.live_bytes_by_tag(tagging.get_frozen_map()))
            if e.category is Category.CUDA_GRAPHS
        ]
        assert rows and rows[0].owner == "in use by graphs", "graph rows use a readable owner"
    finally:
        del graph, kept


def test_pools_are_classified_ahead_of_reports_and_the_walk_cannot_break_them(monkeypatch):
    import weakref

    class _Gone:
        pass

    class _Hostile:
        def __getattribute__(self, name):
            raise RuntimeError("touched")

    pool_mod.clear_registered_pools()
    gone = _Gone()
    # Alive during the walk: a dead proxy raises on isinstance, a hostile
    # object on any attribute access.
    odd = [weakref.proxy(gone), _Hostile()]
    del gone
    unregistered = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(unregistered):
        held = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
    graph, pool, kept = capture_graph(16 * MB, scratch=True)
    later = torch.cuda.MemPool()
    try:
        pool_mod.prime()

        def walked():
            raise AssertionError("pools seen at startup must not be walked for again")

        with monkeypatch.context() as patch:
            patch.setattr(pool_mod, "_live_mempool_ids", walked)
            usage = {p.pool_id: p.kind for p in pool_mod.pool_usage(_segments())}
        assert usage[tuple(unregistered.id)] == "mempool", "primed, without a walk"
        assert usage[tuple(pool)] == "graph"

        # A pool created later is walked for; if the walk fails, it is
        # unclassified for this report and retried on the next one.
        with torch.cuda.use_mem_pool(later):
            late = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")

        def failing():
            raise RuntimeError("walk failed")

        with monkeypatch.context() as patch:
            patch.setattr(pool_mod, "_live_mempool_ids", failing)
            usage = {p.pool_id: p.kind for p in pool_mod.pool_usage(_segments())}
        assert usage[tuple(later.id)] == "unclassified"
        assert usage[tuple(pool)] == "graph", "pools already classified keep their kind"
        usage = {p.pool_id: p.kind for p in pool_mod.pool_usage(_segments())}
        assert usage[tuple(later.id)] == "mempool", "retried once the walk works"
        assert odd  # kept alive through every walk above
        del late
    finally:
        del held, graph, kept


def test_one_report_reads_one_sleep_release_state(monkeypatch):
    """Sleep or wakeup on another thread must not change the state mid-report."""
    import threading

    from tensorrt_llm._torch.memory.monitor import providers

    seen = []

    class _Probe:
        name = "probe"
        category = None

        def available(self):
            return True

        def collect(self):
            seen.append(pool_mod.is_tag_released("kv_cache"))
            flip = threading.Thread(target=pool_mod.mark_released, args=(("kv_cache",),))
            flip.start()
            flip.join()
            seen.append(pool_mod.is_tag_released("kv_cache"))
            return []

    pool_mod.clear_registered_pools()
    providers.clear_registered_providers()
    try:
        pool_mod.register_pool((7, 4), "sleep: kv_cache", sleep_tag="kv_cache")
        providers.register_provider(_Probe())
        assert _capture() is not None
        assert seen == [False, False], "the release state changed during one report"
        assert pool_mod.is_tag_released("kv_cache"), "the next report sees the release"
    finally:
        providers.clear_registered_providers()
        pool_mod.clear_registered_pools()
