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

from tensorrt_llm._torch.memory.monitor import pools as pool_mod
from tensorrt_llm._torch.memory.monitor import tagging
from tensorrt_llm._torch.memory.monitor.summary import Group, group_totals

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

MB = 1 << 20


@pytest.fixture(autouse=True)
def _monitor_on(monkeypatch):
    monkeypatch.setenv("TLLM_MEM_MONITOR", "1")
    yield
    tagging.set_frozen_map(tagging.TagSnapshot(supported=False))


def _capture_graph(size):
    """Capture a graph that allocates ``size`` bytes, and free its output."""
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    warm = torch.ones(1024, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm * 2
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph, pool=pool):
        kept = torch.empty(size, dtype=torch.uint8, device="cuda")
        scratch = torch.empty(size, dtype=torch.uint8, device="cuda")
        del scratch  # freed inside capture: held by the pool, not reusable
    return graph, pool, kept


def _capture():
    from tensorrt_llm._torch.memory.monitor import capture

    return capture(tag="unit")


class TestClassification:
    def test_default_graph_and_registered_mempool(self):
        pool_mod.clear_registered_pools()
        mem = torch.cuda.MemPool()
        pool_mod.register_pool(mem.id, "test pool")
        assert pool_mod.classify((0, 0))[0] == "default"
        assert pool_mod.classify(mem.id) == ("mempool", "test pool")
        assert pool_mod.classify(torch.cuda.graph_pool_handle())[0] == "graph"

    def test_per_pool_reserved_adds_up_to_memory_reserved(self):
        graph, pool, kept = _capture_graph(32 * MB)
        usage = pool_mod.pool_usage(torch.cuda.memory._snapshot()["segments"])
        assert sum(p.reserved for p in usage) == torch.cuda.memory_reserved()
        del graph, kept

    def test_sleep_mode_pool_registers_itself(self):
        from tensorrt_llm._torch.virtual_memory import RestoreMode
        from tensorrt_llm._torch.virtual_memory import scope as vm_scope

        pool_mod.clear_registered_pools()
        with vm_scope("unit_scope", RestoreMode.NONE) as proxy:
            held = torch.empty(16 * MB, dtype=torch.uint8, device="cuda")
        kinds = {
            pool_mod.classify(seg["segment_pool_id"])
            for seg in torch.cuda.memory._snapshot()["segments"]
            if seg["allocated_size"]
        }
        # Without registration the sleep pool would be reported as a graph.
        assert ("mempool", "sleep: unit_scope") in kinds
        del held, proxy

    def test_resumed_parent_pool_keeps_the_parent_tag(self):
        from tensorrt_llm._torch.virtual_memory import RestoreMode
        from tensorrt_llm._torch.virtual_memory import scope as vm_scope

        pool_mod.clear_registered_pools()
        with vm_scope("unit_outer", RestoreMode.NONE) as proxy:
            with vm_scope("unit_inner", RestoreMode.NONE):
                inner = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")
            after = torch.empty(8 * MB, dtype=torch.uint8, device="cuda")
        labels = {
            pool_mod.classify(seg["segment_pool_id"])[1]
            for seg in torch.cuda.memory._snapshot()["segments"]
            if seg["allocated_size"]
        }
        # Memory allocated after the inner scope still belongs to the outer
        # tag and is released with it.
        assert "sleep: unit_outer" in labels
        assert "sleep: resumed parent scope" not in labels
        del inner, after, proxy


class TestSleepRelease:
    """Released sleep-mode memory: torch keeps counting it, the GPU does not."""

    def _scoped(self, tag, size):
        from tensorrt_llm._torch.virtual_memory import RestoreMode
        from tensorrt_llm._torch.virtual_memory import scope as vm_scope

        with vm_scope(tag, RestoreMode.NONE) as proxy:
            held = torch.empty(size, dtype=torch.uint8, device="cuda")
        return proxy, held

    def test_released_memory_leaves_the_footprint(self):
        from tensorrt_llm._torch.virtual_memory import materialize_with_tag, release_with_tag

        pool_mod.clear_registered_pools()
        proxy, held = self._scoped("unit_sleep", 256 * MB)
        try:
            torch.cuda.synchronize()
            awake = _capture().frame
            release_with_tag("unit_sleep")
            torch.cuda.synchronize()
            asleep = _capture().frame
            # torch's own counter does not move ...
            assert asleep.torch.reserved == awake.torch.reserved
            # ... but the footprint does, so the residual stays put instead
            # of dropping by the released size.
            assert asleep.torch.released >= 256 * MB
            assert abs(asleep.nontorch_residual - awake.nontorch_residual) < 64 * MB
            totals = group_totals(asleep)
            assert totals.sizes[Group.SLEEP_RELEASED] == -asleep.torch.released
            assert sum(s for s in totals.sizes.values() if s is not None) == totals.process_total
            materialize_with_tag("unit_sleep")
            torch.cuda.synchronize()
            assert _capture().frame.torch.released == 0
        finally:
            del held, proxy


class TestGraphMemory:
    def test_graph_blocks_are_cuda_graphs_even_inside_a_tagged_step(self):
        # Capture runs inside an executor creation step, whose tag would
        # otherwise claim the graph's memory as that step's.
        if not tagging.start_tagging_window():
            pytest.skip("tagging unsupported here")
        try:
            with tagging.tag("executor_extra"):
                graph, pool, kept = _capture_graph(32 * MB)
            frozen = tagging.freeze_tag_map()
        finally:
            tagging.stop_tagging_window()
        by_tag = tagging.live_bytes_by_tag(frozen)
        assert by_tag.get(tagging.CUDA_GRAPHS, 0) >= 32 * MB
        del graph, kept

    def test_graph_free_space_is_not_reusable_cache(self):
        graph, pool, kept = _capture_graph(64 * MB)
        totals = group_totals(_capture().frame)
        graph_pool = next(
            p
            for p in pool_mod.pool_usage(torch.cuda.memory._snapshot()["segments"])
            if p.kind == "graph"
        )
        assert graph_pool.free >= 64 * MB, "the freed scratch stays in the graph pool"
        # Counting it as reusable overstated headroom by every captured graph.
        assert totals.sizes[Group.CUDA_GRAPHS] >= graph_pool.reserved
        assert totals.sizes.get(Group.TORCH_CACHE, 0) < graph_pool.free
        del graph, kept

    def test_groups_still_add_up(self):
        graph, pool, kept = _capture_graph(32 * MB)
        totals = group_totals(_capture().frame)
        assert sum(s for s in totals.sizes.values() if s is not None) == totals.process_total
        del graph, kept


def test_graph_rows_use_a_readable_owner():
    from tensorrt_llm._torch.memory.monitor import _tag_entries
    from tensorrt_llm._torch.memory.monitor.schema import Category

    graph, pool, kept = _capture_graph(16 * MB)
    try:
        rows = [e for e in _tag_entries() if e.category is Category.CUDA_GRAPHS]
        assert rows and rows[0].owner == "in use by graphs"
    finally:
        del graph, kept


class TestUnregisteredMemPool:
    """A MemPool created by code that never registers, e.g. GMS weight sharing."""

    def test_is_a_mempool_not_a_graph_pool(self):
        pool_mod.clear_registered_pools()
        mem = torch.cuda.MemPool()  # deliberately not registered
        with torch.cuda.use_mem_pool(mem):
            held = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        try:
            usage = {
                p.pool_id: p for p in pool_mod.pool_usage(torch.cuda.memory._snapshot()["segments"])
            }
            assert usage[tuple(mem.id)].kind == "mempool"
            assert usage[tuple(mem.id)].label == "unnamed MemPool"
        finally:
            del held

    def test_its_blocks_are_not_counted_as_cuda_graphs(self):
        pool_mod.clear_registered_pools()
        mem = torch.cuda.MemPool()
        with torch.cuda.use_mem_pool(mem):
            held = torch.empty(32 * MB, dtype=torch.uint8, device="cuda")
        try:
            by_tag = tagging.live_bytes_by_tag(tagging.TagSnapshot(supported=True))
            assert by_tag.get(tagging.CUDA_GRAPHS, 0) == 0
        finally:
            del held

    def test_graph_pools_are_still_graph_pools(self):
        pool_mod.clear_registered_pools()
        mem = torch.cuda.MemPool()
        graph, pool, kept = _capture_graph(16 * MB)
        try:
            kinds = pool_mod.classify_all([tuple(mem.id), tuple(pool)])
            assert kinds[tuple(pool)][0] == "graph"
            assert kinds[tuple(mem.id)][0] == "mempool"
        finally:
            del graph, kept
