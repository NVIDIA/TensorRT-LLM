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
"""CPU tests for the symmetric-memory pool (no GPU): the layout arithmetic against a pure-Python
reference, the rank-agreed probe declining a gloo group without CUDA, ``reserve`` raising on an
unsupported group, and the pool's input checks (the two-rank case rendezvous through a file, no
port). The GPU protocol (rendezvous, peer views, signals) is covered by the token-sharded TP
multi-GPU tests that use the pool.
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.distributed import symm_mem_pool
from tensorrt_llm._torch.distributed.symm_mem_pool import (
    DEFAULT_ALIGN,
    DEFAULT_TIMEOUT_MS,
    PoolLayout,
    SymmMemPool,
    probe_symmetric_memory,
)

pytestmark = pytest.mark.cpu_only


# =============================================================================
# Layout arithmetic
# =============================================================================


def _ref_region_nbytes(slot_nbytes: int, world_size: int, align: int) -> int:
    """Pure-Python reference: ceil(slot / world) rounded up to ``align``."""
    per_rank = (slot_nbytes + world_size - 1) // world_size
    return ((per_rank + align - 1) // align) * align


@pytest.mark.parametrize("world_size", [2, 3, 4, 8])
@pytest.mark.parametrize("slots", [1, 2, 3])
@pytest.mark.parametrize(
    "slot_nbytes",
    [1, 255, 256, 257, 4096, 5 * 256 + 3, 151_200 * 5120 + 151_200 * 10 * 4],
)
def test_pool_layout_matches_reference(world_size, slots, slot_nbytes):
    lay = PoolLayout.for_request(slot_nbytes, slots, world_size)
    region = _ref_region_nbytes(slot_nbytes, world_size, DEFAULT_ALIGN)
    assert lay.align == DEFAULT_ALIGN
    assert lay.region_nbytes == region
    assert lay.region_nbytes % DEFAULT_ALIGN == 0
    assert lay.slot_nbytes == world_size * region
    assert lay.slot_nbytes >= slot_nbytes
    assert lay.slot_nbytes - slot_nbytes < world_size * DEFAULT_ALIGN  # smallest such layout
    assert lay.nbytes == slots * lay.slot_nbytes
    # Regions tile the allocation in (slot, rank) order without gaps or overlaps, all aligned.
    starts = [lay.region_start(s, r) for s in range(slots) for r in range(world_size)]
    assert starts == [i * region for i in range(slots * world_size)]
    assert all(start % DEFAULT_ALIGN == 0 for start in starts)
    # A span inside the last region of the last slot ends inside the allocation.
    start, stop = lay.region_span(slots - 1, world_size - 1, 16, 8)
    assert (start, stop) == (lay.nbytes - region + 16, lay.nbytes - region + 24)
    assert lay.region_span(0, 0, 0, region) == (0, region)


def test_pool_layout_custom_align():
    lay = PoolLayout.for_request(1000, 2, 3, align=512)
    assert lay.region_nbytes == 512 and lay.slot_nbytes == 1536 and lay.nbytes == 3072
    assert lay.region_start(1, 2) == 1536 + 1024


def test_pool_layout_rejects_out_of_range():
    lay = PoolLayout(4, 2, 512)
    with pytest.raises(ValueError, match="slot 2 out of range"):
        lay.region_span(2, 0, 0, 1)
    with pytest.raises(ValueError, match="source rank 4 out of range"):
        lay.region_span(0, 4, 0, 1)
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, 512, 1)  # starts past the region
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, 500, 13)  # ends past the region
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, 0, 0)  # empty
    with pytest.raises(ValueError, match="does not fit"):
        lay.region_span(0, 0, -1, 1)  # negative offset
    with pytest.raises(ValueError, match="positive multiple of align"):
        PoolLayout(4, 2, 500)
    with pytest.raises(ValueError, match="slots >= 1"):
        PoolLayout(4, 0, 512)
    with pytest.raises(ValueError, match="world_size >= 1"):
        PoolLayout(0, 1, 512)
    with pytest.raises(ValueError, match="power of two"):
        PoolLayout(4, 1, 512, align=100)
    with pytest.raises(ValueError, match="slot_nbytes >= 1"):
        PoolLayout.for_request(0, 1, 4)


def test_pool_layout_grow_only_merge():
    a = PoolLayout(4, 2, 1024)
    b = PoolLayout(4, 3, 512)
    assert a.covers(a) and not a.covers(b) and not b.covers(a)
    merged = a.merged(b)
    assert merged == PoolLayout(4, 3, 1024)
    assert merged.covers(a) and merged.covers(b)
    assert a.merged(PoolLayout(4, 1, 256)) == a  # a smaller request never shrinks the pool
    with pytest.raises(ValueError, match="different pools"):
        a.merged(PoolLayout(2, 2, 1024))
    assert not a.covers(PoolLayout(4, 2, 1024, align=512))


# =============================================================================
# Probe and reserve on a gloo group (one rank, in-process)
# =============================================================================


@pytest.fixture
def gloo_world():
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def test_probe_declines_gloo_without_cuda(gloo_world):
    ok, reason = probe_symmetric_memory(gloo_world, torch.device("cpu"))
    assert ok is False
    assert reason.startswith("rank 0: ")
    assert "needs >= 2 ranks" in reason
    assert "device cpu is not a CUDA device" in reason
    assert "ProcessGroupGloo" in reason and "ProcessGroupNCCL" in reason
    # The same verdict with a string device and an explicit channel need.
    assert probe_symmetric_memory(gloo_world, "cpu", min_channels=4)[0] is False
    with pytest.raises(ValueError, match="process group"):
        probe_symmetric_memory(None, "cpu")


def test_reserve_raises_on_unsupported_group(gloo_world):
    pool = SymmMemPool(gloo_world, torch.device("cpu"), timeout_ms=1000)
    assert pool.rank == 0 and pool.world_size == 1 and pool.timeout_ms == 1000
    assert not pool.reserved
    assert pool.layout is None
    assert (pool.nbytes, pool.slots, pool.slot_nbytes, pool.region_nbytes) == (0, 0, 0, 0)
    assert pool.channels == 0 and pool.signal_channels == 0 and pool.offset == 0
    assert "unreserved" in repr(pool)
    with pytest.raises(RuntimeError, match="unsupported group / device.*ProcessGroupNCCL"):
        pool.reserve(4096, slots=2, min_channels=4)
    assert not pool.reserved and pool.generation == 0
    # Nothing works before a reservation, and every message says so.
    for call in (
        lambda: pool.region(0, 0, 0, 1),
        lambda: pool.peer_region(0, 0, 0, 1),
        lambda: pool.signal(0, 0),
        lambda: pool.wait(0, 0),
        pool.barrier,
    ):
        with pytest.raises(RuntimeError, match=r"reserve\(\) first"):
            call()
    pool.release()  # no-op without an allocation
    pool.release()  # idempotent
    assert not pool.reserved


def test_reserve_rejects_bad_requests_before_touching_the_group(gloo_world):
    pool = SymmMemPool(gloo_world, "cpu")
    assert pool.timeout_ms == DEFAULT_TIMEOUT_MS and pool.align == DEFAULT_ALIGN
    with pytest.raises(ValueError, match="slot_nbytes >= 1"):
        pool.reserve(0)
    with pytest.raises(ValueError, match="slots >= 1"):
        pool.reserve(1024, slots=0)
    with pytest.raises(ValueError, match="min_channels >= 1"):
        pool.reserve(1024, min_channels=0)


def test_constructor_validation(gloo_world):
    with pytest.raises(ValueError, match="got None"):
        SymmMemPool(None, "cpu")
    with pytest.raises(ValueError, match="timeout_ms >= 0"):
        SymmMemPool(gloo_world, "cpu", timeout_ms=-1)
    with pytest.raises(ValueError, match="power of two"):
        SymmMemPool(gloo_world, "cpu", align=100)
    pool = SymmMemPool(gloo_world, "cpu", align=512)
    assert pool.device == torch.device("cpu") and pool.align == 512
    assert pool.group_name == gloo_world.group_name


def test_local_preconditions_name_every_failure(gloo_world):
    reasons = symm_mem_pool._local_preconditions(gloo_world, torch.device("cpu"))
    assert len(reasons) == 3, reasons
    assert any("needs >= 2 ranks" in r for r in reasons)
    assert any("not a CUDA device" in r for r in reasons)
    assert any("ProcessGroupNCCL" in r for r in reasons)


# =============================================================================
# Rank agreement across two gloo ranks (spawned processes, CPU)
# =============================================================================


def _agreement_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        group = dist.group.WORLD
        # The probe declines on every rank, and every rank's reason names every rank.
        ok, reason = probe_symmetric_memory(group, torch.device("cpu"))
        assert ok is False
        assert all(f"rank {r}: " in reason for r in range(world_size)), reason
        assert "ProcessGroupNCCL" in reason and "needs >= 2 ranks" not in reason
        # MIN semantics: one dissenting rank fails everyone, and the reason names it.
        dissenter = world_size - 1
        ok, reason = symm_mem_pool._agree(
            group, rank != dissenter, "" if rank != dissenter else "dissent"
        )
        assert (ok, reason) == (False, f"rank {dissenter}: dissent")
        assert symm_mem_pool._agree(group, True, "") == (True, "")
    finally:
        dist.destroy_process_group()


def test_probe_and_agreement_two_gloo_ranks(tmp_path):
    init_file = str(tmp_path / "rendezvous")
    mp.spawn(_agreement_worker, args=(2, init_file), nprocs=2, join=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
