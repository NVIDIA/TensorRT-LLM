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
"""Multi-rank TP tests for the KVCacheManagerV2 rebalance trigger.

``tests/unittest/_torch/executor/test_kv_pool_rebalance.py`` covers the same
logic against a mocked ``dist``; these tests run it across real MPI ranks with
a real ``MPIDist``, which is the only way to show the collective actually
agrees (and that ranks reach it in lockstep rather than deadlocking).

What is being protected
-----------------------
Every input to ``need_adjustment`` is a deterministic function of the request
stream -- the sample counters and the moving averages behind the target ratios
are all request-derived, with no randomness -- **except** the 120s cooldown,
which compares against a per-rank ``steady_clock`` reading
(``kvCacheManager.cpp:855``).  Two TP ranks can therefore straddle that boundary
on different iterations and rebalance one iteration apart.  TP ranks compute
``_schedule()`` independently with no broadcast, so for that one iteration they
could admit different requests and issue mismatched collectives.

``_agreed_need_adjustment`` closes this by letting TP rank 0 decide and
broadcasting.  The tests below inject exactly the skew the mechanism exists to
absorb: ranks whose *local* readings disagree.

These tests need no model weights and no KV cache -- they exercise the
agreement and throttle logic directly.
"""

import pickle
import sys
import traceback
from unittest.mock import MagicMock

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
from tensorrt_llm._torch.distributed.communicator import MPIDist
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# MPIPoolExecutor leaks a worker thread on first use; keep CI green.
pytestmark = pytest.mark.threadleak(enabled=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_single_rank(tensor_parallel_size, single_rank_forward_func, *args):
    """Wrapper used by MPIPoolExecutor; matches test_allgather.py."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        single_rank_forward_func(tensor_parallel_size, rank, *args)
    except Exception:
        traceback.print_exc()
        raise
    return True


def _make_executor(
    dist,
    *,
    need_adjustment: bool,
    enable_attention_dp: bool = False,
    rebalance_check_interval: int = 1,
):
    """A PyExecutor stand-in carrying a *real* MPIDist.

    Only the attributes the rebalance gate and the agreement helper read are
    populated; everything else stays a MagicMock so an accidental new read
    shows up as a test failure rather than silently passing.
    """
    exe = MagicMock(spec=PyExecutor)
    exe.dist = dist
    exe.enable_attention_dp = enable_attention_dp
    exe.enable_kv_pool_rebalance = True
    exe.kv_cache_transceiver = None
    exe.is_warmup = False
    exe.is_shutdown = False
    exe.drafter = None
    exe.kv_cache_manager = MagicMock()
    exe.kv_cache_manager.max_beam_width = 1
    exe.kv_cache_manager.impl = MagicMock()
    exe.kv_cache_manager.impl.need_adjustment = need_adjustment
    exe._rebalance_check_interval = rebalance_check_interval
    exe._rebalance_check_counter = 0
    return exe


def _tp_dist(world_size: int, rank: int) -> MPIDist:
    return MPIDist(Mapping(world_size=world_size, rank=rank, tp_size=world_size))


# ---------------------------------------------------------------------------
# Per-rank work
# ---------------------------------------------------------------------------


def run_agreement(world_size, rank, local_flags, expected):
    """Every TP rank must end up with rank 0's decision, not its own."""
    dist = _tp_dist(world_size, rank)
    exe = _make_executor(dist, need_adjustment=local_flags[rank])

    got = PyExecutor._agreed_need_adjustment(exe)

    assert got == expected, (
        f"rank {rank}: local reading was {local_flags[rank]}, rank 0 said "
        f"{local_flags[0]}, so the agreed decision should be {expected}, got {got}"
    )

    # And every rank must have reached the same answer, which is the property
    # that actually keeps the schedulers from diverging.
    all_decisions = dist.allgather(got)
    assert len(set(all_decisions)) == 1, f"ranks disagreed after the collective: {all_decisions}"


def run_attention_dp_independence(world_size, rank, local_flags):
    """Under attention DP each rank keeps its own decision.

    ADP ranks own independent request streams and independent KV caches, so
    forcing rank 0's decision on them would starve a rank that needs to
    rebalance when rank 0 does not.
    """
    dist = _tp_dist(world_size, rank)
    exe = _make_executor(dist, need_adjustment=local_flags[rank], enable_attention_dp=True)

    got = PyExecutor._agreed_need_adjustment(exe)

    assert got == local_flags[rank], (
        f"rank {rank}: attention DP must decide locally; expected {local_flags[rank]}, got {got}"
    )


def run_throttle_lockstep(world_size, rank, interval, iterations):
    """Ranks must reach the agreement collective on the *same* iterations.

    This is the deadlock guard.  ``_can_pause_for_rebalance`` gates the
    collective, so ranks whose throttle counters drifted apart would enter
    ``tp_broadcast`` on different iterations.  Driving the real collective
    inside the loop means such a divergence hangs here rather than passing
    silently; the allgather afterwards pins down that the firing iterations
    were in fact identical.

    Ranks are deliberately given *different* local ``need_adjustment`` readings,
    so it is the throttle cadence -- not agreement on the value -- that is under
    test here.  Everything ``_can_pause_for_rebalance`` reads is rank-uniform,
    which is precisely the property being confirmed.
    """
    dist = _tp_dist(world_size, rank)
    exe = _make_executor(dist, need_adjustment=(rank % 2 == 0), rebalance_check_interval=interval)

    fired_on = []
    for i in range(iterations):
        if PyExecutor._can_pause_for_rebalance(exe):
            fired_on.append(i)
            # Real collective -- diverging ranks hang instead of passing.
            PyExecutor._agreed_need_adjustment(exe)

    all_fired = dist.allgather(fired_on)
    assert all(f == all_fired[0] for f in all_fired), (
        f"ranks fired the rebalance check on different iterations: {all_fired}"
    )

    # Sanity: the throttle actually throttled, and it fired when expected.
    expected = [i for i in range(iterations) if (i + 1) % interval == 0]
    assert fired_on == expected, (
        f"throttle fired on {fired_on}, expected {expected} for interval "
        f"{interval} over {iterations} iterations"
    )
    assert fired_on, "throttle never fired; the test would be vacuous"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _skip_if_not_enough_gpus(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.cuda.device_count()}")


def _flags_for(case: str, world_size: int):
    """Local per-rank ``need_adjustment`` readings for each skew scenario."""
    if case == "all_true":
        return [True] * world_size
    if case == "all_false":
        return [False] * world_size
    if case == "only_rank0_true":
        # Rank 0 wants to rebalance, nobody else does -> all must follow it.
        return [i == 0 for i in range(world_size)]
    if case == "only_rank0_false":
        # The dangerous one: every follower's clock says "rebalance now" but
        # rank 0's does not.  Without agreement the followers would resize and
        # rank 0 would not.
        return [i != 0 for i in range(world_size)]
    raise ValueError(case)


@pytest.mark.parametrize(
    "case",
    ["all_true", "all_false", "only_rank0_true", "only_rank0_false"],
)
@pytest.mark.parametrize("world_size", [2, 4], ids=lambda x: f"tp:{x}")
def test_tp_ranks_agree_on_rebalance_trigger(world_size, case):
    """Rank 0's decision wins on every TP rank, whatever the local readings."""
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for(case, world_size)
    expected = flags[0]

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_agreement, flags, expected)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize("world_size", [2, 4], ids=lambda x: f"tp:{x}")
def test_attention_dp_ranks_decide_independently(world_size):
    """Attention DP opts out of the agreement: each rank keeps its own answer."""
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for("only_rank0_false", world_size)

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_attention_dp_independence, flags)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize("interval", [1, 8], ids=lambda x: f"interval:{x}")
@pytest.mark.parametrize("world_size", [2, 4], ids=lambda x: f"tp:{x}")
def test_rebalance_check_stays_in_lockstep_across_ranks(world_size, interval):
    """The throttle must fire on identical iterations on every rank.

    A drift here would put ranks into ``tp_broadcast`` on different iterations,
    which deadlocks -- so this test hanging is itself the failure signal.
    """
    _skip_if_not_enough_gpus(world_size)

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_throttle_lockstep, interval, 40)] * world_size),
        )
        for r in results:
            assert r is True
