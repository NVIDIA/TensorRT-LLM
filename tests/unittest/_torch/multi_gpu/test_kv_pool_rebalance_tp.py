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

``tests/unittest/_torch/executor/kv_cache/test_kv_pool_rebalance.py`` covers the same
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
    exe.iter_counter = 0
    return exe


def _tp_dist(world_size: int, rank: int) -> MPIDist:
    """Build a pure-TP communicator for this rank.

    Args:
        world_size: Total number of ranks, all of them TP ranks.
        rank: This process's global rank.

    Returns:
        An ``MPIDist`` whose mapping has ``tp_size == world_size``.
    """
    return MPIDist(Mapping(world_size=world_size, rank=rank, tp_size=world_size))


def _cp_dist(world_size: int, rank: int) -> MPIDist:
    """Build a pure-CP communicator for this rank.

    ``tp_size`` is 1 here, so ``cp_broadcast`` alone carries the agreement and
    keying the mechanism on ``tp_size`` would miss this configuration entirely.

    Args:
        world_size: Total number of ranks, all of them CP ranks.
        rank: This process's global rank.

    Returns:
        An ``MPIDist`` whose mapping has ``cp_size == world_size`` and
        ``tp_size == 1``.
    """
    return MPIDist(Mapping(world_size=world_size, rank=rank, cp_size=world_size))


def _cp_tp_dist(cp_size: int, tp_size: int, rank: int) -> MPIDist:
    """Build a combined CP x TP communicator for this rank.

    This is the only mapping that makes ``_agreed_need_adjustment`` run *both*
    broadcasts, which is the shape the production chain actually takes.

    Args:
        cp_size: Context-parallel width.
        tp_size: Tensor-parallel width.
        rank: This process's global rank.

    Returns:
        An ``MPIDist`` over ``cp_size * tp_size`` ranks with both dimensions > 1.
    """
    return MPIDist(
        Mapping(
            world_size=cp_size * tp_size,
            rank=rank,
            tp_size=tp_size,
            cp_size=cp_size,
        )
    )


def _pp_dist(world_size: int, rank: int) -> MPIDist:
    """Build a pure-PP communicator for this rank.

    ``tp_size`` and ``cp_size`` are both 1, so ``pp_broadcast`` alone carries
    the agreement.  PP ranks hold different layers and therefore different
    pools, so without this hop they would never agree by construction.

    Args:
        world_size: Total number of ranks, all of them PP ranks.
        rank: This process's global rank.

    Returns:
        An ``MPIDist`` whose mapping has ``pp_size == world_size``.
    """
    return MPIDist(Mapping(world_size=world_size, rank=rank, pp_size=world_size))


def _tp_pp_dist(tp_size: int, pp_size: int, rank: int) -> MPIDist:
    """Build a combined TP x PP communicator for this rank.

    Ranks are laid out ``rank == pp_rank * tp_size + tp_rank``, so with
    ``tp_size == pp_size == 2`` the TP groups are ``[0, 1]`` and ``[2, 3]``
    while the PP groups are ``[0, 2]`` and ``[1, 3]``.

    Args:
        tp_size: Tensor-parallel width.
        pp_size: Pipeline-parallel width.
        rank: This process's global rank.

    Returns:
        An ``MPIDist`` over ``tp_size * pp_size`` ranks with both dimensions > 1.
    """
    return MPIDist(
        Mapping(
            world_size=tp_size * pp_size,
            rank=rank,
            tp_size=tp_size,
            pp_size=pp_size,
        )
    )


# ---------------------------------------------------------------------------
# Per-rank work
# ---------------------------------------------------------------------------


def run_agreement(world_size: int, rank: int, local_flags: list[bool], expected: bool) -> None:
    """Every TP rank must end up with rank 0's decision, not its own.

    Args:
        world_size: Number of TP ranks.
        rank: This process's global rank.
        local_flags: Per-rank local ``need_adjustment`` readings.
        expected: The decision every rank should agree on.
    """
    dist = _tp_dist(world_size, rank)
    exe = _make_executor(dist, need_adjustment=local_flags[rank])

    got = PyExecutor._agreed_need_adjustment(exe)

    # Collect *before* asserting.  Every rank has to reach this collective: if
    # some ranks bailed out on a failed assert first, the ranks that passed
    # would block here until the pytest timeout, turning a clean failure into a
    # five-minute hang.
    all_decisions = dist.allgather(got)

    assert got == expected, (
        f"rank {rank}: local reading was {local_flags[rank]}, rank 0 said "
        f"{local_flags[0]}, so the agreed decision should be {expected}, got {got}"
    )
    # And every rank must have reached the same answer, which is the property
    # that actually keeps the schedulers from diverging.
    assert len(set(all_decisions)) == 1, f"ranks disagreed after the collective: {all_decisions}"


def run_cp_agreement(world_size: int, rank: int, local_flags: list[bool], expected: bool) -> None:
    """Same guarantee as TP, but over a pure-CP mapping (``tp_size == 1``).

    A request is split across CP ranks, so they must admit it together; and CP
    runs on the same executor loops as TP, which compute ``_schedule()`` per
    rank with no broadcast.  Keying the agreement on ``tp_size`` alone would
    leave this configuration unsynchronized.

    Args:
        world_size: Number of CP ranks.
        rank: This process's global rank.
        local_flags: Per-rank local ``need_adjustment`` readings.
        expected: The decision every rank should agree on.
    """
    dist = _cp_dist(world_size, rank)
    assert dist.tp_size == 1, "this test is meaningless unless tp_size is 1"
    exe = _make_executor(dist, need_adjustment=local_flags[rank])

    got = PyExecutor._agreed_need_adjustment(exe)

    # Collect before asserting -- see run_agreement for why the order matters.
    all_decisions = dist.allgather(got)

    assert got == expected, (
        f"CP rank {rank}: local reading was {local_flags[rank]}, rank 0 said "
        f"{local_flags[0]}, so the agreed decision should be {expected}, got {got}"
    )
    assert len(set(all_decisions)) == 1, f"CP ranks disagreed: {all_decisions}"


def run_cp_tp_agreement(
    world_size: int, rank: int, cp_size: int, local_flags: list[bool], expected: bool
) -> None:
    """Global rank 0's decision must reach every rank through *both* broadcasts.

    This is the only case that exercises the production chain end to end:
    ``cp_broadcast`` and then ``tp_broadcast``.  The pure-TP test has
    ``cp_size == 1`` and the pure-CP test has ``tp_size == 1``, so each skips one
    of the two steps; neither can show that the TP step consumes the *result* of
    the CP step rather than the rank's own local reading.

    With ``local_flags`` true only at global rank 0, a rank whose ``tp_rank`` is
    non-zero can only end up ``True`` if its CP root picked up rank 0's decision
    in the first step and then passed it on in the second.

    Args:
        world_size: Total number of ranks, equal to ``cp_size * tp_size``.
        rank: This process's global rank.
        cp_size: Context-parallel width; TP width is derived from it.
        local_flags: Per-rank local ``need_adjustment`` readings.
        expected: The decision every rank should agree on.
    """
    tp_size = world_size // cp_size
    dist = _cp_tp_dist(cp_size, tp_size, rank)
    # Both dimensions must really be >1, or this degenerates into one of the
    # single-dimension tests and stops covering the chain.
    assert dist.cp_size > 1 and dist.tp_size > 1, (
        f"this test only means something when both dimensions are > 1, got "
        f"cp_size={dist.cp_size} tp_size={dist.tp_size}"
    )
    exe = _make_executor(dist, need_adjustment=local_flags[rank])

    got = PyExecutor._agreed_need_adjustment(exe)

    # Collect before asserting -- see run_agreement for why the order matters.
    all_decisions = dist.allgather(got)

    assert got == expected, (
        f"rank {rank} (tp_rank={dist.mapping.tp_rank}, "
        f"cp_rank={dist.mapping.cp_rank}): local reading was {local_flags[rank]}, "
        f"global rank 0 said {local_flags[0]}, so the agreed decision should be "
        f"{expected}, got {got}"
    )
    assert len(set(all_decisions)) == 1, f"CPxTP ranks disagreed: {all_decisions}"


def run_attention_dp_or_coupling(world_size: int, rank: int, local_flags: list[bool]) -> None:
    """Under attention DP the ranks OR their readings rather than decide alone.

    ADP ranks own independent request streams and independent KV caches, so the
    resulting *ratios* are legitimately per-rank -- but the *timing* cannot be.
    The rebalance path is collective under ADP:
    ``_consume_previous_batch_for_rebalance`` calls
    ``_flush_pending_transfer_responses``, which enters ``_enqueue_responses``
    on every DP rank even with an empty response list (its own docstring
    requires that), and that runs a ``tp_gather``.  A rank that rebalanced alone
    would join a collective its peers are not in.  Observed on a live 4-rank
    Qwen3-Next run before this was fixed: the gather paired against the
    ``tp_allgather`` of batch sizes in ``_can_queue`` and the executor loop died
    with ``cannot unpack non-iterable int object``.

    ``any()`` rather than rank 0's broadcast is what preserves the
    anti-starvation property the old suppression was reaching for: a rank that
    needs a rebalance still gets one instead of waiting for rank 0 to want the
    same thing.  A rank that does not need one is not free, though -- it runs
    the whole hook, and only the ``adjust()`` within it is a no-op (0.04 ms
    measured, inside ~0.5 ms of suspend/resume; the preceding drain is
    unmeasured).

    Args:
        world_size: Number of TP ranks.
        rank: This process's global rank.
        local_flags: Per-rank local ``need_adjustment`` readings.
    """
    dist = _tp_dist(world_size, rank)
    exe = _make_executor(dist, need_adjustment=local_flags[rank], enable_attention_dp=True)

    got = PyExecutor._agreed_need_adjustment(exe)

    expected = any(local_flags)
    assert got == expected, (
        f"rank {rank}: attention DP must OR the readings so every rank enters the "
        f"collective rebalance path together; local reading was {local_flags[rank]}, "
        f"the group read {local_flags}, expected {expected}, got {got}"
    )


def run_adp_cp_agreement(
    world_size: int, rank: int, cp_size: int, local_flags: list[bool], expected: list[bool]
) -> None:
    """Under ADP the CP hop broadcasts and the TP hop then OR-reduces.

    Two different mechanisms, chained, and this case pins down both.  CP first:
    inside one DP replica the CP ranks split the *same* request along the
    sequence dimension and must admit it together, so they take their CP root's
    reading.  Then TP: under ADP the replicas must still enter the rebalance
    together, because the path is collective (see
    ``run_attention_dp_or_coupling``), so the post-CP values are OR-ed across
    the TP dimension.

    The observable difference from the old suppressed-TP behaviour is that the
    replicas no longer end on different answers.  That was never safe -- the
    ``tp_gather`` in ``_enqueue_responses`` spans the replicas.

    Args:
        world_size: Total ranks, equal to ``cp_size * tp_size``.
        rank: This process's global rank.
        cp_size: Context-parallel width; TP width is derived from it.
        local_flags: Per-rank local ``need_adjustment`` readings.
        expected: Per-rank agreed decision after the CP broadcast and TP OR.
    """
    tp_size = world_size // cp_size
    dist = _cp_tp_dist(cp_size, tp_size, rank)
    exe = _make_executor(dist, need_adjustment=local_flags[rank], enable_attention_dp=True)

    got = PyExecutor._agreed_need_adjustment(exe)

    # Collect before asserting -- see run_agreement for why the order matters.
    all_decisions = dist.allgather(got)

    assert got == expected[rank], (
        f"rank {rank} (tp_rank={dist.mapping.tp_rank}, "
        f"cp_rank={dist.mapping.cp_rank}): local reading was {local_flags[rank]}, "
        f"its CP root read {expected[rank]}, so the agreed decision should be "
        f"{expected[rank]}, got {got}"
    )
    assert all_decisions == expected, (
        f"ADP+CP decisions were {all_decisions}, expected {expected}: CP ranks "
        "must follow their CP root and the replicas must then OR together; "
        "must follow their replica's root while replicas stay independent"
    )


def run_pp_agreement(world_size: int, rank: int, local_flags: list[bool], expected: bool) -> None:
    """Every PP rank must end up with the first stage's decision, not its own.

    PP ranks hold different layers, so each has its own pools, its own
    ``need_adjustment`` reading and its own cooldown clock -- left alone they
    would not agree by construction.  They must agree because rebalancing under
    PP means draining the microbatch ring, and a rank that stopped feeding the
    ring while its peers kept going would desynchronize the per-iteration
    send/recv chain and hang the pipeline.

    Args:
        world_size: Number of PP ranks.
        rank: This process's global rank.
        local_flags: Per-rank local ``need_adjustment`` readings.
        expected: The decision every rank should agree on.
    """
    dist = _pp_dist(world_size, rank)
    # A pure-PP mapping collapses TP and CP, so pp_broadcast alone carries this.
    assert dist.pp_size > 1 and dist.tp_size == 1 and dist.cp_size == 1, (
        f"expected a pure-PP mapping, got pp_size={dist.pp_size} "
        f"tp_size={dist.tp_size} cp_size={dist.cp_size}"
    )
    exe = _make_executor(dist, need_adjustment=local_flags[rank])

    got = PyExecutor._agreed_need_adjustment(exe)

    # Collect before asserting -- see run_agreement for why the order matters.
    all_decisions = dist.allgather(got)

    assert got == expected, (
        f"rank {rank} (pp_rank={dist.mapping.pp_rank}): local reading was "
        f"{local_flags[rank]}, the first PP stage said {local_flags[0]}, so the "
        f"agreed decision should be {expected}, got {got}"
    )
    assert len(set(all_decisions)) == 1, f"PP ranks disagreed: {all_decisions}"


def run_tp_pp_agreement(
    world_size: int, rank: int, tp_size: int, local_flags: list[bool], expected: bool
) -> None:
    """The TP and PP hops must chain, leaving global rank 0's decision everywhere.

    Ranks are laid out ``rank == pp_rank * tp_size + tp_rank``, so on a 2x2 the
    TP groups are ``[0, 1]`` and ``[2, 3]`` and the PP groups are ``[0, 2]`` and
    ``[1, 3]``.  With only global rank 0 reading ``True``:

    * the TP step gives ranks 0 and 1 rank 0's ``True``, and ranks 2 and 3 rank
      2's ``False``;
    * the PP step then broadcasts rank 0's ``True`` to rank 2, and rank 1's
      *post-TP* ``True`` to rank 3.

    So every rank ends ``True`` -- but only if the PP hop consumes the TP hop's
    result.  Had it broadcast each rank's own local reading, ranks 1 and 3 would
    come back ``False``, which is exactly what makes this case discriminating.

    Args:
        world_size: Total ranks, equal to ``tp_size * pp_size``.
        rank: This process's global rank.
        tp_size: Tensor-parallel width; PP width is derived from it.
        local_flags: Per-rank local ``need_adjustment`` readings.
        expected: The decision every rank should agree on.
    """
    pp_size = world_size // tp_size
    dist = _tp_pp_dist(tp_size, pp_size, rank)
    assert dist.pp_size > 1 and dist.tp_size > 1, (
        f"this test only means something when both dimensions are > 1, got "
        f"pp_size={dist.pp_size} tp_size={dist.tp_size}"
    )
    exe = _make_executor(dist, need_adjustment=local_flags[rank])

    got = PyExecutor._agreed_need_adjustment(exe)

    # Collect before asserting -- see run_agreement for why the order matters.
    all_decisions = dist.allgather(got)

    assert got == expected, (
        f"rank {rank} (tp_rank={dist.mapping.tp_rank}, "
        f"pp_rank={dist.mapping.pp_rank}): local reading was {local_flags[rank]}, "
        f"global rank 0 said {local_flags[0]}, so the agreed decision should be "
        f"{expected}, got {got}"
    )
    assert len(set(all_decisions)) == 1, f"TPxPP ranks disagreed: {all_decisions}"


def run_adp_pp_agreement(
    world_size: int, rank: int, tp_size: int, local_flags: list[bool], expected: list[bool]
) -> None:
    """Under ADP both hops run: TP OR-reduces, then PP broadcasts.

    A replica's *pipeline stages* all serve that replica's one request stream
    and must drain together or the replica's pipeline hangs -- that is the PP
    hop, and it is unchanged.  What changed is the TP hop: it no longer skips
    under ADP, because the rebalance path is collective across the TP dimension
    (see ``run_attention_dp_or_coupling``), so the replicas must enter together
    too.

    On a 2x2 with only global rank 0 reading ``True`` the outcome is therefore
    all ``True``: the TP OR carries rank 0's need to rank 1 within stage 0, and
    the PP broadcast then carries stage 0's decision to stage 1.  The previous
    expectation here was ``[True, False, True, False]``, which encoded the
    replica independence a live 4-rank Qwen3-Next run disproved.

    Args:
        world_size: Total ranks, equal to ``tp_size * pp_size``.
        rank: This process's global rank.
        tp_size: Tensor-parallel width; PP width is derived from it.
        local_flags: Per-rank local ``need_adjustment`` readings.
        expected: Per-rank agreed decision after the PP-only broadcast.
    """
    pp_size = world_size // tp_size
    dist = _tp_pp_dist(tp_size, pp_size, rank)
    exe = _make_executor(dist, need_adjustment=local_flags[rank], enable_attention_dp=True)

    got = PyExecutor._agreed_need_adjustment(exe)

    # Collect before asserting -- see run_agreement for why the order matters.
    all_decisions = dist.allgather(got)

    assert got == expected[rank], (
        f"rank {rank} (tp_rank={dist.mapping.tp_rank}, "
        f"pp_rank={dist.mapping.pp_rank}): local reading was {local_flags[rank]}, "
        f"its first stage agreed on {expected[rank]} after the TP OR, so the "
        f"agreed decision should be {expected[rank]}, got {got}"
    )
    assert all_decisions == expected, (
        f"ADP+PP decisions were {all_decisions}, expected {expected}: the TP OR "
        "must couple the replicas within a stage and the PP broadcast must then "
        "carry the first stage's decision to the rest"
    )


def run_throttle_lockstep(world_size: int, rank: int, interval: int, iterations: int) -> None:
    """Ranks must reach the agreement collective on the *same* iterations.

    This is the deadlock guard.  ``_can_pause_for_rebalance`` gates the
    collective, so ranks whose throttle cadence drifted apart would enter
    ``tp_broadcast`` on different iterations.  Driving the real collective
    inside the loop means such a divergence hangs here rather than passing
    silently; the allgather afterwards pins down that the firing iterations
    were in fact identical.

    Ranks are deliberately given *different* local ``need_adjustment`` readings,
    so it is the throttle cadence -- not agreement on the value -- that is under
    test here.

    Note the cadence is a pure function of ``iter_counter``, so drift can only
    come from ranks disagreeing about the iteration index itself.  The related
    hazard -- a rank skipping a check because of a rank-local gate -- cannot be
    covered here, because by construction it would leave the other ranks
    blocked in a collective that rank never enters; that one is pinned down
    without a real collective in ``TestRebalanceCheckThrottle``.

    Args:
        world_size: Number of TP ranks.
        rank: This process's global rank.
        interval: Throttle interval to configure.
        iterations: How many executor iterations to simulate.
    """
    dist = _tp_dist(world_size, rank)
    exe = _make_executor(dist, need_adjustment=(rank % 2 == 0), rebalance_check_interval=interval)

    fired_on = []
    for i in range(iterations):
        exe.iter_counter = i
        if PyExecutor._can_pause_for_rebalance(exe):
            fired_on.append(i)
            # Real collective -- diverging ranks hang instead of passing.
            PyExecutor._agreed_need_adjustment(exe)

    all_fired = dist.allgather(fired_on)
    assert all(f == all_fired[0] for f in all_fired), (
        f"ranks fired the rebalance check on different iterations: {all_fired}"
    )

    # Sanity: the throttle actually throttled, and it fired when expected.
    expected = [i for i in range(iterations) if i % interval == 0]
    assert fired_on == expected, (
        f"throttle fired on {fired_on}, expected {expected} for interval "
        f"{interval} over {iterations} iterations"
    )
    assert fired_on, "throttle never fired; the test would be vacuous"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _skip_if_not_enough_gpus(world_size: int) -> None:
    """Skip the calling test unless ``world_size`` GPUs are visible.

    Args:
        world_size: Number of ranks the test needs.
    """
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.cuda.device_count()}")


def _flags_for(case: str, world_size: int) -> list[bool]:
    """Local per-rank ``need_adjustment`` readings for each skew scenario.

    Args:
        case: Name of the skew scenario.
        world_size: Number of ranks to produce readings for.

    Returns:
        One local reading per rank.

    Raises:
        ValueError: If ``case`` is not a known scenario.
    """
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
def test_tp_ranks_agree_on_rebalance_trigger(world_size: int, case: str) -> None:
    """Rank 0's decision wins on every TP rank, whatever the local readings.

    Args:
        world_size: Number of TP ranks to run on.
        case: Skew scenario name, resolved by ``_flags_for``.
    """
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


@pytest.mark.parametrize(
    "case",
    ["only_rank0_true", "only_rank0_false"],
)
@pytest.mark.parametrize("world_size", [2, 4], ids=lambda x: f"cp:{x}")
def test_cp_ranks_agree_on_rebalance_trigger(world_size: int, case: str) -> None:
    """Pure CP (``tp_size == 1``) must agree just as TP does.

    Args:
        world_size: Number of CP ranks to run on.
        case: Skew scenario name, resolved by ``_flags_for``.
    """
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for(case, world_size)
    expected = flags[0]

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_cp_agreement, flags, expected)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize("cp_size", [2], ids=lambda x: f"cp:{x}")
@pytest.mark.parametrize("world_size", [4], ids=lambda x: f"world:{x}")
def test_cp_and_tp_ranks_agree_on_rebalance_trigger(world_size: int, cp_size: int) -> None:
    """Both broadcasts chained: CP x TP propagates global rank 0's decision.

    The pure-TP and pure-CP cases each collapse one dimension to 1, so between
    them they never run ``cp_broadcast`` and ``tp_broadcast`` back to back.  This
    case does, on a 2x2 topology.

    ``only_rank0_true`` is the discriminating scenario: every rank ends ``True``
    only if the TP step consumed the CP step's *result*.  Had it broadcast each
    rank's own local reading instead, the ranks in the second TP group would
    come back ``False``.

    Args:
        world_size: Total ranks; must equal ``cp_size * tp_size``.
        cp_size: Context-parallel width; TP width is ``world_size // cp_size``.
    """
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for("only_rank0_true", world_size)
    expected = flags[0]
    assert flags == [True, False, False, False], (
        "this case only discriminates the chained broadcast when rank 0 alone "
        f"reads True, got {flags}"
    )

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_cp_tp_agreement, cp_size, flags, expected)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize(
    "case",
    ["all_true", "all_false", "only_rank0_true", "only_rank0_false"],
)
@pytest.mark.parametrize("world_size", [2, 4], ids=lambda x: f"pp:{x}")
def test_pp_ranks_agree_on_rebalance_trigger(world_size: int, case: str) -> None:
    """The first PP stage's decision wins on every stage of the pipeline.

    Args:
        world_size: Number of PP ranks to run on.
        case: Skew scenario name, resolved by ``_flags_for``.
    """
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for(case, world_size)
    expected = flags[0]

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_pp_agreement, flags, expected)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize("tp_size", [2], ids=lambda x: f"tp:{x}")
@pytest.mark.parametrize("world_size", [4], ids=lambda x: f"world:{x}")
def test_tp_and_pp_ranks_agree_on_rebalance_trigger(world_size: int, tp_size: int) -> None:
    """TP then PP chained: the pair propagates global rank 0's decision.

    ``only_rank0_true`` is the discriminating scenario -- see
    ``run_tp_pp_agreement`` for the rank-by-rank walk-through.

    Args:
        world_size: Total ranks; must equal ``tp_size * pp_size``.
        tp_size: Tensor-parallel width; PP width is ``world_size // tp_size``.
    """
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for("only_rank0_true", world_size)
    expected = flags[0]
    assert flags == [True, False, False, False], (
        "this case only discriminates the chained broadcast when rank 0 alone "
        f"reads True, got {flags}"
    )

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_tp_pp_agreement, tp_size, flags, expected)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize("tp_size", [2], ids=lambda x: f"tp:{x}")
@pytest.mark.parametrize("world_size", [4], ids=lambda x: f"world:{x}")
def test_attention_dp_still_agrees_across_pp_stages(world_size: int, tp_size: int) -> None:
    """ADP must never suppress the PP hop, and no longer suppresses the TP hop.

    Each DP replica is itself a pipeline whose stages have to drain together;
    suppressing the PP hop would hang the replica rather than merely let it
    drift.  The TP hop now OR-reduces instead of being skipped, so a need
    anywhere in a stage reaches every rank in it.

    Args:
        world_size: Total ranks; must equal ``tp_size * pp_size``.
        tp_size: Tensor-parallel width; PP width is ``world_size // tp_size``.
    """
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for("only_rank0_true", world_size)
    # rank = pp_rank * tp_size + tp_rank.  Stage 0 is {0, 1} and ORs rank 0's
    # True across itself; the PP hop then carries that to stage 1 = {2, 3}.
    expected = [True] * world_size

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_adp_pp_agreement, tp_size, flags, expected)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize("world_size", [2, 4], ids=lambda x: f"tp:{x}")
def test_attention_dp_ranks_or_couple(world_size: int) -> None:
    """Attention DP agrees on *when*, not on *what*: the readings are OR-ed.

    ``only_rank0_false`` is the discriminating vector: rank 0 reads False while
    its peers read True.  A broadcast from rank 0 would give everyone False and
    starve the peers that need the rebalance; the old suppression would leave
    rank 0 out of a collective its peers enter.  Only the OR gives all True.

    Args:
        world_size: Number of TP ranks to run on.
    """
    _skip_if_not_enough_gpus(world_size)
    flags = _flags_for("only_rank0_false", world_size)

    # Non-vacuity: the vector must distinguish OR from broadcast-from-rank-0.
    assert any(flags) != flags[0], (
        "flags must differ between rank 0 and the OR, or the test cannot tell "
        "an OR-reduction from a rank-0 broadcast"
    )

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_attention_dp_or_coupling, flags)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize(
    "flags",
    [[False, True, False, False], [True, False, False, False]],
    ids=["cp_root_wins", "or_across_replicas"],
)
@pytest.mark.parametrize("cp_size", [2], ids=lambda x: f"cp:{x}")
@pytest.mark.parametrize("world_size", [4], ids=lambda x: f"world:{x}")
def test_attention_dp_agrees_over_cp_then_ors_across_tp(
    world_size: int, cp_size: int, flags: list[bool]
) -> None:
    """ADP + CP: CP ranks follow their root, then the replicas OR together.

    Nothing in ``Mapping`` or ``LlmArgs`` rejects ``enable_attention_dp`` with
    ``cp_size > 1``, so this topology is reachable and both hops have to be
    pinned down in it.

    Two flag vectors, because one cannot pin both properties:

    * ``cp_root_wins`` puts the only True on a *non-root* CP rank.  The CP
      broadcast discards it, so the OR that follows finds nothing and every rank
      ends False.  A plain global OR of the local readings would give True
      everywhere, so this is what proves the CP hop runs *before* the TP hop and
      is not merely folded into it.
    * ``or_across_replicas`` puts the only True on a CP root in one replica.  It
      survives the broadcast and the OR then carries it to the other replica, so
      every rank ends True -- the coupling the collective rebalance path needs.

    Args:
        world_size: Total ranks; must equal ``cp_size * tp_size``.
        cp_size: Context-parallel width; TP width is ``world_size // cp_size``.
        flags: Per-rank local ``need_adjustment`` readings for this case.
    """
    _skip_if_not_enough_gpus(world_size)

    # rank = tp_rank * cp_size + cp_rank, so cp_groups are consecutive ranks
    # ([[0, 1], [2, 3]]) and tp_groups stride by cp_size ([[0, 2], [1, 3]]).
    after_cp = [flags[(r // cp_size) * cp_size] for r in range(world_size)]
    expected = [
        any(after_cp[x] for x in range(world_size) if x % cp_size == r % cp_size)
        for r in range(world_size)
    ]

    # Non-vacuity: the chain must not collapse to a plain global OR of `flags`,
    # which is exactly the mistake this pair of vectors exists to catch.
    assert expected != flags, "flags must force some hop to change some rank"
    if not any(after_cp):
        assert any(flags), (
            "the cp_root_wins vector must start with a True somewhere, or it "
            "proves nothing about ordering"
        )
        assert expected == [False] * world_size, (
            "a True on a non-root CP rank must be discarded by the CP broadcast"
        )

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_adp_cp_agreement, cp_size, flags, expected)] * world_size),
        )
        for r in results:
            assert r is True


@pytest.mark.parametrize("interval", [1, 8], ids=lambda x: f"interval:{x}")
@pytest.mark.parametrize("world_size", [2, 4], ids=lambda x: f"tp:{x}")
def test_rebalance_check_stays_in_lockstep_across_ranks(world_size: int, interval: int) -> None:
    """The throttle must fire on identical iterations on every rank.

    A drift here would put ranks into ``tp_broadcast`` on different iterations,
    which deadlocks -- so this test hanging is itself the failure signal.

    Args:
        world_size: Number of TP ranks to run on.
        interval: Throttle interval to configure on every rank.
    """
    _skip_if_not_enough_gpus(world_size)

    with MPIPoolExecutor(max_workers=world_size) as ex:
        results = ex.map(
            run_single_rank,
            *zip(*[(world_size, run_throttle_lockstep, interval, 40)] * world_size),
        )
        for r in results:
            assert r is True
