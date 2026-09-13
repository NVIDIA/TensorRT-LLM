# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seq-slot pool sizing.

Two pools are sized here, and keeping them distinct is the whole point:

The **seat pool** (``compute_max_num_sequences``) is how many sequences the
executor can seat at once: ``max_batch_size * pp_size`` under pipeline
parallelism, and otherwise ``max_batch_size`` doubled by the overlap headroom.
Under the overlap scheduler, requests finished in the previous iteration still
hold their sequence slots when the next iteration's ``prepare_resources`` runs,
while the V2 capacity scheduler has already dropped them from its budget
(``no_schedule_after_state=GENERATION_TO_COMPLETE``) and the ADP router has
already excluded them from the counts admission subtracts from its capacity --
so a replacement cohort is admitted while the retiring one is still resident.
``should_enable_overlap_headroom`` is the single gate for that. Its overlap term
matches ``KVCacheManagerV2``'s by construction -- non-PP and overlap-on -- so the
seat pool and the index pool move together; attention DP is deliberately *not* a
condition even though it is the only deployment where the extra seats are ever
occupied, because narrowing the seat gate below the manager's term is what forces
the two pools apart on the most ordinary path there is. It stays off for V1 (the
V1 capacity schedulers hardcode ``GENERATION_COMPLETE``) and for hybrid models
(SSM state is sized from ``max_batch_size``).

The **index pool** (``KVCacheManagerV2.max_admissible_sequences``) therefore
equals the seat pool in the common case, and is the more generous of the two
wherever they differ. Its ``is_disagg`` term has no seat-pool counterpart: a
request awaiting its KV transfer holds an index lease and no seat at all, because
``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``. And a
hybrid model suppresses the seat headroom while a V2 index pool behind it still
widens. Hence ``validate_seq_slot_pool_covers_admission`` requires the index pool
to *cover* the seat pool rather than to equal it.

``compute_max_num_sequences`` is the single seat-pool implementation used both
for the executor's ``SeqSlotManager`` pool (``create_py_executor_instance``) and
for the sampler state (``create_torch_sampler_args``);
``resolve_max_num_sequences`` is how a consumer obtains it without re-deriving
it. Every other slot-indexed buffer follows the same number, since
``py_seq_slot`` indexes them all.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    KvCacheCreator,
    compute_max_num_sequences,
    create_torch_sampler_args,
    is_disagg_enabled,
    resolve_max_num_sequences,
    should_enable_adp_dummy_fixes,
    should_enable_non_overlap_adp_forward_intent,
    should_enable_overlap_headroom,
    should_enable_scheduler_aware_adp_dummy,
    validate_seq_slot_pool_covers_admission,
)
from tensorrt_llm.mapping import Mapping

# (pp_size, disable_overlap, enable_overlap_headroom, expected_factor)
#
# The two terms are mutually exclusive rather than combined: pipeline depth
# already puts pp_size micro-batches in flight, so the PP branch takes pp_size
# and ignores the headroom entirely. The gate never turns the headroom on under
# PP (see test_overlap_headroom_gate), so the pp>1 rows only pin that the branch
# cannot start compounding.
#
# Disaggregation is absent from this table on purpose -- see
# test_seat_pool_has_no_disagg_term.
SIZING_CASES = [
    # No PP: aggregated baseline, then the extra micro-batch.
    (1, False, False, 1),
    (1, False, True, 2),
    # The headroom is a no-op with the overlap scheduler off: the retiring cohort
    # is torn down in-line, so it never coexists with its replacement.
    (1, True, True, 1),
    # PP: sized by pp_size, with or without the flag set.
    (4, False, False, 4),
    (4, True, True, 4),
    (2, False, True, 2),
    (4, False, True, 4),
]


@pytest.mark.parametrize(
    "enable_attention_dp,pp_size,disable_overlap,is_v2,is_hybrid,expected",
    [
        # The scenario this PR exists for: attention DP, no PP, overlap on, V2.
        (True, 1, False, True, False, True),
        # Attention DP is deliberately absent from the predicate, so a plain TP
        # run of the same shape widens too. The extra seats are never occupied
        # there -- admission subtracts len(active_requests) with retirees
        # included -- but matching KVCacheManagerV2's own term keeps the two
        # pools equal instead of making the index pool run ahead on the most
        # ordinary path there is.
        (False, 1, False, True, False, True),
        # Overlap off: the retiring cohort is torn down before the next
        # iteration's admission runs.
        (True, 1, True, True, False, False),
        # Pipeline parallelism: already sized by pp_size, and only the last stage
        # marks a generation request GENERATION_TO_COMPLETE, so the router's
        # correction is not rank-consistent. Out of scope.
        (True, 2, False, True, False, False),
        (True, 4, False, True, False, False),
        # V1: BindCapacityScheduler and PyCapacityScheduler both hardcode
        # no_schedule_after_state=GENERATION_COMPLETE, so the retiring cohort is
        # still inside the capacity budget and no overcommit is possible.
        (True, 1, False, False, False, False),
        # Hybrid/SSM: recurrent state is sized from max_batch_size in every
        # manager, so extra seats would let residency outrun the state slots.
        (True, 1, False, True, True, False),
        # Both exclusions at once, as in
        # examples/configs/curated/qwen3.8-high-throughput-mtp3.yaml.
        (True, 1, False, False, True, False),
    ],
)
def test_overlap_headroom_gate(
    enable_attention_dp, pp_size, disable_overlap, is_v2, is_hybrid, expected
):
    mapping = Mapping(
        world_size=pp_size,
        tp_size=1,
        pp_size=pp_size,
        enable_attention_dp=enable_attention_dp,
    )

    assert (
        should_enable_overlap_headroom(
            mapping,
            disable_overlap,
            kv_cache_manager_is_v2=is_v2,
            is_hybrid=is_hybrid,
        )
        is expected
    )


def test_overlap_headroom_gate_does_not_depend_on_disaggregation():
    """Disagg is not a reason to widen the *seat* pool, only the index pool.

    The gate used to take ``cache_transceiver_config`` and return True for a
    disaggregated server regardless of attention DP. That bought seats that could
    never be occupied: a request in KV transfer holds an index lease and no seat,
    and outside attention DP admission subtracts ``len(active_requests)`` so
    residency stays at ``max_batch_size * pp_size``. Asserting on the signature
    states the intent that a value test cannot: the parameter's mere presence is
    what invites the conflation back.
    """
    params = inspect.signature(should_enable_overlap_headroom).parameters
    assert "cache_transceiver_config" not in params
    assert "is_disagg" not in params


@pytest.mark.parametrize("disable_overlap", [False, True])
def test_overlap_headroom_gate_does_not_depend_on_attention_dp(disable_overlap):
    """Attention DP is where the seats are *used*, not where they are sized.

    The gate deliberately ignores it so that its overlap term is the same shape
    as ``KVCacheManagerV2``'s (``not has_pp() and overlap on``). Making the seat
    pool the narrower of the two is what forces
    ``validate_seq_slot_pool_covers_admission`` to tolerate a permanent gap on
    plain TP with the overlap scheduler -- a default configuration -- and a
    tolerated gap is one nobody can distinguish from an undersized seat pool
    later. The cost of ignoring it is idle seats outside attention DP, where
    admission subtracts ``len(active_requests)`` with retirees included; the
    benefit is that the two pools are computed from the same condition.
    """
    kwargs = dict(kv_cache_manager_is_v2=True, is_hybrid=False)
    with_adp = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=True)
    without_adp = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=False)

    assert should_enable_overlap_headroom(
        with_adp, disable_overlap, **kwargs
    ) is should_enable_overlap_headroom(without_adp, disable_overlap, **kwargs)


@pytest.mark.parametrize("pp_size,expected", [(1, True), (2, False)])
def test_adp_dummy_fix_gate(pp_size, expected):
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert should_enable_adp_dummy_fixes(mapping) is expected


@pytest.mark.parametrize(
    "model_type,pp_size,disable_overlap,expected",
    [
        ("kimi_k2", 1, True, True),
        ("kimi_k2", 1, False, False),
        ("deepseek_v4", 1, False, True),
        ("qwen3_5_moe", 1, False, True),
        ("deepseek_v4", 2, True, False),
    ],
)
def test_scheduler_aware_adp_dummy_scope(model_type, pp_size, disable_overlap, expected):
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert should_enable_scheduler_aware_adp_dummy(model_type, mapping, disable_overlap) is expected


@pytest.mark.parametrize(
    "pp_size,disable_overlap,expected",
    [
        (1, True, True),
        (1, False, False),
        (2, True, False),
    ],
)
def test_non_overlap_adp_forward_intent_scope(pp_size, disable_overlap, expected):
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert should_enable_non_overlap_adp_forward_intent(mapping, disable_overlap) is expected


@pytest.mark.parametrize(
    "pp_size,disable_overlap,enable_overlap_headroom,expected_factor", SIZING_CASES
)
def test_compute_max_num_sequences_scopes_overlap_headroom(
    pp_size, disable_overlap, enable_overlap_headroom, expected_factor
):
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert (
        compute_max_num_sequences(
            mapping,
            max_batch_size,
            disable_overlap,
            enable_overlap_headroom=enable_overlap_headroom,
        )
        == max_batch_size * expected_factor
    )


def test_seat_pool_has_no_disagg_term():
    """The disaggregation 2x is confined to KVCacheManagerV2's index pool.

    A request awaiting its KV transfer holds an *index* lease and no seat at all:
    ``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``
    requests outright and only seats one once its transmission completes, while
    admission stays at ``max_batch_size * pp_size``. So the index pool
    legitimately runs ahead of the seat pool, and doubling the *seat* pool would
    buy nothing while doubling everything keyed by seat -- sampler state,
    ``[seats, draft_len, vocab]`` draft probabilities (~800 MB at 512 seats), the
    penalty tensors and the pinned-host block-offset tables.

    Asserting on the signature rather than on a return value is deliberate: a
    value test cannot distinguish "the parameter is gone" from "the parameter
    defaults to False", and it is the parameter's *existence* that invites a
    caller to propagate the factor.
    """
    assert "is_disagg" not in inspect.signature(compute_max_num_sequences).parameters
    assert "is_disagg" not in inspect.signature(resolve_max_num_sequences).parameters


@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
def test_sizing_matches_kv_manager_admission_bound(
    enable_attention_dp, pp_size, is_disagg, disable_overlap_scheduler
):
    """The two pools are computed in different modules; hold them in step.

    Asserting the two expressions against each other rather than against literals
    means a drift in either one fails here rather than at startup in
    ``validate_seq_slot_pool_covers_admission`` -- and the assertion is exactly
    that validator's invariant.

    The overlap term is shared, so the two agree everywhere except where the
    manager's ``is_disagg`` term fires without the shared term also firing --
    i.e. a disaggregated run that is either PP or overlap-off. There the index
    pool runs ahead, which is permitted; what must never happen is the reverse.
    """
    max_batch_size = 8
    mapping = Mapping(
        world_size=pp_size,
        tp_size=1,
        pp_size=pp_size,
        enable_attention_dp=enable_attention_dp,
    )
    enable_overlap_headroom = should_enable_overlap_headroom(
        mapping,
        disable_overlap_scheduler,
        kv_cache_manager_is_v2=True,
    )

    seats = compute_max_num_sequences(
        mapping,
        max_batch_size,
        disable_overlap_scheduler,
        enable_overlap_headroom=enable_overlap_headroom,
    )

    # Mirrors the arithmetic in KVCacheManagerV2.__init__.
    overlap_term = not disable_overlap_scheduler and pp_size == 1
    extra_leases = is_disagg or overlap_term
    admission_bound = max_batch_size * pp_size * (2 if extra_leases else 1)

    # The invariant the startup validator enforces.
    assert admission_bound >= seats
    # Equal unless the manager's disagg-only term is what widened it: the overlap
    # term is shared with the seat gate, so it can never open a gap.
    if is_disagg and not overlap_term:
        assert admission_bound == 2 * seats
    else:
        assert admission_bound == seats


class _FakeManager:
    """Stands in for a KV cache manager that publishes an admission bound."""

    def __init__(self, max_admissible_sequences):
        self.max_admissible_sequences = max_admissible_sequences


def test_validator_accepts_the_matching_pair():
    validate_seq_slot_pool_covers_admission(16, _FakeManager(16))


def test_validator_rejects_an_index_pool_below_the_seat_pool():
    """The direction that shipped as nvbug 6627795, and it is never legitimate.

    A one-sided ``seats >= admissible`` guard is what let it through: the seat
    pool grew to 2B while the index pool stayed at B+1, which satisfies the
    one-sided form and silently defers admitted requests one at a time.
    """
    with pytest.raises(ValueError, match="smaller than the seat"):
        validate_seq_slot_pool_covers_admission(16, _FakeManager(8))


def test_validator_permits_an_index_pool_above_the_seat_pool():
    """A surplus of index leases is the design, in two independent ways.

    Under disaggregation a request in KV transfer holds its index lease with no
    seat (``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``),
    so the manager's ``is_disagg`` term deliberately has no seat-pool
    counterpart. And a hybrid model suppresses the seat headroom -- SSM state is
    sized from ``max_batch_size`` -- while a V2 index pool behind it still
    widens. Admission is bounded independently at ``max_batch_size * pp_size``,
    so the surplus is never extra concurrency -- and it is the cheap direction:
    spare leases cost page-table rows, spare seats would cost sampler and
    speculative-decoding state.

    The shared overlap term is what keeps this from being the *common* case
    rather than the exception: were the seat gate narrowed back to attention DP,
    every plain-TP overlap run would land here and the tolerated gap would stop
    being evidence of anything.

    Asserting on the signature as well: ``is_disagg`` used to select which
    surplus was tolerated, and its absence is what stops the validator from
    growing a second copy of the manager's predicate.
    """
    validate_seq_slot_pool_covers_admission(16, _FakeManager(32))
    assert "is_disagg" not in inspect.signature(validate_seq_slot_pool_covers_admission).parameters


def test_validate_seq_slot_pool_ignores_managers_without_a_bound():
    """The V1/C++ manager does not publish one; the check must not fire."""
    validate_seq_slot_pool_covers_admission(1, Mock(spec=[]))
    validate_seq_slot_pool_covers_admission(1, None)


def test_validate_seq_slot_pool_ignores_a_non_integer_bound():
    """A bare ``Mock`` auto-creates the attribute, so ``is None`` is not enough.

    ``create_py_executor_instance`` is called with a ``Mock()`` cache manager in
    other modules' tests (e.g. test_dual_pool_kv_cache). Keying the opt-in on
    ``is None`` would let a ``Mock`` attribute reach the comparison and raise
    ``TypeError`` from a startup validator.
    """
    validate_seq_slot_pool_covers_admission(1, Mock())


@pytest.mark.parametrize(
    "cache_transceiver_config,expected",
    [
        (None, False),
        (SimpleNamespace(backend=None), False),
        (SimpleNamespace(backend="UCX"), True),
    ],
)
def test_is_disagg_enabled_is_the_single_definition(cache_transceiver_config, expected):
    """One definition of "this is a disaggregated server".

    The ``backend is not None`` test used to be inlined at each use site, which is
    how a derived fact acquires copies that then disagree.
    """
    assert is_disagg_enabled(cache_transceiver_config) is expected


@pytest.mark.parametrize(
    "explicit,engine_seats,expected",
    [
        (24, 16, 24),  # an explicit value wins
        (None, 16, 16),  # otherwise the engine's own pool
        (None, None, 16),  # only then recompute, *with* the engine's gate
    ],
)
def test_resolve_max_num_sequences_prefers_the_published_pool(explicit, engine_seats, expected):
    """The fallback must never be able to undercut the pool it indexes.

    The recomputing branch used to be the *first* branch and was called without
    ``enable_overlap_headroom``, so a caller that omitted ``max_num_sequences``
    silently sized the sampler and the executor's SeqSlotManager below the index
    pool they share indices with. The third row is that branch, and it must still
    land on the headroom value.
    """
    engine = SimpleNamespace(
        max_num_seq_slots=engine_seats,
        _enable_overlap_headroom=True,
    )
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=True)
    llm_args = SimpleNamespace(disable_overlap_scheduler=False)

    assert (
        resolve_max_num_sequences(
            engine,
            mapping,
            8,
            llm_args,
            max_num_sequences=explicit,
        )
        == expected
    )


def test_resolve_max_num_sequences_reads_llm_args_only_in_the_fallback():
    """The two short-circuit branches must not touch ``llm_args`` at all.

    Reading ``disable_overlap_scheduler`` at the *call site* made every caller
    depend on a field only the third branch uses, which broke callers that hold a
    lighter args object and pass ``max_num_sequences`` explicitly. An args object
    that raises on attribute access is the only way to state that as a test:
    asserting on the return value cannot distinguish "not used" from "used and
    happened to agree".
    """

    class _Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"llm_args.{name} read on a path that must not need it")

    engine_with_pool = SimpleNamespace(max_num_seq_slots=16)
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=True)

    # Branch 1: an explicit value wins, even with no pool published at all.
    assert (
        resolve_max_num_sequences(
            SimpleNamespace(),
            mapping,
            8,
            _Exploding(),
            max_num_sequences=24,
        )
        == 24
    )
    # Branch 2: the engine's published pool.
    assert resolve_max_num_sequences(engine_with_pool, mapping, 8, _Exploding()) == 16


def test_sampler_args_require_the_resolved_pool():
    """``max_num_sequences`` is required, and the raw material for re-deriving it
    is gone from the signature.

    ``create_torch_sampler_args`` used to default it by recomputing from
    ``mapping``/``max_batch_size`` without the headroom gate, i.e. it could only
    ever produce a number smaller than the slots the sampler indexes.
    """
    params = inspect.signature(create_torch_sampler_args).parameters

    assert params["max_num_sequences"].default is inspect.Parameter.empty
    assert "mapping" not in params
    assert "max_batch_size" not in params


@pytest.mark.parametrize("slot_factor", [1, 2])
def test_sampler_uses_executor_slot_pool_capacity(slot_factor):
    max_batch_size = 8
    max_num_sequences = max_batch_size * slot_factor
    args = create_torch_sampler_args(
        max_seq_len=1024,
        speculative_config=None,
        max_beam_width=1,
        disable_overlap_scheduler=False,
        enable_async_worker=False,
        enable_speculative_beam_history_d2h=False,
        max_num_sequences=max_num_sequences,
    )
    assert args.max_num_sequences == max_num_sequences


def _make_kv_cache_creator(disable_overlap_scheduler: bool, is_v2: bool = True) -> KvCacheCreator:
    """Minimal creator whose only job is to reach _create_kv_cache_manager."""
    c = object.__new__(KvCacheCreator)
    c._mapping = Mapping(world_size=1, tp_size=1, pp_size=1)
    c._kv_cache_config = Mock()
    c._tokens_per_block = 32
    c._max_seq_len = 1024
    c._max_batch_size = 8
    c._max_num_tokens = 8192
    c._max_beam_width = 1
    c._speculative_config = None
    c._sparse_attention_config = None
    c._kv_connector_manager = None
    c._execution_stream = None
    c._is_disagg = False
    c._is_kv_cache_manager_v2 = is_v2
    c._disable_overlap_scheduler = disable_overlap_scheduler
    # Short-circuit the post-construction max_seq_len fixup.
    c._skip_est = True
    c._get_model_kv_cache_manager_cls = Mock(return_value=Mock())
    c._should_create_separate_draft_kv_cache = Mock(return_value=False)
    c._enable_kv_cache_stats = Mock(return_value=False)
    return c


@pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
def test_kv_cache_manager_receives_the_overlap_scheduler_flag(disable_overlap_scheduler):
    """The index pool needs the overlap flag, and only the creator can supply it.

    ``KVCacheManagerV2`` derives its own lease headroom from
    ``is_disagg or (overlap on and not has_pp)`` -- deliberately looser than the
    seat gate, since over-leasing is cheap and under-leasing hangs. That makes
    ``disable_overlap_scheduler`` load-bearing on the constructor: drop it and
    every aggregated V2 deployment silently reverts to a single cohort of leases,
    which is the nvbug 6627795 shortfall.
    """
    creator = _make_kv_cache_creator(disable_overlap_scheduler)
    model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(is_generation=True)),
    )

    with patch(
        "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
        return_value=None,
    ) as create:
        creator._create_kv_cache_manager(model_engine)

    assert create.call_args.kwargs["disable_overlap_scheduler"] is disable_overlap_scheduler
