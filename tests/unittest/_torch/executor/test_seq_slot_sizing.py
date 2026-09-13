# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seq-slot pool sizing.

Two pools are sized here, and keeping them distinct is the whole point:

The **seat pool** (``compute_max_num_sequences``) is how many sequences the
executor can seat at once. Baseline ``max_batch_size * pp_size``, plus one extra
micro-batch under the overlap headroom. Under the overlap scheduler, requests
finished in the previous iteration still hold their sequence slots when the next
iteration's ``prepare_resources`` runs, while the V2 capacity scheduler has
already dropped them from its budget
(``no_schedule_after_state=GENERATION_TO_COMPLETE``) and the ADP router has
already excluded them from the counts admission subtracts from its capacity --
so a replacement cohort is admitted while the retiring one is still resident.
``should_enable_overlap_headroom`` is the single gate for that, and it is
deliberately narrow: attention DP only (outside it, admission subtracts
``len(active_requests)`` including retirees), non-PP only, V2 only (the V1
capacity schedulers hardcode ``GENERATION_COMPLETE``), non-hybrid only (SSM state
is sized from ``max_batch_size``).

The **index pool** (``KVCacheManagerV2.max_admissible_sequences``) carries the
same headroom term *plus* a disaggregation term the seat pool must not have: a
request awaiting its KV transfer holds an index lease and no seat at all, because
``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``. That is
why ``validate_seq_slot_pool_covers_admission`` is asymmetric rather than an
equality.

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
# The terms are additive, not multiplicative: pipeline depth costs pp_size
# micro-batches of seats, and the overlap deferral costs exactly one more
# generation on top -- not one more per stage. At pp_size == 1 the two readings
# coincide at 2x, which is why the headroom used to be expressible as a factor of
# 2; the pp>1 rows are where they part company (5x, not 8x, at pp=4). The gate
# never turns the headroom on under PP, so those rows only pin the arithmetic.
#
# Disaggregation is absent from this table on purpose -- see
# test_seat_pool_has_no_disagg_term.
SIZING_CASES = [
    # No PP: aggregated baseline, then the one extra micro-batch.
    (1, False, False, 1),
    (1, False, True, 2),
    # The headroom is a no-op with the overlap scheduler off: the retiring cohort
    # is torn down in-line, so it never coexists with its replacement.
    (1, True, True, 1),
    # PP without the headroom.
    (4, False, False, 4),
    (4, True, True, 4),
    # PP with the headroom: additive, not per-stage.
    (2, False, True, 3),
    (4, False, True, 5),
]


@pytest.mark.parametrize(
    "enable_attention_dp,pp_size,disable_overlap,is_v2,is_hybrid,expected",
    [
        # The scenario this PR exists for: attention DP, no PP, overlap on, V2.
        (True, 1, False, True, False, True),
        # Outside attention DP the extra seats can never be occupied: admission
        # subtracts len(active_requests), retirees included, so residency is
        # capped at max_batch_size * pp_size however the run is deployed.
        (False, 1, False, True, False, False),
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

    ``KVCacheManagerV2`` sizes its index pool from ``max_batch_size``, ``pp_size``,
    ``is_disagg`` and the *same* resolved headroom flag the seat pool uses.
    Asserting the two expressions against each other rather than against literals
    means a drift in either one fails here rather than at startup in
    ``validate_seq_slot_pool_covers_admission`` -- and the assertion is exactly
    that validator's invariant.
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
    admission_bound = max_batch_size * pp_size + max(
        max_batch_size * pp_size if is_disagg else 0,
        max_batch_size if enable_overlap_headroom else 0,
    )

    # The invariant the startup validator enforces, in both its directions.
    assert admission_bound >= seats
    if not is_disagg:
        assert admission_bound == seats


class _FakeManager:
    """Stands in for a KV cache manager that publishes an admission bound."""

    def __init__(self, max_admissible_sequences):
        self.max_admissible_sequences = max_admissible_sequences


@pytest.mark.parametrize("is_disagg", [False, True])
def test_validator_accepts_the_matching_pair(is_disagg):
    validate_seq_slot_pool_covers_admission(16, _FakeManager(16), is_disagg=is_disagg)


@pytest.mark.parametrize("is_disagg", [False, True])
def test_validator_always_rejects_an_index_pool_below_the_seat_pool(is_disagg):
    """The direction that shipped as nvbug 6627795, and it is never legitimate.

    A one-sided ``seats >= admissible`` guard is what let it through: the seat
    pool grew to 2B while the index pool stayed at B+1, which satisfies the
    one-sided form and silently defers admitted requests one at a time.
    Disaggregation is no excuse here -- its 2x makes the index pool *larger*, so
    a shortfall under disagg means the two numbers were derived separately.
    """
    with pytest.raises(ValueError, match="smaller than the seat"):
        validate_seq_slot_pool_covers_admission(16, _FakeManager(8), is_disagg=is_disagg)


def test_validator_rejects_a_larger_index_pool_only_when_aggregated():
    """The other direction is a bug when aggregated and by design under disagg.

    Aggregated, every indexed sequence is also a seated one, so a surplus can
    only mean one of the two numbers was re-derived -- and a request would be
    admitted that cannot be seated, with ``SlotManager.add_slot`` raising on the
    executor's event-loop thread. Under disaggregation the surplus *is* the
    mechanism: a request in KV transfer holds its index lease with no seat
    (``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``), so
    the index pool carries a 2x that must not reach the seat pool. Admission is
    bounded independently at ``max_batch_size * pp_size`` either way, so the
    surplus is never extra concurrency.
    """
    with pytest.raises(ValueError, match="larger than the seat"):
        validate_seq_slot_pool_covers_admission(16, _FakeManager(32), is_disagg=False)

    validate_seq_slot_pool_covers_admission(16, _FakeManager(32), is_disagg=True)


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


def _make_kv_cache_creator(engine_headroom_flag, is_v2: bool = True) -> KvCacheCreator:
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
    c._enable_overlap_headroom = engine_headroom_flag
    # Short-circuit the post-construction max_seq_len fixup.
    c._skip_est = True
    c._get_model_kv_cache_manager_cls = Mock(return_value=Mock())
    c._should_create_separate_draft_kv_cache = Mock(return_value=False)
    c._enable_kv_cache_stats = Mock(return_value=False)
    return c


@pytest.mark.parametrize("enable_overlap_headroom", [False, True])
def test_kv_cache_manager_receives_the_resolved_headroom_flag(enable_overlap_headroom):
    """The manager consumes the gate's answer instead of re-deriving it.

    It used to take ``disable_overlap_scheduler`` and rebuild
    ``enable_attention_dp and not disable_overlap and not has_pp`` for itself.
    Once the seat-pool gate also excludes V1 and hybrid, that second copy
    disagrees: hybrid V2 + attention DP + overlap would size the index pool at
    ``2 * B * pp`` against a seat pool of ``B * pp``, and
    ``validate_seq_slot_pool_covers_admission`` would raise at startup on an
    aggregated server. One resolved flag, passed down.
    """
    creator = _make_kv_cache_creator(enable_overlap_headroom)
    model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(is_generation=True)),
    )

    with patch(
        "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
        return_value=None,
    ) as create:
        creator._create_kv_cache_manager(model_engine)

    assert create.call_args.kwargs["enable_overlap_headroom"] is enable_overlap_headroom
    assert "disable_overlap_scheduler" not in create.call_args.kwargs


@pytest.mark.parametrize(
    "engine_flag,is_v2,expected",
    [
        (True, True, True),
        # The engine gates on kv_cache_config.use_kv_cache_manager_v2 -- the
        # resolved request -- while the creator knows the class that was actually
        # selected. When they disagree the seat pool has already been sized, so
        # the surplus seats are accepted (inert: V1 publishes no admission bound,
        # so the startup validator skips it) and the index pool is left alone.
        (True, False, False),
        (False, True, False),
        (False, False, False),
    ],
)
def test_creator_reconciles_the_headroom_flag_with_the_selected_manager(
    engine_flag, is_v2, expected
):
    creator = _make_kv_cache_creator(engine_flag, is_v2=is_v2)
    creator._kv_cache_manager_cls = type("FakeManager", (), {})
    engine = SimpleNamespace(_enable_overlap_headroom=engine_flag)

    assert creator._resolve_overlap_headroom(engine) is expected


def test_creator_treats_a_headroom_free_engine_as_no_headroom():
    """mm-encoder-only engines and test stubs predate the attribute."""
    creator = _make_kv_cache_creator(False)
    creator._kv_cache_manager_cls = type("FakeManager", (), {})

    assert creator._resolve_overlap_headroom(SimpleNamespace()) is False
