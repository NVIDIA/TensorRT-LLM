# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seq-slot pool sizing.

Two independent reasons the pool must exceed max_batch_size, both selected from
runtime topology rather than model architecture:

Attention-DP overlap headroom. Under the overlap scheduler, requests finished in
the previous iteration still hold their sequence slots when the next iteration's
prepare_resources runs, while the capacity scheduler has already dropped them
from its budget (no_schedule_after_state=GENERATION_TO_COMPLETE) and backfilled
their seats. Transient slot demand is therefore one extra micro-batch worth of
slots, regardless of whether speculative decoding is enabled. Pipeline
parallelism is out of scope for this term: the pool is already sized by pp_size
there, and the ADP router's retiring-request correction is not rank-consistent
because only the last pipeline stage marks generation requests
GENERATION_TO_COMPLETE.

Disaggregated serving. On a generation server the admission bound is
KVCacheManagerV2's IndexMapper rather than the seat pool, and it is sized at
twice max_num_sequences. Unlike the headroom above, this holds regardless of
attention DP, overlap, or PP.

compute_max_num_sequences is the single sizing implementation used both for the
executor's SeqSlotManager pool (create_py_executor_instance) and for the sampler
state (create_torch_sampler_args); resolve_max_num_sequences is how a consumer
obtains it without re-deriving it. Every other slot-indexed buffer follows the
same number, since py_seq_slot indexes them all.
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
    should_enable_disagg_adp_overlap_headroom,
    should_enable_non_overlap_adp_forward_intent,
    should_enable_scheduler_aware_adp_dummy,
    validate_seq_slot_pool_covers_admission,
)
from tensorrt_llm.mapping import Mapping

_UCX = SimpleNamespace(backend="UCX")

# (pp_size, enable_overlap_headroom, expected_factor) for an aggregated server.
# The disaggregated counterpart is DISAGG_SIZING_CASES below.
SIZING_CASES = [
    # No PP: the headroom buys one extra micro-batch worth of seats.
    (1, False, 1),
    (1, True, 2),
    # PP sizes the pool by pipeline depth and ignores the headroom.
    (4, False, 4),
    (2, True, 2),
    (4, True, 4),
]


@pytest.mark.parametrize(
    "enable_attention_dp,pp_size,cache_transceiver_config,disable_overlap,expected",
    [
        # Aggregated ADP with overlap on: nvbug 6627795 reproduced here, on a
        # context-only run with no cache transceiver configured.
        (True, 1, None, False, True),
        (False, 1, None, False, False),
        # Aggregated ADP with overlap off: teardown is in-line, no headroom.
        (True, 1, None, True, False),
        # Disagg needs the headroom even with overlap off: a request awaiting its
        # KV transfer keeps its lease.
        (True, 1, _UCX, True, True),
        (True, 1, _UCX, False, True),
        (False, 1, _UCX, False, False),
        # Pipeline parallelism is out of scope in both directions.
        (True, 2, None, False, False),
        (True, 4, _UCX, False, False),
    ],
)
def test_disagg_adp_overlap_headroom_gate(
    enable_attention_dp, pp_size, cache_transceiver_config, disable_overlap, expected
):
    mapping = Mapping(
        world_size=pp_size,
        tp_size=1,
        pp_size=pp_size,
        enable_attention_dp=enable_attention_dp,
    )

    assert (
        should_enable_disagg_adp_overlap_headroom(
            mapping, cache_transceiver_config, disable_overlap
        )
        is expected
    )


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


@pytest.mark.parametrize("pp_size,enable_overlap_headroom,expected_factor", SIZING_CASES)
def test_compute_max_num_sequences_scopes_overlap_headroom(
    pp_size, enable_overlap_headroom, expected_factor
):
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert (
        compute_max_num_sequences(
            mapping,
            max_batch_size,
            disable_overlap_scheduler=False,
            enable_overlap_headroom=enable_overlap_headroom,
        )
        == max_batch_size * expected_factor
    )


# (pp_size, enable_overlap_headroom, expected_factor). The factor is relative to
# max_batch_size and, for disagg, must cover the IndexMapper's 2x.
DISAGG_SIZING_CASES = [
    # Disagg gets the coefficient on topology alone, with no headroom opt-in.
    (1, False, 2),
    # The two factors overlap rather than compose, so this stays 2x.
    (1, True, 2),
    # PP composes: the IndexMapper is likewise sized pp_size * 2.
    (2, False, 4),
    (4, False, 8),
]


@pytest.mark.parametrize("pp_size,enable_overlap_headroom,expected_factor", DISAGG_SIZING_CASES)
def test_disagg_seats_cover_index_mapper_capacity(
    pp_size, enable_overlap_headroom, expected_factor
):
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)

    seats = compute_max_num_sequences(
        mapping,
        max_batch_size,
        disable_overlap_scheduler=False,
        enable_overlap_headroom=enable_overlap_headroom,
        is_disagg=True,
    )

    assert seats == max_batch_size * expected_factor

    # Seats must cover every request admission can let through. Mirrors the
    # expression in KVCacheManagerV2.__init__.
    index_mapper_capacity = max_batch_size * pp_size * 2
    assert seats >= index_mapper_capacity


@pytest.mark.parametrize("disable_overlap", [False, True])
def test_disagg_seats_do_not_depend_on_overlap_scheduler(disable_overlap):
    """The IndexMapper is sized the same either way, so seats must be too.

    Turning the overlap scheduler off removes the terminal-slot race but not the
    transfer/generate overlap that the 2x coefficient exists for.
    """
    max_batch_size = 8
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1)

    assert (
        compute_max_num_sequences(mapping, max_batch_size, disable_overlap, is_disagg=True)
        == max_batch_size * 2
    )


def test_aggregate_sizing_is_unchanged():
    """Aggregated deployments size the pool at one forward batch."""
    max_batch_size = 8
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1)

    assert (
        compute_max_num_sequences(
            mapping, max_batch_size, disable_overlap_scheduler=False, is_disagg=False
        )
        == max_batch_size
    )


@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("is_disagg", [False, True])
@pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
def test_sizing_matches_kv_manager_admission_bound(
    enable_attention_dp, pp_size, is_disagg, disable_overlap_scheduler
):
    """The two coefficients are computed independently; hold them in step.

    KVCacheManagerV2 derives its own admission bound from max_batch_size,
    pp_size, is_disagg and the attention-DP overlap condition. Assert the two
    expressions against each other rather than against a literal, so a drift in
    either one fails here rather than at startup in
    validate_seq_slot_pool_covers_admission.
    """
    max_batch_size = 8
    mapping = Mapping(
        world_size=pp_size,
        tp_size=1,
        pp_size=pp_size,
        enable_attention_dp=enable_attention_dp,
    )

    seats = compute_max_num_sequences(
        mapping,
        max_batch_size,
        disable_overlap_scheduler,
        enable_overlap_headroom=should_enable_disagg_adp_overlap_headroom(
            mapping,
            _UCX if is_disagg else None,
            disable_overlap_scheduler,
        ),
        is_disagg=is_disagg,
    )

    # Mirrors the coefficient in KVCacheManagerV2.__init__.
    needs_extra_index_slots = is_disagg or (
        enable_attention_dp and not disable_overlap_scheduler and pp_size == 1
    )
    admission_bound = max_batch_size * pp_size * (2 if needs_extra_index_slots else 1)

    assert seats >= admission_bound


def test_validate_seq_slot_pool_accepts_sufficient_pool():
    validate_seq_slot_pool_covers_admission(16, Mock(max_admissible_sequences=16))


def test_validate_seq_slot_pool_rejects_undersized_pool():
    with pytest.raises(ValueError, match="smaller than the number of"):
        validate_seq_slot_pool_covers_admission(8, Mock(max_admissible_sequences=16))


def test_validate_seq_slot_pool_ignores_managers_without_a_bound():
    """The V1/C++ manager does not publish one; the check must not fire."""
    manager = Mock(spec=[])
    validate_seq_slot_pool_covers_admission(1, manager)
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
        _enable_disagg_adp_overlap_headroom=True,
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


def _make_kv_cache_creator(disable_overlap_scheduler: bool) -> KvCacheCreator:
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
    c._disable_overlap_scheduler = disable_overlap_scheduler
    # Short-circuit the post-construction max_seq_len fixup.
    c._skip_est = True
    c._get_model_kv_cache_manager_cls = Mock(return_value=Mock())
    c._should_create_separate_draft_kv_cache = Mock(return_value=False)
    c._enable_kv_cache_stats = Mock(return_value=False)
    return c


@pytest.mark.parametrize("disable_overlap_scheduler", [False, True])
def test_kv_cache_manager_receives_the_overlap_flag(disable_overlap_scheduler):
    """The manager sizes its own index pool, but needs the overlap flag to do it.

    The index pool must cover both the retiring cohort and its replacement when
    attention DP runs with the overlap scheduler, otherwise ``_create_kv_cache``
    silently defers admitted requests one at a time (nvbug 6627795). The flag is
    passed rather than the seat-pool size so the manager keeps deriving its
    capacity from ``max_batch_size * pp_size``, which is not comparable with a
    seat pool that also carries the PP multiplier.
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
