# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP seq-slot sizing includes overlap headroom.

Under the overlap scheduler, requests finished in the previous iteration
still hold their sequence slots when the next iteration's
prepare_resources runs, while the capacity scheduler has already dropped
them from its budget (no_schedule_after_state=GENERATION_TO_COMPLETE) and
backfilled their seats. Transient slot demand is therefore one extra
micro-batch worth of slots, regardless of whether speculative decoding is
enabled. The headroom is selected from runtime topology, not model
architecture.

Pipeline parallelism is out of scope: the pool is already sized by pp_size
there, and the ADP router's retiring-request correction is not rank-consistent
because only the last pipeline stage marks generation requests
GENERATION_TO_COMPLETE.

compute_max_num_sequences is the single sizing implementation used both
for the executor's SeqSlotManager pool (create_py_executor_instance) and
for the sampler state (create_torch_sampler_args); resolve_max_num_sequences
is how a consumer obtains it without re-deriving it.
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
)
from tensorrt_llm.mapping import Mapping

_UCX = SimpleNamespace(backend="UCX")

# (pp_size, enable_overlap_headroom, expected_factor)
#
# Disaggregation is absent from this table on purpose -- see
# test_seat_pool_has_no_disagg_term.
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


def test_seat_pool_has_no_disagg_term():
    """The disaggregation 2x reaches the seat pool only through the gate.

    A request awaiting its KV transfer holds an *index* lease and no seat at all:
    ``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``
    requests outright and only seats one once its transmission completes. So the
    sizing function itself must not carry a disaggregation term -- the single
    ``enable_overlap_headroom`` flag is the only way in.

    Asserting on the signature rather than on a return value is deliberate: a
    value test cannot distinguish "the parameter is gone" from "the parameter
    defaults to False", and it is the parameter's *existence* that invites a
    caller to propagate the factor.
    """
    assert "is_disagg" not in inspect.signature(compute_max_num_sequences).parameters
    assert "is_disagg" not in inspect.signature(resolve_max_num_sequences).parameters


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


def _make_kv_cache_creator(enable_overlap_scheduler: bool) -> KvCacheCreator:
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
    c._enable_overlap_scheduler = enable_overlap_scheduler
    # Short-circuit the post-construction max_seq_len fixup.
    c._skip_est = True
    c._get_model_kv_cache_manager_cls = Mock(return_value=Mock())
    c._should_create_separate_draft_kv_cache = Mock(return_value=False)
    c._enable_kv_cache_stats = Mock(return_value=False)
    return c


@pytest.mark.parametrize("enable_overlap_scheduler", [False, True])
def test_kv_cache_manager_receives_the_overlap_flag(enable_overlap_scheduler):
    """The manager sizes its own index pool, but needs the overlap flag to do it.

    The index pool must cover both the retiring cohort and its replacement when
    attention DP runs with the overlap scheduler, otherwise ``_create_kv_cache``
    silently defers admitted requests one at a time (nvbug 6627795). The flag is
    passed rather than the seat-pool size so the manager keeps deriving its
    capacity from ``max_batch_size * pp_size``, which is not comparable with a
    seat pool that also carries the PP multiplier.
    """
    creator = _make_kv_cache_creator(enable_overlap_scheduler)
    model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(is_generation=True)),
    )

    with patch(
        "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
        return_value=None,
    ) as create:
        creator._create_kv_cache_manager(model_engine)

    assert create.call_args.kwargs["enable_overlap_scheduler"] is enable_overlap_scheduler
