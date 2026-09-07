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
architecture -- except that hybrid/SSM architectures are excluded, because
their state-slot pool is not sized from this number.

That extra generation is **additive** in pp_size, not multiplicative:
pipeline depth already accounts for the pp_size micro-batches structurally
in flight, and the overlap deferral is one iteration on top of them. So the
pool is (pp_size + 1) * max_batch_size, which at pp_size == 1 coincides with
the historical 2 * max_batch_size.

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
    should_enable_adp_overlap_seq_slot_headroom,
    should_enable_non_overlap_adp_forward_intent,
    should_enable_scheduler_aware_adp_dummy,
    validate_seq_slot_pool_covers_admission,
)
from tensorrt_llm.mapping import Mapping

# (pp_size, disable_overlap, enable_overlap_headroom, is_disagg, expected_factor)
#
# The terms are additive, not multiplicative: pipeline depth costs pp_size
# micro-batches of seats, and the overlap deferral costs exactly one more
# generation on top -- not one more per stage. At pp_size == 1 the two readings
# coincide at 2x, which is why the headroom used to be expressible as a factor of
# 2; the pp>1 rows are where they part company (5x, not 8x, at pp=4).
SIZING_CASES = [
    # No PP. Every cell here is unchanged by this PR.
    (1, False, False, False, 1),
    (1, False, True, False, 2),
    (1, True, True, False, 1),
    (1, False, False, True, 2),
    (1, False, True, True, 2),
    # PP without the headroom: unchanged.
    (4, False, False, False, 4),
    (4, True, True, False, 4),
    (2, False, False, True, 4),
    (4, False, False, True, 8),
    # PP with the headroom: the one intended behaviour change (was pp_size).
    (2, False, True, False, 3),
    (4, False, True, False, 5),
    # Disagg dominates the additive term rather than compounding with it: the two
    # coefficients each cover one extra cohort of in-flight sequences.
    (4, False, True, True, 8),
]


@pytest.mark.parametrize(
    "enable_attention_dp,pp_size,disable_overlap,is_hybrid,expected",
    [
        # No cache-transceiver term: the gate no longer looks at disaggregation
        # at all, because nvbug-6627795 reproduced on an aggregated context-only
        # run with no transceiver configured.
        (True, 1, False, False, True),
        (False, 1, False, False, False),
        (True, 1, True, False, False),
        # Pipeline parallelism is in scope. The extra seats are additive in
        # pp_size and are only spendable because the ADP router now derives the
        # retiring-request count identically on every stage.
        (True, 2, False, False, True),
        (True, 4, False, False, True),
        # Hybrid/SSM architectures are excluded: MambaHybridCacheManagerV2 sizes
        # its state-index pool from max_batch_size alone, so an extra seat would
        # have no state slot behind it.
        (True, 1, False, True, False),
        (True, 4, False, True, False),
    ],
)
def test_adp_overlap_seq_slot_headroom_gate(
    enable_attention_dp, pp_size, disable_overlap, is_hybrid, expected
):
    mapping = Mapping(
        world_size=pp_size,
        tp_size=1,
        pp_size=pp_size,
        enable_attention_dp=enable_attention_dp,
    )

    assert (
        should_enable_adp_overlap_seq_slot_headroom(
            mapping, disable_overlap, is_hybrid=is_hybrid
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


@pytest.mark.parametrize(
    "pp_size,disable_overlap,enable_overlap_headroom,is_disagg,expected_factor", SIZING_CASES
)
def test_compute_max_num_sequences_scopes_overlap_headroom(
    pp_size, disable_overlap, enable_overlap_headroom, is_disagg, expected_factor
):
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert (
        compute_max_num_sequences(
            mapping,
            max_batch_size,
            disable_overlap,
            enable_overlap_headroom=enable_overlap_headroom,
            is_disagg=is_disagg,
        )
        == max_batch_size * expected_factor
    )


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
    pool they share indices with -- the very skew this number exists to remove.
    The third row is that branch, and it must still land on the headroom value.
    """
    engine = SimpleNamespace(
        max_num_seq_slots=engine_seats,
        _enable_adp_overlap_seq_slot_headroom=True,
    )
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=True)

    assert (
        resolve_max_num_sequences(
            engine,
            mapping,
            8,
            False,
            False,
            max_num_sequences=explicit,
        )
        == expected
    )


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


class _FakeManager:
    """Stands in for a KV cache manager that publishes its seating capacity."""

    def __init__(self, max_admissible_sequences):
        self.max_admissible_sequences = max_admissible_sequences


def test_validator_accepts_the_matching_pair():
    validate_seq_slot_pool_covers_admission(16, _FakeManager(16))


@pytest.mark.parametrize("admissible", [8, 32])
def test_validator_is_two_sided(admissible):
    """Both directions of the skew are bugs, and both have shipped.

    A one-sided ``seats >= admissible`` guard is what let nvbug 6627795 through:
    the seat pool grew to 2B while the index pool stayed at B+1, which satisfies
    the one-sided form and silently defers admitted requests one at a time. The
    other direction -- index pool larger than the seat pool -- admits a request
    that cannot be seated and raises inside the executor's event loop.
    """
    with pytest.raises(ValueError, match="sequence-slot pool"):
        validate_seq_slot_pool_covers_admission(16, _FakeManager(admissible))


@pytest.mark.parametrize(
    "manager",
    [
        None,  # non-generation models have no KV cache manager
        SimpleNamespace(),  # V1 sizes its index pool by an unrelated rule
    ],
)
def test_validator_skips_managers_that_do_not_publish_capacity(manager):
    """Opt-in, not guessed at.

    A manager whose index pool is sized by some other rule -- V1, which receives
    none of this plumbing and re-derives from bare max_batch_size -- must not be
    compared against a number it never consumed.

    Hybrid is deliberately *not* an example here: MambaHybridCacheManagerV2
    derives from KVCacheManagerV2 and therefore does publish
    max_admissible_sequences, so it *is* validated. That comparison holds
    because the headroom is withheld from hybrid on both sides -- seats are
    B*pp (or 2*B*pp under disagg) and, with max_num_seq_slots withheld, its
    index pool lands on exactly the same number. What hybrid does not receive
    is the seat pool, not the check.
    """
    validate_seq_slot_pool_covers_admission(16, manager)


def _make_kv_cache_creator(max_num_seq_slots) -> KvCacheCreator:
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
    # Short-circuit the post-construction max_seq_len fixup.
    c._skip_est = True
    c._get_model_kv_cache_manager_cls = Mock(return_value=Mock())
    c._should_create_separate_draft_kv_cache = Mock(return_value=False)
    c._enable_kv_cache_stats = Mock(return_value=False)
    c._model_engine = SimpleNamespace(max_num_seq_slots=max_num_seq_slots)
    return c


@pytest.mark.parametrize("max_num_seq_slots", [8, 16, None])
def test_kv_cache_manager_receives_executor_seq_slot_pool(max_num_seq_slots):
    """The seat pool size must reach the KV cache manager verbatim.

    The manager sizes its IndexMapper from this number so that every sequence
    the executor can admit is guaranteed an index. Recomputing the coefficient
    inside the manager would let the two pools drift apart (nvbug 6627795).
    """
    creator = _make_kv_cache_creator(max_num_seq_slots)
    model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(is_generation=True)),
        max_num_seq_slots=max_num_seq_slots,
    )

    with patch(
        "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
        return_value=None,
    ) as create:
        creator._create_kv_cache_manager(model_engine)

    assert create.call_args.kwargs["max_num_seq_slots"] == max_num_seq_slots


def test_draft_manager_uses_the_target_engines_seq_slot_pool():
    """A draft engine's own seat count must not size the draft index pool.

    The executor has a single SeqSlotManager, sized from the target engine. Two-
    model speculative decoding builds the draft KV cache manager by passing the
    *draft* engine to the same helper; if that engine's (smaller) number were
    used, the draft manager's IndexMapper would become the new bottleneck and
    reintroduce the silent deferral this sizing exists to prevent.
    """
    creator = _make_kv_cache_creator(16)
    draft_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=SimpleNamespace(is_generation=True)),
        max_num_seq_slots=8,
    )

    with patch(
        "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
        return_value=None,
    ) as create:
        creator._create_kv_cache_manager(draft_engine)

    assert create.call_args.kwargs["max_num_seq_slots"] == 16
