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
``should_enable_overlap_headroom`` is the single gate for that, and it requires
attention DP: ``ADPRouter.exclude_retiring_requests`` is read only inside the
``if self.enable_attention_dp:`` branch of ``_fetch_new_requests``, so outside
attention DP the else-branch subtracts ``len(active_requests)`` with retirees
included and residency is capped at ``max_batch_size * pp_size`` however many
seats exist. It stays off for V1 (the V1 capacity schedulers hardcode
``GENERATION_COMPLETE``) and for hybrid models (SSM state is sized from
``max_batch_size``). "V1" there is the manager the creator *selects*, not the
``use_kv_cache_manager_v2`` request: a plain model with ``max_beam_width > 1`` is
demoted to V1 after the request is honoured, so the gate reads
``resolved_kv_cache_manager_is_v2``. It also stays off for the Qwen-VL models that
keep an MRoPE delta cache, which is sized ``max_num_tokens * pp_size + 1`` yet
indexed by ``py_seq_slot``.

The **index pool** (``KVCacheManagerV2.max_admissible_sequences``) is widened by
a deliberately *broader* predicate -- ``is_disagg or (non-PP and overlap-on)``,
with no attention-DP term -- so it equals the seat pool under attention DP and
runs ahead of it elsewhere. Both reasons for the surplus are real: a request
awaiting its KV transfer holds an index lease and no seat at all (because
``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``), and a
non-ADP overlap run needs the leases without being able to occupy the seats.
Hence ``validate_seq_slot_pool_covers_admission`` requires the index pool to
*cover* the seat pool rather than to equal it. Equalising them by widening the
seat pool is the thing not to do: surplus leases cost page-table rows, surplus
seats cost sampler and speculative-decoding state.

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

from tensorrt_llm._torch.pyexecutor import seq_slot_manager as seq_slot_manager_module
from tensorrt_llm._torch.pyexecutor._util import (
    KvCacheCreator,
    compute_max_num_sequences,
    create_torch_sampler_args,
    is_disagg_enabled,
    kv_cache_manager_v2_incompatible_features,
    resolve_max_num_sequences,
    resolved_kv_cache_manager_is_v2,
    should_enable_adp_dummy_fixes,
    should_enable_non_overlap_adp_forward_intent,
    should_enable_overlap_headroom,
    should_enable_scheduler_aware_adp_dummy,
    validate_seq_slot_pool_covers_admission,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_engine import resolve_mrope_position_deltas_cache
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# (pp_size, disable_overlap, enable_overlap_headroom, expected_factor)
SIZING_CASES = [
    # No PP: aggregated baseline, then the extra micro-batch.
    (1, False, False, 1),
    (1, False, True, 2),
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
        (False, 1, False, True, False, False),
        (True, 1, True, True, False, False),
        (True, 2, False, True, False, False),
        (True, 4, False, True, False, False),
        (True, 1, False, False, False, False),
        (True, 1, False, True, True, False),
        (True, 1, False, False, True, False),
        (False, 1, True, True, False, False),
        (False, 2, False, True, False, False),
        (False, 1, False, False, False, False),
        (False, 1, False, True, True, False),
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


def test_overlap_headroom_gate_requires_attention_dp():
    """Attention DP is the only place the extra seats can be occupied.

    ``ADPRouter.exclude_retiring_requests`` is what lets a replacement cohort be
    admitted while the retiring cohort still holds its seats, and that correction
    is read only inside the ``if self.enable_attention_dp:`` branch of
    ``_fetch_new_requests``. The else-branch subtracts ``len(active_requests)``
    with retirees included, so residency outside attention DP is capped at
    ``max_batch_size * pp_size`` no matter how many seats exist. Widening the
    pool there would allocate sampler state, ``[seats, draft_len, vocab]`` draft
    probabilities and pinned-host block-offset tables for seats that admission
    can never hand out.

    The index pool still widens on this path, so the two pools are unequal here
    by design -- see ``test_sizing_matches_kv_manager_admission_bound``.
    """
    kwargs = dict(kv_cache_manager_is_v2=True, is_hybrid=False)
    with_adp = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=True)
    without_adp = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=False)

    assert should_enable_overlap_headroom(with_adp, False, **kwargs) is True
    assert should_enable_overlap_headroom(without_adp, False, **kwargs) is False


@pytest.mark.parametrize("enable_attention_dp", [False, True])
def test_non_adp_seat_pool_stays_at_max_batch_size(enable_attention_dp):
    """End-to-end: the gate's ADP term reaches the seat count itself.

    Asserting on ``compute_max_num_sequences`` rather than on the predicate is
    what pins the consequence the reviewer asked for -- ``B`` seats for a non-ADP
    overlap run, ``2B`` only under attention DP.
    """
    max_batch_size = 8
    mapping = Mapping(
        world_size=1,
        tp_size=1,
        pp_size=1,
        enable_attention_dp=enable_attention_dp,
    )
    seats = compute_max_num_sequences(
        mapping,
        max_batch_size,
        False,
        enable_overlap_headroom=should_enable_overlap_headroom(
            mapping, False, kv_cache_manager_is_v2=True
        ),
    )
    assert seats == (2 * max_batch_size if enable_attention_dp else max_batch_size)


def test_overlap_headroom_gate_excludes_mrope_delta_cache_models():
    """Qwen-VL sizes its MRoPE delta cache from ``max_num_tokens``, not the seats.

    ``modeling_qwen2vl.py`` and ``modeling_qwen3vl.py`` allocate
    ``max_num_tokens * pp_size + 1`` deltas and then index them by
    ``py_seq_slot``, which only stays in bounds because ``max_batch_size <=
    max_num_tokens``. Doubling the seat pool breaks that, so these models keep the
    single micro-batch until the cache is sized from the seat pool instead.
    """
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=True)
    kwargs = dict(kv_cache_manager_is_v2=True, is_hybrid=False)

    assert should_enable_overlap_headroom(mapping, False, **kwargs) is True
    assert (
        should_enable_overlap_headroom(mapping, False, has_mrope_delta_cache=True, **kwargs)
        is False
    )


@pytest.mark.parametrize("has_mrope_delta_cache", [False, True])
def test_mrope_delta_cache_bounds_the_seat_pool(has_mrope_delta_cache):
    """The arithmetic the exclusion exists for (nvbug 6704146 review).

    With ``max_num_tokens == max_batch_size == 64`` and PP=1 -- the tightest
    configuration ``max_batch_size <= max_num_tokens`` permits -- the delta cache
    holds 65 rows and its top row, index 64, is the reserved dummy slot that
    padded and CUDA-graph requests read. A ``2B`` pool hands out slot 64 (silently
    overwriting the dummy's permanently-zero delta) and then slots 65..127, which
    are past the end. Pinning the seat count against the dummy index is what
    catches a future re-widening; asserting only on the predicate would not.
    """
    max_batch_size = 64
    max_num_tokens = 64
    pp_size = 1
    mapping = Mapping(world_size=1, tp_size=1, pp_size=pp_size, enable_attention_dp=True)
    seats = compute_max_num_sequences(
        mapping,
        max_batch_size,
        False,
        enable_overlap_headroom=should_enable_overlap_headroom(
            mapping,
            False,
            kv_cache_manager_is_v2=True,
            has_mrope_delta_cache=has_mrope_delta_cache,
        ),
    )
    # model_engine._prepare_inputs and cuda_graph_runner both derive this index.
    mrope_dummy_seq_slot = max_num_tokens * pp_size

    if has_mrope_delta_cache:
        assert seats == max_batch_size
        # Highest slot handed out is seats - 1, so every real slot stays below
        # the dummy and inside the max_num_tokens * pp_size + 1 rows.
        assert seats <= mrope_dummy_seq_slot
    else:
        assert seats == 2 * max_batch_size
        assert seats > mrope_dummy_seq_slot


def test_resolve_mrope_delta_cache_finds_the_model_and_draft_buffers():
    """The gate's input is the buffer's presence, not an architecture list.

    ``_pad_batch_seed_mrope_delta_cache`` and the gate must agree on which models
    hold the cache, so both read it through this one resolver -- including the
    draft-model fallback, since speculation puts the multimodal weights there.
    """
    cache = object()

    assert resolve_mrope_position_deltas_cache(SimpleNamespace()) is None
    assert resolve_mrope_position_deltas_cache(None) is None
    assert (
        resolve_mrope_position_deltas_cache(SimpleNamespace(mrope_position_deltas_cache=cache))
        is cache
    )
    assert (
        resolve_mrope_position_deltas_cache(
            SimpleNamespace(draft_model=SimpleNamespace(mrope_position_deltas_cache=cache))
        )
        is cache
    )
    assert (
        resolve_mrope_position_deltas_cache(SimpleNamespace(draft_model=SimpleNamespace())) is None
    )


@pytest.mark.parametrize("max_beam_width,expected_factor", [(1, 2), (2, 1), (4, 1)])
def test_v2_requested_but_beam_search_selects_v1_keeps_b_slots(max_beam_width, expected_factor):
    """``use_kv_cache_manager_v2=True`` is a request, not the manager selected.

    ``KvCacheCreator._validate_or_fallback_kv_cache_manager_v2`` demotes a plain
    V2 manager to ``KVCacheManager`` when ``max_beam_width > 1``, so gating the
    seat pool on the *configured* preference left V2 geometry -- ``2B`` seats,
    plus ``ADPRouter.exclude_retiring_requests`` admitting a replacement cohort
    on top of a retiring one -- on a V1 executor whose capacity scheduler
    hardcodes ``GENERATION_COMPLETE`` and never releases the retirees early. The
    seats would be unusable and the sampler/spec-dec state sized for them
    wasted, so the request must be resolved through
    ``resolved_kv_cache_manager_is_v2`` before it reaches the gate.
    """
    max_batch_size = 8
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, enable_attention_dp=True)
    kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)

    is_v2 = resolved_kv_cache_manager_is_v2(kv_cache_config, max_beam_width)
    assert is_v2 is (max_beam_width == 1)

    seats = compute_max_num_sequences(
        mapping,
        max_batch_size,
        False,
        enable_overlap_headroom=should_enable_overlap_headroom(
            mapping, False, kv_cache_manager_is_v2=is_v2
        ),
    )
    assert seats == expected_factor * max_batch_size


@pytest.mark.parametrize(
    "max_beam_width,has_kv_connector",
    [(1, False), (2, False), (1, True), (4, True)],
)
def test_resolved_v2_agrees_with_the_manager_the_creator_selects(max_beam_width, has_kv_connector):
    """Pin the predicate to the selection instead of restating its conditions.

    A value test over ``max_beam_width`` would still pass if the creator grew a
    third demotion trigger, and the pool would silently go back to being sized
    for a manager the executor does not hold. Driving both off
    ``kv_cache_manager_v2_incompatible_features`` and asserting they agree is
    what makes a new trigger a test failure rather than a regression.

    The ``has_kv_connector`` arms carry their weight in the other direction: a
    connector is served through the pool layout registration path and must not
    demote, so they fail if the fallback it used to force comes back.
    """
    kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    # A plain model: not Gemma4 hybrid (no per-layer head_dim) and not hybrid
    # linear, so the creator falls back rather than raising.
    model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(architectures=["LlamaForCausalLM"], num_hidden_layers=2),
        sparse_attention_config=None,
    )
    creator = object.__new__(KvCacheCreator)
    creator._max_beam_width = max_beam_width
    creator._kv_connector_manager = object() if has_kv_connector else None

    selected = creator._validate_or_fallback_kv_cache_manager_v2(
        KVCacheManagerV2, model_config, kv_cache_config
    )
    resolved = resolved_kv_cache_manager_is_v2(kv_cache_config, max_beam_width)

    assert resolved is issubclass(selected, KVCacheManagerV2)
    assert selected is (KVCacheManagerV2 if resolved else KVCacheManager)


def test_resolved_v2_respects_an_explicit_v1_request():
    """A V1 request stays V1 however compatible the runtime features are."""
    assert resolved_kv_cache_manager_is_v2(KvCacheConfig(use_kv_cache_manager_v2=False), 1) is False
    # "auto" is resolved to a bool during model loading; an unresolved value is
    # not a V2 selection.
    assert (
        resolved_kv_cache_manager_is_v2(KvCacheConfig(use_kv_cache_manager_v2="auto"), 1) is False
    )


def test_v2_incompatible_features_reports_every_trigger():
    """The strings reach the creator's user-facing fallback/rejection message."""
    assert kv_cache_manager_v2_incompatible_features(1) == []
    assert kv_cache_manager_v2_incompatible_features(None) == []
    assert kv_cache_manager_v2_incompatible_features(2) == ["max_beam_width > 1"]


def test_v2_incompatibility_does_not_depend_on_the_kv_connector():
    """A KV connector is served by pool layout registration, not a V1 fallback.

    Asserting this by signature rather than by value is what keeps the demotion
    from creeping back: a connector argument that no longer changes the answer
    would still invite callers to reason as though it did, exactly the
    conflation ``test_overlap_headroom_gate_does_not_depend_on_disaggregation``
    guards against for the seat pool.
    """
    for fn in (kv_cache_manager_v2_incompatible_features, resolved_kv_cache_manager_is_v2):
        params = inspect.signature(fn).parameters
        assert "has_kv_connector" not in params
        assert not [p for p in params if "connector" in p]


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

    The seat gate is the narrower predicate: it additionally requires attention
    DP, because that is the only branch of ``_fetch_new_requests`` that consumes
    ``ADPRouter.exclude_retiring_requests``. So the index pool runs ahead
    wherever the manager widens and the seat gate does not -- a non-ADP overlap
    run, or any disaggregated run. That direction is permitted; what must never
    happen is the reverse, which is what shipped as nvbug 6627795.
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

    assert admission_bound >= seats

    if extra_leases and not enable_overlap_headroom:
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
    """A surplus of index leases is the design, in three independent ways.

    Under disaggregation a request in KV transfer holds its index lease with no
    seat (``SeqSlotManager.prepare_resources`` skips ``DISAGG_GENERATION_INIT``),
    so the manager's ``is_disagg`` term deliberately has no seat-pool
    counterpart. A hybrid model suppresses the seat headroom -- SSM state is
    sized from ``max_batch_size`` -- while a V2 index pool behind it still
    widens. And a non-ADP overlap run widens the index pool while the seat gate
    stays shut, because only the attention-DP branch of ``_fetch_new_requests``
    can admit a replacement before its predecessor releases a seat.

    So the gap is the *ordinary* case on plain TP, not a rare exception, and that
    is accepted rather than engineered away: admission is bounded independently
    at ``max_batch_size * pp_size``, so a surplus lease is never extra
    concurrency, and it is the cheap direction -- spare leases cost page-table
    rows, spare seats would cost sampler state, the eagerly allocated
    ``[seats, draft_len, vocab]`` draft probabilities and the pinned-host
    block-offset tables. Widening the seat pool to make the two numbers equal
    would buy allocation and no admission, which is why the validator checks
    coverage rather than equality.

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
    c._llm_args = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_events_config=None),
        enable_locality_domains=False,
    )
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
        is_draft_model=False,
    )

    with patch(
        "tensorrt_llm._torch.pyexecutor._util._create_kv_cache_manager",
        return_value=None,
    ) as create:
        creator._create_kv_cache_manager(model_engine)

    assert create.call_args.kwargs["disable_overlap_scheduler"] is disable_overlap_scheduler


def _fake_req(request_id, *, init_state=False, transmission_complete=False):
    """Minimal stand-in for the attributes ``prepare_resources`` touches."""
    return SimpleNamespace(
        request_id=request_id,
        seq_slot=None,
        py_seq_slot=None,
        return_perf_metrics=False,
        is_disagg_generation_init_state=init_state,
        is_disagg_generation_transmission_complete=transmission_complete,
    )


def _batch(*requests):
    return SimpleNamespace(all_requests=lambda: list(requests))


def test_seq_slot_manager_skips_disagg_generation_init_without_raising():
    """Regression: the skip branch referenced an unimported ``logger``.

    ``SeqSlotManager.prepare_resources`` logs before ``continue``-ing past a
    ``DISAGG_GENERATION_INIT`` request, but ``seq_slot_manager.py`` never
    imported ``logger``, so reaching this branch raised ``NameError`` instead of
    deferring the request. Every disaggregated generation server takes this path
    on its first look at a request whose KV transfer has not completed.

    Patching the logger and asserting it was called is what pins the branch as
    *executed*: without it an empty body would satisfy the no-raise assertion
    just as well.
    """
    manager = SeqSlotManager(max_num_sequences=4)
    request = _fake_req(7, init_state=True)

    with patch.object(seq_slot_manager_module, "logger") as log:
        manager.prepare_resources(_batch(request))

    log.info.assert_called_once()
    # Deferred, not seated -- and no slot was consumed on its behalf.
    assert request.seq_slot is None
    assert request.py_seq_slot is None
    assert manager.slot_manager.get_slot(7) is None
    assert len(manager.slot_manager.free_slots) == 4


def test_seq_slot_manager_seats_the_request_once_its_transfer_completes():
    """The other half of the skip branch: the same request is seated later.

    Asserting both halves against one request is what shows the skip is a
    deferral rather than a drop.
    """
    manager = SeqSlotManager(max_num_sequences=4)
    request = _fake_req(7, init_state=True)

    with patch.object(seq_slot_manager_module, "logger"):
        manager.prepare_resources(_batch(request))
    assert request.seq_slot is None

    request.is_disagg_generation_init_state = False
    request.is_disagg_generation_transmission_complete = True
    manager.prepare_resources(_batch(request))

    assert request.seq_slot is not None
    assert request.py_seq_slot == request.seq_slot
    assert manager.slot_manager.get_slot(7) == request.seq_slot


def test_seq_slot_turnover_reuses_slots_and_stays_in_range():
    """Full retirement/replacement turnover over a pool sized 2B.

    Walks several generations of requests through the pool, freeing each cohort
    before admitting the next, and checks the two properties that matter for a
    ``py_seq_slot``-indexed buffer: every id stays inside ``[0, capacity)``, and
    ids are distinct among co-resident requests. A leak in ``free_resources``
    shows up as exhaustion; an off-by-one in sizing shows up as an id equal to
    the capacity, which would be an out-of-bounds write into sampler state.
    """
    capacity = 8
    manager = SeqSlotManager(max_num_sequences=capacity)

    next_id = 0
    for _ in range(5):
        cohort = [_fake_req(next_id + i) for i in range(capacity)]
        next_id += capacity
        manager.prepare_resources(_batch(*cohort))

        slots = [r.py_seq_slot for r in cohort]
        assert all(0 <= s < capacity for s in slots), slots
        assert len(set(slots)) == capacity
        assert not manager.slot_manager.free_slots

        for r in cohort:
            manager.free_resources(r)
        assert len(manager.slot_manager.free_slots) == capacity


def test_seq_slot_ids_remain_valid_after_partial_reuse():
    """High slot ids after allocation/reuse, which is where sizing bugs bite.

    ``SlotManager.free_slots`` is a ``set``, so a freed high id is handed back on
    a later ``add_slot`` in an order nothing should depend on. Retiring the
    lower half and admitting a replacement cohort is the turnover shape that
    nvbug 6627795 hit: the replacements coexist with the still-resident upper
    half, so the pool must hold both at once.
    """
    capacity = 8
    manager = SeqSlotManager(max_num_sequences=capacity)

    resident = [_fake_req(i) for i in range(capacity)]
    manager.prepare_resources(_batch(*resident))
    assert not manager.slot_manager.free_slots

    retiring, staying = resident[: capacity // 2], resident[capacity // 2 :]
    for r in retiring:
        manager.free_resources(r)

    replacements = [_fake_req(100 + i) for i in range(capacity // 2)]
    manager.prepare_resources(_batch(*replacements))

    live = staying + replacements
    slots = [r.py_seq_slot for r in live]
    assert all(0 <= s < capacity for s in slots), slots
    assert len(set(slots)) == len(live)
    assert not manager.slot_manager.free_slots
