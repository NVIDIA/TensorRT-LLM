# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seq-slot pool sizing.

Two independent reasons the pool must exceed max_batch_size, both selected
from runtime topology rather than model architecture:

Attention-DP overlap headroom. Requests finished in the previous iteration
still hold their sequence slots when the next iteration's prepare_resources
runs, while the V2 scheduler has already dropped them from its budget
(no_schedule_after_state=GENERATION_TO_COMPLETE) and backfilled their seats.
Transient slot demand is therefore twice max_batch_size, whether or not
speculative decoding is enabled.

Disaggregated serving. On a generation server the admission bound is
KVCacheManagerV2's IndexMapper rather than the seat pool, and it is sized at
twice max_num_sequences. Unlike the headroom above, this holds regardless of
attention-DP, overlap, or PP.

compute_max_num_sequences is the single sizing implementation, used for the
executor's SeqSlotManager pool and for the sampler state. Every other
slot-indexed buffer follows the same number, since py_seq_slot indexes them
all.
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    compute_max_num_sequences,
    create_torch_sampler_args,
    is_disagg_enabled,
    should_enable_disagg_adp_overlap_headroom,
    should_enable_dsv4_adp_dummy_fixes,
    validate_seq_slot_pool_covers_admission,
)
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

SIZING_CASES = [
    # (pp_size, disable_overlap, enable_overlap_headroom, expected_factor)
    (1, False, True, 2),
    (1, False, False, 1),
    (1, True, True, 1),
    # PP sizing is independent of the headroom opt-in.
    (2, False, True, 2),
    (4, False, True, 4),
    (4, True, False, 4),
]


@pytest.mark.parametrize(
    "enable_attention_dp,is_disagg,pp_size,disable_overlap,expected",
    [
        (True, True, 1, False, True),
        (False, True, 1, False, False),
        (True, False, 1, False, False),
        (True, True, 2, False, False),
        (True, True, 1, True, False),
    ],
)
def test_disagg_adp_overlap_headroom_gate(
    enable_attention_dp, is_disagg, pp_size, disable_overlap, expected
):
    mapping = Mapping(
        world_size=pp_size,
        tp_size=1,
        pp_size=pp_size,
        enable_attention_dp=enable_attention_dp,
    )
    cache_config = CacheTransceiverConfig(backend="NIXL") if is_disagg else None

    assert (
        should_enable_disagg_adp_overlap_headroom(mapping, cache_config, disable_overlap)
        is expected
    )


@pytest.mark.parametrize(
    "model_type,pp_size,expected",
    [
        ("deepseek_v4", 1, True),
        ("deepseek_v3", 1, False),
        ("deepseek_v4", 2, False),
    ],
)
def test_dsv4_adp_dummy_fix_gate(model_type, pp_size, expected):
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert should_enable_dsv4_adp_dummy_fixes(model_type, mapping) is expected


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


@pytest.mark.parametrize(
    "cache_transceiver_config,expected",
    [
        (None, False),
        (Mock(backend=None), False),
        (Mock(backend="NIXL"), True),
    ],
)
def test_is_disagg_enabled(cache_transceiver_config, expected):
    assert is_disagg_enabled(cache_transceiver_config) is expected


# (pp_size, enable_overlap_headroom, expected_factor). The factor is relative
# to max_batch_size and, for disagg, must cover the IndexMapper's 2x.
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

    Turning the overlap scheduler off removes the terminal-slot race but not
    the transfer/generate overlap that the 2x coefficient exists for.
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


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("is_disagg", [False, True])
def test_sizing_matches_kv_manager_admission_bound(pp_size, is_disagg):
    """The two coefficients are computed independently; hold them in step.

    KVCacheManagerV2 derives its own admission bound from max_batch_size,
    pp_size and is_disagg. Assert the two expressions against each other
    rather than against a literal, so a drift in either one fails here.
    """
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)

    seats = compute_max_num_sequences(
        mapping, max_batch_size, disable_overlap_scheduler=False, is_disagg=is_disagg
    )
    admission_bound = max_batch_size * pp_size * (2 if is_disagg else 1)

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


@pytest.mark.parametrize("slot_factor", [1, 2])
def test_sampler_uses_executor_slot_pool_capacity(slot_factor):
    max_batch_size = 8
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1)
    max_num_sequences = max_batch_size * slot_factor
    args = create_torch_sampler_args(
        mapping,
        max_seq_len=1024,
        max_batch_size=max_batch_size,
        speculative_config=None,
        max_beam_width=1,
        disable_overlap_scheduler=False,
        enable_async_worker=False,
        enable_speculative_beam_history_d2h=False,
        max_num_sequences=max_num_sequences,
    )
    assert args.max_num_sequences == max_num_sequences
