# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP seq-slot sizing includes overlap headroom.

Under the overlap scheduler, requests finished in the previous iteration
still hold their sequence slots when the next iteration's
prepare_resources runs, while the capacity scheduler has already dropped
them from its budget (no_schedule_after_state=GENERATION_TO_COMPLETE) and
backfilled their seats. Transient slot demand is therefore
2 * max_batch_size, regardless of whether speculative decoding is enabled.
The headroom is selected from runtime topology rather than model architecture.

compute_max_num_sequences is the single sizing implementation used both
for the executor's SeqSlotManager pool (create_py_executor_instance) and
for the sampler state (create_torch_sampler_args).
"""

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    compute_max_num_sequences,
    create_torch_sampler_args,
    should_enable_adp_dummy_fixes,
    should_enable_adp_overlap_seq_slot_headroom,
    should_enable_non_overlap_adp_forward_intent,
    should_enable_scheduler_aware_adp_dummy,
)
from tensorrt_llm.mapping import Mapping

SIZING_CASES = [
    # (pp_size, disable_overlap, enable_overlap_headroom, expected_factor)
    (1, False, True, 2),
    (1, False, False, 1),
    (1, True, True, 1),
    # Existing PP sizing is preserved regardless of the headroom opt-in.
    (2, False, True, 2),
    (4, False, True, 4),
    (4, True, False, 4),
]


@pytest.mark.parametrize(
    "enable_attention_dp,pp_size,disable_overlap,expected",
    [
        # No cache-transceiver term: the gate no longer looks at disaggregation
        # at all, because nvbug-6627795 reproduced on an aggregated context-only
        # run with no transceiver configured.
        (True, 1, False, True),
        (False, 1, False, False),
        (True, 2, False, False),
        (True, 1, True, False),
    ],
)
def test_adp_overlap_seq_slot_headroom_gate(
    enable_attention_dp, pp_size, disable_overlap, expected
):
    mapping = Mapping(
        world_size=pp_size,
        tp_size=1,
        pp_size=pp_size,
        enable_attention_dp=enable_attention_dp,
    )

    assert should_enable_adp_overlap_seq_slot_headroom(mapping, disable_overlap) is expected


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
