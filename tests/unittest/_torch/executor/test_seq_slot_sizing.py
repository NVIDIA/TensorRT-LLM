# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 seq-slot sizing includes overlap headroom.

Under the overlap scheduler, requests finished in the previous iteration
still hold their sequence slots when the next iteration's
prepare_resources runs, while the V2 scheduler has already dropped them
from its budget (no_schedule_after_state=GENERATION_TO_COMPLETE) and
backfilled their seats. Transient slot demand is therefore
2 * max_batch_size. The headroom is intentionally limited to DeepSeek-V4;
other models preserve their established sizing pending separate validation.

compute_max_num_sequences is the single sizing implementation used both
for the executor's SeqSlotManager pool (create_py_executor_instance) and
for the sampler state (create_torch_sampler_args).
"""

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    compute_max_num_sequences,
    create_torch_sampler_args,
    should_enable_dsv4_adp_dummy_fixes,
    should_enable_dsv4_overlap_headroom,
)
from tensorrt_llm.mapping import Mapping

SIZING_CASES = [
    # (pp_size, disable_overlap, enable_overlap_headroom, expected_factor)
    (1, False, True, 2),
    (1, False, False, 1),
    (1, True, True, 1),
    # Existing PP sizing is preserved regardless of the DSv4 opt-in.
    (2, False, True, 2),
    (4, False, True, 4),
    (4, True, False, 4),
]


@pytest.mark.parametrize(
    "model_type,has_spec,is_mtp_one_model,pp_size,disable_overlap,expected",
    [
        ("deepseek_v4", True, True, 1, False, True),
        ("deepseek_v3", True, True, 1, False, False),
        ("deepseek_v4", False, False, 1, False, False),
        ("deepseek_v4", True, False, 1, False, False),
        ("deepseek_v4", True, True, 2, False, False),
        ("deepseek_v4", True, True, 1, True, False),
    ],
)
def test_dsv4_overlap_headroom_gate(
    model_type, has_spec, is_mtp_one_model, pp_size, disable_overlap, expected
):
    spec_config = None
    if has_spec:
        spec_config = Mock()
        spec_config.spec_dec_mode.is_mtp_eagle_one_model.return_value = is_mtp_one_model
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)

    assert (
        should_enable_dsv4_overlap_headroom(model_type, spec_config, mapping, disable_overlap)
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
