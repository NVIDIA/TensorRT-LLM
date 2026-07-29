# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seq-slot pool and slot-indexed state include overlap headroom.

Under the overlap scheduler, requests finished in the previous iteration
still hold their sequence slots when the next iteration's
prepare_resources runs, while the V2 scheduler has already dropped them
from its budget (no_schedule_after_state=GENERATION_TO_COMPLETE) and
backfilled their seats. Transient slot demand is therefore
2 * max_batch_size for every non-PP overlap configuration.

compute_max_num_sequences is the single sizing implementation used both
for the executor's SeqSlotManager pool (create_py_executor_instance) and
for the sampler state (create_torch_sampler_args).
"""

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    compute_max_num_sequences,
    create_torch_sampler_args,
    should_enable_adp_dummy_fixes,
)
from tensorrt_llm.mapping import Mapping

SIZING_CASES = [
    # (pp_size, disable_overlap, expected_factor)
    (1, False, 2),
    (1, True, 1),
    # PP already sizes the pool for its micro-batch count.
    (2, False, 2),
    (4, False, 4),
    (4, True, 4),
]


@pytest.mark.parametrize("pp_size,expected", [(1, True), (2, False)])
def test_adp_dummy_fix_gate(pp_size, expected):
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert should_enable_adp_dummy_fixes(mapping) is expected


@pytest.mark.parametrize("pp_size,disable_overlap,expected_factor", SIZING_CASES)
def test_compute_max_num_sequences_includes_overlap_headroom(
    pp_size, disable_overlap, expected_factor
):
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert (
        compute_max_num_sequences(
            mapping,
            max_batch_size,
            disable_overlap,
        )
        == max_batch_size * expected_factor
    )


@pytest.mark.parametrize("pp_size,disable_overlap,expected_factor", SIZING_CASES)
def test_sampler_uses_executor_slot_pool_capacity(pp_size, disable_overlap, expected_factor):
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    args = create_torch_sampler_args(
        mapping,
        max_seq_len=1024,
        max_batch_size=max_batch_size,
        speculative_config=None,
        max_beam_width=1,
        disable_overlap_scheduler=disable_overlap,
        enable_async_worker=False,
        enable_speculative_beam_history_d2h=False,
    )
    assert args.max_num_sequences == compute_max_num_sequences(
        mapping, max_batch_size, disable_overlap
    )
    assert args.max_num_sequences == max_batch_size * expected_factor
