# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seq-slot pool / sampler sizing must include overlap headroom.

Under the overlap scheduler, requests finished in the previous iteration
still hold their sequence slots when the next iteration's
prepare_resources runs, while the V2 scheduler has already dropped them
from its budget (no_schedule_after_state=GENERATION_TO_COMPLETE) and
backfilled their seats.  Transient slot demand is therefore
2 * max_batch_size; sizing the pool (and the sampler state it indexes)
at max_batch_size alone exhausts it ("No free slots").

compute_max_num_sequences is the single sizing implementation used both
for the executor's SeqSlotManager pool (create_py_executor_instance) and
for the sampler state (create_torch_sampler_args).
"""

import pytest

from tensorrt_llm._torch.pyexecutor._util import (
    compute_max_num_sequences,
    create_torch_sampler_args,
)
from tensorrt_llm.mapping import Mapping

SIZING_CASES = [
    # (pp_size, disable_overlap_scheduler, expected_factor)
    # Overlap without PP: headroom for two in-flight iterations.
    (1, False, 2),
    # No overlap: exact batch size suffices.
    (1, True, 1),
    # With PP the micro-batch count is pp_size, regardless of overlap.
    (2, False, 2),
    (4, True, 4),
]


@pytest.mark.parametrize("pp_size,disable_overlap,expected_factor", SIZING_CASES)
def test_compute_max_num_sequences_includes_overlap_headroom(
    pp_size, disable_overlap, expected_factor
):
    max_batch_size = 8
    mapping = Mapping(world_size=pp_size, tp_size=1, pp_size=pp_size)
    assert (
        compute_max_num_sequences(mapping, max_batch_size, disable_overlap)
        == max_batch_size * expected_factor
    )


@pytest.mark.parametrize("pp_size,disable_overlap,expected_factor", SIZING_CASES)
def test_sampler_max_num_sequences_matches_executor_slot_pool(
    pp_size, disable_overlap, expected_factor
):
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
