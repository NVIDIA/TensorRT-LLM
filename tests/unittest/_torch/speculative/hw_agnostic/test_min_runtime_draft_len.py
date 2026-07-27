# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dynamic draft length never drops below the drafter's minimum sustainable length.

One-engine speculation carries cross-iteration drafter state (draft KV cache,
hidden-state pools, per-request context buffers) that is only refreshed while
drafting runs. A draft_len_schedule that resolves to 0 stops those updates while
the target keeps committing tokens, so the drafter later attends to positions
that were allocated but never written and its acceptance rate silently drops.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.utils import get_draft_len_for_batch_size
from tensorrt_llm.llmapi.llm_args import (
    DraftTargetDecodingConfig,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
)

SCHEDULE = {4: 4, 8: 2, 32: 1}


class _ScheduleOnlyEngine:
    """Minimal stand-in exposing what the graph mapping reads."""

    def __init__(self, spec_config, cuda_graph_batch_sizes):
        self.spec_config = spec_config
        self._cuda_graph_batch_sizes = cuda_graph_batch_sizes


class _SpecConfigStub:
    def __init__(self, schedule, min_runtime_draft_len, supports_dynamic=True):
        self.draft_len_schedule = schedule
        self.min_runtime_draft_len = min_runtime_draft_len
        self.spec_dec_mode = _ModeStub(supports_dynamic)


class _ModeStub:
    def __init__(self, supports_dynamic):
        self._supports_dynamic = supports_dynamic

    def support_dynamic_draft_len(self):
        return self._supports_dynamic


@pytest.mark.parametrize(
    "batch_size,expected",
    [(1, 4), (4, 4), (5, 2), (8, 2), (9, 1), (32, 1)],
)
def test_scheduled_draft_lengths_are_unchanged(batch_size, expected):
    assert get_draft_len_for_batch_size(SCHEDULE, batch_size, 4, 1) == expected


@pytest.mark.parametrize("batch_size", [33, 64, 4096])
def test_implicit_tier_above_largest_key_is_floored(batch_size):
    """Above the largest key drafting continues at the floor instead of stopping."""
    assert get_draft_len_for_batch_size(SCHEDULE, batch_size, 4, 1) == 1


def test_explicit_zero_tier_is_floored_without_touching_the_schedule():
    schedule = {4: 4, 8: 0}
    assert get_draft_len_for_batch_size(schedule, 8, 4, 1) == 1
    assert schedule == {4: 4, 8: 0}, "the user's schedule must not be rewritten"


def test_floor_defaults_to_zero_for_callers_that_do_not_opt_in():
    """The two-model drafter path keeps its own semantics, including explicit 0."""
    assert get_draft_len_for_batch_size(SCHEDULE, 33, 4) == 0
    assert get_draft_len_for_batch_size({4: 4, 8: 0}, 8, 4) == 0


def test_no_schedule_uses_max_draft_len():
    assert get_draft_len_for_batch_size(None, 4096, 4, 1) == 4


@pytest.mark.parametrize(
    "spec_config,expected",
    [
        # One-engine: the drafter runs inside the target forward and owns state
        # that only drafting refreshes.
        (
            Eagle3DecodingConfig(
                max_draft_len=8,
                speculative_model="/path/to/draft",
                num_eagle_layers=8,
            ),
            1,
        ),
        (MTPDecodingConfig(max_draft_len=1), 1),
        (
            DraftTargetDecodingConfig(max_draft_len=4, speculative_model="/path/to/draft"),
            1,
        ),
        # Two-model and stateless drafters keep the existing behaviour, including
        # the ability to turn speculation off through the schedule.
        (
            Eagle3DecodingConfig(
                max_draft_len=8,
                speculative_model="/path/to/draft",
                num_eagle_layers=8,
                eagle3_one_model=False,
            ),
            0,
        ),
        (NGramDecodingConfig(max_draft_len=4, max_matching_ngram_size=2), 0),
    ],
)
def test_min_runtime_draft_len_is_one_only_for_one_engine_modes(spec_config, expected):
    assert spec_config.min_runtime_draft_len == expected


def test_vanilla_mtp_opts_out_of_dynamic_draft_len():
    """Vanilla MTP owns one KV-cache layer per module and only runs
    mtp_layers[:runtime_draft_len], so no floor above 0 keeps every module warm."""
    assert not SpeculativeDecodingMode.MTP.support_dynamic_draft_len()
    assert SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL.support_dynamic_draft_len()
    for mode in (
        SpeculativeDecodingMode.EAGLE3_ONE_MODEL,
        SpeculativeDecodingMode.DFLASH,
        SpeculativeDecodingMode.PARD,
        SpeculativeDecodingMode.SA,
        SpeculativeDecodingMode.DRAFT_TARGET_ONE_MODEL,
    ):
        assert mode.support_dynamic_draft_len(), mode


@pytest.mark.parametrize(
    "schedule",
    [{4: 4, 8: 2, 32: 1}, {50: 4, 200: 3, 350: 2}, {4: 4, 8: 0}, {1: 4}],
)
def test_graph_mapping_matches_runtime_resolution(schedule):
    """A captured graph is looked up by an exact (batch_size, draft_len) match, so a
    floor applied to one of the two resolution sites and not the other would silently
    drop those batches to eager execution."""
    graph_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 200, 256, 350, 384, 500]
    max_draft_len = max(schedule.values())
    engine = _ScheduleOnlyEngine(_SpecConfigStub(schedule, 1), graph_batch_sizes)

    mapping = PyTorchModelEngine._compute_dynamic_draft_len_mapping(engine)

    for batch_size in graph_batch_sizes:
        assert mapping[batch_size] == get_draft_len_for_batch_size(
            schedule, batch_size, max_draft_len, 1
        )
    assert 0 not in mapping.values(), "no graph may be captured with drafting disabled"


def test_graph_mapping_is_inert_when_the_mode_opts_out():
    engine = _ScheduleOnlyEngine(
        _SpecConfigStub(SCHEDULE, 1, supports_dynamic=False), [1, 4, 8, 64]
    )
    assert PyTorchModelEngine._compute_dynamic_draft_len_mapping(engine) is None
