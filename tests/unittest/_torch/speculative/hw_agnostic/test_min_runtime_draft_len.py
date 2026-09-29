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
"""Draft lengths resolved from draft_len_schedule are floored at min_runtime_draft_len."""

import pytest

from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.utils import get_draft_len_for_batch_size
from tensorrt_llm.llmapi.llm_args import (
    DraftTargetDecodingConfig,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
)

SCHEDULE = {4: 4, 8: 2, 32: 1}


@pytest.mark.parametrize(
    "schedule,batch_size,min_draft_len,expected",
    [
        (SCHEDULE, 1, 1, 4),
        (SCHEDULE, 8, 1, 2),
        (SCHEDULE, 32, 1, 1),
        # The implicit tier above the largest key and an explicit 0 are floored.
        (SCHEDULE, 33, 1, 1),
        (SCHEDULE, 4096, 1, 1),
        ({4: 4, 8: 0}, 8, 1, 1),
        # A floor of 0 keeps the schedule as written.
        (SCHEDULE, 33, 0, 0),
        ({4: 4, 8: 0}, 8, 0, 0),
        (None, 4096, 1, 4),
    ],
)
def test_get_draft_len_for_batch_size_applies_floor(schedule, batch_size, min_draft_len, expected):
    assert get_draft_len_for_batch_size(schedule, batch_size, 4, min_draft_len) == expected


@pytest.mark.parametrize(
    "spec_config,expected",
    [
        (Eagle3DecodingConfig(max_draft_len=4, speculative_model="/path/to/draft"), 1),
        (MTPDecodingConfig(max_draft_len=1), 1),
        (DraftTargetDecodingConfig(max_draft_len=4, speculative_model="/path/to/draft"), 1),
        (NGramDecodingConfig(max_draft_len=4, max_matching_ngram_size=2), 0),
    ],
)
def test_min_runtime_draft_len_is_one_for_one_engine_modes(spec_config, expected):
    assert spec_config.min_runtime_draft_len == expected


def test_shared_target_kv_drafter_needs_no_floor():
    spec_config = MTPDecodingConfig(max_draft_len=1)
    spec_config._use_shared_kv_cache = True
    assert spec_config.min_runtime_draft_len == 0


def test_vanilla_mtp_opts_out_of_dynamic_draft_len():
    assert not SpeculativeDecodingMode.MTP.support_dynamic_draft_len()
    assert SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL.support_dynamic_draft_len()
