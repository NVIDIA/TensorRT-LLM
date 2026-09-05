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
"""CPU tests for the host-side page-table check.

The module under test only depends on numpy, so these run anywhere; no GPU and
no attention backend are involved.
"""

import numpy as np
import pytest

from tensorrt_llm._torch.attention.backends.page_table_check import (
    PLACEHOLDER_PAGE_INDEX,
    check_page_table,
)

PAGE_SIZE = 4


def _sound_table():
    """Two rows of 2 and 3 pages, fully filled except the last page of row 1."""
    return dict(
        pool_indices={0: np.array([1, 2, 3, 4, 5], dtype=np.int32)},
        num_blocks=[2, 3],
        logical_num_blocks=[2, 3],
        kv_lens=[8, 10],
        page_size=PAGE_SIZE,
        pool_size=16,
    )


def test_sound_table_reports_nothing():
    assert check_page_table(**_sound_table()) == []


def test_none_pool_is_skipped():
    args = _sound_table()
    args["pool_indices"] = {0: None}
    assert check_page_table(**args) == []


def test_empty_batch():
    assert (
        check_page_table({0: np.array([], dtype=np.int32)}, [], [], [], PAGE_SIZE, pool_size=16)
        == []
    )


@pytest.mark.parametrize("bad_id", [-1, 16, 999])
def test_page_id_outside_pool(bad_id):
    args = _sound_table()
    args["pool_indices"] = {0: np.array([1, 2, 3, bad_id, 5], dtype=np.int64)}
    problems = check_page_table(**args)
    assert len(problems) == 1
    assert "outside [0, 16)" in problems[0]
    assert "flat index 3" in problems[0]


def test_no_pool_size_skips_range_check():
    args = _sound_table()
    args["pool_indices"] = {0: np.array([1, 2, 3, 999, 5], dtype=np.int64)}
    args["pool_size"] = None
    assert check_page_table(**args) == []


def test_duplicate_page_within_a_row():
    args = _sound_table()
    args["pool_indices"] = {0: np.array([1, 2, 3, 4, 3], dtype=np.int32)}
    problems = check_page_table(**args)
    assert len(problems) == 1
    assert "pool 0 row 1: repeated page id" in problems[0]


def test_duplicate_across_rows_is_not_reported():
    # Rows are checked independently: sharing a page across rows is how prefix
    # reuse works and is not a fault.
    args = _sound_table()
    args["pool_indices"] = {0: np.array([1, 2, 1, 4, 5], dtype=np.int32)}
    assert check_page_table(**args) == []


def test_repeated_placeholder_page_is_allowed():
    # The placeholder marks evicted out-of-window pages; the window mask drops
    # them, so repeats are expected rather than a fault.
    args = _sound_table()
    args["pool_indices"] = {
        0: np.array(
            [PLACEHOLDER_PAGE_INDEX, PLACEHOLDER_PAGE_INDEX, PLACEHOLDER_PAGE_INDEX, 4, 5],
            dtype=np.int32,
        )
    }
    assert check_page_table(**args) == []


def test_index_count_disagrees_with_indptr():
    args = _sound_table()
    args["pool_indices"] = {0: np.array([1, 2, 3], dtype=np.int32)}
    problems = check_page_table(**args)
    assert len(problems) == 1
    assert "3 page ids for an indptr that ends at 5" in problems[0]


@pytest.mark.parametrize(
    "kv_lens,description",
    [
        ([8, 20], "too long for its page count"),
        ([8, 8], "too short for its page count"),
    ],
)
def test_last_page_len_out_of_range(kv_lens, description):
    args = _sound_table()
    args["kv_lens"] = kv_lens
    problems = check_page_table(**args)
    assert len(problems) == 1, description
    assert f"last_page_len outside [1, {PAGE_SIZE}] on rows [1]" in problems[0]


def test_multiple_pools_are_all_checked():
    args = _sound_table()
    args["pool_indices"] = {
        0: np.array([1, 2, 3, 4, 5], dtype=np.int32),
        1: np.array([1, 2, 3, 4, 4], dtype=np.int32),
    }
    problems = check_page_table(**args)
    assert len(problems) == 1
    assert problems[0].startswith("pool 1 row 1")


def test_several_faults_are_all_reported():
    args = _sound_table()
    args["pool_indices"] = {0: np.array([1, 2, 99, 4, 4], dtype=np.int32)}
    args["kv_lens"] = [8, 20]
    problems = check_page_table(**args)
    assert len(problems) == 3
