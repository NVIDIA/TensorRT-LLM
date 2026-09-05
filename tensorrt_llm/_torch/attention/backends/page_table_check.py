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
"""Host-side structural checks for a paged-attention page table.

Kept free of torch and of the rest of the package so it can be exercised
directly from a unit test: it takes numpy arrays and returns a list of
human-readable problem descriptions, and never raises on malformed input.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

# Page id used as a placeholder for evicted, out-of-window pages. Repeats of it
# inside one row are expected -- the sliding-window mask drops those pages -- so
# the duplicate check ignores it.
PLACEHOLDER_PAGE_INDEX = 0


def check_page_table(
    pool_indices: Dict[int, Optional[np.ndarray]],
    num_blocks: Sequence[int],
    logical_num_blocks: Sequence[int],
    kv_lens: Sequence[int],
    page_size: int,
    pool_size: Optional[int] = None,
    max_rows_reported: int = 8,
) -> List[str]:
    """Return a list of structural problems with a page table, empty if it is sound.

    Checks, per pool:

    * a page id outside ``[0, pool_size)``, which makes the kernel read memory
      no one wrote;
    * the same page claimed twice inside one row, so two positions of the same
      sequence append over each other;
    * a flat index array whose length disagrees with the indptr implied by
      ``num_blocks``.

    And, across rows:

    * a ``last_page_len`` outside ``[1, page_size]``, derived from ``kv_lens``
      and the committed page count;
    Every input is already on the host when a plan is built, so this costs no
    GPU work and no device synchronization.
    """
    problems: List[str] = []
    num_blocks_arr = np.asarray(num_blocks, dtype=np.int64)
    logical_arr = np.asarray(logical_num_blocks, dtype=np.int64)
    kv_lens_arr = np.asarray(kv_lens, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(num_blocks_arr)])

    for pool_id, indices in sorted(pool_indices.items()):
        if indices is None:
            continue
        flat = np.asarray(indices).reshape(-1)
        if pool_size is not None and flat.size:
            out_of_range = np.flatnonzero((flat < 0) | (flat >= pool_size))
            if out_of_range.size:
                worst = int(flat[out_of_range[0]])
                problems.append(
                    f"pool {pool_id}: {out_of_range.size} page ids outside "
                    f"[0, {pool_size}), first at flat index "
                    f"{int(out_of_range[0])} value {worst} "
                    f"(0x{worst & 0xFFFFFFFF:08x})"
                )
        if flat.size != int(starts[-1]):
            problems.append(
                f"pool {pool_id}: {flat.size} page ids for an indptr that ends at {int(starts[-1])}"
            )
            continue
        for row in range(num_blocks_arr.size):
            span = flat[int(starts[row]) : int(starts[row + 1])]
            real = span[span != PLACEHOLDER_PAGE_INDEX]
            if real.size and np.unique(real).size != real.size:
                problems.append(
                    f"pool {pool_id} row {row}: repeated page id in a span of {span.size}"
                )

    if logical_arr.size == kv_lens_arr.size:
        last_page_len = kv_lens_arr - (logical_arr - 1) * page_size
        bad_len = np.flatnonzero((last_page_len < 1) | (last_page_len > page_size))
        if bad_len.size:
            problems.append(
                f"last_page_len outside [1, {page_size}] on rows "
                f"{bad_len.tolist()[:max_rows_reported]}: "
                f"{last_page_len[bad_len].tolist()[:max_rows_reported]}"
            )

    return problems
