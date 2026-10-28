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
"""The one mapper of the native transfer: byte runs within a pool slot, from ``layout_mapping``,
added to each block's or slot's base pointers."""

from typing import List, Tuple

import numpy as np

from tensorrt_llm._utils import nvtx_range


class RunsMapper:
    """Copies run ``i``, ``sizes[i]`` bytes from ``src_offsets[i]`` of every source slot to
    ``dst_offsets[i]`` of the destination slot paired with it.

    Args:
        src_offsets: Byte offsets within a source pool slot.
        dst_offsets: Byte offsets within a destination pool slot.
        sizes: Run lengths in bytes.
    """

    def __init__(self, src_offsets: np.ndarray, dst_offsets: np.ndarray, sizes: np.ndarray):
        src_offsets = np.asarray(src_offsets, dtype=np.int64)
        dst_offsets = np.asarray(dst_offsets, dtype=np.int64)
        sizes = np.asarray(sizes, dtype=np.int64)
        if not src_offsets.size == dst_offsets.size == sizes.size:
            raise ValueError(
                f"runs need as many offsets as sizes: {src_offsets.size} source, "
                f"{dst_offsets.size} destination, {sizes.size} sizes"
            )
        # The runs grouped by size, so each group's fragments share one size.
        self._groups = [
            (int(size), src_offsets[sizes == size], dst_offsets[sizes == size])
            for size in np.unique(sizes)
        ]

    @property
    def bytes_per_slot(self) -> int:
        """Bytes moved per slot or block."""
        return sum(size * src.size for size, src, _ in self._groups)

    @property
    def runs_per_slot(self) -> int:
        """Fragments, and so transfer descriptors, per slot or block."""
        return sum(src.size for _, src, _ in self._groups)

    @nvtx_range("RunsMapper.map")
    def map(
        self, src_bases: np.ndarray, dst_bases: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """The fragments that copy the runs from each source slot starting at ``src_bases[i]`` to
        the destination slot starting at ``dst_bases[i]``: per run size, the source and
        destination pointers of every run of that size, slot by slot, and the size.

        Raises:
            ValueError: If the two sides list different numbers of slots.
        """
        src_bases = np.asarray(src_bases, dtype=np.int64)
        dst_bases = np.asarray(dst_bases, dtype=np.int64)
        if src_bases.size != dst_bases.size:
            raise ValueError(
                f"Number of regions of src({src_bases.size}) and dst({dst_bases.size}) must match"
            )
        return [
            (np.add.outer(src_bases, src).ravel(), np.add.outer(dst_bases, dst).ravel(), size)
            for size, src, dst in self._groups
        ]
