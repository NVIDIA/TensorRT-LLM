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

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import (
    BufferRuns,
    TransferWindow,
    merge_runs,
    page_runs,
)
from tensorrt_llm._utils import nvtx_range


class RunsMapper:
    """Copies what ``layout_mapping.map_buffers`` gives one pool pair from every source slot to the
    destination slot paired with it.

    Runs alike on every page (``BufferRuns.fixed``) are merged once and added to each slot's base.
    The others depend on the token positions each page has written (``layout_mapping.page_runs``);
    buffers whose runs differ only by where the buffer lies share one computation per page.

    Args:
        runs: What the source sends of each buffer of the pool pair.

    Raises:
        ValueError: If a buffer's runs list unequal numbers of offsets and sizes.
    """

    def __init__(self, runs: Sequence[BufferRuns]):
        for r in runs:
            if not r.src_offsets.size == r.dst_offsets.size == r.sizes.size:
                raise ValueError(
                    f"runs need as many offsets as sizes: {r.src_offsets.size} source, "
                    f"{r.dst_offsets.size} destination, {r.sizes.size} sizes"
                )
        fixed = [r for r in runs if r.fixed]
        if fixed:
            src_offsets, dst_offsets, sizes = merge_runs(
                np.concatenate([r.src_offsets for r in fixed]),
                np.concatenate([r.dst_offsets for r in fixed]),
                np.concatenate([r.sizes for r in fixed]),
            )
        else:
            src_offsets = dst_offsets = sizes = np.zeros(0, dtype=np.int64)
        # The runs grouped by size, so each group's fragments share one size.
        self._groups = [
            (int(size), src_offsets[sizes == size], dst_offsets[sizes == size])
            for size in np.unique(sizes)
        ]
        # Runs that move by page: one template per shape, its runs relative to its first, and
        # where each buffer of that shape starts on either side.
        shapes: Dict[tuple, Tuple[BufferRuns, List[int], List[int]]] = {}
        for r in runs:
            if r.fixed or r.sizes.size == 0:
                continue
            src_start, dst_start = int(r.src_offsets[0]), int(r.dst_offsets[0])
            template = dataclasses.replace(
                r, src_offsets=r.src_offsets - src_start, dst_offsets=r.dst_offsets - dst_start
            )
            key = (
                r.by_position,
                r.sizes.tobytes(),
                template.src_offsets.tobytes(),
                template.dst_offsets.tobytes(),
            )
            entry = shapes.setdefault(key, (template, [], []))
            entry[1].append(src_start)
            entry[2].append(dst_start)
        self._by_page = [
            (template, np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64))
            for template, src, dst in shapes.values()
        ]

    @property
    def bytes_per_slot(self) -> int:
        """Bytes moved per slot or block by the runs alike on every page."""
        return sum(size * src.size for size, src, _ in self._groups)

    @property
    def runs_per_slot(self) -> int:
        """Fragments, and so transfer descriptors, per slot or block of the runs alike on every
        page."""
        return sum(src.size for _, src, _ in self._groups)

    @nvtx_range("RunsMapper.map")
    def map(
        self,
        src_bases: np.ndarray,
        dst_bases: np.ndarray,
        pages: Optional[np.ndarray] = None,
        window: Optional[TransferWindow] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """The fragments that copy the runs from each source slot starting at ``src_bases[i]`` to
        the destination slot starting at ``dst_bases[i]``: per run size, the source and
        destination pointers of every run of that size, slot by slot, and the size.

        Args:
            src_bases: Base address of each source slot.
            dst_bases: Base address of the destination slot paired with each.
            pages: Position of each slot pair's page among the request's pages, from 0.
            window: What the transfer covers of the request.

        Raises:
            ValueError: If the two sides list different numbers of slots, or runs moving by page
                lack ``pages`` for every slot or ``window``.
        """
        src_bases = np.asarray(src_bases, dtype=np.int64)
        dst_bases = np.asarray(dst_bases, dtype=np.int64)
        if src_bases.size != dst_bases.size:
            raise ValueError(
                f"Number of regions of src({src_bases.size}) and dst({dst_bases.size}) must match"
            )
        out = [
            (np.add.outer(src_bases, src).ravel(), np.add.outer(dst_bases, dst).ravel(), size)
            for size, src, dst in self._groups
        ]
        if not self._by_page:
            return out
        if pages is None or window is None or np.asarray(pages).size != src_bases.size:
            raise ValueError(
                "runs moving by token position need the page of every slot pair and the "
                "transfer's window"
            )
        parts: Dict[int, Tuple[list, list]] = {}
        for src_base, dst_base, page in zip(src_bases, dst_bases, np.asarray(pages)):
            for template, src_starts, dst_starts in self._by_page:
                src, dst, sizes = page_runs(template, int(page), window)
                for size in np.unique(sizes):
                    pick = sizes == size
                    group = parts.setdefault(int(size), ([], []))
                    group[0].append(np.add.outer(src_base + src_starts, src[pick]).ravel())
                    group[1].append(np.add.outer(dst_base + dst_starts, dst[pick]).ravel())
        out.extend(
            (np.concatenate(src), np.concatenate(dst), size)
            for size, (src, dst) in sorted(parts.items())
        )
        return out
