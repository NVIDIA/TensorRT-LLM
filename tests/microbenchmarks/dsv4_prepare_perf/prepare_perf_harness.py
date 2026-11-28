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
"""Signal collection for the DeepSeek-V4 block-table preparation host path.

Why dispatch counts rather than wall-clock
-----------------------------------------
The block-table half of ``_prepare_inputs`` is pure host work: on a profiled
GB300 generation worker it issued ~1480 ATen dispatches per iteration of which
only ~5% launched any GPU work, the rest being interpreter, dispatcher and
allocator cost. Wall-clock on that shape is dominated by machine noise (the
same region's per-iteration cost moved by more than the effect size between two
runs of identical code), so a timing gate would either be blind or flaky.

The ATen dispatch count is the quantity the host-side work actually consists of.
It is an integer, it is deterministic for a fixed batch shape, it needs no GPU
clock lock, and it moves for exactly the reasons we want to catch:

* a per-request Python/ATen loop reintroduced where a batched op or a cached
  result was used (count scales with ``num_seqs``),
* a redundant ``IndexMapper.get_copy_index`` walk added back (each walk is
  O(num_seqs) ``select``/``as_strided``/``fill_`` triples),
* a full-buffer ``fill_`` restored where only the untouched padding needs one.

So the gate is a zero-threshold ``==`` against a blessed golden, in the spirit of
the ``discrete`` tests in ``tests/microbenchmarks/attention_perf``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torch.utils._python_dispatch import TorchDispatchMode


class DispatchCounter(TorchDispatchMode):
    """Counts ATen dispatches, optionally excluding a set of op names.

    Runs on CPU-only hosts and inside CUDA streams alike: it intercepts at the
    dispatcher, so it observes the host-side call regardless of where the work
    lands.
    """

    def __init__(self) -> None:
        super().__init__()
        self.total = 0
        self.by_op: Counter = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.total += 1
        # `aten::select.int` -> `select`; keeps the report readable and stable
        # across the overload renames that happen between torch versions.
        self.by_op[func.overloadpacket.__name__] += 1
        return func(*args, **(kwargs or {}))

    def top(self, n: int = 8) -> List[str]:
        return [f"{op}={cnt}" for op, cnt in self.by_op.most_common(n)]


@dataclass
class PrepareSignals:
    """Signals from one block-table preparation. None = not observed here."""

    case_id: str
    num_seqs: int
    num_ratios: int
    # Total ATen dispatches for the whole prepare sequence.
    dispatch_total: Optional[int] = None
    # Dispatches attributable to the shared request->slot mapping walk. This is
    # the signal the mapping memo moves.
    mapping_walks: Optional[int] = None
    per_op: Dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"{self.case_id}: num_seqs={self.num_seqs} ratios={self.num_ratios} "
            f"dispatch_total={self.dispatch_total} mapping_walks={self.mapping_walks}"
        )


def count_mapping_walks(cache_manager) -> "MappingWalkProbe":
    """Context manager counting `IndexMapper.get_copy_index` invocations.

    Counting the *walks* separately from the dispatch total matters because the
    two failure modes are different: an extra walk is a redundancy bug (the same
    mapping recomputed), while a higher dispatch total with the same walk count
    means new per-request work somewhere else.
    """
    return MappingWalkProbe(cache_manager)


class MappingWalkProbe:
    def __init__(self, cache_manager) -> None:
        self._mgr = cache_manager
        self._mapper = cache_manager.index_mapper
        self._orig = None
        self.calls = 0

    def __enter__(self) -> "MappingWalkProbe":
        orig = self._mapper.get_copy_index
        self._orig = orig

        def counting(request_ids, num_contexts, beam_width):
            self.calls += 1
            return orig(request_ids, num_contexts, beam_width)

        # Patch on the instance; the C++ binding object tolerates attribute
        # assignment for this purpose and it is restored on exit.
        try:
            self._mapper.get_copy_index = counting  # type: ignore[assignment]
        except (AttributeError, TypeError):
            # Immutable binding: fall back to "not observed" rather than failing
            # the test for an environment reason.
            self._orig = None
        return self

    def __exit__(self, *exc) -> None:
        if self._orig is not None:
            try:
                self._mapper.get_copy_index = self._orig  # type: ignore[assignment]
            except (AttributeError, TypeError):
                pass

    @property
    def observed(self) -> bool:
        return self._orig is not None


def sliding_block_tables_shape(cache_manager, num_rows: int):
    """Destination shape for `copy_batch_sliding_block_tables`.

    Matches the helper of the same purpose in
    ``tests/unittest/_torch/attention/sparse/deepseek_v4/test_deepseek_v4_cache_manager.py``.
    ``num_rows`` is deliberately a separate argument from the batch size: the
    production buffer is sized by the mapper capacity, not by the batch, which is
    exactly why only the untouched tail needs the BAD_PAGE_INDEX fill.
    """
    from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
        DEEPSEEK_V4_SLIDING_ATTENTION,
    )

    return (
        cache_manager.num_local_layers,
        len(DEEPSEEK_V4_SLIDING_ATTENTION),
        num_rows,
        cache_manager.max_blocks_per_seq,
    )
