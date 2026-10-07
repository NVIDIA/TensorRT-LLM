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
"""Host side of the one-model speculative decoding step's copy kernels (``spec_step_copies_kernel``).

``SlotScatter`` moves a step's per-row outputs into the sampler's slot stores (one launch for the sampler's four
``index_copy_``). ``StepInputStage`` writes a decode step's per-step inputs with one launch: the overlap scheduler's
gathers from those stores, host-to-device copies whose values the kernel reads from a pinned host record, and KV cache
block-offset copies. Each kernel takes every size and address as a launch argument, so it is compiled once per process,
on the first launch, which must happen outside CUDA-graph capture. The kernels are called through TVM-FFI: a launch
costs a few microseconds of host time.

Callers use them only where ``is_supported()`` holds. A call whose arguments are outside what the kernel covers launches
and stages nothing and returns False; the caller then does that work itself.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from ...._utils import get_sm_version, prefer_pinned
from ...cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE


def is_supported() -> bool:
    """Whether the kernels run on the current device: the SM 100 family (SM 100, 103, 107) with the CuTe DSL installed.

    The kernels are plain SIMT code (thread and block indices, global loads and stores), with nothing specific to an
    architecture.
    """
    # TODO: validated on SM 100 (B200 / GB200) only; SM 103 (GB300) and SM 107 (Rubin) are untested.
    return (
        IS_CUTLASS_DSL_AVAILABLE
        and torch.cuda.is_available()
        and get_sm_version() in (100, 103, 107)
    )


def _is_i32(t: torch.Tensor) -> bool:
    """A contiguous CUDA int32 tensor."""
    return t.is_cuda and t.dtype == torch.int32 and t.is_contiguous()


def _is_pinned_i32(t: torch.Tensor) -> bool:
    """A contiguous pinned host int32 tensor."""
    return not t.is_cuda and t.dtype == torch.int32 and t.is_contiguous() and t.is_pinned()


def _compile(name: str, args: tuple):
    """Compile kernel ``name`` for its scalar ``args`` with TVM-FFI; a call then takes ``args`` and the CUDA stream
    handle to launch on."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"spec_step_copies: the {name} kernel must run once outside CUDA-graph capture first "
            "(it compiles on its first launch)."
        )
    import cutlass.cute as cute

    from . import spec_step_copies_kernel as kernel

    return cute.compile(
        getattr(kernel, name), *args, cute.runtime.make_fake_stream(), options="--enable-tvm-ffi"
    )


def _gather_shapes_ok(
    store_next_new_tokens: torch.Tensor,
    store_next_draft: torch.Tensor,
    store_lens: torch.Tensor,
    rows: int,
    tokens_per_row: int,
    draft_width: int,
) -> bool:
    """The overlap gather's stores are [width, slots, 1], [slots, draft width] and [slots], wide enough for a row."""
    num_slots = store_lens.numel()
    return (
        store_lens.dim() == 1
        and store_next_new_tokens.dim() == 3
        and store_next_new_tokens.shape[1:] == (num_slots, 1)
        and store_next_draft.dim() == 2
        and store_next_draft.shape[0] == num_slots
        and 0 < tokens_per_row <= store_next_new_tokens.shape[0]
        and 0 <= draft_width < tokens_per_row
        and draft_width <= store_next_draft.shape[1]
        and rows >= 0
    )


class SlotScatter:
    """The speculative sampler's store update as one kernel.

    ``store[..., slots[r]] = outputs[row_begin + r]`` for every row ``r``, each row padded with zeros or cut to its
    store's width. A row's ``new_tokens`` at or past its ``new_tokens_lens`` are stored as zeros: the forward writes
    only column 0 of a context row, and readers of the store stop at that length.
    """

    _kernel: ClassVar[Any] = None

    def scatter(
        self,
        outputs: dict[str, torch.Tensor],
        row_begin: int,
        rows: int,
        slots: torch.Tensor,
        store_new_tokens: torch.Tensor,
        store_next_new_tokens: torch.Tensor,
        store_lens: torch.Tensor,
        store_next_draft: torch.Tensor,
    ) -> bool:
        """Launch the store update on the current stream.

        Args:
            outputs: The forward's ``new_tokens`` [rows_total, a], ``next_new_tokens`` [rows_total, b],
                ``new_tokens_lens`` [rows_total] and ``next_draft_tokens`` [rows_total, c], int32.
            row_begin: The first row of ``outputs`` to move.
            rows: The number of rows to move.
            slots: The distinct slot of each moved row, int32 [>= rows] on the device.
            store_new_tokens: int32 [width, num_slots, 1].
            store_next_new_tokens: int32 [width, num_slots, 1].
            store_lens: int32 [num_slots].
            store_next_draft: int32 [num_slots, width].

        Returns:
            False, launching nothing, when an argument is outside what the kernel covers (a non-contiguous or
            non-int32 tensor, a store of another layout, fewer output rows or slots than ``rows``); True otherwise.
        """
        if rows == 0:
            return True
        new_tokens = outputs["new_tokens"]
        next_new_tokens = outputs["next_new_tokens"]
        lens = outputs["new_tokens_lens"]
        next_draft = outputs["next_draft_tokens"]
        num_slots = store_lens.numel()
        tensors = (
            new_tokens,
            next_new_tokens,
            lens,
            next_draft,
            slots,
            store_new_tokens,
            store_next_new_tokens,
            store_lens,
            store_next_draft,
        )
        if not (
            all(_is_i32(t) for t in tensors)
            and new_tokens.dim() == next_new_tokens.dim() == next_draft.dim() == 2
            and lens.dim() == store_lens.dim() == 1
            and min(t.shape[0] for t in (new_tokens, next_new_tokens, lens, next_draft))
            >= row_begin + rows
            and row_begin >= 0
            and slots.numel() >= rows
            and store_new_tokens.dim() == store_next_new_tokens.dim() == 3
            and store_new_tokens.shape[1:] == (num_slots, 1)
            and store_next_new_tokens.shape[1:] == (num_slots, 1)
            and store_next_draft.dim() == 2
            and store_next_draft.shape[0] == num_slots
        ):
            return False
        columns = max(
            store_new_tokens.shape[0], store_next_new_tokens.shape[0], store_next_draft.shape[1], 1
        )
        args = (
            new_tokens.data_ptr(),
            next_new_tokens.data_ptr(),
            lens.data_ptr(),
            next_draft.data_ptr(),
            slots.data_ptr(),
            store_new_tokens.data_ptr(),
            store_next_new_tokens.data_ptr(),
            store_lens.data_ptr(),
            store_next_draft.data_ptr(),
            rows,
            row_begin,
            columns,
            num_slots,
            store_new_tokens.shape[0],
            new_tokens.shape[1],
            store_next_new_tokens.shape[0],
            next_new_tokens.shape[1],
            store_next_draft.shape[1],
            next_draft.shape[1],
        )
        if SlotScatter._kernel is None:
            SlotScatter._kernel = _compile("scatter", args)
        SlotScatter._kernel(*args, torch.cuda.current_stream().cuda_stream)
        return True


class StepInputStage:
    """One decode step's per-step device inputs, written by one ``stage_kernel`` launch.

    A step calls ``begin()``, then stages the overlap gathers (``gather``), host-to-device copies whose values go into a
    pinned host record that the kernel reads in place (``copy``, up to ``max_copies``) and KV cache block-offset copies
    (``block_copy``, up to ``max_block_copies``), then ``commit()`` launches one kernel on the current stream that
    performs all of them. The record is reused every step: the first ``copy`` after a commit that staged copies waits
    for that commit's kernel (in the overlap loop it has run by then: the host is at most one step ahead).

    Args:
        capacity: The record's size in int32 values, the most a step's copies stage in total.
    """

    _kernel: ClassVar[Any] = None

    def __init__(self, capacity: int) -> None:
        from . import spec_step_copies_kernel as kernel

        self.max_copies = kernel.STAGED_COPIES
        self.max_block_copies = kernel.BLOCK_COPIES
        self.record = torch.empty((capacity,), dtype=torch.int32, pin_memory=prefer_pinned())
        self._copies: list[tuple[int, int, int]] = []
        self._block_copies: list[tuple[int, ...]] = []
        self._gather: tuple[int, ...] | None = None
        self._used = 0
        self._done: torch.cuda.Event | None = None
        self._record_free = True

    def begin(self) -> None:
        """Start a step: drop anything staged by a step that did not commit (its preparation raised)."""
        self._copies = []
        self._block_copies = []
        self._gather = None
        self._used = 0

    def gather(
        self,
        store_next_new_tokens: torch.Tensor,
        store_next_draft: torch.Tensor,
        store_lens: torch.Tensor,
        slots: torch.Tensor,
        pos_indices: torch.Tensor,
        rows: int,
        tokens_per_row: int,
        draft_width: int,
        input_ids: torch.Tensor,
        input_begin: int,
        draft_tokens: torch.Tensor,
        draft_begin: int,
        pos_offsets: torch.Tensor,
        pos_begin: int,
        kv_offsets: torch.Tensor,
        kv_begin: int,
    ) -> bool:
        """Stage the overlap scheduler's gathers of the rows whose request ran in the previous step.

        For ``r < rows`` with ``s = slots[r]`` and ``j < tokens_per_row``:
        ``input_ids[input_begin + r * tokens_per_row + j] = store_next_new_tokens[j, s, 0]``,
        ``pos_offsets[pos_begin + r * tokens_per_row + j] = store_lens[pos_indices[r * tokens_per_row + j]]``,
        ``draft_tokens[draft_begin + r * draft_width + j] = store_next_draft[s, j]`` for ``j < draft_width`` and
        ``kv_offsets[kv_begin + r] = store_lens[s] - tokens_per_row``. Every tensor is int32 on the device; the
        stores are [width, slots, 1], [slots, width] and [slots].

        Returns:
            False, staging nothing, when an argument is outside what the kernel covers or a gather is already
            staged; True otherwise.
        """
        tensors = (
            store_next_new_tokens,
            store_next_draft,
            store_lens,
            slots,
            pos_indices,
            input_ids,
            draft_tokens,
            pos_offsets,
            kv_offsets,
        )
        if not (
            self._gather is None
            and all(_is_i32(t) for t in tensors)
            and _gather_shapes_ok(
                store_next_new_tokens,
                store_next_draft,
                store_lens,
                rows,
                tokens_per_row,
                draft_width,
            )
            and slots.numel() >= rows
            and pos_indices.numel() >= rows * tokens_per_row
            and input_ids.numel() >= input_begin + rows * tokens_per_row
            and draft_tokens.numel() >= draft_begin + rows * draft_width
            and pos_offsets.numel() >= pos_begin + rows * tokens_per_row
            and kv_offsets.numel() >= kv_begin + rows
            and min(input_begin, draft_begin, pos_begin, kv_begin) >= 0
        ):
            return False
        if rows == 0:
            return True
        self._gather = (
            store_next_new_tokens.data_ptr(),
            store_next_draft.data_ptr(),
            store_lens.data_ptr(),
            slots.data_ptr(),
            pos_indices.data_ptr(),
            input_ids.data_ptr(),
            draft_tokens.data_ptr(),
            pos_offsets.data_ptr(),
            kv_offsets.data_ptr(),
            rows,
            tokens_per_row,
            draft_width,
            store_lens.numel(),
            store_next_draft.shape[1],
            input_begin,
            draft_begin,
            pos_begin,
            kv_begin,
        )
        return True

    def copy(self, dst: torch.Tensor, values: torch.Tensor) -> bool:
        """Stage ``dst[:n] = values`` for the ``n`` int32 host ``values``; ``dst`` is a 1-D int32 device tensor.

        Returns:
            False, staging nothing, when ``dst`` is not a contiguous 1-D CUDA int32 tensor of at least ``n``
            values, the step already staged ``max_copies`` copies or the record cannot hold the values; True
            otherwise.
        """
        host = values.reshape(-1)
        n = host.numel()
        if not (
            _is_i32(dst)
            and dst.dim() == 1
            and n <= dst.numel()
            and host.dtype == torch.int32
            and not host.is_cuda
            and len(self._copies) < self.max_copies
            and self._used + n <= self.record.numel()
            and self.record.is_pinned()
        ):
            return False
        if n == 0:
            return True
        if not self._record_free:
            self._done.synchronize()
            self._record_free = True
        self.record[self._used : self._used + n].copy_(host)
        self._copies.append((dst.data_ptr(), self._used, n))
        self._used += n
        return True

    def block_copy(
        self,
        offsets: torch.Tensor,
        table: torch.Tensor,
        copy_index: torch.Tensor,
        index_scales: torch.Tensor,
        kv_offset: torch.Tensor,
    ) -> bool:
        """Stage a KV cache manager's block-offset copy (``copy_batch_block_offsets_to_device``).

        Args:
            offsets: The device block offsets, int32 [pools, seqs_cap, 2, blocks].
            table: The pinned host page table, int32 [pools, table_seqs, 2, blocks].
            copy_index: The table row of each of the step's sequences, pinned host int32 [seqs].
            index_scales: The per-pool page index scale, pinned host int32 [pools].
            kv_offset: The per-pool V offset, pinned host int32 [pools].

        For every pool and sequence the K row is ``index_scale * page`` and the V row ``index_scale * page +
        kv_offset`` over the table row ``copy_index[sequence]``; a bad page (-1) gives 0.

        Returns:
            False, staging nothing, when an argument is outside what the kernel covers or the step already staged
            ``max_block_copies`` block copies; True otherwise.
        """
        if not (
            _is_i32(offsets)
            and offsets.dim() == 4
            and table.dim() == 4
            and all(_is_pinned_i32(t) for t in (table, copy_index, index_scales, kv_offset))
            and len(self._block_copies) < self.max_block_copies
        ):
            return False
        pools, table_seqs, kv, blocks = table.shape
        seqs = copy_index.numel()
        if not (
            kv == 2
            and offsets.shape[0] >= pools
            and offsets.shape[1] >= seqs
            and offsets.shape[2] == 2
            and offsets.shape[3] == blocks
            and index_scales.numel() >= pools
            and kv_offset.numel() >= pools
        ):
            return False
        if pools * seqs * blocks == 0:
            return True
        self._block_copies.append(
            (
                table.data_ptr(),
                offsets.data_ptr(),
                copy_index.data_ptr(),
                index_scales.data_ptr(),
                kv_offset.data_ptr(),
                pools,
                table_seqs,
                offsets.shape[1],
                blocks,
                seqs,
            )
        )
        return True

    def commit(self) -> None:
        """Launch the staged gathers and copies as one kernel on the current stream, then start the next staging."""
        if self._gather is None and not self._copies and not self._block_copies:
            return
        gather = self._gather or (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0)
        copies = self._copies + [(0, 0, 0)] * (self.max_copies - len(self._copies))
        blocks = self._block_copies + [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)] * (
            self.max_block_copies - len(self._block_copies)
        )
        args = (
            *gather,
            self.record.data_ptr(),
            *(c[0] for c in copies),
            *(c[1] for c in copies),
            *(c[2] for c in copies),
            *(v for b in blocks for v in b),
        )
        if StepInputStage._kernel is None:
            StepInputStage._kernel = _compile("stage", args)
        StepInputStage._kernel(*args, torch.cuda.current_stream().cuda_stream)
        if self._copies:
            if self._done is None:
                self._done = torch.cuda.Event()
            self._done.record()
            self._record_free = False
        self.begin()
