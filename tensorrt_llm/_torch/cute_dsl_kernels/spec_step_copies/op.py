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
``index_copy_``). ``StepInputGather`` performs the overlap scheduler's gathers from those stores into a decode step's
inputs (one launch for the engine's four). Each kernel takes every size and address as a launch argument, so it is
compiled once per process, on the first launch, which must happen outside CUDA-graph capture. The kernels are called
through TVM-FFI: a launch costs a few microseconds of host time.

Callers use them only where ``is_supported()`` holds. A call whose arguments are outside what the kernel covers launches
nothing and returns False; the caller then does that work itself.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from ...._utils import get_sm_version
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


class StepInputGather:
    """The overlap scheduler's gathers of a decode step's inputs from the sampler's slot stores, as one kernel."""

    _kernel: ClassVar[Any] = None

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
        """Launch the gathers of the rows whose request ran in the previous step on the current stream.

        For ``r < rows`` with ``s = slots[r]`` and ``j < tokens_per_row``:
        ``input_ids[input_begin + r * tokens_per_row + j] = store_next_new_tokens[j, s, 0]``,
        ``pos_offsets[pos_begin + r * tokens_per_row + j] = store_lens[pos_indices[r * tokens_per_row + j]]``,
        ``draft_tokens[draft_begin + r * draft_width + j] = store_next_draft[s, j]`` for ``j < draft_width`` and
        ``kv_offsets[kv_begin + r] = store_lens[s] - tokens_per_row``. Every tensor is int32 on the device; the
        stores are [width, slots, 1], [slots, width] and [slots].

        Returns:
            False, launching nothing, when an argument is outside what the kernel covers; True otherwise.
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
            all(_is_i32(t) for t in tensors)
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
        args = (
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
        if StepInputGather._kernel is None:
            StepInputGather._kernel = _compile("gather", args)
        StepInputGather._kernel(*args, torch.cuda.current_stream().cuda_stream)
        return True
