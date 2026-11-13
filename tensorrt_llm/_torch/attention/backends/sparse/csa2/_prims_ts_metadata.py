# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact live CSA2 packed-pool selections into PrimTS physical source metadata."""

# Annotations stay evaluated: the CuTe DSL resolves kernel annotations against
# module globals, and its classes import CuTe lazily.
from functools import cache
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

if TYPE_CHECKING:
    from collections.abc import Callable

    from .metadata import CSA2PrimsTSMainBank
    from .params import CSA2BackendForwardArgs
    from .prims_ts import _PrimsTSMetadata

__all__ = ["prepare_prims_ts_metadata"]

# One warp compacts one query row; lane ``l`` owns column ``32 c + l`` of chunk ``c``.
_WARPS = 8
_INT32_LIMIT = 1 << 31


@cache
def _compact_kernel() -> type:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Uint32, Uint64

    from tensorrt_llm._torch.kernels.blackwell.utils import (
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    def _min(a, b):
        return cutlass.select_(a < b, a, b)

    def _max(a, b):
        return cutlass.select_(a < b, b, a)

    def _as(kind, value):
        """``value`` (INT64) clamped into ``kind``'s range and converted."""
        if kind == Int64:
            return value
        return Int32(_max(_min(value, Int64(_INT32_LIMIT - 1)), Int64(-_INT32_LIMIT)))

    def _within(kind, value, first, count):
        """``first <= value < first + count`` for ``kind`` values in one unsigned compare.

        ``count`` is INT64; exact while ``0 <= first`` and ``first + count`` is at
        most one past ``kind``'s maximum.
        """
        unsigned = Uint32 if kind == Int32 else Uint64
        return unsigned(kind(value) - kind(first)) < unsigned(count)

    def _split(logical, kind, page_size):
        """``logical``'s page and offset; power-of-two pages use shift and mask.

        Only non-negative rows' splits are used.
        """
        bits = page_size.bit_length() - 1
        if page_size == 1 << bits:
            return logical >> kind(bits), logical & kind(page_size - 1)
        return logical // kind(page_size), logical % kind(page_size)

    def _column(selections, chunk, lane, width):
        """Selection ``32 chunk + lane`` of a row in its own type, or -1 past the width."""
        kind = selections.element_type
        column = Int32(chunk * 32) + lane
        if (chunk + 1) * 32 <= width:
            return kind(selections[column])
        inside = column < Int32(width)
        value = kind(selections[cutlass.select_(inside, column, Int32(width - 1))])
        return kind(cutlass.select_(inside, value, kind(-1)))

    class _Compact:
        """Write each list's valid rows as a prefix, then -1 up to its capacity.

        With ``claim``, every kept MAIN row also claims its owner slot for the selection
        bank decode, as ``_prims_ts_bank.dedup_main_records`` claims: the slot keeps the
        largest ``tag | (2^31 - 1 - f)``, so the smallest flat selection ``f`` of this
        call reading the row, where ``tag`` is the epoch after the device's current one.
        Without ``writes_main`` only the SWA list is written; the MAIN list and its
        lengths keep their contents.
        """

        def __init__(
            self,
            swa_width: int,
            swa_capacity: int,
            main_width: int,
            main_capacity: int,
            has_main: bool,
            writes_main: bool,
            mapped: bool,
            bank: bool,
            page_size: int,
            marks: bool,
            claim: bool,
            use_pdl: bool,
        ) -> None:
            if has_main and not writes_main:
                raise ValueError("CSA2 PrimTS compaction reads MAIN selections only to write them")
            self.swa_width = swa_width
            self.swa_capacity = swa_capacity
            self.main_width = main_width
            self.main_capacity = main_capacity
            self.has_main = has_main
            self.writes_main = writes_main
            self.mapped = mapped
            self.bank = bank
            self.page_size = page_size
            self.marks = marks
            self.claim = claim
            self.use_pdl = use_pdl

        @cute.jit
        def _claim(self, owners, tag, row, column, record):
            """Offer flat selection ``row * main_capacity + column`` as ``record``'s owner."""
            flat = row * Int32(self.main_capacity) + column
            value = tag | Int64(Int32(_INT32_LIMIT - 1) - flat)
            cute.arch.atomic_max(owners.iterator + Int64(record), value, sem="relaxed", scope="gpu")

        @cute.jit
        def _finish(self, compact, lengths, row, lane, count, capacity):
            for chunk in cutlass.range_constexpr((capacity + 31) // 32):
                column = Int32(chunk * 32) + lane
                if cutlass.const_expr((chunk + 1) * 32 <= capacity):
                    if column >= count:
                        compact[column] = Int32(-1)
                else:
                    if column >= count and column < Int32(capacity):
                        compact[column] = Int32(-1)
            if lane == Int32(0):
                lengths[row] = count

        @cute.kernel
        def kernel(
            self,
            swa: cute.Tensor,
            main: cute.Tensor,
            pages: cute.Tensor,
            requests: cute.Tensor,
            visible: cute.Tensor,
            indices: cute.Tensor,
            lengths: cute.Tensor,
            extra_indices: cute.Tensor,
            extra_lengths: cute.Tensor,
            offsets: cute.Tensor,
            marks: cute.Tensor,
            owners: cute.Tensor,
            epoch: cute.Tensor,
            num_rows: Int32,
            swa_rows: Int64,
            main_rows: Int64,
            main_pool_pages: Int64,
            max_positions: Int64,
            table_rows: Int64,
            table_columns: Int64,
            bank_base: Int64,
            bank_rows: Int64,
            bank_requests: Int64,
        ) -> None:
            lane = cute.arch.lane_idx()
            row = cute.arch.block_idx()[0] * Int32(_WARPS) + cute.arch.warp_idx()
            if cutlass.const_expr(self.use_pdl):
                # Selections, tables and owner slots may still be written by the
                # preceding kernel; the attention that reads these lists waits for
                # this whole grid, so it may launch now.
                griddepcontrol_wait()
                griddepcontrol_launch_dependents()
            if row < num_rows:
                below = cute.arch.lanemask_lt()
                swa_chunks = (self.swa_width + 31) // 32
                main_chunks = (self.main_width + 31) // 32
                # Row views keep per-element addressing to the column offset.
                compact = indices[row, None]
                extra_compact = extra_indices[row, None]
                # Every selection load of the row issues before any store.
                swa_row = swa[row, None]
                swa_values = []
                for chunk in cutlass.range_constexpr(swa_chunks):
                    swa_values.append(_column(swa_row, chunk, lane, self.swa_width))
                main_values = []
                if cutlass.const_expr(self.has_main):
                    main_row = main[row, None]
                    for chunk in cutlass.range_constexpr(main_chunks):
                        main_values.append(_column(main_row, chunk, lane, self.main_width))

                # Selections are compared in their own type: every bound is narrowed
                # into it, and the host keeps INT32 physical rows from overflowing.
                kind = swa.element_type
                count = Int32(0)
                for chunk in cutlass.range_constexpr(swa_chunks):
                    value = swa_values[chunk]
                    # swa_rows bounds physical rows by both the pool and INT32.
                    keep = _within(kind, value, 0, swa_rows)
                    ballot = cute.arch.vote_ballot_sync(keep)
                    if keep:
                        compact[count + cute.arch.popc(ballot & below)] = Int32(value)
                    count = count + cute.arch.popc(ballot)
                self._finish(compact, lengths, row, lane, count, self.swa_capacity)

                extra = Int32(0)
                if cutlass.const_expr(self.has_main):
                    kind = main.element_type
                    tag = Int64(0)
                    if cutlass.const_expr(self.claim):
                        # The bank decode advances the device epoch to this call's.
                        tag = (Int64(epoch[0]) + Int64(1)) << Int64(32)
                    if cutlass.const_expr(self.mapped):
                        request = Int64(requests[row])
                        # Every row-wide bound folds into one logical range [lo, hi):
                        # the visible prefix, the position limit and the table's columns.
                        hi = _min(
                            _min(Int64(visible[row]), max_positions),
                            table_columns * Int64(self.page_size),
                        )
                        lo = Int64(0)
                        owned = (request >= Int64(0)) & (request < table_rows)
                        shift = Int64(0)
                        if cutlass.const_expr(self.bank):
                            # Valid rows address their request's bank rows instead of
                            # the packed records the bank decoded them from, and must
                            # land in [0, min(bank_rows, 2^31)).
                            in_bank = (request >= Int64(0)) & (request < bank_requests)
                            slot = cutlass.select_(in_bank, request, Int64(0))
                            begin = Int64(offsets[slot])
                            shift = begin - bank_base
                            hi = _min(hi, Int64(offsets[slot + Int64(1)]) - begin)
                            hi = _min(hi, _min(bank_rows, Int64(_INT32_LIMIT)) - shift)
                            lo = _max(lo, Int64(0) - shift)
                            owned = owned & in_bank
                        hi = cutlass.select_(owned, hi, Int64(0))
                        # [lo, hi) narrowed to the selections' type, which every
                        # selection already lies in.
                        if cutlass.const_expr(kind == Int32):
                            lo = _min(lo, Int64(_INT32_LIMIT))
                            hi = _min(hi, Int64(_INT32_LIMIT))
                        first = _as(kind, lo)
                        span = _max(hi - lo, Int64(0))
                        page_kind = pages.element_type
                        page_size = kind(self.page_size)
                        # Selections outside the range read the row's first page,
                        # which exists: the host skips empty tables, and unowned rows
                        # read table row 0.
                        table = pages[cutlass.select_(owned, request, Int64(0)), None]
                        physical_pages = []
                        for chunk in cutlass.range_constexpr(main_chunks):
                            logical = main_values[chunk]
                            page = _split(logical, kind, self.page_size)[0]
                            inside = _within(kind, logical, first, span)
                            physical_pages.append(table[cutlass.select_(inside, page, kind(0))])
                        for chunk in cutlass.range_constexpr(main_chunks):
                            logical = main_values[chunk]
                            physical_page = page_kind(physical_pages[chunk])
                            backed = _within(kind, logical, first, span) & _within(
                                page_kind, physical_page, 0, main_pool_pages
                            )
                            physical = (
                                kind(cutlass.select_(backed, physical_page, page_kind(0)))
                                * page_size
                                + _split(logical, kind, self.page_size)[1]
                            )
                            kept = backed & (physical < _as(kind, main_rows))
                            target = physical
                            if cutlass.const_expr(self.bank):
                                # Inside the bank, so the sum fits INT32.
                                target = logical + kind(shift)
                            votes = cute.arch.vote_ballot_sync(kept)
                            if kept:
                                column = extra + cute.arch.popc(votes & below)
                                extra_compact[column] = Int32(target)
                                if cutlass.const_expr(self.marks):
                                    # The bank decodes only the rows some selection keeps.
                                    marks[target] = cutlass.Uint8(1)
                                if cutlass.const_expr(self.claim):
                                    self._claim(owners, tag, row, column, target)
                            extra = extra + cute.arch.popc(votes)
                    else:
                        for chunk in cutlass.range_constexpr(main_chunks):
                            value = main_values[chunk]
                            keep = _within(kind, value, 0, main_rows)
                            ballot = cute.arch.vote_ballot_sync(keep)
                            if keep:
                                column = extra + cute.arch.popc(ballot & below)
                                extra_compact[column] = Int32(value)
                                if cutlass.const_expr(self.claim):
                                    self._claim(owners, tag, row, column, value)
                            extra = extra + cute.arch.popc(ballot)
                if cutlass.const_expr(self.writes_main):
                    self._finish(extra_compact, extra_lengths, row, lane, extra, self.main_capacity)

        @cute.jit
        def __call__(
            self,
            swa: cute.Tensor,
            main: cute.Tensor,
            pages: cute.Tensor,
            requests: cute.Tensor,
            visible: cute.Tensor,
            indices: cute.Tensor,
            lengths: cute.Tensor,
            extra_indices: cute.Tensor,
            extra_lengths: cute.Tensor,
            offsets: cute.Tensor,
            marks: cute.Tensor,
            owners: cute.Tensor,
            epoch: cute.Tensor,
            num_rows: Int32,
            swa_rows: Int64,
            main_rows: Int64,
            main_pool_pages: Int64,
            max_positions: Int64,
            table_rows: Int64,
            table_columns: Int64,
            bank_base: Int64,
            bank_rows: Int64,
            bank_requests: Int64,
            stream: object,
        ) -> None:
            self.kernel(
                swa,
                main,
                pages,
                requests,
                visible,
                indices,
                lengths,
                extra_indices,
                extra_lengths,
                offsets,
                marks,
                owners,
                epoch,
                num_rows,
                swa_rows,
                main_rows,
                main_pool_pages,
                max_positions,
                table_rows,
                table_columns,
                bank_base,
                bank_rows,
                bank_requests,
            ).launch(
                grid=((num_rows + Int32(_WARPS - 1)) // Int32(_WARPS), 1, 1),
                block=(32 * _WARPS, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )

    return _Compact


@cache
def _compiled(key: tuple) -> "Callable[..., object]":
    import cutlass
    import cutlass.cute as cute

    (
        dtypes,
        swa_width,
        swa_capacity,
        main_width,
        main_capacity,
        has_main,
        writes_main,
        mapped,
        bank,
        page_size,
        marks,
        claim,
        use_pdl,
    ) = key
    element = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}

    def matrix(dtype: torch.dtype) -> object:
        return cute.runtime.make_fake_tensor(
            element[dtype],
            (cute.sym_int(), cute.sym_int()),
            stride=(cute.sym_int64(divisibility=1), 1),
            assumed_align=4,
        )

    def vector(dtype: torch.dtype) -> object:
        return cute.runtime.make_fake_tensor(
            element[dtype],
            (cute.sym_int(),),
            stride=(cute.sym_int64(divisibility=1),),
            assumed_align=4,
        )

    def slots(dtype: torch.dtype) -> object:
        # Owner slots take 8-byte atomics; INT32 stand-ins are never read.
        return cute.runtime.make_fake_compact_tensor(
            element[dtype],
            (cute.sym_int(),),
            stride_order=(0,),
            assumed_align=8 if dtype == torch.int64 else 4,
        )

    swa, main, pages, requests, visible, offsets, owners = dtypes
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile[cute.FrontendNext](
        _compact_kernel()(
            swa_width,
            swa_capacity,
            main_width,
            main_capacity,
            has_main,
            writes_main,
            mapped,
            bank,
            page_size,
            marks,
            claim,
            use_pdl,
        ),
        matrix(swa),
        matrix(main),
        matrix(pages),
        vector(requests),
        vector(visible),
        matrix(torch.int32),
        vector(torch.int32),
        matrix(torch.int32),
        vector(torch.int32),
        vector(offsets),
        cute.runtime.make_fake_compact_tensor(
            cutlass.Uint8, (cute.sym_int(),), stride_order=(0,), assumed_align=1
        ),
        slots(owners),
        slots(owners),
        cutlass.Int32(0),
        *(cutlass.Int64(0) for _ in range(9)),
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def _validate_indices(
    indices: torch.Tensor,
    rows: int,
    capacity: int,
    device: torch.device,
    name: str,
) -> None:
    if (
        not isinstance(indices, torch.Tensor)
        or indices.device != device
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.ndim != 2
        or indices.shape[0] != rows
        or indices.shape[1] > capacity
        or indices.stride(1) != 1
        or indices.stride(0) < indices.shape[1]
    ):
        raise ValueError(
            f"CSA2 {name} selections must be CUDA integer[queries,width] within capacity"
        )


def prepare_prims_ts_metadata(
    inputs: "CSA2BackendForwardArgs",
    prepared: "_PrimsTSMetadata",
    main_bank: "CSA2PrimsTSMainBank | None" = None,
    marks: torch.Tensor | None = None,
    claims: tuple[torch.Tensor, torch.Tensor] | None = None,
    swa_only: bool = False,
) -> None:
    """Refresh two compact lists without allocations or host tensor reads.

    SWA selections address physical packed rows. MAIN selections either address
    physical rows or, with ``main_mapping``, logical rows in the owner's live
    page table. Bounds are checked before INT32 conversion or page-table loads;
    -1 denotes invalid, while physical row zero remains a valid source token.
    All output metadata is preallocated, and padded outer strides are allowed.

    With ``main_bank``, the mapped MAIN rows that would be valid address their
    request's rows of that decoded bank instead of the packed pool. ``marks``, a
    contiguous UINT8[main_bank.rows] vector, then also gets a nonzero byte at
    every bank row a selection keeps; other bytes are left as they are.

    ``claims``, the device's ``(owners, epoch)`` from ``main_record_owners``, makes
    every kept MAIN pool row claim its owner slot for the next
    ``dedup_main_records(..., claimed=True)`` over ``prepared.extra_indices``, which must
    then be contiguous: flat selection ``r * width + c`` is its entry ``[r, c]``.

    With ``swa_only``, only the SWA list and its lengths are refreshed: the MAIN list keeps
    what the last compaction into ``prepared`` wrote, for a layer whose MAIN selections,
    pool and page table are that compaction's. It takes no bank, marks or claims.
    """
    rows, swa_capacity = prepared.indices.shape
    main_capacity = prepared.extra_indices.shape[1]
    device = prepared.indices.device
    if not prepared.indices.is_cuda:
        raise ValueError("CSA2 PrimTS compact metadata requires CUDA")
    if inputs.swa_pool is None or inputs.swa_pool.device != device:
        raise ValueError("CSA2 SWA pool must be on the compact metadata device")
    _validate_indices(inputs.swa_indices, rows, swa_capacity, device, "SWA")
    if swa_only and (main_bank is not None or marks is not None or claims is not None):
        raise ValueError("CSA2 PrimTS SWA-only compaction takes no MAIN bank, marks or claims")
    has_main = inputs.topk_indices is not None and not swa_only
    if has_main:
        if inputs.main_pool is None or inputs.main_pool.device != device:
            raise ValueError("CSA2 MAIN selections require a pool on the compact metadata device")
        _validate_indices(inputs.topk_indices, rows, main_capacity, device, "MAIN")
    mapping = None if swa_only else inputs.main_mapping
    if mapping is not None:
        if not has_main:
            raise ValueError("CSA2 MAIN mapping requires logical MAIN selections")
        pages, page_size, max_positions, requests, visible = mapping
        if (
            pages.device != device
            or pages.dtype not in (torch.int32, torch.int64)
            or pages.ndim != 2
            or pages.stride(1) != 1
            or page_size <= 0
            or max_positions < 0
        ):
            raise ValueError(
                "CSA2 MAIN mapping requires a CUDA integer page table and valid bounds"
            )
        for values in (requests, visible):
            if (
                values.device != device
                or values.dtype not in (torch.int32, torch.int64)
                or values.shape != (rows,)
            ):
                raise ValueError("CSA2 MAIN requests and visibility must be CUDA integer[queries]")
    else:
        pages = requests = visible = None
        page_size, max_positions = 1, 0
    if main_bank is not None:
        offsets = main_bank.offsets
        if (
            mapping is None
            or offsets.device != device
            or offsets.dtype != torch.int64
            or offsets.shape != (main_bank.num_requests + 1,)
            or not offsets.is_contiguous()
            or main_bank.rows <= 0
            or main_bank.num_requests <= 0
        ):
            raise ValueError(
                "CSA2 PrimTS MAIN banks need logical selections and INT64[requests + 1] offsets"
            )
    if marks is not None and (
        main_bank is None
        or marks.dtype != torch.uint8
        or marks.shape != (main_bank.rows,)
        or not marks.is_contiguous()
        or marks.device != device
    ):
        raise ValueError("CSA2 PrimTS bank row marks must be a CUDA UINT8[bank rows] vector")
    if claims is not None:
        owners, epoch = claims
        if (
            main_bank is not None
            or not prepared.extra_indices.is_contiguous()
            or rows * main_capacity >= _INT32_LIMIT
            or owners.dtype != torch.int64
            or owners.ndim != 1
            or not owners.is_contiguous()
            or owners.device != device
            or owners.shape[0] < (inputs.main_pool.shape[0] if has_main else 0)
            or epoch.dtype != torch.int64
            or epoch.shape != (1,)
            or epoch.device != device
        ):
            raise ValueError(
                "CSA2 PrimTS MAIN claims need pool rows, contiguous compact MAIN rows and an "
                "INT64 owner slot per pool row"
            )
    if rows == 0:
        return
    # Clamped page reads need a table entry; an empty table admits no MAIN row, so
    # its rows compact like a layer without MAIN selections.
    table_rows, table_columns = pages.shape if mapping is not None else (0, 0)
    reads_main = has_main and (mapping is None or pages.numel() > 0)
    # Absent operands are stood in by live metadata tensors the kernel never reads.
    main = inputs.topk_indices if reads_main else inputs.swa_indices
    if not reads_main or mapping is None:
        pages, requests, visible = prepared.indices, prepared.lengths, prepared.lengths
    offsets = main_bank.offsets if reads_main and main_bank is not None else prepared.lengths
    main_pool_rows = inputs.main_pool.shape[0] if has_main else 0
    # INT32 indices bound every final row; bank rows are bounded by the bank instead.
    main_rows = main_pool_rows if main_bank is not None else min(main_pool_rows, _INT32_LIMIT)
    mapped = reads_main and mapping is not None
    marked = mapped and marks is not None
    claimed = reads_main and claims is not None
    # An unclaimed launch reads no owner slots; any INT32 vector stands in.
    owners, epoch = claims if claimed else (prepared.lengths, prepared.lengths)
    if (
        mapped
        and inputs.topk_indices.dtype == torch.int32
        and main_pool_rows > _INT32_LIMIT - page_size
    ):
        # Physical rows are formed in the selections' type before their bound check.
        raise ValueError(
            "CSA2 PrimTS INT32 MAIN selections need a pool below 2^31 - page_size rows"
        )
    key = (
        (
            inputs.swa_indices.dtype,
            main.dtype,
            pages.dtype,
            requests.dtype,
            visible.dtype,
            offsets.dtype,
            owners.dtype,
        ),
        inputs.swa_indices.shape[1],
        swa_capacity,
        main.shape[1] if reads_main else 0,
        main_capacity,
        reads_main,
        not swa_only,
        mapped,
        mapped and main_bank is not None,
        page_size if mapped else 1,
        marked,
        claimed,
        get_env_enable_pdl(),
    )
    _compiled(key)(
        inputs.swa_indices,
        main,
        pages,
        requests,
        visible,
        prepared.indices,
        prepared.lengths,
        prepared.extra_indices,
        prepared.extra_lengths,
        offsets,
        # An unmarked launch writes no marks; any UINT8 vector stands in.
        marks if marked else prepared.indices[0].view(torch.uint8),
        owners,
        epoch,
        rows,
        min(inputs.swa_pool.shape[0], _INT32_LIMIT),
        main_rows,
        (main_pool_rows + page_size - 1) // page_size,
        max_positions,
        table_rows,
        table_columns,
        main_bank.base if main_bank is not None else 0,
        main_bank.rows if main_bank is not None else 0,
        main_bank.num_requests if main_bank is not None else 0,
    )
