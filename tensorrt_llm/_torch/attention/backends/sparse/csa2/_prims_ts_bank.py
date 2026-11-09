# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode a context phase's CSA2 MAIN rows once into a unit-scale E4M3 bank for PrimTS.

Context queries of one request select their MAIN rows from the same owner pages,
so decoding every selection inside the attention kernel converts each row many
times. The bank holds each context request's MAIN rows, decoded once with the
producer's own conversion, and the kernel copies its operands from it.

A phase that selects fewer rows than its requests own decodes only the bank rows
its compact metadata marks, each once however many queries select it.
"""

# Annotations stay evaluated: the CuTe DSL resolves kernel annotations against
# module globals, and its classes import CuTe lazily.
from functools import cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from .metadata import CSA2PrimsTSMainBank

__all__ = [
    "decode_main_bank",
    "decode_main_records",
    "main_bank_marks",
    "main_bank_rows_limit",
    "main_bank_storage",
    "main_records_rows_limit",
    "main_records_storage",
]

_HEAD_DIM = 512
_MAIN_ROW_BYTES = 288
_BANK_ROW_BYTES = 512
_WARPS = 8
# Rows a warp owns, and how many of them it reads before converting any.
_GROUP = 8
_BATCH = 4
# Bank storage is retained per device and bounded, so it is reserved with the
# PrimTS workspace; larger phases keep decoding MAIN rows inside the kernel.
_MAX_BANK_BYTES = 64 << 20
_BANK_CACHE_KEY = "csa2_prims_ts_main_bank"
_MARKS_CACHE_KEY = "csa2_prims_ts_main_bank_marks"


def main_bank_rows_limit() -> int:
    """Most MAIN rows one bank holds; also the bound its reservation covers."""
    return _MAX_BANK_BYTES // _BANK_ROW_BYTES


def main_bank_storage(arenas: dict, device: torch.device, rows: int) -> torch.Tensor:
    """Return a UINT8[rows,512] view of the device's retained bank storage.

    Storage grows to the next power of two of rows, up to
    :func:`main_bank_rows_limit`, and is only allocated outside CUDA Graph capture.
    """
    if not 0 < rows <= main_bank_rows_limit():
        raise ValueError("CSA2 PrimTS MAIN bank rows exceed its reserved capacity")
    key = (_BANK_CACHE_KEY, device)
    storage = arenas.get(key)
    if storage is None or storage.shape[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CSA2 PrimTS MAIN bank storage cannot grow during CUDA Graph capture"
            )
        capacity = min(1 << (rows - 1).bit_length(), main_bank_rows_limit())
        storage = torch.empty((capacity, _BANK_ROW_BYTES), device=device, dtype=torch.uint8)
        arenas[key] = storage
    return storage[:rows]


def main_bank_marks(arenas: dict, device: torch.device, rows: int) -> torch.Tensor:
    """Return a UINT8[rows] view of the device's retained bank row marks.

    The marks are zero between phases: the compact metadata sets the bank rows a
    phase keeps, and :func:`decode_main_bank` clears each mark it decodes. Storage
    grows like :func:`main_bank_storage`, zero-filled.
    """
    if not 0 < rows <= main_bank_rows_limit():
        raise ValueError("CSA2 PrimTS MAIN bank rows exceed its reserved capacity")
    key = (_MARKS_CACHE_KEY, device)
    marks = arenas.get(key)
    if marks is None or marks.shape[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CSA2 PrimTS MAIN bank marks cannot grow during CUDA Graph capture")
        capacity = min(1 << (rows - 1).bit_length(), main_bank_rows_limit())
        marks = torch.zeros((capacity,), device=device, dtype=torch.uint8)
        arenas[key] = marks
    return marks[:rows]


@cache
def _bank_kernel() -> type:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    from ...prims_ts.kernels.fmha_decode.fmha_decode_resources.smem_transformed_kv import (
        _cvt_e2m1_word_to_e4m3_words,
    )
    from ._prims_ts_producer import _load_packed_words

    class _MainBankDecode:
        """Decode bank rows; lane ``l`` converts channels ``[16 l, 16 l + 16)`` of a row.

        Warp ``w`` owns rows ``[8 w, 8 w + 8)``. Lane ``i < 8`` resolves row
        ``8 w + i``'s pool record, so the warp's offset and page reads overlap,
        and with ``marked`` reads and clears that row's mark. The warp then
        decodes its rows (with ``marked``, its marked rows) four at a time,
        issuing every pool read of a batch before converting any of its rows.

        With ``readable``, the pool has rows and 8-byte aligned row strides, so
        every slot of a batch reads unconditionally, a slot without a record
        from row 0, and no branch keeps a later read waiting on an earlier one.
        """

        def __init__(
            self, page_size: int, marked: bool, readable: bool, direct: bool = False
        ) -> None:
            self.page_size = page_size
            self.marked = marked
            self.readable = readable
            # With ``direct``, ``pages`` is an INT32[rows, 1] list of each bank row's pool
            # record (negative for none) instead of a request page table.
            self.direct = direct
            self.rows_per_block = _WARPS * _GROUP

        @cute.jit
        def _read(self, pool, record, lane):
            """Return whether ``record`` is a pool row, and lane's payload words and scale of it."""
            valid = record >= Int32(0)
            low, high, scale = Int32(0), Int32(0), Int32(0)
            if cutlass.const_expr(self.readable):
                row = Int64(cutlass.select_(valid, record, Int32(0)))
                packed = cutlass.Pointer(
                    pool.iterator + row * Int64(pool.stride[0]) + Int64(lane * Int32(8)),
                    dtype=Int32,
                ).load(count=2, alignment=8)
                low, high = Int32(packed[0]), Int32(packed[1])
                scale = Int32(pool[row, Int32(_HEAD_DIM // 2) + lane])
            else:
                if valid:
                    words = _load_packed_words(pool, record, lane * Int32(8), 2)
                    low, high = words[0], words[1]
                    scale = Int32(pool[record, Int32(_HEAD_DIM // 2) + lane])
            return valid, low, high, scale

        @cute.jit
        def _resolve(self, pool, pages, offsets, base, requests, bank_row):
            """Return the pool row behind ``bank_row``, or -1 when it has none."""
            target = Int64(bank_row) + base
            # The owning request has the last offset at or before the row; the
            # search runs only as many steps as the request count needs.
            lo, hi = Int32(0), requests
            while lo + Int32(1) < hi:
                mid = (lo + hi) // Int32(2)
                if Int64(offsets[mid]) <= target:
                    lo = mid
                else:
                    hi = mid
            logical = target - Int64(offsets[lo])
            # Pages are a compile-time size, so power-of-two pages split by shift
            # and mask instead of 64-bit division.
            page = logical // Int64(self.page_size)
            offset = logical % Int64(self.page_size)
            bits = self.page_size.bit_length() - 1
            if cutlass.const_expr(self.page_size == 1 << bits):
                page = logical >> Int64(bits)
                offset = logical & Int64(self.page_size - 1)
            physical = Int32(-1)
            if logical >= Int64(0) and page < Int64(pages.shape[1]):
                physical_page = Int64(pages[lo, page])
                row = physical_page * Int64(self.page_size) + offset
                # Unbacked pages and rows outside the pool decode to zero rows,
                # which the compact metadata masks like the physical path does.
                if physical_page >= Int64(0) and row < Int64(pool.shape[0]):
                    physical = Int32(row)
            return physical

        @cute.kernel
        def kernel(
            self,
            pool: cute.Tensor,
            pages: cute.Tensor,
            offsets: cute.Tensor,
            bank: cute.Tensor,
            marks: cute.Tensor,
            base: Int64,
            rows: Int32,
            requests: Int32,
        ) -> None:
            lane = cute.arch.thread_idx()[0] % Int32(32)
            warp = cute.arch.block_idx()[0] * Int32(_WARPS) + cute.arch.thread_idx()[0] // Int32(32)
            first = warp * Int32(_GROUP)
            own = first + lane
            decode = Int32(0)
            physical = Int32(-1)
            if lane < Int32(_GROUP) and own < rows:
                decode = Int32(1)
                if cutlass.const_expr(self.marked):
                    decode = Int32(marks[own])
                # Resolved whether marked or not, so the mark read overlaps it.
                if cutlass.const_expr(self.direct):
                    record = Int32(pages[own, 0])
                    # A bank row without a pool record is left as it is: no selection
                    # reads it.
                    decode = Int32(0)
                    if record >= Int32(0) and record < Int32(pool.shape[0]):
                        physical = record
                        decode = Int32(1)
                else:
                    physical = self._resolve(pool, pages, offsets, base, requests, own)
            if cutlass.const_expr(self.marked):
                if decode != Int32(0):
                    marks[own] = cutlass.Uint8(0)
            pending = cute.arch.vote_ballot_sync(decode != Int32(0))
            while pending != Int32(0):
                sources, active, reads = [], [], []
                for _ in cutlass.range_constexpr(_BATCH):
                    # The lowest pending lane's row; a slot past the last one stores nothing.
                    source = cute.arch.clz(cute.arch.brev(pending))
                    active.append(pending != Int32(0))
                    pending = pending & (pending - Int32(1))
                    sources.append(source)
                    reads.append(self._read(pool, cute.arch.shuffle_sync(physical, source), lane))
                for slot in cutlass.range_constexpr(_BATCH):
                    if active[slot]:
                        valid, low, high, scale = reads[slot]
                        # A row without a record converts a zero payload and scale
                        # to a zero row.
                        low = cutlass.select_(valid, low, Int32(0))
                        high = cutlass.select_(valid, high, Int32(0))
                        scale = cutlass.select_(valid, scale, Int32(0))
                        values = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                        values[0], values[1] = _cvt_e2m1_word_to_e4m3_words(
                            Int32(low), Int32(scale)
                        )
                        values[2], values[3] = _cvt_e2m1_word_to_e4m3_words(
                            Int32(high), Int32(scale)
                        )
                        destination = bank.iterator + Int64(first + sources[slot]) * Int64(
                            _BANK_ROW_BYTES
                        )
                        cutlass.Pointer(destination + Int64(lane * Int32(16)), dtype=Int32).store(
                            values.load(0, 4), alignment=16
                        )

        @cute.jit
        def __call__(
            self,
            pool: cute.Tensor,
            pages: cute.Tensor,
            offsets: cute.Tensor,
            bank: cute.Tensor,
            marks: cute.Tensor,
            base: Int64,
            rows: Int32,
            requests: Int32,
            stream: object,
        ) -> None:
            self.kernel(pool, pages, offsets, bank, marks, base, rows, requests).launch(
                grid=(
                    (rows + Int32(self.rows_per_block - 1)) // Int32(self.rows_per_block),
                    1,
                    1,
                ),
                block=(32 * _WARPS, 1, 1),
                stream=stream,
            )

    return _MainBankDecode


@cache
def _compiled(
    page_dtype: torch.dtype, page_size: int, marked: bool, readable: bool, direct: bool = False
) -> "Callable[..., object]":
    import cutlass
    import cutlass.cute as cute

    page_element = {torch.int32: cutlass.Int32, torch.int64: cutlass.Int64}[page_dtype]
    pool = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (cute.sym_int(), _MAIN_ROW_BYTES),
        stride=(cute.sym_int64(divisibility=1), 1),
        assumed_align=16,
    )
    pages = cute.runtime.make_fake_tensor(
        page_element,
        (cute.sym_int(), cute.sym_int()),
        stride=(cute.sym_int64(divisibility=1), 1),
        assumed_align=4,
    )
    offsets = cute.runtime.make_fake_compact_tensor(
        cutlass.Int64, (cute.sym_int(),), stride_order=(0,), assumed_align=8
    )
    bank = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int(), _BANK_ROW_BYTES), stride_order=(1, 0), assumed_align=16
    )
    marks = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int(),), stride_order=(0,), assumed_align=1
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile[cute.FrontendNext](
        _bank_kernel()(page_size, marked, readable, direct),
        pool,
        pages,
        offsets,
        bank,
        marks,
        cutlass.Int64(0),
        cutlass.Int32(0),
        cutlass.Int32(1),
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def decode_main_bank(
    main_pool: torch.Tensor,
    plan: "CSA2PrimsTSMainBank",
    bank: torch.Tensor,
    marks: torch.Tensor | None = None,
) -> None:
    """Decode each context request's MAIN rows into its bank rows on the current stream.

    Args:
        main_pool: The owner's packed MAIN records, CUDA UINT8[pool_rows,288]
            with a contiguous inner axis; outer strides may be padded.
        plan: Bank geometry of the context phase; request ``r`` owns bank rows
            ``[offsets[r] - base, offsets[r + 1] - base)``, which hold its
            logical rows resolved through ``pages[r]``.
        bank: Contiguous CUDA UINT8[plan.rows,512] destination.
        marks: Optional contiguous CUDA UINT8[plan.rows] row marks. With them,
            only marked rows are decoded and their marks are cleared; other bank
            rows keep their contents.

    Rows are rounded RN/satfinite to E4M3 at unit scale exactly as the PrimTS
    producer decodes them, so attention over the bank reads the same operands.
    """
    if (
        main_pool.dtype != torch.uint8
        or main_pool.ndim != 2
        or main_pool.shape[1] != _MAIN_ROW_BYTES
        or main_pool.stride(1) != 1
        or main_pool.data_ptr() % 16
    ):
        raise ValueError("CSA2 MAIN bank source must be an aligned CUDA UINT8[N,288] pool")
    if (
        bank.dtype != torch.uint8
        or bank.shape != (plan.rows, _BANK_ROW_BYTES)
        or not bank.is_contiguous()
        or bank.data_ptr() % 16
        or bank.device != main_pool.device
    ):
        raise ValueError("CSA2 MAIN bank must be contiguous aligned CUDA UINT8[rows,512]")
    pages, offsets = plan.pages, plan.offsets
    if (
        pages.dtype not in (torch.int32, torch.int64)
        or pages.ndim != 2
        or pages.shape[0] != plan.num_requests
        or pages.stride(1) != 1
        or offsets.dtype != torch.int64
        or offsets.shape != (plan.num_requests + 1,)
        or not offsets.is_contiguous()
        or pages.device != bank.device
        or offsets.device != bank.device
    ):
        raise ValueError("CSA2 MAIN bank plan has invalid page or offset tables")
    if plan.page_size <= 0 or plan.num_requests <= 0 or plan.rows <= 0:
        raise ValueError("CSA2 MAIN bank plan needs rows, requests and a positive page size")
    if marks is not None and (
        marks.dtype != torch.uint8
        or marks.shape != (plan.rows,)
        or not marks.is_contiguous()
        or marks.device != bank.device
    ):
        raise ValueError("CSA2 MAIN bank marks must be contiguous CUDA UINT8[rows]")
    # Rows on 8-byte strides read their payloads as words, so any row can be read.
    readable = main_pool.shape[0] > 0 and main_pool.stride(0) % 8 == 0
    _compiled(pages.dtype, plan.page_size, marks is not None, readable)(
        main_pool,
        pages,
        offsets,
        bank,
        # An unmarked decode reads no marks; any UINT8 vector stands in.
        marks if marks is not None else bank[0],
        plan.base,
        plan.rows,
        plan.num_requests,
    )


_RECORDS_CACHE_KEY = "csa2_prims_ts_main_records"
_RETIRED_RECORDS_CACHE_KEY = "csa2_prims_ts_main_records_retired"
_OFFSETS_CACHE_KEY = "csa2_prims_ts_main_records_offsets"
# A selection bank holds one row per MAIN selection of a generation tile (rows x 512), so
# it scales with the largest tile; tiles beyond this bound keep decoding inside the kernel.
_MAX_RECORDS_BYTES = 1 << 30


def main_records_rows_limit() -> int:
    """Most rows a device's selection bank holds."""
    return _MAX_RECORDS_BYTES // _BANK_ROW_BYTES


def main_records_storage(
    arenas: dict, device: torch.device, rows: int, capacity: int = 0
) -> torch.Tensor:
    """Return a UINT8[rows,512] view of the device's retained selection bank.

    It holds one decoded row per MAIN selection of a tile that has no MAIN bank, for
    :func:`decode_main_records`. It is allocated once, outside CUDA Graph capture, at
    ``capacity`` rows (the selections of the largest tile its plans serve), at least
    :func:`main_bank_rows_limit` rows and at most :func:`main_records_rows_limit`, so CUDA
    Graphs that captured it never see it move. A larger tile replaces it with a larger
    allocation and retains the previous one for the graphs that captured it.
    """
    if not 0 < rows <= main_records_rows_limit():
        raise ValueError("CSA2 PrimTS selection bank rows exceed its reserved capacity")
    key = (_RECORDS_CACHE_KEY, device)
    storage = arenas.get(key)
    if storage is None or storage.shape[0] < rows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CSA2 PrimTS selection bank cannot be allocated during CUDA Graph capture"
            )
        if storage is not None:
            arenas.setdefault((_RETIRED_RECORDS_CACHE_KEY, device), []).append(storage)
        capacity = min(max(capacity, rows, main_bank_rows_limit()), main_records_rows_limit())
        storage = torch.empty((capacity, _BANK_ROW_BYTES), device=device, dtype=torch.uint8)
        arenas[key] = storage
        # The direct decode reads no offsets; any INT64 vector stands in.
        arenas[(_OFFSETS_CACHE_KEY, device)] = torch.zeros(1, device=device, dtype=torch.int64)
    return storage[:rows]


def decode_main_records(
    main_pool: torch.Tensor, records: torch.Tensor, bank: torch.Tensor, arenas: dict
) -> None:
    """Decode MAIN pool rows ``records[r, c]`` into bank rows ``r * width + c`` on the current stream.

    Args:
        main_pool: The owner's packed MAIN records, CUDA UINT8[pool_rows,288]
            with a contiguous inner axis; outer strides may be padded.
        records: Contiguous CUDA INT32[rows, width] pool rows, negative for none.
        bank: Contiguous CUDA UINT8[rows * width, 512] destination from
            :func:`main_records_storage`.
        arenas: The arenas that hold ``bank``.

    Bank row ``r * width + c`` decodes record ``records[r, c]``; a negative or out-of-pool
    record leaves its bank row as it was, since no selection reads it. Rows convert exactly
    as :func:`decode_main_bank`.
    """
    if (
        main_pool.dtype != torch.uint8
        or main_pool.ndim != 2
        or main_pool.shape[1] != _MAIN_ROW_BYTES
        or main_pool.stride(1) != 1
        or main_pool.data_ptr() % 16
    ):
        raise ValueError("CSA2 MAIN bank source must be an aligned CUDA UINT8[N,288] pool")
    if records.dtype != torch.int32 or records.ndim != 2 or not records.is_contiguous():
        raise ValueError("CSA2 MAIN selection records must be contiguous INT32[rows, width]")
    rows = records.numel()
    if (
        bank.dtype != torch.uint8
        or bank.shape != (rows, _BANK_ROW_BYTES)
        or not bank.is_contiguous()
        or bank.data_ptr() % 16
        or bank.device != main_pool.device
        or records.device != bank.device
    ):
        raise ValueError("CSA2 MAIN selection bank must be contiguous aligned CUDA UINT8[rows,512]")
    if rows == 0:
        return
    offsets = arenas[(_OFFSETS_CACHE_KEY, bank.device)]
    readable = main_pool.shape[0] > 0 and main_pool.stride(0) % 8 == 0
    _compiled(torch.int32, 1, False, readable, True)(
        main_pool, records.view(rows, 1), offsets, bank, bank[0], 0, rows, 1
    )
