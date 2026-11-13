# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWIGLU_MXFP8_ROWS (K5 U1): the launch geometry of ``mv2_minimax_m3::minimax_m31_swiglu_oai_mxfp8``'s rows form, in
plain Python (no torch, no CUDA) so it runs in CPU tests. Shared by the DEP2CTX and TP2CTX targets (both call the op).

``invokeSwigluOaiMxfp8`` (``cpp/tensorrt_llm/kernels/minimaxM31/swigluOaiMxfp8Sm103.cu``) runs one CTA of ``d / 8``
threads per row. The one-row launch sizes its grid for 2048 threads per SM (``2048 / threads`` CTAs), but an sm_107 SM
holds 1024, so at d 3072 two 384-thread CTAs are resident (not five) with one 32-byte (gate, up) load each in flight.
The rows form (from :data:`ROWS_MIN` rows, unless ``TRTLLM_M31_DISABLE_SWIGLU_MXFP8_ROWS`` is set) keeps ``k_rows``
rows in flight per CTA on an occupancy-true grid, ``min(pad128(rows), resident CTAs x SMs)``. CTA ``c`` still visits
rows ``c, c + grid, ...`` in that order, only ``k_rows`` at a time, and every row runs the one-row kernel's
instructions, so every output byte is the one-row kernel's (bitwise by construction). The functions mirror the C++
formulas one for one.
"""

from typing import List, Optional, Tuple

ELTS_PER_THREAD = 8
#: The rows form engages from this many rows (``kSwigluRowsMinRows``).
ROWS_MIN = 1024
#: ``kSwigluRowsDefaultK``; ``TRTLLM_M31_SWIGLU_MXFP8_ROWS_K`` picks another of :data:`K_ROWS_CHOICES`.
K_ROWS_DEFAULT = 4
K_ROWS_CHOICES = (2, 4, 8)
#: The thread count the one-row grid assumes per SM, and what sm_100 / sm_103 / sm_107 hold.
ONE_ROW_THREADS_PER_SM = 2048
SM107_THREADS_PER_SM = 1024
REGS_PER_SM = 65536
REG_ALLOC_UNIT = 8  # registers per thread, rounded up (256 per warp)
MAX_CTAS_PER_SM = 32
TILE_ROWS = 128
DISABLE_ENV = "TRTLLM_M31_DISABLE_SWIGLU_MXFP8_ROWS"
K_ROWS_ENV = "TRTLLM_M31_SWIGLU_MXFP8_ROWS_K"


def threads(d: int) -> int:
    """The block: one 8-column chunk per thread."""
    return d // ELTS_PER_THREAD


def padded_rows(rows: int) -> int:
    """Rows rounded up to the 128-row scale tile (the pad rows get zero scales)."""
    return (rows + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS


def one_row_grid(rows: int, d: int, sms: int) -> int:
    """The one-row launch's grid (also the kill switch's): ``min(pad128(rows), SMs x max(1, 2048 / threads))``."""
    return min(padded_rows(rows), sms * max(1, ONE_ROW_THREADS_PER_SM // threads(d)))


def ctas_per_sm(
    block: int,
    threads_per_sm: int = SM107_THREADS_PER_SM,
    regs_per_thread: Optional[int] = None,
    regs_per_sm: int = REGS_PER_SM,
) -> int:
    """Resident CTAs per SM as the occupancy API counts them for a kernel without shared memory: the thread limit and,
    when given, the register file (``regs_per_thread`` rounded up to the allocation unit). The launcher reads this from
    ``cudaOccupancyMaxActiveBlocksPerMultiprocessor``; this mirror documents the arithmetic."""
    ctas = min(MAX_CTAS_PER_SM, threads_per_sm // block)
    if regs_per_thread is not None:
        regs = -(-regs_per_thread // REG_ALLOC_UNIT) * REG_ALLOC_UNIT
        ctas = min(ctas, regs_per_sm // (regs * block))
    return ctas


def grid(rows: int, ctas: int, sms: int) -> int:
    """The rows form's grid: ``max(1, min(pad128(rows), ctas x SMs))``. Every SM gets resident CTAs at any row count
    (``ceil(pad128 / k_rows)`` CTAs would leave SMs idle from 1024 rows up to about ``k_rows x ctas x SMs``); a CTA
    with fewer than ``k_rows`` rows left batches the rows it has."""
    return max(1, min(padded_rows(rows), ctas * sms))


def uses_rows_form(rows: int, disabled: bool, ctas: int) -> bool:
    """Whether ``invokeSwigluOaiMxfp8`` launches the rows form: switch unset, at least :data:`ROWS_MIN` rows and at
    least one resident CTA (else the one-row launch)."""
    return not disabled and rows >= ROWS_MIN and ctas >= 1


def cta_batches(cta: int, grid_size: int, rows: int, k_rows: int) -> List[List[int]]:
    """The rows CTA ``cta`` loads per main-loop iteration, in order: rows ``base + i * grid`` (i < k_rows) below
    ``rows``, all loaded before the first of them is computed."""
    out = []
    for base in range(cta, rows, grid_size * k_rows):
        out.append([base + i * grid_size for i in range(k_rows) if base + i * grid_size < rows])
    return out


def cta_rows(cta: int, grid_size: int, rows: int, k_rows: int) -> List[Tuple[int, int, int]]:
    """(iteration, slot, row) of every real row CTA ``cta`` computes, in its order. The C++ compute loop stops at the
    first slot past ``rows``; the slots of an iteration are increasing, so that is the same set."""
    out = []
    for it, base in enumerate(range(cta, rows, grid_size * k_rows)):
        for i in range(k_rows):
            row = base + i * grid_size
            if row >= rows:
                break
            out.append((it, i, row))
    return out


def cta_pad_rows(cta: int, grid_size: int, rows: int) -> List[int]:
    """The pad rows ``[rows, pad128(rows))`` whose scales CTA ``cta`` zeroes (the one-row kernel's loop, unchanged)."""
    return list(range(rows + cta, padded_rows(rows), grid_size))


def one_row_cta_rows(cta: int, grid_size: int, rows: int) -> List[int]:
    """The rows the one-row kernel's CTA ``cta`` computes, in its order."""
    return list(range(cta, rows, grid_size))


def loads_in_flight_bytes(ctas: int, block: int, k_rows: int) -> int:
    """Bytes of (gate, up) loads one SM can have in flight: two 16-byte loads per thread per row."""
    return ctas * block * k_rows * 32


def bytes_moved(rows: int, d: int, write_h: bool) -> int:
    """HBM bytes of one call: x bf16 [rows, 2d] read, E4M3 q [rows, d] and the swizzled scales pad128(rows) * d / 32
    written, and the bf16 h [rows, d] when written."""
    return (
        rows * 2 * d * 2 + rows * d + padded_rows(rows) * d // 32 + (rows * d * 2 if write_h else 0)
    )


def ms_per_1k(delta_us: float, calls: int, rows: int) -> float:
    """The cost model: a per-call saving of ``delta_us`` over ``calls`` launches per ``rows``-row chunk, in ms per 1K
    new tokens (delta_us x calls / rows)."""
    return delta_us * calls / rows


def env_word(value: Optional[str]) -> str:
    """``swigluEnvWord``: the value without surrounding whitespace, lower-cased (``""`` when unset)."""
    return "" if value is None else value.strip().lower()


def switch_set(value: Optional[str]) -> bool:
    """``swigluRowsForm``'s parse of the kill switch: 1 / true / on / yes (any case, surrounding whitespace ignored)
    set it; anything else, or unset, leaves the rows form on."""
    return env_word(value) in ("1", "true", "on", "yes")


def switch_warns(value: Optional[str]) -> bool:
    """Whether ``swigluRowsForm`` warns about the kill switch's value: neither a set nor an unset word
    (0 / false / off / no / empty)."""
    return env_word(value) not in ("", "0", "false", "off", "no", "1", "true", "on", "yes")


def k_rows_from_env(value: Optional[str]) -> int:
    """``swigluRowsForm``'s parse of ``TRTLLM_M31_SWIGLU_MXFP8_ROWS_K``: ``2``, ``4`` or ``8`` (surrounding whitespace
    ignored), else :data:`K_ROWS_DEFAULT` (with a warning when the value is not empty)."""
    word = env_word(value)
    return int(word) if word in ("2", "4", "8") else K_ROWS_DEFAULT
