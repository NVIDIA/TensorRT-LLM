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

"""Host-side access to the DWDP fabric-gather kernel (``trtllm::dwdp_fabric_gather``).

Opt-in via ``TRTLLM_DWDP_PREFETCH_GATHER=1``: an SM kernel issuing 16-byte loads out-runs the copy
engine but occupies SMs for the whole step, so it only pays off while compute hides behind the copy.
``TRTLLM_DWDP_GATHER_LIB`` loads the op from a prebuilt library; ``TRTLLM_DWDP_GATHER_BLOCKS`` and
``TRTLLM_DWDP_GATHER_THREADS`` size the grid; the launcher reads ``TRTLLM_DWDP_GATHER_CLUSTER`` and
``TRTLLM_DWDP_GATHER_KERNEL=tma``.  See ``apply_compute_side_defaults`` for the compute-side knobs.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import torch

try:
    from tensorrt_llm.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

_ENABLE_ENV = "TRTLLM_DWDP_PREFETCH_GATHER"
_LIB_ENV = "TRTLLM_DWDP_GATHER_LIB"
_BLOCKS_ENV = "TRTLLM_DWDP_GATHER_BLOCKS"
_THREADS_ENV = "TRTLLM_DWDP_GATHER_THREADS"

#: Load width of the kernel; every descriptor pointer and byte count must be a multiple of it.
LINE_BYTES = 16

#: Block size the op accepts: a positive multiple of 32 up to 1024 (the kernel's launch bound).
MAX_THREADS = 1024


def _int_env(name: str, default: int) -> int:
    """Integer env knob; an unparsable value logs a warning and keeps ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"[DWDP] {name}={raw!r} is not an integer; using {default}")
        return default


def _threads() -> int:
    threads = _int_env(_THREADS_ENV, MAX_THREADS)
    if threads <= 0 or threads % 32 or threads > MAX_THREADS:
        logger.warning(
            f"[DWDP] {_THREADS_ENV}={threads} must be a positive multiple of 32 up to "
            f"{MAX_THREADS}; using {MAX_THREADS}"
        )
        return MAX_THREADS
    return threads


#: Threads per block (``TRTLLM_DWDP_GATHER_THREADS`` overrides).  Bytes in flight scale with
#: blocks x threads, so 1024 keeps the fabric saturated from as few SMs as possible.
THREADS = _threads()

#: Total blocks, split evenly across the groups (peers) of one layer: the smallest grid that still
#: saturates the fabric.  Each row's blocks launch as one cluster so a row occupies whole TPCs.
TARGET_BLOCKS = 30

_op: Optional[Callable] = None
_probed = False


def is_enabled() -> bool:
    """True if the env knob asks for the gather transport."""
    return os.environ.get(_ENABLE_ENV, "0") not in ("", "0", "false", "False")


def total_blocks() -> int:
    """Total gather blocks, from ``TRTLLM_DWDP_GATHER_BLOCKS`` or ``TARGET_BLOCKS``."""
    return max(1, _int_env(_BLOCKS_ENV, TARGET_BLOCKS))


def blocks_per_row(rows: int) -> int:
    """Blocks to give each group so the whole grid lands near ``TARGET_BLOCKS``."""
    return max(1, total_blocks() // max(rows, 1))


#: Compute-side knobs the SM gather sets for itself unless the user pinned them.
_SM_RESERVE_ENV = "TRTLLM_CUTEDSL_SM_RESERVE"
_DENSE_GEMM_SCHEDULER_ENV = "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER"


def apply_compute_side_defaults() -> None:
    """Tune the CuTeDSL persistent kernels for a resident gather; call before any kernel compiles.

    Reserves the gather's block count of SMs (``TRTLLM_CUTEDSL_SM_RESERVE``) and pins the Rubin
    dense GEMM to the CLC dynamic scheduler (``TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER``) so static
    persistent grids do not run a second wave on the occupied SMs.  An explicit user setting wins.
    """
    reserve = str(total_blocks() if THREADS >= 512 else 0)
    applied = []
    if os.environ.setdefault(_SM_RESERVE_ENV, reserve) == reserve:
        applied.append(f"{_SM_RESERVE_ENV}={reserve}")
    if os.environ.setdefault(_DENSE_GEMM_SCHEDULER_ENV, "clc_dynamic") == "clc_dynamic":
        applied.append(f"{_DENSE_GEMM_SCHEDULER_ENV}=clc_dynamic")
    if applied:
        logger.info(f"[DWDP] SM gather on: {' '.join(applied)} (set an explicit value to override)")


def gather_op() -> Optional[Callable]:
    """Resolve the op once, honoring ``TRTLLM_DWDP_GATHER_LIB``; None if unavailable."""
    global _op, _probed
    if _probed:
        return _op
    _probed = True

    lib = os.environ.get(_LIB_ENV)
    if lib:
        try:
            torch.ops.load_library(lib)
        except OSError as exc:
            logger.warning(f"[DWDP] {_LIB_ENV}={lib!r} could not be loaded: {exc}")

    try:
        _op = torch.ops.trtllm.dwdp_fabric_gather
    except (AttributeError, RuntimeError):
        _op = None
    return _op


def build_plan(
    groups: List[List[Tuple[int, int, int]]], device: torch.device
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """Turn per-group ``(dst_ptr, src_ptr, size_bytes)`` descriptors into launch arguments.

    Each group becomes one grid row walked by the same blocks.  Returns None (caller keeps the copy
    engine) if any descriptor is not ``LINE_BYTES``-aligned; the kernel has no unaligned path.

    Returns:
        ``(dst_ptrs, src_ptrs, lines, row_starts, blocks_per_row)`` with device-resident int64
        tensors, or None if the plan cannot be gathered.
    """
    dst_ptrs: List[int] = []
    src_ptrs: List[int] = []
    lines: List[int] = []
    row_starts: List[int] = [0]

    for descriptors in groups:
        for dst_ptr, src_ptr, size in descriptors:
            if dst_ptr % LINE_BYTES or src_ptr % LINE_BYTES or size % LINE_BYTES:
                return None
            dst_ptrs.append(dst_ptr)
            src_ptrs.append(src_ptr)
            lines.append(size // LINE_BYTES)
        row_starts.append(len(dst_ptrs))

    if not dst_ptrs:
        return None

    def as_tensor(values: List[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device)

    rows = len(row_starts) - 1
    return (
        as_tensor(dst_ptrs),
        as_tensor(src_ptrs),
        as_tensor(lines),
        as_tensor(row_starts),
        blocks_per_row(rows),
    )
