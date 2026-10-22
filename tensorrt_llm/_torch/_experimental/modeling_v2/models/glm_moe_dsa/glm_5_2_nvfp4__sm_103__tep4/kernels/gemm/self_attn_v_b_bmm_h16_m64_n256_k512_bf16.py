# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 absorbed-MLA V-B batched BF16 GEMM.

The registry ABI is ``build(active_m=M) -> launcher(input, weight, output)`` for
exact serving views ``[16,M,512] x [16,256,512] -> [16,M,256]``; ``M`` is one of
:data:`ACTIVE_M_CELLS` and selects the schedule group (:data:`SCHEDULE_GROUPS`).
V-B runs on the same
persistent swapped-operand GEMM family as K-B (``self_attn_k_b_bmm_*``: the
token rows are the GEMM's N, the immutable weight is preloaded before
``griddepcontrol.wait``): ``(128, 8)`` for M <= 8 and ``(64, 32)`` for M >= 16,
TMA store (clips the N edge), PDL wait/signal role with the weight preload, one
compiled object per group (dynamic layouts).  Output is
bit-identical to ``torch.bmm`` at every cell.  One grid covers every head and
the input, weight and output keep their serving strides (no copy on the decode
path).
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

from . import _swapped_dense_gemm_pdl_chain as _swappeddensegemmpdlchain
from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersistent
from . import self_attn_k_b_bmm_h16_m64_n512_k192_bf16 as _kbbmm

GemmSchedule = _swappeddensegemmpersistent.GemmSchedule
WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE = _swappeddensegemmpdlchain.WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE

BatchedGemmAbi = _kbbmm.BatchedGemmAbi
make_persistent_build = _kbbmm.make_build


# ---------------------------------------------------------------------------
# V-B owner contract.
# ---------------------------------------------------------------------------

_NUM_HEADS = 16
_ACTIVE_M = 64
_OUTPUT_FEATURES = 256
_INPUT_FEATURES = 512
_INPUT_STRIDE = (512, 8_192, 1)
_OUTPUT_STRIDE = (256, 4_096, 1)


@dataclass(frozen=True)
class BatchedTcgenAbi:
    """Public shape metadata used by the shared numerical report."""

    module_name: str
    num_heads: int
    active_m: int
    output_features: int
    input_features: int


ABI = BatchedTcgenAbi(
    module_name="self_attn.v_b_bmm",
    num_heads=_NUM_HEADS,
    active_m=_ACTIVE_M,
    output_features=_OUTPUT_FEATURES,
    input_features=_INPUT_FEATURES,
)


# Compiled cells: the padded decode row counts of the serving config
# (CUDA-graph buckets 1/2/4/8/16/24/32 requests x MTP 2 rows).  ``active_m`` is
# the GEMM's N in the swapped orientation; one compiled object per schedule
# group serves its whole row range (dynamic layouts).
ACTIVE_M_CELLS: tuple[int, ...] = (2, 4, 8, 16, 32, 48, 64)


def make_abi(active_m: int) -> BatchedGemmAbi:
    """The V-B contract at one compiled cell (strides are M-independent)."""
    if int(active_m) not in ACTIVE_M_CELLS:
        raise ValueError(
            f"{ABI.module_name}: no compiled cell for active_m={active_m}; "
            f"cells are {ACTIVE_M_CELLS}"
        )
    return BatchedGemmAbi(
        module_name=ABI.module_name,
        num_heads=_NUM_HEADS,
        active_m=int(active_m),
        output_features=_OUTPUT_FEATURES,
        input_features=_INPUT_FEATURES,
        kernel_name_prefix="glm52_self_attn_v_b_dense_persistent",
        input_stride=_INPUT_STRIDE,
        output_stride=_OUTPUT_STRIDE,
    )


# Per-M schedule on the persistent swapped-GEMM family, both groups with the
# weight preload role: (128, 8) for M <= 8 and (64, 32) for M >= 16. Every cell
# is bit-identical to torch.bmm. The cells keep the TMA store
# (``use_tma_store=True``): the library's register store is unpredicated on the N edge
# and rejects N < tile_n, so without it M=2/4 vs (128, 8) and M=16 vs (64, 32) cannot
# compile and M=48 writes past its rows.  M = 4 takes the M <= 8 group and
# M = 16 / 48 the M >= 16 group by extrapolation from the nearest measured cell.
SCHEDULE_SMALL_M = GemmSchedule(False, (128, 8), (1, 1))
SCHEDULE_LARGE_M = GemmSchedule(False, (64, 32), (1, 1))
# (upper row bound inclusive, group name) in ascending order.
SCHEDULE_GROUPS: tuple[tuple[int, str], ...] = ((8, "m_le_8"), (64, "m_ge_16"))
_SCHEDULE_BY_GROUP = {"m_le_8": SCHEDULE_SMALL_M, "m_ge_16": SCHEDULE_LARGE_M}


def schedule_group_for_m(active_m: int) -> str:
    """Name of the schedule group serving ``active_m`` (one of ACTIVE_M_CELLS)."""
    if int(active_m) not in ACTIVE_M_CELLS:
        raise ValueError(
            f"{ABI.module_name}: no compiled cell for active_m={active_m}; "
            f"cells are {ACTIVE_M_CELLS}"
        )
    for upper, name in SCHEDULE_GROUPS:
        if int(active_m) <= upper:
            return name
    raise AssertionError("SCHEDULE_GROUPS does not cover ACTIVE_M_CELLS")


def rows_of_group(group: str) -> tuple[int, ...]:
    """The cells one compiled object of ``group`` serves."""
    return tuple(m for m in ACTIVE_M_CELLS if schedule_group_for_m(m) == group)


@lru_cache(maxsize=None)
def _build_factory(group: str):
    rows = rows_of_group(group)
    return make_persistent_build(
        make_abi(rows[-1]),
        _SCHEDULE_BY_GROUP[group],
        pdl_role=WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE,
        m_rows_allowed=rows,
    )


def build(
    *,
    active_m: int = _ACTIVE_M,
    schedule_group: str | None = None,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]:
    """Registry factory.

    ``active_m`` selects the compiled cell (one of :data:`ACTIVE_M_CELLS`) and
    thereby the schedule group; ``schedule_group`` (optional) must agree with
    the cell -- a disagreement is a routing fault and raises.  The runtime
    passes ``{"active_m": M}`` only.
    """
    group = schedule_group_for_m(int(active_m))
    if schedule_group is not None and schedule_group != group:
        raise ValueError(
            f"{ABI.module_name}: schedule_group={schedule_group!r} does not serve "
            f"active_m={active_m} (expected {group!r})"
        )
    return _build_factory(group)()
