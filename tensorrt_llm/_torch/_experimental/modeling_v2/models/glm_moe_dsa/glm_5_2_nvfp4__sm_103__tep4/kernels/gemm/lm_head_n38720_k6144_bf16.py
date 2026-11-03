# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 TP4 local-vocabulary LM-head GEMM with per-M dispatch.

The vocabulary-parallel ``lm_head`` leaves each rank one ``[38720, 6144]``
shard of the 154880-token vocabulary. This is that shard's local projection:
contiguous BF16 ``input[M,6144]``, ``weight[38720,6144]`` and
``output[M,38720]``. The shard gather and the fp32 widening are the caller's.

Every M from 1 to 64 has an instance (a decode step's request count can drain
through any value), plus the context buckets 128 to 8192 and MTP5's 192 verify
rows (``MTP_DECODE_M_VALUES``, built only for an engine drafting more than one
token). The M=1..64 rows are
weight-read bound: each instance moves the 476 MB shard once, at the same
76-85 us as cuBLAS. The swapped orientation writes the logical
``[38720, M]`` product through a column-major epilogue into the row-major
``[M, 38720]`` output.

Why the target owns this GEMM rather than calling ``catalog/gemm/cublas_mm``: at decode
M the two run at the same speed, so the reason is the bits. Each M here has one fixed
schedule (tile, cluster, orientation, K order) in this tree; cuBLASLt picks its own
algorithm (tile and K split, hence the fp32 summation order) per shape, heuristic and
library version, so its bf16 logits can differ from these in the last bit. Those bits
decide the greedy tokens, and the target's served outputs, its accuracy and MTP
acceptance gates and the bitwise checks against them were taken on this kernel. Moving
to cuBLAS is therefore a numerics change to be gated on its own, not a drop-in
replacement. The stock model's
``LMHead`` does not call ``cublas_mm`` either (``UnquantizedLinearMethod`` uses it only
with ``use_custom_cublas_mm``).
"""

from ._projection_module_dispatch import ProjectionAbi, make_build, persistent_instance
from ._swapped_dense_gemm_persistent import GemmSchedule

# Own symbol name: the persistent GEMM's kernel object is shared by every projection, so
# without a prefix this one keeps whichever name the last-built projection set, and a
# profile attributes the lm_head launches to that op.
ABI = ProjectionAbi("lm_head", 38_720, 6_144, kernel_name_prefix="glm52_lm_head")

_SMALL_M = {
    1: GemmSchedule(True, (128, 32), (2, 1)),
    2: GemmSchedule(False, (64, 32), (8, 1)),
    4: GemmSchedule(False, (64, 32), (8, 1)),
    8: GemmSchedule(False, (64, 32), (8, 1)),
    16: GemmSchedule(False, (64, 32), (4, 1)),
    24: GemmSchedule(True, (128, 32), (4, 1)),
    32: GemmSchedule(False, (64, 32), (4, 1)),
    48: GemmSchedule(True, (128, 64), (4, 1), raster_along="n"),
    64: GemmSchedule(True, (128, 64), (4, 1)),
}
_CONTEXT_M = {
    128: GemmSchedule(True, (128, 128), (4, 1)),
    256: GemmSchedule(True, (256, 176), (2, 1), orientation="direct"),
    512: GemmSchedule(True, (256, 176), (2, 1), orientation="direct"),
    1_024: GemmSchedule(True, (256, 176), (2, 1), orientation="direct"),
    2_048: GemmSchedule(True, (256, 176), (2, 1), orientation="direct"),
    4_096: GemmSchedule(True, (256, 256), (2, 1), orientation="direct"),
    8_192: GemmSchedule(True, (256, 256), (2, 1), swizzle_size=8),
}


def _schedule(m: int) -> GemmSchedule:
    """An M without its own schedule takes the next measured bucket's."""
    if m in _SMALL_M:
        return _SMALL_M[m]
    return _SMALL_M[min(b for b in _SMALL_M if b >= m)]


# MTP5's verify rows (32 requests x 6): the 256 bucket's schedule at exactly 192 rows.
# B300, cold weights: 88.4 us, against 93.8 us for the pad to 256 those rows took (zero
# rows, a copy and the 256-row GEMM; the gather then also widens 64 padded rows), and
# 92.4 / 94.6 / 98.8 / 105.5 us for (256, 192) direct, (128, 192), (256, 224) direct and
# (128, 96).
_MTP_M = {192: GemmSchedule(True, (256, 176), (2, 1), orientation="direct")}
MTP_DECODE_M_VALUES = tuple(_MTP_M)

SCHEDULE_BY_M = (
    {m: persistent_instance(_schedule(m)) for m in range(1, 65)}
    | {m: persistent_instance(s) for m, s in _CONTEXT_M.items()}
    | {m: persistent_instance(s) for m, s in _MTP_M.items()}
)
SUPPORTED_M_VALUES = tuple(SCHEDULE_BY_M)

build = make_build(ABI, SCHEDULE_BY_M)
