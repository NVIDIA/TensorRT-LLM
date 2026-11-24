# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference of the MXFP8 form of a BF16 activation, as the K3.1 Rubin kernels make it.

The low-M FP8 GEMM quantizes its BF16 activation in-kernel; the producers that hand it the MXFP8
form instead take the same 1x32 group amax and call the same PTX helpers, so their tests compare
the codes and the scales against this reference. A multi-GPU test that calls it in its workers
registers this module with ``cloudpickle.register_pickle_by_value``.
"""

import torch

#: Values per UE8M0 scale: one 1x32 group.
SF_VEC = 32
#: BF16 bit pattern of ~1e-4; groups with a smaller amax share its scale.
AMAX_FLOOR_BF16 = 0x38D2


def mxfp8_quantize_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(xq, xs)`` of a BF16 ``[M, K]`` activation as the low-M kernel quantizes it: per 1x32
    group, ``e = biased exponent of 2^ceil(log2(amax / 448))`` with ``amax`` floored at
    ``AMAX_FLOOR_BF16``, codes ``satfinite_rn_e4m3(x * 2^(127 - e))`` (the power-of-two multiply
    is exact in BF16)."""
    m, k = x.shape
    g = x.view(m, k // SF_VEC, SF_VEC)
    bits = g.abs().view(torch.int16).to(torch.int32) & 0x7FFF
    amax = bits.amax(dim=-1).clamp_min(AMAX_FLOOR_BF16)
    e = (amax >> 7) - 8 + ((amax & 0x7F) > 0x60).to(torch.int32)
    inv = torch.pow(2.0, (127 - e).float()).to(torch.bfloat16)
    scaled = (g * inv.unsqueeze(-1)).float()
    xq = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)
    return xq, e.to(torch.uint8)
