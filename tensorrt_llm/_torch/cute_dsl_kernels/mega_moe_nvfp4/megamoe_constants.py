# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the fused fc1+fc2 MegaMoE path.

Deliberately cutlass-free (a TRT-LLM-local trim of the upstream
``common/megamoe_constants.py``): the package ``__init__`` imports these for
capability probing on non-SM100 / no-cutlass-dsl hosts, so this module must NOT
``from cutlass...`` import. Only the constants actually consumed by the ported
kernel package are kept; the cutlass-typed ``Log2E`` / ``Fp32Max`` upstream
constants are unused here (the kernels inline their own ``cutlass.Float32``).
"""

Nvfp4BlockSize = 16
SfPaddingBlock = 128
TmaLeadingDimByteAlign = 16

Nvfp4E2M1Max = 6.0
Fp8E4M3FNMax = 448.0

SupportedMmaTileM = (128, 256)
SupportedMmaTileN = (64, 128, 256)
