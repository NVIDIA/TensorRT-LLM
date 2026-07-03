# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared constants for the fused fc1+fc2 MegaMoE path."""

from cutlass.cutlass_dsl import Float32

Log2E = Float32(1.4426950408889634)
Fp32Max = Float32(3.40282346638528859812e38)

Nvfp4BlockSize = 16
Mxfp8BlockSize = 32
SfPaddingBlock = 128
TmaLeadingDimByteAlign = 16

Nvfp4E2M1Max = 6.0
Fp8E4M3FNMax = 448.0
Fp8E5M2Max = 57344.0

Nvfp4E2M1RcpLimit = 1.0 / Nvfp4E2M1Max
Fp8E4M3RcpLimit = 1.0 / Fp8E4M3FNMax
Fp8E5M2RcpLimit = 1.0 / Fp8E5M2Max

Nvfp4E2M1RcpLimit = 1.0 / Nvfp4E2M1Max
Fp8E4M3RcpLimit = 1.0 / Fp8E4M3FNMax
Fp8E5M2RcpLimit = 1.0 / Fp8E5M2Max

SupportedMmaTileM = (128, 256)
SupportedMmaTileN = (64, 128, 256)
