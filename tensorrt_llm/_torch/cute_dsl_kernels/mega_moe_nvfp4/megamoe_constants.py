# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared constants for the fused fc1+fc2 MegaMoE path."""

Nvfp4BlockSize = 16
SfPaddingBlock = 128
TmaLeadingDimByteAlign = 16

Nvfp4E2M1Max = 6.0
Fp8E4M3FNMax = 448.0

SupportedMmaTileM = (128, 256)
SupportedMmaTileN = (64, 128, 256)
