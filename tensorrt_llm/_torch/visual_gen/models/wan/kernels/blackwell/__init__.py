# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell (sm100) CuTe DSL kernels of WinoVae's spatial Winograd: the F(4x4, 3x3) transforms
(:mod:`.transforms`) around the shared batched tcgen05 GEMM, the fused F(2,3)-horizontal /
direct-vertical 256-channel conv (:mod:`.fused_conv`), and their torch entry points
(:mod:`.spatial`)."""
