# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash NVFP4 / sm_103 / tp8."""

from ...._native import load_target_ops

# torch.ops.mv2_minimax_m3_minimax_m3_1_flash_nvfp4_sm_103_tp8
native_ops = load_target_ops(__name__)
