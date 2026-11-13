# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3.1-Flash NVFP4 / sm_107 / dep2ctx.

The disaggregated context (prefill) worker: attention DP 2 and EP2 experts.
"""

from ...._native import load_target_ops

# torch.ops.mv2_minimax_m3_minimax_m3_1_flash_nvfp4_sm_107_dep2ctx
native_ops = load_target_ops(__name__)
