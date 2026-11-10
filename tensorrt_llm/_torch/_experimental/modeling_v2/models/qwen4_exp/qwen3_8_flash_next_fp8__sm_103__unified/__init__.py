# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next-FP8 / sm_103 / unified: tp1, tp2, tep2, dep2, tp4, tep4 and dep4."""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

# torch.ops.mv2_qwen4_exp_qwen3_8_flash_next_fp8_sm_103_unified
native_ops = load_target_ops(__name__)
