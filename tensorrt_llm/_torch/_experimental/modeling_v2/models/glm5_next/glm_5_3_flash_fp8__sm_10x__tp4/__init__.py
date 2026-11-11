# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash block-FP8 on sm_100 / sm_103: tensor-parallel attention and experts over 4 ranks."""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

native_ops = load_target_ops(__name__)  # torch.ops.mv2_glm5_next_glm_5_3_flash_fp8_sm_10x_tp4
