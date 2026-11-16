# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 / sm_103 / tp8."""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

native_ops = load_target_ops(__name__)  # torch.ops.mv2_glm_moe_dsa_glm_5_2_nvfp4_sm_103_tp8
