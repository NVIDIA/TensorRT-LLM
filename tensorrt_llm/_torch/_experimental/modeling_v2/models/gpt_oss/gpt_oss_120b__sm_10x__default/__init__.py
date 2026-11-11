# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gpt-oss-120b / sm_10x / default: every parallel layout the routing table admits."""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

native_ops = load_target_ops(__name__)  # torch.ops.mv2_gpt_oss_gpt_oss_120b_sm_10x_default
