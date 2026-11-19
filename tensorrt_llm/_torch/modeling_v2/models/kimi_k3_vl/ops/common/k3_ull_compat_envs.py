# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-in for ``vllm.envs``: the switches the vendored MR8 decode code reads, at their defaults."""

VLLM_KIMI_K3_AUX_ATTN_RES_STREAM = False
VLLM_KIMI_K3_DS_LPC = False
VLLM_KIMI_K3_DS_T2P = False
VLLM_MOE_SKIP_PADDING = False
VLLM_USE_FUSED_MOE_GROUPED_TOPK = False
