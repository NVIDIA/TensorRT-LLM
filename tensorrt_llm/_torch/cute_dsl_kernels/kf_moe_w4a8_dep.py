# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert-parallel module of the default-off TLLM_MOE_KF_DECODE_DEP path: the same kernel as kf_moe_w4a8, which
accepts the 4-byte-aligned per-rank slices of the per-expert alpha/beta/limit arrays."""

from tensorrt_llm._torch.cute_dsl_kernels.kf_moe_w4a8 import run  # noqa: F401
