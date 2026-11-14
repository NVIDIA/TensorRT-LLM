# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-Flash-Next-FP8 / sm_103 / unified: tp1, tp2, tep2, dep2, tp4, tep4 and dep4.

The collective matrices (`tests/unittest/.../qwen3_8_flash_next_fp8__sm_103__unified/comm`) run in
the pre-merge GB300 multi-GPU stage (`l0_gb300.yml`), which CI schedules only when a file it lists
as multi-GPU changes (`getMultiGpuFileChanged` in `jenkins/L0_MergeRequest.groovy`); this
directory is not among them. A change to `csrc/` or `ops/comm/` is therefore run with
`/bot run --add-multi-gpu-test` (`docs/source/developer-guide/ci-overview.md`, Multi-GPU Tests).
"""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

# torch.ops.mv2_qwen4_exp_qwen3_8_flash_next_fp8_sm_103_unified
native_ops = load_target_ops(__name__)
