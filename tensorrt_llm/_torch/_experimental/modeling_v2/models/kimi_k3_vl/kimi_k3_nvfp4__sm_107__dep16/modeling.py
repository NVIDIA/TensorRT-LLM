# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 NVFP4 on Rubin (sm107), attention DP over 16 ranks, routed experts expert parallel over the same 16.

Runs the Rubin K3 model in ``kimi_k3_vl/k3_rubin`` (K3 Rubin agentic optimizations
merged onto main): text model, MLA, KDA, DSpark drafter and worker, K3 CuTe DSL kernels.
"""

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.wrapper import (
    KimiK3ForConditionalGeneration,
)


class ModelingV2KimiK3Nvfp4Sm107Dep16(KimiK3ForConditionalGeneration):
    """Kimi K3 NVFP4, sm107, dep16; the k3_rubin model."""
