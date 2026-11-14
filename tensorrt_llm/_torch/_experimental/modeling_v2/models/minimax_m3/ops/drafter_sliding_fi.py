# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family SM107_DRAFT_FMHA_SM100F (the TP8 and TP4 targets install it from here): FlashInfer's trtllm-gen FMHA for the
standalone DSpark drafter's attention on SM 10.7.

The stock library declines SM 10.7, so the drafter's sliding-window decode falls through to ``thop.attention``'s
in-tree trtllm-gen runner, which picks the SM107a Q64 KeepsAb kernel. FlashInfer's own selector runs its SM100f
kernels on SM 10.7 and picks the Q8 SwapsAb one. A different kernel, so not bitwise the in-tree output (the draft can
change, the verified output does not). Everything else
is the stock library: ``is_available`` keeps its per-layer checks (head size, KV dtype, layout), so a geometry it does
not serve keeps the in-tree runner. The installing target owns the kill switch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import get_sm_version


class M3DrafterSlidingFiFmha(FlashInferTrtllmGenFmha):
    """FlashInfer's trtllm-gen FMHA, available on SM 10.7 (the stock class declines it)."""

    @classmethod
    def is_environment_available(cls) -> bool:
        if not IS_FLASHINFER_AVAILABLE or cls._missing_fused_nanobind_ops():
            return False
        return get_sm_version() == 107


def install_sliding_fi(draft_model: torch.nn.Module) -> int:
    """Put ``M3DrafterSlidingFiFmha`` ahead of the fallback in the FMHA list of every attention module of
    ``draft_model`` it can serve. Call before warmup. Returns the number of attention layers it was installed on."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    count = 0
    for module in draft_model.modules():
        backend = getattr(module, "attn", None)
        if not isinstance(backend, TrtllmAttention) or not hasattr(backend, "_fmha_manager"):
            continue
        libs = backend._fmha_manager.fmha_libs
        if any(isinstance(lib, M3DrafterSlidingFiFmha) for lib in libs):
            continue
        if not M3DrafterSlidingFiFmha.is_available(backend):
            continue
        at = next((i for i, lib in enumerate(libs) if isinstance(lib, FallbackFmha)), len(libs))
        libs.insert(at, M3DrafterSlidingFiFmha(backend))
        count += 1
    return count
