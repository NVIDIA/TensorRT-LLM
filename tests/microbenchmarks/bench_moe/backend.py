# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local MoE backend registry for ``bench_moe``.

The benchmark should not import unittest-wide MoE helpers at module import time:
those helpers import every backend, including CUTEDSL, even when a run only asks
for TRTLLM/CUTLASS/MegaMoE. Keep the small registry needed by the benchmark here
and import concrete backend classes lazily.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MoeBackendType(str, Enum):
    """MoE backend identifiers accepted by bench_moe."""

    CUTLASS = "CUTLASS"
    TRTLLM = "TRTLLM"
    CUTEDSL = "CUTEDSL"
    DEEPGEMM = "DEEPGEMM"
    DENSEGEMM = "DENSEGEMM"
    MEGAMOE = "MEGAMOE_DEEPGEMM"


@dataclass
class MoeModelConfig:
    """MoE model shape used by routing-method construction."""

    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    n_group: Optional[int] = None
    topk_group: Optional[int] = None

    def __str__(self) -> str:
        return f"e{self.num_experts}_k{self.top_k}_h{self.hidden_size}_i{self.intermediate_size}"


def resolve_deepseek_group_config(model_config: MoeModelConfig) -> tuple[int, int]:
    """Resolve DeepSeek-V3 routing group settings for built-in or custom shapes."""
    if model_config.n_group is not None and model_config.topk_group is not None:
        return model_config.n_group, model_config.topk_group
    n_group = max(1, model_config.num_experts // 2)
    topk_group = min(n_group, max(1, n_group // 2))
    return n_group, topk_group


def ensure_cute_dsl_importable_for_benchmark() -> None:
    """Install a local sentinel module when optional CUTEDSL imports are absent.

    Production MoE modules import ``fused_moe_cute_dsl`` at module load time.
    ``bench_moe`` can still benchmark non-CUTEDSL backends in environments that
    do not package CUTLASS DSL; keep that fallback local to the benchmark rather
    than weakening the production module.
    """
    module_name = "tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl"
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ImportError as exc:
        import_error = exc

    class CuteDslFusedMoE:
        @classmethod
        def can_implement(cls, *_args, **_kwargs):
            return False, f"CUTLASS DSL is unavailable: {import_error}"

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(f"CUTLASS DSL is unavailable: {import_error}")

    module = types.ModuleType(module_name)
    module.CuteDslFusedMoE = CuteDslFusedMoE
    sys.modules[module_name] = module


def get_backend_class(backend_type: MoeBackendType):
    """Import and return the concrete backend class for ``backend_type`` lazily."""
    if backend_type == MoeBackendType.CUTLASS:
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE

        return CutlassFusedMoE
    if backend_type == MoeBackendType.TRTLLM:
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE

        return TRTLLMGenFusedMoE
    if backend_type == MoeBackendType.CUTEDSL:
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE

        return CuteDslFusedMoE
    if backend_type == MoeBackendType.DEEPGEMM:
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE

        return DeepGemmFusedMoE
    if backend_type == MoeBackendType.DENSEGEMM:
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_densegemm import DenseGEMMFusedMoE

        return DenseGEMMFusedMoE
    if backend_type == MoeBackendType.MEGAMOE:
        from tensorrt_llm._torch.modules.fused_moe.mega_moe import MegaMoEDeepGemm

        return MegaMoEDeepGemm
    raise ValueError(f"unknown MoE backend {backend_type!r}")
