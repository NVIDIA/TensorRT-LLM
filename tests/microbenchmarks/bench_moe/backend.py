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
    MEGAMOE_DEEPGEMM = "MEGAMOE_DEEPGEMM"
    MEGAMOE_CUTEDSL = "MEGAMOE_CUTEDSL"


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
    """Install a local sentinel package when optional CUTEDSL imports are absent.

    Production MoE modules import the ``cutedsl`` package at module load time.
    ``bench_moe`` can still benchmark non-CUTEDSL backends in environments that
    do not package CUTLASS DSL; keep that fallback local to the benchmark rather
    than weakening the production package.
    """
    module_name = "tensorrt_llm._torch.moe.fused_moe.cutedsl"
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ImportError as exc:
        import_error = exc

    class CuteDslFusedMoEBase:
        @classmethod
        def can_implement(cls, p, d):
            from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
                MoEEligibility,
                MoERejectReason,
            )

            return MoEEligibility.no(
                MoERejectReason.DEP_MISSING, f"CUTLASS DSL is unavailable: {import_error}"
            )

        def __init__(self, *_args, **_kwargs):
            raise RuntimeError(f"CUTLASS DSL is unavailable: {import_error}")

    def find_cutedsl_grouped_gemm_leaf(quant_algo: object) -> type:
        del quant_algo
        return CuteDslFusedMoEBase

    module = types.ModuleType(module_name)
    module.CuteDslFusedMoEBase = CuteDslFusedMoEBase
    # Stubs for the leaves the MoE package imports by name; they inherit the
    # root stub's DEP_MISSING verdict.
    for leaf_name in (
        "TrtllmCutedslGroupedGemmNvfp4Impl",
        "TrtllmCutedslGroupedGemmBf16Impl",
        "TrtllmCutedslFusedFc12Nvfp4Impl",
        "TrtllmCutedslDenseGemmNvfp4Impl",
    ):
        setattr(module, leaf_name, type(leaf_name, (CuteDslFusedMoEBase,), {}))
    module.find_cutedsl_grouped_gemm_leaf = find_cutedsl_grouped_gemm_leaf
    module.cutedsl_grouped_gemm_leaf = find_cutedsl_grouped_gemm_leaf
    sys.modules[module_name] = module


def find_backend_class(backend_type: MoeBackendType, quant_algo=None):
    """As :func:`get_backend_class`, but ``None`` when no leaf serves the format.

    For a caller enumerating combinations to decide which are worth running,
    "no leaf publishes this format" is an answer and not a failure. Only the
    TRTLLM, CUTLASS and CUTEDSL branches can give it, so the rest is delegated
    rather than restated.

    Names one class. A caller deciding whether a configuration is *runnable*
    wants :func:`find_backend_classes` instead, because for CUTLASS
    ``fp8_block_scales`` that question spans two leaves.
    """
    if backend_type == MoeBackendType.TRTLLM:
        from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import find_trtllm_gen_leaf

        return find_trtllm_gen_leaf(quant_algo)
    if backend_type == MoeBackendType.CUTLASS:
        from tensorrt_llm._torch.moe.fused_moe.cutlass import find_cutlass_leaf

        return find_cutlass_leaf(quant_algo)
    if backend_type == MoeBackendType.CUTEDSL:
        from tensorrt_llm._torch.moe.fused_moe.cutedsl import find_cutedsl_grouped_gemm_leaf

        return find_cutedsl_grouped_gemm_leaf(quant_algo)
    return get_backend_class(backend_type, quant_algo)


def find_backend_classes(backend_type: MoeBackendType, quant_algo=None) -> tuple:
    """The classes a resolution walk would try, in its order; empty if none.

    A tuple and not one class because CUTLASS publishes ``fp8_block_scales``
    twice, as the SM90 and SM120 leaves, and resolution falls through to the
    second when the first rejects. Asking a single pre-picked class prunes
    candidates a run would have served.
    """
    if backend_type == MoeBackendType.CUTLASS:
        from tensorrt_llm._torch.moe.fused_moe.cutlass import cutlass_leaves_in_resolution_order

        return cutlass_leaves_in_resolution_order(quant_algo)
    cls = find_backend_class(backend_type, quant_algo)
    return () if cls is None else (cls,)


def get_backend_class(backend_type: MoeBackendType, quant_algo=None):
    """Import and return the concrete backend class for ``backend_type`` lazily.

    ``quant_algo`` is only consulted for ``TRTLLM``, ``CUTLASS`` and
    ``CUTEDSL``, whose leaves are keyed by quantization format; every other
    backend is one class. Raises when no leaf publishes the format -- see
    :func:`find_backend_class` for the lookup that reports absence instead.
    """
    if backend_type == MoeBackendType.CUTLASS:
        from tensorrt_llm._torch.moe.fused_moe.cutlass import cutlass_leaf

        return cutlass_leaf(quant_algo)
    if backend_type == MoeBackendType.TRTLLM:
        # Leaves are keyed by quant, so the format has to be named.
        # ``trtllm_gen_leaf`` returns the native leaf for a quantized format
        # and the FlashInfer one for the unquantized format.
        from tensorrt_llm._torch.moe.fused_moe.fused_moe_trtllm_gen import trtllm_gen_leaf

        return trtllm_gen_leaf(quant_algo)
    if backend_type == MoeBackendType.CUTEDSL:
        from tensorrt_llm._torch.moe.fused_moe.cutedsl import cutedsl_grouped_gemm_leaf

        return cutedsl_grouped_gemm_leaf(quant_algo)
    if backend_type == MoeBackendType.DEEPGEMM:
        from tensorrt_llm._torch.moe.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE

        return DeepGemmFusedMoE
    if backend_type == MoeBackendType.DENSEGEMM:
        from tensorrt_llm._torch.moe.fused_moe.fused_moe_densegemm import DenseGEMMFusedMoE

        return DenseGEMMFusedMoE
    if backend_type == MoeBackendType.MEGAMOE_DEEPGEMM:
        from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoEDeepGemm

        return MegaMoEDeepGemm
    if backend_type == MoeBackendType.MEGAMOE_CUTEDSL:
        from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl

        return MegaMoECuteDsl
    raise ValueError(f"unknown MoE backend {backend_type!r}")
