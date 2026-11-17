# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine settings the Kimi K3 targets are built for: the GPU architecture, the KV cache block size, the attention
and routed-expert backends, and the model config values `declines` checks (compute dtype, KV cache quantization,
checkpoint quantization, all-reduce strategy, attention backend).

``routing.py`` and each target's ``_check_construction`` apply `declines`: the target raises ``NoTarget`` with the
reasons it returns, and ``route`` declines the deployment with the same reasons before it imports a target, so
``modeling_v2="auto"`` falls back to Modeling V1. The targets' first-step checks and the family's MoE block read the
constants below. This module imports no target and, at import time, nothing but the standard library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

#: The one GPU architecture the targets are written for. sm is part of a target's identity, not a knob: a different
#: SM is a different target. The K3 decode kernels (the tcgen05 GEMVs, the fused MoE) are built for sm_100 only.
SM = (10, 0)

#: The KV cache block size. The MLA generation kernels for K3's 96 query heads exist only at a 64-token page (the
#: built-in model's ``get_model_defaults`` sets it for the same reason).
TOKENS_PER_BLOCK = 64

#: The attention backend whose metadata the targets' attention layers read.
ATTN_BACKEND = "TRTLLM"

#: The routed-expert backend the targets' MoE runs (trtllm-gen).
MOE_BACKEND = "TRTLLM"

#: The ``moe_config.backend`` values a deployment may set: the engine resolves AUTO to `MOE_BACKEND` for Kimi K3
#: before it routes, so the model config the targets are built from holds `MOE_BACKEND`.
MOE_BACKENDS = frozenset({"AUTO", MOE_BACKEND})

#: The all-reduce strategies the targets take, by ``AllReduceStrategy`` name.
ALLREDUCE_STRATEGIES = ("AUTO", "MNNVL")


def dtype_declines(torch_dtype: torch.dtype) -> str | None:
    """Why the model config's compute dtype does not fit, or None for bf16."""
    import torch

    if torch_dtype != torch.bfloat16:
        return f"dtype {torch_dtype}: the Kimi K3 targets compute in bf16"
    return None


def kv_cache_quant_declines(kv_cache_quant_algo: QuantAlgo | str | None) -> str | None:
    """Why the KV cache quantization does not fit, or None for an unquantized (bf16) KV cache."""
    if kv_cache_quant_algo is not None:
        return (
            f"kv_cache_config.dtype {getattr(kv_cache_quant_algo, 'name', kv_cache_quant_algo)}: the Kimi K3 targets' "
            "MLA kernels read a bf16 KV cache"
        )
    return None


def quantization_declines(
    quant_algo: QuantAlgo | None, layer_quant: dict[str, QuantConfig] | None
) -> str | None:
    """Why the quantization the engine resolved from the checkpoint does not fit, or None for the MXFP4 checkpoint's:
    its compressed-tensors config reads as W4A16_MXFP4 with no per-layer declarations (the routed experts then run
    the W4A8_MXFP4_MXFP8 default)."""
    from tensorrt_llm.quantization.mode import QuantAlgo

    if quant_algo != QuantAlgo.W4A16_MXFP4 or layer_quant:
        return (
            f"resolved quantization {quant_algo} with {len(layer_quant or {})} per-layer declarations: the Kimi K3 "
            "targets load the MXFP4 checkpoint, W4A16_MXFP4 with no per-layer declarations"
        )
    return None


def allreduce_strategy_declines(strategy: AllReduceStrategy) -> str | None:
    """Why the all-reduce strategy does not fit, or None for one of `ALLREDUCE_STRATEGIES`."""
    from tensorrt_llm.functional import AllReduceStrategy

    if strategy not in tuple(AllReduceStrategy[name] for name in ALLREDUCE_STRATEGIES):
        return (
            f"allreduce_strategy {strategy.name}: the Kimi K3 targets take "
            f"{' or '.join(ALLREDUCE_STRATEGIES)}"
        )
    return None


def attn_backend_declines(attn_backend: str) -> str | None:
    """Why the attention backend does not fit, or None for `ATTN_BACKEND` (in any case)."""
    name = str(attn_backend).upper()
    if name != ATTN_BACKEND:
        return f"attn_backend {name}: the Kimi K3 targets' attention reads the {ATTN_BACKEND} backend's metadata"
    return None


def declines(model_config: ModelConfig, kv_cache_quant_algo: QuantAlgo | str | None) -> list[str]:
    """The reasons ``model_config`` does not meet the settings above, in a fixed order; empty when it meets them.

    ``kv_cache_quant_algo`` is the caller's reading of the KV cache quantization: ``route`` runs before the engine
    applies ``kv_cache_config.dtype`` to the model config and passes ``engine_kv_cache_quant_algo``; a target passes
    ``model_config.quant_config.kv_cache_quant_algo``.
    """
    reasons = (
        dtype_declines(model_config.torch_dtype),
        kv_cache_quant_declines(kv_cache_quant_algo),
        quantization_declines(model_config.quant_config.quant_algo, model_config.quant_config_dict),
        allreduce_strategy_declines(model_config.allreduce_strategy),
        attn_backend_declines(model_config.attn_backend),
    )
    return [reason for reason in reasons if reason is not None]
