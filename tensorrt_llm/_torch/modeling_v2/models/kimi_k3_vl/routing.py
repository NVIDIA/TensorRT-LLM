# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a KimiK3ForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in Kimi K3 implementation; in ``require`` it raises that error.

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower and refuse image input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture these targets are written for. sm is part of a
# target's identity, not a knob: a different SM is a different target. The K3
# decode kernels (the tcgen05 GEMVs, the fused MoE) are built for GB200 only.
_SM = (10, 0)

# The MXFP4 checkpoint's config shape; see the note in the sibling gpt_oss
# routing module for what shape-sniffing does and does not pin.
#
# (num_hidden_layers, hidden_size, num_experts, routed_expert_hidden_size), all
# read from ``text_config``. The NVFP4 requant of the same checkpoint has this
# shape too; the quantization criterion below reads only what ``config.json``
# declares, since ``route`` does not see ``hf_quant_config.json``.
_SHAPE = (93, 7168, 896, 3584)


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _quant_algo(pretrained_config: Any) -> Optional[QuantAlgo]:
    """The quantization the checkpoint's config declares, as the engine reads it.

    The MXFP4 checkpoint declares its quantization inside
    ``text_config.quantization_config`` (compressed-tensors,
    ``mxfp4-pack-quantized``); ``KimiK3Config`` surfaces it and the model
    config reads it as W4A16_MXFP4. A checkpoint without quantization reads as
    none; the experts of either would land in a loader that reads packed MXFP4
    tensors.
    """
    from tensorrt_llm._torch.model_config import ModelConfig

    declared = getattr(pretrained_config, "quantization_config", None)
    if declared is None:
        return None
    try:
        quant_config, _ = ModelConfig.load_hf_quant_config(declared, "TRTLLM")
    except ValueError:
        return None
    return quant_config.quant_algo


def _parallel(m) -> Optional[str]:
    """Name the parallel topology, or None if no target implements it.

    Attention and the dense layers are sharded 16 ways. The routed experts are
    split either 4 ways by tensor and 4 ways by expert, or 16 ways by tensor
    with every expert on every rank. The expert split decides which expert
    weights each rank loads and into which shapes, so it selects a target
    rather than a runtime branch.

    The 16-way tensor split counts only when the configuration asks for it. A
    mapping with no expert split resolves to the same sizes, but the built-in
    Kimi K3 model reads that default as expert parallelism over the 16 ranks
    (`moe_tp_ep_user_specified`), and so does this tree: no target serves it.
    """
    if not (
        m.world_size == 16 and m.tp_size == 16 and m.pp_size == 1 and not m.enable_attention_dp
    ):
        return None
    if m.moe_tp_size == 4 and m.moe_ep_size == 4:
        return "tp16_moetp4ep4"
    if m.moe_tp_size == 16 and m.moe_ep_size == 1 and getattr(m, "moe_tp_ep_user_specified", False):
        return "tp16_moetp16ep1"
    return None


@routes("KimiK3ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    text_config = getattr(pretrained_config, "text_config", None)
    shape = (
        None
        if text_config is None
        else tuple(
            _field(text_config, name)
            for name in (
                "num_hidden_layers",
                "hidden_size",
                "num_experts",
                "routed_expert_hidden_size",
            )
        )
    )
    quant_algo = _quant_algo(pretrained_config)
    parallel = _parallel(mapping)
    mxfp4_on_sm100 = (
        compute_capability == _SM and shape == _SHAPE and quant_algo == QuantAlgo.W4A16_MXFP4
    )

    if mxfp4_on_sm100 and parallel == "tp16_moetp4ep4":
        from .kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling import (
            ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4,
        )

        return ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4

    if mxfp4_on_sm100 and parallel == "tp16_moetp16ep1":
        from .kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.modeling import (
            ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1,
        )

        return ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1

    raise NoTarget(
        f"compute capability {compute_capability}, text_config shape {shape}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, parallel "
        f"ws={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"split_set={getattr(mapping, 'moe_tp_ep_user_specified', False)} "
        f"attention_dp={mapping.enable_attention_dp}; the kimi_k3_vl targets are the MXFP4 "
        "Kimi K3 checkpoint on sm100 (GB200), 16 ranks of tensor-parallel attention, with the "
        "experts split moe_tp 4 x moe_ep 4 or, set explicitly, moe_tp 16 x moe_ep 1"
    )
