# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen4ExpForConditionalGeneration.

On the GPUs in `archs.SERVED`, Qwen3.8-Flash-Next-NVFP4 lands on
`qwen3_8_flash_next_nvfp4__sm_10x__default` and Qwen3.8-Flash-Next-FP8 (FP8_BLOCK_SCALES) on
`qwen3_8_flash_next_fp8__sm_10x__default`, at one of the layouts in `_LAYOUTS` or, over several
nodes, at dep<world size>, each with one-model MTP-Eagle or no speculative
decoding. The target's `check_deployment` then declines the checkpoints and engine choices it
does not run, so `auto` falls back to the built-in implementation instead of failing in the build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, routes
from . import archs

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

# (world size, attention DP, MoE TP, MoE EP) -> the target's layout in one node. tp1 holds
# everything on one rank; tp2 and tp4 split the routed experts by intermediate, tep2, tep4 and
# dep4 by expert; dep2 is attention DP with two 256-expert windows.
_LAYOUTS = {
    (1, False, 1, 1): "tp1",
    (2, False, 2, 1): "tp2",
    (2, False, 1, 2): "tep2",
    (2, True, 1, 2): "dep2",
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
}


def layout(mapping: Mapping) -> Optional[str]:
    """The layout ``mapping`` describes, or None.

    Attention runs TP over the whole world, or DP with the LM head whole. Across nodes only
    attention DP with the routed experts split by expert over the whole world is served
    (dep<world size>). Pipeline and context parallelism and MoE clusters are other forwards.
    """
    if mapping.enable_attention_dp:
        attention_as_built = not mapping.enable_lm_head_tp_in_adp
    else:
        attention_as_built = mapping.attn_tp_size == mapping.tp_size
    if (
        not attention_as_built
        or mapping.tp_size != mapping.world_size
        or (mapping.pp_size, mapping.cp_size, mapping.moe_cluster_size) != (1, 1, 1)
    ):
        return None
    if mapping.is_multi_node():
        expert_split = mapping.moe_tp_size == 1 and mapping.moe_ep_size == mapping.world_size
        return f"dep{mapping.world_size}" if mapping.enable_attention_dp and expert_split else None
    return _LAYOUTS.get(
        (mapping.world_size, mapping.enable_attention_dp, mapping.moe_tp_size, mapping.moe_ep_size)
    )


# Qwen3.8-Flash-Next publishes ``Qwen4ExpForConditionalGeneration``, and the
# config loader's composite branch writes the same name back.
@routes("Qwen4ExpForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits."""
    mapping, spec = model_config.mapping, model_config.spec_config
    fp8 = getattr(model_config.quant_config, "quant_algo", None) == QuantAlgo.FP8_BLOCK_SCALES
    member = layout(mapping)
    if not archs.served(compute_capability) or member is None:
        raise NoTarget(
            f"compute capability {compute_capability}, world_size={mapping.world_size} "
            f"(tp={mapping.tp_size}, pp={mapping.pp_size}, cp={mapping.cp_size}, "
            f"attention_dp={mapping.enable_attention_dp}, moe_tp={mapping.moe_tp_size}, "
            f"moe_ep={mapping.moe_ep_size}); the qwen4_exp targets serve {archs.served_names()} "
            f"at {', '.join(_LAYOUTS.values())} in one node, or dep<world size> across nodes"
        )
    if spec is not None and not (
        isinstance(spec, MTPDecodingConfig) and spec.spec_dec_mode.is_mtp_eagle_one_model()
    ):
        raise NoTarget(
            f"speculative decoding {spec.spec_dec_mode.name}; the target runs one-model "
            "MTP-Eagle or no speculative decoding"
        )
    if fp8:
        from .qwen3_8_flash_next_fp8__sm_10x__default.modeling import (
            ModelingV2Qwen38FlashNextFp8Sm10xDefault,
        )

        ModelingV2Qwen38FlashNextFp8Sm10xDefault.check_deployment(model_config, llm_args, member)
        return ModelingV2Qwen38FlashNextFp8Sm10xDefault
    if str(getattr(model_config, "moe_backend_requested", "")).upper() == "CUTEDSL" and (
        not archs.fast_path("cute_dsl_moe", compute_capability)
    ):
        raise NoTarget(f"moe_config.backend=CUTEDSL does not run on {compute_capability}")
    from .qwen3_8_flash_next_nvfp4__sm_10x__default.modeling import (
        ModelingV2Qwen38FlashNextNvfp4Sm10xDefault,
    )

    ModelingV2Qwen38FlashNextNvfp4Sm10xDefault.check_deployment(model_config, llm_args, member)
    return ModelingV2Qwen38FlashNextNvfp4Sm10xDefault
