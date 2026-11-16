# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for Qwen4ExpForConditionalGeneration.

Qwen3.8-Flash-Next-NVFP4 on sm103 lands on `qwen3_8_flash_next_nvfp4__sm_103__default` at one
of the layouts in `_LAYOUTS`, with one-model MTP-Eagle (not at dep2) or no speculative
decoding. The target's `check_deployment` then declines the checkpoints and engine choices it
does not run, so `auto` falls back to the built-in implementation instead of failing in the
build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

from ..._router_index import NoTarget, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

# The GPU the target is certified on.
_SM = (10, 3)

# (world size, attention DP, MoE TP, MoE EP) -> the target's layout, each in one node. tp1 holds
# everything on one rank; tp4 splits the routed experts by intermediate, tep4 and dep4 by
# expert; dep2 is attention DP with two 256-expert windows. MTP runs at every layout but dep2,
# where no draft acceptance was measured.
_LAYOUTS = {
    (1, False, 1, 1): "tp1",
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
    (2, True, 1, 2): "dep2",
}


def layout(mapping: Mapping) -> Optional[str]:
    """The layout ``mapping`` describes, or None.

    Attention runs TP over the whole world, or DP with the LM head whole. Pipeline and context
    parallelism, MoE clusters and a world spanning nodes are other forwards.
    """
    if mapping.enable_attention_dp:
        attention_as_built = not mapping.enable_lm_head_tp_in_adp
    else:
        attention_as_built = mapping.attn_tp_size == mapping.tp_size
    if (
        not attention_as_built
        or mapping.tp_size != mapping.world_size
        or (mapping.pp_size, mapping.cp_size, mapping.moe_cluster_size) != (1, 1, 1)
        or mapping.is_multi_node()
    ):
        return None
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
    member = layout(mapping)
    if compute_capability != _SM or member is None:
        raise NoTarget(
            f"compute capability {compute_capability}, world_size={mapping.world_size} "
            f"(tp={mapping.tp_size}, pp={mapping.pp_size}, cp={mapping.cp_size}, "
            f"attention_dp={mapping.enable_attention_dp}, moe_tp={mapping.moe_tp_size}, "
            f"moe_ep={mapping.moe_ep_size}); the qwen4_exp target is Qwen3.8-Flash-Next-NVFP4 "
            "on sm103 in one node: tp1, tp4 (experts split by intermediate), tep4, dep4 "
            "(attention DP) or dep2 (attention DP, experts EP2)"
        )
    if spec is not None and not (
        isinstance(spec, MTPDecodingConfig) and spec.spec_dec_mode.is_mtp_eagle_one_model()
    ):
        raise NoTarget(
            f"speculative decoding {spec.spec_dec_mode.name}; the target runs one-model "
            "MTP-Eagle or no speculative decoding"
        )
    if spec is not None and member == "dep2":
        raise NoTarget(
            "speculative decoding with attention DP (dep2): no draft acceptance was measured at "
            "this layout; MTP under attention DP is admitted at dep4"
        )
    from .qwen3_8_flash_next_nvfp4__sm_103__default.modeling import (
        ModelingV2Qwen38FlashNextNvfp4Sm103Default,
    )

    ModelingV2Qwen38FlashNextNvfp4Sm103Default.check_deployment(model_config, llm_args, member)
    return ModelingV2Qwen38FlashNextNvfp4Sm103Default
