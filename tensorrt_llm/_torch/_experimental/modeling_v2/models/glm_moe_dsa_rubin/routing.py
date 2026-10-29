# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2-NVFP4 targets for Rubin, or ``NoTarget`` outside their deployments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# (layers, hidden, routed experts, q_lora_rank). The accuracy gate pins identity.
_SHAPE = (78, 6144, 256, 2048)


@routes("GlmMoeDsaForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: tuple[int, int]
) -> type:
    """Select the Rubin target without importing modeling code for a declined deployment."""
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    shape = (
        pretrained_config.num_hidden_layers,
        pretrained_config.hidden_size,
        pretrained_config.n_routed_experts,
        pretrained_config.q_lora_rank,
    )
    pure_tp = (
        mapping.tp_size in (4, 8, 16)
        and mapping.world_size == mapping.tp_size
        and mapping.moe_ep_size == 1
        and mapping.moe_tp_size == mapping.tp_size
        and not mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
    )
    # Replay the checkpoint's one MTP module as a linear chain, or run without speculation.
    mtp_eagle_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and not getattr(spec_config, "use_dynamic_tree", False)
        and getattr(spec_config, "eagle_choices", None) is None
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
    )
    if (
        compute_capability == (10, 7)
        and shape == _SHAPE
        and pure_tp
        and mtp_eagle_or_none
        and kv_cache_quant_algo == QuantAlgo.FP8
    ):
        # Keep the target import on one line for static claims and CI selection.
        # fmt: off
        # isort: off
        from .glm_5_2_nvfp4__sm_107__low_latency.modeling import ModelingV2Glm52Nvfp4Sm107LowLatency
        # isort: on
        # fmt: on
        return ModelingV2Glm52Nvfp4Sm107LowLatency

    spec = (
        "off"
        if spec_config is None
        else f"{spec_config.spec_dec_mode.name} "
        f"(max_draft_len={getattr(spec_config, 'max_draft_len', None)}, "
        f"max_total_draft_tokens={getattr(spec_config, 'max_total_draft_tokens', None)})"
    )
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} moe_ep={mapping.moe_ep_size} "
        f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp} "
        f"pp={mapping.pp_size} cp={mapping.cp_size}, speculative decoding {spec}; "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        "the glm_moe_dsa_rubin target is GLM-5.2-NVFP4 on sm107, TP4/TP8/TP16 "
        "(equal world and TP sizes, tensor-parallel attention and experts, no attention DP), "
        "with one-model MTP-Eagle as a linear chain or no speculative decoding. "
        "It needs an fp8 latent KV cache (not the packed fp8_ds_mla layout)"
    )
