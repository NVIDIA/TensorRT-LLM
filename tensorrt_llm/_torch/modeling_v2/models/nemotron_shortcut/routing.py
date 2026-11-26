# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the default text generation target for the checkpoint precision and mapping."""

import torch

from tensorrt_llm._torch.configs.nemotron_shortcut import NemotronShortcutTextConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import (
    NoTarget,
    engine_kv_cache_quant_algo,
    input_processors,
    recurrent_state_specs,
    routes,
)

# (world size, attention DP, MoE TP, MoE EP), all on a single SM103 node.
_FOUR_RANK_LAYOUTS = {
    (4, False, 1, 4),
    (4, False, 2, 2),
    (4, False, 4, 1),
    (4, True, 1, 4),
    (4, True, 2, 2),
    (4, True, 4, 1),
}
_QUANTIZED_LAYOUTS = _FOUR_RANK_LAYOUTS | {
    (1, False, 1, 1),
    (2, False, 1, 2),
    (2, False, 2, 1),
    (2, True, 1, 2),
    (2, True, 2, 1),
}


@routes(NemotronShortcutTextConfig.causal_lm_architecture)
def route(model_config, llm_args, compute_capability):
    from tensorrt_llm._torch.pyexecutor.model_loader import effective_mamba_ssm_cache_dtype

    cfg = model_config.pretrained_config
    mapping = model_config.mapping
    if compute_capability != (10, 3):
        raise NoTarget("The text decoder requires SM103.")
    if model_config.lora_config is not None:
        if mapping.enable_attention_dp:
            raise NoTarget("The decoder's LoRA projections use tensor-parallel weight shards.")
        if model_config.lora_config.lora_ckpt_source != "hf":
            raise NoTarget("The decoder requires Hugging Face PEFT adapters.")
    spec = model_config.spec_config
    if spec is not None:
        from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

        if (
            not isinstance(spec, MTPDecodingConfig)
            or not spec.spec_dec_mode.is_mtp_eagle_one_model()
            or spec.use_dynamic_tree
            or spec.speculative_model is not None
            or not getattr(cfg, "mtp_use_repeated_layer", False)
            or not getattr(cfg, "moe_shortcut_disable_mtp", False)
            or not getattr(cfg, "num_nextn_predict_layers", 0)
        ):
            raise NoTarget("The checkpoint requires linear MTP with a shared prediction layer.")
        if spec.draft_lm_head_dtype != "auto" or spec.draft_moe_dtype != "auto":
            raise NoTarget("MTP uses the checkpoint dtypes for the shared head and draft experts.")
        if spec.moe_backend not in (None, "AUTO"):
            raise NoTarget("MTP uses the decoder's expert backend.")
        if spec.use_rejection_sampling:
            # The sampler pads the rejected suffix with -1; the draft embeds every input row.
            raise NoTarget("The repeated prediction layer requires token IDs in every draft row.")
    if model_config.torch_dtype != torch.bfloat16:
        raise NoTarget("The text decoder requires BF16 activations.")
    quant = model_config.quant_config.quant_algo
    kv_quant = engine_kv_cache_quant_algo(model_config, llm_args)
    if kv_quant is not None and (
        kv_quant != QuantAlgo.NVFP4 or quant != QuantAlgo.MIXED_PRECISION or cfg.index_topk is None
    ):
        raise NoTarget(
            "Quantized KV requires the mixed-precision checkpoint with sparse attention "
            "and NVFP4 pages."
        )
    cache = llm_args.kv_cache_config
    if (
        effective_mamba_ssm_cache_dtype(model_config, cache.mamba_ssm_cache_dtype) != torch.float32
        or cache.mamba_ssm_stochastic_rounding
    ):
        raise NoTarget(
            "The text decoder requires FP32 recurrent state without stochastic rounding."
        )
    if (
        mapping.world_size != mapping.tp_size
        or mapping.attn_cp_size != 1
        or mapping.moe_tp_size * mapping.moe_ep_size != mapping.world_size
        or mapping.enable_lm_head_tp_in_adp
        or mapping.dwdp_enabled
        or mapping.is_multi_node()
    ):
        raise NoTarget("The text decoder requires single-node attention TP or DP with MoE TP/EP.")
    tp = 1 if mapping.enable_attention_dp else mapping.tp_size
    if (
        cfg.num_attention_heads % tp
        or cfg.vocab_size % tp
        or cfg.gdp_num_heads % tp
        or cfg.intermediate_size % tp
        or not cfg.moe_shared_expert_intermediate_size
        or cfg.moe_shared_expert_intermediate_size % tp
        or cfg.n_routed_experts % mapping.moe_ep_size
        or cfg.moe_intermediate_size % mapping.moe_tp_size
    ):
        raise NoTarget(
            "The checkpoint dimensions do not divide across the selected parallel groups."
        )
    if (
        not cfg.residual_in_fp32
        or cfg.moe_router_bias_space != "logit"
        or cfg.gdp_g_clamp_value is None
        or "full_attention" in cfg.block_types
    ):
        raise NoTarget("The decoder requires FP32 streams, logit-space routing and clamped GDP.")
    if (
        cfg.gdp_num_heads != cfg.gdp_num_groups
        or cfg.gdp_head_dim > cfg.gdp_key_head_dim
        or cfg.gdp_key_head_dim != 128
        or cfg.num_key_value_heads != 1
        or cfg.head_dim not in (64, 128, 256)
        or cfg.tie_word_embeddings
        or cfg.mlp_hidden_act != "relu2"
        or not cfg.activation_clamp_value
        or not cfg.wide_residual_learned_retention
        or not cfg.moe_use_norm_before_up_proj
        or cfg.moe_shortcut_post_norm
    ):
        raise NoTarget("The checkpoint has unsupported mixer, residual or expert settings.")
    layout = (
        mapping.world_size,
        mapping.enable_attention_dp,
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    media = bool(getattr(cfg, "media_config", None)) and not llm_args.disable_mm_encoder
    if quant is None:
        if layout not in _FOUR_RANK_LAYOUTS:
            raise NoTarget(f"Unsupported BF16 text decoder mapping: {layout}.")
        if media:
            from .bf16__sm_103__default.multimodal import ShortcutBf16ForConditionalGeneration

            return ShortcutBf16ForConditionalGeneration
        from .bf16__sm_103__default.modeling import ShortcutBf16ForCausalLM

        return ShortcutBf16ForCausalLM
    if quant == QuantAlgo.MXFP8:
        recipe = model_config.quant_config
        if "group_size" in recipe.model_fields_set and recipe.group_size not in (None, 32):
            raise NoTarget("The MXFP8 text decoder requires 32-element scale blocks.")
        if layout not in _QUANTIZED_LAYOUTS or mapping.world_size == 1:
            raise NoTarget(f"Unsupported MXFP8 text decoder mapping: {layout}.")
        if media:
            from .mxfp8__sm_103__default.multimodal import ShortcutMxfp8ForConditionalGeneration

            return ShortcutMxfp8ForConditionalGeneration
        from .mxfp8__sm_103__default.modeling import ShortcutMxfp8ForCausalLM

        return ShortcutMxfp8ForCausalLM
    if quant == QuantAlgo.MIXED_PRECISION:
        if layout not in _QUANTIZED_LAYOUTS:
            raise NoTarget(f"Unsupported mixed-precision text decoder mapping: {layout}.")
        if media:
            from .nvfp4__sm_103__default.multimodal import ShortcutNvfp4ForConditionalGeneration

            return ShortcutNvfp4ForConditionalGeneration
        from .nvfp4__sm_103__default.modeling import ShortcutNvfp4ForCausalLM

        return ShortcutNvfp4ForCausalLM
    raise NoTarget(f"Unsupported checkpoint precision: {quant}.")


@input_processors(NemotronShortcutTextConfig.causal_lm_architecture)
def input_processor(config, *, disable_mm_encoder):
    if disable_mm_encoder:
        from .input_processor import ShortcutTextInputProcessor

        return config, ShortcutTextInputProcessor
    media_config = config.get_media_config()
    if media_config is None:
        return config, None
    # Importing the module registers the image and video placeholders.
    from .multimodal import ShortcutInputProcessor

    return media_config, ShortcutInputProcessor


@recurrent_state_specs(NemotronShortcutTextConfig.causal_lm_architecture)
def recurrent_state_spec(config, *, quant_config=None):
    from tensorrt_llm._torch.modules.nemotron_shortcut.cache_manager import (
        get_nemotron_shortcut_cache_params,
    )
    from tensorrt_llm._torch.recurrent_state_spec import recurrent_state_spec_from_mamba_params

    params = get_nemotron_shortcut_cache_params(config, quant_config=quant_config)
    # The convolution plane packs more sections than the two shared orders describe.
    return recurrent_state_spec_from_mamba_params(
        params, conv_sections=tuple(config.gdp_conv_section_widths)
    )
