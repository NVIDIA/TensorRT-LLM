# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GlmMoeDsaForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in implementation (``modeling_deepseekv3``); in ``require`` it
raises that error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# (layers, hidden, routed experts, q_lora_rank). Shape-sniffing also matches
# fine-tunes of the same shape; the accuracy gate pins identity.
_SHAPE = (78, 6144, 256, 2048)


@routes("GlmMoeDsaForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
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
    # tp16: attention heads and the expert intermediate dimension sharded over
    # 16 ranks, no attention DP, no expert parallelism. A weight-layout
    # question, which is why it selects a target rather than a runtime branch.
    tp16 = (
        mapping.world_size == 16
        and mapping.tp_size == 16
        and mapping.moe_ep_size == 1
        and mapping.moe_tp_size == 16
        and not mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
    )
    # The target drives the checkpoint's one MTP module through the one-model
    # MTP-Eagle loop as a linear chain, at any draft length, or runs without
    # speculative decoding.
    mtp_eagle_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and not getattr(spec_config, "use_dynamic_tree", False)
        and getattr(spec_config, "eagle_choices", None) is None
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
    )
    # GLM-5.2-NVFP4 on 16 Rubin GPUs (sm107), tp16, MTP-Eagle or no speculation.
    if (
        compute_capability == (10, 7)
        and shape == _SHAPE
        and tp16
        and mtp_eagle_or_none
        and kv_cache_quant_algo == QuantAlgo.FP8
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .glm_5_2_nvfp4__sm_107__tp16.modeling import ModelingV2Glm52Nvfp4Sm107Tp16
        # isort: on
        # fmt: on
        return ModelingV2Glm52Nvfp4Sm107Tp16
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} moe_ep={mapping.moe_ep_size} "
        f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp} "
        f"pp={mapping.pp_size} cp={mapping.cp_size}, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}; "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        "the one glm_moe_dsa target is GLM-5.2-NVFP4 on sm107, tp16 (16 ranks, "
        "tensor-parallel attention and experts, no attention DP), with one-model "
        "MTP-Eagle as a linear chain or no speculative decoding, and an fp8 latent "
        "KV cache (not the packed fp8_ds_mla layout)"
    )
