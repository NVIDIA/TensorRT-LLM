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
    index_n_heads = getattr(pretrained_config, "index_n_heads", None)
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

    # tep4: the attention heads and the dense MLPs sharded four ways, the experts
    # sharded four ways. Attention DP would replicate attention, which is a
    # different weight layout and so a different target.
    tep4 = (
        mapping.world_size == 4
        and mapping.tp_size == 4
        and mapping.pp_size == 1
        and mapping.moe_ep_size == 4
        and mapping.moe_tp_size == 1
        and not mapping.enable_attention_dp
    )
    # The sm103 target drives the checkpoint's MTP module (layer 78) through the
    # one-model MTP-Eagle loop with one draft token, or runs without speculation.
    # A generation step then carries at most 2 query rows per sequence: the decode
    # attention's grid, the base-row index kind and the specialised MLA rope kernel
    # are built for that width. The engine resolves an unset draft length to 1
    # before this runs.
    draft_len = getattr(spec_config, "max_draft_len", None) or 1
    draft_tokens = getattr(spec_config, "max_total_draft_tokens", None) or draft_len
    mtp1_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and draft_len == 1
        and draft_tokens == 1
    )
    # GLM-5.2-NVFP4 on 4 B300s (sm103), tep4, MTP-Eagle with one draft token or
    # no speculation. index_n_heads is checked because a DeepSeek-V3.2-shaped DSA
    # checkpoint would otherwise match the shape.
    if (
        compute_capability == (10, 3)
        and shape == _SHAPE
        and index_n_heads == 32
        and tep4
        and mtp1_or_none
        and kv_cache_quant_algo == QuantAlgo.FP8
    ):
        # fmt: off
        # isort: off
        from .glm_5_2_nvfp4__sm_103__tep4.modeling import ModelingV2Glm52Nvfp4Sm103Tep4
        # isort: on
        # fmt: on
        return ModelingV2Glm52Nvfp4Sm103Tep4

    spec = (
        "off"
        if spec_config is None
        else f"{spec_config.spec_dec_mode.name} (max_draft_len={draft_len}, "
        f"max_total_draft_tokens={draft_tokens})"
    )
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"index_n_heads={index_n_heads}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} moe_ep={mapping.moe_ep_size} "
        f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp} "
        f"pp={mapping.pp_size} cp={mapping.cp_size}, speculative decoding {spec}; "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        "the glm_moe_dsa targets are GLM-5.2-NVFP4 on sm107, tp16 (16 ranks, "
        "tensor-parallel attention and experts, no attention DP), with one-model "
        "MTP-Eagle as a linear chain or no speculative decoding; and GLM-5.2-NVFP4 "
        "on sm103, tep4 (4 ranks, tensor-parallel attention, expert-parallel MoE), "
        "with one-model MTP-Eagle at max_draft_len=1 or no speculative decoding. "
        "Both need an fp8 latent KV cache (not the packed fp8_ds_mla layout)"
    )
