# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GlmMoeDsaForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in GlmMoeDsa implementation; in ``require`` it raises that
error.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
    kv_cache_quant_algo: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    ``kv_cache_quant_algo`` is the KV-cache algorithm the engine will run with
    (see ``_router_index._kv_cache_quant_algo``), not the checkpoint's alone.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    # The checkpoint is recognized by its config shape; see the gpt_oss routing
    # module for what shape-sniffing does and does not pin. index_n_heads is in
    # the shape because a DeepSeek-V3.2-shaped DSA checkpoint would otherwise
    # match on the first four.
    shape = (
        pretrained_config.num_hidden_layers,
        pretrained_config.hidden_size,
        pretrained_config.n_routed_experts,
        pretrained_config.q_lora_rank,
        getattr(pretrained_config, "index_n_heads", None),
    )
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
    # The target carries the checkpoint's own MTP module (layer 78) driven by the
    # one-model MTP-Eagle loop, and no other draft path. One draft token per step,
    # so a generation step carries at most 2 query rows per sequence: the decode
    # attention's grid, the base-row index kind and the specialised MLA rope kernel
    # are built for that width. A longer chain (or a dynamic tree) is a different
    # forward. The engine resolves an unset draft length to 1 before this runs.
    draft_len = getattr(spec_config, "max_draft_len", None) or 1
    draft_tokens = getattr(spec_config, "max_total_draft_tokens", None) or draft_len
    mtp_eagle_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and draft_len == 1
        and draft_tokens == 1
    )
    # The sparse-MLA kernels read and write an fp8 latent pool only, the one the
    # checkpoint declares.
    kv_fp8 = kv_cache_quant_algo == QuantAlgo.FP8
    # GLM-5.2-NVFP4 on 4 B300s (sm103), tep4, fp8 KV cache, MTP-Eagle with one
    # draft token or no speculation.
    if (
        compute_capability == (10, 3)
        and shape == (78, 6144, 256, 2048, 32)
        and tep4
        and kv_fp8
        and mtp_eagle_or_none
    ):
        from .glm_5_2_nvfp4__sm_103__tep4.modeling import ModelingV2Glm52Nvfp4Sm103Tep4

        return ModelingV2Glm52Nvfp4Sm103Tep4

    spec = (
        "off"
        if spec_config is None
        else f"{spec_config.spec_dec_mode.name} (max_draft_len={draft_len}, "
        f"max_total_draft_tokens={draft_tokens})"
    )
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"moe_ep={mapping.moe_ep_size} moe_tp={mapping.moe_tp_size} "
        f"attention_dp={mapping.enable_attention_dp}, KV cache {kv_cache_quant_algo or 'bf16'}, "
        f"speculative decoding {spec}; "
        "the one glm_moe_dsa target is GLM-5.2-NVFP4 on sm103, tep4 "
        "(4 ranks, tensor-parallel attention, expert-parallel MoE), with an fp8 KV "
        "cache and one-model MTP-Eagle at max_draft_len=1 or no speculative decoding"
    )
