# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a DeepseekV3ForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in DeepseekV3 implementation; in ``require`` it raises that
error.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

from ..._router_index import NoTarget


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    # The checkpoint is recognized by its config shape; see the gpt_oss routing
    # module for what shape-sniffing does and does not pin. q_lora_rank is in
    # the shape because it changes the attention weight layout: a checkpoint
    # matching the first three but not this one is a different assembly.
    shape = (
        pretrained_config.num_hidden_layers,
        pretrained_config.hidden_size,
        pretrained_config.n_routed_experts,
        pretrained_config.q_lora_rank,
    )

    # dep4: attention replicated across the 4 ranks (attention DP), only the
    # experts sharded. This is a weight-layout question, which is why it
    # selects a target rather than a runtime branch: tep4 (not in this batch)
    # shards the attention heads and loads different weights into different
    # shapes.
    dep4 = (
        mapping.world_size == 4
        and mapping.moe_ep_size == 4
        and mapping.moe_tp_size == 1
        and mapping.enable_attention_dp
    )

    # The target carries the checkpoint's own MTP module (layer 61) driven by
    # the one-model MTP-Eagle loop, and no other draft path. ``use_mtp_vanilla``
    # selects the multi-model MTP loop instead, so it does not route here.
    mtp_eagle_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
    )

    # DeepSeek-R1-0528-NVFP4 on 4 GB300s (sm103), dep4, MTP-Eagle or no speculation.
    if (
        compute_capability == (10, 3)
        and shape == (61, 7168, 256, 1536)
        and dep4
        and mtp_eagle_or_none
    ):
        from .r1_0528_nvfp4__sm_103__dep4.modeling import ModelingV2DeepseekR10528Nvfp4Sm103Dep4

        return ModelingV2DeepseekR10528Nvfp4Sm103Dep4

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"world_size={mapping.world_size} moe_ep={mapping.moe_ep_size} "
        f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp}, "
        f"speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}; "
        "the one deepseek_v3 target is DeepSeek-R1-0528-NVFP4 on sm103, dep4 "
        "(4 ranks, expert-parallel MoE, attention DP), with one-model MTP-Eagle or no "
        "speculative decoding"
    )
