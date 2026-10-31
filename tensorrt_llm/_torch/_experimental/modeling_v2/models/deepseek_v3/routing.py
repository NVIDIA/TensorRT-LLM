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

from typing import TYPE_CHECKING, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes, tp_reaches_across_nodes

# The GPUs the target is certified on: sm_100 (B200, GB200) and sm_103 (B300,
# GB300). They run the same kernels.
_SMS = frozenset({(10, 0), (10, 3)})

# A parallel layout, as (world_size, attention_dp, moe_tp, moe_ep).
#
# The layouts the default target has passed its gates on. The default forward
# computes others too -- any split of 128 heads, 256 experts and the
# 2048-wide expert intermediate on 128-row tiles -- but a layout is admitted
# here only once it has been measured, so ``require`` cannot quietly run one
# nobody checked. A GB200 or GB300 tray holds four GPUs, so its 8- and 16-rank
# layouts span two and four trays of one NVLink domain; a B200 or B300 node holds
# eight, so its 8-rank layouts stay inside one node.
_DEFAULT_LAYOUTS = frozenset(
    {
        (2, False, 2, 1),  # tp2, experts split by intermediate (the LLM API default)
        (2, False, 1, 2),  # tp2 + ep2
        (2, True, 1, 2),  # dep2
        (4, False, 4, 1),  # tp4, experts split by intermediate (the LLM API default)
        (4, False, 1, 4),  # tp4 + ep4
        (4, False, 2, 2),  # tp4, experts tp2 x ep2
        (4, True, 1, 4),  # dep4
        (4, True, 2, 2),  # dep4, experts tp2 x ep2
        (4, True, 4, 1),  # dep4, experts split by intermediate
        (8, False, 8, 1),  # tp8, experts split by intermediate (the LLM API default)
        (8, False, 1, 8),  # tp8 + ep8
        (8, False, 4, 2),  # tp8, experts tp4 x ep2
        (8, False, 2, 4),  # tp8, experts tp2 x ep4
        (8, True, 1, 8),  # dep8
        (8, True, 8, 1),  # dep8, experts split by intermediate
        (8, True, 2, 4),  # dep8, experts tp2 x ep4
        (16, True, 1, 16),  # dep16
    }
)

# The layouts at which the checkpoint's MTP layer (one-model MTP-Eagle) is
# admitted. The MTP layer is cut like every other layer, so the forward
# computes it at every layout above; these are the ones its draft acceptance
# was measured at against stock modeling: each residual-norm path (attention
# TP in one tray and across trays, attention DP) and each expert split.
_MTP_LAYOUTS = frozenset(
    {
        (4, True, 1, 4),  # dep4
        (4, False, 4, 1),  # tp4
        (4, False, 1, 4),  # tp4 + ep4
        (4, True, 4, 1),  # dep4, experts split by intermediate
        (8, False, 2, 4),  # tp8, experts tp2 x ep4
        (8, True, 1, 8),  # dep8
    }
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


@routes("DeepseekV3ForCausalLM")
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
    if compute_capability not in _SMS or shape != (61, 7168, 256, 1536):
        raise NoTarget(
            f"compute capability {compute_capability}, config shape {shape}; the "
            "deepseek_v3 target is DeepSeek-R1-0528-NVFP4 on sm100 or sm103"
        )

    # The target's MLA ops read and write an fp8 latent pool only, the one the
    # checkpoint declares; it asserts so at construction.
    if kv_cache_quant_algo != QuantAlgo.FP8:
        raise NoTarget(
            f"KV cache {kv_cache_quant_algo or 'bf16'}; the default target runs an fp8 "
            "latent KV cache only"
        )

    # Which ranks hold which heads, experts and intermediate columns. The
    # default target computes its shares from the mapping, so the layout is a
    # question of which ones were measured, not of which code runs.
    layout = (
        mapping.world_size,
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    split = (
        f"world_size={mapping.world_size} pp={mapping.pp_size} cp={mapping.cp_size} "
        f"attention_dp={mapping.enable_attention_dp} moe_tp={mapping.moe_tp_size} "
        f"moe_ep={mapping.moe_ep_size}"
    )
    if mapping.pp_size != 1 or mapping.cp_size != 1:
        raise NoTarget(
            f"parallel layout {split}; no deepseek_v3 target implements pipeline or "
            "context parallelism"
        )

    # Two mapping facts the default target asserts on at construction. Refused
    # here so that ``auto`` falls back instead of crashing in the model: an
    # lm_head split across the attention-DP ranks gathers logits the target
    # never computes, and an attention TP group narrower than the TP group
    # (helix attention CP) holds heads on ranks the target does not expect.
    if getattr(mapping, "enable_lm_head_tp_in_adp", False):
        raise NoTarget(
            f"parallel layout {split} with enable_lm_head_tp_in_adp; the default "
            "target keeps lm_head whole under attention DP"
        )
    if not mapping.enable_attention_dp and mapping.attn_tp_size != mapping.tp_size:
        raise NoTarget(
            f"parallel layout {split} with attn_tp_size={mapping.attn_tp_size} != "
            f"tp_size={mapping.tp_size}; the default target splits the heads over the "
            "whole TP group"
        )

    if layout not in _DEFAULT_LAYOUTS:
        raise NoTarget(
            f"parallel layout {split}; the default target is gated on tp2, tp2+ep2, "
            "dep2, tp4, tp4+ep4, tp4 with moe tp2 x ep2, dep4 with moe ep4, tp2 x ep2 "
            "or tp4, tp8, tp8+ep8, tp8 with moe tp4 x ep2 or tp2 x ep4, dep8 with moe "
            "ep8, tp8 or tp2 x ep4, and dep16"
        )

    # Attention TP across nodes reduces only over multi-node NVLink.
    if not tp_reaches_across_nodes(mapping):
        raise NoTarget(
            f"parallel layout {split} with attention TP over {mapping.tp_size} ranks across "
            f"nodes of {mapping.gpus_per_node} GPUs and no multi-node NVLink; the default "
            "target reduces attention TP across nodes only over MNNVL"
        )

    # The checkpoint's own MTP module (layer 61), driven by the one-model
    # MTP-Eagle loop, is the only draft path. ``use_mtp_vanilla`` selects the
    # multi-model MTP loop instead, so it does not route here.
    if spec_config is not None:
        spec = spec_config.spec_dec_mode.name
        mtp_eagle = (
            isinstance(spec_config, MTPDecodingConfig)
            and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        )
        if not mtp_eagle:
            raise NoTarget(
                f"speculative decoding {spec} at parallel layout {split}; the default "
                "target implements one-model MTP-Eagle or no speculative decoding"
            )
        if layout not in _MTP_LAYOUTS:
            raise NoTarget(
                f"speculative decoding {spec} at parallel layout {split}; MTP is gated "
                "on dep4, tp4, tp4+ep4, dep4 with moe tp4, tp8 with moe tp2 x ep4, "
                "and dep8"
            )

    from .r1_0528_nvfp4__sm_10x__default.modeling import ModelingV2DeepseekR10528Nvfp4Sm10xDefault

    return ModelingV2DeepseekR10528Nvfp4Sm10xDefault
