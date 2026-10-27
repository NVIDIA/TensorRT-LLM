# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GptOssForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in GptOss implementation; in ``require`` it raises that error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes, tp_reaches_across_nodes

# The GPUs the target is certified on: sm_100 (B200, GB200) and sm_103 (B300,
# GB300). They run the same kernels.
_SMS = frozenset({(10, 0), (10, 3)})

# The layouts the default target has passed its gates on, as
# (world_size, attention_dp, moe_tp, moe_ep). The default forward computes
# others too -- any split of 8 kv heads and 128 experts -- but a layout is
# admitted here only once it has been measured, so `require` cannot quietly
# run one nobody checked. A GB200 or GB300 tray holds four GPUs, so its 8-rank
# layouts span two trays of one NVLink domain; a B200 or B300 node holds eight,
# so they stay inside one node.
_DEFAULT_LAYOUTS = frozenset(
    {
        (1, False, 1, 1),  # tp1: one GPU, no collective
        (2, False, 2, 1),  # tp2, MoE split by intermediate (the LLM API default)
        (2, False, 1, 2),  # tp2 + ep2
        (4, False, 4, 1),  # tp4, MoE split by intermediate (the LLM API default)
        (4, False, 1, 4),  # tp4 + ep4
        (4, False, 2, 2),  # tp4, MoE tp2 x ep2
        (2, True, 1, 2),  # dep2
        (4, True, 1, 4),  # dep4
        (4, True, 2, 2),  # dep4, MoE tp2 x ep2
        (8, False, 8, 1),  # tp8, MoE split by intermediate (the LLM API default)
        (8, False, 1, 8),  # tp8 + ep8
        (8, True, 1, 8),  # dep8
    }
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


@routes("GptOssForCausalLM")
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
    # The checkpoint is recognized by its config shape, the upstream idiom
    # (``is_mla``, ``is_nemotron_hybrid`` do the same). A fine-tune with the
    # same shape routes here too; the accuracy gate is what pins the identity.
    # The expert count is in the shape because it fixes the MoE operand
    # geometry the target declares; layer count alone separates 120b from 20b.
    shape = (
        pretrained_config.num_hidden_layers,
        pretrained_config.hidden_size,
        pretrained_config.num_local_experts,
    )

    # gpt-oss-120b on sm100 or sm103, no speculative decoding: the target
    # implements no draft path, so claiming such a deployment would crash in
    # the model instead of falling back.
    gpt_oss_120b_sm10x = (
        compute_capability in _SMS and shape == (36, 2880, 128) and spec_config is None
    )

    # The parallel layout. Pipeline and context parallelism change which rank
    # holds which layers or tokens, and the target implements neither. Two
    # mapping facts the table does not show change which rank holds which
    # weights, so they are refused here too, before the target's own asserts
    # on them could crash a deployment `auto` should have sent back: an
    # attention TP group narrower than the TP group (heads split over fewer
    # ranks than the MoE), and lm_head split over the attention-DP ranks.
    layout = (
        mapping.world_size,
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    attn_tp_whole = mapping.enable_attention_dp or mapping.attn_tp_size == mapping.tp_size
    lm_head_tp_in_adp = bool(getattr(mapping, "enable_lm_head_tp_in_adp", False))
    no_pp_cp = mapping.pp_size == 1 and mapping.cp_size == 1
    split_as_built = no_pp_cp and attn_tp_whole and not lm_head_tp_in_adp
    # Attention TP across nodes reduces only over multi-node NVLink.
    tp_reach = tp_reaches_across_nodes(mapping)
    # The target attends over a bf16 KV pool only; it asserts so at construction.
    kv_bf16 = kv_cache_quant_algo is None

    # Every gated layout, one GPU included, is the default target's.
    if (
        gpt_oss_120b_sm10x
        and split_as_built
        and tp_reach
        and kv_bf16
        and layout in _DEFAULT_LAYOUTS
    ):
        from .gpt_oss_120b__sm_10x__default.modeling import ModelingV2GptOss120bSm10xDefault

        return ModelingV2GptOss120bSm10xDefault

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"{mapping.world_size} ranks (pp={mapping.pp_size} cp={mapping.cp_size} "
        f"attention_dp={bool(mapping.enable_attention_dp)} moe_tp={mapping.moe_tp_size} "
        f"moe_ep={mapping.moe_ep_size} attn_tp={mapping.attn_tp_size} tp={mapping.tp_size} "
        f"lm_head_tp_in_adp={lm_head_tp_in_adp} "
        f"tp_reaches_across_nodes={tp_reach}), speculative decoding "
        f"{type(spec_config).__name__ if spec_config is not None else 'off'}, "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        "the gpt_oss target is gpt-oss-120b on sm100 or sm103 without speculative decoding "
        "and with a bf16 KV cache, "
        "at one of the default target's layouts (tp1, tp2, tp2+ep2, tp4, tp4+ep4, "
        "tp4 with MoE tp2 x ep2, dep2, dep4, dep4 with MoE tp2 x ep2, tp8, tp8+ep8, dep8; "
        "no pipeline or context parallelism, attention TP equal to TP unless attention DP "
        "is on, no lm_head TP under attention DP, and multi-node NVLink when attention TP "
        "spans nodes)"
    )
