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

from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.modelopt_config import (
    canonicalize_quant_algo,
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# (layers, hidden, routed experts, q_lora_rank). Shape-sniffing also matches
# fine-tunes of the same shape; the accuracy gate pins identity.
_SHAPE = (78, 6144, 256, 2048)

# The fields beyond _SHAPE that the sm103 tp8 target's fast paths are built for. The
# head count fixes the per-rank attention operands at tp8, and index_topk fixes the
# sparse-attention selection width the decode kernels are built for. The last four
# fields fix the layer pattern: the dense layers before the first MoE layer, and which
# layers run their own indexer and which reuse an earlier layer's selection (an
# explicit index_topk_pattern overrides the frequency and offset, so it must be absent).
_TP8_PATTERN_FIELDS = (
    "num_attention_heads",
    "index_topk",
    "first_k_dense_replace",
    "index_topk_freq",
    "index_skip_topk_offset",
    "index_topk_pattern",
)
_TP8_PATTERN = (64, 2048, 3, 4, 3, None)

# Draft tokens per request of the sm103 tp8 target. Its decode kernels take the target
# token plus this many draft tokens per request in one step.
_TP8_DRAFT_LEN = 5


def _nvfp4_with_bf16_mtp_layer(pretrained_config: Any) -> Tuple[bool, str]:
    """Whether the checkpoint is NVFP4 with a bf16 MTP layer, and its quantization in words.

    Reads the inline ``quantization_config`` of config.json, and parses only the
    modelopt shape, with the modelopt normalizer (which raises only
    ``ValueError``). The engine itself prefers ``hf_quant_config.json`` when the
    checkpoint has one, which ``route`` does not see, and warns when the inline
    copy diverges from it. Any other declaration is named and declined, never
    parsed, so a config this target does not serve cannot fail here. The export
    keeps the whole MTP layer (layer ``num_hidden_layers``) in bf16; the draft
    fast paths read bf16 weights there.
    """
    from tensorrt_llm.models.modeling_utils import QuantConfig

    declared = getattr(pretrained_config, "quantization_config", None)
    if declared is None:
        return False, "none"
    if not isinstance(declared, dict) and callable(getattr(declared, "to_dict", None)):
        declared = declared.to_dict()
    if not isinstance(declared, dict):
        return False, f"declared as a {type(declared).__name__}, not a dict"
    if not is_modelopt_quant_config(declared):
        return False, f"quant_method {declared.get('quant_method')!r} (not a modelopt export)"
    try:
        normalized = read_modelopt_quant_config(declared)
        quant_algo = QuantAlgo(canonicalize_quant_algo(normalized.get("quant_algo")))
    except ValueError as why:
        return False, f"a modelopt config without a readable global algorithm ({why})"
    excluded = normalized.get("exclude_modules")
    if excluded is not None and not (
        isinstance(excluded, (list, tuple)) and all(isinstance(m, str) for m in excluded)
    ):
        return False, f"modelopt {quant_algo.name} with unreadable excluded modules"
    mtp_layer = f"model.layers.{pretrained_config.num_hidden_layers}"
    unquantized = QuantConfig(exclude_modules=excluded).is_module_excluded_from_quantization(
        mtp_layer
    )
    words = f"{quant_algo.name} with the MTP layer {'unquantized' if unquantized else 'quantized'}"
    return quant_algo == QuantAlgo.NVFP4 and unquantized, words


def _mtp_eagle_linear(spec_config: Optional[Any], draft_len: int) -> bool:
    """The checkpoint's MTP layer driven by the one-model MTP-Eagle loop, as a
    linear chain of exactly ``draft_len`` draft tokens.

    No speculation and other draft lengths do not route here: the decode
    kernels are built for the target's rows per request. Nor do the
    suffix-automaton draft enhancer and relaxed acceptance: both are fixed for
    the engine's lifetime, and the draft fast paths implement neither.
    """
    return (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and spec_config.max_draft_len == draft_len
        and spec_config.max_total_draft_tokens == draft_len
        and not spec_config.use_dynamic_tree
        and spec_config.sa_config is None
        and not spec_config.use_relaxed_acceptance_for_thinking
    )


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

    # tp8: the attention heads and the experts both tensor-parallel over eight ranks,
    # every expert on every rank, no attention DP. The low-latency decode fast paths
    # of the sm103 tp8 target are built for this split and for its layer pattern.
    tp8 = (
        mapping.world_size == 8
        and mapping.tp_size == 8
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and mapping.moe_tp_size == 8
        and mapping.moe_ep_size == 1
        and not mapping.enable_attention_dp
    )
    pattern = tuple(getattr(pretrained_config, name, None) for name in _TP8_PATTERN_FIELDS)
    # GLM-5.2-NVFP4 (bf16 MTP layer) on 8 B300s (sm103), tp8, one-model MTP-Eagle as a
    # linear chain of five draft tokens. The quantization is read last, and only for
    # such a deployment.
    quantization = "not read"
    if (
        compute_capability == (10, 3)
        and shape == _SHAPE
        and pattern == _TP8_PATTERN
        and tp8
        and _mtp_eagle_linear(spec_config, _TP8_DRAFT_LEN)
    ):
        served, quantization = _nvfp4_with_bf16_mtp_layer(pretrained_config)
        if served:
            # fmt: off
            # isort: off
            from .glm_5_2_nvfp4__sm_103__tp8.modeling import ModelingV2Glm52Nvfp4Sm103Tp8
            # isort: on
            # fmt: on
            return ModelingV2Glm52Nvfp4Sm103Tp8

    spec = (
        "off"
        if spec_config is None
        else f"{spec_config.spec_dec_mode.name} (max_draft_len={draft_len}, "
        f"max_total_draft_tokens={draft_tokens})"
    )
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"index_n_heads={index_n_heads}, layer pattern {pattern}, "
        f"quantization {quantization}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} moe_ep={mapping.moe_ep_size} "
        f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp} "
        f"pp={mapping.pp_size} cp={mapping.cp_size}, speculative decoding {spec}; "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        "the glm_moe_dsa targets are GLM-5.2-NVFP4 on sm107, tp16 (16 ranks, "
        "tensor-parallel attention and experts, no attention DP), with one-model "
        "MTP-Eagle as a linear chain or no speculative decoding; GLM-5.2-NVFP4 "
        "on sm103, tep4 (4 ranks, tensor-parallel attention, expert-parallel MoE), "
        "with one-model MTP-Eagle at max_draft_len=1 or no speculative decoding, both "
        "with an fp8 latent KV cache (not the packed fp8_ds_mla layout); and the "
        "GLM-5.2 modelopt NVFP4 checkpoint (bf16 MTP layer) on sm103, tp8 (8 ranks, "
        "tensor-parallel attention and experts, no attention DP), with one-model "
        "MTP-Eagle as a linear chain at max_draft_len=5, without the suffix-automaton "
        "enhancer or relaxed acceptance"
    )
