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

import os
import re
from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, PrefillCudaGraphBackend
from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode
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


# Dense MLP projections a mixed-precision export may quantize in the layers before
# the first MoE layer. The tp8 target's dense MLP fast path runs only on bf16
# projections and leaves quantized ones to the built-in MLP.
_DENSE_MLP_MODULES = frozenset(
    ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.gate_up_proj")
)
_LAYER_MODULE = re.compile(r"model\.layers\.(\d+)\.(.+)")


def _nvfp4_experts_mixed_precision(normalized: dict, pretrained_config: Any) -> Tuple[bool, str]:
    """Whether a MIXED_PRECISION export quantizes only NVFP4 routed experts (in every MoE
    layer) and dense MLPs before the first MoE layer, as GLM-5.3-NVFP4 does, and why not.
    """
    layers = normalized.get("quantized_layers")
    if not isinstance(layers, dict) or not layers:
        return False, "MIXED_PRECISION without a readable per-layer map"
    num_layers = pretrained_config.num_hidden_layers
    first_moe = getattr(pretrained_config, "first_k_dense_replace", None)
    if not isinstance(first_moe, int):
        return False, "MIXED_PRECISION on a config without first_k_dense_replace"
    nvfp4_expert_layers = set()
    for name, entry in layers.items():
        match = _LAYER_MODULE.fullmatch(name) if isinstance(name, str) else None
        try:
            # read_modelopt_quant_config canonicalized the algorithm aliases; any other
            # spelling (e.g. lowercase) is unreadable here and declines.
            algo = QuantAlgo(entry.get("quant_algo") if isinstance(entry, dict) else None)
        except ValueError:
            return False, f"MIXED_PRECISION with an unreadable algorithm for {name!r}"
        if match is None:
            return False, f"MIXED_PRECISION quantizing {name!r} ({algo.name})"
        layer, module = int(match.group(1)), match.group(2)
        if layer >= num_layers:  # the per-layer map, not only exclude_modules, must agree
            return False, f"MIXED_PRECISION with the MTP layer quantized ({name!r}, {algo.name})"
        if module == "mlp.experts" and layer >= first_moe and algo == QuantAlgo.NVFP4:
            nvfp4_expert_layers.add(layer)
        elif not (module in _DENSE_MLP_MODULES and layer < first_moe):
            return False, f"MIXED_PRECISION quantizing {name!r} as {algo.name}"
    missing = sorted(set(range(first_moe, num_layers)) - nvfp4_expert_layers)
    if missing:
        return False, f"MIXED_PRECISION without NVFP4 routed experts in layers {missing}"
    return True, "MIXED_PRECISION (NVFP4 routed experts)"


def _nvfp4_with_bf16_mtp_layer(pretrained_config: Any) -> Tuple[bool, str]:
    """Whether the checkpoint has NVFP4 MoE layers and a bf16 MTP layer, and its quantization
    in words: a uniform NVFP4 export (GLM-5.2), or a MIXED_PRECISION one whose per-layer map
    ``_nvfp4_experts_mixed_precision`` accepts (GLM-5.3).

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
    if quant_algo == QuantAlgo.MIXED_PRECISION and unquantized:
        served, layers = _nvfp4_experts_mixed_precision(normalized, pretrained_config)
        return served, f"{layers} with the MTP layer unquantized"
    words = f"{quant_algo.name} with the MTP layer {'unquantized' if unquantized else 'quantized'}"
    return quant_algo == QuantAlgo.NVFP4 and unquantized, words


def _unquantized_mlps(model_config: ModelConfig) -> tuple[bool, str]:
    """Check dense/shared projections, including an active MTP shared expert.

    A physical split declaration takes precedence over a fused exclusion or
    entry; fused entries apply only to gate/up projections. Read weight-only
    flags without caching quantization modes before the loader's KV override.
    """
    quant_config = model_config.quant_config
    pretrained_config = model_config.pretrained_config
    num_layers = pretrained_config.num_hidden_layers + (model_config.spec_config is not None)
    for layer_idx in range(num_layers):
        mlp_name = f"model.layers.{layer_idx}.mlp"
        if layer_idx >= pretrained_config.first_k_dense_replace:
            mlp_name += ".shared_experts"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            projection_name = f"{mlp_name}.{projection}"
            if quant_config.is_module_excluded_from_quantization(projection_name):
                continue
            projection_quant = None
            if model_config.quant_config_dict is not None:
                projection_quant = model_config.quant_config_dict.get(projection_name)
            if projection_quant is None:
                if projection != "down_proj" and quant_config.is_module_excluded_from_quantization(
                    f"{mlp_name}.gate_up_proj"
                ):
                    continue
                if model_config.quant_config_dict is None:
                    projection_quant = quant_config
                elif projection != "down_proj":
                    projection_quant = model_config.quant_config_dict.get(
                        f"{mlp_name}.gate_up_proj"
                    )
            if (
                projection_quant is not None
                and QuantMode.from_quant_algo(projection_quant.quant_algo).has_any_quant()
            ):
                return (
                    False,
                    f"quantized MLP projection {projection_name}: {projection_quant.quant_algo}",
                )
    return True, "unquantized dense and shared MLPs"


def _tep4_checkpoint(model_config: ModelConfig) -> tuple[bool, str]:
    """Check indexer ownership and unquantized dense/shared MLP declarations.

    This does not establish the other loader requirements: NVFP4 trunk experts,
    bf16 attention and remaining MTP weights, or exact tensor shapes and dtypes.
    """
    pretrained_config = model_config.pretrained_config
    indexer_types = getattr(pretrained_config, "indexer_types", None)
    if indexer_types is None:
        return False, "indexer_types absent"
    if not isinstance(indexer_types, (list, tuple)):
        return False, f"indexer_types is {type(indexer_types).__name__}, not a list or tuple"
    num_layers = pretrained_config.num_hidden_layers
    if len(indexer_types) < num_layers:
        return False, f"indexer_types covers {len(indexer_types)} of {num_layers} trunk layers"
    if indexer_types[0] != "full":
        return False, f"indexer_types[0]={indexer_types[0]!r} (layer 0 must own an indexer)"
    sparse_config = model_config.sparse_attention_config
    if sparse_config is None:
        return False, "sparse_attention_config absent; no indexer K-cache ownership policy"
    from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
        derive_indexer_k_cache_layer_mask,
    )

    mtp_active = model_config.spec_config is not None
    cache_owners = derive_indexer_k_cache_layer_mask(
        sparse_config, pretrained_config, num_layers + mtp_active
    )
    for layer_idx, indexer_type in enumerate(indexer_types[: pretrained_config.num_hidden_layers]):
        if indexer_type == "full" and not cache_owners[layer_idx]:
            return False, (
                f"indexer_types layer {layer_idx} has no indexer K-cache under sparse_attention_config "
                "(index_topk_pattern / index_topk_freq / index_skip_topk_offset)"
            )
    if mtp_active and not cache_owners[num_layers]:
        return False, f"active MTP indexer layer {num_layers} has no indexer K-cache"
    return _unquantized_mlps(model_config)


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
    # pure TP: attention heads and the expert intermediate dimension sharded over
    # 4, 8, or 16 ranks, no attention DP, no expert parallelism. A weight-layout
    # question, which is why it selects a target rather than a runtime branch.
    pure_tp = (
        mapping.tp_size in (4, 8, 16)
        and mapping.world_size == mapping.tp_size
        and mapping.moe_ep_size == 1
        and mapping.moe_tp_size == mapping.tp_size
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
    # GLM-5.2-NVFP4 on Rubin (sm107), TP4/TP8/TP16, MTP-Eagle or no speculation.
    if (
        compute_capability == (10, 7)
        and shape == _SHAPE
        and index_n_heads == 32
        and pure_tp
        and mtp_eagle_or_none
        and kv_cache_quant_algo == QuantAlgo.FP8
    ):
        from .glm_5_2_nvfp4__sm_107__low_latency.modeling import ModelingV2Glm52Nvfp4Sm107LowLatency

        return ModelingV2Glm52Nvfp4Sm107LowLatency

    # Replicated attention and expert-sharded MoE across 4 or 16 ranks.
    dep4_or_16 = (
        mapping.world_size == mapping.tp_size == mapping.moe_ep_size
        and mapping.world_size in (4, 16)
        and mapping.moe_tp_size == 1
        and mapping.enable_attention_dp
        and mapping.pp_size == mapping.cp_size == 1
    )
    if (
        compute_capability == (10, 7)
        and shape == _SHAPE
        and index_n_heads == 32
        and dep4_or_16
        and mtp_eagle_or_none
        and kv_cache_quant_algo == QuantAlgo.NVFP4
        and model_config.moe_backend == "MEGAMOE_CUTEDSL"
    ):
        if (
            llm_args.moe_config.load_balancer is not None
            or model_config.moe_load_balancer is not None
        ):
            raise NoTarget(
                "GLM high-throughput requires fixed expert placement; "
                "moe_config.load_balancer (EPLB) is not supported"
            )
        if os.environ.get("MEGAMOE_TACTIC_AUTOTUNE", "0") == "1":
            raise NoTarget(
                "MEGAMOE_TACTIC_AUTOTUNE=1 is not supported by the "
                "GLM high-throughput MegaMoE kernels"
            )
        if llm_args.prefill_cuda_graph_backend == PrefillCudaGraphBackend.PIECEWISE:
            raise NoTarget(
                "GLM high-throughput does not support prefill_cuda_graph_backend='piecewise'"
            )
        # fmt: off
        # isort: off
        from .glm_5_2_nvfp4__sm_107__high_throughput.modeling import ModelingV2Glm52Nvfp4Sm107HighThroughput
        # isort: on
        # fmt: on
        return ModelingV2Glm52Nvfp4Sm107HighThroughput

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
    # one-model MTP-Eagle loop, and no other draft path. A linear chain of 1, 3 or 5
    # draft tokens per step: a generation step carries D + 1 query rows per sequence,
    # and the decode attention's grid is compiled for that width. The specialised MLA
    # rope kernel and the base-row index kind are built for width 2 and, at MTP3 /
    # MTP5, also for D + 1 (4 / 6). A dynamic tree is a different forward: its tree
    # masks and position offsets are consumed by neither the MLA / RoPE path nor the
    # top-k, which assume a linear chain. That holds even when the tree is capped at
    # the chain length (max_total_draft_tokens == max_draft_len), so the flag itself
    # is declined. The engine resolves an unset draft length to 1 before this runs.
    draft_len = getattr(spec_config, "max_draft_len", None) or 1
    draft_tokens = getattr(spec_config, "max_total_draft_tokens", None) or draft_len
    dynamic_tree = bool(getattr(spec_config, "use_dynamic_tree", False))
    mtp_linear_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and not dynamic_tree
        and draft_len in (1, 3, 5)
        and draft_tokens == draft_len
    )
    tep4_checkpoint = "not read"
    # GLM-5.2-NVFP4 on 4 B300s (sm103), tep4, MTP-Eagle with 1, 3 or 5 draft
    # tokens or no speculation. index_n_heads is checked because a
    # DeepSeek-V3.2-shaped DSA checkpoint would otherwise match the shape.
    if (
        compute_capability == (10, 3)
        and shape == _SHAPE
        and index_n_heads == 32
        and tep4
        and mtp_linear_or_none
        and kv_cache_quant_algo == QuantAlgo.FP8
    ):
        served, tep4_checkpoint = _tep4_checkpoint(model_config)
        if served:
            from .glm_5_2_nvfp4__sm_103__tep4.modeling import ModelingV2Glm52Nvfp4Sm103Tep4

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
    # GLM-5.2-NVFP4 (bf16 MTP layer) or GLM-5.3-NVFP4 (MIXED_PRECISION, same MoE layers)
    # on 8 B300s (sm103), tp8, one-model MTP-Eagle as a
    # linear chain of five draft tokens, with an fp8 latent KV cache. The quantization is
    # read last, and only for such a deployment.
    quantization = "not read"
    if (
        compute_capability == (10, 3)
        and shape == _SHAPE
        and pattern == _TP8_PATTERN
        and tp8
        and _mtp_eagle_linear(spec_config, _TP8_DRAFT_LEN)
        and kv_cache_quant_algo == QuantAlgo.FP8
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
        f"max_total_draft_tokens={draft_tokens}"
        f"{', dynamic tree' if dynamic_tree else ''})"
    )
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"index_n_heads={index_n_heads}, layer pattern {pattern}, "
        f"tep4 checkpoint {tep4_checkpoint}, "
        f"quantization {quantization}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} moe_ep={mapping.moe_ep_size} "
        f"moe_tp={mapping.moe_tp_size} attention_dp={mapping.enable_attention_dp} "
        f"moe_backend={getattr(model_config, 'moe_backend', None)} "
        f"pp={mapping.pp_size} cp={mapping.cp_size}, speculative decoding {spec}; "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        "the glm_moe_dsa targets are GLM-5.2-NVFP4 on sm107, TP4/TP8/TP16 (4/8/16 ranks, "
        "tensor-parallel attention and experts, no attention DP), with one-model "
        "MTP-Eagle as a linear chain or no speculative decoding; GLM-5.2-NVFP4 "
        "on sm103, tep4 (4 ranks, tensor-parallel attention, expert-parallel MoE; "
        "owning indexer schedule and unquantized dense/shared MLP projections), "
        "with one-model MTP-Eagle as a linear chain at max_draft_len 1, 3 or 5 (no dynamic "
        "tree) or no speculative decoding, both "
        "with an fp8 latent KV cache (not the packed fp8_ds_mla layout); and the "
        "GLM-5.2 modelopt NVFP4 checkpoint (bf16 MTP layer) or GLM-5.3-NVFP4 (MIXED_PRECISION, "
        "same MoE layers) on sm103, tp8 (8 ranks, "
        "tensor-parallel attention and experts, no attention DP), with one-model "
        "MTP-Eagle as a linear chain at max_draft_len=5, without the suffix-automaton "
        "enhancer or relaxed acceptance, also with an fp8 latent KV cache"
        "; the sm107 high-throughput target uses attention DP with TP4/EP4 or TP16/EP16, "
        "NVFP4 KV cache and MEGAMOE_CUTEDSL"
    )
