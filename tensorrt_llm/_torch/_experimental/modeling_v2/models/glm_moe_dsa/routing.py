# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a GlmMoeDsaForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the target below was written for. When none
fits, ``route`` raises ``NoTarget`` stating those facts. In ``auto`` the engine
then uses the built-in GlmMoeDsa implementation; in ``require`` it raises that
error.

One target, ``glm_5_2_nvfp4__sm_103__default``, serves one checkpoint on one GPU
architecture at the layouts below; it picks the fast paths of each layout when
it is built. The fast paths are built for the checkpoint's operand shapes and
layer pattern, the ranks and the rows per request (the target token and its
draft tokens), so each of those facts is a routing criterion rather than a
runtime branch. A layout is admitted only once it has passed its gate:

* tp8: attention and experts tensor-parallel over eight ranks, five draft
  tokens. The fast paths were designed and tuned for this layout;
* tep: attention tensor-parallel over four or eight ranks and the experts
  expert-parallel over the same ranks, five draft tokens;
* dep: attention data parallelism and the experts expert-parallel over the same
  ranks, at the rank counts, LM-head splits and draft lengths of ``_DEP``. Two
  more facts fix what its fast paths serve, the KV-cache dtype and the MoE
  backend; the target checks both when it is built, since ``route`` does not see
  the MoE backend.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.modelopt_config import (
    canonicalize_quant_algo,
    is_modelopt_quant_config,
    read_modelopt_quant_config,
)

from ..._router_index import NoTarget

# The one GPU architecture the fast paths are built and validated for.
_SM = (10, 3)

# The checkpoint's config shape; see the gpt_oss routing module for what
# shape-sniffing does and does not pin. The head count fixes the per-rank
# attention operands at tp8, and index_topk fixes the sparse-attention selection
# width the decode kernels are built for. The last four fields fix the layer
# pattern: the dense layers before the first MoE layer, and which layers run
# their own indexer and which reuse an earlier layer's selection (an explicit
# index_topk_pattern overrides the frequency and offset, so it must be absent).
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "n_routed_experts",
    "q_lora_rank",
    "num_attention_heads",
    "index_topk",
    "first_k_dense_replace",
    "index_topk_freq",
    "index_skip_topk_offset",
    "index_topk_pattern",
)
_SHAPE = (78, 6144, 256, 2048, 64, 2048, 3, 4, 3, None)

# Ranks of tensor-parallel attention and tensor-parallel experts.
_TP = 8

# Ranks of tensor-parallel attention with the experts expert-parallel over the same ranks.
_TEP = (4, 8)

# Draft tokens per request. The decode kernels take the target token plus this
# many draft tokens per request in one step.
_DRAFT_LEN = 5

# Attention data parallelism with the experts expert-parallel over the same ranks: per
# (ranks, LM head tensor-parallel in attention DP), the draft tokens per request.
_DEP = {(16, False): 3, (8, False): 3, (8, True): 5}


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


def _tp8(m: Any) -> bool:
    """Attention heads and experts both tensor-parallel over eight ranks, with every
    expert on every rank and no attention data parallelism.

    The split decides which weights each rank loads and into which shapes, so it is
    a routing criterion; the target picks the layout's fast paths when it is built.
    """
    return (
        m.world_size == _TP
        and m.tp_size == _TP
        and m.pp_size == 1
        and m.cp_size == 1
        and m.moe_tp_size == _TP
        and m.moe_ep_size == 1
        and not m.enable_attention_dp
    )


def _tep(m: Any) -> bool:
    """Attention heads tensor-parallel over four or eight ranks and the experts expert-parallel
    over the same ranks (each rank holds whole experts), with no attention data parallelism.

    The split decides which weights each rank loads and into which shapes, so it is
    a routing criterion; the target picks the layout's fast paths when it is built.
    """
    return (
        m.world_size == m.tp_size
        and m.tp_size in _TEP
        and m.pp_size == 1
        and m.cp_size == 1
        and m.moe_tp_size == 1
        and m.moe_ep_size == m.tp_size
        and not m.enable_attention_dp
    )


def _dep(m: Any) -> bool:
    """Attention data parallelism (each rank runs the whole attention on its own requests)
    and the experts expert-parallel over the same ranks, at a rank count and LM-head split
    of ``_DEP``.

    The split decides which weights each rank loads and into which shapes, so it is
    a routing criterion; the target picks the layout's fast paths when it is built.
    """
    return (
        m.world_size == m.tp_size
        and m.pp_size == 1
        and m.cp_size == 1
        and m.moe_tp_size == 1
        and m.moe_ep_size == m.world_size
        and m.enable_attention_dp
        and (m.world_size, bool(m.enable_lm_head_tp_in_adp)) in _DEP
    )


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


def _speculation(spec_config: Optional[Any]) -> str:
    if spec_config is None:
        return "off"
    words = (
        f"{spec_config.spec_dec_mode.name} max_draft_len={spec_config.max_draft_len} "
        f"max_total_draft_tokens={spec_config.max_total_draft_tokens}"
    )
    if getattr(spec_config, "use_dynamic_tree", False):
        words += " dynamic tree"
    if getattr(spec_config, "sa_config", None) is not None:
        words += " with the suffix-automaton enhancer"
    if getattr(spec_config, "use_relaxed_acceptance_for_thinking", False):
        words += " with relaxed acceptance"
    return words


def _layout_and_speculation(mapping: Any, spec_config: Optional[Any]) -> bool:
    """One of the admitted layouts, with the speculation its fast paths are built for."""
    if _tp8(mapping) or _tep(mapping):
        return _mtp_eagle_linear(spec_config, _DRAFT_LEN)
    if _dep(mapping):
        draft_len = _DEP[(mapping.world_size, bool(mapping.enable_lm_head_tp_in_adp))]
        return _mtp_eagle_linear(spec_config, draft_len)
    return False


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
    kv_cache_quant_algo: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    ``kv_cache_quant_algo`` is the KV-cache algorithm the engine will run with. It
    selects the fast paths of a dep deployment, which the target checks together
    with the MoE backend when it is built, so ``route`` only reports it.

    The target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    # A GlmMoeDsa config without one of these fields is another assembly: it reads as
    # None and the shape does not match.
    shape = tuple(getattr(pretrained_config, name, None) for name in _SHAPE_FIELDS)

    # GLM-5.2-NVFP4 (bf16 MTP layer) on sm103 at an admitted layout with its speculation.
    # The quantization is read last, and only for such a deployment.
    quantization = "not read"
    if (
        compute_capability == _SM
        and shape == _SHAPE
        and _layout_and_speculation(mapping, spec_config)
    ):
        served, quantization = _nvfp4_with_bf16_mtp_layer(pretrained_config)
        if served:
            from .glm_5_2_nvfp4__sm_103__default.modeling import ModelingV2Glm52Nvfp4Sm103Default

            return ModelingV2Glm52Nvfp4Sm103Default

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, quantization "
        f"{quantization}, parallel ws={mapping.world_size} tp={mapping.tp_size} "
        f"pp={mapping.pp_size} cp={mapping.cp_size} moe_tp={mapping.moe_tp_size} "
        f"moe_ep={mapping.moe_ep_size} attention_dp={mapping.enable_attention_dp} "
        f"lm_head_tp_in_adp={mapping.enable_lm_head_tp_in_adp}, "
        f"speculative decoding {_speculation(spec_config)}, KV cache "
        f"{kv_cache_quant_algo or 'bf16'}; the glm_moe_dsa target is "
        "the GLM-5.2 modelopt NVFP4 checkpoint (bf16 MTP layer) on sm103, with one-model "
        "MTP-Eagle as a linear chain, without the suffix-automaton enhancer or relaxed "
        "acceptance, at tp8 (8 ranks of tensor-parallel attention and experts) or tep (4 or 8 "
        "ranks of tensor-parallel attention with the experts expert-parallel over the same "
        "ranks), both without attention DP and at max_draft_len 5, or at dep (attention DP "
        "with the experts expert-parallel over the same ranks: 16 ranks at max_draft_len 3, "
        "8 ranks at max_draft_len 3, or 8 ranks with the LM head tensor-parallel in attention "
        "DP at max_draft_len 5)"
    )
