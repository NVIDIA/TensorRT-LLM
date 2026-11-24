# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing for HYV4FlashForConditionalGeneration (HY4-Flash, recognized by its config shape)."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Optional, Tuple

from tensorrt_llm._torch.pyexecutor.config_utils import (
    get_hyv4_flash_layer_masks,
    unwrap_hyv4_flash_text_config,
)
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, embedded_dfly_draft_config
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes, tp_reaches_across_nodes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

_SM = (10, 3)

# (world size, attention DP, MoE TP, MoE EP); tep2 is the disaggregated context engine.
_LAYOUTS = {
    (2, False, 1, 2): "tep2",
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
    (8, False, 1, 8): "tep8",
}

_SHAPE = (42, 3072, 512, 16, 1536, 120832)
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "vocab_size",
)

# Compiled into the target's kernels: the DSA indexer's geometry and the window.
_GEOMETRY = {
    "index_topk": 2048,
    "index_n_heads": 16,
    "index_head_dim": 128,
    "sliding_window": 512,
}

_SPARSE_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41)


def layout(mapping: Mapping) -> Optional[str]:
    """The ``_LAYOUTS`` name of ``mapping``, or None.

    Attention runs TP over the whole world, or DP with the LM head whole; no PP, CP or MoE
    clusters.
    """
    if mapping.enable_attention_dp:
        attention_as_built = not mapping.enable_lm_head_tp_in_adp
    else:
        attention_as_built = mapping.attn_tp_size == mapping.tp_size
    if (
        not attention_as_built
        or mapping.tp_size != mapping.world_size
        or (mapping.pp_size, mapping.cp_size, mapping.moe_cluster_size) != (1, 1, 1)
    ):
        return None
    return _LAYOUTS.get(
        (mapping.world_size, mapping.enable_attention_dp, mapping.moe_tp_size, mapping.moe_ep_size)
    )


def _draft_is_embedded(spec_config: DFlashDecodingConfig, model: Optional[str]) -> bool:
    """Whether the DFlash draft is a checkpoint's own ``dfly_*`` draft, not a separate drafter.

    The loader may swap a hub id for its snapshot, so a path other than ``model`` counts too
    when its config declares an embedded draft.
    """
    draft = spec_config.speculative_model
    if draft is None:
        return True
    if model is not None and os.path.realpath(str(draft)) == os.path.realpath(str(model)):
        return True
    config_path = os.path.join(str(draft), "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, encoding="utf-8") as f:
            return bool(embedded_dfly_draft_config(json.load(f)))
    except (OSError, ValueError):
        return False


def speculation_refused(spec_config, model: Optional[str] = None) -> list[str]:
    """The speculative features the target does not run: only the embedded DFly2, five drafts."""
    if spec_config is None:
        return []
    if not isinstance(spec_config, DFlashDecodingConfig):
        return [f"{type(spec_config).__name__} (only the checkpoint's DFly2)"]
    refused = {
        "with a separate draft checkpoint": not _draft_is_embedded(spec_config, model),
        f"of variant {spec_config.spec_dec_variant} (only DFly2, variant 2)": (
            spec_config.spec_dec_variant not in (None, 2)
        ),
        f"with {spec_config.max_draft_len} drafts (only five)": spec_config.max_draft_len != 5,
        "with a draft tree": (
            spec_config.max_total_draft_tokens not in (None, spec_config.max_draft_len)
        ),
    }
    return [name for name, present in refused.items() if present]


@routes("HYV4FlashForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits."""
    pretrained_config = model_config.pretrained_config
    mapping, spec = model_config.mapping, model_config.spec_config
    text = unwrap_hyv4_flash_text_config(pretrained_config)
    shape = tuple(getattr(text, name, None) for name in _SHAPE_FIELDS)
    geometry = {name: getattr(text, name, None) for name in _GEOMETRY}
    try:
        sparse_mask, _ = get_hyv4_flash_layer_masks(pretrained_config)
    except ValueError as why:
        raise NoTarget(f"the config's layer types are not HY4-Flash's: {why}") from None
    sparse_layers = tuple(layer_idx for layer_idx, is_sparse in enumerate(sparse_mask) if is_sparse)
    quant = getattr(model_config.quant_config, "quant_algo", None)
    kv_cache_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    member = layout(mapping)
    tp_reach = tp_reaches_across_nodes(mapping)

    if not (
        compute_capability == _SM
        and quant == QuantAlgo.FP8_BLOCK_SCALES
        and kv_cache_algo in (None, QuantAlgo.FP8)
        and shape == _SHAPE
        and geometry == _GEOMETRY
        and sparse_layers == _SPARSE_LAYERS
        and member is not None
        and tp_reach
    ):
        raise NoTarget(
            f"compute capability {compute_capability}, world_size={mapping.world_size} "
            f"(tp={mapping.tp_size}, pp={mapping.pp_size}, cp={mapping.cp_size}, "
            f"attention_dp={mapping.enable_attention_dp}, moe_tp={mapping.moe_tp_size}, "
            f"moe_ep={mapping.moe_ep_size}, tp_reaches_across_nodes={tp_reach}), "
            f"quantization {quant}, KV cache {kv_cache_algo or 'bf16'}, config shape {shape}, "
            f"indexer/window {geometry}, sparse layers {sparse_layers}; the hyv4_flash target is "
            "the block-FP8 HY4-Flash checkpoint on sm103 with a bf16 or FP8 KV cache, at tp4, "
            "tep4, dep4, tep8 or tep2"
        )
    refused = speculation_refused(spec, llm_args.model)
    if refused:
        raise NoTarget(
            f"speculative decoding {', '.join(refused)}; the target runs the checkpoint's DFly2 "
            "drafting (five drafts) or no speculative decoding"
        )
    from .hy4_flash_fp8__sm_103__default.modeling import ModelingV2Hy4FlashFp8Sm103Default

    return ModelingV2Hy4FlashFp8Sm103Default
