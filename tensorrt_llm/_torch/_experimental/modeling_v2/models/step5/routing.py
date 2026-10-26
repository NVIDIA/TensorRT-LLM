# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Step5ForConditionalGeneration config lands. Read this file and you know.

``route`` compares the facts that fix which code runs against the deployments the
target below was written for, and raises ``NoTarget`` stating those facts when none
fits (``auto`` then falls back to the built-in implementation, of which Step-5 has
none; ``require`` raises). The config loader keeps the checkpoint's published
architecture name, so ``_router_index`` keys on it. Shape criteria are read off the
composite's ``text_config`` and ``vision_config``, never off the fields mirrored onto
the composite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, tp_reaches_across_nodes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The GPU the target is written for: sm_103 (B300, GB300).
_SMS = frozenset({(10, 3)})

# (decoder layers, hidden, vocab, query heads, KV groups, head_dim, sliding window,
#  routed experts, experts per token). Shape-sniffing also matches fine-tunes; the
# accuracy gate pins identity.
_TEXT_SHAPE = (92, 4096, 128896, 64, 4, 192, 512, 352, 8)

# The sparse indexer and sparse attention of the full-attention layers:
# (indexer heads, indexer K heads, proxy_dim, indexer rope dim, top-k, region block size,
#  provider groups, compression method, attention impl).
_SPARSE_SHAPE = (16, 1, 256, 32, 512, 8, 4, "csa_block_compress", "sparse_gqa")

# (layers, width, heads, patch, image size, activation) of the perception encoder.
_VISION_SHAPE = (47, 1536, 16, 14, 728, "quick_gelu")

# The weight format the target loads: ModelOpt NVFP4 routed experts, the rest bf16.
_WEIGHTS = QuantAlgo.NVFP4

# The parallel layouts, as (world_size, attention_dp, moe_tp, moe_ep), the target is
# written for. Its shares come from the mapping, so others may compute too; a layout is
# admitted here once it is one the target serves.
_LAYOUTS = {
    (4, False, 4, 1): "tp4",  # attention TP4, experts split by intermediate (LLM API default)
    (4, False, 1, 4): "tep4",  # attention TP4, experts EP4
    (4, True, 1, 4): "dep4",  # attention DP4, experts EP4
    (2, True, 1, 2): "dep2",  # attention DP2, experts EP2
}

# KV-cache algorithms the target runs: the checkpoint's bf16 (None), FP8 and NVFP4.
_KV_ALGOS = (None, QuantAlgo.FP8, QuantAlgo.NVFP4)


def _sparse_shape(text: Any) -> tuple:
    sparse = getattr(text, "sparse_config", None) or {}
    return (
        sparse.get("sparse_indexer_num_heads"),
        sparse.get("sparse_indexer_num_k_heads"),
        sparse.get("proxy_dim"),
        sparse.get("sparse_indexer_rope_dim"),
        sparse.get("topk"),
        sparse.get("region_block_size"),
        sparse.get("num_provider_groups"),
        sparse.get("compression_method"),
        sparse.get("attention_impl"),
    )


def _weight_format(pretrained_config: Any, model_config: Optional[ModelConfig]) -> Any:
    """The checkpoint's weight quantization: the loaded ``QuantConfig`` when given, else
    the composite's inline ``quantization_config``."""
    if model_config is not None:
        return model_config.quant_config.quant_algo
    inline = getattr(pretrained_config, "quantization_config", None) or {}
    algo = inline.get("quant_algo")
    return QuantAlgo(algo) if algo in QuantAlgo.__members__ else algo


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
    kv_cache_quant_algo: Optional[Any],
    model_config: Optional[ModelConfig] = None,
    llm_args: Optional[TorchLlmArgs] = None,
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    ``kv_cache_quant_algo`` is the KV-cache algorithm the engine will run with (see
    ``_router_index._kv_cache_quant_algo``), not the checkpoint's alone. The target is
    imported where it is returned, so declining costs no import of modeling code.
    """
    text = getattr(pretrained_config, "text_config", None)
    vision = getattr(pretrained_config, "vision_config", None)
    if text is None or vision is None:
        raise NoTarget(
            f"{type(pretrained_config).__name__} carries no text_config/vision_config; "
            "the step5 target reads both halves of the step5_vl composite"
        )

    shape = (
        text.num_hidden_layers,
        text.hidden_size,
        text.vocab_size,
        text.num_attention_heads,
        text.num_attention_groups,
        text.head_dim,
        text.sliding_window,
        text.moe_num_experts,
        text.moe_top_k,
    )
    sparse = _sparse_shape(text)
    vshape = (
        vision.layers,
        vision.width,
        vision.heads,
        vision.patch_size,
        vision.image_size,
        vision.hidden_act,
    )
    weights = _weight_format(pretrained_config, model_config)
    if (
        compute_capability not in _SMS
        or shape != _TEXT_SHAPE
        or sparse != _SPARSE_SHAPE
        or vshape != _VISION_SHAPE
        or weights != _WEIGHTS
    ):
        raise NoTarget(
            f"compute capability {compute_capability}, text shape {shape}, sparse shape "
            f"{sparse}, vision shape {vshape}, weights {weights}; the step5 target is "
            "Step-5-Preview-NVFP4 on sm103"
        )

    if kv_cache_quant_algo not in _KV_ALGOS:
        raise NoTarget(
            f"KV cache {kv_cache_quant_algo}; the step5 target stores K/V as the "
            "checkpoint's bf16, as FP8 or as NVFP4"
        )

    split = (
        f"world_size={mapping.world_size} pp={mapping.pp_size} cp={mapping.cp_size} "
        f"attention_dp={mapping.enable_attention_dp} moe_tp={mapping.moe_tp_size} "
        f"moe_ep={mapping.moe_ep_size}"
    )
    if mapping.pp_size != 1 or mapping.cp_size != 1:
        raise NoTarget(
            f"parallel layout {split}; the step5 target implements no pipeline or context "
            "parallelism"
        )
    if getattr(mapping, "enable_lm_head_tp_in_adp", False):
        raise NoTarget(
            f"parallel layout {split} with enable_lm_head_tp_in_adp; the step5 target "
            "keeps lm_head whole under attention DP"
        )
    if not mapping.enable_attention_dp and mapping.attn_tp_size != mapping.tp_size:
        raise NoTarget(
            f"parallel layout {split} with attn_tp_size={mapping.attn_tp_size} != "
            f"tp_size={mapping.tp_size}; the step5 target splits the heads over the whole "
            "TP group"
        )
    layout = (
        mapping.world_size,
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    if layout not in _LAYOUTS:
        raise NoTarget(
            f"parallel layout {split}; the step5 target serves tp4 (experts split by "
            "intermediate), tep4 (experts EP4), dep4 (attention DP, experts EP4) and dep2 "
            "(attention DP, experts EP2)"
        )
    if not tp_reaches_across_nodes(mapping):
        raise NoTarget(
            f"parallel layout {split} with attention TP over {mapping.tp_size} ranks across "
            f"nodes of {mapping.gpus_per_node} GPUs and no multi-node NVLink; the step5 "
            "target reduces attention TP inside one NVLink domain only"
        )

    # The checkpoint's MTP layers (num_nextn_predict_layers) are not built: no draft
    # path routes here.
    if spec_config is not None:
        raise NoTarget(
            f"speculative decoding {spec_config.spec_dec_mode.name}; the step5 target "
            "implements no speculative decoding"
        )

    # One line on purpose: the claims suite reads the routed targets off
    # `from .<target>.modeling import <Class>` lines.
    # fmt: off
    # isort: off
    from .step5_preview_nvfp4__sm_103__unified.modeling import ModelingV2Step5PreviewNvfp4Sm103Unified
    # isort: on
    # fmt: on

    return ModelingV2Step5PreviewNvfp4Sm103Unified
