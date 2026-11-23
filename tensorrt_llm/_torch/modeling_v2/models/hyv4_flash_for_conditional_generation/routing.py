# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a HYV4FlashForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployment the target below was written for. When none fits
it raises ``NoTarget`` stating those facts. In ``auto`` the engine then uses the
built-in implementation; in ``require`` it raises that error.

One target serves the whole checkpoint on sm_103, at tp4, tep4, dep4 and tep8
(two nodes), with no speculative decoding or the checkpoint's own DFly2 drafting
(a ``DFlashDecodingConfig`` of variant 2, five drafts). Its two-rank half --
the disaggregated deployment's context engine, tp2 with the experts split two
ways -- is the same code at attention TP 2. The checkpoint is
recognized by its config shape, the upstream idiom (``is_mla``,
``is_nemotron_hybrid`` do the same); a fine-tune with the same shape routes here
too, and the accuracy gate is what pins the identity.
"""

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

# The one GPU architecture the target is written for: GB300 (sm103).
_SM = (10, 3)

# The layouts the target has passed its gates on, as (world size, attention DP,
# MoE TP, MoE EP). tp4 splits the routed experts by intermediate, tep4 and dep4
# by expert, tep8 (two nodes) does too, and tep2 is the same split at the two
# ranks the disaggregated deployment runs its context engine on.
_LAYOUTS = {
    (2, False, 1, 2): "tep2",
    (4, False, 4, 1): "tp4",
    (4, False, 1, 4): "tep4",
    (4, True, 1, 4): "dep4",
    (8, False, 1, 8): "tep8",
}

# HY4-Flash's config shape, read from ``text_config``: (num_hidden_layers,
# hidden_size, n_routed_experts, num_experts_per_tok, moe_intermediate_size,
# vocab_size).
_SHAPE = (42, 3072, 512, 16, 1536, 120832)
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "vocab_size",
)

# What the target's kernels compile in beyond that shape: the DSA indexer's
# top-k, head count and head width, and the windowed layers' window.
_GEOMETRY = {
    "index_topk": 2048,
    "index_n_heads": 16,
    "index_head_dim": 128,
    "sliding_window": 512,
}

# The decoder layers that run the sparse MLA + DSA indexer (0-based); the rest
# run windowed MLA. The DFly2 draft layers reuse five of these.
_SPARSE_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41)


def layout(mapping: Mapping, layouts: dict = _LAYOUTS) -> Optional[str]:
    """The layout ``mapping`` describes in ``layouts``, or None.

    Attention runs TP over the whole world, or DP with the LM head whole. Pipeline and context
    parallelism and MoE clusters are other forwards. ``tep8`` spans two nodes, so a world
    spanning nodes is not itself a refusal: the attention allreduce's reach over them is
    checked in ``route``.
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
    return layouts.get(
        (mapping.world_size, mapping.enable_attention_dp, mapping.moe_tp_size, mapping.moe_ep_size)
    )


def declares_its_own_draft(checkpoint_dir: str) -> bool:
    """Whether the checkpoint at ``checkpoint_dir`` keeps a DFly2 draft inside itself.

    Such a checkpoint states the draft's geometry -- the target layers each draft
    layer reuses, the mask token -- as its own ``dfly_*`` keys, and a standalone
    drafter does not: it spells its geometry in a ``dflash_config`` block. A path
    that is not a readable checkpoint declares nothing either way.
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.isfile(config_path):
        return False
    try:
        with open(config_path, encoding="utf-8") as f:
            return bool(embedded_dfly_draft_config(json.load(f)))
    except (OSError, ValueError):
        return False


def draft_is_embedded_in_target(
    spec_config: DFlashDecodingConfig, model: Optional[str] = None
) -> bool:
    """Whether this DFlash config drafts from the checkpoint it is served with.

    HY4-Flash's DFly2 draft is the target checkpoint's own ``model.dfly_layers.*``
    namespace, so the draft is found by naming the target checkpoint: the engine's
    DFlash validation fills an unset ``speculative_model`` with the model the
    arguments carry (``TorchLlmArgs.model``), and the drafter reads the draft out
    of that same directory. A deployment may still spell that directory
    differently -- the loader swaps a hub repo id for the snapshot it downloaded
    -- and a path that is not ``model``'s is the embedded draft too when the
    checkpoint at it is one that declares the draft's geometry inside itself. A
    path that names neither is a separate draft checkpoint, which this target
    does not run.
    """
    draft = spec_config.speculative_model
    if draft is None:
        return True
    if model is not None and os.path.realpath(str(draft)) == os.path.realpath(str(model)):
        return True
    return declares_its_own_draft(str(draft))


def speculation_refused(spec_config, model: Optional[str] = None) -> list[str]:
    """The speculative features the target does not run; empty when it runs this config.

    The checkpoint drafts from five layers inside itself, so it takes no separate
    draft checkpoint and no draft tree, and only the DFly2 (variant 2) algorithm
    with its five drafts. ``model`` is the checkpoint the engine arguments carry;
    a DFlash config that names it (or another checkpoint holding its own draft)
    is that embedded draft, not a separate one.
    """
    if spec_config is None:
        return []
    if not isinstance(spec_config, DFlashDecodingConfig):
        return [f"{type(spec_config).__name__} (only the checkpoint's DFly2)"]
    refused = {
        "with a separate draft checkpoint": not draft_is_embedded_in_target(spec_config, model),
        f"of variant {spec_config.spec_dec_variant} (only DFly2, variant 2)": (
            spec_config.spec_dec_variant not in (None, 2)
        ),
        f"with {spec_config.max_draft_len} drafts (only five)": spec_config.max_draft_len != 5,
        "with a draft tree": (
            spec_config.max_total_draft_tokens not in (None, spec_config.max_draft_len)
        ),
    }
    return [name for name, present in refused.items() if present]


# HY4-Flash publishes ``HYV4FlashForConditionalGeneration``, and the config
# loader's composite branch writes the same name back.
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
        and all(geometry[name] == value for name, value in _GEOMETRY.items())
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
            "tep4, dep4 or tep8 (two nodes), or at the disaggregated deployment's context "
            "width tep2 (tp2 with the experts split two ways)"
        )
    refused = speculation_refused(spec, llm_args.model)
    if refused:
        raise NoTarget(
            f"speculative decoding {', '.join(refused)}; the target runs the checkpoint's DFly2 "
            "drafting (five drafts) or no speculative decoding"
        )
    from .hf_003107f_orig__sm_103__unified.modeling import (
        ModelingV2Hyv4FlashForConditionalGeneration,
    )

    return ModelingV2Hyv4FlashForConditionalGeneration
