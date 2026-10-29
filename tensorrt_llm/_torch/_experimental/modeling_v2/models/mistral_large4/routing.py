# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MistralLarge4ForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs, compared
against the deployments the targets below were written for. When none fits it raises
``NoTarget`` stating those facts. There is no built-in implementation of this architecture,
so ``auto`` then has nothing to fall back to and ``require`` raises the same message.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from ..._router_index import NoTarget

_LAYERS = 72

# The per-layer structure the forward dispatches on, as this checkpoint states it. Every
# fourth layer takes no rope, in MLA and in its indexer; the NoPE layers past 35 own no latent
# and read layer 35's (itself NoPE); the first 17 layers run their own indexer and after that
# the two layers following each indexer pair reuse its selection. The forward is written
# against exactly this pattern -- every KV sharer NoPE, its source NoPE, layer 0 an indexer
# layer -- so another pattern is another forward.
_NOPE = [i % 4 == 3 for i in range(_LAYERS)]
_KV_SOURCE = [35 if i > 35 and i % 4 == 3 else i for i in range(_LAYERS)]
_INDEX_SHARING = ["full" if i < 17 or i % 4 in (0, 3) else "shared" for i in range(_LAYERS)]

# (layers, hidden, heads, q_lora, kv_lora, qk_nope, qk_rope, v, routed experts, top-k, shared
# experts, expert width, dense width, dense layers, indexer heads, indexer head dim, index_topk,
# vocab), in `route`'s order.
_SHAPE = (72, 6144, 64, None, 512, 64, 64, 128, 128, 4, 1, 6144, 24576, 1, 32, 128, 2048, 204800)

# The ModelOpt MIXED_PRECISION export's per-module quantization, as the checkpoint states it,
# by Mistral module name: the routed experts of every MoE layer NVFP4 W4A4 (16-input blocks);
# `wq`, `wo`, each indexer's `wq_b`, the shared experts and layer 0's dense MLP FP8 128x128
# blocks with dynamic 1x128 activations (FP8_PB_WO); every other module bf16. Which kernel each
# projection runs follows from it, so the whole map is compared.
_NVFP4_EXPERTS = {"quant_algo": "NVFP4", "group_size": 16}
_FP8_PB_WO = {"quant_algo": "FP8_PB_WO"}
_MIXED_PRECISION_LAYERS = {
    **{f"layers.{i}.experts": _NVFP4_EXPERTS for i in range(1, _LAYERS)},
    **{f"layers.{i}.attention.{m}": _FP8_PB_WO for i in range(_LAYERS) for m in ("wq", "wo")},
    **{
        f"layers.{i}.attention.indexer.wq_b": _FP8_PB_WO
        for i in range(_LAYERS)
        if _INDEX_SHARING[i] == "full"
    },
    **{f"layers.{i}.shared_experts.w{j}": _FP8_PB_WO for i in range(1, _LAYERS) for j in (1, 2, 3)},
    **{f"layers.0.feed_forward.w{j}": _FP8_PB_WO for j in (1, 2, 3)},
}


def _native_dspark(spec: Any) -> bool:
    """The checkpoint's trained DSpark block and the runtime this target implements."""
    if spec is None:
        return True
    native = getattr(spec, "_draft_model_config", None) or {}
    head = native.get("dspark") or {}
    return (
        spec.spec_dec_mode.name == "DSPARK"
        and spec.max_draft_len == spec.block_size == 3
        and spec.target_layer_ids == [1, 18, 36, 54, 71]
        and spec.mask_token_id == _SHAPE[-1]
        and spec.markov_rank == 512
        and spec.markov_head_type == "rnn"
        and spec.attention_backend in ("AUTO", "VANILLA")
        and spec.draft_len_schedule is None
        and not spec._use_shared_kv_cache
        and spec._allow_separate_draft_kv_cache
        and native.get("n_layers") == 3
        and native.get("depth") == 3
        and native.get("ragged_attention") == "2048,2048,2048"
        and native.get("norm_strategy") == "eagle"
        and head.get("draft_input") == "position_specific"
        and head.get("head_type") == "rnn"
        and head.get("markov_rank") == 512
    )


def _yarn_mscale(factor: float, mscale: float) -> float:
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def _per_layer_structure(cfg: Any) -> bool:
    return (
        [bool(f) for f in getattr(cfg, "nope_layer_flags", None) or []] == _NOPE
        and [bool(f) for f in getattr(cfg, "indexer_nope_layer_flags", None) or []] == _NOPE
        and [int(s) for s in getattr(cfg, "kv_source_layers", None) or []] == _KV_SOURCE
        and list(getattr(cfg, "index_sharing", None) or []) == _INDEX_SHARING
    )


def _math_switches(cfg: Any) -> bool:
    """The operations the forward contains beyond the shape and the weight formats: SSMax on
    both queries, LeRoPE, the logit soft-cap, softmax top-k routing renormalised without
    groups, interleaved YaRN with no magnitude scaling, and an fp8 latent cache."""
    rope = getattr(cfg, "rope_parameters", None) or {}
    factor = float(rope.get("factor", 1.0))
    return (
        getattr(cfg, "ssmax_scaling", False) is True
        and getattr(cfg, "indexer_ssmax_scaling", False) is True
        and getattr(cfg, "lerope", False) is True
        and getattr(cfg, "softmax_tanh_gating", None) is not None
        and getattr(cfg, "scoring_func", None) == "softmax"
        and getattr(cfg, "norm_topk_prob", False) is True
        and getattr(cfg, "n_group", 1) == 1
        and getattr(cfg, "topk_group", 1) == 1
        and getattr(cfg, "topk_method", None) is None
        and rope.get("rope_type") == "yarn"
        and rope.get("attention_factor") == 1.0
        and _yarn_mscale(factor, float(rope.get("mscale", 1)))
        == _yarn_mscale(factor, float(rope.get("mscale_all_dim", 0)))
        and getattr(cfg, "fp8_kv_cache", False) is True
    )


def _nvfp4_mixed_precision(cfg: Any) -> bool:
    """The NVFP4 export: ModelOpt MIXED_PRECISION, 16-input NVFP4 blocks, and exactly the
    per-module map the forward is written against."""
    quant = getattr(cfg, "quantization_config", None) or {}
    return (
        quant.get("quant_method") == "modelopt"
        and quant.get("quant_algo") == "MIXED_PRECISION"
        and quant.get("group_size") == 16
        and quant.get("quantized_layers") == _MIXED_PRECISION_LAYERS
    )


def serves_images(pretrained_config: Any) -> bool:
    """Whether this config's target encodes images: an NVFP4 export with a vision encoder."""
    return getattr(pretrained_config, "vision_config", None) is not None and (
        _nvfp4_mixed_precision(pretrained_config)
    )


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The target is imported where it is returned, so declining costs no import of modeling code.
    """
    cfg = pretrained_config
    # The targets encode images with exactly this vision tower.
    vision = cfg.vision_config if serves_images(cfg) else None
    if vision is not None:
        expected_vision = {
            "num_hidden_layers": 30,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "intermediate_size": 6144,
            "num_channels": 3,
            "image_size": 1540,
            "patch_size": 14,
            "rope_theta": 10000.0,
            "spatial_merge_size": 2,
            "mm_projector_id": "patch_merge",
            "adapter_bias": False,
            "add_pre_mm_projector_layer_norm": True,
            "image_token_id": 10,
            "image_break_token_id": 12,
            "image_end_token_id": 13,
        }
        if any(getattr(vision, key, None) != value for key, value in expected_vision.items()):
            raise NoTarget("unsupported Mistral-Large-4 vision encoder or projector geometry")
        if getattr(vision, "rms_norm_eps", 1e-5) != 1e-5:
            raise NoTarget("Mistral-Large-4 vision RMSNorm requires eps=1e-5")
    # The checkpoint is recognized by its config shape: the trunk, the MLA head split, the
    # MoE geometry and the indexer, every width the declared weights are written against.
    shape = (
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_attention_heads,
        getattr(cfg, "q_lora_rank", None),
        getattr(cfg, "kv_lora_rank", None),
        getattr(cfg, "qk_nope_head_dim", None),
        getattr(cfg, "qk_rope_head_dim", None),
        getattr(cfg, "v_head_dim", None),
        getattr(cfg, "n_routed_experts", None),
        getattr(cfg, "num_experts_per_tok", None),
        getattr(cfg, "n_shared_experts", None),
        getattr(cfg, "moe_intermediate_size", None),
        getattr(cfg, "intermediate_size", None),
        getattr(cfg, "first_k_dense_replace", None),
        getattr(cfg, "index_n_heads", None),
        getattr(cfg, "index_head_dim", None),
        getattr(cfg, "index_topk", None),
        cfg.vocab_size,
    )
    # The targets serve 4 ranks on one node or 8 ranks over two nodes of four: every per-rank
    # width -- heads, expert windows, shard widths -- is read off the mapping, so the two sizes
    # are one weight layout and one forward.
    nvfp4_world = mapping.world_size in (4, 8)
    # tep4: the attention heads, the dense and shared MLPs and lm_head sharded over every rank,
    # the routed experts too, by expert (32 per rank at 4 ranks, 16 at 8). Attention DP would
    # replicate attention: another weight layout. At 8 ranks this is the tep8 deployment;
    # tensor_parallel_size 8 alone shards every expert eight ways instead (moe_tp 8, moe_ep 1),
    # the tp4 branch's layout.
    tep4 = (
        mapping.tp_size == mapping.world_size
        and mapping.pp_size == 1
        and mapping.moe_ep_size == mapping.world_size
        and mapping.moe_tp_size == 1
        and not mapping.enable_attention_dp
    )
    # dep4: attention data parallel over the ranks -- every rank holds the whole attention, the
    # dense and shared MLPs and the full-vocabulary lm_head and serves its own requests -- and
    # the routed experts split by expert over every rank (32 per rank at 4 ranks, 16 at 8),
    # gathered over the ranks' tokens. lm_head tensor parallelism under attention DP would make
    # the shell's lm_head vocab-parallel: another weight layout. At 8 ranks this is the dep8
    # deployment.
    dep4 = (
        mapping.tp_size == mapping.world_size
        and mapping.pp_size == 1
        and mapping.moe_ep_size == mapping.world_size
        and mapping.moe_tp_size == 1
        and mapping.enable_attention_dp
        and not mapping.enable_lm_head_tp_in_adp
    )
    # tp4: tensor parallel throughout -- the attention heads, the dense and shared MLPs and
    # lm_head sharded over every rank as at tep4 -- and the routed experts tensor parallel too:
    # every rank holds all 128 experts, each cut to its 1/world_size of the intermediate width
    # (1536 at 4 ranks, 768 at 8), and the ranks' partial sums close on the same all-reduce. At 8
    # ranks this is the tp8 deployment, and what tensor_parallel_size 8 alone resolves to.
    # Expert parallelism, or a mixed split of the experts, would be another weight layout;
    # attention DP another attention.
    tp4 = (
        mapping.tp_size == mapping.world_size
        and mapping.pp_size == 1
        and mapping.moe_tp_size == mapping.world_size
        and mapping.moe_ep_size == 1
        and not mapping.enable_attention_dp
    )
    layers = _per_layer_structure(cfg)
    math_ok = _math_switches(cfg)
    nvfp4 = _nvfp4_mixed_precision(cfg)
    draft = _native_dspark(spec_config)
    # The NVFP4 checkpoint carries a BF16 DSpark block; the target and draft
    # follow the same parallel topology, on 4 ranks or 8 over two nodes.
    if (
        compute_capability == (10, 3)
        and shape == _SHAPE
        and nvfp4_world
        and tep4
        and draft
        and layers
        and math_ok
        and nvfp4
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .mistral_large_4_nvfp4__sm_103__tep4.modeling import ModelingV2MistralLarge4Nvfp4Sm103Tep4
        # isort: on
        # fmt: on

        return ModelingV2MistralLarge4Nvfp4Sm103Tep4

    # DEP replicates the attention and dense draft for each rank's requests.
    if (
        compute_capability == (10, 3)
        and shape == _SHAPE
        and nvfp4_world
        and dep4
        and draft
        and layers
        and math_ok
        and nvfp4
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .mistral_large_4_nvfp4__sm_103__dep4.modeling import ModelingV2MistralLarge4Nvfp4Sm103Dep4
        # isort: on
        # fmt: on

        return ModelingV2MistralLarge4Nvfp4Sm103Dep4

    # TP shards the draft attention and dense MLP like the target's dense layer.
    if (
        compute_capability == (10, 3)
        and shape == _SHAPE
        and nvfp4_world
        and tp4
        and draft
        and layers
        and math_ok
        and nvfp4
    ):
        # One line on purpose: the claims suite reads the routed targets off
        # `from .<target>.modeling import <Class>` lines.
        # fmt: off
        # isort: off
        from .mistral_large_4_nvfp4__sm_103__tp4.modeling import ModelingV2MistralLarge4Nvfp4Sm103Tp4
        # isort: on
        # fmt: on

        return ModelingV2MistralLarge4Nvfp4Sm103Tp4

    if nvfp4:
        quant = "ModelOpt MIXED_PRECISION (NVFP4 experts, FP8_PB_WO linears)"
    else:
        quant = "differs"
    spec = "off" if spec_config is None else spec_config.spec_dec_mode.name
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, "
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"moe_ep={mapping.moe_ep_size} moe_tp={mapping.moe_tp_size} "
        f"attention_dp={mapping.enable_attention_dp} "
        f"lm_head_tp_in_adp={mapping.enable_lm_head_tp_in_adp}, speculative decoding {spec}, "
        f"per-layer NoPE/KV-sharing/indexer-sharing pattern "
        f"{'as written' if layers else 'differs'}, math switches "
        f"(SSMax, LeRoPE, soft-cap, softmax routing, YaRN magnitude 1, fp8 KV) "
        f"{'as written' if math_ok else 'differ'}, quantization {quant}; the mistral_large4 "
        "targets are Mistral-Large-4-1T-A52B-NVFP4 (ModelOpt MIXED_PRECISION: NVFP4 routed "
        "experts, FP8_PB_WO linears, group 16, its 475-module map) on sm103 at tep4, at dep4 "
        "and at tp4; each target serves 4 ranks on one node or 8 ranks over two nodes of "
        "four (world_size=tp=4 or 8, pp=1); tep4 is tensor-parallel attention, expert-parallel "
        "MoE, attention DP off (world_size=4 tp=4 pp=1 moe_ep=4 moe_tp=1, or world_size=8 tp=8 "
        "pp=1 moe_ep=8 moe_tp=1: tep8); dep4 is attention data parallel with lm_head TP off, "
        "expert-parallel MoE (world_size=4 tp=4 pp=1 moe_ep=4 moe_tp=1, or world_size=8 tp=8 "
        "pp=1 moe_ep=8 moe_tp=1: dep8); tp4 is tensor-parallel attention, tensor-parallel MoE "
        "(every expert on every rank, cut by width), attention DP off (world_size=4 tp=4 pp=1 "
        "moe_ep=1 moe_tp=4, or world_size=8 tp=8 pp=1 moe_ep=1 moe_tp=8: tp8). "
        "Speculative decoding is off or native DSpark: fixed block/depth 3, layers "
        "1,18,36,54,71, three dense draft layers, window 2048, eagle norm, "
        "position-specific masks and rank-512 RNN, AUTO/VANILLA attention with "
        "separate draft KV"
    )
