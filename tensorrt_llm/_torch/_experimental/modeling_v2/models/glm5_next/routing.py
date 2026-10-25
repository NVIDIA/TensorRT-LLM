# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Glm5NextForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs, compared against
the deployments the targets below were written for. When none fits it raises ``NoTarget``
stating those facts. In ``auto`` the engine then uses the built-in glm5_next implementation; in
``require`` it raises that error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The GPUs the target runs on: sm_100 (B200, GB200) and sm_103 (B300, GB300), each with its own
# build of the same kernels.
_SMS = frozenset({(10, 0), (10, 3)})

# A parallel layout, as (world_size, attention_dp, moe_tp, moe_ep): tensor-parallel attention
# over 4 ranks with the routed experts expert parallel (tp4 + ep4) or split by intermediate
# (tp4, the LLM API default; the faster one at small batches). Attention DP splits the
# attention weights differently, so it would be another layout of the target, not a knob.
_LAYOUTS = frozenset(
    {
        (4, False, 1, 4),  # tp4 + ep4
        (4, False, 4, 1),  # tp4, experts split by intermediate
    }
)

# Sequences per step the target's step program takes (its head's sampled rows, 4 per sequence
# with MTP).
_MAX_BATCH = 256


def _quant_algo(pretrained_config: Any) -> str:
    """The checkpoint's weight quantization as its config declares it (ModelOpt export)."""
    quant = getattr(pretrained_config, "quantization_config", None) or {}
    if not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else {}
    return str(quant.get("quant_algo", "none")).upper()


@routes("Glm5NextForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import of modeling code.
    """
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    # The checkpoint is recognized by its text config's shape (the vision tower is not part of
    # any target here) and by its weight quantization: an NVFP4 export loads different tensors
    # into different kernels than the BF16 or FP8 release of the same shape.
    text = unwrap_glm5_next_text_config(pretrained_config)
    shape = (
        text.num_hidden_layers,
        text.hidden_size,
        text.n_routed_experts,
        text.q_lora_rank,
    )
    quant_algo = _quant_algo(pretrained_config)
    if compute_capability not in _SMS or shape != (45, 4096, 288, 1536) or quant_algo != "NVFP4":
        raise NoTarget(
            f"compute capability {compute_capability}, config shape {shape}, quantization "
            f"{quant_algo}; the glm5_next target is GLM-5.3-Flash-NVFP4 on sm100 or sm103"
        )

    layout = (
        mapping.world_size,
        bool(mapping.enable_attention_dp),
        mapping.moe_tp_size,
        mapping.moe_ep_size,
    )
    split = (
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} attention_dp={mapping.enable_attention_dp} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size}"
    )
    if (
        layout not in _LAYOUTS
        or mapping.tp_size != 4
        or mapping.pp_size != 1
        or mapping.cp_size != 1
        or mapping.gpus_per_node < 4
    ):
        raise NoTarget(
            f"parallel layout {split}; the glm5_next target is tp4 (4 ranks of one node, "
            "tensor-parallel attention) with the routed experts expert parallel (moe_ep 4) or "
            "split by intermediate (moe_tp 4)"
        )

    # The kernels read and write a BF16 latent cache; the checkpoint declares FP8 without
    # scales, which the target replaces (its KV cache dtype "auto").
    kv_dtype = llm_args.kv_cache_config.dtype
    if kv_dtype not in ("auto", "bfloat16"):
        raise NoTarget(
            f"KV cache dtype {kv_dtype} ({kv_cache_quant_algo}); the glm5_next target runs a "
            "BF16 latent cache (kv_cache_config.dtype auto)"
        )
    if llm_args.max_batch_size is not None and llm_args.max_batch_size > _MAX_BATCH:
        raise NoTarget(
            f"max_batch_size {llm_args.max_batch_size}; the glm5_next target takes up to "
            f"{_MAX_BATCH} sequences per step"
        )
    if llm_args.max_num_tokens is not None and llm_args.max_num_tokens % 4:
        raise NoTarget(
            f"max_num_tokens {llm_args.max_num_tokens}; the glm5_next target splits a "
            "step's rows over its 4 ranks: a multiple of 4"
        )

    # The checkpoint's MTP module driven by the one-model MTP-Eagle loop at three drafts (the
    # KDA verify's per-slot replay records hold three), or no speculation.
    if spec_config is not None and not (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and spec_config.max_draft_len == 3
    ):
        raise NoTarget(
            f"speculative decoding {spec_config.spec_dec_mode.name} with max_draft_len="
            f"{spec_config.max_draft_len}; the glm5_next target implements one-model "
            "MTP-Eagle at max_draft_len=3 or no speculative decoding"
        )

    from .glm_5_3_flash_nvfp4__sm_10x__tp4.modeling import ModelingV2Glm53FlashNvfp4Sm10xTp4

    return ModelingV2Glm53FlashNvfp4Sm10xTp4
