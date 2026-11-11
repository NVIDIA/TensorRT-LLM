# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Glm5NextForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in GLM-5.3-Flash implementation; in ``require`` it raises that
error.

Two NVFP4 targets share one checkpoint, GPU and parallel coverage and differ in how
speculative decoding runs:

* ``glm_5_3_flash_nvfp4__sm_100__default_mtp_fused`` runs one-model MTP-Eagle
  inside its own forward. It is chosen whenever it supports the deployment.
* ``glm_5_3_flash_nvfp4__sm_100__default`` is the split target: the engine's
  speculative worker runs after its forward. It takes every other deployment it
  supports, including no speculative decoding.

The checkpoint's quantization picks the targets that can apply: the two above
serve the NVFP4 checkpoint, and ``glm_5_3_flash_fp8__sm_10x__tp4`` serves the
block-FP8 (``FP8_BLOCK_SCALES``) checkpoint on sm_100 / sm_103 with tensor
parallelism 4 (see ``_check_fp8``).

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, SADecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture the targets are written for: B200 (sm100). They have
# been run on nothing else, so routing sends nothing else to them.
_SM = (10, 0)

# GLM-5.3-Flash's config shape, read from ``text_config``: (num_hidden_layers,
# hidden_size, n_routed_experts, q_lora_rank, linear_num_heads). See the note in
# the gpt_oss routing module for what shape-sniffing does and does not pin.
_SHAPE = (45, 4096, 288, 1536, 64)
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "n_routed_experts",
    "q_lora_rank",
    "linear_num_heads",
)

# The block-FP8 target's deployment: (layers, hidden, routed experts, q_lora_rank) of the
# text config on sm_100 (B200, GB200) and sm_103 (B300, GB300).
_FP8_SHAPE = (45, 4096, 288, 1536)
_FP8_SMS = frozenset({(10, 0), (10, 3)})

# Folds the shared expert into the routed experts' weights when set to "1".
_SHARED_EXPERT_FUSION_ENV = "TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION"


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _quant_algo(model_config: ModelConfig) -> Optional[QuantAlgo]:
    """The checkpoint's weight quantization, as the engine read it from the checkpoint."""
    quant_config = getattr(model_config, "quant_config", None)
    return None if quant_config is None else quant_config.quant_algo


def _mtp_with_split_head(model_config: ModelConfig, spec_config: Any) -> bool:
    """MTP under attention DP with the LM head split across the attention-DP ranks.

    Neither target's MTP drafter computes its draft logits over such a head.
    """
    mapping = model_config.mapping
    return (
        isinstance(spec_config, MTPDecodingConfig)
        and mapping.enable_attention_dp
        and mapping.enable_lm_head_tp_in_adp
    )


def fused_speculation_refused(spec_config: Any) -> list[str]:
    """The speculative features the MTP-fused target does not compute; empty when it computes this config.

    The target runs the checkpoint's MTP layer as a linear one-model MTP-Eagle chain with strict acceptance (greedy or
    plain sampling) inside its own forward. ``route`` declines these features so the split target takes the
    deployment, and the fused target's constructor refuses them against direct construction.
    """
    if spec_config is None:
        return ["no speculative decoding"]
    if not isinstance(spec_config, MTPDecodingConfig):
        return [f"{type(spec_config).__name__} (only MTP)"]
    refused = {
        "vanilla MTP (only MTP-Eagle)": not spec_config.spec_dec_mode.is_mtp_eagle_one_model(),
        "a dynamic draft tree": spec_config.use_dynamic_tree,
        "draft trees": spec_config.max_total_draft_tokens not in (None, spec_config.max_draft_len),
        "rejection sampling": spec_config.use_rejection_sampling,
        "relaxed acceptance": spec_config.use_relaxed_acceptance_for_thinking,
        "occurrence penalties": spec_config.enable_penalty,
        "sa_config": spec_config.sa_config is not None,
        "a separate speculative checkpoint": spec_config.speculative_model is not None,
        "a draft-length schedule": spec_config.draft_len_schedule is not None,
    }
    return [name for name, present in refused.items() if present]


def _fused_refused(
    model_config: ModelConfig, llm_args: TorchLlmArgs, spec_config: Any
) -> list[str]:
    """What the MTP-fused target does not run for this deployment; empty when it is admitted.

    Its speculative features (:func:`fused_speculation_refused`), plus the engine features the step does not run.
    """
    refused = fused_speculation_refused(spec_config)
    # The step runs no guided decoder; the engine would reject guided decoding at start-up.
    if llm_args.guided_decoding_backend is not None:
        refused.append("guided decoding")
    if _mtp_with_split_head(model_config, spec_config):
        refused.append("an LM head split across attention-DP ranks")
    return refused


def _split_refused(model_config: ModelConfig, spec_config: Any) -> list[str]:
    """What the split target does not run for this deployment; empty when it is admitted.

    The engine's speculative worker drives the target. A mode is admitted when its worker commits the KDA recurrent
    state for the accepted tokens and it reads no hidden states captured from the decoder layers, which this model
    does not capture: one-model MTP (vanilla or MTP-Eagle, from the checkpoint's own MTP layer) and suffix-automaton
    drafting.
    """
    if spec_config is None:
        return []
    if isinstance(spec_config, SADecodingConfig):
        return []
    if not isinstance(spec_config, MTPDecodingConfig):
        return [f"{type(spec_config).__name__} (only MTP or SA)"]
    refused = {
        "a separate speculative checkpoint": spec_config.speculative_model is not None,
        "an LM head split across attention-DP ranks": _mtp_with_split_head(
            model_config, spec_config
        ),
    }
    return [name for name, present in refused.items() if present]


@routes("Glm5NextForConditionalGeneration")
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
    text_config = getattr(pretrained_config, "text_config", None)
    shape = (
        None if text_config is None else tuple(_field(text_config, name) for name in _SHAPE_FIELDS)
    )
    quant_algo = _quant_algo(model_config)
    if quant_algo == QuantAlgo.FP8_BLOCK_SCALES:
        _check_fp8(model_config, llm_args, compute_capability)
        # One line on purpose: the native loader and CI test selection read the routed
        # targets off `from .<target>.modeling import <Class>` and `return <Class>` here.
        # fmt: off
        # isort: off
        from .glm_5_3_flash_fp8__sm_10x__tp4.modeling import ModelingV2Glm53FlashFp8Sm10xTp4
        # isort: on
        # fmt: on
        return ModelingV2Glm53FlashFp8Sm10xTp4
    # Both targets compute their weight shapes from the mapping, as the built-in
    # model does; neither runs pipeline or context parallelism.
    single_stage = mapping.pp_size == 1 and mapping.cp_size == 1
    deployment = (
        f"compute capability {compute_capability}, text_config shape {shape}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, "
        f"pp={mapping.pp_size} cp={mapping.cp_size}"
    )
    if not (compute_capability == _SM and shape == _SHAPE and quant_algo == QuantAlgo.NVFP4):
        raise NoTarget(
            f"{deployment}; the glm5_next targets are the NVFP4 GLM-5.3-Flash checkpoint on "
            "sm100 (B200) and the block-FP8 (FP8_BLOCK_SCALES) one on sm100 / sm103"
        )
    if not single_stage:
        raise NoTarget(
            f"{deployment}; the glm5_next targets run without pipeline or context parallelism"
        )

    if not _fused_refused(model_config, llm_args, spec_config):
        from .glm_5_3_flash_nvfp4__sm_100__default_mtp_fused.modeling import (
            ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused,
        )

        return ModelingV2Glm53FlashNvfp4Sm100DefaultMtpFused

    split_refused = _split_refused(model_config, spec_config)
    if not split_refused:
        # fmt: off
        # isort: off
        from .glm_5_3_flash_nvfp4__sm_100__default.modeling import ModelingV2Glm53FlashNvfp4Sm100Default
        # isort: on
        # fmt: on
        return ModelingV2Glm53FlashNvfp4Sm100Default

    spec = spec_config.spec_dec_mode.name
    raise NoTarget(
        f"{deployment}, speculative decoding {spec} with {', '.join(split_refused)}; the glm5_next "
        "targets run one-model MTP (vanilla or MTP-Eagle, from the checkpoint's MTP layer), "
        "suffix-automaton drafting or no speculative decoding"
    )


def _check_fp8(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> None:
    """Raise ``NoTarget`` unless the block-FP8 target fits this deployment."""
    from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_glm5_next_text_config

    mapping = model_config.mapping
    spec_config = model_config.spec_config
    text = unwrap_glm5_next_text_config(model_config.pretrained_config)
    shape = tuple(
        _field(text, name)
        for name in ("num_hidden_layers", "hidden_size", "n_routed_experts", "q_lora_rank")
    )
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)

    # tp4: tensor-parallel attention over the 4 ranks of one node, the routed experts
    # split by intermediate (moe_tp 4). A weight-layout question, which is why it
    # selects a target rather than a runtime branch.
    tp4 = (
        mapping.world_size == 4
        and mapping.tp_size == 4
        and mapping.moe_tp_size == 4
        and mapping.moe_ep_size == 1
        and not mapping.enable_attention_dp
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and mapping.gpus_per_node >= 4
    )
    # The checkpoint's MTP layer driven by the one-model MTP-Eagle loop at three linear
    # drafts, or no speculation. The target builds the drafter from the checkpoint's own
    # MTP layer, so a separate draft checkpoint, a draft tree, an MXFP8 draft head and a
    # draft MoE backend, none of which it builds, are declined rather than ignored.
    mtp3_or_none = spec_config is None or (
        isinstance(spec_config, MTPDecodingConfig)
        and spec_config.spec_dec_mode.is_mtp_eagle_one_model()
        and spec_config.max_draft_len == 3
        and spec_config.max_total_draft_tokens == spec_config.max_draft_len
        and not spec_config.use_dynamic_tree
        and spec_config.speculative_model is None
        and spec_config.draft_lm_head_dtype == "auto"
        and spec_config.moe_backend is None
    )
    # The target's MoE kernels read the routed experts in the checkpoint's layout, which
    # only the TRTLLM backend keeps (AUTO resolves to it here). It does no load balancing,
    # DWDP or shared-expert fusion, so those are declined rather than ignored.
    moe = llm_args.moe_config
    moe_as_built = (
        moe.backend in ("AUTO", "TRTLLM")
        and moe.load_balancer is None
        and llm_args.dwdp_config is None
        and os.environ.get(_SHARED_EXPERT_FUSION_ENV) != "1"
    )

    # GLM-5.3-Flash, block-FP8 release, text only, on sm_100 / sm_103, tp4, with an FP8
    # latent cache: the sparse attention reads FP8 latent rows.
    if (
        compute_capability in _FP8_SMS
        and shape == _FP8_SHAPE
        and tp4
        and mtp3_or_none
        and moe_as_built
        and kv_cache_quant_algo == QuantAlgo.FP8
        and llm_args.disable_mm_encoder
    ):
        return
    draft = ""
    if isinstance(spec_config, MTPDecodingConfig):
        draft = (
            f" max_draft_len={spec_config.max_draft_len}"
            f" max_total_draft_tokens={spec_config.max_total_draft_tokens}"
            f" use_dynamic_tree={spec_config.use_dynamic_tree}"
            f" speculative_model={spec_config.speculative_model}"
            f" draft_lm_head_dtype={spec_config.draft_lm_head_dtype}"
            f" draft moe_backend={spec_config.moe_backend}"
        )
    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, quantization "
        f"{QuantAlgo.FP8_BLOCK_SCALES}, world_size={mapping.world_size} tp={mapping.tp_size} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"attention_dp={mapping.enable_attention_dp} pp={mapping.pp_size} cp={mapping.cp_size}, "
        f"speculative decoding {spec_config.spec_dec_mode.name if spec_config else 'off'}{draft}, "
        f"moe backend={moe.backend} load_balancer={moe.load_balancer} "
        f"dwdp_config={llm_args.dwdp_config} "
        f"{_SHARED_EXPERT_FUSION_ENV}={os.environ.get(_SHARED_EXPERT_FUSION_ENV)}, KV cache "
        f"{kv_cache_quant_algo or 'bf16'}, disable_mm_encoder={llm_args.disable_mm_encoder}; "
        "the block-FP8 glm5_next target is GLM-5.3-Flash on sm100 / sm103, tp4 (4 ranks of "
        "one node, tensor-parallel attention, experts split by intermediate), with one-model "
        "MTP-Eagle at max_draft_len=3 (linear drafts from the checkpoint's MTP layer) or no "
        "speculative decoding, the AUTO or TRTLLM MoE backend without load balancing, DWDP or "
        "shared-expert fusion, an FP8 latent cache (kv_cache_config.dtype fp8) and the vision "
        "tower disabled (disable_mm_encoder)"
    )
