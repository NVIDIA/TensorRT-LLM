# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Glm5NextForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in GLM-5.3-Flash implementation; in ``require`` it raises that
error.

Two targets share one checkpoint, GPU and parallel coverage and differ in how
speculative decoding runs:

* ``glm_5_3_flash_nvfp4__sm_100__default_mtp_fused`` runs one-model MTP-Eagle
  inside its own forward. It is chosen whenever it supports the deployment.
* ``glm_5_3_flash_nvfp4__sm_100__default`` runs on the stock speculative
  framework and takes every other deployment it supports, including no
  speculative decoding.

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig, SADecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, routes

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
        and bool(mapping.enable_attention_dp)
        and bool(getattr(mapping, "enable_lm_head_tp_in_adp", False))
    )


def _fused_refused(
    model_config: ModelConfig, llm_args: TorchLlmArgs, spec_config: Any
) -> list[str]:
    """What the MTP-fused target does not compute for this deployment; empty when it is admitted.

    The target runs the checkpoint's MTP layer as a linear one-model MTP-Eagle chain with strict acceptance (greedy or
    plain sampling) inside its own forward. Its constructor refuses the same speculative features
    (``_check_speculation``); routing declines them first, so the stock target takes the deployment instead.
    """
    if spec_config is None:
        return ["no speculative decoding"]
    if not isinstance(spec_config, MTPDecodingConfig):
        return [f"{type(spec_config).__name__} (only MTP)"]
    refused = {
        "vanilla MTP (only MTP-Eagle)": not spec_config.spec_dec_mode.is_mtp_eagle_one_model(),
        "a dynamic draft tree": bool(getattr(spec_config, "use_dynamic_tree", False)),
        "draft trees": spec_config.max_total_draft_tokens not in (None, spec_config.max_draft_len),
        "rejection sampling": bool(getattr(spec_config, "use_rejection_sampling", False)),
        "relaxed acceptance": bool(
            getattr(spec_config, "use_relaxed_acceptance_for_thinking", False)
        ),
        "occurrence penalties": bool(getattr(spec_config, "enable_penalty", False)),
        "sa_config": getattr(spec_config, "sa_config", None) is not None,
        "a separate speculative checkpoint": getattr(spec_config, "speculative_model", None)
        is not None,
        "a draft-length schedule": getattr(spec_config, "draft_len_schedule", None) is not None,
        # The step runs no guided decoder; the engine would reject guided decoding at start-up.
        "guided decoding": getattr(llm_args, "guided_decoding_backend", None) is not None,
        "an LM head split across attention-DP ranks": _mtp_with_split_head(
            model_config, spec_config
        ),
    }
    return [name for name, present in refused.items() if present]


def _stock_refused(model_config: ModelConfig, spec_config: Any) -> list[str]:
    """What the stock-framework target does not run for this deployment; empty when it is admitted.

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
        "a separate speculative checkpoint": getattr(spec_config, "speculative_model", None)
        is not None,
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
            "sm100 (B200)"
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

    stock_refused = _stock_refused(model_config, spec_config)
    if not stock_refused:
        # fmt: off
        # isort: off
        from .glm_5_3_flash_nvfp4__sm_100__default.modeling import ModelingV2Glm53FlashNvfp4Sm100Default
        # isort: on
        # fmt: on
        return ModelingV2Glm53FlashNvfp4Sm100Default

    spec = spec_config.spec_dec_mode.name
    raise NoTarget(
        f"{deployment}, speculative decoding {spec} with {', '.join(stock_refused)}; the glm5_next "
        "targets run one-model MTP (vanilla or MTP-Eagle, from the checkpoint's MTP layer), "
        "suffix-automaton drafting or no speculative decoding"
    )
