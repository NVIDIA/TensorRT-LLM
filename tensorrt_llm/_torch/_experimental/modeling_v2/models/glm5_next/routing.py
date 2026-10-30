# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a Glm5NextForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in GLM-5.3-Flash implementation; in ``require`` it raises that
error.

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture the target is written for: B200 (sm100). The target
# has been run on nothing else, so routing sends nothing else to it.
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


def _speculation_refused(spec_config: Any) -> list[str]:
    """What in ``spec_config`` the target's in-model MTP step does not compute; empty when it is admitted.

    The target runs the checkpoint's MTP layer as a linear one-model MTP-Eagle chain with strict acceptance (greedy or
    plain sampling) inside its own forward. The target's constructor refuses the same features
    (``_check_speculation``); routing declines them first, so ``auto`` falls back instead of failing mid-build.
    """
    if spec_config is None:
        return []
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
    # The default target computes its weight shapes from the mapping, as the
    # built-in model does; it does not run pipeline or context parallelism.
    single_stage = mapping.pp_size == 1 and mapping.cp_size == 1
    refused = _speculation_refused(spec_config)

    if (
        compute_capability == _SM
        and shape == _SHAPE
        and quant_algo == QuantAlgo.NVFP4
        and single_stage
        and not refused
    ):
        # fmt: off
        # isort: off
        from .glm_5_3_flash_nvfp4__sm_100__default.modeling import ModelingV2Glm53FlashNvfp4Sm100Default
        # isort: on
        # fmt: on
        return ModelingV2Glm53FlashNvfp4Sm100Default

    spec = "off" if spec_config is None else spec_config.spec_dec_mode.name
    if refused:
        spec += f" with {', '.join(refused)}"
    raise NoTarget(
        f"compute capability {compute_capability}, text_config shape {shape}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, "
        f"pp={mapping.pp_size} cp={mapping.cp_size}, speculative decoding {spec}; "
        "the glm5_next target is the NVFP4 GLM-5.3-Flash checkpoint on sm100 (B200), "
        "without pipeline or context parallelism, with one-model MTP-Eagle as a linear chain "
        "(greedy or plain sampling, strict acceptance) or no speculative decoding"
    )
