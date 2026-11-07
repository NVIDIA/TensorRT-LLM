# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam family routing: which ModelingV2 target serves a ``BeamForCausalLM`` deployment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm.quantization.modelopt_config import (
    canonicalize_quant_algo,
    read_modelopt_quant_config,
)

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def _quant_algo(pretrained_config: PretrainedConfig | None) -> str | None:
    quant = getattr(pretrained_config, "quantization_config", None)
    if isinstance(quant, dict):
        if "quantization" in quant or "quant_method" in quant or "producer" in quant:
            return read_modelopt_quant_config(quant).get("quant_algo")
        return canonicalize_quant_algo(quant.get("quant_algo"))
    return canonicalize_quant_algo(getattr(quant, "quant_algo", None))


def engine_refusal(llm_args: TorchLlmArgs | None) -> str | None:
    """Report engine options this eager target cannot implement."""
    if llm_args is None:
        return None
    reasons = []
    if getattr(llm_args, "cuda_graph_config", None) is not None:
        reasons.append("CUDA graphs are not supported by this eager Beam target")
    if not getattr(llm_args, "disable_overlap_scheduler", True):
        reasons.append("the overlap scheduler is not supported by this eager Beam target")
    load_format = getattr(llm_args, "load_format", "auto")
    if str(getattr(load_format, "name", load_format)).lower() != "auto":
        reasons.append("load_format must be auto")
    if getattr(llm_args, "max_beam_width", 1) not in (None, 1):
        reasons.append("beam search is not supported; max_beam_width must be 1")
    if getattr(llm_args, "attn_backend", "TRTLLM") != "TRTLLM":
        reasons.append("attn_backend must be TRTLLM")
    for field in ("lora_config", "kv_connector_config", "cache_transceiver_config"):
        if getattr(llm_args, field, None) is not None:
            reasons.append(f"{field} is not supported by this Beam target")
    if getattr(llm_args, "enable_return_routed_experts", False):
        reasons.append("returning routed experts is not supported")
    max_tokens = getattr(llm_args, "max_num_tokens", None)
    if max_tokens is not None:
        from .full_model_nvfp4__sm_103__tp4.model import MAX_NUM_TOKENS

        if not 1 <= max_tokens <= MAX_NUM_TOKENS:
            reasons.append(f"max_num_tokens {max_tokens} is outside [1, {MAX_NUM_TOKENS}]")
    return "; ".join(reasons) or None


@routes("BeamForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    The one target is the 52-layer NVFP4-expert checkpoint on four sm_103 ranks: attention
    TP4, MoE TP4/EP1, PP1, BF16 KV cache, no attention data parallelism, no speculative
    decoding. ``kv_cache_quant_algo`` is the effective engine setting, including runtime
    overrides. No additional engine choice in ``model_config`` or ``llm_args`` selects a
    different target.
    """
    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    shape = (
        pretrained_config.num_hidden_layers,
        pretrained_config.hidden_size,
        getattr(pretrained_config, "num_experts", None),
    )
    layout = (
        mapping.world_size,
        mapping.tp_size,
        mapping.pp_size,
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        bool(mapping.enable_attention_dp),
    )
    refusal = engine_refusal(llm_args)
    try:
        quant_algo = _quant_algo(pretrained_config)
    except ValueError as exc:
        raise NoTarget(str(exc)) from exc
    if quant_algo is None:
        algo = getattr(model_config.quant_config, "quant_algo", None)
        quant_algo = getattr(algo, "value", algo)
    if (
        compute_capability == (10, 3)
        and shape == (52, 5120, 256)
        and layout == (4, 4, 1, 4, 1, False)
        and quant_algo == "NVFP4"
        and spec_config is None
        and kv_cache_quant_algo is None
        and refusal is None
    ):
        from .full_model_nvfp4__sm_103__tp4.weights import BeamNvfp4CheckpointLoader

        loader = getattr(llm_args, "checkpoint_loader", None)
        registered_format = (
            loader is None and getattr(llm_args, "checkpoint_format", None) == "beam_nvfp4"
        )
        if not registered_format and not isinstance(loader, BeamNvfp4CheckpointLoader):
            raise NoTarget(
                "this Beam target requires checkpoint_format='beam_nvfp4' or "
                "checkpoint_loader=BeamNvfp4CheckpointLoader(); "
                "the default HF checkpoint loader cannot populate its packed weights"
            )
        from .full_model_nvfp4__sm_103__tp4.layout import check_checkpoint, checkpoint_dir_of

        try:
            check_checkpoint(checkpoint_dir_of(pretrained_config))
        except ValueError as exc:
            raise NoTarget(str(exc)) from exc
        from .full_model_nvfp4__sm_103__tp4.modeling import ModelingV2BeamFullModelNvfp4Sm103Tp4

        return ModelingV2BeamFullModelNvfp4Sm103Tp4

    raise NoTarget(
        f"compute capability {compute_capability}, config shape {shape}, quantization "
        f"{quant_algo}, layout (world, tp, pp, moe_tp, moe_ep, attention_dp) {layout}, "
        f"speculative decoding {type(spec_config).__name__ if spec_config is not None else 'off'}; "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; engine: {refusal or 'supported'}; "
        "the one Beam target is the 52-layer NVFP4 checkpoint on sm103, attention TP4, MoE "
        "TP4/EP1, PP1, with BF16 KV cache and without attention DP or speculative decoding"
    )
