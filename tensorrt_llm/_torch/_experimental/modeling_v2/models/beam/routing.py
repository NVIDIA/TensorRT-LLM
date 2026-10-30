# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam family routing: which ModelingV2 target serves a ``BeamForCausalLM`` deployment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def _quant_algo(pretrained_config: Any) -> Optional[str]:
    quant = getattr(pretrained_config, "quantization_config", None)
    if isinstance(quant, dict):
        return quant.get("quant_algo")
    return getattr(quant, "quant_algo", None)


def engine_refusal(llm_args: Optional[Any]) -> Optional[str]:
    """Why the engine configuration cannot execute this eager target, or None.

    The target runs at most ``model.MAX_NUM_TOKENS`` tokens per call (the attention op's
    measured step size), so ``max_num_tokens`` must lie in ``[1, MAX_NUM_TOKENS]``.
    """
    if llm_args is None:
        return None
    if getattr(llm_args, "cuda_graph_config", None) is not None:
        return "this target requires cuda_graph_config=None (eager execution)"
    if not getattr(llm_args, "disable_overlap_scheduler", False):
        return "this target requires disable_overlap_scheduler=True"
    if getattr(llm_args, "max_num_tokens", None) is None:
        return None
    from .full_model_nvfp4__sm_103__tp4.model import MAX_NUM_TOKENS

    if not 1 <= llm_args.max_num_tokens <= MAX_NUM_TOKENS:
        return f"max_num_tokens {llm_args.max_num_tokens} is outside [1, {MAX_NUM_TOKENS}]"
    return None


@routes("BeamForCausalLM")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
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
    quant_algo = _quant_algo(pretrained_config)
    refusal = engine_refusal(llm_args)
    if (
        compute_capability == (10, 3)
        and shape == (52, 5120, 256)
        and layout == (4, 4, 1, 4, 1, False)
        and quant_algo == "NVFP4"
        and spec_config is None
        and kv_cache_quant_algo is None
        and refusal is None
    ):
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
        f"KV cache {kv_cache_quant_algo or 'bf16'}; engine configuration: {refusal or 'supported'}; "
        "the one Beam target is the 52-layer NVFP4 checkpoint on sm103, attention TP4, MoE "
        "TP4/EP1, PP1, with BF16 KV cache and without attention DP or speculative decoding"
    )
