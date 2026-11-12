# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam family routing: which ModelingV2 target serves a ``BeamForCausalLM`` deployment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from ..._router_index import NoTarget
from ._constants import SM, SUPPORTED_CONFIG, WORLD_SIZE
from .full_model_nvfp4__sm_103__tp4.limits import MAX_NUM_TOKENS

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

#: Proposals per DFlash draft block the target verifies: the drafter's serving block of eight
#: rows (the pending token and seven proposals), the largest verification width the target's
#: attention op is measured at.
DFLASH_MAX_DRAFT_LEN = 7


def _quant_algo(pretrained_config: Any) -> Optional[str]:
    quant = getattr(pretrained_config, "quantization_config", None)
    if isinstance(quant, dict):
        return quant.get("quant_algo")
    return getattr(quant, "quant_algo", None)


def speculation_refusal(spec_config: Any, llm_args: Optional[Any] = None) -> Optional[str]:
    """Why the Beam target cannot run ``spec_config``, or None when it can.

    The one speculative mode is DFlash with a separate drafter checkpoint, the TRTLLM draft
    attention backend (the drafter's context in its paged KV-cache layers), a fixed
    draft length of ``DFLASH_MAX_DRAFT_LEN`` and strict greedy verification (no rejection
    sampling, no draft-length schedule). The engine must run it eagerly: the overlap
    scheduler would hand the target device-corrected positions and KV lengths its host-side
    step check cannot state.
    """
    if spec_config is None:
        return None
    mode = getattr(spec_config, "spec_dec_mode", None)
    if mode is None or not mode.is_dflash():
        return f"speculative decoding {type(spec_config).__name__} is not DFlash"
    reasons = []
    if spec_config.speculative_model is None:
        reasons.append("DFlash needs its drafter checkpoint (speculative_model)")
    if spec_config.attention_backend != "TRTLLM":
        reasons.append(
            f"DFlash attention_backend {spec_config.attention_backend!r}; the drafter's context "
            "lives in paged KV-cache layers, which only the TRTLLM backend reads"
        )
    if spec_config.max_draft_len != DFLASH_MAX_DRAFT_LEN:
        reasons.append(
            f"DFlash max_draft_len {spec_config.max_draft_len}, not {DFLASH_MAX_DRAFT_LEN}"
        )
    if spec_config.draft_len_schedule:
        reasons.append("a DFlash draft_len_schedule changes the verification width per step")
    if spec_config.use_rejection_sampling:
        reasons.append("DFlash rejection sampling; verification is strict greedy")
    if llm_args is not None and not getattr(llm_args, "disable_overlap_scheduler", True):
        reasons.append("the overlap scheduler with DFlash")
    return "; ".join(reasons) or None


def engine_refusal(llm_args: Optional[Any]) -> Optional[str]:
    """Report all unsupported scheduling options and token budgets for this eager target.

    CUDA graphs and overlap scheduling are unsupported by this target's host-synchronizing
    expert provider. The per-step token budget must lie in ``[1, MAX_NUM_TOKENS]``.
    """
    if llm_args is None:
        return None
    reasons = []
    if getattr(llm_args, "cuda_graph_config", None) is not None:
        reasons.append("CUDA graphs are not supported by this eager Beam target")
    if getattr(llm_args, "torch_compile_config", None) is not None:
        reasons.append("torch_compile_config must be None for this eager Beam target")
    prefill_backend = getattr(llm_args, "prefill_cuda_graph_backend", "disabled")
    if getattr(prefill_backend, "value", prefill_backend) != "disabled":
        reasons.append("prefill_cuda_graph_backend must be disabled for this eager Beam target")
    if not getattr(llm_args, "disable_overlap_scheduler", True):
        reasons.append("the overlap scheduler is not supported by this eager Beam target")
    max_tokens = getattr(llm_args, "max_num_tokens", None)
    if max_tokens is not None and not 1 <= max_tokens <= MAX_NUM_TOKENS:
        reasons.append(f"max_num_tokens {max_tokens} is outside [1, {MAX_NUM_TOKENS}]")
    return "; ".join(reasons) or None


def checkpoint_refusal(pretrained_config: Any, spec_config: Optional[Any]) -> Optional[str]:
    """Why the target's construction would refuse this checkpoint or its drafter, or None.

    Runs the target's own validators on the files its construction reads: the checkpoint's
    ``config.json`` and NVFP4 recipe (``layout.check_checkpoint``) and, with DFlash, the
    drafter's ``config.json`` against them (``layout.check_drafter_config``). Pure file reads;
    nothing is allocated.
    """
    from .full_model_nvfp4__sm_103__tp4 import layout

    try:
        config = layout.check_checkpoint(layout.checkpoint_dir_of(pretrained_config))
        if spec_config is not None:
            drafter = layout.read_config(spec_config.speculative_model)
            # Unset capture layers resolve from the drafter checkpoint, as the runtime
            # resolves them (DFlashDecodingConfig.resolve_from_checkpoint).
            layers = spec_config.target_layer_ids
            if layers is None:
                layers = (drafter.get("dflash_config") or {}).get("target_layer_ids")
            layout.check_drafter_config(drafter, config, layers)
    except ValueError as exc:
        return str(exc)
    return None


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

    The one target is the 52-layer NVFP4-expert checkpoint on four sm_103 ranks: attention
    TP4, MoE TP4/EP1, PP1, a BF16 or FP8 KV cache, no attention data parallelism, without
    speculative decoding or with the DFlash drafter ``speculation_refusal`` admits.
    ``kv_cache_quant_algo`` is the effective engine setting, including runtime overrides; an
    FP8 cache holds the target's and the drafter's K/V with per-layer scales the checkpoint
    loader supplies. No other engine choice in ``model_config`` or ``llm_args`` selects a
    different target.
    """
    from tensorrt_llm.quantization.mode import QuantAlgo

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
    speculation = speculation_refusal(spec_config, llm_args)
    engine = engine_refusal(llm_args)
    refusal = None
    if (
        compute_capability == SM
        and shape
        == tuple(
            SUPPORTED_CONFIG[key] for key in ("num_hidden_layers", "hidden_size", "num_experts")
        )
        and layout == (WORLD_SIZE, WORLD_SIZE, 1, WORLD_SIZE, 1, False)
        and quant_algo == "NVFP4"
        and speculation is None
        and engine is None
        and kv_cache_quant_algo in (None, QuantAlgo.FP8)
    ):
        refusal = checkpoint_refusal(pretrained_config, spec_config)
        if refusal is None:
            from .full_model_nvfp4__sm_103__tp4.modeling import ModelingV2BeamFullModelNvfp4Sm103Tp4

            return ModelingV2BeamFullModelNvfp4Sm103Tp4

    raise NoTarget(
        f"{f'the checkpoint is not one the target computes: {refusal}; ' if refusal else ''}"
        f"compute capability {compute_capability}, config shape {shape}, quantization "
        f"{quant_algo}, layout (world, tp, pp, moe_tp, moe_ep, attention_dp) {layout}, "
        f"speculative decoding {type(spec_config).__name__ if spec_config is not None else 'off'}"
        f"{f' ({speculation})' if speculation else ''}; "
        f"KV cache {kv_cache_quant_algo or 'bf16'}; "
        f"{f'{engine}; ' if engine else ''}"
        "the one Beam target is the 52-layer NVFP4 checkpoint on sm103, attention TP4, MoE "
        f"TP4/EP1, PP1, with a BF16 or FP8 KV cache and without attention DP, at most {MAX_NUM_TOKENS} "
        "tokens per engine step, with speculative decoding "
        f"off or DFlash (TRTLLM draft attention, max_draft_len {DFLASH_MAX_DRAFT_LEN}, strict "
        "greedy verification, no overlap scheduler)"
    )
