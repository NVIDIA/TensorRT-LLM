# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a KimiK3ForConditionalGeneration config lands. Read this file and you know.

Kimi K3 NVFP4 on Rubin (sm107) in the disaggregated AgentX deployments:

* context workers, one rank per process (DWDP context servers)   -> ``__tp1``
* generation over Helix context parallelism on 8 ranks           -> ``__helix8``
* generation over Helix context parallelism on 16 ranks,
  experts split MoE TP4 x EP4                                    -> ``__helix16_moetp4ep4``
* generation with attention DP over 16 ranks                     -> ``__dep16``

Every target runs the package-internal Rubin K3 model in ``k3_rubin/`` (the text model,
MLA, KDA, DSpark drafter / worker and K3 kernels of the K3 Rubin agentic optimizations),
with the optimizations of its profile in ``k3_rubin/features.py``.
``modeling_v2="off"`` keeps the built-in K3 class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# (layers, hidden, routed experts, experts per token, attention heads, q_lora_rank, kv_lora_rank, vocab).
# Shape-sniffing also matches fine-tunes of the same shape; the accuracy gate pins identity.
_SHAPE = (93, 7168, 896, 16, 96, 1536, 512, 163840)
_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_experts",
    "num_experts_per_token",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "vocab_size",
)


def _parallel(mapping) -> str:
    """The deployment this mapping is, or a description of it when it is none of them."""
    helix = mapping.cp_size > 1 and mapping.has_cp_helix()
    if mapping.pp_size != 1:
        return f"pp{mapping.pp_size}"
    if mapping.world_size == 1:
        return "tp1"
    if helix and not mapping.enable_attention_dp and mapping.tp_size == 1:
        if mapping.cp_size == 8:
            return "helix8"
        if mapping.cp_size == 16 and (mapping.moe_tp_size, mapping.moe_ep_size) == (4, 4):
            return "helix16_moetp4ep4"
    if (
        mapping.enable_attention_dp
        and mapping.world_size == 16
        and mapping.tp_size == 16
        and mapping.cp_size == 1
    ):
        return "dep16"
    return (
        f"world_size={mapping.world_size} tp={mapping.tp_size} cp={mapping.cp_size} "
        f"helix={helix} attention_dp={mapping.enable_attention_dp} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size}"
    )


@routes("KimiK3ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits."""
    pretrained_config = model_config.pretrained_config
    text = getattr(pretrained_config, "text_config", None)
    if text is None:
        raise NoTarget("Kimi K3 targets need the composite checkpoint's text_config")
    shape = tuple(getattr(text, name, None) for name in _SHAPE_FIELDS)
    if shape != _SHAPE:
        raise NoTarget(f"text_config shape {shape} is not Kimi K3's {_SHAPE}")
    if compute_capability != (10, 7):
        raise NoTarget(
            f"the Kimi K3 targets are written for sm107 (Rubin), got {compute_capability}"
        )
    quant = getattr(pretrained_config, "quantization_config", None) or {}
    experts = [
        value
        for name, value in (quant.get("quantized_layers", {}) or {}).items()
        if name.endswith(".experts")
    ]
    if not experts or any(value.get("quant_algo") != "NVFP4" for value in experts):
        raise NoTarget("the Kimi K3 targets need the NVFP4 checkpoint (NVFP4 routed experts)")
    spec_config = model_config.spec_config
    if spec_config is not None and not spec_config.spec_dec_mode.is_dspark():
        raise NoTarget(
            f"the Kimi K3 targets run DSpark or no speculative decoding, got "
            f"{spec_config.spec_dec_mode.name}"
        )
    kv_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    if kv_algo not in (QuantAlgo.NVFP4, QuantAlgo.FP8):
        raise NoTarget(f"the Kimi K3 targets run an NVFP4 or FP8 KV cache, got {kv_algo}")

    mapping = model_config.mapping
    parallel = _parallel(mapping)
    logger.info(
        f"modeling_v2 kimi_k3_vl: parallel={parallel} world_size={mapping.world_size} "
        f"rank={mapping.rank} tp={mapping.tp_size} cp={mapping.cp_size} "
        f"attention_dp={mapping.enable_attention_dp} moe_tp={mapping.moe_tp_size} "
        f"moe_ep={mapping.moe_ep_size} dwdp_size={getattr(mapping, 'dwdp_size', None)} "
        f"kv={kv_algo} spec={spec_config.spec_dec_mode.name if spec_config else None}"
    )
    from .k3_rubin import features

    if parallel in features.PROFILES:
        # Before the target's import: the k3_rubin modules read the profile when imported.
        features.select(parallel)
    # One line per import on purpose: the claims suite reads the routed targets off
    # `from .<target>.modeling import <Class>` lines.
    # fmt: off
    # isort: off
    if parallel == "tp1":
        from .kimi_k3_nvfp4__sm_107__tp1.modeling import ModelingV2KimiK3Nvfp4Sm107Tp1
        return ModelingV2KimiK3Nvfp4Sm107Tp1
    if parallel == "helix8":
        from .kimi_k3_nvfp4__sm_107__helix8.modeling import ModelingV2KimiK3Nvfp4Sm107Helix8
        return ModelingV2KimiK3Nvfp4Sm107Helix8
    if parallel == "helix16_moetp4ep4":
        from .kimi_k3_nvfp4__sm_107__helix16_moetp4ep4.modeling import ModelingV2KimiK3Nvfp4Sm107Helix16Moetp4ep4
        return ModelingV2KimiK3Nvfp4Sm107Helix16Moetp4ep4
    if parallel == "dep16":
        from .kimi_k3_nvfp4__sm_107__dep16.modeling import ModelingV2KimiK3Nvfp4Sm107Dep16
        return ModelingV2KimiK3Nvfp4Sm107Dep16
    # isort: on
    # fmt: on
    raise NoTarget(
        f"{parallel}: the Kimi K3 targets are one rank per process (DWDP context workers), Helix "
        "over 8 ranks, Helix over 16 ranks with MoE TP4 x EP4, or attention DP over 16 ranks"
    )
