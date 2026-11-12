# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a KimiK3ForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in Kimi K3 implementation; in ``require`` it raises that error.

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The MXFP4
targets on sm100 are text-only: they load no vision tower and refuse image input.

The NVFP4 checkpoint on sm107 (Rubin) has one target per disaggregated layout
(``_SM107_TARGETS``): the one-GPU DWDP context servers of every deployment, and
the generation servers over Helix CP8 (MoE EP8, or MoE TP2 x EP4) and over
attention DP16. ``_sm107_target`` reads the layout from the mapping, the MoE
backend and the context-parallel config; the layouts exclude each other, so the
disaggregated role is not needed to choose. Their SM107 kernels launch with PDL as a
constant, so a deployment that turns PDL off (``TRTLLM_ENABLE_PDL`` other than ``1``, as the
workers see it) gets ``NoTarget`` rather than launches that ignore the switch.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture these targets are written for. sm is part of a
# target's identity, not a knob: a different SM is a different target. The K3
# decode kernels (the tcgen05 GEMVs, the fused MoE) are built for GB200 only.
_SM = (10, 0)

# The MXFP4 checkpoint's config shape; see the note in the sibling gpt_oss
# routing module for what shape-sniffing does and does not pin.
#
# (num_hidden_layers, hidden_size, num_experts, routed_expert_hidden_size), all
# read from ``text_config``. The NVFP4 requant of the same checkpoint has this
# shape too; the quantization criterion below reads only what ``config.json``
# declares, since ``route`` does not see ``hf_quant_config.json``.
_SHAPE = (93, 7168, 896, 3584)


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _quant_algo(pretrained_config: Any) -> Optional[QuantAlgo]:
    """The quantization the checkpoint's config declares, as the engine reads it.

    The MXFP4 checkpoint declares its quantization inside
    ``text_config.quantization_config`` (compressed-tensors,
    ``mxfp4-pack-quantized``); ``KimiK3Config`` surfaces it and the model
    config reads it as W4A16_MXFP4. A checkpoint without quantization reads as
    none; the experts of either would land in a loader that reads packed MXFP4
    tensors.
    """
    from tensorrt_llm._torch.model_config import ModelConfig

    declared = getattr(pretrained_config, "quantization_config", None)
    if declared is None:
        return None
    try:
        quant_config, _ = ModelConfig.load_hf_quant_config(declared, "TRTLLM")
    except ValueError:
        return None
    return quant_config.quant_algo


def _parallel(m) -> Optional[str]:
    """Name the parallel topology, or None if no target implements it.

    Attention and the dense layers are sharded 16 ways. The routed experts are
    split either 4 ways by tensor and 4 ways by expert, or 16 ways by tensor
    with every expert on every rank. The expert split decides which expert
    weights each rank loads and into which shapes, so it selects a target
    rather than a runtime branch.

    The 16-way tensor split counts only when the configuration asks for it. A
    mapping with no expert split resolves to the same sizes, but the built-in
    Kimi K3 model reads that default as expert parallelism over the 16 ranks
    (`moe_tp_ep_user_specified`), and so does this tree: no target serves it.
    """
    if not (
        m.world_size == 16 and m.tp_size == 16 and m.pp_size == 1 and not m.enable_attention_dp
    ):
        return None
    if m.moe_tp_size == 4 and m.moe_ep_size == 4:
        return "tp16_moetp4ep4"
    if m.moe_tp_size == 16 and m.moe_ep_size == 1 and getattr(m, "moe_tp_ep_user_specified", False):
        return "tp16_moetp16ep1"
    return None


# ---------------------------------------------------------------------------
# Kimi K3 NVFP4 on Rubin (sm107).
# ---------------------------------------------------------------------------

_SM107 = (10, 7)

# (layers, hidden, routed experts, experts per token, attention heads, q_lora_rank,
# kv_lora_rank, vocab), read from ``text_config``.
_SM107_SHAPE = (93, 7168, 896, 16, 96, 1536, 512, 163840)
_SM107_SHAPE_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "num_experts",
    "num_experts_per_token",
    "num_attention_heads",
    "q_lora_rank",
    "kv_lora_rank",
    "vocab_size",
)


# Target directory -> the layout it serves (for the NoTarget message).
_SM107_TARGETS = {
    "kimi_k3_nvfp4__sm_107__tp1": (
        "context servers: one rank (world 1), DWDP, moe_config.backend CUTEDSL"
    ),
    "kimi_k3_nvfp4__sm_107__helix8_moeep8": (
        "generation over Helix CP8 (tp 1, no attention DP, NCCL off for the all-to-all), "
        "MoE EP8, moe_config.backend CUTEDSL_FC12"
    ),
    "kimi_k3_nvfp4__sm_107__helix8_moetp2ep4": (
        "generation over Helix CP8 (tp 1, no attention DP, NCCL off for the all-to-all), "
        "MoE TP2 x EP4, moe_config.backend CUTEDSL_FC12"
    ),
    "kimi_k3_nvfp4__sm_107__dep16": (
        "generation with attention DP over 16 ranks (tp 16), MoE EP16, "
        "moe_config.backend MEGAMOE_CUTEDSL"
    ),
}


def _sm107_target(mapping, llm_args) -> Optional[str]:
    """The SM107 target directory serving this layout, or None."""
    backend = llm_args.moe_config.backend
    moe = (mapping.moe_tp_size, mapping.moe_ep_size)
    helix = mapping.cp_size > 1 and mapping.has_cp_helix()
    if mapping.pp_size != 1:
        return None
    if (
        mapping.world_size == 1
        and mapping.cp_size == 1
        and moe == (1, 1)
        and backend == "CUTEDSL"
        and getattr(llm_args, "dwdp_config", None) is not None
    ):
        return "kimi_k3_nvfp4__sm_107__tp1"
    if (
        helix
        and mapping.cp_size == 8
        and mapping.tp_size == 1
        and not mapping.enable_attention_dp
        and backend == "CUTEDSL_FC12"
        and mapping.cp_config.get("use_nccl_for_alltoall") is False
    ):
        if moe == (1, 8):
            return "kimi_k3_nvfp4__sm_107__helix8_moeep8"
        if moe == (2, 4):
            return "kimi_k3_nvfp4__sm_107__helix8_moetp2ep4"
    if (
        mapping.enable_attention_dp
        and mapping.world_size == 16
        and mapping.tp_size == 16
        and mapping.cp_size == 1
        and moe == (1, 16)
        and backend == "MEGAMOE_CUTEDSL"
    ):
        return "kimi_k3_nvfp4__sm_107__dep16"
    return None


def _worker_env(llm_args: TorchLlmArgs, name: str) -> Optional[str]:
    """``name`` as the workers see it: ``llm_args.env_overrides`` (applied on every rank, MPI
    ranks spawned before it was set included), else this process's environment."""
    overrides = getattr(llm_args, "env_overrides", None) or {}
    if name in overrides:
        return str(overrides[name])
    return os.environ.get(name)


def _check_sm107(model_config: ModelConfig, llm_args: TorchLlmArgs) -> str:
    """Return the sm107 target directory, or raise ``NoTarget`` saying why none fits."""
    pretrained_config = model_config.pretrained_config
    text = getattr(pretrained_config, "text_config", None)
    if text is None:
        raise NoTarget("the sm107 Kimi K3 targets need the composite checkpoint's text_config")
    shape = tuple(getattr(text, name, None) for name in _SM107_SHAPE_FIELDS)
    if shape != _SM107_SHAPE:
        raise NoTarget(f"text_config shape {shape} is not Kimi K3's {_SM107_SHAPE}")
    quant = getattr(pretrained_config, "quantization_config", None) or {}
    experts = [
        value
        for name, value in (quant.get("quantized_layers", {}) or {}).items()
        if name.endswith(".experts")
    ]
    if not experts or any(value.get("quant_algo") != "NVFP4" for value in experts):
        raise NoTarget("the sm107 Kimi K3 targets need the NVFP4 checkpoint (NVFP4 routed experts)")
    spec_config = model_config.spec_config
    if spec_config is None or not spec_config.spec_dec_mode.is_dspark():
        raise NoTarget(
            "the sm107 Kimi K3 targets run DSpark speculative decoding, got "
            f"{spec_config.spec_dec_mode.name if spec_config is not None else 'none'}"
        )
    kv_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    if kv_algo != QuantAlgo.NVFP4:
        raise NoTarget(f"the sm107 Kimi K3 targets run an NVFP4 KV cache, got {kv_algo}")
    pdl = _worker_env(llm_args, "TRTLLM_ENABLE_PDL")
    if pdl not in (None, "1"):
        raise NoTarget(
            f"the sm107 Kimi K3 targets launch their kernels with PDL, got TRTLLM_ENABLE_PDL={pdl!r}"
        )
    mapping = model_config.mapping
    backend = llm_args.moe_config.backend
    target = _sm107_target(mapping, llm_args)
    seen = (
        f"world_size={mapping.world_size} pp={mapping.pp_size} tp={mapping.tp_size} "
        f"cp={mapping.cp_size} helix={mapping.cp_size > 1 and mapping.has_cp_helix()} "
        f"attention_dp={mapping.enable_attention_dp} moe_tp={mapping.moe_tp_size} "
        f"moe_ep={mapping.moe_ep_size} moe_backend={backend} "
        f"cp_nccl_alltoall={mapping.cp_config.get('use_nccl_for_alltoall')} "
        f"dwdp={getattr(llm_args, 'dwdp_config', None) is not None}"
    )
    if target is None:
        layouts = "; ".join(f"{name}: {layout}" for name, layout in _SM107_TARGETS.items())
        raise NoTarget(
            f"{seen}: no sm107 Kimi K3 target serves this layout. The targets: {layouts}"
        )
    logger.info(
        f"modeling_v2 kimi_k3_vl: {target} rank={mapping.rank} {seen} kv={kv_algo} "
        f"spec={spec_config.spec_dec_mode.name}"
    )
    return target


@routes("KimiK3ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    if compute_capability == _SM107:
        target = _check_sm107(model_config, llm_args)
        if target == "kimi_k3_nvfp4__sm_107__tp1":
            from .kimi_k3_nvfp4__sm_107__tp1.modeling import ModelingV2KimiK3Nvfp4Sm107Tp1

            return ModelingV2KimiK3Nvfp4Sm107Tp1
        if target == "kimi_k3_nvfp4__sm_107__helix8_moeep8":
            from .kimi_k3_nvfp4__sm_107__helix8_moeep8.modeling import (
                ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8,
            )

            return ModelingV2KimiK3Nvfp4Sm107Helix8Moeep8
        if target == "kimi_k3_nvfp4__sm_107__helix8_moetp2ep4":
            from .kimi_k3_nvfp4__sm_107__helix8_moetp2ep4.modeling import (
                ModelingV2KimiK3Nvfp4Sm107Helix8Moetp2ep4,
            )

            return ModelingV2KimiK3Nvfp4Sm107Helix8Moetp2ep4
        from .kimi_k3_nvfp4__sm_107__dep16.modeling import ModelingV2KimiK3Nvfp4Sm107Dep16

        return ModelingV2KimiK3Nvfp4Sm107Dep16

    pretrained_config = model_config.pretrained_config
    mapping = model_config.mapping
    text_config = getattr(pretrained_config, "text_config", None)
    shape = (
        None
        if text_config is None
        else tuple(
            _field(text_config, name)
            for name in (
                "num_hidden_layers",
                "hidden_size",
                "num_experts",
                "routed_expert_hidden_size",
            )
        )
    )
    quant_algo = _quant_algo(pretrained_config)
    parallel = _parallel(mapping)
    mxfp4_on_sm100 = (
        compute_capability == _SM and shape == _SHAPE and quant_algo == QuantAlgo.W4A16_MXFP4
    )

    if mxfp4_on_sm100 and parallel == "tp16_moetp4ep4":
        from .kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling import (
            ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4,
        )

        return ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4

    if mxfp4_on_sm100 and parallel == "tp16_moetp16ep1":
        from .kimi_k3_mxfp4__sm_100__tp16_moetp16ep1.modeling import (
            ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1,
        )

        return ModelingV2KimiK3Mxfp4Sm100Tp16Moetp16ep1

    raise NoTarget(
        f"compute capability {compute_capability}, text_config shape {shape}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, parallel "
        f"ws={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"split_set={getattr(mapping, 'moe_tp_ep_user_specified', False)} "
        f"attention_dp={mapping.enable_attention_dp}; the kimi_k3_vl targets are the MXFP4 "
        "Kimi K3 checkpoint on sm100 (GB200), 16 ranks of tensor-parallel attention, with the "
        "experts split moe_tp 4 x moe_ep 4 or, set explicitly, moe_tp 16 x moe_ep 1"
    )
