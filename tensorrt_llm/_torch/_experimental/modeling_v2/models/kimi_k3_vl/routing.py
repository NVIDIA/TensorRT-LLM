# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a KimiK3ForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in Kimi K3 implementation; in ``require`` it raises that error.

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower and refuse image input.

This module and the family modules it reads (``drafter_config``,
``environment``) import no target package: declining a deployment loads no
modeling code and no native library.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes
from . import drafter_config as _drafter_config
from . import environment as _environment

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
# shape too; the quantization in ``config.json`` tells the two apart, and
# ``_unmet_settings`` also checks the quantization the engine resolved.
_SHAPE = (93, 7168, 896, 3584)

# The KV-cache block size the targets' MLA kernels are built for.
_TOKENS_PER_BLOCK = 64

# The routed-expert backend the targets' MoE runs. The engine resolves AUTO to
# TRTLLM for Kimi K3 before it routes.
_MOE_BACKENDS = frozenset({"AUTO", "TRTLLM"})

# Environment switches the built-in implementation reads and the targets do
# not: the name and the values that leave the built-in on the path the targets
# run (None: unset). The switches the targets read are in ``environment.py``.
_ENV_SWITCHES = (
    ("ENABLE_PERFECT_ROUTER", (None, "0")),
    ("EXPERT_STATISTIC_ITER_RANGE", (None,)),
    ("TRTLLM_LOW_M_GEMM_BACKEND", (None, "off")),
    ("TLLM_KDA_ENABLE_OPT_PREFILL", (None, "1")),
    ("TLLM_KDA_ENABLE_INDEXED_STATE_POOL", (None, "1")),
    ("FLA_CONV_BACKEND", (None, "triton")),
)


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


def _unmet_settings(
    model_config: ModelConfig,
    llm_args: TorchLlmArgs,
    parallel: str,
    compute_capability: Tuple[int, int],
) -> List[str]:
    """The settings the ``parallel`` target needs that this deployment does not have.

    The targets check the same settings when they are built or on their first
    step. Checking them here first lets ``auto`` fall back to the built-in
    implementation instead of failing after the target is chosen.
    """
    import torch

    from tensorrt_llm.functional import AllReduceStrategy

    unmet = []

    spec_config = model_config.spec_config
    if spec_config is not None:
        mode = spec_config.spec_dec_mode
        if parallel == "tp16_moetp16ep1":
            unmet.append(
                f"speculative decoding {mode.name}: the tp16_moetp16ep1 target decodes without "
                "speculation"
            )
        elif not (mode.is_sa() or mode.is_dspark()):
            unmet.append(
                f"speculative decoding {mode.name}: the tp16_moetp4ep4 target supports SA and "
                "DSpark with a standalone drafter"
            )
        elif mode.is_dspark() and spec_config.draft_is_embedded_in_target:
            unmet.append(
                "speculative decoding DSPARK with the draft embedded in the target: the "
                "tp16_moetp4ep4 target supports DSpark with a standalone drafter"
            )
        elif mode.is_dspark():
            # The drafter checkpoint's config, read by the functions the target's drafters apply when they are
            # built.
            sm = compute_capability[0] * 10 + compute_capability[1]
            reason = _drafter_config.standalone_drafter_declines(model_config, spec_config, sm)
            if reason is not None:
                unmet.append(
                    f"speculative decoding DSPARK: the tp16_moetp4ep4 target does not build this drafter: {reason}"
                )

    if model_config.torch_dtype != torch.bfloat16:
        unmet.append(f"dtype {model_config.torch_dtype}: the targets compute in bf16")

    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    if kv_cache_quant_algo is not None:
        unmet.append(
            f"KV cache {getattr(kv_cache_quant_algo, 'name', kv_cache_quant_algo)}: the targets' "
            "MLA kernels read a bf16 KV cache"
        )

    quant_algo = model_config.quant_config.quant_algo
    layer_quant = model_config.quant_config_dict
    if quant_algo != QuantAlgo.W4A16_MXFP4 or layer_quant:
        unmet.append(
            f"resolved quantization {quant_algo} with {len(layer_quant or {})} per-layer "
            "declarations: the targets load the MXFP4 checkpoint, W4A16_MXFP4 with no "
            "per-layer declarations"
        )

    strategy = model_config.allreduce_strategy
    if strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL):
        unmet.append(f"allreduce_strategy {strategy.name}: the targets take AUTO or MNNVL")

    kv_cache_config = llm_args.kv_cache_config
    if kv_cache_config.tokens_per_block != _TOKENS_PER_BLOCK:
        unmet.append(
            f"kv_cache_config.tokens_per_block {kv_cache_config.tokens_per_block}: the targets "
            f"need {_TOKENS_PER_BLOCK}"
        )
    if kv_cache_config.enable_block_reuse:
        unmet.append("kv_cache_config.enable_block_reuse true: the targets need it false")
    if kv_cache_config.use_kv_cache_manager_v2 is not True:
        unmet.append(
            f"kv_cache_config.use_kv_cache_manager_v2 {kv_cache_config.use_kv_cache_manager_v2}: "
            "the targets keep the KDA states in the V2 hybrid cache manager"
        )
    if kv_cache_config.mamba_ssm_cache_dtype not in ("auto", "float32"):
        unmet.append(
            f"kv_cache_config.mamba_ssm_cache_dtype {kv_cache_config.mamba_ssm_cache_dtype}: "
            "the targets' KDA kernels keep fp32 recurrent states"
        )

    moe_backend = str(model_config.moe_backend).upper()
    if moe_backend not in _MOE_BACKENDS:
        unmet.append(
            f"moe_config.backend {moe_backend}: the targets' routed experts run on "
            f"{', '.join(sorted(_MOE_BACKENDS))}"
        )
    if model_config.moe_load_balancer is not None:
        unmet.append("moe_config.load_balancer set: the targets do not support EPLB")
    if getattr(llm_args, "enable_return_routed_experts", False):
        unmet.append(
            "enable_return_routed_experts true: the targets' MoE does not record routed experts"
        )
    for name in ("use_cute_dsl_bf16_gemm", "use_cute_dsl_bf16_bmm"):
        if getattr(llm_args, name, False):
            unmet.append(f"{name} true: the targets run their bf16 projections on cuBLAS")

    for name, built_for in _ENV_SWITCHES:
        value = os.environ.get(name)
        if value not in built_for:
            unmet.append(f"environment {name}={value}: the targets do not read this switch")
    for reason in _environment.declines():
        unmet.append(f"environment: {reason}")

    return unmet


@routes("KimiK3ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining imports no
    target package. With a standalone DSpark drafter, the tp16_moetp4ep4 check
    loads the drafter checkpoint's config.json through the family's
    ``drafter_config``.
    """
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

    if mxfp4_on_sm100 and parallel is not None:
        unmet = _unmet_settings(model_config, llm_args, parallel, compute_capability)
        if unmet:
            raise NoTarget(f"the {parallel} target needs other settings: " + "; ".join(unmet))

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
