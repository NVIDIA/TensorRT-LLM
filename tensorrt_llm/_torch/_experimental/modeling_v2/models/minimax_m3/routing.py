# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MiniMaxM3SparseForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in MiniMax-M3 implementation; in ``require`` it raises that error.

The sparse-attention runtime is part of the decision too (``ROUTES_ON_RUNTIME``):
the targets are written for the MSA implementation with an FP8 indexer and an
FP8 or NVFP4 main KV cache, and any other runtime is the built-in model's.

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower and refuse image input.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, RuntimeFacts

# ``route`` also reads the engine's sparse-attention runtime and KV cache dtype (``RuntimeFacts``).
ROUTES_ON_RUNTIME = True

# The GPU architectures these targets are written for: the tp4 / tp8 decode
# kernels (the tcgen05 GEMVs, the fused MoE engines, the attention megakernels
# and the multicast all-reduce) are built and measured on GB300 (sm103); the
# tp2 and tp4 Rubin (sm107) targets are ports, switched on one certified path
# at a time.
_SM = (10, 3)
_SM_RUBIN = (10, 7)

# The routing knobs of the former separate Rubin tp2 wide-decode target. Its 9..192-row decode cells are now part of the
# one Rubin tp2 target, so there is nothing left to select: a deployment that still sets either knob is stale.
_REMOVED_TP2_KNOBS = ("TRTLLM_M3_TP2_WIDE", "TRTLLM_M3_TP2_WIDE_MIN_BATCH")

# The checkpoint's config shape, read from ``text_config``; see the note in the
# gpt_oss routing module for what shape-sniffing does and does not pin.
#
# (num_hidden_layers, hidden_size, num_local_experts, intermediate_size,
#  num_attention_heads, num_key_value_heads). The expert width and the head
# counts fix the per-rank operand shapes the decode kernels are compiled for.
_SHAPE = (60, 6144, 128, 3072, 64, 4)

# The Eagle3 drafter geometry the targets' drafter kernels and weight tables are built for (the GQA drafter: the
# target's hidden size, heads, KV heads and vocabulary, a 18432-wide MLP).
_DRAFTER_GEOMETRY = dict(
    hidden_size=6144,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=128,
    intermediate_size=18432,
    vocab_size=200064,
)


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _quant_algo(pretrained_config: Any) -> Optional[QuantAlgo]:
    """The quantization the checkpoint's config declares, as the engine reads it.

    The NVFP4 checkpoint is a modelopt mixed-precision export (NVFP4 routed
    experts, MXFP8 dense projections, bf16 router, embedding and head), declared
    in ``config.json`` as ``MIXED_PRECISION``; the per-layer split lives in
    ``hf_quant_config.json``, which ``route`` does not see.
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

    Attention is head-split and the routed experts are tensor-parallel with
    every expert on every rank: the expert split decides which expert weights
    each rank loads and into which shapes (per-rank expert width 768 at tp4,
    384 at tp8, 1536 at tp2), so it selects a target rather than a runtime branch.
    """
    for tp in (2, 4, 8):
        if (
            m.world_size == tp
            and m.tp_size == tp
            and m.pp_size == 1
            and m.cp_size == 1
            and m.moe_tp_size == tp
            and m.moe_ep_size == 1
            and not m.enable_attention_dp
        ):
            return f"tp{tp}"
    return None


def _eagle3_one_model_or_none(spec_config: Optional[Any]) -> bool:
    """No speculation, or the one-model Eagle3 loop the target's worker extends."""
    if spec_config is None:
        return True
    return (
        isinstance(spec_config, EagleDecodingConfig)
        and spec_config.spec_dec_mode.is_eagle3_one_model()
    )


def _drafter_mismatch(spec_config: Optional[Any]) -> Optional[str]:
    """Why the speculative drafter is not the one the targets' flat drafter builds, or None when it is (or there is
    no speculation).

    The targets draft with their own one-layer Eagle3 drafter (``M3Eagle3Draft``): one Llama-style layer, per-layer
    fc_norms over several captured target layers, no eh_proj before attention or norm before the fc, no MLA, the
    attention and MLP geometry of ``_DRAFTER_GEOMETRY``, the target's hidden size and vocabulary, no biases and an
    untied head. Any other Eagle3 drafter is the built-in
    implementation's. Read from the drafter checkpoint's ``config.json`` (the engine has resolved
    ``speculative_model`` to a local directory before it builds the model).
    """
    if spec_config is None:
        return None
    path = getattr(spec_config, "speculative_model", None)
    if path is None:
        return "no Eagle3 drafter checkpoint (speculative_model)"
    try:
        with open(Path(path) / "config.json") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as err:
        return f"the Eagle3 drafter's config.json is not readable ({err})"
    eagle = cfg.get("eagle_config") or {}
    hidden = cfg.get("hidden_size")
    vocab = cfg.get("vocab_size")
    captured = getattr(spec_config, "eagle3_layers_to_capture", None)
    checks = (
        (
            cfg.get("num_hidden_layers") == 1,
            f"{cfg.get('num_hidden_layers')} drafter layers, not 1",
        ),
        (bool(cfg.get("fc_norm", False)), "no per-layer fc_norm"),
        (not eagle.get("eh_proj_before_attn", False), "eh_proj before attention"),
        (not eagle.get("norm_before_fc", False), "norm before the fc"),
        (not cfg.get("kv_lora_rank"), "an MLA drafter layer"),
        (
            cfg.get("target_hidden_size", hidden) == hidden,
            "a target hidden size other than the drafter's",
        ),
        (
            cfg.get("draft_vocab_size", vocab) in (None, vocab),
            "a draft vocabulary other than the target's",
        ),
        (
            not (cfg.get("attention_bias") or cfg.get("mlp_bias") or cfg.get("bias")),
            "biased projections",
        ),
        (not cfg.get("tie_word_embeddings", False), "a tied head"),
        (captured is None or len(captured) > 1, "a single captured target layer"),
    )
    heads = cfg.get("num_attention_heads")
    defaults = dict(
        num_key_value_heads=heads,
        head_dim=hidden // heads if hidden and heads else None,
    )
    geometry = []
    for name, want in _DRAFTER_GEOMETRY.items():
        value = cfg.get(name)
        if value is None:
            value = defaults.get(name)
        geometry.append((value == want, f"{name}={value}, not {want}"))
    reasons = [why for ok, why in (*checks, *geometry) if not ok]
    return "; ".join(reasons) if reasons else None


def _runtime_mismatch(runtime: RuntimeFacts) -> Optional[str]:
    """Why the engine's runtime is not the one the targets are written for, or None when it is.

    The MiniMax-M3 MSA backend (it owns the step metadata and the index-K cache), an FP8 indexer and an FP8 or
    NVFP4 main KV cache, with the fused five-way QKV/index projection or the separate projections. Every other
    runtime the engine accepts (the Triton implementation, the BF16 indexer, the BF16 main KV cache) is the built-in
    implementation's. The targets' ``_validate_runtime`` checks the same at construction.
    """
    sparse = runtime.sparse_attention_config
    if sparse is None or _field(sparse, "algorithm") != "minimax_m3":
        return "no MiniMax-M3 sparse_attention_config"
    reasons = []
    implementation = _field(sparse, "implementation")
    if implementation != "msa":
        reasons.append(f"sparse attention implementation {implementation!r}, not 'msa'")
    indexer = _field(sparse, "indexer_kv_dtype")
    if indexer != "fp8":
        reasons.append(f"indexer_kv_dtype {indexer!r}, not 'fp8'")
    kv = runtime.kv_cache_quant_algo
    if kv not in (QuantAlgo.FP8, QuantAlgo.NVFP4):
        reasons.append(f"a {kv if kv is not None else 'BF16'} main KV cache, not FP8 or NVFP4")
    return "; ".join(reasons) if reasons else None


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
    kv_cache_quant_algo: Optional[Any],
    *,
    runtime: RuntimeFacts,
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    ``kv_cache_quant_algo`` is the resolver's main KV-cache algorithm, passed to every routing module; this router
    reads the same value, with the sparse attention config, from ``runtime`` (``ROUTES_ON_RUNTIME``).

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    text_config = getattr(pretrained_config, "text_config", None)
    shape = (
        None
        if text_config is None
        else tuple(
            _field(text_config, name)
            for name in (
                "num_hidden_layers",
                "hidden_size",
                "num_local_experts",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
            )
        )
    )
    quant_algo = _quant_algo(pretrained_config)
    parallel = _parallel(mapping)
    speculation = _eagle3_one_model_or_none(spec_config)
    drafter = _drafter_mismatch(spec_config) if speculation else None
    runtime_miss = _runtime_mismatch(runtime)

    # MiniMax-M3 NVFP4 on 4 GB300s (sm103), tensor-parallel attention and
    # experts, one-model Eagle3 or no speculative decoding.
    if (
        compute_capability == _SM
        and shape == _SHAPE
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp4"
        and speculation
        and drafter is None
        and runtime_miss is None
    ):
        from .minimax_m3_nvfp4__sm_103__tp4.modeling import ModelingV2MiniMaxM3Nvfp4Sm103Tp4

        return ModelingV2MiniMaxM3Nvfp4Sm103Tp4

    # The same checkpoint on 8 GB300s over two trays (sm103), tensor-parallel
    # attention and experts, one-model Eagle3 or no speculative decoding.
    if (
        compute_capability == _SM
        and shape == _SHAPE
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp8"
        and speculation
        and drafter is None
        and runtime_miss is None
    ):
        from .minimax_m3_nvfp4__sm_103__tp8.modeling import ModelingV2MiniMaxM3Nvfp4Sm103Tp8

        return ModelingV2MiniMaxM3Nvfp4Sm103Tp8

    # The same checkpoint on 2 Rubin GPUs (sm107), tensor-parallel attention and
    # experts, one-model Eagle3 or no speculative decoding, at any batch size (decode rows 1..192).
    if (
        compute_capability == _SM_RUBIN
        and shape == _SHAPE
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp2"
        and speculation
        and drafter is None
        and runtime_miss is None
    ):
        stale = [knob for knob in _REMOVED_TP2_KNOBS if knob in os.environ]
        if stale:
            raise ValueError(
                f"{', '.join(stale)}: removed; the MiniMax-M3 sm107 tp2 target serves every batch size (its "
                "wide-decode cells are part of it), so unset these variables"
            )
        from .minimax_m3_nvfp4__sm_107__tp2.modeling import ModelingV2MiniMaxM3Nvfp4Sm107Tp2

        return ModelingV2MiniMaxM3Nvfp4Sm107Tp2

    # The same checkpoint on 4 Rubin GPUs (sm107), tensor-parallel attention and
    # experts, one-model Eagle3 or no speculative decoding.
    if (
        compute_capability == _SM_RUBIN
        and shape == _SHAPE
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp4"
        and speculation
        and drafter is None
        and runtime_miss is None
    ):
        from .minimax_m3_nvfp4__sm_107__tp4.modeling import ModelingV2MiniMaxM3Nvfp4Sm107Tp4

        return ModelingV2MiniMaxM3Nvfp4Sm107Tp4

    # The same checkpoint on 8 Rubin GPUs over two trays (sm107), tensor-parallel
    # attention and experts, one-model Eagle3 or no speculative decoding.
    if (
        compute_capability == _SM_RUBIN
        and shape == _SHAPE
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp8"
        and speculation
        and drafter is None
        and runtime_miss is None
    ):
        from .minimax_m3_nvfp4__sm_107__tp8.modeling import ModelingV2MiniMaxM3Nvfp4Sm107Tp8

        return ModelingV2MiniMaxM3Nvfp4Sm107Tp8

    raise NoTarget(
        f"compute capability {compute_capability}, text_config shape {shape}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, parallel "
        f"ws={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} cp={mapping.cp_size} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"attention_dp={mapping.enable_attention_dp}, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
        f"{f' (Eagle3 drafter: {drafter})' if drafter else ''}"
        f"{f', runtime: {runtime_miss}' if runtime_miss else ''}; the "
        "minimax_m3 targets are the MiniMax-M3 NVFP4 checkpoint on sm103 (GB300) with 4 or 8 ranks, or on sm107 "
        "(Rubin) with 2, 4 or 8 ranks, of tensor-parallel attention and experts, with one-model Eagle3 (a one-layer "
        "drafter with per-layer fc_norms) or no speculative decoding, on the MSA sparse attention runtime with an FP8 "
        "indexer and an FP8 or NVFP4 KV cache"
    )
