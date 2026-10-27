# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a MiniMaxM3SparseForConditionalGeneration config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating those facts. In ``auto`` the engine then
uses the built-in MiniMax-M3 implementation; in ``require`` it raises that error.

The KV cache the engine runs with is part of the decision: the targets are
written for an FP8 or NVFP4 main KV cache. The rest of their runtime (the MSA
sparse attention implementation with an FP8 indexer) is an engine argument
rather than a checkpoint fact, so the targets check it at construction
(``_validate_runtime``).

The checkpoint declares the vision-language wrapper, and the language model's
shape lives in its ``text_config``, so that is what this tree reads. The
targets are text-only: they load no vision tower and refuse image input.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# The one GPU architecture these targets are written for: the decode kernels
# (the tcgen05 GEMVs, the fused MoE engines, the attention megakernels and the
# multicast all-reduce) are built and measured on GB300 only.
_SM = (10, 3)

# The checkpoint's config shape, read from ``text_config``; see the note in the
# gpt_oss routing module for what shape-sniffing does and does not pin.
#
# (num_hidden_layers, hidden_size, num_local_experts, intermediate_size,
#  num_attention_heads, num_key_value_heads). The expert width and the head
# counts fix the per-rank operand shapes the decode kernels are compiled for.
_SHAPE = (60, 6144, 128, 3072, 64, 4)

# The architecture class and the shape do not pin everything the targets are built for: a newer
# checkpoint revision of this model family has the same architecture class and shapes but a different
# quant packing (compressed-tensors NVFP4 experts). The targets build a 5e6 RoPE table and dense
# attention on layers 0-2 (sparse on 3-59), so these two facts, with the quantization below, keep any
# other revision off the targets.
_ROPE_THETA = 5_000_000
_SPARSE_ATTENTION_FREQ = (0,) * 3 + (1,) * 57


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _layout_mismatch(text_config: Any) -> Optional[str]:
    """Why the text model's RoPE base or sparse-attention layers are not the ones the targets build, or None."""
    if text_config is None:
        return None  # the shape check declines it
    theta = _field(text_config, "rope_theta")
    if theta is None:
        rope = _field(text_config, "rope_parameters") or {}
        theta = _field(
            rope.get("full_attention", rope) if isinstance(rope, dict) else rope, "rope_theta"
        )
    sparse = _field(text_config, "sparse_attention_config")
    freq = None if sparse is None else _field(sparse, "sparse_attention_freq")
    # The built-in model reads use_sparse_attention=False as dense attention on every layer.
    enabled = sparse is not None and _field(sparse, "use_sparse_attention") is not False
    reasons = []
    if theta != _ROPE_THETA:
        reasons.append(f"rope_theta {theta}, not {_ROPE_THETA}")
    if not enabled or freq is None or tuple(int(f) for f in freq) != _SPARSE_ATTENTION_FREQ:
        reasons.append("sparse attention layers other than 3-59")
    return "; ".join(reasons) if reasons else None


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
    384 at tp8), so it selects a target rather than a runtime branch.
    """
    for tp in (4, 8):
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


@routes("MiniMaxM3SparseForConditionalGeneration")
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
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
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
    layout_miss = _layout_mismatch(text_config)
    speculation = _eagle3_one_model_or_none(spec_config)
    kv_cache = kv_cache_quant_algo in (QuantAlgo.FP8, QuantAlgo.NVFP4)

    # MiniMax-M3 NVFP4 on 4 GB300s (sm103), tensor-parallel attention and
    # experts, one-model Eagle3 or no speculative decoding.
    if (
        compute_capability == _SM
        and shape == _SHAPE
        and layout_miss is None
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp4"
        and speculation
        and kv_cache
    ):
        from .minimax_m3_nvfp4__sm_103__tp4.modeling import ModelingV2MiniMaxM3Nvfp4Sm103Tp4

        return ModelingV2MiniMaxM3Nvfp4Sm103Tp4

    # The same checkpoint on 8 GB300s over two trays (sm103), tensor-parallel
    # attention and experts, one-model Eagle3 or no speculative decoding.
    if (
        compute_capability == _SM
        and shape == _SHAPE
        and layout_miss is None
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp8"
        and speculation
        and kv_cache
    ):
        from .minimax_m3_nvfp4__sm_103__tp8.modeling import ModelingV2MiniMaxM3Nvfp4Sm103Tp8

        return ModelingV2MiniMaxM3Nvfp4Sm103Tp8

    raise NoTarget(
        f"compute capability {compute_capability}, text_config shape {shape}"
        f"{f' ({layout_miss})' if layout_miss else ''}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, parallel "
        f"ws={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} cp={mapping.cp_size} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"attention_dp={mapping.enable_attention_dp}, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
        f", KV cache {kv_cache_quant_algo or 'bf16'}; the "
        "minimax_m3 targets are the MiniMax-M3 NVFP4 checkpoint on sm103 (GB300), 4 or 8 ranks of "
        "tensor-parallel attention and experts, with one-model Eagle3 or no speculative decoding, and an FP8 or "
        "NVFP4 KV cache"
    )
