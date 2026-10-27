# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route the MiniMax-M3.1 sm_103 TP4/TP8 and sm_107 TP2/TP4/TP8 targets; other shapes decline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget

TEXT_ONLY = True
_SM = (10, 3)
_SM_RUBIN = (10, 7)
_SHAPE = (60, 6144, 128, 3072, 64, 4)
_M31_ROPE_THETA = 10_000_000
_M31_SPARSE_ATTENTION_FREQ = (1,) * 60
_M31_EXPERTS = ("NVFP4", "nvfp4-pack-quantized")
_M31_DSPARK_ARCH = "DSparkMiniMaxDraftModel"
_M31_DSPARK_GEOMETRY = dict(
    hidden_size=6144, vocab_size=200064, num_key_value_heads=4, head_dim=128
)
_M31_DSPARK_TARGET_LAYERS = [3, 17, 31, 45, 59]


def _field(config: Any, name: str) -> Any:
    """Read ``name`` off a sub-config that may be an object or a plain dict."""
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def _rope_theta(text_config: Any) -> Any:
    theta = _field(text_config, "rope_theta")
    if theta is None:
        rope = _field(text_config, "rope_parameters") or {}
        theta = _field(
            rope.get("full_attention", rope) if isinstance(rope, dict) else rope, "rope_theta"
        )
    return theta


def _m31_mismatch(
    pretrained_config: Any, text_config: Any, quant_algo: Optional[QuantAlgo]
) -> Optional[str]:
    """Why the checkpoint is not the MiniMax-M3.1 export the m31 targets build, or None."""
    if text_config is None:
        return "no text_config"
    reasons = []
    theta = _rope_theta(text_config)
    if theta != _M31_ROPE_THETA:
        reasons.append(f"rope_theta {theta}, not {_M31_ROPE_THETA}")
    sparse = _field(text_config, "sparse_attention_config")
    freq = None if sparse is None else _field(sparse, "sparse_attention_freq")
    enabled = sparse is not None and _field(sparse, "use_sparse_attention") is not False
    if not enabled or freq is None or tuple(int(f) for f in freq) != _M31_SPARSE_ATTENTION_FREQ:
        reasons.append("sparse attention not on every layer")
    declared = getattr(pretrained_config, "quantization_config", None) or {}
    experts = (_field(declared, "moe_quant_algo"), _field(declared, "moe_quant_format"))
    if quant_algo != QuantAlgo.MXFP8 or experts != _M31_EXPERTS:
        reasons.append(
            f"quantization {quant_algo} with experts {experts}, not MXFP8 with {_M31_EXPERTS}"
        )
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
    each rank loads and into which shapes (per-rank expert widths 1536 at TP2, 768 at TP4 and 384 at TP8),
    so it selects a target rather than a runtime branch.
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


def _m31_tp2_routed() -> bool:
    """Whether the MiniMax-M3.1 sm_107 TP2 decode target takes its topology. On by default;
    ``TRTLLM_M31_DISABLE_TP2=1`` keeps the built-in model."""
    from .minimax_m31_nvfp4__sm_107__tp2.m31_switches import m31_disabled

    return not m31_disabled("TP2")


def _m31_speculation_mismatch(spec_config: Optional[Any]) -> Optional[str]:
    """Why the speculation is not one the M3.1 targets serve, or None when it is (none, or a standalone MiniMax DSpark
    drafter). Read from the drafter checkpoint's ``config.json``."""
    if spec_config is None:
        return None
    mode = spec_config.spec_dec_mode
    if not (mode.use_one_engine() and mode.is_dspark()):
        return f"speculative decoding {mode.name}, not standalone DSpark"
    if getattr(spec_config, "draft_is_embedded_in_target", False):
        return "a DSpark drafter embedded in the target checkpoint"
    path = getattr(spec_config, "speculative_model", None)
    if path is None:
        return "no DSpark drafter checkpoint (speculative_model)"
    try:
        with open(Path(path) / "config.json") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as err:
        return f"the DSpark drafter's config.json is not readable ({err})"
    text = cfg.get("text_config") or {}

    def field(name):
        return text.get(name, cfg.get(name))

    arches = cfg.get("architectures") or text.get("architectures") or []
    if not arches or arches[0] != _M31_DSPARK_ARCH:
        return f"drafter architecture {arches[:1]}, not {_M31_DSPARK_ARCH}"
    if list(field("dspark_target_layer_ids") or []) != _M31_DSPARK_TARGET_LAYERS:
        return f"drafter target layers {field('dspark_target_layer_ids')}, not {_M31_DSPARK_TARGET_LAYERS}"
    for name, want in _M31_DSPARK_GEOMETRY.items():
        if field(name) != want:
            return f"drafter {name} {field(name)}, not {want}"
    return None


def arm_sm107_shared_opt_ins() -> None:
    """Arm, process-wide and before the model is built, the shared paths the MiniMax-M3.1 sm_107 targets take: the
    shipped ahead-of-time CuTe DSL objects, the drafter KV reserve and guard page, the host-tier KV prefetch, the
    context-only int32 prompt tokens, the BF16X / E4M3X sparse-prefill fork, the indexer proxy max-score buffer reuse,
    the context one-hot index_fill, the DFlash host-side context allocation and GPU timestamps for the executor's
    timing events. Each keeps its kill switch. The MSA context paths and warmup stay the ones these targets run
    (msa_indexer.arm_sm107_msa_paths), and CUDA-graph padding rows stay zeroed
    (interface.arm_sm107_spec_paths). Armed here
    rather than in a target's RuntimeHooks so that the target directories (the AOT objects' source digests) are not
    edited."""
    from tensorrt_llm._torch import cute_aot
    from tensorrt_llm._torch.attention.backends.fmha import msa_prefill
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
    from tensorrt_llm._torch.pyexecutor import llm_request, perf_metrics_manager, py_executor
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
    from tensorrt_llm._torch.speculative import dflash, interface

    cute_aot.arm()
    kv_cache_manager_v2.arm_draft_kv_reserve_guard()
    py_executor.arm_kv_gpu_prefetch()
    llm_request.arm_context_token_array()
    msa_prefill.arm_prefill_fork()
    msa_indexer.arm_proxy_ms_reuse()
    msa_indexer.arm_sm107_msa_paths()
    interface.arm_onehot_index_fill()
    interface.arm_sm107_spec_paths()
    dflash.arm_host_ctx_alloc()
    perf_metrics_manager.arm_gpu_timestamps()


def route(
    pretrained_config: Any,
    mapping: Any,
    compute_capability: Tuple[int, int],
    spec_config: Optional[Any],
    kv_cache_quant_algo: Optional[Any],
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    ``kv_cache_quant_algo`` is the KV-cache algorithm the engine will run with
    (see ``_router_index._kv_cache_quant_algo``), not the checkpoint's alone.

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
    kv_cache = kv_cache_quant_algo in (QuantAlgo.FP8, QuantAlgo.NVFP4)
    m31_miss = _m31_mismatch(pretrained_config, text_config, quant_algo)
    m31_spec_miss = _m31_speculation_mismatch(spec_config) if m31_miss is None else None

    if (
        compute_capability == _SM
        and shape == _SHAPE
        and m31_miss is None
        and parallel in ("tp4", "tp8")
        and m31_spec_miss is None
        and kv_cache
    ):
        if parallel == "tp4":
            from .minimax_m31_nvfp4__sm_103__tp4.modeling import ModelingV2MiniMaxM31Nvfp4Sm103Tp4

            return ModelingV2MiniMaxM31Nvfp4Sm103Tp4
        from .fake_ops import register

        register()
        from .minimax_m31_nvfp4__sm_103__tp8.modeling import ModelingV2MiniMaxM31Nvfp4Sm103Tp8

        return ModelingV2MiniMaxM31Nvfp4Sm103Tp8

    # The same export on 2, 4 or 8 Rubin GPUs (sm_107), on the same terms (TP2: moe_tp 2, every expert on both ranks).
    if (
        compute_capability == _SM_RUBIN
        and shape == _SHAPE
        and m31_miss is None
        and parallel in ("tp2", "tp4", "tp8")
        and (parallel != "tp2" or _m31_tp2_routed())
        and m31_spec_miss is None
        and kv_cache
    ):
        from .fake_ops import register

        register()
        arm_sm107_shared_opt_ins()
        logger.info(
            "M3.1 DSpark MXDRAFT projection counts describe registered hooks, not executed GEMVs. "
            "Row-parallel projections can bypass those hooks via the BF16 NCCL-window path."
        )
        if parallel == "tp2":
            from ...catalog.gemm.m31_router_half_pin import arm_router_half_pin

            arm_router_half_pin()
            from ...catalog.spec_decode.m31_draft_bf16 import arm_draft_bf16

            arm_draft_bf16()
            from .minimax_m31_nvfp4__sm_107__tp2.modeling import ModelingV2MiniMaxM31Nvfp4Sm107Tp2

            return ModelingV2MiniMaxM31Nvfp4Sm107Tp2
        if parallel == "tp4":
            from .minimax_m31_nvfp4__sm_107__tp4.modeling import ModelingV2MiniMaxM31Nvfp4Sm107Tp4

            return ModelingV2MiniMaxM31Nvfp4Sm107Tp4
        from .minimax_m31_nvfp4__sm_107__tp8.modeling import ModelingV2MiniMaxM31Nvfp4Sm107Tp8

        return ModelingV2MiniMaxM31Nvfp4Sm107Tp8

    raise NoTarget(
        f"MiniMax-M3.1 requires sm_103 TP4 or TP8/EP1 or sm_107 TP2, TP4 or TP8/EP1, shape {_SHAPE}, "
        "MXFP8/NVFP4 experts, "
        f"FP8 or NVFP4 KV, and standalone MiniMax DSpark or no speculation; "
        f"got SM={compute_capability}, shape={shape}, topology={parallel}, "
        f"KV={kv_cache_quant_algo}, checkpoint={m31_miss}, speculation={m31_spec_miss}"
    )
