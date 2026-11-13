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

MiniMax-M3.1-Flash has the same architecture class and text shape but another
RoPE base, sparse attention on every layer and another quantization, so the
MiniMax-M3 checks decline it. ``route`` then checks its own targets (sm103 tp4 /
tp8 and sm107, a standalone MiniMax DSpark drafter or no speculative decoding)
before it raises; any other checkpoint keeps the MiniMax-M3 message.

"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

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


# MiniMax-M3.1-Flash: the MiniMax-M3 text shape (``_SHAPE``) with a 1e7 RoPE base, sparse attention on every layer, and
# MXFP8 dense projections with packed NVFP4 routed experts. Its speculative decoding is a standalone MiniMax DSpark
# drafter with these target layers and geometry.
_M31_ROPE_THETA = 10_000_000
_M31_SPARSE_ATTENTION_FREQ = (1,) * 60
_M31_EXPERTS = ("NVFP4", "nvfp4-pack-quantized")
_M31_DSPARK_ARCH = "DSparkMiniMaxDraftModel"
_M31_DSPARK_GEOMETRY = dict(
    hidden_size=6144, vocab_size=200064, num_key_value_heads=4, head_dim=128
)
_M31_DSPARK_TARGET_LAYERS = [3, 17, 31, 45, 59]
# The attention-DP generation worker's sizes (DEP4 / DEP8 / DEP16).
_M31_DEP_SIZES = (4, 8, 16)
# The MiniMax-M3.1-Flash topologies with a GB300 (sm103) target.
_M31_SM103_TOPOLOGIES = ("tp4", "tp8")


def _rope_theta(text_config: Any) -> Any:
    """The text model's RoPE base, from ``rope_theta`` or ``rope_parameters``."""
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
    """Why the checkpoint is not the MiniMax-M3.1-Flash export the M3.1 targets build, or None."""
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


def _tp2ctx(m) -> bool:
    """The MiniMax-M3.1-Flash context worker's topology: 2 ranks of tensor-parallel attention (attention DP off) with
    the routed experts split over them by expert parallelism (moe_ep 2)."""
    return (
        m.world_size == 2
        and m.tp_size == 2
        and m.pp_size == 1
        and m.cp_size == 1
        and m.moe_tp_size == 1
        and m.moe_ep_size == 2
        and not m.enable_attention_dp
    )


def _dep2ctx(m) -> bool:
    """The MiniMax-M3.1-Flash context worker on attention DP: 2 ranks, each running the attention on its own requests
    at full heads, with the routed experts split over them by expert parallelism (moe_ep 2)."""
    return (
        m.world_size == 2
        and m.tp_size == 2
        and m.pp_size == 1
        and m.cp_size == 1
        and m.moe_tp_size == 1
        and m.moe_ep_size == 2
        and m.enable_attention_dp
    )


def _dep(m) -> bool:
    """The MiniMax-M3.1-Flash generation worker's DEP topology: one attention-DP group over every rank, with the routed
    experts split over all of them by expert parallelism (DEP4 / DEP8 / DEP16). The DEP target checks the same at
    construction."""
    return bool(
        m.enable_attention_dp
        and m.world_size in _M31_DEP_SIZES
        and m.tp_size == m.world_size
        and m.moe_ep_size == m.world_size
        and m.moe_tp_size == 1
        and m.pp_size == 1
        and m.cp_size == 1
    )


def _m31_topology(m) -> Optional[str]:
    """Name the MiniMax-M3.1-Flash target topology, or None if no M3.1 target implements it (tp4 runs on sm103
    only; ``_m31_refused`` checks the architecture)."""
    parallel = _parallel(m)
    if parallel in ("tp2", "tp4", "tp8"):
        return parallel
    if _tp2ctx(m):
        return "tp2ctx"
    if _dep2ctx(m):
        return "dep2ctx"
    if _dep(m):
        return "dep"
    return None


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


def _m31_drafter_window(spec_config: Any) -> Optional[int]:
    """The DSpark drafter's attention window as its ``config.json`` declares it, or None for full attention.

    A window from ``dflash_config`` (``use_swa`` with ``swa_window_size``) or from ``sliding_window`` (unless
    ``use_sliding_window`` is false or ``layer_types`` names no sliding layer) counts.
    """
    path = getattr(spec_config, "speculative_model", None)
    if path is None:
        return None
    try:
        with open(Path(path) / "config.json") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    text = cfg.get("text_config") or {}

    def field(name):
        return text.get(name, cfg.get(name))

    def positive_int(value) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    dflash = field("dflash_config")
    if (
        isinstance(dflash, dict)
        and dflash.get("use_swa")
        and positive_int(dflash.get("swa_window_size"))
    ):
        return dflash["swa_window_size"]
    window = field("sliding_window")
    layer_types = field("layer_types")
    if field("use_sliding_window") is False or not positive_int(window):
        return None
    if layer_types and not any("sliding" in str(layer_type).lower() for layer_type in layer_types):
        return None
    return window


def _m31_disagg_generation(llm_args: TorchLlmArgs) -> bool:
    """Whether this worker is a disaggregated generation worker: the generation role, or a KV cache transceiver
    without the context role."""
    role = os.environ.get("TRTLLM_DISAGG_ROLE", "").strip().lower()
    if role.startswith(("generation", "gen")):
        return True
    return llm_args.cache_transceiver_config is not None and not role.startswith(("context", "ctx"))


def _m31_refused(
    llm_args: TorchLlmArgs,
    compute_capability: Tuple[int, int],
    topology: Optional[str],
    kv_cache_quant_algo: Any,
    spec_config: Optional[Any],
) -> List[str]:
    """What the MiniMax-M3.1-Flash targets do not run for this deployment of the M3.1 checkpoint; empty when one is
    admitted (``topology`` names it)."""
    from .runtime_hooks_compat import missing_defaults

    refused = []
    speculation_miss = _m31_speculation_mismatch(spec_config)
    if speculation_miss is not None:
        refused.append(speculation_miss)
    if compute_capability == _SM:
        if topology is not None and topology not in _M31_SM103_TOPOLOGIES:
            refused.append(f"{topology} on sm103 (GB300), where only tp4 and tp8 run")
    elif compute_capability != _SM_RUBIN:
        refused.append("neither sm103 (GB300) nor sm107 (Rubin)")
    elif topology == "tp4":
        refused.append("tp4 on sm107 (Rubin), where no tp4 target runs")
    if kv_cache_quant_algo not in (QuantAlgo.FP8, QuantAlgo.NVFP4):
        refused.append("a KV cache other than FP8 or NVFP4")
    if topology is None:
        refused.append("a parallel layout no target runs")
    if spec_config is not None and speculation_miss is None:
        window = _m31_drafter_window(spec_config)
        max_seq_len = getattr(llm_args, "max_seq_len", None)
        windowed = window is not None and (max_seq_len is None or window < max_seq_len)
        if (
            windowed
            and llm_args.enable_chunked_prefill
            and llm_args.kv_cache_config.host_cache_size in (None, 0)
        ):
            refused.append(
                f"a DSpark drafter with a {window}-token window, chunked prefill and no host KV cache, whose "
                "separate draft pool needs a history-first resize this engine does not have"
            )
        if (
            windowed
            and topology in ("tp2", "tp4", "tp8", "dep")
            and _m31_disagg_generation(llm_args)
        ):
            refused.append(
                f"a disaggregated generation worker with a {window}-token DSpark draft window, which this engine "
                "does not carry from the context worker"
            )
    refused.extend(missing_defaults())
    return refused


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

    # The same checkpoint on 2 Rubin GPUs (sm107), tensor-parallel attention and
    # experts, one-model Eagle3 or no speculative decoding, at any batch size (decode rows 1..192).
    if (
        compute_capability == _SM_RUBIN
        and shape == _SHAPE
        and layout_miss is None
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp2"
        and speculation
        and kv_cache
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
        and layout_miss is None
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp4"
        and speculation
        and kv_cache
    ):
        from .minimax_m3_nvfp4__sm_107__tp4.modeling import ModelingV2MiniMaxM3Nvfp4Sm107Tp4

        return ModelingV2MiniMaxM3Nvfp4Sm107Tp4

    # The same checkpoint on 8 Rubin GPUs over two trays (sm107), tensor-parallel
    # attention and experts, one-model Eagle3 or no speculative decoding.
    if (
        compute_capability == _SM_RUBIN
        and shape == _SHAPE
        and layout_miss is None
        and quant_algo == QuantAlgo.MIXED_PRECISION
        and parallel == "tp8"
        and speculation
        and kv_cache
    ):
        from .minimax_m3_nvfp4__sm_107__tp8.modeling import ModelingV2MiniMaxM3Nvfp4Sm107Tp8

        return ModelingV2MiniMaxM3Nvfp4Sm107Tp8

    # MiniMax-M3.1-Flash (the same architecture class and shape, another checkpoint) on Rubin GPUs (sm107): tp2 / tp8
    # with tensor-parallel attention and experts; the 2-rank context worker with EP2 experts and tensor-parallel
    # (tp2ctx) or attention-DP (dep2ctx) attention; the DEP4/8/16 generation worker (dep, built only when
    # modeling_v2 is set). A standalone MiniMax DSpark drafter or no speculative decoding.
    if shape == _SHAPE and _m31_mismatch(pretrained_config, text_config, quant_algo) is None:
        m31_topology = _m31_topology(mapping)
        m31_refused = _m31_refused(
            llm_args, compute_capability, m31_topology, kv_cache_quant_algo, spec_config
        )
        # GB300 (sm103): tp4 / tp8 with tensor-parallel attention and experts.
        if not m31_refused and compute_capability == _SM and m31_topology == "tp4":
            from .minimax_m3_1_flash_nvfp4__sm_103__tp4.modeling import (
                ModelingV2MiniMaxM31FlashNvfp4Sm103Tp4,
            )

            return ModelingV2MiniMaxM31FlashNvfp4Sm103Tp4
        if not m31_refused and compute_capability == _SM and m31_topology == "tp8":
            from .minimax_m3_1_flash_nvfp4__sm_103__tp8.modeling import (
                ModelingV2MiniMaxM31FlashNvfp4Sm103Tp8,
            )

            return ModelingV2MiniMaxM31FlashNvfp4Sm103Tp8
        if not m31_refused and m31_topology == "tp2":
            from .minimax_m3_1_flash_nvfp4__sm_107__tp2.modeling import (
                ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2,
            )

            return ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2
        if not m31_refused and m31_topology == "tp8":
            from .minimax_m3_1_flash_nvfp4__sm_107__tp8.modeling import (
                ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8,
            )

            return ModelingV2MiniMaxM31FlashNvfp4Sm107Tp8
        if not m31_refused and m31_topology == "tp2ctx":
            from .minimax_m3_1_flash_nvfp4__sm_107__tp2ctx.modeling import (
                ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx,
            )

            return ModelingV2MiniMaxM31FlashNvfp4Sm107Tp2ctx
        if not m31_refused and m31_topology == "dep2ctx":
            from .minimax_m3_1_flash_nvfp4__sm_107__dep2ctx.modeling import (
                ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx,
            )

            return ModelingV2MiniMaxM31FlashNvfp4Sm107Dep2ctx
        if not m31_refused and m31_topology == "dep":
            from .minimax_m3_1_flash_nvfp4__sm_107__dep.modeling import (
                ModelingV2MiniMaxM31FlashNvfp4Sm107Dep,
            )

            return ModelingV2MiniMaxM31FlashNvfp4Sm107Dep
        raise NoTarget(
            f"MiniMax-M3.1-Flash: {'; '.join(m31_refused)}; got compute capability {compute_capability}, parallel "
            f"ws={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} cp={mapping.cp_size} "
            f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
            f"attention_dp={mapping.enable_attention_dp}, speculative decoding "
            f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
            f", KV cache {kv_cache_quant_algo or 'bf16'}; the MiniMax-M3.1-Flash targets run on sm103 (GB300) with 4 "
            "or 8 ranks of tensor-parallel attention and experts, or on sm107 (Rubin) with "
            "2 or 8 ranks of tensor-parallel attention and experts, 2 ranks of tensor-parallel or attention-DP "
            "attention with EP2 experts (the context worker), or attention DP with expert parallelism over 4, 8 "
            "or 16 ranks, with a standalone MiniMax DSpark drafter or no speculative decoding, and an FP8 or "
            "NVFP4 KV cache"
        )

    raise NoTarget(
        f"compute capability {compute_capability}, text_config shape {shape}"
        f"{f' ({layout_miss})' if layout_miss else ''}, "
        f"quantization {quant_algo.name if quant_algo is not None else 'none'}, parallel "
        f"ws={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} cp={mapping.cp_size} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size} "
        f"attention_dp={mapping.enable_attention_dp}, speculative decoding "
        f"{spec_config.spec_dec_mode.name if spec_config is not None else 'off'}"
        f", KV cache {kv_cache_quant_algo or 'bf16'}; the "
        "minimax_m3 targets are the MiniMax-M3 NVFP4 checkpoint on sm103 (GB300) with 4 or 8 ranks, or on sm107 "
        "(Rubin) with 2, 4 or 8 ranks, of tensor-parallel attention and experts, with one-model Eagle3 or no "
        "speculative decoding, and an FP8 or NVFP4 KV cache"
    )
