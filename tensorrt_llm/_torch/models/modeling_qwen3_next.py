# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/qwen3_next.py
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from torch import nn
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.fused_shared_expert import \
    fused_sigmoid_gate_mul_add
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.config_utils import \
    get_qwen3_hybrid_layer_types
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...logger import logger
from ..attention_backend import AttentionMetadata
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduce, MoEAllReduceParams, allgather)
from ..distributed.ops import q35_mnnvl_fusable, q35_zero_residual
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (BaseMoeRoutingMethod, MoEWeightLoadingMode,
                                 RenormalizeMoeRoutingMethod,
                                 RenormalizeNaiveMoeRoutingMethod,
                                 RoutingMethodType, create_moe)

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from ..modules.fused_moe.q35_fused_front import (
    Q35_MOE_FRONT_BIG, Q35_MOE_FUSED_FRONT, Q35_MOE_FUSED_FRONT_MAX_TOKENS,
    Q35_MOE_GATE_LOWM, Q35_MOE_ROUTER_GEMM, Q35_MOE_ROUTER_LOWM, Q35_MOE_STATS,
    Q35RouterGemm, q35_lowm_router, q35_moe_stats_record, q35_routed_forward)
# isort: on
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import FP8QDQLinearMethod, Linear, TensorParallelMode
from ..modules.lowm_linear import (lowm_check, lowm_gemm, lowm_rows_ok,
                                   lowm_se_enabled, lowm_skipped, swiglu_pairs)
from ..modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from ..modules.multi_stream_utils import (do_multi_stream,
                                          maybe_execute_in_parallel)
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, create_lm_head_tp_mapping
from .modeling_qwen3 import Qwen3Attention
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, EagerFusionConfig, register_auto_model


def _fused_norm_weight(norm: RMSNorm) -> torch.Tensor:
    """Weight to feed the fused AllReduce+RMSNorm op for ``norm``.

    Gemma RMSNorm scales by ``(1 + weight)`` (see RMSNorm.forward), but the
    fused AR+RMSNorm kernels (and the NCCL / NCCL_SYMMETRIC fallbacks the
    AUTO strategy may pick) only apply ``weight``. Baking the ``+1`` into the
    weight makes EVERY allreduce backend produce the correct gemma result
    without any backend-specific flag.

    For gemma norms the ``(1 + weight)`` tensor is precomputed once in
    ``Qwen3NextForCausalLM.cache_derived_state`` and cached on the module as
    ``_fused_norm_weight``. Computing it inline here would re-run a cast+add
    elementwise kernel every forward inside the CUDA graph. The inline path
    below is only a correctness fallback if the cache is absent.
    """
    cached = getattr(norm, "_fused_norm_weight", None)
    if cached is not None:
        return cached
    w = norm.weight
    if getattr(norm, "use_gemma", False):
        return (w.float() + 1.0).to(w.dtype)
    return w


def _precompute_fused_norm_weights(module: nn.Module) -> None:
    """Bake ``(1 + weight)`` once for every gemma RMSNorm under ``module``.

    Caches the result on each norm as ``_fused_norm_weight`` so the fused
    AllReduce+RMSNorm path reads a ready tensor instead of recomputing the
    cast+add every forward. Non-gemma norms are left untouched (the fused op
    uses their ``weight`` directly). Must run after weights are loaded onto the
    device; the cached tensor is a plain attribute, not a registered buffer, so
    it stays out of the state dict.

    Norms whose ``weight`` was stripped are skipped: the layer-wise benchmark
    runs ``remove_weights`` on unused layers (``skip_forward``), leaving a
    ``use_gemma`` norm without a ``weight`` parameter; those layers never run
    the fused path, so there is nothing to precompute.
    """
    for norm in module.modules():
        if isinstance(norm, RMSNorm) and getattr(norm, "use_gemma", False):
            w = getattr(norm, "weight", None)
            if w is None:
                continue
            norm._fused_norm_weight = (w.float() + 1.0).to(w.dtype)


def _eager_fusion_enabled(enable_attention_dp: bool) -> bool:
    return (os.environ.get("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "0") == "0"
            and not enable_attention_dp)


# Qwen3.5 low-latency comm flags (default off).
# TRTLLM_Q35_AR_FP8_OUT=1: the fused all-reduce in front of a GDN / full-attention layer also emits
#   that layer's FP8 (E4M3, static per-tensor scale) projection input, bitwise what
#   static_quantize_e4m3_per_tensor makes of the BF16 norm output, so in_proj_qkvz / qkv_proj skip
#   their quantize kernel (GDN keeps the BF16 output for in_proj_ba).
# TRTLLM_Q35_AR_NORM_FUSE=1: fold the standalone RMSNorms that follow plain all-reduces into them
#   (embedding + first input_layernorm; per MTP pass: embedding + pre_fc_norm_embedding, fc +
#   input_layernorm, MoE + shared_head.norm).
# TRTLLM_Q35_REPLICATED_EMBED=1: full embedding table on every rank (+~1.8 GB/GPU), no all-reduce.
_Q35_AR_FP8_OUT = os.environ.get("TRTLLM_Q35_AR_FP8_OUT", "0") == "1"
_Q35_AR_NORM_FUSE = os.environ.get("TRTLLM_Q35_AR_NORM_FUSE", "0") == "1"
_Q35_REPLICATED_EMBED = os.environ.get("TRTLLM_Q35_REPLICATED_EMBED",
                                       "0") == "1"
# TRTLLM_Q35_EMBED_GATHER=1 (with TRTLLM_Q35_REPLICATED_EMBED): every target / MTP lookup of the full table
# is one gather launch instead of the 9-kernel masked lookup (bitwise).
_Q35_EMBED_GATHER = os.environ.get("TRTLLM_Q35_EMBED_GATHER", "0") == "1"


def _q35_mtp_fused_input() -> bool:
    """TRTLLM_Q35_MTP_FUSED_INPUT=1 (default off): see modules/q35_mtp_glue.py."""
    return os.environ.get("TRTLLM_Q35_MTP_FUSED_INPUT", "0") == "1"


# TRTLLM_Q35_TRITON_AR=se: the MoE-output (POST_MOE) all-reduce runs the Triton one-shot kernel of
#   distributed/q35_allreduce.py with the shared-expert combine (routed + sigmoid(gate) * shared)
#   folded into its input, removing the separate combine kernel; =all: additionally every other
#   fused all-reduce + RMSNorm (PRE_MOE, folds) on that kernel (one-shot also at 28 rows).
#   Decode shapes only (<= 64 rows, MNNVL); otherwise the library op.
# TRTLLM_Q35_FUSED_AR=se|all: same, on the CuTe DSL kernel (cute_dsl_kernels/blackwell/
#   q35_allreduce_norm.py: bitwise the library op, one-shot up to 64 rows); takes precedence.
_Q35_TRITON_AR = os.environ.get("TRTLLM_Q35_TRITON_AR", "")
_Q35_FUSED_AR = os.environ.get("TRTLLM_Q35_FUSED_AR", "")
# TRTLLM_Q35_FIN_FOLD=1 (with TRTLLM_Q35_FUSED_AR): the trtllm-gen NVFP4 MoE runs with
#   do_finalize=False and the CuTe all-reduce finalizes its permuted FC2 rows in-kernel (the finalize
#   kernel's arithmetic, bitwise), removing the finalize launch. TRTLLM_Q35_FIN_CHECK=1: in eager
#   calls, also run the MoE with do_finalize=True and compare (log once per layer).
_Q35_FIN_FOLD = os.environ.get("TRTLLM_Q35_FIN_FOLD", "0") == "1"
# TRTLLM_Q35_MTP_MOE_CALIB=<path> (debug, default off): MoE activation maxima for the NVFP4 MTP experts' activation
# scales (modules/q35_mtp_moe_quant.py).
_Q35_MTP_MOE_CALIB = bool(
    os.environ.get("TRTLLM_Q35_MTP_MOE_CALIB", "").strip())
_Q35_FIN_CHECK = os.environ.get("TRTLLM_Q35_FIN_CHECK", "0") == "1"
# TRTLLM_Q35_AR2=fe|fp|fpm|sl|slm (with TRTLLM_Q35_FUSED_AR): the one-shot fused all-reduces on the ar2 kernel
#   (distributed/q35_ar2.py); two-shot calls (TP16 >= 12 rows) stay on the CuTe kernel (slot two-shot: opt-in with
#   TRTLLM_Q35_AR2_SLOT_TWOSHOT=1). fe / fp: same protocol and outputs as the CuTe
#   kernel (fp: buffer rotation inside the poll loop); sl: flag-free slot protocol on its own workspace (same
#   outputs); m: st.async + mbarrier norm exchange. TRTLLM_Q35_AR2_SITES=all|moe|mixer limits it to the MoE-output
#   (fold) or the other calls.
_Q35_AR2 = os.environ.get("TRTLLM_Q35_AR2", "")
_Q35_AR2_SITES = os.environ.get("TRTLLM_Q35_AR2_SITES", "all")

# TRTLLM_Q35_FUSED_AR=all also takes the ARs without a fold (PRE_MOE) only while the library's MNNVL path would
# be one-shot (rows x hidden x TP x 2 B <= 1 MiB): above that the library two-shot + Lamport RMSNorm pair is the
# faster one (TP8 28 rows 6.34 us vs CuTe one-shot 6.93 / two-shot 7.28). TRTLLM_Q35_FUSED_AR_PLAIN_MAX_ROWS=<n>
# overrides the bound (A/B).
_Q35_FUSED_AR_PLAIN_MAX_ROWS = os.environ.get(
    "TRTLLM_Q35_FUSED_AR_PLAIN_MAX_ROWS")
# TRTLLM_Q35_FUSED_AR_MAX_ROWS=<n> (default 64; CuTe kernel only, i.e. with TRTLLM_Q35_FUSED_AR): largest call the
# fused all-reduces take. Above 64 rows (TEP8 b16 / b32 verify: 112 / 224 rows) the MoE-output all-reduce keeps the
# shared-expert combine and the finalize (TRTLLM_Q35_FIN_FOLD) in its launch instead of the library two-shot +
# Lamport RMSNorm pair behind separate finalize and combine kernels; pair it with TRTLLM_Q35_FUSED_AR_TWOSHOT_ROWS
# (the CuTe two-shot moves ~1/4 of the one-shot's NVLink ingress at TP8).
_Q35_FUSED_AR_MAX_ROWS = int(
    os.environ.get("TRTLLM_Q35_FUSED_AR_MAX_ROWS", "64"))


def _q35_triton_ar(allreduce, rows: int, is_post_moe: bool, hidden: int = 4096):
    """The in-kernel-fused all-reduce for this call (CuTe DSL or Triton), or None (library op)."""
    mode = _Q35_FUSED_AR or _Q35_TRITON_AR
    if mode not in ("se", "all") or (not is_post_moe and mode != "all"):
        return None
    max_rows = _Q35_FUSED_AR_MAX_ROWS if _Q35_FUSED_AR else 64
    if (allreduce is None or getattr(allreduce, "mnnvl_allreduce", None) is None
            or not 0 < rows <= max_rows):
        return None
    if not is_post_moe and _Q35_FUSED_AR:
        if _Q35_FUSED_AR_PLAIN_MAX_ROWS is not None:
            if rows > int(_Q35_FUSED_AR_PLAIN_MAX_ROWS):
                return None
        elif rows * hidden * allreduce.mapping.tp_size * 2 > 64 * 1024 * 8 * 2:
            return None
    if _Q35_FUSED_AR:
        from ..distributed.q35_allreduce import get_q35_cute_allreduce
        inst = get_q35_cute_allreduce(allreduce.mapping)
        if _Q35_AR2 and inst is not None and (_Q35_AR2_SITES == "all" or
                                              (_Q35_AR2_SITES
                                               == "moe") == is_post_moe):
            from ..distributed.q35_ar2 import get_q35_ar2
            inst = get_q35_ar2(allreduce.mapping, inst, _Q35_AR2) or inst
        return inst
    from ..distributed.q35_allreduce import get_q35_fused_allreduce
    return get_q35_fused_allreduce(allreduce.mapping)


def _q35_fin_fold(decoder_layer, rows: int) -> bool:
    """Run this layer's MoE with do_finalize=False and finalize inside the CuTe all-reduce?"""
    mlp = decoder_layer.mlp
    return (_Q35_FIN_FOLD and _Q35_FUSED_AR in ("se", "all")
            and isinstance(mlp, Qwen3NextSparseMoeBlock)
            and _q35_triton_ar(decoder_layer.allreduce, rows, True) is not None
            and mlp.q35_can_defer_finalize())


# TRTLLM_Q35_MTP_FC_CUTE_AR=1 (with TRTLLM_Q35_FUSED_AR=all and TRTLLM_Q35_AR_NORM_FUSE): the MTP fc all-reduce +
#   input_layernorm (zero residual) runs on the CuTe kernel instead of the library MNNVL one-shot, under the same
#   policy as the other fold-free ARs (only where the library would be one-shot; bitwise there).
_Q35_MTP_FC_CUTE_AR = os.environ.get("TRTLLM_Q35_MTP_FC_CUTE_AR", "0") == "1"

# TRTLLM_Q35_GEMM_AR=1: the attention o_proj / GDN out_proj (FP8 per-tensor on the low-M kernel) and the
# following all-reduce + residual + post_attention_layernorm run as one CuTe launch
# (distributed/q35_allreduce.Q35GemmArNorm): the projection defers itself to the decoder layer's fused norm
# all-reduce. Same projection values; the RMSNorm sums its squares in another order.
_Q35_GEMM_AR = os.environ.get("TRTLLM_Q35_GEMM_AR", "0") == "1"
# TRTLLM_Q35_GEMM_AR_MIN_ROWS=N: the fused path only for calls of at least N rows (e.g. 21 = b4 only with MTP 6:
# b1 / b2 / b4 verify 7 / 14 / 28 rows); every batch size is its own CUDA graph, so this is a per-batch policy.
_Q35_GEMM_AR_MIN_ROWS = int(os.environ.get("TRTLLM_Q35_GEMM_AR_MIN_ROWS", "1"))


class _Q35DeferredProj:
    """A row-parallel projection's input, handed to the decoder layer instead of its output."""

    def __init__(self, linear, x, args, kwargs):
        self.linear = linear
        self.x = x
        self.args = args
        self.kwargs = kwargs

    def materialize(self) -> torch.Tensor:
        return type(self.linear).forward(self.linear, self.x, *self.args,
                                         **self.kwargs)


def _q35_install_defer(linear) -> None:
    """Let ``linear`` hand its input back (once, when armed) instead of projecting it."""
    if linear.__dict__.get("_q35_defer_installed", False):
        return
    cls_forward = type(linear).forward

    def forward(input, *args, **kwargs):
        if linear.__dict__.get("_q35_defer_once", False):
            linear.__dict__["_q35_defer_once"] = False
            return _Q35DeferredProj(linear, input, args, kwargs)
        return cls_forward(linear, input, *args, **kwargs)

    linear.__dict__["forward"] = forward
    linear.__dict__["_q35_defer_installed"] = True


def _q35_gemm_ar_arm(decoder_layer, proj, rows: int, lora_params) -> None:
    """Arm ``proj`` to defer into the fused GEMM + all-reduce + norm when the call qualifies."""
    if (not _Q35_GEMM_AR or lora_params is not None
            or not decoder_layer.fusion_config.PRE_MOE_FUSION
            or not proj.__dict__.get("_q35_defer_installed", False)):
        return
    allreduce = decoder_layer.allreduce
    if (getattr(allreduce, "mnnvl_allreduce", None) is None
            or not 0 < rows <= 32 or rows < _Q35_GEMM_AR_MIN_ROWS):
        return
    from ..modules.lowm_linear import (lowm_fp8_act, lowm_fp8_enabled,
                                       lowm_rows_ok, lowm_skipped)
    label = getattr(proj, "_q35_lowm", False)
    if not label or not lowm_fp8_enabled() or lowm_skipped(label):
        return
    weight = getattr(proj, "weight", None)
    if (weight is None or weight.dtype != torch.float8_e4m3fn
            or getattr(proj, "input_scale", None) is None
            or getattr(proj, "force_dynamic_quantization", False)
            or getattr(proj, "bias", None) is not None):
        return
    if not lowm_rows_ok(rows, weight.shape[0], weight.shape[1], lowm_fp8_act()):
        return
    from ..distributed.q35_allreduce import get_q35_gemm_ar
    if get_q35_gemm_ar(allreduce.mapping) is None:
        return
    proj.__dict__["_q35_defer_once"] = True


def _q35_gemm_ar_run(allreduce, deferred: _Q35DeferredProj, residual, norm):
    """The fused projection + all-reduce + residual + RMSNorm, or None (caller materializes)."""
    from ..distributed.q35_allreduce import get_q35_gemm_ar
    fused = get_q35_gemm_ar(allreduce.mapping)
    x = deferred.x
    if fused is None or x.dim() < 2:
        return None
    x2 = x.reshape(-1, x.shape[-1])
    if (x2.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or x2.stride(-1) != 1 or x2.stride(0) % 16):
        return None
    lin = deferred.linear
    outs = fused(x2, lin.weight, lin.weight_scale, lin.input_scale, residual,
                 _fused_norm_weight(norm), norm.variance_epsilon)
    if outs is not None and not _q35_gemm_ar_logged:
        _q35_gemm_ar_logged.add(True)
        logger.info(
            f"[Q35 comm] TRTLLM_Q35_GEMM_AR: fused projection + all-reduce + norm in use "
            f"(first call: {x2.shape[0]} rows, K={x2.shape[1]}, {x2.dtype})")
    return outs


_q35_gemm_ar_logged: set = set()


def _q35_finalize_ref(fc2: torch.Tensor, weights: torch.Tensor,
                      perm: torch.Tensor, hidden: int) -> torch.Tensor:
    """The trtllm-gen finalize kernel's result (FP32 sum of w * row in k order, BF16 once)."""
    acc = torch.zeros((perm.shape[0], hidden),
                      dtype=torch.float32,
                      device=fc2.device)
    for k in range(perm.shape[1]):
        p = perm[:, k]
        rows = fc2[p.clamp(min=0).long(), :hidden].float()
        acc = torch.where((p >= 0)[:, None],
                          acc + weights[:, k:k + 1].float() * rows, acc)
    return acc.to(torch.bfloat16)


def _q35_fp8_input_linear(layer: nn.Module) -> Optional[Linear]:
    """The FP8 per-tensor (static scale) input projection of a decoder layer, or None."""
    if isinstance(layer, Qwen3NextLinearDecoderLayer):
        linear = layer.linear_attn.in_proj_qkvz
    elif isinstance(layer, Qwen3NextFullAttentionDecoderLayer):
        linear = layer.self_attn.qkv_proj
    else:
        return None
    if (not isinstance(getattr(linear, "quant_method", None),
                       FP8QDQLinearMethod)
            or getattr(linear, "input_scale", None) is None
            or getattr(linear, "force_dynamic_quantization", False)):
        return None
    return linear


def _q35_fused_norm_allreduce(allreduce,
                              hidden_states,
                              residual,
                              norm: RMSNorm,
                              next_layer: Optional[nn.Module],
                              shared_fold=None,
                              is_post_moe: bool = False,
                              se_moe: Optional[nn.Module] = None):
    """``allreduce`` + residual + ``norm`` (RESIDUAL_RMS_NORM), returning (hidden, residual).

    With TRTLLM_Q35_AR_FP8_OUT and an FP8 input projection in ``next_layer`` the same launch also
    quantizes the norm output with that projection's static input scale: before a GDN layer the
    BF16 output is kept (in_proj_ba) and the FP8 copy is handed to the layer as
    ``_q35_fp8_input``; before a full-attention layer the FP8 tensor replaces it (qkv_proj is the
    only consumer).

    ``shared_fold`` = (routed, shared, gate logits) of a MoE block that deferred its shared-expert
    combine (TRTLLM_Q35_TRITON_AR): the Triton kernel folds it into the all-reduce input.

    ``se_moe`` = the MoE block after this (pre-MoE) all-reduce: with TRTLLM_Q35_SE_AUX=fp8 the library
    launch also emits its shared expert's FP8 input (RESIDUAL_RMS_NORM_OUT_QUANT_FP8 with gate_up_proj's
    static input scale: 1 / scale times the BF16 norm output, E4M3 satfinite, as its quantize kernel) and
    hands it to the block as ``_q35_se_fp8_input``.
    """
    if isinstance(hidden_states, _Q35DeferredProj):
        if next_layer is None and shared_fold is None:
            outs = _q35_gemm_ar_run(allreduce, hidden_states, residual, norm)
            if outs is not None:
                return outs
        hidden_states = hidden_states.materialize()
    linear = (_q35_fp8_input_linear(next_layer)
              if _Q35_AR_FP8_OUT and next_layer is not None else None)
    rows = (shared_fold[1] if shared_fold is not None else residual).shape[0]
    triton_ar = _q35_triton_ar(allreduce, rows, is_post_moe, residual.shape[-1])
    if shared_fold is not None and isinstance(shared_fold[0], (tuple, list)):
        # Un-finalized MoE output: only the CuTe kernel takes it (the caller checked _q35_fin_fold).
        assert triton_ar is not None and _Q35_FUSED_AR
    if triton_ar is not None and hasattr(allreduce, "mapping"):
        gdn_next = isinstance(next_layer, Qwen3NextLinearDecoderLayer)
        outs = triton_ar(
            residual,
            _fused_norm_weight(norm),
            norm.variance_epsilon,
            x=hidden_states if shared_fold is None else None,
            shared_fold=shared_fold,
            scale=linear.input_scale if linear is not None else None,
            norm_out=linear is None or gdn_next)
        if linear is None:
            return outs
        if gdn_next:
            norm_out, quant_out, residual = outs
            next_layer.__dict__["_q35_fp8_input"] = quant_out
            return norm_out, residual
        return outs
    if shared_fold is not None:
        routed, shared, gate = shared_fold
        hidden_states = fused_sigmoid_gate_mul_add(
            routed, gate, shared, output=torch.empty_like(routed))
    params = AllReduceParams(fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                             residual=residual,
                             norm_weight=_fused_norm_weight(norm),
                             eps=norm.variance_epsilon)
    se_linear = (se_moe.q35_se_fp8_linear(rows) if se_moe is not None
                 and linear is None and shared_fold is None else None)
    if se_linear is not None:
        # TRTLLM_Q35_SE_AUX=fp8: norm output for the router / gate, FP8 copy for the shared expert.
        params.scale = se_linear.quant_method.get_static_input_scale(se_linear)
        params.fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8
        norm_out, quant_out, residual = allreduce(hidden_states,
                                                  all_reduce_params=params)
        se_moe.__dict__["_q35_se_fp8_input"] = quant_out
        return norm_out, residual
    if linear is None:
        return allreduce(hidden_states, all_reduce_params=params)
    params.scale = linear.input_scale
    if isinstance(next_layer, Qwen3NextLinearDecoderLayer):
        params.fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8
        norm_out, quant_out, residual = allreduce(hidden_states,
                                                  all_reduce_params=params)
        # A plain attribute (not a registered buffer); the next layer pops it.
        next_layer.__dict__["_q35_fp8_input"] = quant_out
        return norm_out, residual
    params.fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
    quant_out, residual = allreduce(hidden_states, all_reduce_params=params)
    return quant_out, residual


def _q35_pop_fp8_input(layer: nn.Module,
                       hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    fp8 = layer.__dict__.pop("_q35_fp8_input", None)
    if fp8 is not None and fp8.shape != hidden_states.shape:
        return None
    return fp8


def _experts_excluded_from_quant(model_config: ModelConfig[Qwen3NextConfig],
                                 layer_idx: Optional[int]) -> bool:
    """Is this layer's routed-experts module listed in ``exclude_modules``?

    ``exclude_modules`` is applied only after every module is built, but
    ``create_moe`` has to pick the MoE backend class before that. A backend
    picked for FP8/NVFP4 weights rejects the experts if they turn out to be
    bf16, so look the answer up now instead of waiting for that pass.
    """
    quant_config = model_config.quant_config
    if layer_idx is None or not quant_config.exclude_modules:
        return False
    candidates = [f"model.layers.{layer_idx}.mlp.experts"]
    n_hidden_layers = getattr(model_config.pretrained_config,
                              "num_hidden_layers", None)
    # One MTP layer has two names: mtp.layers.<i> in the checkpoint and
    # model.layers.<n_hidden_layers + i> at runtime. Only the Qwen3.5 entry
    # points rewrite the first into the second, so try both names here.
    if n_hidden_layers is not None and layer_idx >= n_hidden_layers:
        candidates.append(
            f"mtp.layers.{layer_idx - n_hidden_layers}.mlp.experts")
    return any(
        quant_config.is_module_excluded_from_quantization(candidate)
        for candidate in candidates)


class Qwen3NextGate(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dtype: Optional[torch.dtype] = None,
        apply_routing: bool = False,
        routing_method_type: RoutingMethodType = RoutingMethodType.Renormalize,
        moe_backend: str = "CUTLASS",
    ):
        super().__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size),
                                               dtype=dtype),
                                   requires_grad=False)
        self.routing_method_type = routing_method_type
        # FIXME: out_dtype=float32 does not work
        # self.out_dtype = torch.float32 if moe_backend == "TRTLLM" else dtype
        self.out_dtype = dtype
        self._q35_router_gemm = None

        assert not apply_routing, "Qwen3NextGate routing is called inside MoE"

    def q35_router_ok(self, hidden_states: torch.Tensor) -> bool:
        return (hidden_states.dim() == 2
                and hidden_states.shape[0] <= Q35_MOE_FUSED_FRONT_MAX_TOKENS
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.stride(1) == 1
                and self.weight.dtype == torch.bfloat16
                and self.out_dtype == torch.bfloat16)

    def q35_router(self,
                   hidden_states: torch.Tensor,
                   global_scale: Optional[torch.Tensor] = None,
                   route: Optional[tuple] = None):
        # Single-launch Triton router GEMM (TRTLLM_Q35_MOE_ROUTER_GEMM=1);
        # with global_scale it also returns the NVFP4 quantization of x, and with
        # route (TRTLLM_Q35_MOE_PREROUTE) also the routing and its tables.
        if route is not None:
            if self._q35_router_gemm is None:
                self._q35_router_gemm = Q35RouterGemm()
            return self._q35_router_gemm(hidden_states, self.weight,
                                         global_scale, route)
        if Q35_MOE_ROUTER_LOWM or (global_scale is None and Q35_MOE_GATE_LOWM):
            # TRTLLM_Q35_MOE_GATE_LOWM: plain (no-quant) router calls, e.g. the BF16 MTP layer's gate, on the
            # low-M GEMM
            out = q35_lowm_router(hidden_states, self.weight, global_scale)
            if out is not None:
                return out
        if self._q35_router_gemm is None:
            self._q35_router_gemm = Q35RouterGemm()
        return self._q35_router_gemm(hidden_states, self.weight, global_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if Q35_MOE_ROUTER_GEMM and self.q35_router_ok(hidden_states):
            return self.q35_router(hidden_states)
        logits: torch.Tensor = torch.ops.trtllm.cublas_mm(
            hidden_states, self.weight.t(), bias=None, out_dtype=self.out_dtype)
        return logits

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        assert len(weights) == 1
        weight = weights[0].get("weight")
        if not allow_partial_loading:
            assert weight is not None
        if weight is not None:
            self.weight.copy_(weight[:])

    @property
    def routing_method(self) -> BaseMoeRoutingMethod:
        if self.routing_method_type == RoutingMethodType.RenormalizeNaive:
            return RenormalizeNaiveMoeRoutingMethod(top_k=self.top_k)
        elif self.routing_method_type == RoutingMethodType.Renormalize:
            return RenormalizeMoeRoutingMethod(top_k=self.top_k)
        else:
            raise ValueError(
                f"Unsupported routing method: {self.routing_method_type}")


# TRTLLM_Q35_SE_AUX=gate|prio|fp8, combined with '+' (default off): the shared expert's chain on the aux stream
#   (FP8 quant of x -> gate_up -> SwiGLU -> down, then the shared_expert_gate GEMV hidden -> 1 and its cuBLAS split-K
#   reduce) next to the routed MoE. With a short MoE front (TRTLLM_Q35_MOE_FRONT_BIG) the routed FC1 launches early
#   (PDL) and fills the SMs: chain kernels launched after it wait for the FC1 / FC2 CTA queues to drain, and the
#   MoE-output all-reduce waits for the chain.
#   gate: the GEMV + reduce run on a second aux stream, in parallel with the rest of the chain (same kernels, bitwise;
#     CUDA-graph multi-stream only);
#   prio: the chain runs on its own aux stream(s) created at the highest stream priority, so its CTAs are dispatched
#     ahead of queued FC1 / FC2 CTAs (same kernels, bitwise; inside a CUDA graph this needs node priorities);
#   fp8: the pre-MoE all-reduce + RMSNorm (library launch) also writes the shared expert's FP8 input, so the chain
#     starts at gate_up without its quantize kernel (the same arithmetic, bitwise).
#   Use gate. Same-job decode A/B (TEP8 MTP-6, TRTLLM_Q35_MOE_FRONT_BIG=1, step ms): gate b8 -0.13..-0.22, b16 -0.11,
#   b32 -0.19..-0.21, greedy tokens identical; prio and gate+prio lose most of it (the high-priority chain takes SMs
#   from FC1 / the route launch); gate+fp8 is 0.02-0.07 slower than gate (gate_up then runs next to the router GEMM).
# TRTLLM_Q35_SE_AUX_MIN_ROWS=<n> (default 1): only calls of at least n rows (e.g. 112: the TEP8 b16 / b32 verify).
# TRTLLM_Q35_SE_AUX_CHECK=1 (debug, with fp8): eager calls compare the all-reduce's FP8 copy with the shared expert's
#   own quantization of x (bitwise; logged once per layer and row count, and on every mismatch).
_Q35_SE_AUX = frozenset(t for t in os.environ.get(
    "TRTLLM_Q35_SE_AUX", "").replace("+", ",").split(",") if t)
_Q35_SE_AUX_MIN_ROWS = int(os.environ.get("TRTLLM_Q35_SE_AUX_MIN_ROWS", "1"))
_Q35_SE_AUX_CHECK = os.environ.get("TRTLLM_Q35_SE_AUX_CHECK", "0") == "1"
_Q35_SE_STREAMS: Dict[tuple, torch.cuda.Stream] = {}
_q35_se_logged: set = set()


def _q35_se_stream(name: str) -> torch.cuda.Stream:
    """The shared-expert chain's own aux stream ``name`` (TRTLLM_Q35_SE_AUX), one per device; high priority with
    ``prio`` (the most negative priority torch hands out; the main stream has 0)."""
    key = (torch.cuda.current_device(), name)
    stream = _Q35_SE_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(priority=-8 if "prio" in _Q35_SE_AUX else 0)
        _Q35_SE_STREAMS[key] = stream
        prange = getattr(torch.cuda.Stream, "priority_range", None)
        logger.info(
            f"[q35 se aux] TRTLLM_Q35_SE_AUX={'+'.join(sorted(_Q35_SE_AUX))}: stream '{name}' on "
            f"device {key[0]} has priority {getattr(stream, 'priority', '?')} (torch range "
            f"{prange() if prange is not None else '?'})")
    return stream


class Qwen3NextSparseMoeBlock(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.mapping = model_config.mapping
        self._q35_layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy,
                                   dtype=config.torch_dtype)

        self.aux_stream = aux_stream

        self.gate = Qwen3NextGate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=config.torch_dtype,
            apply_routing=False,
            routing_method_type=RoutingMethodType.Renormalize,
            moe_backend=model_config.moe_backend,
        )

        # Qwen3.5 BF16 checkpoints store fused [gate_up_proj, down_proj] per
        # expert, while Qwen3Next and Qwen3.5 FP8 checkpoints store separate
        # gate_proj/up_proj/down_proj (vanilla).  The model_type is set by HF
        # config.json: "qwen3_5_moe_text" for Qwen3.5, "qwen3_next" otherwise.
        weight_loading_mode = (MoEWeightLoadingMode.FUSED_GATE_UP_PROJ
                               if config.model_type == "qwen3_5_moe_text" else
                               MoEWeightLoadingMode.VANILLA)
        # For MIXED_PRECISION checkpoints (e.g. ModelOpt), the experts' real
        # quant algo lives in the per-layer quant_config_dict keyed by TRT-LLM
        # module name. Pass it as override_quant_config so the backend resolves
        # the correct quant method (e.g. NVFP4) at construction, instead of
        # inheriting the unquantized MIXED_PRECISION global and relying on
        # apply_layerwise_quant_config (which rebinds the ConfigurableMoE
        # wrapper but not its delegated backend).
        expert_quant_config = None
        quant_config_dict = getattr(model_config, "quant_config_dict", None)
        if quant_config_dict and layer_idx is not None:
            expert_quant_config = quant_config_dict.get(
                f"model.layers.{layer_idx}.mlp.experts")
        moe_model_config = model_config
        stock_weight_loading_mode = weight_loading_mode
        # Q35 (opt-in): TRTLLM_Q35_MTP_MOE_QUANT=fp8block|mxfp4 quantizes the
        # MTP layer's bf16 experts at load time for TRTLLM-gen
        # (modules/q35_mtp_moe_quant.py). The NVFP4 checkpoint is
        # MIXED_PRECISION: its MTP experts have no per-layer quant entry (and
        # may or may not be listed in exclude_modules), i.e. they are bf16.
        q35_mtp_quant = None
        n_hidden = getattr(config, "num_hidden_layers", None)
        if layer_idx is not None and n_hidden is not None and layer_idx >= n_hidden:
            from ..modules.q35_mtp_moe_quant import (experts_unquantized,
                                                     mtp_moe_quant_mode,
                                                     quant_config_for)
            if (_experts_excluded_from_quant(model_config, layer_idx)
                    or experts_unquantized(model_config, expert_quant_config)):
                q35_mtp_quant = mtp_moe_quant_mode()
        if q35_mtp_quant is not None:
            expert_quant_config = quant_config_for(
                q35_mtp_quant, model_config.quant_config.kv_cache_quant_algo)
            weight_loading_mode = MoEWeightLoadingMode.VANILLA
            if model_config.moe_backend != "TRTLLM":
                moe_model_config = copy.copy(model_config)
                moe_model_config._frozen = False
                moe_model_config.moe_backend = "TRTLLM"
                moe_model_config._frozen = True
        elif _experts_excluded_from_quant(model_config, layer_idx):
            # These experts end up bf16 whatever the per-layer entry says, so
            # build them on CUTLASS, the only backend that serves bf16
            # unconditionally.
            expert_quant_config = QuantConfig(kv_cache_quant_algo=model_config.
                                              quant_config.kv_cache_quant_algo)
            # Q35 (opt-in): TRTLLM_Q35_MTP_MOE_BACKEND=TRTLLM runs these bf16
            # experts on TRTLLM-gen (FlashInfer trtllm_bf16_moe) instead.
            bf16_backend = ("TRTLLM" if os.environ.get(
                "TRTLLM_Q35_MTP_MOE_BACKEND", "").upper() == "TRTLLM" else
                            "CUTLASS")
            if model_config.moe_backend != bf16_backend:
                moe_model_config = copy.copy(model_config)
                moe_model_config._frozen = False
                moe_model_config.moe_backend = bf16_backend
                moe_model_config._frozen = True
                logger.warning(
                    f"Layer {layer_idx} MoE experts are excluded from "
                    f"quantization; using moe_backend={bf16_backend} for this "
                    f"layer (other layers keep {model_config.moe_backend}).")
        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=moe_model_config,
            layer_idx=layer_idx,
            weight_loading_mode=weight_loading_mode,
            override_quant_config=expert_quant_config,
        )
        self._q35_quant_max_m = 0
        self.q35_experts_bf16 = None
        if q35_mtp_quant is not None:
            from ..modules.q35_mtp_moe_quant import (mtp_moe_quant_max_m,
                                                     wrap_experts_loader)
            self._q35_quant_max_m = mtp_moe_quant_max_m()
            if self._q35_quant_max_m > 0:
                # Inputs above the cap (TRTLLM_Q35_MTP_MOE_QUANT_MAX_M) run on a
                # BF16 twin built as the stock MTP experts (CUTLASS).
                twin_config = copy.copy(model_config)
                twin_config._frozen = False
                twin_config.moe_backend = "CUTLASS"
                twin_config._frozen = True
                self.q35_experts_bf16 = create_moe(
                    num_experts=self.num_experts,
                    routing_method=self.gate.routing_method,
                    hidden_size=self.hidden_dim,
                    intermediate_size=self.moe_intermediate_size,
                    aux_stream_dict={
                        AuxStreamType.MoeChunkingOverlap: aux_stream
                    },
                    dtype=config.torch_dtype,
                    reduce_results=False,
                    model_config=twin_config,
                    # Not registered under this layer's index (that slot is the
                    # quantized experts'; no EPLB / torch.compile custom op).
                    layer_idx=None,
                    weight_loading_mode=stock_weight_loading_mode,
                    override_quant_config=QuantConfig(
                        kv_cache_quant_algo=model_config.quant_config.
                        kv_cache_quant_algo),
                )
            wrap_experts_loader(self.experts, q35_mtp_quant,
                                self.q35_experts_bf16)
            logger.info(
                f"Layer {layer_idx} (MTP) MoE experts: bf16 checkpoint "
                f"weights quantized at load time ({q35_mtp_quant}) for "
                "moe_backend=TRTLLM; inputs above "
                f"{self._q35_quant_max_m or 'no'} tokens on a BF16 twin")

        self.shared_expert = GatedMLP(
            hidden_size=self.hidden_dim,
            intermediate_size=config.shared_expert_intermediate_size,
            bias=config.mlp_bias if hasattr(config, 'mlp_bias') else False,
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=False,
            layer_idx=layer_idx,
            is_shared_expert=True,
            overridden_tp_size=1 if self.enable_attention_dp else None)

        self.shared_expert_gate = Linear(self.hidden_dim,
                                         1,
                                         bias=False,
                                         dtype=config.torch_dtype,
                                         quant_config=None)

        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.MoeShared]
        }
        # TRTLLM_Q35_SE_AUX: (chain stream, gate stream or None, x-ready event, gate-done event), else None.
        self._q35_se_aux = None
        if _Q35_SE_AUX:
            self._q35_se_aux = (
                _q35_se_stream("se") if "prio" in _Q35_SE_AUX else aux_stream,
                _q35_se_stream("gate") if "gate" in _Q35_SE_AUX else None,
                torch.cuda.Event(), torch.cuda.Event())
        # TRTLLM_Q35_LOWM_SE: (SwiGLU-paired gate_up weight, zero-padded gate weight), built on the
        # first eager decode-sized call; False when the shared expert does not qualify.
        self._se_lowm = None

    def post_load_weights(self) -> None:
        # Rebuilt from the final (rescaled) weights on the first eager decode-sized call; CUDA-graph
        # capture is always preceded by eager warmup forwards.
        self._se_lowm = None

    def q35_se_fp8_linear(self, rows: int) -> Optional[Linear]:
        """TRTLLM_Q35_SE_AUX=fp8: the shared expert's gate_up_proj when the pre-MoE all-reduce should also emit
        its FP8 input for a call of ``rows`` rows, else None (the low-M shared expert takes BF16 x)."""
        if ("fp8" not in _Q35_SE_AUX or rows < _Q35_SE_AUX_MIN_ROWS
                or self.enable_attention_dp):
            return None
        mlp = self.shared_expert
        gate_up = getattr(mlp, "gate_up_proj", None)
        down = getattr(mlp, "down_proj", None)
        if (gate_up is None or down is None
                or not isinstance(gate_up.quant_method, FP8QDQLinearMethod)
                or gate_up.quant_method.get_static_input_scale(gate_up) is None
                or mlp._can_fuse_gate_up_swiglu()
                or mlp._can_fuse_gate_up_swiglu_fp4out()):
            return None
        if (lowm_se_enabled() and self._se_lowm is not False
                and not lowm_skipped("se", rows)
                and lowm_rows_ok(rows, gate_up.weight.shape[0],
                                 gate_up.weight.shape[1], "quant")
                and lowm_rows_ok(rows, down.weight.shape[0],
                                 down.weight.shape[1], "fp8")):
            return None
        return gate_up

    def _q35_se_check(self, x: torch.Tensor, x_fp8: torch.Tensor) -> None:
        """TRTLLM_Q35_SE_AUX_CHECK: the all-reduce's FP8 copy vs gate_up_proj's own static quantization of x."""
        gate_up = self.shared_expert.gate_up_proj
        ref, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, gate_up.quant_method.get_static_input_scale(gate_up))
        diff = int((ref.view(torch.uint8)
                    != x_fp8.view(torch.uint8)).sum().item())
        key = (self._q35_layer_idx, x.shape[0])
        if key not in _q35_se_logged or diff:
            _q35_se_logged.add(key)
            logger.info(
                f"[q35 se aux] fp8 check layer {self._q35_layer_idx} rows {x.shape[0]}: all-reduce FP8 "
                f"copy {'== (bitwise)' if diff == 0 else '!='} static_quantize_e4m3_per_tensor(x) "
                f"({diff} of {ref.numel()} bytes differ)")

    def _prepare_shared_expert_lowm(self) -> None:
        """Weights of the shared expert's low-M path (``TRTLLM_Q35_LOWM_SE=1``).

        Decode-sized batches run the FP8 (W8A8) shared expert as two low-M launches instead of seven
        kernels (input quantize, gate_up GEMM, split-K reduce, SwiGLU + quantize, down GEMM, the
        shared_expert_gate GEMV and its reduce): (1) gate_up with the SwiGLU in the epilogue (FP8
        output quantized with down_proj.input_scale) plus shared_expert_gate as a BF16 segment;
        (2) down_proj on that FP8 output. The gate_up rows are re-paired for the epilogue (a copy;
        the original stays for larger batches).
        """
        self._se_lowm = False
        mlp = self.shared_expert
        gate_up = getattr(mlp, "gate_up_proj", None)
        down = getattr(mlp, "down_proj", None)
        gate = self.shared_expert_gate
        if gate_up is None or down is None:
            return
        # LoRA: GatedMLP always attaches LoRA layers (applied only with lora_params, which keep the
        # stock path in _compute_shared_output), so they do not disqualify the low-M path.
        for proj in (gate_up, down):
            if (not isinstance(proj.quant_method, FP8QDQLinearMethod)
                    or proj.bias is not None
                    or proj.quant_method.get_static_input_scale(proj) is None):
                return
        if (mlp.activation is not torch.nn.functional.silu
                or not mlp._is_plain_swiglu() or mlp.swiglu_limit is not None
                or gate.weight.dtype != torch.bfloat16
                or gate.weight.shape[0] != 1 or gate.bias is not None):
            return
        inter, k = gate_up.weight.shape[0] // 2, gate_up.weight.shape[1]
        if inter % 8 or k % 64 or down.weight.shape[1] % 64:
            return
        gate_w = torch.zeros((8, k),
                             dtype=gate.weight.dtype,
                             device=gate.weight.device)
        gate_w[:1].copy_(gate.weight)
        self._se_lowm = (swiglu_pairs(gate_up.weight.data), gate_w)

    def _shared_expert_lowm(self, x: torch.Tensor):
        """(shared expert output, gate logits [M, 1]) from two low-M launches, or None."""
        if (not lowm_se_enabled() or x.dim() != 2
                or lowm_skipped("se", x.shape[0]) or x.dtype != torch.bfloat16):
            return None
        gate_up_w = self.shared_expert.gate_up_proj.weight
        down_w = self.shared_expert.down_proj.weight
        if not (lowm_rows_ok(x.shape[0], gate_up_w.shape[0], x.shape[1],
                             "quant")
                and lowm_rows_ok(x.shape[0], down_w.shape[0], down_w.shape[1],
                                 "fp8")):
            return None
        if self._se_lowm is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            self._prepare_shared_expert_lowm()
        if not self._se_lowm:
            return None
        pairs, gate_w = self._se_lowm
        mlp = self.shared_expert
        gate_up, down = mlp.gate_up_proj, mlp.down_proj
        inter = pairs.shape[0] // 2
        down_scale = down.quant_method.get_static_input_scale(down)
        hidden, gate_logits = lowm_gemm(
            x16=x,
            w_fp8=pairs,
            weight_scale=gate_up.weight_scale,
            input_scale=gate_up.quant_method.get_static_input_scale(gate_up),
            swiglu_fp8=True,
            w_bf16=gate_w,
            split_out=inter,
            n_valid=inter + 1,
            out_scale=down_scale,
            out_dtype=x.dtype,
        )
        out = lowm_gemm(x8=hidden,
                        w_fp8=down.weight,
                        weight_scale=down.weight_scale,
                        input_scale=down_scale,
                        fp8_act="fp8",
                        out_dtype=x.dtype)
        lowm_check("fp8:se(gate_up+swiglu+down)", out,
                   lambda: self.shared_expert(x))
        lowm_check("bf16:se.gate(fused)", gate_logits,
                   lambda: self.shared_expert_gate(x))
        return out, gate_logits

    def q35_can_defer_finalize(self) -> bool:
        """trtllm-gen NVFP4 experts with in-kernel routing (their do_finalize=False output carries
        valid top-k weights) and no MoE communication / load balancer."""
        cached = self.__dict__.get("_q35_can_defer_finalize")
        if cached is None:
            from ..modules.fused_moe import moe_scheduler
            from ..modules.fused_moe.fused_moe_trtllm_gen import \
                TRTLLMGenFusedMoE
            experts = self.experts
            backend = getattr(experts, "backend", experts)
            routing = getattr(experts, "routing_method", None)
            cached = (
                isinstance(backend, TRTLLMGenFusedMoE)
                and bool(getattr(backend, "has_nvfp4", False))
                # TRTLLM_Q35_MTP_MOE_QUANT_MAX_M: the BF16 twin (CUTLASS)
                # always finalizes, so a layer with one never defers.
                and getattr(self, "q35_experts_bf16", None) is None and
                not self.enable_attention_dp and
                not backend._supports_load_balancer() and
                getattr(experts, "comm", None) is None and
                getattr(experts, "layer_load_balancer", None) is None and
                routing is not None and not routing.requires_separated_routing
                and not moe_scheduler.FORCE_SEPARATED_ROUTING)
            self.__dict__["_q35_can_defer_finalize"] = cached
        return cached

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
        lora_params: Optional[dict] = None,
        q35_defer_shared_combine: bool = False,
        q35_defer_finalize: bool = False,
    ) -> torch.Tensor:
        """``q35_defer_shared_combine`` (TRTLLM_Q35_TRITON_AR, caller does the all-reduce): return
        (routed, shared, gate logits) instead of routed + sigmoid(gate) * shared.
        ``q35_defer_finalize`` (TRTLLM_Q35_FIN_FOLD, with the former): ``routed`` is the experts'
        do_finalize=False output (fc2 rows, top-k weights, expanded -> permuted index)."""
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        use_dp_padding = False
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.enable_attention_dp and self.mapping.tp_size > 1 and get_sm_version(
        ) == 120:
            use_dp_padding = True
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0]))

        if not do_finalize:
            # TODO: support do_finalize == False
            raise NotImplementedError(
                "do_finalize == False is not supported yet")

        # TRTLLM_Q35_MTP_MOE_QUANT_MAX_M: the quantized MTP experts only up to
        # the cap; larger inputs take the BF16 twin (static per CUDA graph). A
        # layer with a twin never defers its finalize (_q35_can_defer_finalize).
        experts = self.experts
        if (self.q35_experts_bf16 is not None
                and hidden_states.shape[0] > self._q35_quant_max_m):
            experts = self.q35_experts_bf16
        experts_finalize = do_finalize and not (q35_defer_finalize
                                                and q35_defer_shared_combine
                                                and lora_params is None)

        def _compute_routed_output():
            if _Q35_MTP_MOE_CALIB:
                # Debug (TRTLLM_Q35_MTP_MOE_CALIB): activation maxima of every MoE layer's input, and of the
                # SwiGLU output of layers with BF16 experts (the stock MTP layer; needs their router logits).
                from ..modules.q35_mtp_moe_quant import mtp_moe_calib_record
                w31 = getattr(getattr(experts, "backend", experts),
                              "w3_w1_weight", None)
                calib_logits = (self.gate(hidden_states) if w31 is not None
                                and w31.dtype == torch.bfloat16 else None)
                mtp_moe_calib_record(self._q35_layer_idx, experts,
                                     hidden_states, calib_logits, self.top_k)
            if Q35_MOE_FUSED_FRONT or Q35_MOE_ROUTER_GEMM or Q35_MOE_FRONT_BIG:
                out = q35_routed_forward(
                    self.gate,
                    experts,
                    hidden_states,
                    all_rank_num_tokens=all_rank_num_tokens,
                    use_dp_padding=use_dp_padding,
                    do_finalize=experts_finalize,
                )
                if out is not None:
                    return out
            router_logits = self.gate(hidden_states)
            if Q35_MOE_STATS:
                q35_moe_stats_record(self._q35_layer_idx, router_logits,
                                     self.top_k)
            final_hidden_states = experts(
                hidden_states,
                router_logits,
                output_dtype=hidden_states.dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                do_finalize=experts_finalize,
            )
            if not experts_finalize:
                final_hidden_states = tuple(final_hidden_states)
                if _Q35_FIN_CHECK and not torch.cuda.is_current_stream_capturing(
                ):
                    self._q35_check_finalize(hidden_states, router_logits,
                                             all_rank_num_tokens,
                                             use_dp_padding,
                                             final_hidden_states)

            return final_hidden_states

        se_aux = (self._q35_se_aux if self._q35_se_aux is not None
                  and hidden_states.shape[0] >= _Q35_SE_AUX_MIN_ROWS else None)
        # TRTLLM_Q35_SE_AUX=fp8: the shared expert's FP8 input from the pre-MoE all-reduce (set by
        # _q35_fused_norm_allreduce for this call; popped either way).
        se_x = self.__dict__.pop("_q35_se_fp8_input", None)
        if se_x is None or se_x.shape != hidden_states.shape or lora_params:
            se_x = hidden_states
        elif _Q35_SE_AUX_CHECK and not torch.cuda.is_current_stream_capturing():
            self._q35_se_check(hidden_states, se_x)

        def _compute_shared_output():
            if not lora_params:
                lowm = self._shared_expert_lowm(hidden_states)
                if lowm is not None:
                    return lowm
            if (se_aux is not None and se_aux[1] is not None
                    and do_multi_stream()
                    and not torch.compiler.is_compiling()):
                # TRTLLM_Q35_SE_AUX=gate: the gate GEMV + split-K reduce on their own stream, beside the shared
                # expert (this runs on the chain's aux stream, after x is ready).
                _, gate_stream, x_ready, gate_done = se_aux
                x_ready.record()
                with torch.cuda.stream(gate_stream):
                    x_ready.wait()
                    shared_expert_gate_logits = self.shared_expert_gate(
                        hidden_states)
                    gate_done.record()
                shared_expert_output = self.shared_expert(
                    se_x,
                    lora_params=lora_params,
                )
                gate_done.wait()
                return shared_expert_output, shared_expert_gate_logits
            shared_expert_output = self.shared_expert(
                se_x,
                lora_params=lora_params,
            )
            shared_expert_gate_logits = self.shared_expert_gate(hidden_states)
            return shared_expert_output, shared_expert_gate_logits

        final_hidden_states, shared_expert_outputs = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            se_aux[0] if se_aux is not None else self.aux_stream,
            disable_on_compile=True,
        )
        if not do_finalize:
            return final_hidden_states

        shared_expert_output, shared_expert_gate_logits = shared_expert_outputs
        if q35_defer_shared_combine:
            return (final_hidden_states, shared_expert_output,
                    shared_expert_gate_logits)
        assert experts_finalize
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            output_tensor, _ = torch.ops.trtllm.allocate_output(
                final_hidden_states, self.allreduce.output_buffer_kind,
                self.mapping.tp_group)
            final_hidden_states = fused_sigmoid_gate_mul_add(
                final_hidden_states,
                shared_expert_gate_logits,
                shared_expert_output,
                output=output_tensor,
            )
            final_hidden_states = self.allreduce(
                final_hidden_states, all_reduce_params=all_reduce_params)
        else:
            final_hidden_states = fused_sigmoid_gate_mul_add(
                final_hidden_states, shared_expert_gate_logits,
                shared_expert_output)
        return final_hidden_states.view(orig_shape)

    def _q35_check_finalize(self, hidden_states, router_logits,
                            all_rank_num_tokens, use_dp_padding, unfinalized):
        """TRTLLM_Q35_FIN_CHECK: the do_finalize=False operands, finalized as the trtllm-gen
        finalize kernel does, must equal the experts' own finalized output (bitwise)."""
        ref = self.experts(hidden_states,
                           router_logits,
                           output_dtype=hidden_states.dtype,
                           all_rank_num_tokens=all_rank_num_tokens,
                           use_dp_padding=use_dp_padding,
                           do_finalize=True)
        fc2, weights, perm = unfinalized
        emu = _q35_finalize_ref(fc2, weights, perm, self.hidden_dim)
        same = torch.equal(emu, ref[:, :self.hidden_dim])
        key = (id(self), hidden_states.shape[0])
        logged = Qwen3NextSparseMoeBlock.__dict__.get("_q35_fin_logged")
        if logged is None:
            logged = set()
            Qwen3NextSparseMoeBlock._q35_fin_logged = logged
        if key not in logged or not same:
            logged.add(key)
            diff = (emu.float() - ref[:, :self.hidden_dim].float()).abs().max()
            logger.info(
                f"[q35 fin fold] rows={hidden_states.shape[0]} fc2={tuple(fc2.shape)} "
                f"weights={weights.dtype}{tuple(weights.shape)} local={(perm >= 0).sum().item()}"
                f"/{perm.numel()}: finalize emulation {'== (bitwise)' if same else '!='} "
                f"do_finalize=True output (max abs diff {diff.item():.3g})")


class _DenseMlpAdapter(nn.Module):
    """Wraps GatedMLP to match Qwen3NextSparseMoeBlock's forward interface.

    This allows the decoder layer forward methods to call self.mlp(...)
    with the same arguments regardless of whether MoE or dense MLP is used.
    """

    def __init__(self, mlp: GatedMLP, mapping):
        super().__init__()
        self.mlp = mlp
        self.mapping = mapping
        self.enable_attention_dp = mapping.enable_attention_dp
        # Provide a dummy `experts` attribute so that
        # `self.mlp.experts.has_nvfp4` checks in decoder forward don't crash.
        self.experts = SimpleNamespace(has_nvfp4=False)

    def forward(
        self,
        hidden_states,
        attn_metadata=None,
        all_reduce_params=None,
        do_finalize=True,
        lora_params=None,
    ):
        all_rank_num_tokens = (attn_metadata.all_rank_num_tokens
                               if attn_metadata is not None else None)
        return self.mlp(
            hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            final_all_reduce_params=all_reduce_params,
            lora_params=lora_params,
        )


def _create_mlp(model_config, aux_stream, layer_idx):
    """Create the appropriate MLP for this layer: MoE or dense GatedMLP."""
    config = model_config.pretrained_config
    num_experts = getattr(config, "num_experts", 0) or 0
    if num_experts > 0:
        return Qwen3NextSparseMoeBlock(model_config,
                                       aux_stream,
                                       layer_idx=layer_idx)
    return _DenseMlpAdapter(
        GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=getattr(config, "mlp_bias", False),
            dtype=config.torch_dtype,
            overridden_tp_size=1
            if model_config.mapping.enable_attention_dp else None,
            config=model_config,
        ),
        model_config.mapping,
    )


class Qwen3NextLinearDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config
        self.linear_attn = Qwen3NextGatedDeltaNet(model_config, aux_stream,
                                                  layer_idx)

        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = _create_mlp(model_config, aux_stream, layer_idx)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype,
                                       use_gemma=True)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype,
                                                use_gemma=True)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy,
                                   dtype=config.torch_dtype)

        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = _eager_fusion_enabled(self.enable_attention_dp)

        has_tp = self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)

        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        h_fp8 = _q35_pop_fp8_input(self, hidden_states)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False
        # Linear Attention
        ### FIXME: 1. forward_batch; 2. allreduce
        if hidden_states.shape[0] != 0:
            if h_fp8 is not None:
                kwargs["h_fp8"] = h_fp8
            if _Q35_GEMM_AR:
                _q35_install_defer(self.linear_attn.out_proj)
                _q35_gemm_ar_arm(self, self.linear_attn.out_proj,
                                 hidden_states.shape[0], lora_params)
            hidden_states = self.linear_attn(
                hidden_states,
                attn_metadata,
                spec_metadata=spec_metadata,
                all_reduce_params=AllReduceParams(
                    enable_allreduce=not self.disable_attn_allreduce),
                **kwargs)
        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = _q35_fused_norm_allreduce(
                self.allreduce,
                hidden_states,
                residual,
                self.post_attention_layernorm,
                None,
                se_moe=self.mlp if lora_params is None
                and isinstance(self.mlp, Qwen3NextSparseMoeBlock) else None)
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # Qwen3NextSparseMoeBlock does not implement do_finalize=False. Defer
        # only its final all-reduce so the decoder can fuse it with RMSNorm.
        do_finalize = True

        defer_combine = (self.fusion_config.POST_MOE_FUSION and isinstance(
            self.mlp, Qwen3NextSparseMoeBlock) and _q35_triton_ar(
                self.allreduce, hidden_states.shape[0], True) is not None)
        defer_kwargs = {}
        if defer_combine:
            defer_kwargs["q35_defer_shared_combine"] = True
            if lora_params is None and _q35_fin_fold(self,
                                                     hidden_states.shape[0]):
                defer_kwargs["q35_defer_finalize"] = True
        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
            lora_params=lora_params,
            **defer_kwargs,
        )

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = _q35_fused_norm_allreduce(
                    self.allreduce,
                    None if defer_combine else hidden_states,
                    residual,
                    self.next_layer_layernorm,
                    self.__dict__.get("_q35_next_layer"),
                    shared_fold=hidden_states if defer_combine else None,
                    is_post_moe=True)
            else:
                assert len(
                    hidden_states
                ) == 3, f"hidden_states must have 3 elements, but got {len(hidden_states)}"

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params)

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)
        return hidden_states, residual


class Qwen3NextAttention(Qwen3Attention):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int, fuse_qk_norm_rope: bool):
        super().__init__(model_config,
                         layer_idx,
                         fuse_qk_norm_rope=fuse_qk_norm_rope,
                         attn_output_gate=True,
                         use_gemma_rms_norm=True)
        self._fuse_qk_norm_rope_gate = True
        # Decode-sized projections on the low-M kernel (TRTLLM_Q35_LOWM_FP8 for the FP8 target
        # layers, TRTLLM_Q35_LOWM_BF16 for the BF16 MTP layer; read per call, inert when off).
        for proj, name in ((self.qkv_proj, "attn.qkv_proj"), (self.o_proj,
                                                              "attn.o_proj")):
            proj._q35_lowm = name
            proj._q35_lowm_bf16 = name


class Qwen3NextFullAttentionDecoderLayer(DecoderLayer):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int, aux_stream: torch.cuda.Stream):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.self_attn = Qwen3NextAttention(
            model_config,
            layer_idx=layer_idx,
            # Gemma-style QK-norm is now supported by the fused qk_norm_rope
            # kernel (use_gemma path), so fuse instead of running separate
            # split + q/k RMSNorm + RoPE kernels.
            fuse_qk_norm_rope=True,
        )

        self.mlp = _create_mlp(model_config, aux_stream, layer_idx)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype,
                                       use_gemma=True)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype,
                                                use_gemma=True)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy,
                                   dtype=config.torch_dtype)

        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = _eager_fusion_enabled(self.enable_attention_dp)

        has_tp = self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp

        # POST_MOE_FUSION fuses the MoE-output all-reduce with the next layer's
        # RMSNorm. It is a tensor-parallel (TEP) optimization: it is only valid
        # when ranks share the same tokens (not attention_dp, where each rank holds
        # different tokens and the MoE block does no cross-rank all-reduce). This
        # mirrors the DeepSeek-V3 pattern (POST == PRE in the non-attention_dp path).
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        # When PRE_MOE_FUSION is on, the attention all-reduce is deferred to the
        # fused PRE all-reduce+RMSNorm, so disable the in-attention all-reduce to
        # avoid reducing twice.
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)
        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False

        # Self Attention
        if _Q35_GEMM_AR and hidden_states.shape[0] != 0:
            _q35_install_defer(self.self_attn.o_proj)
            _q35_gemm_ar_arm(self, self.self_attn.o_proj,
                             hidden_states.shape[0], lora_params)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_attn_allreduce),
            lora_params=lora_params,
            **kwargs,
        )

        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = _q35_fused_norm_allreduce(
                self.allreduce,
                hidden_states,
                residual,
                self.post_attention_layernorm,
                None,
                se_moe=self.mlp if lora_params is None
                and isinstance(self.mlp, Qwen3NextSparseMoeBlock) else None)
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # The fully-fused do_finalize=False MoE path (MoEAllReduce on the
        # unfinalized expert output) is not implemented by Qwen3NextSparseMoeBlock
        # (it raises NotImplementedError). Keep do_finalize=True so POST_MOE_FUSION
        # still fuses the *finalized* MoE all-reduce with the next layer's RMSNorm
        # via the do_finalize branch below, without hitting the unimplemented path.
        do_finalize = True
        defer_combine = (self.fusion_config.POST_MOE_FUSION and isinstance(
            self.mlp, Qwen3NextSparseMoeBlock) and _q35_triton_ar(
                self.allreduce, hidden_states.shape[0], True) is not None)
        defer_kwargs = {}
        if defer_combine:
            defer_kwargs["q35_defer_shared_combine"] = True
            if lora_params is None and _q35_fin_fold(self,
                                                     hidden_states.shape[0]):
                defer_kwargs["q35_defer_finalize"] = True
        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
            lora_params=lora_params,
            **defer_kwargs,
        )

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = _q35_fused_norm_allreduce(
                    self.allreduce,
                    None if defer_combine else hidden_states,
                    residual,
                    self.next_layer_layernorm,
                    self.__dict__.get("_q35_next_layer"),
                    shared_fold=hidden_states if defer_combine else None,
                    is_post_moe=True)
            else:
                assert len(
                    hidden_states
                ) == 3, f"hidden_states must have 3 elements, but got {len(hidden_states)}"

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params)

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)

        return hidden_states, residual


class Qwen3NextMTPHead(nn.Module):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.mapping_lm_head_tp = None

    @torch.compile(options={"max-autotune": True})
    def get_last_token_states(self, hidden_states, attn_metadata):
        last_tokens = torch.cumsum(
            attn_metadata.seq_lens_cuda,
            dim=0,
            dtype=torch.long,
        ) - 1
        return hidden_states[last_tokens]

    def forward(self,
                hidden_states: torch.Tensor,
                lm_head: Linear,
                attn_metadata: AttentionMetadata,
                return_context_logits: bool = False) -> torch.Tensor:
        if not return_context_logits:
            if attn_metadata is not None:
                hidden_states = self.get_last_token_states(
                    hidden_states, attn_metadata)
            else:
                hidden_states = hidden_states[-1].unsqueeze(0)

        enable_attention_dp = self.model_config.mapping.enable_attention_dp
        enable_lm_head_tp_in_adp = enable_attention_dp and self.model_config.mapping.enable_lm_head_tp_in_adp

        if enable_lm_head_tp_in_adp:
            self.mapping_lm_head_tp = create_lm_head_tp_mapping(
                self.model_config.mapping, hidden_states.shape[0])
            hidden_states = allgather(hidden_states,
                                      self.mapping_lm_head_tp,
                                      dim=0)

        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = False
        logits = lm_head(hidden_states,
                         mapping_lm_head_tp=self.mapping_lm_head_tp,
                         is_spec_decoding_head=True)
        if not enable_attention_dp or enable_lm_head_tp_in_adp:
            lm_head.gather_output = True
        return logits


class Qwen3NextMTP(Qwen3NextFullAttentionDecoderLayer):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        # Some HF checkpoints (e.g. Qwen3.5 NVFP4) keep the whole MTP layer in
        # bf16. Qwen3NextSparseMoeBlock handles that for every layer, MTP
        # included, so nothing layer-specific is needed here.
        super().__init__(model_config, layer_idx,
                         aux_stream_dict[AuxStreamType.Attention])
        config = model_config.pretrained_config
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.MoeShared]
        }

        self.pre_fc_norm_embedding = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.pre_fc_norm_hidden = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )

        if model_config.mapping.enable_attention_dp:
            self.fc = Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                use_cute_dsl_blockscaling_mm=False,
            )
        else:
            self.fc = Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                use_cute_dsl_blockscaling_mm=False,
            )
        self.fc._q35_lowm_bf16 = "mtp.fc"  # TRTLLM_Q35_LOWM_BF16 (strided input rows are fine)
        self.shared_head = Qwen3NextMTPHead(model_config)
        # MTP applies shared_head.norm after the base decoder forward, so its
        # MoE-output all-reduce cannot consume next_layer_layernorm.
        self.fusion_config.POST_MOE_FUSION = False
        # TRTLLM_Q35_DRAFT_ATTN_WINDOW=W (default 0 = off; experiment): the
        # draft layer's decode attention reads only the last W keys (sliding
        # window); prefill and the KV cache are unchanged. Changes the draft
        # distribution: gate with the long-context AL check.
        q35_window = int(
            os.environ.get("TRTLLM_Q35_DRAFT_ATTN_WINDOW", "0") or 0)
        if q35_window > 0:
            self.self_attn.attn.q35_draft_window = q35_window
            logger.info(f"[Q35] MTP layer {layer_idx}: draft decode attention "
                        f"window {q35_window} keys")
        if _Q35_AR_NORM_FUSE and self.fusion_config.PRE_MOE_FUSION:
            # TRTLLM_Q35_AR_NORM_FUSE: the MoE-output all-reduce does residual + shared_head.norm
            # itself (RESIDUAL_RMS_NORM), and forward skips the separate FusedAddRMSNorm. A plain
            # attribute, so the norm is not registered a second time as a submodule.
            self.fusion_config.POST_MOE_FUSION = not model_config.mapping.has_pp(
            )
            self.__dict__["next_layer_layernorm"] = self.shared_head.norm

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        del all_rank_num_tokens
        tp_size = self.model_config.mapping.tp_size
        tp_rank = self.model_config.mapping.tp_rank
        chunked = tp_size > 1 and not self.model_config.mapping.enable_attention_dp
        # TRTLLM_Q35_DRAFT_BOOKKEEPING: a per-pass ``kv_lens_cuda[:n] += 1`` the
        # draft loop deferred to this pass (folded into the fused input launch
        # below, else applied here before anything reads kv_lens).
        kv_inc = getattr(attn_metadata, "q35_pending_kv_inc", 0)
        if kv_inc:
            attn_metadata.q35_pending_kv_inc = 0
        # TRTLLM_Q35_MTP_FUSED_INPUT (default off): with a replicated embedding
        # table, the masked embedding gather, both pre-fc Gemma RMSNorms, the
        # concat and the row-parallel chunk in one launch per pass
        # (modules/q35_mtp_glue.py).
        if (_q35_mtp_fused_input() and getattr(embed_tokens, "tp_size", 1) == 1
                and embed_tokens.weight.dtype == hidden_states.dtype
                and hidden_states.stride(-1) == 1):
            from ..modules.q35_mtp_glue import mtp_fused_input
            hidden_states = mtp_fused_input(
                input_ids,
                embed_tokens.weight,
                hidden_states,
                self.pre_fc_norm_embedding.weight,
                self.pre_fc_norm_hidden.weight,
                self.pre_fc_norm_embedding.variance_epsilon,
                self.pre_fc_norm_hidden.variance_epsilon,
                tp_size if chunked else 1,
                tp_rank if chunked else 0,
                kv_lens=attn_metadata.kv_lens_cuda if kv_inc else None,
                kv_rows=kv_inc)
        else:
            if kv_inc:
                attn_metadata.kv_lens_cuda[:kv_inc] += 1
            hidden_states = self._q35_stock_fc_input(input_ids, hidden_states,
                                                     embed_tokens, chunked,
                                                     tp_size, tp_rank)

        residual = None
        fc_zero = None
        if (_Q35_AR_NORM_FUSE
                and self.fc.all_reduce is not None and q35_mnnvl_fusable(
                    self.fc.all_reduce, hidden_states.shape[0])):
            fc_zero = q35_zero_residual(
                (hidden_states.shape[0], self.fc.out_features),
                hidden_states.dtype, hidden_states.device)
        fc_cute = None
        if fc_zero is not None and _Q35_MTP_FC_CUTE_AR and _Q35_FUSED_AR == "all":
            fc_cute = _q35_triton_ar(self.fc.all_reduce, hidden_states.shape[0],
                                     False, self.fc.out_features)
        if fc_cute is not None:
            # TRTLLM_Q35_MTP_FC_CUTE_AR: this rank's fc partial, then the CuTe all-reduce +
            # input_layernorm with the zero residual (residual output = the fc sum).
            partial = self.fc(
                hidden_states,
                all_reduce_params=AllReduceParams(enable_allreduce=False))
            hidden_states, residual = fc_cute(
                fc_zero,
                _fused_norm_weight(self.input_layernorm),
                self.input_layernorm.variance_epsilon,
                x=partial)
        elif fc_zero is not None:
            # fc all-reduce + input_layernorm in one launch; the residual output is the fc sum.
            hidden_states, residual = self.fc(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=fc_zero,
                    norm_weight=_fused_norm_weight(self.input_layernorm),
                    eps=self.input_layernorm.variance_epsilon))
        else:
            hidden_states = self.fc(hidden_states)

        hidden_states, residual = super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=residual,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if not (_Q35_AR_NORM_FUSE and self.fusion_config.POST_MOE_FUSION):
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(0, hidden_states, None)

        return hidden_states

    def _q35_stock_fc_input(self, input_ids, hidden_states, embed_tokens,
                            chunked, tp_size, tp_rank):
        """The stock fc input: chunk(concat([norm(embed(ids)), norm(hidden)]))"""

        def norm_embeds():
            if (_Q35_AR_NORM_FUSE and embed_tokens.tp_size > 1
                    and embed_tokens.tp_mode == TensorParallelMode.COLUMN
                    and q35_mnnvl_fusable(embed_tokens.all_reduce,
                                          input_ids.numel())):
                # Embedding all-reduce + pre_fc_norm_embedding in one launch.
                return embed_tokens(
                    input_ids,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RMS_NORM,
                        norm_weight=_fused_norm_weight(
                            self.pre_fc_norm_embedding),
                        eps=self.pre_fc_norm_embedding.variance_epsilon))
            return self.pre_fc_norm_embedding(embed_tokens(input_ids))

        def norm_hidden():
            return self.pre_fc_norm_hidden(hidden_states)

        inputs_embeds, hidden_states = maybe_execute_in_parallel(
            norm_embeds,
            norm_hidden,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
            disable_on_compile=True,
        )
        hidden_states = torch.concat([inputs_embeds, hidden_states], dim=-1)
        if chunked:
            hidden_states = torch.chunk(hidden_states, tp_size, dim=-1)[tp_rank]
        return hidden_states


ALL_DECODER_LAYER_TYPES = {
    "full_attention": Qwen3NextFullAttentionDecoderLayer,
    "linear_attention": Qwen3NextLinearDecoderLayer,
}


class Qwen3NextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig]):
        super().__init__(model_config)
        config = self.model_config
        pretrained_config = self.model_config.pretrained_config
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }
        self.aux_stream = self.aux_stream_dict[AuxStreamType.Attention]
        self.preload_weight_modules = []
        if config.moe_backend == "TRTLLM":
            self.preload_weight_modules = [
                "experts",
                "routing_method",
                "all_reduce",
            ]

        if model_config.mapping.enable_attention_dp or _Q35_REPLICATED_EMBED:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            # TRTLLM_Q35_REPLICATED_EMBED: the full table on every rank removes the
            # vocab-parallel all-reduce of every target and MTP embedding lookup.
            self.embed_tokens = Embedding(pretrained_config.vocab_size,
                                          pretrained_config.hidden_size,
                                          dtype=pretrained_config.torch_dtype)
            if _Q35_REPLICATED_EMBED and _Q35_EMBED_GATHER:
                self.embed_tokens.__dict__["q35_gather"] = True
        else:
            self.embed_tokens = Embedding(
                pretrained_config.vocab_size,
                pretrained_config.hidden_size,
                dtype=pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        layer_types = get_qwen3_hybrid_layer_types(pretrained_config)
        self.layers = nn.ModuleList([
            ALL_DECODER_LAYER_TYPES[layer_types[layer_idx]](model_config,
                                                            layer_idx,
                                                            self.aux_stream)
            for layer_idx in range(pretrained_config.num_hidden_layers)
        ])
        self.num_hidden_layers = pretrained_config.num_hidden_layers

        self.norm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype,
            use_gemma=True,
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        mamba_metadata = attn_metadata.mamba_metadata
        if mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            attn_metadata.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests, chunk_size=128)

        residual = None
        if inputs_embeds is None:
            emb = self.embed_tokens
            zero = None
            if (_Q35_AR_NORM_FUSE and emb.tp_size > 1
                    and emb.tp_mode == TensorParallelMode.COLUMN
                    and q35_mnnvl_fusable(emb.all_reduce, input_ids.numel())):
                zero = q35_zero_residual((input_ids.numel(), emb.embedding_dim),
                                         emb.dtype, input_ids.device)
            if zero is not None:
                # Embedding all-reduce + layer 0's input_layernorm (+ its FP8 input) in one
                # launch; layer 0 then skips its norm (residual given).
                inputs_embeds, residual = _q35_fused_norm_allreduce(
                    lambda x, all_reduce_params: emb(
                        input_ids, all_reduce_params=all_reduce_params), None,
                    zero, self.layers[0].input_layernorm, self.layers[0])
            else:
                inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers[:self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mamba_metadata=mamba_metadata,
                lora_params=lora_params)
        return hidden_states


@register_auto_model("Qwen3NextForCausalLM")
class Qwen3NextForCausalLM(SpecDecOneEngineForCausalLM[Qwen3NextModel,
                                                       Qwen3NextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
    ):
        if (model_config.spec_config is not None
                and model_config.spec_config.spec_dec_mode.is_mtp_one_model()):
            ckpt_num_nextn = getattr(model_config.pretrained_config,
                                     "num_nextn_predict_layers", None)
            if ckpt_num_nextn not in (None, 1):
                logger.warning(
                    "Qwen3Next one-model MTP uses one shared MTP layer, but "
                    f"checkpoint/config reports num_nextn_predict_layers={ckpt_num_nextn}. "
                    "Forcing num_nextn_predict_layers=1 to keep eagle-style recurrence."
                )
            model_config.pretrained_config.num_nextn_predict_layers = 1

        super().__init__(
            Qwen3NextModel(model_config),
            model_config,
        )
        self.preload_weight_modules = self.model.preload_weight_modules

        if (model_config.spec_config is not None
                and model_config.spec_config.spec_dec_mode.is_mtp_one_model()):

            self.model.layers.extend(self.draft_model.mtp_layers)

    @classmethod
    def get_model_defaults(cls, llm_args: 'TorchLlmArgs') -> dict:
        """Use V2 by default; explicit V1 selections remain supported."""
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": True,
            }
        }

    @classmethod
    def get_preferred_transceiver_runtime(cls,
                                          pretrained_config: object
                                          | None = None) -> Literal["PYTHON"]:
        """Use the Python transceiver for hybrid-state transfers."""
        return "PYTHON"

    def load_weights(self,
                     weights: dict,
                     weight_mapper: BaseWeightMapper,
                     params_map: Optional[Dict[str, str]] = None,
                     allow_partial_loading: bool = False):
        new_weights = weight_mapper.preprocess_weights(
            weights, allow_partial_loading=allow_partial_loading)
        # `new_weights` aliases the source tensors for every key
        # `preprocess_weights` did not rewrite -- the routed experts, i.e. most
        # of a MoE checkpoint. Holding `weights` too would pin them, so
        # consuming `new_weights` during the load would free nothing.
        if hasattr(weights, "clear"):
            weights.clear()
        super().load_weights(
            new_weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )

    def setup_aliases(self) -> None:
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
                if _Q35_AR_FP8_OUT:
                    # Plain attribute (not a registered submodule) for _q35_fused_norm_allreduce.
                    layer.__dict__["_q35_next_layer"] = self.model.layers[idx +
                                                                          1]

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        _precompute_fused_norm_weights(self)
        self._build_q35_draft_head()

    def _build_q35_draft_head(self) -> None:
        """TRTLLM_Q35_DRAFT_FP8_HEAD=1: draft-only MXFP8 copy of this rank's LM-head shard for the
        one-model MTP draft loop (see ``modules/q35_draft_head.py``); the target head stays BF16."""
        from ..modules.q35_draft_head import (build_draft_head_if_enabled,
                                              draft_fp8_head_enabled)
        if self.draft_model is None or not draft_fp8_head_enabled():
            return
        spec_config = self.model_config.spec_config
        if (spec_config is None
                or not spec_config.spec_dec_mode.is_mtp_eagle_one_model()):
            return
        # Rebuild from the current BF16 weights (idempotent on reloads).
        self.draft_model.q35_draft_head = None
        self.draft_model.q35_draft_head = build_draft_head_if_enabled(
            self.lm_head, self.model_config.mapping)
