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
from typing import TYPE_CHECKING, Dict, List, Literal, NamedTuple, Optional

import torch

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from torch import nn
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.moe.fused_shared_expert import \
    fused_sigmoid_gate_mul_add
from tensorrt_llm._torch.pyexecutor.config_utils import \
    get_qwen3_hybrid_layer_types
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...logger import logger
from .. import q35_env
from ..attention.backends import AttentionMetadata
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduce, MoEAllReduceParams, allgather)
from ..distributed.q35_allreduce import (FUSED_AR_MAX_ROWS,
                                         FUSED_AR_TWOSHOT_ROWS,
                                         get_q35_cute_allreduce,
                                         get_q35_gemm_ar, q35_mnnvl_fusable,
                                         q35_zero_residual)
from ..distributed.q35_ar2 import get_q35_ar2
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import FP8QDQLinearMethod, Linear, TensorParallelMode
from ..modules.low_m_gemm import _should_apply_low_m_gemm, apply_low_m_gemm
from ..modules.lowm_linear import (LOWM_FP8_ACT, lowm_enabled, lowm_gemm,
                                   lowm_rows_ok, swiglu_pairs)
from ..modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from ..modules.multi_stream_utils import (do_multi_stream,
                                          maybe_execute_in_parallel)
from ..modules.rms_norm import RMSNorm
from ..moe.fused_moe import (BaseMoeRoutingMethod, MoEWeightLoadingMode,
                             RenormalizeMoeRoutingMethod,
                             RenormalizeNaiveMoeRoutingMethod,
                             RoutingMethodType, create_moe)
from ..moe.fused_moe.q35_fused_front import (Q35_MOE_ROUTER_MAX_TOKENS,
                                             Q35RouterGemm, q35_routed_forward)
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, create_lm_head_tp_mapping
from .modeling_qwen3 import Qwen3Attention
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, EagerFusionConfig, register_auto_model

# Every Qwen3.5 path in this file is entered only when q35_env.fast_paths_enabled().
# The fused all-reduces serve decode-sized calls on MNNVL without attention DP;
# every other call takes the library op.


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


def _q35_replicated_embed() -> bool:
    """Qwen3.5: the full embedding table on every rank (+~1.8 GB/GPU on Qwen3.5-397B), so no target or MTP
    lookup needs the vocab-parallel all-reduce. A memory-for-latency trade that depends on the worker's role."""
    return (q35_env.fast_paths_enabled()
            and os.environ.get("TRTLLM_Q35_REPLICATED_EMBED", "0") == "1")


# A plain all-reduce + RMSNorm (no fold) takes the fused kernel only while the library's MNNVL path would be
# one-shot (rows x hidden x TP x 2 B <= 1 MiB); above that the library two-shot + Lamport RMSNorm is faster.
_Q35_PLAIN_AR_MAX_BYTES = 1 << 20


@q35_env.register
def _q35_read_switches() -> None:
    """This module's TRTLLM_Q35_* switches; q35_env.activate() re-reads them."""
    global _Q35_PLAIN_AR_TWOSHOT
    # TRTLLM_Q35_FUSED_AR_PLAIN_TWOSHOT=<lo>-<hi> (default off): the plain all-reduces of lo..hi rows above that
    # bound also take the CuTe kernel's two-shot, bitwise the library two-shot + Lamport RMSNorm pair it replaces
    # (lo >= FUSED_AR_TWOSHOT_ROWS: below it the CuTe call is one-shot, (x * rstd) * w, not bitwise). On TP8
    # the CuTe two-shot is faster within one wave of CTAs (up to ~2 rows per SM); above it the library pair is
    # faster, even with TRTLLM_Q35_FUSED_AR_TPC.
    _Q35_PLAIN_AR_TWOSHOT = tuple(
        int(v) for v in os.environ.get("TRTLLM_Q35_FUSED_AR_PLAIN_TWOSHOT",
                                       "0-0").split("-"))


def _q35_fused_ar(allreduce, rows: int, is_post_moe: bool, hidden: int = 4096):
    """The fused all-reduce + RMSNorm callable for this call, or None (the library op).

    The one-shot calls run on the slot kernel (``q35_ar2``), the two-shot ones on the CuTe DSL kernel
    (``q35_allreduce``); both produce the library op's outputs. Only with the Qwen3.5 fast paths, on MNNVL and
    up to ``FUSED_AR_MAX_ROWS`` rows.
    """
    if (not q35_env.fast_paths_enabled() or allreduce is None
            or getattr(allreduce, "mnnvl_allreduce", None) is None
            or not 0 < rows <= FUSED_AR_MAX_ROWS):
        return None
    lo, hi = _Q35_PLAIN_AR_TWOSHOT
    if (not is_post_moe and rows * hidden * allreduce.mapping.tp_size * 2
            > _Q35_PLAIN_AR_MAX_BYTES
            and not FUSED_AR_TWOSHOT_ROWS <= lo <= rows <= hi):
        return None
    inst = get_q35_cute_allreduce(allreduce.mapping)
    return get_q35_ar2(allreduce.mapping, inst) or inst


def _q35_moe_defer_kwargs(decoder_layer, rows: int,
                          lora_params: Optional[dict]) -> dict:
    """MoE kwargs that hand the shared-expert combine to the fused MoE-output all-reduce, or {} when that
    all-reduce is the library op.

    Where the experts support it the MoE also skips its finalize (trtllm-gen NVFP4 with ``do_finalize=False``)
    and the all-reduce finalizes the permuted FC2 rows in-kernel (the finalize kernel's arithmetic).
    """
    mlp = decoder_layer.mlp
    if (not decoder_layer.fusion_config.POST_MOE_FUSION
            or not isinstance(mlp, Qwen3NextSparseMoeBlock)
            or _q35_fused_ar(decoder_layer.allreduce, rows, True) is None):
        return {}
    kwargs = {"q35_defer_shared_combine": True}
    if lora_params is None and mlp.q35_can_defer_finalize():
        kwargs["q35_defer_finalize"] = True
    return kwargs


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
    """Arm ``proj`` (attention o_proj / GDN out_proj) to defer into the fused GEMM + all-reduce + norm.

    The projection (FP8 per-tensor on the low-M kernel) and the following all-reduce + residual +
    post_attention_layernorm then run as one launch (``q35_allreduce.Q35GemmArNorm``): same projection values;
    the RMSNorm sums its squares in another order. Only for calls the low-M kernel takes (decode sizes).
    """
    if (lora_params is not None or rows <= 0
            or not decoder_layer.fusion_config.PRE_MOE_FUSION
            or not proj.__dict__.get("_q35_defer_installed", False)
            or not getattr(proj, "_q35_lowm", False)):
        return
    allreduce = decoder_layer.allreduce
    if getattr(allreduce, "mnnvl_allreduce", None) is None:
        return
    weight = getattr(proj, "weight", None)
    if (weight is None or weight.dtype != torch.float8_e4m3fn
            or getattr(proj, "input_scale", None) is None
            or getattr(proj, "force_dynamic_quantization", False)
            or getattr(proj, "bias", None) is not None):
        return
    if not lowm_rows_ok(rows, weight.shape[0], weight.shape[1], LOWM_FP8_ACT):
        return
    if get_q35_gemm_ar(allreduce.mapping) is None:
        return
    proj.__dict__["_q35_defer_once"] = True


_q35_gemm_ar_logged: set = set()


def _q35_gemm_ar_run(allreduce, deferred: _Q35DeferredProj, residual, norm):
    """The fused projection + all-reduce + residual + RMSNorm, or None (caller materializes)."""
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
            f"[Q35] fused projection + all-reduce + norm in use "
            f"(first call: {x2.shape[0]} rows, K={x2.shape[1]}, {x2.dtype})")
    return outs


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
                              is_post_moe: bool = False):
    """Qwen3.5: ``allreduce`` + residual + ``norm`` (RESIDUAL_RMS_NORM), returning (hidden, residual).

    With an FP8 input projection in ``next_layer`` the same launch also quantizes the norm output with that
    projection's static input scale (bitwise what static_quantize_e4m3_per_tensor makes of it), so the projection
    skips its quantize kernel: before a GDN layer the BF16 output is kept (in_proj_ba) and the FP8 copy is handed
    to the layer as ``_q35_fp8_input``; before a full-attention layer the FP8 tensor replaces it (qkv_proj is the
    only consumer).

    ``hidden_states`` may be a ``_Q35DeferredProj`` (``_q35_gemm_ar_arm``): the projection then runs inside the
    fused launch when it qualifies, else it is materialized here.

    ``shared_fold`` = (routed, shared, gate logits) of a MoE block that deferred its shared-expert combine: the
    fused all-reduce folds it into its input. ``routed`` may be the experts' un-finalized output (fc2 rows, top-k
    weights, expanded -> permuted index), finalized in-kernel.
    """
    if isinstance(hidden_states, _Q35DeferredProj):
        if next_layer is None and shared_fold is None:
            outs = _q35_gemm_ar_run(allreduce, hidden_states, residual, norm)
            if outs is not None:
                return outs
        hidden_states = hidden_states.materialize()
    linear = (_q35_fp8_input_linear(next_layer)
              if next_layer is not None else None)
    rows = (shared_fold[1] if shared_fold is not None else residual).shape[0]
    fused_ar = _q35_fused_ar(allreduce, rows, is_post_moe, residual.shape[-1])
    if shared_fold is not None and isinstance(shared_fold[0], (tuple, list)):
        # Only the fused all-reduce takes an un-finalized MoE output (_q35_moe_defer_kwargs checked it).
        assert fused_ar is not None
    if fused_ar is not None and hasattr(allreduce, "mapping"):
        gdn_next = isinstance(next_layer, Qwen3NextLinearDecoderLayer)
        outs = fused_ar(
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

    # The router weight is a bare parameter, not a Linear, so the dispatcher's
    # name-based scan needs an explicit opt-in to reach it.
    _low_m_gemm_opt_in = True

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
        """Whether the Qwen3.5 router GEMM takes ``hidden_states`` (decode-sized BF16 rows, eager or CUDA graph).

        Not under torch.compile: dynamo cannot analyze the kernel's inline PTX, functionalizes it as mutating every
        argument, and the logits come out wrong.
        """
        return (not torch.compiler.is_compiling() and hidden_states.dim() == 2
                and hidden_states.shape[0] <= Q35_MOE_ROUTER_MAX_TOKENS
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.stride(1) == 1
                and self.weight.dtype == torch.bfloat16
                and self.out_dtype == torch.bfloat16)

    def q35_router(self,
                   hidden_states: torch.Tensor,
                   global_scale: Optional[torch.Tensor] = None):
        """Router logits from the single-launch Qwen3.5 router GEMM; with ``global_scale`` also the NVFP4
        quantization of ``hidden_states``: (logits, xq, xsf)."""
        if self._q35_router_gemm is None:
            self._q35_router_gemm = Q35RouterGemm()
        return self._q35_router_gemm(hidden_states, self.weight, global_scale)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if q35_env.fast_paths_enabled() and self.q35_router_ok(hidden_states):
            return self.q35_router(hidden_states)
        # The low-M kernels return the input dtype, so cublas_mm below still
        # owns any out_dtype promotion.
        if _should_apply_low_m_gemm(
                hidden_states) and self.out_dtype == hidden_states.dtype:
            logits = apply_low_m_gemm(self, hidden_states, self.weight, None)
            if logits is not None:
                return logits
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


class _DeferredSharedExpertFinalize(NamedTuple):
    """The three tensors the shared-expert finalize would have combined.

    Returned in place of ``routed + sigmoid(gate) * shared`` when the caller
    asks to defer and no collective follows, so that a downstream consumer can
    fold that expression into a kernel it already runs over these rows. The
    Qwen3.5 fused MoE-output all-reduce takes it too (``q35_defer_shared_combine``);
    there ``routed_output`` may be the experts' un-finalized
    ``(fc2, topk_weights, expanded_idx_to_permuted_idx)``.
    """

    routed_output: torch.Tensor
    shared_output: torch.Tensor
    shared_gate_logits: torch.Tensor


_Q35_SE_GATE_STREAMS: Dict[int, torch.cuda.Stream] = {}


def _q35_se_gate_stream() -> torch.cuda.Stream:
    """The aux stream of the shared-expert gate GEMV, one per device (shared by every MoE layer)."""
    device = torch.cuda.current_device()
    stream = _Q35_SE_GATE_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream()
        _Q35_SE_GATE_STREAMS[device] = stream
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
        # Qwen3.5: the MTP layer's BF16 experts (no per-layer quant entry in the
        # MIXED_PRECISION NVFP4 checkpoint) are quantized to NVFP4 at load time
        # and run on TRTLLM-gen like the target layers.
        q35_mtp_quant = False
        n_hidden = getattr(config, "num_hidden_layers", None)
        if (q35_env.fast_paths_enabled() and layer_idx is not None
                and n_hidden is not None and layer_idx >= n_hidden):
            from ..modules.q35_mtp_moe_quant import experts_unquantized
            q35_mtp_quant = (
                _experts_excluded_from_quant(model_config, layer_idx)
                or experts_unquantized(model_config, expert_quant_config))
        if q35_mtp_quant:
            from ..modules.q35_mtp_moe_quant import quant_config_for
            expert_quant_config = quant_config_for(
                model_config.quant_config.kv_cache_quant_algo)
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
            if model_config.moe_backend != "CUTLASS":
                moe_model_config = copy.copy(model_config)
                moe_model_config._frozen = False
                moe_model_config.moe_backend = "CUTLASS"
                moe_model_config._frozen = True
                logger.warning(
                    f"Layer {layer_idx} MoE experts are excluded from "
                    "quantization; using moe_backend=CUTLASS for this layer "
                    f"(other layers keep {model_config.moe_backend}).")
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
            # A MegaMoE request that degraded to CUTLASS would still be
            # measured as MegaMoE, so report the rejection trail instead.
            allow_backend_degradation=moe_model_config.moe_backend
            not in ("MEGAMOE_DEEPGEMM", "MEGAMOE_CUTEDSL"),
        )
        self._q35_quant_max_m = 0
        self.q35_experts_bf16 = None
        if q35_mtp_quant:
            from ..modules.q35_mtp_moe_quant import (MTP_MOE_QUANT_MAX_M,
                                                     wrap_experts_loader)
            self._q35_quant_max_m = MTP_MOE_QUANT_MAX_M
            if self._q35_quant_max_m > 0:
                # Inputs above MTP_MOE_QUANT_MAX_M rows run on a BF16 twin built
                # as the stock MTP experts (CUTLASS).
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
            wrap_experts_loader(self.experts, self.q35_experts_bf16)
            logger.info(
                f"Layer {layer_idx} (MTP) MoE experts: bf16 checkpoint "
                "weights quantized to NVFP4 at load time for "
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
        # Qwen3.5: (gate stream, x-ready event, gate-done event). The routed
        # MoE's early-launched (PDL) FC1 fills the SMs, so the shared_expert_gate
        # GEMV + reduce get their own aux stream instead of queueing behind it.
        self._q35_se_aux = None
        if q35_env.fast_paths_enabled():
            self._q35_se_aux = (_q35_se_gate_stream(), torch.cuda.Event(),
                                torch.cuda.Event())
        # Qwen3.5 low-M shared expert: (SwiGLU-paired gate_up weight,
        # zero-padded gate weight), built on the first eager decode-sized call;
        # False when the shared expert does not qualify.
        self._se_lowm = None

    def post_load_weights(self) -> None:
        # Rebuilt from the final (rescaled) weights on the first eager decode-sized call; CUDA-graph
        # capture is always preceded by eager warmup forwards.
        self._se_lowm = None

    def _prepare_shared_expert_lowm(self) -> None:
        """Weights of the shared expert's low-M path.

        Decode-sized batches run the FP8 shared expert as two low-M launches: (1) gate_up with the
        SwiGLU in the epilogue (FP8 output at down_proj.input_scale) plus shared_expert_gate as a BF16
        segment; (2) down_proj. The gate_up rows are re-paired into a copy; the original stays for
        larger batches.
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
        if not lowm_enabled() or x.dim() != 2 or x.dtype != torch.bfloat16:
            return None
        gate_up_w = self.shared_expert.gate_up_proj.weight
        down_w = self.shared_expert.down_proj.weight
        if not (lowm_rows_ok(x.shape[0], gate_up_w.shape[0], x.shape[1],
                             LOWM_FP8_ACT)
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
        return out, gate_logits

    def q35_can_defer_finalize(self) -> bool:
        """trtllm-gen NVFP4 experts with in-kernel routing (their do_finalize=False output carries
        valid top-k weights) and no MoE communication / load balancer."""
        cached = self.__dict__.get("_q35_can_defer_finalize")
        if cached is None:
            from ..moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
            from ..moe.fused_moe.interface import FORCE_SEPARATED_ROUTING
            experts = self.experts
            backend = getattr(experts, "backend", experts)
            routing = getattr(experts, "routing_method", None)
            cached = (
                isinstance(backend, TRTLLMGenFusedMoE)
                and bool(getattr(backend, "has_nvfp4", False))
                # The BF16 twin (CUTLASS) always finalizes, so a layer with one
                # never defers.
                and self.q35_experts_bf16 is None and
                not self.enable_attention_dp and
                not backend._supports_load_balancer() and
                getattr(experts, "comm", None) is None and
                getattr(experts, "layer_load_balancer", None) is None and
                routing is not None and not routing.requires_separated_routing
                and not FORCE_SEPARATED_ROUTING)
            self.__dict__["_q35_can_defer_finalize"] = cached
        return cached

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
        lora_params: Optional[dict] = None,
        defer_shared_expert_finalize: bool = False,
        q35_defer_shared_combine: bool = False,
        q35_defer_finalize: bool = False,
    ) -> torch.Tensor | _DeferredSharedExpertFinalize:
        """``q35_defer_shared_combine`` (Qwen3.5; the caller runs the fused MoE-output all-reduce): return the
        three combine operands instead of routed + sigmoid(gate) * shared. ``q35_defer_finalize`` (with the
        former): the routed operand is the experts' do_finalize=False output (fc2 rows, top-k weights,
        expanded -> permuted index)."""
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

        q35 = q35_env.fast_paths_enabled()
        experts_finalize = do_finalize and not (q35_defer_finalize
                                                and q35_defer_shared_combine
                                                and lora_params is None)

        def _compute_routed_output():
            experts = self.experts
            if q35:
                # The quantized MTP experts serve inputs up to the cap, larger
                # ones the BF16 twin (static per CUDA graph).
                if (self.q35_experts_bf16 is not None
                        and hidden_states.shape[0] > self._q35_quant_max_m):
                    experts = self.q35_experts_bf16
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

            return final_hidden_states

        se_aux = self._q35_se_aux if q35 else None

        def _compute_shared_output():
            if q35 and not lora_params:
                lowm = self._shared_expert_lowm(hidden_states)
                if lowm is not None:
                    return lowm
            if (se_aux is not None and do_multi_stream()
                    and not torch.compiler.is_compiling()):
                # This runs on the chain's aux stream once x is ready; the gate
                # GEMV + split-K reduce fork onto their own stream.
                gate_stream, x_ready, gate_done = se_aux
                x_ready.record()
                with torch.cuda.stream(gate_stream):
                    x_ready.wait()
                    shared_expert_gate_logits = self.shared_expert_gate(
                        hidden_states)
                    gate_done.record()
                shared_expert_output = self.shared_expert(
                    hidden_states,
                    lora_params=lora_params,
                )
                gate_done.wait()
                return shared_expert_output, shared_expert_gate_logits
            shared_expert_output = self.shared_expert(
                hidden_states,
                lora_params=lora_params,
            )
            shared_expert_gate_logits = self.shared_expert_gate(hidden_states)
            return shared_expert_output, shared_expert_gate_logits

        final_hidden_states, shared_expert_outputs = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
            disable_on_compile=True,
        )
        if not do_finalize:
            return final_hidden_states

        shared_expert_output, shared_expert_gate_logits = shared_expert_outputs
        if q35_defer_shared_combine:
            return _DeferredSharedExpertFinalize(
                final_hidden_states,
                shared_expert_output,
                shared_expert_gate_logits,
            )
        assert experts_finalize
        # Deferring is sound only where no collective follows the finalize,
        # which is exactly the negation of the all-reduce condition below.
        # DP padding and >2D inputs are excluded: the rows handed over would no
        # longer line up with the Hyper-Connection residual.
        no_collective_after_finalize = (self.enable_attention_dp
                                        or self.mapping.tp_size == 1)
        if (defer_shared_expert_finalize and no_collective_after_finalize
                and not use_dp_padding and len(orig_shape) == 2):
            return _DeferredSharedExpertFinalize(
                final_hidden_states,
                shared_expert_output,
                shared_expert_gate_logits,
            )
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
        defer_shared_expert_finalize=False,
    ):
        assert not defer_shared_expert_finalize, \
            "a dense MLP has no shared-expert finalize to defer"
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

        # MoEAllReduce is not merely unused under attention DP -- constructing it
        # is destructive. Its __init__ eagerly calls get_allreduce_workspace(),
        # which opens a CUDA-IPC workspace across the TP group
        # (ops.py -> allreduce_helper.allocate_allreduce_fusion_workspace ->
        # IpcMemory.open_ipc_memory). With attention DP that peer topology does
        # not exist and the import faults with cudaErrorIllegalAddress, which is
        # sticky: every later CUDA call in the context fails too, so the crash
        # surfaces somewhere unrelated (create_rope_const_params).
        #
        # It is also genuinely unreachable here: _eager_fusion_enabled() returns
        # False under attention DP, so PRE/POST_MOE_FUSION are False and the only
        # call sites (inside `if self.fusion_config.POST_MOE_FUSION`) never run.
        # Gate on the fusion flag itself: it already excludes attention DP and
        # pipeline-parallel layers, and also skips the workspace when eager
        # fusion is disabled (TRTLLM_QWEN3_EAGER_FUSION_DISABLED=1). Forward
        # only ever clears the flag (spec-metadata layer capture), never sets
        # it, so the construction-time value is a superset of runtime use.
        self.moe_allreduce = None
        if self.fusion_config.POST_MOE_FUSION and self.mapping.tp_size > 1:
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
        q35 = q35_env.fast_paths_enabled()
        # Qwen3.5: in_proj_qkvz's FP8 input, emitted by the preceding fused
        # all-reduce (_q35_fused_norm_allreduce).
        h_fp8 = _q35_pop_fp8_input(self, hidden_states) if q35 else None
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
            if q35:
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
            if q35:
                hidden_states, residual = _q35_fused_norm_allreduce(
                    self.allreduce, hidden_states, residual,
                    self.post_attention_layernorm, None)
            else:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=_fused_norm_weight(
                            self.post_attention_layernorm),
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ))
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # Qwen3NextSparseMoeBlock does not implement do_finalize=False. Defer
        # only its final all-reduce so the decoder can fuse it with RMSNorm.
        do_finalize = True

        defer_kwargs = (_q35_moe_defer_kwargs(self, hidden_states.shape[0],
                                              lora_params) if q35 else {})
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
            if do_finalize and q35:
                hidden_states, residual = _q35_fused_norm_allreduce(
                    self.allreduce,
                    None if defer_kwargs else hidden_states,
                    residual,
                    self.next_layer_layernorm,
                    self.__dict__.get("_q35_next_layer"),
                    shared_fold=hidden_states if defer_kwargs else None,
                    is_post_moe=True)
            elif do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=_fused_norm_weight(
                            self.next_layer_layernorm),
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
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
        # Qwen3.5 decode-sized projections on the low-M kernel (FP8 target
        # layers / BF16 MTP layer); linear.py reads these labels only under the
        # Qwen3.5 fast paths.
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
        # MoEAllReduce is not merely unused under attention DP -- constructing it
        # is destructive. Its __init__ eagerly calls get_allreduce_workspace(),
        # which opens a CUDA-IPC workspace across the TP group
        # (ops.py -> allreduce_helper.allocate_allreduce_fusion_workspace ->
        # IpcMemory.open_ipc_memory). With attention DP that peer topology does
        # not exist and the import faults with cudaErrorIllegalAddress, which is
        # sticky: every later CUDA call in the context fails too, so the crash
        # surfaces somewhere unrelated (create_rope_const_params).
        #
        # It is also genuinely unreachable here: _eager_fusion_enabled() returns
        # False under attention DP, so PRE/POST_MOE_FUSION are False and the only
        # call sites (inside `if self.fusion_config.POST_MOE_FUSION`) never run.
        # Gate on the fusion flag itself: it already excludes attention DP and
        # pipeline-parallel layers, and also skips the workspace when eager
        # fusion is disabled (TRTLLM_QWEN3_EAGER_FUSION_DISABLED=1). Forward
        # only ever clears the flag (spec-metadata layer capture), never sets
        # it, so the construction-time value is a superset of runtime use.
        self.moe_allreduce = None
        if self.fusion_config.POST_MOE_FUSION and self.mapping.tp_size > 1:
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
        q35 = q35_env.fast_paths_enabled()

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False

        # Self Attention
        if q35 and hidden_states.shape[0] != 0:
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
            if q35:
                hidden_states, residual = _q35_fused_norm_allreduce(
                    self.allreduce, hidden_states, residual,
                    self.post_attention_layernorm, None)
            else:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=_fused_norm_weight(
                            self.post_attention_layernorm),
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ))
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
        defer_kwargs = (_q35_moe_defer_kwargs(self, hidden_states.shape[0],
                                              lora_params) if q35 else {})
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
            if do_finalize and q35:
                hidden_states, residual = _q35_fused_norm_allreduce(
                    self.allreduce,
                    None if defer_kwargs else hidden_states,
                    residual,
                    self.next_layer_layernorm,
                    self.__dict__.get("_q35_next_layer"),
                    shared_fold=hidden_states if defer_kwargs else None,
                    is_post_moe=True)
            elif do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=_fused_norm_weight(
                            self.next_layer_layernorm),
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
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
        # Qwen3.5: the BF16 fc on the low-M kernel (strided input rows are fine).
        self.fc._q35_lowm_bf16 = "mtp.fc"
        self.shared_head = Qwen3NextMTPHead(model_config)
        # MTP applies shared_head.norm after the base decoder forward, so its
        # MoE-output all-reduce cannot consume next_layer_layernorm.
        self.fusion_config.POST_MOE_FUSION = False
        if (q35_env.fast_paths_enabled() and self.fusion_config.PRE_MOE_FUSION):
            # Qwen3.5: the MoE-output all-reduce applies residual +
            # shared_head.norm itself. A plain attribute, so the norm is not
            # registered a second time as a submodule.
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
        # Install this draft step's attention-DP token distribution: the draft
        # loop passes it as a kwarg and leaves attn_metadata alone, matching
        # Eagle3DraftModel.forward. The MoE below reads the per-rank counts off
        # attn_metadata, and an MTP Eagle draft runs one token per sequence
        # after step 0, so a stale target value mismatches collective sizes.
        previous_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        try:
            q35 = q35_env.fast_paths_enabled()
            mapping = self.model_config.mapping
            tp_size = mapping.tp_size
            tp_rank = mapping.tp_rank
            chunked = tp_size > 1 and not mapping.enable_attention_dp
            # A ``kv_lens_cuda[:n] += 1`` the draft loop deferred to this pass
            # (TRTLLM_Q35_DRAFT_BOOKKEEPING): folded into the fused input launch
            # below, else applied here before anything reads kv_lens.
            kv_inc = getattr(attn_metadata, "q35_pending_kv_inc", 0)
            if kv_inc:
                attn_metadata.q35_pending_kv_inc = 0
            if (q35 and getattr(embed_tokens, "tp_size", 1) == 1
                    and embed_tokens.weight.dtype == hidden_states.dtype
                    and hidden_states.stride(-1) == 1):
                # Qwen3.5 with a full (replicated) embedding table: the masked
                # embedding gather, both pre-fc Gemma RMSNorms, the concat and
                # the row-parallel chunk in one launch (modules/q35_mtp_glue.py).
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
                hidden_states = self._q35_stock_fc_input(
                    input_ids, hidden_states, embed_tokens, chunked, tp_size,
                    tp_rank)

            residual = None
            fc_all_reduce = getattr(self.fc, "all_reduce", None)
            fc_zero = None
            if q35 and q35_mnnvl_fusable(fc_all_reduce, hidden_states.shape[0]):
                fc_zero = q35_zero_residual(
                    (hidden_states.shape[0], self.fc.out_features),
                    hidden_states.dtype, hidden_states.device)
            fc_fused = (_q35_fused_ar(fc_all_reduce, hidden_states.shape[0],
                                      False, self.fc.out_features)
                        if fc_zero is not None else None)
            if fc_fused is not None:
                # This rank's fc partial, then the fused all-reduce +
                # input_layernorm with the zero residual (residual output = the
                # fc sum).
                partial = self.fc(
                    hidden_states,
                    all_reduce_params=AllReduceParams(enable_allreduce=False))
                hidden_states, residual = fc_fused(
                    fc_zero,
                    _fused_norm_weight(self.input_layernorm),
                    self.input_layernorm.variance_epsilon,
                    x=partial)
            elif fc_zero is not None:
                # fc all-reduce + input_layernorm in one library launch; the
                # residual output is the fc sum.
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
            # The base forward already applied shared_head.norm when it is this
            # layer's next_layer_layernorm (Qwen3.5, see __init__).
            if self.__dict__.get(
                    "next_layer_layernorm") is not self.shared_head.norm:
                hidden_states, _ = self.shared_head.norm(
                    hidden_states, residual)
            if spec_metadata is not None:
                spec_metadata.maybe_capture_hidden_states(
                    0, hidden_states, None)

            return hidden_states
        finally:
            # Shared with the target forward, so restore on the exception path
            # too; otherwise a failed draft step corrupts every later forward.
            if all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = previous_all_rank_num_tokens

    def _q35_stock_fc_input(self, input_ids, hidden_states, embed_tokens,
                            chunked, tp_size, tp_rank):
        """The fc input ``chunk(concat([norm(embed(ids)), norm(hidden)]))`` as separate kernels."""

        def norm_embeds():
            if (q35_env.fast_paths_enabled() and embed_tokens.tp_size > 1
                    and embed_tokens.tp_mode == TensorParallelMode.COLUMN
                    and q35_mnnvl_fusable(embed_tokens.all_reduce,
                                          input_ids.numel())):
                # Qwen3.5: embedding all-reduce + pre_fc_norm_embedding in one
                # launch.
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

        replicated_embed = _q35_replicated_embed()
        if model_config.mapping.enable_attention_dp or replicated_embed:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embed_tokens = Embedding(pretrained_config.vocab_size,
                                          pretrained_config.hidden_size,
                                          dtype=pretrained_config.torch_dtype)
            if replicated_embed:
                # Every lookup of the full table is one gather launch instead
                # of the masked lookup's kernels (bitwise).
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
            if (q35_env.fast_paths_enabled() and emb.tp_size > 1
                    and emb.tp_mode == TensorParallelMode.COLUMN
                    and q35_mnnvl_fusable(emb.all_reduce, input_ids.numel())):
                zero = q35_zero_residual((input_ids.numel(), emb.embedding_dim),
                                         emb.dtype, input_ids.device)
            if zero is not None:
                # Qwen3.5: embedding all-reduce + layer 0's input_layernorm
                # (+ its FP8 input) in one launch; layer 0 then skips its norm
                # (residual given).
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
        """Disable block reuse until a snapshot policy is configured."""
        return {"kv_cache_config": {"enable_block_reuse": False}}

    @classmethod
    def get_preferred_kv_cache_manager_version(cls,
                                               pretrained_config: object
                                               | None = None) -> Literal["V2"]:
        """Prefer KV cache manager V2 for the hybrid state layout."""
        return "V2"

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
        q35 = q35_env.fast_paths_enabled()
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
                if q35:
                    # Plain attribute (not a registered submodule): the next
                    # layer whose FP8 input _q35_fused_norm_allreduce emits.
                    layer.__dict__["_q35_next_layer"] = self.model.layers[idx +
                                                                          1]

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        _precompute_fused_norm_weights(self)
        self._build_q35_draft_head()

    def _build_q35_draft_head(self) -> None:
        """Qwen3.5 with TRTLLM_Q35_DRAFT_FP8_HEAD=1: draft-only MXFP8 copy of this rank's LM-head shard for the
        one-model MTP draft loop (see ``modules/q35_draft_head.py``); the target head stays BF16."""
        if not q35_env.fast_paths_enabled():
            return
        from ..modules.q35_draft_head import (build_draft_head_if_enabled,
                                              draft_fp8_head_enabled)
        draft_model = getattr(self, "draft_model", None)
        if draft_model is None or not draft_fp8_head_enabled():
            return
        spec_config = self.model_config.spec_config
        if (spec_config is None
                or not spec_config.spec_dec_mode.is_mtp_eagle_one_model()):
            return
        # Rebuild from the current BF16 weights (idempotent on reloads).
        draft_model.q35_draft_head = None
        draft_model.q35_draft_head = build_draft_head_if_enabled(
            self.lm_head, self.model_config.mapping)
