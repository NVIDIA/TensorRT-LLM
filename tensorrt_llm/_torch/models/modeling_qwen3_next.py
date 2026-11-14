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
from typing import (TYPE_CHECKING, Dict, List, Literal, NamedTuple, Optional,
                    Sequence, Tuple, Union)

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
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._utils import is_blackwell_datacenter, is_sm_120f
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...logger import logger
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PredefinedAttentionMask
from ..autotuner import AutoTuner
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduceParams, allgather)
from ..distributed.mnnvl_fused_allreduce import SLOT_MAX_ROWS, TWO_SHOT_ROWS
from ..distributed.ops import MNNVLAllReduce
from ..kernels.nvfp4_quant import can_use_nvfp4_quant, nvfp4_fp8_dual_quant
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding, LMHead
from ..modules.expert_l2_prefetch import ExpertL2Prefetch
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (FP8QDQLinearMethod, FP8RowwiseLinearMethod,
                              Linear, TensorParallelMode,
                              is_static_nvfp4_input_eligible)
from ..modules.low_m_gemm import (LowMGemmDispatcher, _should_apply_low_m_gemm,
                                  apply_low_m_gemm, low_m_gemm_dispatcher,
                                  low_m_gemm_takes_rows, prepare_low_m_gemm)
from ..modules.low_m_mma_gemm import MAX_M as LOW_M_MAX_ROWS
from ..modules.low_m_mma_gemm import (SHARED_EXPERT_MAX_M, default_tactic,
                                      fused_shared_expert,
                                      fused_shared_expert_weights,
                                      raced_shared_expert, warp_mma_supported)
from ..modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from ..modules.multi_stream_utils import (do_multi_stream,
                                          maybe_execute_in_parallel)
from ..modules.rms_norm import RMSNorm
from ..moe.fused_moe import (BaseMoeRoutingMethod, ConfigurableMoE,
                             MoEWeightLoadingMode, RenormalizeMoeRoutingMethod,
                             RenormalizeNaiveMoeRoutingMethod,
                             RoutingMethodType, create_moe)
from ..moe.fused_moe.impl_contract import MOE_DECODE_OPTIONS, MoEDecodeOptions
from ..moe.fused_moe.trtllm_gen.decode_front import (FUSED_FRONT_MAX_TOKENS,
                                                     ROUTER_MAX_TOKENS,
                                                     DecodeFront,
                                                     route_front_steps)
from ..moe.fused_moe.trtllm_gen.decode_moe import (LamportFc2Buffer,
                                                   rows_in_lamport)
from ..speculative import SpecMetadata
from ..speculative.kept_rows import DraftKeptRows, gather_rows2
from ..utils import (AuxStreamType, EventType, Fp4QuantizedTensor,
                     create_lm_head_tp_mapping)
from .modeling_qwen3 import Qwen3Attention
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (DecoderModel, EagerFusionConfig,
                             moe_experts_quant_config, register_auto_model)

# Most rows of the MTP fc all-reduce fused with the input RMSNorm.
_FC_FUSED_NORM_MAX_TOKENS = 1024

# Most rows of an MTP draft LM head call that races cuBLAS against the low-M
# GEMMs.
_DRAFT_HEAD_LOW_M_MAX_ROWS = 8


def _is_qwen3_5(model_config: ModelConfig) -> bool:
    """A Qwen3.5 checkpoint (the Qwen3Next architecture with Qwen3.5 weights)."""
    return getattr(model_config.pretrained_config, "model_type",
                   "").startswith("qwen3_5")


def _uses_decode_kernels(model_config: ModelConfig) -> bool:
    """Whether the model class declares the attention / GDN / draft decode
    paths (``Qwen3NextForCausalLM.decode_kernels``) on SM 10.0 / 10.3, in a
    layout they were tuned for (``_decode_kernels_layout``)."""
    return bool(model_config.extra_attrs.get("qwen3_next_decode_kernels",
                                             False))


def _decode_kernels_layout(model_config: ModelConfig) -> bool:
    """Whether the layout is one the decode kernels were tuned for: attention
    DP, or a TP layout on a multi-node NVLink fabric (GB200 / GB300 NVL
    racks), where the MNNVL all-reduce kernels that the TP decode paths
    overlap with and fuse into can run, whatever all-reduce strategy is set.
    A TP layout without one (the NVLink of an HGX node) keeps the stock
    paths.

    Args:
        model_config: The target model's config (mapping, torch dtype).

    Returns:
        True for attention DP or a TP layout with MNNVL.
    """
    mapping = model_config.mapping
    if mapping.enable_attention_dp:
        return True
    return MNNVLAllReduce.is_mnnvl(mapping,
                                   model_config.pretrained_config.torch_dtype,
                                   explicitly_requested=True)


def _uses_prefill_kernels(model_config: ModelConfig) -> bool:
    """Whether the model class declares the kernels of prefill-sized calls
    (``Qwen3NextForCausalLM.prefill_kernels``)."""
    return bool(
        model_config.extra_attrs.get("qwen3_next_prefill_kernels", False))


def _uses_tp_decode_kernels(model_config: ModelConfig) -> bool:
    """The decode kernels of the tensor-parallel layout (no attention DP): the
    ones that overlap a layer's GEMMs with its all-reduces."""
    return (_uses_decode_kernels(model_config)
            and not model_config.mapping.enable_attention_dp)


def _rows_per_request(model_config: ModelConfig) -> int:
    """Rows a generation request adds to a decode step (1 + draft tokens)."""
    spec_config = model_config.spec_config
    return spec_config.tokens_per_gen_step if spec_config is not None else 1


def _decode_step_rows(model_config: ModelConfig) -> int:
    """Rows of the largest CUDA-graph decode step (0 without generation CUDA
    graphs)."""
    return _rows_per_request(model_config) * max(
        model_config.cuda_graph_batch_sizes, default=0)


# From this many requests of a TP decode step, the input projection after a
# MoE-output all-reduce leaves the low-M race for the resident-weight GEMM, or
# for cuBLAS with its weight pulled into L2 by that all-reduce.
_RESIDENT_MIN_REQUESTS = 2


def _resident_min_rows(model_config: ModelConfig) -> int:
    """Rows from which the input projections after a MoE-output all-reduce
    take the resident-weight GEMM (``_RESIDENT_MIN_REQUESTS`` requests of a TP
    decode step); 0 without the TP decode kernels.

    Args:
        model_config: The target model's config.

    Returns:
        The band's first row count, or 0.
    """
    if not _uses_tp_decode_kernels(model_config):
        return 0
    return _RESIDENT_MIN_REQUESTS * _rows_per_request(model_config)


# The routed experts' cross-step L2 prefetch (ExpertL2Prefetch) runs from 2
# requests below the all-reduces' two-shot rows: beside their slot transport at
# more rows, its burst slows the layer chain more than the experts gain.
_EXPERT_PREFETCH_MIN_REQUESTS = 2
_EXPERT_PREFETCH_MAX_ROWS = TWO_SHOT_ROWS - 1
# Steps above the one-launch front's rows of at most this many requests
# rank the prefetched experts by routed rows and stop each layer's prefetch
# after its issue budget: there the engine starts streaming soon after.
_EXPERT_PREFETCH_BUDGET_MAX_REQUESTS = 5
# Issue budget (ns) of a GDN layer's prefetch: measured at 3-5 requests, a
# longer issue overlaps the layer's MoE front and costs chain time.
_EXPERT_PREFETCH_GDN_BUDGET_NS = 18500
# Same for a full-attention layer, whose mixer reaches the MoE front sooner.
_EXPERT_PREFETCH_ATTN_BUDGET_NS = 12500
# From this many requests of an MTP decode step, the GDN layers verify on the
# tcgen05 kernel and the decode MoE engine leaves SMs to the shared expert
# running beside it.
_LARGE_STEP_MIN_REQUESTS = 64


def _expert_prefetch_budget_rows(model_config: ModelConfig) -> Tuple[int, int]:
    """Rows (exclusive, inclusive) of the steps whose expert prefetch ranks
    by routed rows and stops after the layer's issue budget."""
    return (FUSED_FRONT_MAX_TOKENS, _EXPERT_PREFETCH_BUDGET_MAX_REQUESTS *
            _rows_per_request(model_config))


def _large_step_rows(model_config: ModelConfig) -> int:
    """Rows of a decode step of _LARGE_STEP_MIN_REQUESTS requests (0 without
    MTP)."""
    if model_config.spec_config is None:
        return 0
    return _LARGE_STEP_MIN_REQUESTS * _rows_per_request(model_config)


def _verify_tcgen05_min_rows(model_config: ModelConfig) -> int:
    """Rows from which an MTP verify step runs the GDN replay on the tcgen05
    kernel: every step under attention DP (a rank holds all of a layer's
    heads), large steps with TP attention (0 without MTP)."""
    if model_config.spec_config is None:
        return 0
    if model_config.mapping.enable_attention_dp:
        return _rows_per_request(model_config)
    return _large_step_rows(model_config)


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
            fused = (w.float() + 1.0).to(w.dtype)
            cached = getattr(norm, "_fused_norm_weight", None)
            if (cached is not None and cached.shape == fused.shape
                    and cached.dtype == fused.dtype
                    and cached.device == fused.device):
                # In-place update keeps the buffer address stable: captured
                # CUDA graphs bake this tensor's pointer into the fused-AR
                # kernel args, and post_load_weights re-runs on RL weight
                # refit (rlhf_utils.update_weights) without re-capturing
                # graphs -- rebinding would leave replays reading freed
                # memory.
                cached.copy_(fused)
            else:
                norm._fused_norm_weight = fused


def _eager_fusion_enabled(enable_attention_dp: bool) -> bool:
    return (os.environ.get("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "0") == "0"
            and not enable_attention_dp)


class _Fp8HiddenStates(NamedTuple):
    """A decoder layer's input with its E4M3 copy at the input projection's
    static scale, made by the previous layer's fused all-reduce."""

    hidden_states: torch.Tensor
    fp8: torch.Tensor


def _allreduce_fp8_payload(model_config: ModelConfig) -> bool:
    """``mnnvl_allreduce_fp8_payload`` where this model's all-reduces can take
    it, with a warning where they cannot."""
    if not model_config.mnnvl_allreduce_fp8_payload:
        return False
    config = model_config.pretrained_config
    if MNNVLAllReduce.supports_fp8_payload(model_config.mapping,
                                           config.torch_dtype,
                                           config.hidden_size):
        return True
    logger.warning_once(
        "mnnvl_allreduce_fp8_payload is ignored: it needs BF16, TP 2 / 4 / 8 "
        "and a hidden size that is a multiple of 1024",
        key="qwen3_next_mnnvl_fp8_payload")
    return False


def _fp8_input_projection(layer: nn.Module) -> Optional[Linear]:
    """A decoder layer's input projection (GDN in_proj_qkvz / attention qkv_proj), or None."""
    if isinstance(layer, Qwen3NextLinearDecoderLayer):
        return layer.linear_attn.in_proj_qkvz
    if isinstance(layer, Qwen3NextFullAttentionDecoderLayer):
        return layer.self_attn.qkv_proj
    return None


def _fp8_input_scale(layer: nn.Module) -> Optional[torch.Tensor]:
    """Static input scale of a decoder layer's FP8 per-tensor input projection
    (GDN in_proj_qkvz / attention qkv_proj), or None."""
    if isinstance(layer, Qwen3NextLinearDecoderLayer):
        linear = layer.linear_attn.in_proj_qkvz
    elif isinstance(layer, Qwen3NextFullAttentionDecoderLayer):
        linear = layer.self_attn.qkv_proj
    else:
        return None
    # remove_weights (layer-wise benchmark) leaves no input_scale.
    if (getattr(linear, "_weights_removed", False)
            or not isinstance(linear.quant_method, FP8QDQLinearMethod)):
        return None
    scale = linear.quant_method.get_static_input_scale(linear)
    return scale.detach() if scale is not None else None


def _next_layer_l2_prefetch(layer: nn.Module,
                            rows: int) -> Optional[torch.Tensor]:
    """The weight the MoE-output all-reduce pulls into L2 for the next layer's
    input projection at ``rows`` rows, or None: below ``l2_prefetch_min_rows``
    and where the resident-weight GEMM takes the projection, it loads its
    weight itself."""
    w = layer.next_layer_prefetch_weight
    if w is None or rows < layer.l2_prefetch_min_rows:
        return None
    lo, hi = layer.next_layer_resident_rows
    if lo <= rows <= hi:
        return None
    return w


def _takes_resident_band(layer: nn.Module, projection: Optional[Linear],
                         min_rows: int, max_rows: int) -> bool:
    """Whether ``projection`` (the next layer's input projection) takes the
    resident-weight GEMM from ``min_rows`` to ``max_rows`` rows: this layer's
    MoE-output all-reduce hands it an E4M3 input (TP > 1 with POST_MOE_FUSION
    and the next layer's FP8 input scale), and ``resident_fp8_gemm`` covers
    its (K, N) at every row count of the band.

    Args:
        layer: The decoder layer whose MoE-output all-reduce feeds ``projection``.
        projection: The next layer's FP8 input projection, or None.
        min_rows: The band's first row count (``_resident_min_rows``).
        max_rows: Its last (``resident_fp8_gemm.MAX_ROWS``).

    Returns:
        True where the band applies.
    """
    from ..kernels.blackwell.resident_fp8_gemm import plan
    if (projection is None or not min_rows
            or layer.next_layer_input_scale is None
            or not layer.fusion_config.POST_MOE_FUSION
            or layer.mapping.tp_size == 1 or projection._weights_removed):
        return False
    n, k = projection.weight.shape
    return all(
        plan(rows, k, n) is not None for rows in range(min_rows, max_rows + 1))


def _post_moe_allreduce(layer: nn.Module, mlp_output, residual: torch.Tensor):
    """The MoE-output all-reduce + residual + ``next_layer_layernorm``; returns
    (hidden, residual).

    ``mlp_output`` is the MLP output, or the MoE's routed output with its
    shared expert and gate logits (``_DeferredSharedExpertFinalize``), which
    the all-reduce combines (and finalizes, when un-finalized). With
    ``next_layer_input_scale`` set the same launch also quantizes the norm
    output for the next layer's FP8 input projection: the hidden is then the
    E4M3 tensor (attention), or ``_Fp8HiddenStates`` (GDN, whose in_proj_ba
    reads BF16).
    """
    norm = layer.next_layer_layernorm
    scale = layer.next_layer_input_scale
    if scale is None:
        fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
    elif layer.next_layer_reads_bf16:
        fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8
    else:
        fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
    common = dict(fusion_op=fusion_op,
                  residual=residual,
                  norm_weight=_fused_norm_weight(norm),
                  scale=scale,
                  eps=norm.variance_epsilon,
                  l2_prefetch=_next_layer_l2_prefetch(layer, residual.shape[0]))
    if isinstance(mlp_output, _DeferredSharedExpertFinalize):
        routed = mlp_output.routed_output
        if isinstance(routed, tuple):
            routed, topk_weights, expanded_idx_to_permuted_idx = routed
        else:
            topk_weights = expanded_idx_to_permuted_idx = None
        params = MoEAllReduceParams(
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            expert_scale_factor=topk_weights,
            shared_expert_output=mlp_output.shared_output,
            shared_expert_gate=mlp_output.shared_gate_logits,
            fc2_lamport=mlp_output.fc2_lamport,
            **common)
        outputs = layer.allreduce(routed, all_reduce_params=params)
    else:
        outputs = layer.allreduce(mlp_output,
                                  all_reduce_params=AllReduceParams(
                                      fp8_payload=layer.allreduce_fp8_payload,
                                      **common))
    if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8:
        norm_out, fp8, residual = outputs
        return _Fp8HiddenStates(norm_out, fp8), residual
    return outputs


def _post_moe_norm_fold(layer: nn.Module,
                        mlp_output: "_DeferredSharedExpertFinalize",
                        residual: torch.Tensor):
    """Without a MoE-output all-reduce: the shared-expert gate, residual add,
    ``next_layer_layernorm`` and the next layer's FP8 quant in one launch;
    returns (hidden, residual) as ``_post_moe_allreduce`` does."""
    from ..kernels.blackwell.gate_add_rmsnorm_fp8 import gate_add_rmsnorm_fp8
    norm = layer.next_layer_layernorm
    write_y = layer.next_layer_reads_bf16
    norm_out, fp8 = gate_add_rmsnorm_fp8(mlp_output.routed_output,
                                         mlp_output.shared_output,
                                         mlp_output.shared_gate_logits,
                                         residual,
                                         norm.weight,
                                         norm.variance_epsilon,
                                         layer.next_layer_input_scale,
                                         write_y=write_y)
    return (_Fp8HiddenStates(norm_out, fp8) if write_y else fp8), residual


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

        assert not apply_routing, "Qwen3NextGate routing is called inside MoE"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The low-M kernels return the input dtype, so cublas_mm below still
        # owns any out_dtype promotion.
        if _should_apply_low_m_gemm(
                hidden_states, self) and self.out_dtype == hidden_states.dtype:
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
    asks to defer and no collective follows, or asks the MoE-output all-reduce
    to fold it (``fold_shared_expert``), so that a downstream consumer can
    fold that expression into a kernel it already runs over these rows. With
    ``do_finalize=False`` ``routed_output`` is the experts' un-finalized
    ``(fc2, topk_weights, expanded_idx_to_permuted_idx)``, which the MoE-output
    all-reduce finalizes.
    """

    routed_output: torch.Tensor | Tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor]
    shared_output: torch.Tensor
    shared_gate_logits: torch.Tensor
    # The un-finalized fc2 rows are in the Lamport FC2 buffer (the shared
    # expert's stream fenced): the all-reduce polls them.
    fc2_lamport: bool = False


def _with_moe_backend(model_config: ModelConfig, backend: str) -> ModelConfig:
    """``model_config`` with another MoE backend for one layer's experts."""
    if model_config.moe_backend == backend:
        return model_config
    layer_config = copy.copy(model_config)
    layer_config._frozen = False
    layer_config.moe_backend = backend
    layer_config._frozen = True
    return layer_config


# Context-sized MoE inputs (at least this many tokens) are quantized for the
# routed experts (NVFP4) and the shared expert (FP8) in one pass.
_MOE_INPUT_DUAL_QUANT_MIN_TOKENS = 1024


def _shared_expert_default_tactics_exist(gate_up_weight: torch.Tensor,
                                         down_weight: torch.Tensor) -> bool:
    """Whether the warp-MMA kernel has a heuristic tactic (``low_m_mma_gemm.default_tactic``, which an untuned
    launch runs) for both launches of the shared expert's low-M path at up to ``SHARED_EXPERT_MAX_M`` rows (the
    shared memory grows with the rows): the SwiGLU gate_up launch over the [2I, K] weight and down_proj over
    [H, I].

    Args:
        gate_up_weight: The fused gate_up weight [2I, K].
        down_weight: The down_proj weight [H, I].

    Returns:
        False where ``default_tactic`` raises for either launch.
    """
    try:
        default_tactic(SHARED_EXPERT_MAX_M, gate_up_weight.shape[0],
                       gate_up_weight.shape[1], True, True)
        default_tactic(SHARED_EXPERT_MAX_M, down_weight.shape[0],
                       down_weight.shape[1], True)
    except ValueError:
        return False
    return True


class Qwen3NextSparseMoeBlock(nn.Module):

    # The shared-expert combine can be handed to the MoE-output all-reduce.
    has_shared_expert = True
    # The Qwen3.5 decode kernels (_uses_decode_kernels): the BF16 draft experts' output is finalized in the
    # MoE-output all-reduce too.
    decode_kernels = False
    # Context-sized inputs may be quantized once for both expert paths
    # (_moe_input_dual_quant_supported).
    moe_input_dual_quant = False
    # TP decode kernels: the model's shared Lamport FC2 buffer, which the
    # experts of a fold call write and the MoE-output all-reduce polls.
    lamport_fc2: Optional[LamportFc2Buffer] = None
    # Decided once the weights are loaded: the experts take lamport_fc2.
    _experts_take_lamport_fc2 = False
    # TP decode: up to this many rows (the largest CUDA-graph decode step's) the
    # shared expert's two launches run on their own warp-MMA races
    # (raced_shared_expert); 0: the low-M race of the plain projections decides.
    _shared_expert_raced_rows = 0
    # Zeroed after the shared expert of a Lamport FC2 call (allocated on first use).
    _lamport_fence: Optional[torch.Tensor] = None
    # Whether a context-sized input is quantized once for both expert paths
    # (_moe_input_dual_quant); decided in cache_derived_state.
    _dual_quant_input = False
    # Decode kernels with MTP: rows from which the shared expert running beside
    # the routed experts takes the SMs their persistent grid can leave free
    # (_reserve_sms_for_shared_expert); 0: never.
    _shared_expert_reserve_rows = 0

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
        shared_gate_stream: Optional[torch.cuda.Stream] = None,
        moe_input_dual_quant: bool = False,
    ):
        """``shared_gate_stream``: the shared-expert gate GEMV runs there, beside
        the shared expert, instead of after it. ``moe_input_dual_quant``: see
        ``_moe_input_dual_quant`` (not with attention DP)."""
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
        self.decode_kernels = _uses_decode_kernels(model_config)
        self.moe_input_dual_quant = (moe_input_dual_quant
                                     and not self.enable_attention_dp)

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy,
                                   dtype=config.torch_dtype)

        self.aux_stream = aux_stream
        if self.decode_kernels:
            self._shared_expert_reserve_rows = _large_step_rows(model_config)

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
        expert_quant_config = None
        if layer_idx is not None:
            expert_quant_config = moe_experts_quant_config(
                model_config, f"model.layers.{layer_idx}.mlp.experts")
        moe_model_config = model_config
        # The MTP layer's BF16 experts (no per-layer quant entry in a
        # MIXED_PRECISION NVFP4 checkpoint) run as NVFP4 on TRTLLM-gen when the
        # speculative config asks for it (lossy: drafts only).
        nvfp4_draft_experts = False
        spec_config = model_config.spec_config
        n_hidden = getattr(config, "num_hidden_layers", None)
        if (spec_config is not None
                and spec_config.spec_dec_mode.is_mtp_one_model()
                and spec_config.draft_moe_dtype == "nvfp4"
                and layer_idx is not None and n_hidden is not None
                and layer_idx >= n_hidden):
            from ..modules.draft_experts_nvfp4 import experts_unquantized
            nvfp4_draft_experts = (
                _experts_excluded_from_quant(model_config, layer_idx)
                or experts_unquantized(model_config, expert_quant_config))
        if nvfp4_draft_experts:
            from ..modules.draft_experts_nvfp4 import quant_config_for
            expert_quant_config = quant_config_for(
                model_config.quant_config.kv_cache_quant_algo)
            weight_loading_mode = MoEWeightLoadingMode.VANILLA
            moe_model_config = _with_moe_backend(model_config, "TRTLLM")
        elif _experts_excluded_from_quant(model_config, layer_idx):
            # These experts end up bf16 whatever the per-layer entry says, so
            # build them on CUTLASS, the only backend that serves bf16
            # unconditionally.
            expert_quant_config = QuantConfig(kv_cache_quant_algo=model_config.
                                              quant_config.kv_cache_quant_algo)
            if model_config.moe_backend != "CUTLASS":
                moe_model_config = _with_moe_backend(model_config, "CUTLASS")
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
        if nvfp4_draft_experts:
            from ..modules.draft_experts_nvfp4 import wrap_experts_loader
            wrap_experts_loader(self.experts)
            logger.info(f"Layer {layer_idx} (MTP) MoE experts: bf16 checkpoint "
                        "weights quantized to NVFP4 at load time for "
                        "moe_backend=TRTLLM")
        # Qwen3.5's TP decode takes the route front right above the one-launch
        # front's steps.
        self._decode_front = DecodeFront(
            route_front_steps(
                model_config.cuda_graph_batch_sizes,
                spec_config.tokens_per_gen_step
                if spec_config is not None else 1,
                min_rows=(FUSED_FRONT_MAX_TOKENS +
                          1) if _uses_tp_decode_kernels(model_config) else
                ROUTER_MAX_TOKENS + 1),
            decode_kernels=_uses_decode_kernels(model_config),
            # The decode engine routes the steps it takes from the router
            # GEMM's per-tile candidates.
            engine_route=_uses_tp_decode_kernels(model_config))

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
        # (stream, x-ready event, gate-done event) of the shared-expert gate.
        self._shared_gate_aux = None
        if shared_gate_stream is not None:
            self._shared_gate_aux = (shared_gate_stream, torch.cuda.Event(),
                                     torch.cuda.Event())
        # Decided once the weights are loaded (cache_derived_state).
        # Whether the experts can hand back their un-finalized output
        # (do_finalize=False) for the MoE-output all-reduce to finalize.
        self.can_defer_finalize = False
        # The shared expert's low-M weights: (SwiGLU-paired gate_up, gate weight
        # padded to 8 rows), or None when the shared expert does not qualify.
        self._shared_expert_low_m: Optional[Tuple[torch.Tensor,
                                                  torch.Tensor]] = None
        self._shared_expert_raced_rows = (
            min(SHARED_EXPERT_MAX_M, _decode_step_rows(model_config))
            if _uses_tp_decode_kernels(model_config) else 0)

    def cache_derived_state(self) -> None:
        self.can_defer_finalize = self._experts_defer_finalize()
        self._experts_take_lamport_fc2 = self._experts_take_lamport()
        self._shared_expert_low_m = self._shared_expert_low_m_weights()
        self._dual_quant_input = self._moe_input_dual_quant_supported()

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _experts_defer_finalize(self) -> bool:
        """trtllm-gen NVFP4 experts with in-kernel routing (their do_finalize=False output carries valid top-k
        weights) and no MoE communication / load balancer. With the Qwen3.5 decode kernels also the BF16
        trtllm-gen experts (the MTP draft layer), whose routing kernel writes the BF16 top-k weights it returns with
        ``do_finalize=False`` whether it routes from the logits or from precomputed top-k."""
        from ..moe.fused_moe.configurable_moe import ConfigurableMoE
        from ..moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
        from ..moe.fused_moe.interface import FORCE_SEPARATED_ROUTING
        experts = self.experts
        backend = experts.backend if isinstance(experts,
                                                ConfigurableMoE) else experts
        if self.decode_kernels and self._bf16_experts_defer_finalize(
                experts, backend):
            return True
        return (isinstance(backend, TRTLLMGenFusedMoE) and backend.has_nvfp4
                and not self.enable_attention_dp
                and not backend._supports_load_balancer() and
                (not isinstance(experts, ConfigurableMoE) or
                 (experts.comm is None and experts.layer_load_balancer is None))
                and not experts.routing_method.requires_separated_routing
                and not FORCE_SEPARATED_ROUTING)

    def _reserve_sms_for_shared_expert(self, num_rows: int) -> None:
        """Ask the routed experts' backend to keep SMs free of its persistent
        grid for the shared expert on this forward
        (``MoEImplBase.reserve_sms_for_concurrent_kernels``, up to its
        ``sm_reserve_ceiling()``): only where the shared expert runs beside them
        (multi-stream on the aux stream) on a decode step of at least
        ``_shared_expert_reserve_rows`` rows; otherwise 0, so the grid keeps
        every SM. Re-asked per forward: a captured graph bakes the launch's
        reserve in."""
        from ..moe.fused_moe.configurable_moe import ConfigurableMoE
        if (not self._shared_expert_reserve_rows
                or not isinstance(self.experts, ConfigurableMoE)
                or torch.compiler.is_compiling()):
            return
        backend = self.experts.backend
        beside = (do_multi_stream() and self.aux_stream is not None
                  and num_rows >= self._shared_expert_reserve_rows)
        backend.reserve_sms_for_concurrent_kernels(
            backend.sm_reserve_ceiling() if beside else 0)

    def _experts_take_lamport(self) -> bool:
        """Experts that take ``lamport_fc2`` (``takes_lamport_fc2_out``): an
        un-finalized call may write its FC2 rows there and then returns the
        buffer as its rows, else its own rows."""
        from ..moe.fused_moe.configurable_moe import ConfigurableMoE
        experts = self.experts
        return (self.can_defer_finalize and isinstance(experts, ConfigurableMoE)
                and experts.backend.takes_lamport_fc2_out)

    def _bf16_experts_defer_finalize(self, experts: nn.Module,
                                     backend: nn.Module) -> bool:
        """BF16 trtllm-gen experts, no attention DP, no MoE communication, load balancer or
        model-owned combine: the MoE-output all-reduce can finalize their (fc2 rows, BF16 top-k weights,
        expanded -> permuted index) output.

        Args:
            experts: The routed experts module (a ``ConfigurableMoE`` or a backend).
            backend: Its MoE backend (``experts`` itself when it is not a ``ConfigurableMoE``).

        Returns:
            Whether the MoE-output all-reduce can finalize these experts' un-finalized output.
        """
        from ..moe.fused_moe.configurable_moe import ConfigurableMoE
        from ..moe.fused_moe.trtllm_gen.trtllm_bf16 import \
            TrtllmTrtllmGenBf16Impl
        if not isinstance(backend, TrtllmTrtllmGenBf16Impl):
            return False
        if not backend.supports_unfinalized_output(
        ) or self.enable_attention_dp:
            return False
        if isinstance(experts, ConfigurableMoE) and (
                experts.comm is not None or experts.layer_load_balancer
                is not None or experts.unfinalized_combine_fn is not None):
            return False
        return True

    def _shared_expert_low_m_weights(
            self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Weights of the shared expert's two-launch low-M path (``fused_shared_expert``), built from the loaded
        (rescaled) weights, or None when the shared expert does not qualify (also where the warp-MMA kernel has
        no heuristic tactic for a launch, ``_shared_expert_default_tactics_exist``). A rebuild (weight reload)
        writes into the existing copies, whose addresses captured graphs hold.
        """
        mlp = self.shared_expert
        gate_up = getattr(mlp, "gate_up_proj", None)
        down = getattr(mlp, "down_proj", None)
        gate = self.shared_expert_gate
        if (gate_up is None or down is None or not warp_mma_supported()
                or low_m_gemm_dispatcher(gate_up) is None):
            return None
        # LoRA: GatedMLP always attaches LoRA layers (applied only with
        # lora_params, which keep the stock path), so they do not disqualify it.
        for proj in (gate_up, down):
            if (not isinstance(proj.quant_method, FP8QDQLinearMethod)
                    or proj.bias is not None
                    or proj.quant_method.get_static_input_scale(proj) is None):
                return None
        if (mlp.activation is not torch.nn.functional.silu
                or not mlp._is_plain_swiglu() or mlp.swiglu_limit is not None
                or gate.bias is not None):
            return None
        if not _shared_expert_default_tactics_exist(gate_up.weight,
                                                    down.weight):
            return None
        return fused_shared_expert_weights(gate_up.weight.data,
                                           gate.weight,
                                           down.weight,
                                           out=self._shared_expert_low_m)

    def _shared_expert_low_m_forward(
            self,
            x: torch.Tensor) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """(shared expert output, gate logits [M, 1]) from two warp-MMA launches: up to
        ``_shared_expert_raced_rows`` rows on their own races, else where the low-M race picked the warp-MMA
        GEMM for gate_up_proj and down_proj; else None (also for 0 rows, which the warp-MMA GEMM refuses, and
        under torch.compile: AutoTuner and CuTe DSL host code)."""
        weights = self._shared_expert_low_m
        if (weights is None or x.dim() != 2 or x.dtype != torch.bfloat16
                or x.shape[0] == 0 or torch.compiler.is_compiling()):
            return None
        mlp = self.shared_expert
        gate_up, down = mlp.gate_up_proj, mlp.down_proj
        down_input_scale = down.quant_method.get_static_input_scale(down)
        if (x.shape[0] <= self._shared_expert_raced_rows
                or (self._shared_expert_raced_rows
                    and AutoTuner.get().is_tuning_mode)):
            if self._shared_expert_raced_rows and AutoTuner.get(
            ).is_tuning_mode:
                self._tune_fused_shared_expert(x, gate_up, down,
                                               down_input_scale)
            return raced_shared_expert(
                x,
                weights,
                gate_up_weight_scale=gate_up.weight_scale,
                gate_up_input_scale=gate_up.quant_method.get_static_input_scale(
                    gate_up),
                down_weight=down.weight,
                down_weight_scale=down.weight_scale,
                down_input_scale=down_input_scale)
        dispatcher = low_m_gemm_dispatcher(gate_up)
        if dispatcher is None:
            return None
        return fused_shared_expert(
            dispatcher,
            x,
            weights,
            gate_up_weight=gate_up.weight,
            gate_up_weight_scale=gate_up.weight_scale,
            gate_up_input_scale=gate_up.quant_method.get_static_input_scale(
                gate_up),
            down_weight=down.weight,
            down_weight_scale=down.weight_scale,
            down_input_scale=down_input_scale)

    def _tune_fused_shared_expert(self, x: torch.Tensor, gate_up: Linear,
                                  down: Linear,
                                  down_input_scale: torch.Tensor) -> None:
        """Tuning mode: time the low-M races that ``fused_shared_expert`` reads for the rows above
        ``_shared_expert_raced_rows`` (the FP8 race of gate_up_proj with a BF16 input and of down_proj with an
        E4M3 input), whose stock calls never run while the raced path takes every tuning call. Nothing to tune
        where the raced rows reach the low-M GEMM's row bound, so fused_shared_expert never runs.

        Args:
            x: The tuning call's [M, K] BF16 shared-expert input.
            gate_up: The shared expert's fused gate_up projection (FP8, static input scale).
            down: Its down projection (FP8, static input scale).
            down_input_scale: down_proj's static input scale.
        """
        dispatcher = low_m_gemm_dispatcher(gate_up)
        if dispatcher is None or not low_m_gemm_takes_rows(
                gate_up, self._shared_expert_raced_rows + 1):
            return
        # The races time every row bucket of the low-M GEMM whatever rows they are called with.
        rows = min(x.shape[0], LOW_M_MAX_ROWS)
        dispatcher.apply_fp8(
            x[:rows], gate_up.weight, gate_up.weight_scale,
            gate_up.quant_method.get_static_input_scale(gate_up), x.dtype)
        hidden = torch.zeros((rows, down.weight.shape[1]),
                             dtype=torch.float8_e4m3fn,
                             device=x.device)
        dispatcher.apply_fp8(hidden, down.weight, down.weight_scale,
                             down_input_scale, x.dtype)

    def _moe_input_dual_quant_supported(self) -> bool:
        """Whether one pass can quantize the MoE input for both expert paths:
        the routed experts take an NVFP4 input quantized ahead of them
        (``ConfigurableMoE.nvfp4_input_scale``), and the shared expert's fused
        gate_up is an FP8 projection with a static input scale."""
        from ..moe.fused_moe.configurable_moe import ConfigurableMoE
        if not self.moe_input_dual_quant or not isinstance(
                self.experts, ConfigurableMoE):
            return False
        fp4_scale = self.experts.nvfp4_input_scale(1)
        if fp4_scale is None:
            return False
        mlp = self.shared_expert
        if mlp.split_gate_up or mlp._can_fuse_gate_up_swiglu():
            return False
        gate_up = mlp.gate_up_proj
        if (not isinstance(gate_up.quant_method, FP8QDQLinearMethod)
                or gate_up.dtype != torch.bfloat16
                or gate_up._locality_domain_weight_shards is not None):
            return False
        fp8_scale = gate_up.quant_method.get_static_input_scale(gate_up)
        if (fp8_scale is None or fp8_scale.dtype != torch.float32
                or fp8_scale.numel() != 1
                or fp8_scale.device != fp4_scale.device):
            return False
        return True

    def _moe_input_dual_quant(
        self, hidden_states: torch.Tensor, lora_params: Optional[dict]
    ) -> Optional[Tuple[Fp4QuantizedTensor, torch.Tensor]]:
        """(the routed experts' input as an unswizzled Fp4QuantizedTensor, the
        shared expert's E4M3 gate_up input) from one read of a context-sized
        BF16 MoE input, or None to quantize per path. The values equal
        fp4_quantize (linear SF) and static_quantize_e4m3_per_tensor."""
        # The compile check before the row count: under torch.compile the
        # count is symbolic, and comparing it would guard on the row bound.
        if (not self._dual_quant_input or lora_params
                or torch.compiler.is_compiling()
                or hidden_states.shape[0] < _MOE_INPUT_DUAL_QUANT_MIN_TOKENS):
            return None
        # None as well when the forward would run in more than one MoE chunk.
        fp4_scale = self.experts.nvfp4_input_scale(hidden_states.shape[0])
        if fp4_scale is None or not can_use_nvfp4_quant(hidden_states,
                                                        fp4_scale):
            return None
        gate_up = self.shared_expert.gate_up_proj
        fp4, sf, fp8 = nvfp4_fp8_dual_quant(
            hidden_states, fp4_scale,
            gate_up.quant_method.get_static_input_scale(gate_up))
        return Fp4QuantizedTensor(fp4,
                                  sf.view(fp4.shape[0], -1),
                                  is_sf_swizzled=False), fp8

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
        lora_params: Optional[dict] = None,
        defer_shared_expert_finalize: bool = False,
        fold_shared_expert: bool = False,
    ) -> torch.Tensor | _DeferredSharedExpertFinalize:
        """``do_finalize=False`` (``can_defer_finalize``): the experts' un-finalized output (fc2 rows, top-k
        weights, expanded -> permuted index) with the shared expert's output and gate logits, for a MoE-output
        all-reduce that finalizes them. ``fold_shared_expert`` (the caller runs that all-reduce with
        MoEAllReduceParams): the finalized routed output with the shared expert's output and gate logits."""
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        routed_hidden_states = hidden_states
        use_dp_padding = False
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.enable_attention_dp and self.mapping.tp_size > 1 and is_sm_120f(
        ):
            use_dp_padding = True
            # Padding improves context-phase MoE communication on SM120. Keep
            # it local to the routed experts: the shared expert operates on
            # this rank's real tokens and must retain the unpadded row count.
            routed_hidden_states = torch.nn.functional.pad(
                routed_hidden_states,
                (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0]))

        if not do_finalize and not (self.can_defer_finalize
                                    and lora_params is None):
            raise NotImplementedError(
                "do_finalize=False needs experts that can defer the finalize")
        if (not do_finalize and isinstance(self.experts, ConfigurableMoE)
                and hidden_states.shape[0] > self.experts.moe_max_num_tokens):
            # The MoE scheduler concatenates the outputs of its chunks, so a
            # chunked call hands the fold its finalized output.
            do_finalize = True

        # Context sizes: NVFP4 for the routed experts and E4M3 for the shared
        # expert's gate_up from one pass; the router and the shared-expert gate
        # keep the BF16 input.
        # Not with the SM120 attention-DP padding: the routed input then has more rows than the shared one.
        dual_quant = None if use_dp_padding else self._moe_input_dual_quant(
            hidden_states, lora_params)
        shared_input = hidden_states if dual_quant is None else dual_quant[1]

        # On the fold path the experts may write their FC2 rows into the
        # Lamport buffer, which the MoE-output all-reduce polls; the shared
        # expert's stream is then fenced (shared_fn below).
        lamport_out = None
        if (self.lamport_fc2 is not None and self._experts_take_lamport_fc2
                and not do_finalize
                and hidden_states.shape[0] <= self.lamport_fc2.max_rows
                and do_multi_stream() and self.aux_stream is not None
                and not torch.compiler.is_compiling()):
            lamport_out = self.lamport_fc2.get(
                self.hidden_dim, self.experts.intermediate_size_per_partition,
                self.top_k, self.experts.expert_size_per_partition,
                hidden_states.device)
        experts_kwargs = {} if lamport_out is None else {
            "lamport_fc2_out": lamport_out
        }

        def _compute_routed_output():
            experts = self.experts
            out = self._decode_front(
                self.gate,
                experts,
                routed_hidden_states,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                do_finalize=do_finalize,
                **experts_kwargs,
            )
            if out is not None:
                return out
            router_logits = self.gate(routed_hidden_states)
            return experts(
                routed_hidden_states if dual_quant is None else dual_quant[0],
                router_logits,
                output_dtype=routed_hidden_states.dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                do_finalize=do_finalize,
                **experts_kwargs,
            )

        def _compute_shared_output():
            if not lora_params:
                low_m = self._shared_expert_low_m_forward(hidden_states)
                if low_m is not None:
                    return low_m
            if (self._shared_gate_aux is not None and do_multi_stream()
                    and not torch.compiler.is_compiling()):
                # This runs on the chain's aux stream once x is ready; the gate
                # GEMV + split-K reduce fork onto their own stream.
                gate_stream, x_ready, gate_done = self._shared_gate_aux
                x_ready.record()
                with torch.cuda.stream(gate_stream):
                    x_ready.wait()
                    shared_expert_gate_logits = self.shared_expert_gate(
                        hidden_states)
                    gate_done.record()
                shared_expert_output = self.shared_expert(
                    shared_input,
                    lora_params=lora_params,
                )
                gate_done.wait()
                return shared_expert_output, shared_expert_gate_logits
            shared_expert_output = self.shared_expert(
                shared_input,
                lora_params=lora_params,
            )
            shared_expert_gate_logits = self.shared_expert_gate(hidden_states)
            return shared_expert_output, shared_expert_gate_logits

        shared_fn = _compute_shared_output
        if lamport_out is not None:
            if self._lamport_fence is None:
                self._lamport_fence = torch.zeros((1, ),
                                                  dtype=torch.int32,
                                                  device=hidden_states.device)

            def shared_fn():
                outputs = _compute_shared_output()
                # The Lamport fold reads these outputs without a dependency
                # wait: this plain (non-PDL) kernel orders the all-reduce after
                # the aux-stream kernels.
                self._lamport_fence.zero_()
                return outputs

        self._reserve_sms_for_shared_expert(hidden_states.shape[0])
        (final_hidden_states,
         shared_expert_outputs) = maybe_execute_in_parallel(
             _compute_routed_output,
             shared_fn,
             self.event_dict[EventType.Main],
             self.event_dict[EventType.MoeShared],
             self.aux_stream,
             disable_on_compile=True,
         )

        shared_expert_output, shared_expert_gate_logits = shared_expert_outputs
        if not do_finalize:
            return _DeferredSharedExpertFinalize(
                tuple(final_hidden_states),
                shared_expert_output,
                shared_expert_gate_logits,
                fc2_lamport=rows_in_lamport(final_hidden_states[0],
                                            lamport_out),
            )
        if fold_shared_expert:
            return _DeferredSharedExpertFinalize(
                final_hidden_states,
                shared_expert_output,
                shared_expert_gate_logits,
            )
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

    # A dense MLP has no finalize to defer and no shared expert to fold.
    can_defer_finalize = False
    has_shared_expert = False

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
        fold_shared_expert=False,
    ):
        assert not (defer_shared_expert_finalize or fold_shared_expert), \
            "a dense MLP has no shared-expert finalize to defer"
        all_rank_num_tokens = (attn_metadata.all_rank_num_tokens
                               if attn_metadata is not None else None)
        return self.mlp(
            hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            final_all_reduce_params=all_reduce_params,
            lora_params=lora_params,
        )


def _create_mlp(model_config, aux_stream, layer_idx, shared_gate_stream=None):
    """Create the appropriate MLP for this layer: MoE or dense GatedMLP."""
    config = model_config.pretrained_config
    num_experts = getattr(config, "num_experts", 0) or 0
    if num_experts > 0:
        return Qwen3NextSparseMoeBlock(
            model_config,
            aux_stream,
            layer_idx=layer_idx,
            shared_gate_stream=shared_gate_stream,
            moe_input_dual_quant=_uses_prefill_kernels(model_config))
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


class _Qwen3NextDecoderLayerBase(DecoderLayer):
    """Norms, all-reduce and fusion choices shared by the GDN and full-attention
    decoder layers; ``forward`` runs ``mixer_forward`` (attention / GDN) and the
    MLP around them."""

    # Set from the next layer by Qwen3NextModel.cache_derived_state: the static
    # scale of its FP8 input projection, which the MoE-output all-reduce then
    # quantizes for, and whether it also reads the BF16 input.
    next_layer_input_scale: Optional[torch.Tensor] = None
    next_layer_reads_bf16 = False
    # TP decode kernels: the weight of the next layer's input projection, which
    # the MNNVL MoE-output all-reduce pulls into L2 from l2_prefetch_min_rows
    # rows, except in the rows the resident-weight GEMM takes it.
    next_layer_prefetch_weight: Optional[torch.Tensor] = None
    l2_prefetch_min_rows = 0
    next_layer_resident_rows: Tuple[int, int] = (1, 0)
    # Attention-DP decode kernels: the post-MoE chain runs as one launch
    # (_post_moe_norm_fold).
    post_moe_norm_fold = False

    def _init_common(self, model_config: ModelConfig[Qwen3NextConfig],
                     layer_idx: int, aux_stream: torch.cuda.Stream,
                     shared_gate_stream: Optional[torch.cuda.Stream]) -> None:
        self.model_config = model_config
        config = model_config.pretrained_config
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp
        self.hidden_size = config.hidden_size

        self.mlp = _create_mlp(model_config, aux_stream, layer_idx,
                               shared_gate_stream)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype,
                                       use_gemma=True)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype,
                                                use_gemma=True,
                                                quantize_type="nvfp4")
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy,
                                   dtype=config.torch_dtype)
        # Lossy E4M3 payload of the residual RMSNorm all-reduces.
        self.allreduce_fp8_payload = _allreduce_fp8_payload(model_config)

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
        # When PRE_MOE_FUSION is on, the mixer's output projection runs the
        # all-reduce + residual + post_attention_layernorm.
        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)

    def mixer_forward(self, position_ids, hidden_states, attn_metadata,
                      spec_metadata, lora_params, all_reduce_params, **kwargs):
        raise NotImplementedError

    def set_expert_prefetch(self,
                            prefetch: ExpertL2Prefetch,
                            issue_budget_ns: int = 0) -> None:
        """Prefetch the routed experts this layer used at the previous decode
        step into L2, early in the layer (the mixer's own hook point), issuing
        for at most ``issue_budget_ns`` in budgeted steps (0: no limit)."""
        mlp = self.mlp
        if not isinstance(mlp, Qwen3NextSparseMoeBlock):
            return
        from ..moe.fused_moe.configurable_moe import ConfigurableMoE
        experts = mlp.experts
        prefetch.register(
            self.layer_idx, experts.backend if isinstance(
                experts, ConfigurableMoE) else experts, issue_budget_ns)
        mlp._decode_front.set_expert_prefetch(prefetch, self.layer_idx)
        self._attach_expert_prefetch(prefetch)

    def _attach_expert_prefetch(self, prefetch: ExpertL2Prefetch) -> None:
        raise NotImplementedError

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        kept_rows: Optional[DraftKeptRows] = None,
        **kwargs,
    ) -> torch.Tensor:
        """``kept_rows``: the MLP and what follows it run on these rows only
        (an MTP draft pass, speculative/kept_rows.py); the caller restores
        ``attn_metadata.all_rank_num_tokens``."""
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False

        if self.fusion_config.PRE_MOE_FUSION:
            # all-reduce + residual + post_attention_layernorm in the output
            # projection (with the projection GEMM where it fuses).
            norm = self.post_attention_layernorm
            hidden_states, residual = self.mixer_forward(
                position_ids, hidden_states, attn_metadata, spec_metadata,
                lora_params,
                AllReduceParams(fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                                residual=residual,
                                norm_weight=_fused_norm_weight(norm),
                                eps=norm.variance_epsilon,
                                fp8_payload=self.allreduce_fp8_payload),
                **kwargs)
        else:
            hidden_states = self.mixer_forward(
                position_ids, hidden_states, attn_metadata, spec_metadata,
                lora_params,
                AllReduceParams(
                    enable_allreduce=not self.disable_attn_allreduce), **kwargs)
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        if kept_rows is not None:
            hidden_states, residual = gather_rows2(hidden_states, residual,
                                                   kept_rows.ids)
            if kept_rows.all_rank_counts is not None:
                attn_metadata.all_rank_num_tokens = kept_rows.all_rank_counts

        # The MoE-output all-reduce combines the shared expert, and finalizes
        # the experts' output where they can hand it back un-finalized.
        fold_shared_expert = (self.fusion_config.POST_MOE_FUSION
                              and self.mlp.has_shared_expert
                              and self.allreduce.supports_moe_fold(
                                  hidden_states.shape[0], self.hidden_size))
        do_finalize = not (fold_shared_expert and lora_params is None
                           and self.mlp.can_defer_finalize)
        # do_finalize=False only on the fold path: _post_moe_allreduce below
        # consumes the un-finalized rows (and resets a Lamport FC2 buffer).
        # The MoE block defers its shared-expert finalize to
        # _post_moe_norm_fold where no all-reduce follows.
        defer_shared_expert_finalize = (
            self.post_moe_norm_fold and lora_params is None
            and not torch.compiler.is_compiling()
            and not (spec_metadata is not None
                     and spec_metadata.is_layer_capture(self.layer_idx)))
        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
            lora_params=lora_params,
            defer_shared_expert_finalize=defer_shared_expert_finalize,
            fold_shared_expert=fold_shared_expert,
        )

        if self.fusion_config.POST_MOE_FUSION:
            hidden_states, residual = _post_moe_allreduce(
                self, hidden_states, residual)
        elif isinstance(hidden_states, _DeferredSharedExpertFinalize):
            hidden_states, residual = _post_moe_norm_fold(
                self, hidden_states, residual)
        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(
                    hidden_states, residual)
        return hidden_states, residual


class Qwen3NextLinearDecoderLayer(_Qwen3NextDecoderLayerBase):
    # Issues the expert prefetch at the layer's start, while the projections,
    # the mixer and the all-reduce leave DRAM idle.
    expert_prefetch: Optional[ExpertL2Prefetch] = None

    linear_attn_cls = Qwen3NextGatedDeltaNet

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
        shared_gate_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        resident_min_rows = _resident_min_rows(model_config)
        # in_proj_ba's replica from one request more.
        resident_ba_min_rows = resident_min_rows + 1 if resident_min_rows else 0
        self.linear_attn = self.linear_attn_cls(
            model_config,
            aux_stream,
            layer_idx,
            fused_verify=_uses_decode_kernels(model_config),
            resident_min_rows=resident_min_rows,
            resident_ba_min_rows=resident_ba_min_rows,
            verify_tcgen05_min_rows=_verify_tcgen05_min_rows(model_config)
            if _uses_decode_kernels(model_config) else 0,
            fused_prefill=_uses_prefill_kernels(model_config))
        self._init_common(model_config, layer_idx, aux_stream,
                          shared_gate_stream)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: Union[torch.Tensor, _Fp8HiddenStates],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(hidden_states, _Fp8HiddenStates):
            # in_proj_qkvz's E4M3 input, made by the previous layer's all-reduce.
            hidden_states, kwargs["h_fp8"] = hidden_states
        if self.expert_prefetch is not None:
            self.expert_prefetch.prefetch(self.layer_idx,
                                          hidden_states.shape[0])
        return super().forward(position_ids, hidden_states, attn_metadata,
                               residual, *args, **kwargs)

    def _attach_expert_prefetch(self, prefetch: ExpertL2Prefetch) -> None:
        self.expert_prefetch = prefetch

    def mixer_forward(self, position_ids, hidden_states, attn_metadata,
                      spec_metadata, lora_params, all_reduce_params, **kwargs):
        if hidden_states.shape[0] == 0:
            if all_reduce_params.fusion_op == AllReduceFusionOp.NONE:
                return hidden_states
            return hidden_states, all_reduce_params.residual
        return self.linear_attn(hidden_states,
                                attn_metadata,
                                spec_metadata=spec_metadata,
                                all_reduce_params=all_reduce_params,
                                **kwargs)


class Qwen3NextAttention(Qwen3Attention):

    def __init__(self,
                 model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int,
                 fuse_qk_norm_rope: bool,
                 context_last_rows_only: bool = False):
        decode_kernels = _uses_decode_kernels(model_config)
        prefill_kernels = _uses_prefill_kernels(model_config)
        # FP8 paged context attention stages each K / V tile's whole head.
        stage_kv_head_dim = (model_config.pretrained_config.head_dim
                             if prefill_kernels else 0)
        super().__init__(model_config,
                         layer_idx,
                         fuse_qk_norm_rope=fuse_qk_norm_rope,
                         attn_output_gate=True,
                         use_gemma_rms_norm=True,
                         use_gqa_decode_fmha=decode_kernels,
                         cache_context_plan=decode_kernels
                         and not model_config.mapping.enable_attention_dp,
                         fp8_output_gate=prefill_kernels,
                         context_fmha_head_dim_per_stage_kv=stage_kv_head_dim,
                         context_last_rows_only=context_last_rows_only)
        self._fuse_qk_norm_rope_gate = True

    # Issues the layer's expert prefetch right before o_proj, after the
    # attention's KV stream, so that the two do not compete for DRAM.
    expert_prefetch: Optional[ExpertL2Prefetch] = None

    def forward_impl(self, q: torch.Tensor, k: Optional[torch.Tensor],
                     v: Optional[torch.Tensor], *args: object,
                     **kwargs: object) -> torch.Tensor:
        """``Attention.forward_impl``, then this layer's expert prefetch (on
        its side stream).

        Args:
            q: Query (or fused QKV) rows.
            k: Key rows, or None with fused QKV.
            v: Value rows, or None with fused QKV.
            *args: ``Attention.forward_impl``'s other positional arguments.
            **kwargs: Its keyword arguments.

        Returns:
            The attention output rows.
        """
        attn_output = super().forward_impl(q, k, v, *args, **kwargs)
        if self.expert_prefetch is not None:
            self.expert_prefetch.prefetch(self.layer_idx, attn_output.shape[0])
        return attn_output


class Qwen3NextFullAttentionDecoderLayer(_Qwen3NextDecoderLayerBase):

    attention_mask = PredefinedAttentionMask.CAUSAL

    def __init__(self,
                 model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int,
                 aux_stream: torch.cuda.Stream,
                 shared_gate_stream: Optional[torch.cuda.Stream] = None,
                 context_last_rows_only: bool = False):
        super().__init__()
        self.self_attn = Qwen3NextAttention(
            model_config,
            layer_idx=layer_idx,
            # Gemma-style QK-norm is now supported by the fused qk_norm_rope
            # kernel (use_gemma path), so fuse instead of running separate
            # split + q/k RMSNorm + RoPE kernels.
            fuse_qk_norm_rope=True,
            context_last_rows_only=context_last_rows_only,
        )
        self._init_common(model_config, layer_idx, aux_stream,
                          shared_gate_stream)

    def _attach_expert_prefetch(self, prefetch: ExpertL2Prefetch) -> None:
        self.self_attn.expert_prefetch = prefetch

    def mixer_forward(self, position_ids, hidden_states, attn_metadata,
                      spec_metadata, lora_params, all_reduce_params, **kwargs):
        kwargs.setdefault("attention_mask", self.attention_mask)
        return self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            **kwargs,
        )


class Qwen3NextMTPHead(nn.Module):

    _draft_head_low_m: Optional[LowMGemmDispatcher] = None

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
        # This head's own low-M dispatcher (Qwen3.5 decode kernels, no attention DP). Only the draft passes call this
        # head (the target's logits come from the LM head directly), so the verify GEMM keeps its path.
        self._draft_head_low_m = (
            LowMGemmDispatcher() if _uses_decode_kernels(model_config)
            and not model_config.mapping.enable_attention_dp else None)

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
        logits = None
        if (self._draft_head_low_m is not None
                and hidden_states.shape[0] <= _DRAFT_HEAD_LOW_M_MAX_ROWS
                and isinstance(lm_head, LMHead)
                and lm_head.tp_mode == TensorParallelMode.COLUMN
                and lm_head.bias is None and lm_head.lora is None
                and not lm_head._writes_output_dtype()):
            # This rank's vocabulary shard, the GEMM lm_head(...) runs below with gather_output off. None when the
            # rows are not a low-M candidate (more than 32 rows, alignment): then the normal path runs.
            logits = self._draft_head_low_m.apply(lm_head,
                                                  hidden_states,
                                                  lm_head.weight,
                                                  None,
                                                  force_active=True)
        if logits is None:
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
        # Only each context request's last row feeds the draft token, so with the prefill kernels this layer's
        # context attention computes only those rows.
        super().__init__(
            model_config,
            layer_idx,
            aux_stream_dict[AuxStreamType.Attention],
            aux_stream_dict.get(AuxStreamType.MoeSharedGate),
            context_last_rows_only=_uses_prefill_kernels(model_config))
        config = model_config.pretrained_config
        # The one-launch fc input (modules/q35_mtp_glue.py).
        self.decode_kernels = _uses_decode_kernels(model_config)
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
        self.shared_head = Qwen3NextMTPHead(model_config)
        # With POST_MOE_FUSION the MoE-output all-reduce applies residual +
        # shared_head.norm.
        self.next_layer_layernorm = self.shared_head.norm
        # Zero residual of the fused fc all-reduce + input_layernorm
        # (cache_derived_state); its residual output is then the fc sum.
        self._fc_zero_residual: Optional[torch.Tensor] = None

    @staticmethod
    def takes_kept_rows(model_config: ModelConfig) -> bool:
        """Whether the draft loop hands pass 0 its kept rows (``forward``'s
        ``kept_rows``): the decode kernels, with CUDA-graph decode steps of
        more than one request."""
        return (_uses_decode_kernels(model_config)
                and _decode_step_rows(model_config)
                > _rows_per_request(model_config))

    @staticmethod
    def promotes_target_state_beside(model_config: ModelConfig) -> bool:
        """Whether the target's recurrent-state promotion after acceptance runs
        beside the draft loop: this layer (full attention + MoE) reads no
        recurrent state; with the TP decode kernels."""
        return _uses_tp_decode_kernels(model_config)

    def cache_derived_state(self) -> None:
        fc = self.fc
        if (fc.tp_mode == TensorParallelMode.ROW and fc.tp_size > 1
                and fc.reduce_output and self._fc_zero_residual is None):
            rows = min(self.model_config.max_num_tokens,
                       _FC_FUSED_NORM_MAX_TOKENS)
            self._fc_zero_residual = torch.zeros((rows, fc.out_features),
                                                 dtype=fc.dtype,
                                                 device=fc.weight.device)

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        kv_lens_increment: int = 0,
        kept_rows: Optional[DraftKeptRows] = None,
        **kwargs,
    ) -> torch.Tensor:
        """``kv_lens_increment``: ``attn_metadata.kv_lens_cuda[:n] += 1`` with
        ``n = kv_lens_increment``, deferred by the draft loop to this pass and
        applied before anything reads kv_lens (inside the fused input launch
        when this pass takes it). ``kept_rows``: past attention only these rows
        are computed; they come back compact or at their positions (the other
        rows zero)."""
        # Install this draft step's attention-DP token distribution: the draft
        # loop passes it as a kwarg and leaves attn_metadata alone, matching
        # Eagle3DraftModel.forward. The MoE below reads the per-rank counts off
        # attn_metadata, and an MTP Eagle draft runs one token per sequence
        # after step 0, so a stale target value mismatches collective sizes.
        previous_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        try:
            mapping = self.model_config.mapping
            tp_size = mapping.tp_size
            tp_rank = mapping.tp_rank
            chunked = tp_size > 1 and not mapping.enable_attention_dp
            if (self.decode_kernels and getattr(embed_tokens, "tp_size", 1) == 1
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
                    kv_lens=attn_metadata.kv_lens_cuda
                    if kv_lens_increment else None,
                    kv_rows=kv_lens_increment)
            else:
                if kv_lens_increment:
                    attn_metadata.kv_lens_cuda[:kv_lens_increment] += 1
                hidden_states = self._fc_input(input_ids, hidden_states,
                                               embed_tokens, chunked, tp_size,
                                               tp_rank)

            residual = None
            zero = self._fc_zero_residual
            rows = hidden_states.shape[0]
            if zero is not None and 0 < rows <= zero.shape[0]:
                # fc all-reduce + input_layernorm in one launch; the residual
                # output is the fc sum.
                hidden_states, residual = self.fc(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=zero[:rows],
                        norm_weight=_fused_norm_weight(self.input_layernorm),
                        eps=self.input_layernorm.variance_epsilon,
                        fp8_payload=self.allreduce_fp8_payload))
            else:
                hidden_states = self.fc(hidden_states)

            hidden_states, residual = super().forward(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                kept_rows=kept_rows,
                **kwargs,
            )
            if kept_rows is not None and not kept_rows.compact:
                full = hidden_states.new_zeros((rows, hidden_states.shape[-1]))
                full.index_copy_(0, kept_rows.ids, hidden_states)
                hidden_states = full
            # The base forward applied shared_head.norm (next_layer_layernorm).
            if spec_metadata is not None:
                spec_metadata.maybe_capture_hidden_states(
                    0, hidden_states, None)

            return hidden_states
        finally:
            # Shared with the target forward, so restore on the exception path
            # too; otherwise a failed draft step corrupts every later forward.
            if all_rank_num_tokens is not None or kept_rows is not None:
                attn_metadata.all_rank_num_tokens = previous_all_rank_num_tokens

    def _fc_input(self, input_ids, hidden_states, embed_tokens, chunked,
                  tp_size, tp_rank):
        """The fc input ``chunk(concat([norm(embed(ids)), norm(hidden)]))`` as separate kernels."""

        def norm_embeds():
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

    # TP decode kernels: the Lamport FC2 buffer its MoE layers share.
    lamport_fc2: Optional[LamportFc2Buffer] = None
    # TP decode kernels: the routed experts' cross-step L2 prefetch.
    expert_prefetch: Optional[ExpertL2Prefetch] = None
    decoder_layer_types = ALL_DECODER_LAYER_TYPES

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
        if _is_qwen3_5(model_config):
            # The shared-expert gate GEMV runs beside the shared expert (the
            # routed experts' early-launched FC1 fills the SMs).
            self.aux_stream_dict[
                AuxStreamType.MoeSharedGate] = torch.cuda.Stream()
        self.aux_stream = self.aux_stream_dict[AuxStreamType.Attention]
        self.preload_weight_modules = []
        if config.moe_backend == "TRTLLM":
            self.preload_weight_modules = [
                "experts",
                "routing_method",
                "all_reduce",
            ]

        replicated_embed = model_config.replicate_embedding
        if model_config.mapping.enable_attention_dp or replicated_embed:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            # A replicated table looks tokens up in one gather launch instead
            # of the masked lookup's kernels (bitwise).
            self.embed_tokens = Embedding(pretrained_config.vocab_size,
                                          pretrained_config.hidden_size,
                                          dtype=pretrained_config.torch_dtype,
                                          use_gather_kernel=replicated_embed)
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
        shared_gate_stream = self.aux_stream_dict.get(
            AuxStreamType.MoeSharedGate)
        self.layers = nn.ModuleList([
            self.decoder_layer_types[layer_types[layer_idx]](model_config,
                                                             layer_idx,
                                                             self.aux_stream,
                                                             shared_gate_stream)
            for layer_idx in range(pretrained_config.num_hidden_layers)
        ])
        self.num_hidden_layers = pretrained_config.num_hidden_layers
        if _uses_tp_decode_kernels(model_config):
            # Dense (GatedMLP) models have no routed experts to prefetch.
            if any(
                    isinstance(layer.mlp, Qwen3NextSparseMoeBlock)
                    for layer in self.layers):
                self._init_expert_prefetch(model_config)
            # Lamport FC2 calls: the slot all-reduce's rows, which poll per token.
            self.lamport_fc2 = LamportFc2Buffer(SLOT_MAX_ROWS)
            self.share_lamport_fc2(self.layers)

        self.norm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype,
            use_gemma=True,
        )

    def _init_expert_prefetch(self, model_config: ModelConfig) -> None:
        """The routed-expert L2 prefetch of the TP decode layouts, attached to every MoE layer."""
        self.aux_stream_dict[
            AuxStreamType.MoeExpertPrefetch] = torch.cuda.Stream()
        self.expert_prefetch = ExpertL2Prefetch(
            self.num_hidden_layers,
            min_rows=_EXPERT_PREFETCH_MIN_REQUESTS *
            _rows_per_request(model_config),
            max_rows=_EXPERT_PREFETCH_MAX_ROWS,
            top_k=model_config.pretrained_config.num_experts_per_tok,
            stream=self.aux_stream_dict[AuxStreamType.MoeExpertPrefetch],
            budget_rows=_expert_prefetch_budget_rows(model_config))
        for layer in self.layers:
            layer.set_expert_prefetch(
                self.expert_prefetch, _EXPERT_PREFETCH_GDN_BUDGET_NS
                if isinstance(layer, Qwen3NextLinearDecoderLayer) else
                _EXPERT_PREFETCH_ATTN_BUDGET_NS)

    def share_lamport_fc2(self, layers: Sequence[nn.Module]) -> None:
        """Hand ``lamport_fc2`` to the MoE blocks of ``layers``."""
        for layer in layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                layer.mlp.lamport_fc2 = self.lamport_fc2

    def cache_derived_state(self) -> None:
        # A layer's MoE-output all-reduce also quantizes the next layer's
        # input for its FP8 input projection (not with LoRA, whose adapters
        # read the BF16 input).
        layers = self.layers[:self.num_hidden_layers]
        for layer, next_layer in zip(layers, layers[1:]):
            layer.next_layer_input_scale = (_fp8_input_scale(next_layer)
                                            if self.model_config.lora_config
                                            is None else None)
            layer.next_layer_reads_bf16 = isinstance(
                next_layer, Qwen3NextLinearDecoderLayer)
        if _uses_tp_decode_kernels(self.model_config):
            self._cache_tp_decode_gates(layers)
            self._drop_expert_prefetch_without_record(layers)
        elif _uses_decode_kernels(self.model_config):
            self._cache_post_moe_norm_folds(layers)

    def _drop_expert_prefetch_without_record(
            self, layers: Sequence[nn.Module]) -> None:
        """Unregister from the expert prefetch the layers whose decode front
        never records their experts, so that they launch no prefetch on an
        empty or stale selection: the front records only in an un-finalized
        MoE call, which needs POST_MOE_FUSION, a shared expert, experts that
        defer their finalize and an all-reduce that folds the MoE output at
        some row count the prefetch takes (``_Qwen3NextDecoderLayerBase.forward``).

        Args:
            layers: The decoder layers (not the MTP layers).
        """
        prefetch = self.expert_prefetch
        if prefetch is None:
            return
        rows = range(prefetch.min_rows, prefetch.max_rows + 1)
        for layer in layers:
            mlp = layer.mlp
            if (not isinstance(mlp, Qwen3NextSparseMoeBlock)
                    or not prefetch.registered(layer.layer_idx)):
                continue
            # The MoE block's own cache_derived_state (can_defer_finalize) may
            # run after this one: ask the experts directly.
            records = (
                layer.fusion_config.POST_MOE_FUSION and mlp.has_shared_expert
                and mlp._experts_defer_finalize() and any(
                    layer.allreduce.supports_moe_fold(r, layer.hidden_size)
                    for r in rows))
            if not records:
                prefetch.unregister(layer.layer_idx)

    def _cache_post_moe_norm_folds(self, layers: Sequence[nn.Module]) -> None:
        """Attention DP: the layers whose MoE output, residual add,
        ``next_layer_layernorm`` and the next layer's FP8 quant run as one
        launch (_post_moe_norm_fold)."""
        from ..kernels.blackwell.gate_add_rmsnorm_fp8 import supports_hidden
        config = self.model_config.pretrained_config
        if (config.torch_dtype != torch.bfloat16
                or not supports_hidden(config.hidden_size)):
            return
        for layer in layers:
            norm = layer.next_layer_layernorm
            layer.post_moe_norm_fold = (layer.next_layer_input_scale is not None
                                        and isinstance(layer.mlp,
                                                       Qwen3NextSparseMoeBlock)
                                        and layer.mlp.has_shared_expert
                                        and norm is not None and norm.use_gemma
                                        and not norm.use_cuda_tile)

    def _cache_tp_decode_gates(self, layers: Sequence[nn.Module]) -> None:
        """Row gates of the input projections that follow a MoE-output
        all-reduce (see _RESIDENT_MIN_REQUESTS)."""
        from ..kernels.blackwell.resident_fp8_gemm import \
            MAX_ROWS as resident_max_rows
        min_rows = _resident_min_rows(self.model_config)
        for layer, next_layer in zip(layers, layers[1:]):
            projection = _fp8_input_projection(next_layer)
            # Only the MNNVL all-reduce prefetches; remove_weights (layer-wise
            # benchmark) leaves the projection without a weight.
            prefetch = (projection is not None
                        and layer.allreduce.mnnvl_allreduce is not None
                        and not projection._weights_removed)
            # A tensor, not the module: a module attribute would register the
            # projection as a second child of this layer.
            weight = projection.weight.detach() if prefetch else None
            layer.next_layer_prefetch_weight = (
                weight if weight is not None and weight.is_cuda
                and weight.is_contiguous() and weight.numel() > 0 else None)
            layer.l2_prefetch_min_rows = min_rows
            if _takes_resident_band(layer, projection, min_rows,
                                    resident_max_rows):
                # Its input arrives in E4M3 from this all-reduce: from min_rows
                # the resident-weight GEMM takes it instead of the low-M race.
                projection.resident_fp8_min_rows = min_rows
                layer.next_layer_resident_rows = (min_rows, resident_max_rows)
                dispatcher = low_m_gemm_dispatcher(projection)
                if dispatcher is not None:
                    dispatcher.limit_rows(projection, min_rows - 1)

    def post_load_weights(self) -> None:
        self.cache_derived_state()

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers[:self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mamba_metadata=mamba_metadata,
                lora_params=lora_params)
        self.after_decoder_layers()
        return hidden_states

    def after_decoder_layers(self) -> None:
        """What a forward runs after its last decoder layer, also for a caller
        that runs the layers itself (the layer-wise benchmark): join the side
        stream the layers fork for the expert L2 prefetch."""
        if self.expert_prefetch is not None:
            self.expert_prefetch.join()


@register_auto_model("Qwen3NextForCausalLM")
class Qwen3NextForCausalLM(SpecDecOneEngineForCausalLM[Qwen3NextModel,
                                                       Qwen3NextConfig]):
    # load_weights below only pre-processes the dict and delegates to the
    # generic loader, which tolerates a torch.compile wrapper in the tree.
    load_weights_strips_compile_wrapper = True
    model_specific_llm_args = frozenset(
        {"replicate_embedding", "mnnvl_allreduce_fp8_payload"})
    # The fast paths' TP key exchange: temperature rows take the four-draws Gumbel kernel; argmax splits rows.
    spec_decode_fast_gumbel = True
    spec_decode_argmax_row_split = True
    # Declared by a subclass (Qwen3_5MoeForCausalLM): prefill-sized calls run the fused GDN prefill conv, the MoE
    # input dual quant, FP8-output attention gates and staged-KV FP8 context attention (_uses_prefill_kernels).
    prefill_kernels = False
    # Declared by the Qwen3.5 subclasses: on SM 10.0 / 10.3 the attention, GDN, MoE and MTP draft layers build their
    # decode paths (_uses_decode_kernels).
    decode_kernels = False
    # The step-loop hooks of the speculative fast paths (supports_spec_decode_fast_paths): the overlap loop on the
    # execution stream, the sampler's Philox state staged with the step inputs, the timing events off the current
    # stream.
    spec_decode_fast_path_runtime_hooks = RuntimeHooks(
        loop_on_execution_stream=True,
        pack_spec_rng_uploads=True,
        iteration_timing_one_event=True,
        perf_metrics_timing_side_stream=True)
    # Construction-time declarations of its decode kernels in extra_attrs, kept out of a drafter built from another
    # checkpoint (which shares extra_attrs): the MoE decode options and the fused-norm all-reduces.
    target_only_extra_attrs = (MOE_DECODE_OPTIONS, "allreduce_mnnvl_fused_norm",
                               "allreduce_tp2_fused_norm")

    @classmethod
    def supports_spec_decode_fast_paths(
            cls, model_config: ModelConfig[Qwen3NextConfig]) -> bool:
        return _uses_decode_kernels(model_config)

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
    ):
        # The prefill and decode kernels were tuned on SM 10.0 / 10.3 only; the
        # decode kernels also only in the layouts _decode_kernels_layout names.
        if self.prefill_kernels and is_blackwell_datacenter():
            model_config.extra_attrs["qwen3_next_prefill_kernels"] = True
        if self.decode_kernels and is_blackwell_datacenter():
            if _decode_kernels_layout(model_config):
                model_config.extra_attrs["qwen3_next_decode_kernels"] = True
            else:
                logger.info("Qwen3.5 decode kernels off: a TP layout "
                            "without MNNVL (e.g. an HGX node)")
        decode_kernels = _uses_decode_kernels(model_config)
        if decode_kernels:
            # Its MNNVL all-reduces run the residual RMSNorm fusions on the CuTe DSL kernels.
            model_config.extra_attrs["allreduce_mnnvl_fused_norm"] = True
            # Its TP2 all-reduces without MNNVL run large residual RMSNorm fusions on Tp2AllReduceNorm.
            model_config.extra_attrs["allreduce_tp2_fused_norm"] = True
            # FP4 MoE: decode tuning; the engine releases the MoE-output all-reduce after its own wait from its
            # two-shot rows; CuTe DSL decode steps (every attention-DP rank's rows) stream FC2 weights evict-first.
            # Without attention DP its BF16 TRTLLM-Gen MoE layers (the MTP draft experts) take the BF16 decode MoE
            # engine for small logits-routed calls. The SMs the engine leaves to the shared expert are asked per
            # forward (Qwen3NextSparseMoeBlock._reserve_sms_for_shared_expert).
            no_adp = not model_config.mapping.enable_attention_dp
            model_config.extra_attrs[MOE_DECODE_OPTIONS] = MoEDecodeOptions(
                tuning=True,
                early_release_rows=TWO_SHOT_ROWS,
                fc2_weights_evict_first_rows=_decode_step_rows(model_config) *
                model_config.mapping.dp_size,
                bf16_decode_engine=no_adp)
            if no_adp:
                # Its temperature verify steps sample and accept in one launch where that launch takes the step.
                model_config.extra_attrs["spec_fused_sample_accept"] = True
            logger.info("Qwen3.5 decode kernels enabled")
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
            if self.model.lamport_fc2 is not None:
                self.model.share_lamport_fc2(self.draft_model.mtp_layers)

        if _is_qwen3_5(model_config):
            # Decode-sized projections race cuBLAS against the low-M GEMMs.
            prepare_low_m_gemm(
                self,
                force=True,
                per_rank_tuning=model_config.mapping.enable_attention_dp,
                whole_bucket_direct_tiles=_uses_tp_decode_kernels(model_config))

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
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        _precompute_fused_norm_weights(self)
        # Fold the static NVFP4 input quantize of a dense gate_up_proj into the
        # preceding add + RMSNorm.
        for layer in self.model.layers[:self.config.num_hidden_layers]:
            mlp = getattr(layer, "mlp", None)
            norm = getattr(layer, "post_attention_layernorm", None)
            if (isinstance(mlp, _DenseMlpAdapter) and norm is not None
                    and norm.is_nvfp4
                    and is_static_nvfp4_input_eligible(mlp.mlp.gate_up_proj)):
                norm.nvfp4_scale = mlp.mlp.gate_up_proj.input_scale
        # Quantize per token in the input norm when in_proj_qkvz is FP8 rowwise.
        for layer in self.model.layers[:self.config.num_hidden_layers]:
            linear_attn = getattr(layer, "linear_attn", None)
            if linear_attn is not None:
                layer.input_layernorm.fp8_per_token_out = isinstance(
                    linear_attn.in_proj_qkvz.quant_method,
                    FP8RowwiseLinearMethod)
