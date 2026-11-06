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
                    Tuple, Union)

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
from tensorrt_llm._utils import is_blackwell_datacenter, is_sm_120f
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...logger import logger
from ..attention.backends import AttentionMetadata
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduceParams, allgather)
from ..distributed.ops import MNNVLAllReduce
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import FP8QDQLinearMethod, Linear, TensorParallelMode
from ..modules.low_m_gemm import (_should_apply_low_m_gemm, apply_low_m_gemm,
                                  low_m_gemm_dispatcher, prepare_low_m_gemm)
from ..modules.low_m_mma_gemm import (fused_shared_expert,
                                      fused_shared_expert_weights,
                                      warp_mma_supported)
from ..modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from ..modules.multi_stream_utils import (do_multi_stream,
                                          maybe_execute_in_parallel)
from ..modules.rms_norm import RMSNorm
from ..moe.fused_moe import (BaseMoeRoutingMethod, MoEWeightLoadingMode,
                             RenormalizeMoeRoutingMethod,
                             RenormalizeNaiveMoeRoutingMethod,
                             RoutingMethodType, create_moe)
from ..moe.fused_moe.trtllm_gen.decode_front import (DecodeFront,
                                                     route_front_steps)
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, create_lm_head_tp_mapping
from .modeling_qwen3 import Qwen3Attention
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (DecoderModel, EagerFusionConfig,
                             moe_experts_quant_config, register_auto_model)

# Most rows of the MTP fc all-reduce fused with the input RMSNorm.
_FC_FUSED_NORM_MAX_TOKENS = 1024


def _is_qwen3_5(model_config: ModelConfig) -> bool:
    """A Qwen3.5 checkpoint (the Qwen3Next architecture with Qwen3.5 weights)."""
    return getattr(model_config.pretrained_config, "model_type",
                   "").startswith("qwen3_5")


def _uses_decode_kernels(model_config: ModelConfig) -> bool:
    """Whether the Qwen3.5 attention / GDN / draft decode paths are built:
    a Qwen3.5 checkpoint on SM 10.0 / 10.3."""
    return _is_qwen3_5(model_config) and is_blackwell_datacenter()


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
                  eps=norm.variance_epsilon)
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


def _with_moe_backend(model_config: ModelConfig, backend: str) -> ModelConfig:
    """``model_config`` with another MoE backend for one layer's experts."""
    if model_config.moe_backend == backend:
        return model_config
    layer_config = copy.copy(model_config)
    layer_config._frozen = False
    layer_config.moe_backend = backend
    layer_config._frozen = True
    return layer_config


class Qwen3NextSparseMoeBlock(nn.Module):

    # The shared-expert combine can be handed to the MoE-output all-reduce.
    has_shared_expert = True

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
        shared_gate_stream: Optional[torch.cuda.Stream] = None,
    ):
        """``shared_gate_stream``: the shared-expert gate GEMV runs there, beside
        the shared expert, instead of after it."""
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
        self._decode_front = DecodeFront(
            route_front_steps(
                model_config.cuda_graph_batch_sizes,
                spec_config.tokens_per_gen_step
                if spec_config is not None else 1))

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

    def cache_derived_state(self) -> None:
        self.can_defer_finalize = self._experts_defer_finalize()
        self._shared_expert_low_m = self._shared_expert_low_m_weights()

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _experts_defer_finalize(self) -> bool:
        """trtllm-gen NVFP4 experts with in-kernel routing (their do_finalize=False output carries valid top-k
        weights) and no MoE communication / load balancer."""
        from ..moe.fused_moe.configurable_moe import ConfigurableMoE
        from ..moe.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
        from ..moe.fused_moe.interface import FORCE_SEPARATED_ROUTING
        experts = self.experts
        backend = experts.backend if isinstance(experts,
                                                ConfigurableMoE) else experts
        return (isinstance(backend, TRTLLMGenFusedMoE) and backend.has_nvfp4
                and not self.enable_attention_dp
                and not backend._supports_load_balancer() and
                (not isinstance(experts, ConfigurableMoE) or
                 (experts.comm is None and experts.layer_load_balancer is None))
                and not experts.routing_method.requires_separated_routing
                and not FORCE_SEPARATED_ROUTING)

    def _shared_expert_low_m_weights(self):
        """Weights of the shared expert's two-launch low-M path (``fused_shared_expert``), built from the loaded
        (rescaled) weights, or None when the shared expert does not qualify. A rebuild (weight reload) writes
        into the existing copies, whose addresses captured graphs hold.
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
        return fused_shared_expert_weights(gate_up.weight.data,
                                           gate.weight,
                                           down.weight,
                                           out=self._shared_expert_low_m)

    def _shared_expert_low_m_forward(self, x: torch.Tensor):
        """(shared expert output, gate logits [M, 1]) from two warp-MMA launches, where the low-M race picked
        the warp-MMA GEMM for gate_up_proj and down_proj; else None."""
        weights = self._shared_expert_low_m
        if weights is None or x.dim() != 2 or x.dtype != torch.bfloat16:
            return None
        mlp = self.shared_expert
        gate_up, down = mlp.gate_up_proj, mlp.down_proj
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
            down_input_scale=down.quant_method.get_static_input_scale(down))

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
        use_dp_padding = False
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.enable_attention_dp and self.mapping.tp_size > 1 and is_sm_120f(
        ):
            use_dp_padding = True
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0]))

        if not do_finalize and not (self.can_defer_finalize
                                    and lora_params is None):
            raise NotImplementedError(
                "do_finalize=False needs experts that can defer the finalize")

        def _compute_routed_output():
            experts = self.experts
            out = self._decode_front(
                self.gate,
                experts,
                hidden_states,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                do_finalize=do_finalize,
            )
            if out is not None:
                return out
            router_logits = self.gate(hidden_states)
            return experts(
                hidden_states,
                router_logits,
                output_dtype=hidden_states.dtype,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                do_finalize=do_finalize,
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

        shared_expert_output, shared_expert_gate_logits = shared_expert_outputs
        if not do_finalize:
            return _DeferredSharedExpertFinalize(
                tuple(final_hidden_states),
                shared_expert_output,
                shared_expert_gate_logits,
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
        return Qwen3NextSparseMoeBlock(model_config,
                                       aux_stream,
                                       layer_idx=layer_idx,
                                       shared_gate_stream=shared_gate_stream)
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
                                                use_gemma=True)
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(mapping=model_config.mapping,
                                   strategy=model_config.allreduce_strategy,
                                   dtype=config.torch_dtype)
        # Lossy E4M3 payload of the residual RMSNorm all-reduces.
        self.allreduce_fp8_payload = _allreduce_fp8_payload(model_config)

        self.next_layer_layernorm: RMSNorm = None
        # Set from the next layer by Qwen3NextForCausalLM.cache_derived_state:
        # the static scale of its FP8 input projection, which the MoE-output
        # all-reduce then quantizes for, and whether it also reads the BF16 input.
        self.next_layer_input_scale: Optional[torch.Tensor] = None
        self.next_layer_reads_bf16 = False

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

        # The MoE-output all-reduce combines the shared expert, and finalizes
        # the experts' output where they can hand it back un-finalized.
        fold_shared_expert = (self.fusion_config.POST_MOE_FUSION
                              and self.mlp.has_shared_expert
                              and self.allreduce.supports_moe_fold(
                                  hidden_states.shape[0], self.hidden_size))
        do_finalize = not (fold_shared_expert and lora_params is None
                           and self.mlp.can_defer_finalize)
        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MOE_FUSION
                                      or self.mapping.tp_size == 1)),
            do_finalize=do_finalize,
            lora_params=lora_params,
            fold_shared_expert=fold_shared_expert,
        )

        if self.fusion_config.POST_MOE_FUSION:
            hidden_states, residual = _post_moe_allreduce(
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

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
        shared_gate_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.linear_attn = Qwen3NextGatedDeltaNet(
            model_config,
            aux_stream,
            layer_idx,
            fused_verify=_uses_decode_kernels(model_config))
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
        return super().forward(position_ids, hidden_states, attn_metadata,
                               residual, *args, **kwargs)

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

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int, fuse_qk_norm_rope: bool):
        decode_kernels = _uses_decode_kernels(model_config)
        super().__init__(model_config,
                         layer_idx,
                         fuse_qk_norm_rope=fuse_qk_norm_rope,
                         attn_output_gate=True,
                         use_gemma_rms_norm=True,
                         use_gqa_decode_fmha=decode_kernels,
                         cache_context_plan=decode_kernels
                         and not model_config.mapping.enable_attention_dp)
        self._fuse_qk_norm_rope_gate = True


class Qwen3NextFullAttentionDecoderLayer(_Qwen3NextDecoderLayerBase):

    def __init__(self,
                 model_config: ModelConfig[Qwen3NextConfig],
                 layer_idx: int,
                 aux_stream: torch.cuda.Stream,
                 shared_gate_stream: Optional[torch.cuda.Stream] = None):
        super().__init__()
        self.self_attn = Qwen3NextAttention(
            model_config,
            layer_idx=layer_idx,
            # Gemma-style QK-norm is now supported by the fused qk_norm_rope
            # kernel (use_gemma path), so fuse instead of running separate
            # split + q/k RMSNorm + RoPE kernels.
            fuse_qk_norm_rope=True,
        )
        self._init_common(model_config, layer_idx, aux_stream,
                          shared_gate_stream)

    def mixer_forward(self, position_ids, hidden_states, attn_metadata,
                      spec_metadata, lora_params, all_reduce_params, **kwargs):
        return self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            **kwargs,
        )


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
                         aux_stream_dict[AuxStreamType.Attention],
                         aux_stream_dict.get(AuxStreamType.MoeSharedGate))
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
        **kwargs,
    ) -> torch.Tensor:
        """``kv_lens_increment``: ``attn_metadata.kv_lens_cuda[:n] += 1`` with
        ``n = kv_lens_increment``, deferred by the draft loop to this pass and
        applied before anything reads kv_lens (inside the fused input launch
        when this pass takes it)."""
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
                **kwargs,
            )
            # The base forward applied shared_head.norm (next_layer_layernorm).
            if spec_metadata is not None:
                spec_metadata.maybe_capture_hidden_states(
                    0, hidden_states, None)

            return hidden_states
        finally:
            # Shared with the target forward, so restore on the exception path
            # too; otherwise a failed draft step corrupts every later forward.
            if all_rank_num_tokens is not None:
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
            ALL_DECODER_LAYER_TYPES[layer_types[layer_idx]](model_config,
                                                            layer_idx,
                                                            self.aux_stream,
                                                            shared_gate_stream)
            for layer_idx in range(pretrained_config.num_hidden_layers)
        ])
        self.num_hidden_layers = pretrained_config.num_hidden_layers

        self.norm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype,
            use_gemma=True,
        )

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
        return hidden_states


@register_auto_model("Qwen3NextForCausalLM")
class Qwen3NextForCausalLM(SpecDecOneEngineForCausalLM[Qwen3NextModel,
                                                       Qwen3NextConfig]):
    # load_weights below only pre-processes the dict and delegates to the
    # generic loader, which tolerates a torch.compile wrapper in the tree.
    load_weights_strips_compile_wrapper = True
    model_specific_llm_args = frozenset(
        {"replicate_embedding", "mnnvl_allreduce_fp8_payload"})

    @classmethod
    def supports_spec_decode_fast_paths(
            cls, model_config: ModelConfig[Qwen3NextConfig]) -> bool:
        return _uses_decode_kernels(model_config)

    def __init__(
        self,
        model_config: ModelConfig[Qwen3NextConfig],
    ):
        decode_kernels = _uses_decode_kernels(model_config)
        if decode_kernels:
            # Every AllReduce this model builds: no NCCL_SYMMETRIC.
            model_config.extra_attrs["allreduce_exclude_nccl_symmetric"] = True
            # Its FP4 MoE layers tune for the decode steps.
            model_config.extra_attrs["moe_decode_tuning"] = True
            # Its MNNVL all-reduces run the residual RMSNorm fusions on the CuTe DSL kernels.
            model_config.extra_attrs["allreduce_mnnvl_fused_norm"] = True
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

        if _is_qwen3_5(model_config):
            # Decode-sized projections race cuBLAS against the low-M GEMMs.
            prepare_low_m_gemm(
                self,
                force=True,
                per_rank_tuning=model_config.mapping.enable_attention_dp)

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
