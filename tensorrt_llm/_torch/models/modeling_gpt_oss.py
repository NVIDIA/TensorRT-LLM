# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Any, Dict, Literal, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
from transformers import GptOssConfig

from tensorrt_llm._utils import get_hf_rope_theta, get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

from ..attention.attention import PRUNE_LAST_TOKEN_TARGET, Attention
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import (PositionalEmbeddingParams,
                                            PredefinedAttentionMask, RopeParams)
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           allgather)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from ..moe.fused_moe import (MoEWeightLoadingMode, RenormalizeMoeRoutingMethod,
                             SwigluBiasActivation, TritonFusedMoE, create_moe,
                             is_moe_weight_owner)
from ..moe.fused_moe.static_eplb_route import DEP3_NORM_MXFP8_ENABLED
# isort: on
from ..modules.linear import Linear, TensorParallelMode
from ..modules.low_m_gemm import _MAX_M as LOW_M_MAX_M
from ..modules.low_m_gemm import (SMALL_M_PATH_ATTR, small_m_print_once,
                                  register_low_m_linear_op)
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import (Fp4QuantizedTensor, MxFp8QuantizedTensor,
                     is_torch_compiling)
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, filter_weights, register_auto_model

# Use TinyGEMM when the number of tokens is not larger than this threshold
MIN_LATENCY_TINYGEMM_NUM_TOKENS = 128

# Default-off opt-ins for the small-M BF16 GEMMs on SM107 (Rubin), read once at
# model construction. Each is ignored on every other GPU.
# - TLLM_TINYGEMM2_ROUTER_SM107=1: the router gate takes tinygemm2 for up to
#   MIN_LATENCY_TINYGEMM_NUM_TOKENS tokens, as on SM90/100/103 (that list,
#   #8182, predates SM107; the op's own check admits SM107 since #19224).
# - TLLM_LOWM_GEMM_SM107=1: an attention projection whose K is a multiple of
#   128 (o_proj, K=4096; not qkv_proj, K=2880) takes the autotuned low-M
#   dispatcher for up to 32 tokens, which use_custom_cublas_mm otherwise skips.
# - TLLM_TINYGEMM2_PROJ_SM107=qkv_proj[,o_proj]: the named projections take
#   tinygemm2 for up to TLLM_TINYGEMM2_PROJ_MAX_TOKENS (default 8) tokens; this
#   wins over TLLM_LOWM_GEMM_SM107 for a projection named in both.
#   Experimental: this path has not been validated for accuracy; run it only
#   together with an accuracy check, and leave it unset in production
#   configurations.
_SMALL_M_SM_VERSION = 107
_SMALL_M_PROJECTIONS = ("qkv_proj", "o_proj")

# Default-off glue fusions for the layouts with no all-reduce to fuse into (TP1,
# no attention DP), read once here; unset, nothing below changes and their
# module (cute_dsl_kernels/gptoss_glue_fusion.py) is never imported. Each is
# bitwise equal to the kernels it replaces.
# - TLLM_GPTOSS_FUSE_FINALIZE_NORM=1: the MoE skips its finalize kernel, and one
#   kernel gathers the top-k expert rows, sums them, adds the residual and runs
#   the next layer's RMSNorm (the final norm after the last layer).
# - TLLM_GPTOSS_FUSE_NORM_MXFP8=1: the post-attention add+RMSNorm also writes
#   the MoE input's MXFP8 quantization, which the MoE then takes as is.
_FUSE_FINALIZE_NORM = os.environ.get("TLLM_GPTOSS_FUSE_FINALIZE_NORM",
                                     "0") == "1"
_FUSE_NORM_MXFP8 = os.environ.get("TLLM_GPTOSS_FUSE_NORM_MXFP8", "0") == "1"
# - TLLM_GPTOSS_DEP3_GLUE=mxfp8 (or 1/all): the same fused norm+MXFP8 under
#   attention DP, ahead of the one-sided all-to-all's post-quant dispatch
#   (moe/fused_moe/static_eplb_route.py documents the flag's parts).
# The first gate miss or failure per fusion (all blocks share one
# configuration), kept so the other blocks neither retry a failed compile nor
# log again.
_GLUE_GATE_FAILURE: Dict[str, str] = {}


def _glue_gate(name: str, env: str, block: "TransformerBlock"):
    """The gate's result when the fusion `name` serves this block, else None."""
    if name in _GLUE_GATE_FAILURE:
        return None
    try:
        from ..cute_dsl_kernels import gptoss_glue_fusion as glue
        gate = {
            "finalize_norm": glue.gate_finalize_norm,
            "norm_mxfp8": glue.gate_norm_mxfp8,
            "norm_mxfp8_dep3": glue.gate_norm_mxfp8_dep3,
        }[name]
        return gate(block, get_sm_version())
    except Exception as e:  # any miss or failure keeps the unfused kernels
        _GLUE_GATE_FAILURE[name] = repr(e)
        print(
            f"[TRT-LLM] [W] {env}=1 disabled for this process, unfused "
            f"kernels kept: {e!r}",
            file=sys.stderr,
            flush=True)
        return None


def _mark_small_m_projections(attn: Attention, layer_idx: int) -> None:
    """Mark qkv_proj/o_proj for a small-M path (SM107 opt-ins)."""
    if get_sm_version() != _SMALL_M_SM_VERSION:
        return
    low_m = os.environ.get("TLLM_LOWM_GEMM_SM107", "0") == "1"
    tiny = {
        name.strip()
        for name in os.environ.get("TLLM_TINYGEMM2_PROJ_SM107", "").split(",")
        if name.strip()
    }
    if not low_m and not tiny:
        return
    tiny_max_m = int(os.environ.get("TLLM_TINYGEMM2_PROJ_MAX_TOKENS", "8"))
    paths = []
    for name in _SMALL_M_PROJECTIONS:
        linear = getattr(attn, name)
        shape = f"k={linear.in_features},n={linear.out_features}"
        if name in tiny and tiny_max_m > 0:
            setattr(linear, SMALL_M_PATH_ATTR, ("tinygemm2", tiny_max_m))
            paths.append(f"{name}=tinygemm2(m<={tiny_max_m},{shape})")
        elif (low_m and linear.use_custom_cublas_mm
              and linear.in_features % 128 == 0):
            register_low_m_linear_op()
            setattr(linear, SMALL_M_PATH_ATTR, ("low_m", LOW_M_MAX_M))
            paths.append(f"{name}=low_m(m<={LOW_M_MAX_M},{shape})")
        else:
            paths.append(f"{name}=unchanged({shape})")
    ignored = sorted(tiny.difference(_SMALL_M_PROJECTIONS))
    small_m_print_once(
        "projections", f"sm={_SMALL_M_SM_VERSION} layer={layer_idx} " +
        " ".join(paths) + (f" ignored={','.join(ignored)}" if ignored else ""))


class AttentionBlock(Attention):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int = 0,
        reduce_output: bool = True,
        use_custom_cublas_mm: bool = False,
    ):
        pretrained_config = config.pretrained_config

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams(
                dim=pretrained_config.head_dim,
                theta=get_hf_rope_theta(pretrained_config, 10000.0),
                scale_type=RotaryScalingType.yarn,
                scale=pretrained_config.rope_scaling['factor'],
                max_positions=pretrained_config.max_position_embeddings,
                original_max_positions=pretrained_config.
                rope_scaling['original_max_position_embeddings'],
                beta_fast=pretrained_config.rope_scaling['beta_fast'],
                beta_slow=pretrained_config.rope_scaling['beta_slow'],
                duplicate_data=False),
            # GPT-OSS applies NeoX-style (rotate-half) RoPE, matching the HF
            # reference. The fused kernel ignores this flag for yarn (which
            # masked the wrong value); the unfused path honors it.
            is_neox=True,
        )

        super().__init__(
            hidden_size=pretrained_config.hidden_size,
            num_attention_heads=pretrained_config.num_attention_heads,
            num_key_value_heads=pretrained_config.num_key_value_heads,
            max_position_embeddings=pretrained_config.max_position_embeddings,
            bias=True,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=pretrained_config.torch_dtype,
            dense_bias=True,
            config=config,
            q_scaling=1.0,
            attention_chunk_size=None,
            reduce_output=reduce_output,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )

        if self.attn_backend != "TRTLLM":
            raise ValueError(
                f"GPT-OSS model uses attention sinks, which are only supported "
                f"with attn_backend='TRTLLM'. Current backend: {self.attn_backend}."
            )

        # Only apply sliding window to every other layer
        self.sliding_window = pretrained_config.sliding_window if layer_idx % 2 == 0 else None

        self.sinks = Parameter(
            torch.empty(pretrained_config.num_attention_heads // self.tp_size,
                        dtype=torch.float32))
        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)
        _mark_small_m_projections(self, layer_idx)

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor | Fp4QuantizedTensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attention_window_size = self.sliding_window
        attn_output = super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            all_reduce_params=all_reduce_params,
            lora_params=lora_params,
            attention_window_size=attention_window_size,
            attention_sinks=self.sinks.data,
            **kwargs)
        return attn_output, residual

    def load_weights(self, weights: Dict):
        sinks = weights[0]['sinks'][self.num_heads *
                                    self.tp_rank:self.num_heads *
                                    (self.tp_rank + 1)]
        self.sinks.data = sinks.to(torch.float32).to("cuda")


class MLPBlock(torch.nn.Module):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int,
        reduce_results: bool = True,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__()

        self.config = config  # Store config as instance variable
        pretrained_config = config.pretrained_config
        self.num_experts = pretrained_config.num_local_experts
        moe_load_balancer_config = config.moe_load_balancer
        self.num_slots = moe_load_balancer_config.num_slots if moe_load_balancer_config and moe_load_balancer_config.num_slots else self.num_experts

        self.layer_idx = layer_idx
        self.enable_attention_dp = config.mapping.enable_attention_dp
        self.mapping = config.mapping

        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

        self.gate = Linear(
            in_features=pretrained_config.hidden_size,
            out_features=pretrained_config.num_local_experts,
            bias=True,
            dtype=pretrained_config.torch_dtype,
            use_custom_cublas_mm=use_custom_cublas_mm,
        )
        # TLLM_TINYGEMM2_ROUTER_SM107=1 (see the note at the top).
        self._tinygemm2_router_sm107 = (
            get_sm_version() == _SMALL_M_SM_VERSION
            and os.environ.get("TLLM_TINYGEMM2_ROUTER_SM107", "0") == "1")
        if self._tinygemm2_router_sm107:
            small_m_print_once(
                "router",
                f"sm={_SMALL_M_SM_VERSION} layer={layer_idx} router=tinygemm2"
                f"(m<={MIN_LATENCY_TINYGEMM_NUM_TOKENS},"
                f"k={pretrained_config.hidden_size},"
                f"n={pretrained_config.num_local_experts})")

        self.routing_method = RenormalizeMoeRoutingMethod(
            top_k=pretrained_config.num_experts_per_tok,
            output_dtype=torch.bfloat16
            if config.moe_backend.upper() == "TRTLLM" else torch.float32)

        # gpt-oss constants are uniform across experts; the backend broadcasts
        # them to whatever per-expert shape its kernels index, sized by the
        # local slot count it resolved.
        self.moe_activation = SwigluBiasActivation(gate_sigmoid_scale=1.702,
                                                   linear_offset=1.0,
                                                   clamp=7.0)
        # Prepare MoE creation parameters
        moe_params = {
            'routing_method': self.routing_method,
            'num_experts': pretrained_config.num_local_experts,
            'hidden_size': pretrained_config.hidden_size,
            'intermediate_size': pretrained_config.intermediate_size,
            'dtype': pretrained_config.torch_dtype,
            'reduce_results': not self.enable_attention_dp and reduce_results,
            'model_config': config,
            'weight_loading_mode': MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            'bias': True,
            'activation': self.moe_activation,
            'layer_idx': self.layer_idx,
        }

        self.experts = create_moe(**moe_params)

    @staticmethod
    def swiglu(x, alpha: float = 1.702):
        """
        This function is not really used in self.forward(), it's kept here for reference.
        It's implemented as part of the MoE kernels.
        """
        # Note we add an extra bias of 1 to the linear layer
        x_glu, x_linear = torch.chunk(x, 2, dim=-1)
        out_glu = x_glu * torch.sigmoid(alpha * x_glu)
        return out_glu * (x_linear + 1)

    def compute_gate_output(self,
                            x: torch.Tensor,
                            lora_params: Optional[dict] = None) -> torch.Tensor:
        # Skip tinygemm2 optimization when LoRA is active (tinygemm2 doesn't support LoRA)
        use_tinygemm = ((get_sm_version() in [90, 100, 103]
                         or self._tinygemm2_router_sm107)
                        and x.shape[0] <= MIN_LATENCY_TINYGEMM_NUM_TOKENS
                        and (lora_params is None or not bool(lora_params)))

        if use_tinygemm:
            weight = self.gate.weight
            bias = self.gate.bias
            g = torch.ops.trtllm.tinygemm2(x, weight, bias)
        else:
            g = self.gate(x, lora_params=lora_params, layer_idx=self.layer_idx)
        return g

    def forward_normal(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None,
        x_quant: Optional[MxFp8QuantizedTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x = x.view(-1, hidden_dim)

        # t = self.norm(x) was done in the parent block
        t = x

        g = self.compute_gate_output(t, lora_params=lora_params)

        if x_quant is not None:
            # TLLM_GPTOSS_FUSE_NORM_MXFP8: the experts take the MXFP8 copy of t
            # the fused post-attention norm wrote; the router keeps the BF16 t.
            expert_output = self.experts(x=x_quant,
                                         router_logits=g,
                                         output_dtype=t.dtype)
        else:
            # When attention_dp is not enabled, don't pass those parameters
            expert_output = self.experts(x=t, router_logits=g)

        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def forward_finalize_add_rmsnorm(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm: RMSNorm,
        x_quant: Optional[MxFp8QuantizedTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """norm(MoE(x), residual) with the MoE's finalize fused into the norm
        (TLLM_GPTOSS_FUSE_FINALIZE_NORM): (normed output, new residual)."""
        from ..cute_dsl_kernels import gptoss_glue_fusion as glue

        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        t = x.view(-1, hidden_dim)
        g = self.compute_gate_output(t)
        experts = self.experts
        g = experts._maybe_get_perfect_router_logits(g)
        if x_quant is not None:
            x_in, x_sf = x_quant.fp8_tensor, x_quant.scaling_factor
        else:
            x_in, x_sf = t, None
        res = residual.view(-1, hidden_dim)
        if experts.register_to_config and is_torch_compiling():
            out, res = torch.ops.trtllm.gptoss_moe_finalize_add_rmsnorm(
                experts.layer_idx_str, x_in, x_sf, g, res, norm.weight,
                norm.variance_epsilon)
        else:
            out, res = glue.moe_layer_finalize_add_rmsnorm(
                experts, x_in, x_sf, g, res, norm.weight, norm.variance_epsilon)
        return out.view(orig_shape), res.view(orig_shape)

    def forward_attn_dp(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None,
        x_quant: Optional[MxFp8QuantizedTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x = x.view(-1, hidden_dim)

        # t = self.norm(x) was done in the parent block
        t = x

        # Get attention_dp parameters
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.mapping.tp_size > 1 and all_rank_num_tokens is not None:
            if (isinstance(self.experts, (TritonFusedMoE))):
                t = allgather(t, self.mapping, dim=0, sizes=all_rank_num_tokens)

        g = self.compute_gate_output(t, lora_params=lora_params)

        if x_quant is not None and self._dep3_mxfp8_usable(
                all_rank_num_tokens):
            # TLLM_GPTOSS_DEP3_GLUE=mxfp8: the experts take the MXFP8 copy of
            # t the fused post-attention norm wrote; quantize_input passes it
            # to the post-quant dispatch as is. The router keeps the BF16 t.
            expert_output = self.experts(
                x=x_quant,
                router_logits=g,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False,
                output_dtype=t.dtype)
        else:
            # Let CutlassFusedMoE and TRTLLMGenFusedMoE handle allgather internally
            # Pass the normalized tensor (t) as input to experts, not x
            expert_output = self.experts(
                x=t,
                router_logits=g,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False)

        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def _dep3_mxfp8_usable(self, all_rank_num_tokens) -> bool:
        """Whether this forward's MoE takes an MXFP8 input as is: one chunk
        (the chunked path splits and re-pads its input) and no empty rank
        (the scheduler pads the input with zero rows). Otherwise the BF16
        output of the same fused kernel goes in and is quantized as today."""
        if all_rank_num_tokens is None or min(all_rank_num_tokens) <= 0:
            return False
        return self.experts.calculate_num_chunks(all_rank_num_tokens) == 1

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None,
        x_quant: Optional[MxFp8QuantizedTensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.enable_attention_dp:
            # x_quant is set only by TLLM_GPTOSS_DEP3_GLUE=mxfp8's gate.
            return self.forward_attn_dp(x,
                                        attn_metadata,
                                        residual,
                                        lora_params,
                                        x_quant=x_quant)
        else:
            return self.forward_normal(x,
                                       residual,
                                       lora_params,
                                       x_quant=x_quant)


class TransformerBlock(DecoderLayer):

    def __init__(
        self,
        config: ModelConfig[GptOssConfig],
        layer_idx: int,
        use_custom_cublas_mm: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        mapping = config.mapping
        self.enable_attn_dp = mapping.enable_attention_dp
        self.is_tp = mapping.has_tp() and not self.enable_attn_dp

        pretrained_config = config.pretrained_config
        self.input_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        self.attn = AttentionBlock(config,
                                   layer_idx,
                                   reduce_output=False,
                                   use_custom_cublas_mm=use_custom_cublas_mm)

        self.post_attention_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        self.mlp = MLPBlock(config,
                            layer_idx,
                            reduce_results=False,
                            use_custom_cublas_mm=use_custom_cublas_mm)

        self.mapping = config.mapping

        self.next_layer_layernorm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype)

        # setup for tp
        self.allreduce = None
        if self.is_tp:
            self.allreduce = AllReduce(
                mapping=config.mapping,
                strategy=config.allreduce_strategy,
                dtype=config.pretrained_config.torch_dtype)

        # Default-off glue fusions (see _FUSE_FINALIZE_NORM): False / None keep
        # today's kernels.
        self.fuse_finalize_norm = bool(_FUSE_FINALIZE_NORM and _glue_gate(
            "finalize_norm", "TLLM_GPTOSS_FUSE_FINALIZE_NORM", self))
        # The MoE input alignment the fused post-attention kernel was built for.
        self.norm_mxfp8_alignment = (_glue_gate(
            "norm_mxfp8", "TLLM_GPTOSS_FUSE_NORM_MXFP8", self)
                                     if _FUSE_NORM_MXFP8 else None)
        if (self.norm_mxfp8_alignment is None and DEP3_NORM_MXFP8_ENABLED
                and self.enable_attn_dp):
            # TLLM_GPTOSS_DEP3_GLUE=mxfp8: the same kernel under attention DP.
            self.norm_mxfp8_alignment = _glue_gate("norm_mxfp8_dep3",
                                                   "TLLM_GPTOSS_DEP3_GLUE",
                                                   self)

    def _fused_post_attention_norm_mxfp8(self, x: torch.Tensor,
                                         residual: torch.Tensor):
        """post_attention_layernorm(x, residual) plus the MXFP8 quantization
        of its output the MoE would do, from one kernel; None when the MoE
        resolved another input alignment."""
        quant_method = getattr(self.mlp.experts.backend, "quant_method", None)
        alignment = getattr(quant_method, "input_hidden_alignment", None)
        if alignment != self.norm_mxfp8_alignment:
            return None
        norm = self.post_attention_layernorm
        shape = x.shape
        hidden = shape[-1]
        out, residual, x_fp8, x_sf = torch.ops.trtllm.fused_add_rmsnorm_mxfp8(
            x.reshape(-1, hidden), residual.reshape(-1, hidden), norm.weight,
            norm.variance_epsilon, self.norm_mxfp8_alignment)
        return (out.view(shape), residual.view(shape),
                MxFp8QuantizedTensor(x_fp8, x_sf, is_sf_swizzled=False))

    def forward_normal(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        x, residual = self.attn(position_ids,
                                hidden_states,
                                attn_metadata,
                                residual=residual,
                                lora_params=lora_params,
                                **kwargs)
        if kwargs.get("output_row_ids") is not None:
            # TLLM_PRUNE_LAST_TOKEN: the attention output already holds only
            # these rows; keep the residual rows that match.
            residual = residual[kwargs["output_row_ids"]]
        fused = None
        if self.norm_mxfp8_alignment is not None and not lora_params:
            fused = self._fused_post_attention_norm_mxfp8(x, residual)
        if fused is None:
            x, residual = self.post_attention_layernorm(x, residual)
            x_quant = None
        else:
            x, residual, x_quant = fused

        if self.fuse_finalize_norm and not lora_params:
            # The MoE output never goes to memory: the fused kernel writes the
            # next layer's normed input and residual (= MoE output + residual,
            # which is what EAGLE3 captures).
            x, residual = self.mlp.forward_finalize_add_rmsnorm(
                x, residual, self.next_layer_layernorm, x_quant=x_quant)
            if spec_metadata is not None:
                spec_metadata.maybe_capture_hidden_states(
                    self.layer_idx, residual, None)
            return x, residual

        x, residual = self.mlp(x,
                               attn_metadata,
                               residual,
                               lora_params=lora_params,
                               x_quant=x_quant)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, x,
                                                      residual)

        x, residual = self.next_layer_layernorm(x, residual)
        return x, residual

    def forward_tp(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        x, residual = self.attn(
            position_ids,
            hidden_states,
            attn_metadata,
            residual=residual,
            all_reduce_params=AllReduceParams(enable_allreduce=False),
            lora_params=lora_params,
            **kwargs)

        x, residual = self.allreduce(
            x,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
                trigger_completion_at_end=False,
            ))

        x, residual = self.mlp(x,
                               attn_metadata,
                               residual,
                               lora_params=lora_params)

        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            # In eagle3 mode, we capture the value in the boundary of decoder layer.
            # If fusing rms in the next layer, the value is not correct. Thus, if
            # this layer will be captured, we should not fuse the rms in the next
            # layer.
            x = self.allreduce(x,
                               all_reduce_params=AllReduceParams(
                                   fusion_op=AllReduceFusionOp.NONE,
                                   trigger_completion_at_end=False,
                               ))
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, x,
                                                      residual)
            x, residual = self.next_layer_layernorm(x, residual)
        else:
            x, residual = self.allreduce(
                x,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ))

        return x, residual

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = ...,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.is_tp:
            _forward = self.forward_tp
        else:
            _forward = self.forward_normal
        return _forward(position_ids,
                        hidden_states,
                        attn_metadata,
                        residual,
                        spec_metadata=spec_metadata,
                        lora_params=lora_params,
                        **kwargs)


class Transformer(DecoderModel):

    def __init__(self, model_config: ModelConfig[GptOssConfig]):
        super().__init__(model_config)
        config = self.model_config

        # Triton MoE kernels require installing Triton main branch,
        # which may be incompatible with torch.compile due to version mismatch.
        enable_torch_compile_for_embedding = model_config.moe_backend != "TRITON"

        # Use custom cublas since we need LUT to tune the perf.
        prop = torch.cuda.get_device_properties(0)
        sm_version = prop.major * 10 + prop.minor
        # Use custom cublas to bypass F.linear's additional memory copy for biases on SM 100+
        self.use_custom_cublas_mm = sm_version >= 100

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embedding = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                enable_torch_compile_for_embedding=
                enable_torch_compile_for_embedding)
        else:
            self.embedding = Embedding(
                config.pretrained_config.vocab_size,
                config.pretrained_config.hidden_size,
                dtype=config.pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                enable_torch_compile_for_embedding=
                enable_torch_compile_for_embedding,
                use_custom_cublas_mm=self.use_custom_cublas_mm,
            )
        # For modeling_speculative, different name expected
        self.embed_tokens = self.embedding
        self.block = nn.ModuleList([
            TransformerBlock(
                model_config,
                layer_idx,
                use_custom_cublas_mm=self.use_custom_cublas_mm,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            hidden_size=config.pretrained_config.hidden_size,
            eps=config.pretrained_config.rms_norm_eps,
            dtype=config.pretrained_config.torch_dtype,
        )
        # TLLM_PRUNE_LAST_TOKEN: the last layer may run on a row subset only
        # when no collective sees its row count (single GPU, no attention DP
        # or Helix CP) and its attention has no sparse hooks.
        mapping = model_config.mapping
        last = self.block[-1]
        self._prune_last_token = (PRUNE_LAST_TOKEN_TARGET
                                  and mapping.tp_size == 1
                                  and not mapping.enable_attention_dp
                                  and not mapping.has_cp_helix()
                                  and not last.is_tp
                                  and last.attn.sparse_attn_hooks is None)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        hidden_states = inputs_embeds or self.embedding(input_ids)

        # TLLM_PRUNE_LAST_TOKEN: SpecDecOneEngineForCausalLM.forward offers the
        # rows the step reads (spec_metadata.prune_row_ids, else None);
        # the last layer then runs o_proj, the MoE and the final norm on those
        # rows only. Dynamo traces this: an offer is a separate compiled graph
        # (only offered for steps above the piecewise CUDA graph sizes), no
        # offer adds no op, and with the setting unset nothing is read.
        last_row_ids = None
        if (self._prune_last_token and spec_metadata is not None
                and not lora_params):
            last_row_ids = getattr(spec_metadata, "prune_row_ids", None)
            if last_row_ids is not None and spec_metadata.is_layer_capture(
                    len(self.block) - 1):
                last_row_ids = None

        residual = None
        last_idx = len(self.block) - 1
        for idx, block in enumerate(self.block):
            extra = ({} if last_row_ids is None or idx != last_idx else {
                "output_row_ids": last_row_ids
            })
            hidden_states, residual = block(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
                **extra,
            )

        return hidden_states


@register_auto_model("GptOssForCausalLM")
class GptOssForCausalLM(SpecDecOneEngineForCausalLM[Transformer, GptOssConfig]):

    @classmethod
    def get_preferred_kv_cache_manager_version(cls,
                                               pretrained_config: Any = None
                                               ) -> Literal["V2"]:
        """Prefer KV cache manager V2 for the model's VSWA layout.

        GPT-OSS applies a sliding window to every other layer
        (see ``AttentionBlock.__init__``), so the KV cache is VSWA: two
        distinct attention window sizes. V2 groups layers by lifecycle and
        coalesces buffers within each pool group, which sizes the
        sliding-window and full-attention pools independently instead of
        statically dividing memory between them.

        The preference is adopted only when the user leaves
        ``kv_cache_config.use_kv_cache_manager_v2`` at ``"auto"``. Two-model
        speculative decoding demotes it to V1 because V2 sizes both the target
        and draft KV cache managers from the full budget; an explicit ``True``
        is rejected by ``llm_utils._resolve_kv_cache_manager_v2_auto``.
        """
        return "V2"

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        return "PYTHON"

    params_map = {
        # TRTLLM module name : GptOss module name
        "qkv_proj": "qkv",
        "o_proj": "out",
        "lm_head": "unembedding",
    }

    hf_params_map = {
        # TRTLLM module name : HuggingFace module name
        "embedding": "embed_tokens",
        # Order matters for attn.norm and attn.
        'attn.norm': 'input_layernorm',
        'attn': 'self_attn',
        'mlp.norm': 'post_attention_layernorm',
        'block': 'layers',
        'gate': 'router',
    }

    def __init__(
        self,
        model_config: ModelConfig[GptOssConfig],
    ):
        # Map config to HF format.
        if hasattr(model_config.pretrained_config, 'num_experts'):
            model_config.pretrained_config.num_local_experts = model_config.pretrained_config.num_experts
            model_config.pretrained_config.num_experts_per_tok = model_config.pretrained_config.experts_per_token
            model_config.pretrained_config.rope_scaling = {
                'factor':
                model_config.pretrained_config.rope_scaling_factor,
                'beta_fast':
                model_config.pretrained_config.rope_ntk_beta,
                'beta_slow':
                model_config.pretrained_config.rope_ntk_alpha,
                'original_max_position_embeddings':
                model_config.pretrained_config.initial_context_length,
            }
        if model_config.pretrained_config.torch_dtype is None:
            model_config.pretrained_config.torch_dtype = torch.bfloat16

        assert model_config.mapping.pp_size == 1, "Pipeline parallelism is not supported."

        super().__init__(
            Transformer(model_config),
            model_config=model_config,
        )

    def __post_init__(self):
        # Do not call super().__post_init__()
        params_map_reverse = {v: k for k, v in self.params_map.items()}

        quant_config = self.model_config.quant_config
        if quant_config.exclude_modules:
            if quant_config.quant_algo == "NVFP4":
                quant_config.exclude_modules = [
                    'block.*.attn.qkv',
                    'block.*.attn.out',
                    'block.*.mlp.gate',
                    'embedding',
                    'unembedding',
                ]

            for i, module in enumerate(quant_config.exclude_modules):
                names = module.split(".")
                if names[-1] in params_map_reverse:
                    names[-1] = params_map_reverse[names[-1]]
                prefix = [] if names[0] == "model" else ["model"]
                quant_config.exclude_modules[i] = '.'.join(prefix + names)

        super().apply_quant_config_exclude_modules()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def load_weights(self, weights: Dict):
        is_nvfp4 = self.model_config.quant_config.quant_mode.has_nvfp4()

        if is_nvfp4:
            self.load_nvfp4_weights(weights)
        else:
            self.load_hf_weights(weights)

    def setup_aliases(self) -> None:
        for idx, layer in enumerate(
                self.model.block[:self.config.num_hidden_layers]):
            if idx == 0:
                layer.input_layernorm = layer.attn.norm

            layer.post_attention_layernorm = layer.mlp.norm

            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.block[idx + 1].attn.norm

    def load_hf_weights(self, weights: Dict):
        num_expert = self.config.num_local_experts

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue

            module_weights = {}
            for k, v in self.hf_params_map.items():
                name = name.replace(k, v)

            # Special case: ConfigurableMoE.backend (TRTLLMGenFusedMoE)
            # Currently saved MoE weights don't include 'backend' in their names.
            # After MoE refactoring, ConfigurableMoE now has a backend submodule,
            # and weights loading is done in the backend, so module name includes '.backend'.
            # We need to use parent module name (without .backend) to match saved weight names.
            # After MoE refactoring is fully complete, all paths will follow this branch.
            names = name.split('.')
            if names[-1] == "backend" and is_moe_weight_owner(module):
                # Backend is under experts module (ConfigurableMoE wrapper)
                name = '.'.join(names[:-1])

            module_weights = filter_weights(name, weights)

            if is_moe_weight_owner(module):
                try:
                    # For BF16 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights['gate_up_proj']
                    gate, up = gate_up_weight[:, :, ::2], gate_up_weight[:, :,
                                                                         1::2]
                    gate_up_weight = torch.cat([gate, up], dim=-1)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate, up = gate_up_bias[:, ::2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate, up], dim=-1)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight.to(self.model.dtype)[i, :, :]
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj'][i, :, :].to(
                                self.model.dtype) for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ]
                    }
                except:
                    # For MXFP4 ckpt.
                    # Deinterleave for gate and up.
                    gate_up_weight = module_weights[
                        'gate_up_proj_blocks'].flatten(-2, -1)
                    gate_weight, up_weight = gate_up_weight[:, ::
                                                            2, :], gate_up_weight[:,
                                                                                  1::
                                                                                  2, :]
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=-2)
                    gate_up_bias = module_weights['gate_up_proj_bias']
                    gate_bias, up_bias = gate_up_bias[:, ::
                                                      2], gate_up_bias[:, 1::2]
                    gate_up_bias = torch.cat([gate_bias, up_bias], dim=-1)
                    gate_up_weight_scale = module_weights['gate_up_proj_scales']
                    gate_weight_scale, up_weight_scale = gate_up_weight_scale[:, ::
                                                                              2, :], gate_up_weight_scale[:,
                                                                                                          1::
                                                                                                          2, :]
                    gate_up_weight_scale = torch.cat(
                        [gate_weight_scale, up_weight_scale], dim=-2)
                    moe_weights = {
                        'gate_up_proj': [
                            gate_up_weight[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj': [
                            module_weights['down_proj_blocks'].flatten(
                                -2, -1)[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'gate_up_proj.bias':
                        [gate_up_bias[i, :] for i in range(num_expert)],
                        'down_proj.bias': [
                            module_weights['down_proj_bias'][i, :]
                            for i in range(num_expert)
                        ],
                        'gate_up_proj_weight_scale': [
                            gate_up_weight_scale[i, :, :].transpose(0, 1)
                            for i in range(num_expert)
                        ],
                        'down_proj_weight_scale': [
                            module_weights['down_proj_scales']
                            [i, :, :].transpose(0, 1) for i in range(num_expert)
                        ]
                    }

                    if self.model_config.quant_config.quant_algo == 'W4A16_MXFP4':
                        for i in range(num_expert):
                            moe_weights[
                                f"{i}.w1.weight_scale_inv"] = gate_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w3.weight_scale_inv"] = up_weight_scale[
                                    i, :, :]
                            moe_weights[
                                f"{i}.w2.weight_scale_inv"] = module_weights[
                                    'down_proj_scales'][i, :, :]

                module.load_weights(weights=[moe_weights])
            elif hasattr(module, "load_weights"):
                if 'qkv' in name:
                    # For qkv_proj
                    q_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'q_proj'), weights)
                    k_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'k_proj'), weights)
                    v_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'v_proj'), weights)
                    module.load_weights(
                        weights=[q_weight_bias, k_weight_bias, v_weight_bias])
                else:
                    # For o_proj, sinks.
                    module.load_weights(weights=[module_weights])
            else:
                # Load four LN weights (attn.norm, mlp.norm, input_layernorm, post_attention_layernorm).
                if 'next_layer_layernorm' in name:
                    continue

                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n][:])

    def load_nvfp4_weights(self, weights: Dict):
        num_expert = self.config.num_local_experts

        for name, module in tqdm(list(self.named_modules()),
                                 desc="Loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue

            module_weights = {}
            for k, v in self.hf_params_map.items():
                name = name.replace(k, v)

            names = name.split('.')
            if names[-1] == "backend" and is_moe_weight_owner(module):
                # Backend is under experts module (ConfigurableMoE wrapper)
                name = '.'.join(names[:-1])

            module_weights = filter_weights(name, weights)

            if is_moe_weight_owner(module):
                assert getattr(module, "quant_config", None) is not None and \
                   module.quant_config.quant_mode.has_nvfp4()
                gate_up = module_weights.get('gate_up_proj', None)
                down = module_weights.get('down_proj', None)
                gate_up_bias = module_weights.get('gate_up_proj_bias', None)
                down_bias = module_weights.get('down_proj_bias', None)

                def deinterleave(tensor):
                    g, u = tensor[..., ::2], tensor[..., 1::2]
                    return torch.cat([g, u], dim=-1)

                gate_up = deinterleave(gate_up)
                gate_up_bias = deinterleave(gate_up_bias)

                # Only fp32 bias is supported for NVFP4 MoE.
                if gate_up_bias.dtype != torch.float32:
                    gate_up_bias = gate_up_bias.to(torch.float32)
                if down_bias.dtype != torch.float32:
                    down_bias = down_bias.to(torch.float32)

                moe_weights = {}
                if gate_up is not None:
                    moe_weights['gate_up_proj'] = [
                        gate_up[i, :, :] for i in range(num_expert)
                    ]
                if down is not None:
                    moe_weights['down_proj'] = [
                        down[i, :, :] for i in range(num_expert)
                    ]
                if gate_up_bias is not None:
                    moe_weights['gate_up_proj.bias'] = [
                        gate_up_bias[i, :] for i in range(num_expert)
                    ]
                if down_bias is not None:
                    moe_weights['down_proj.bias'] = [
                        down_bias[i, :] for i in range(num_expert)
                    ]

                # Per-expert block scales (transpose to expected layout)
                if 'gate_up_proj_weight_scale' in module_weights:
                    gu_ws = module_weights['gate_up_proj_weight_scale']
                    gu_ws = deinterleave(gu_ws)
                    moe_weights['gate_up_proj_weight_scale'] = [
                        gu_ws[i, :, :] for i in range(num_expert)
                    ]
                if 'down_proj_weight_scale' in module_weights:
                    dp_ws = module_weights['down_proj_weight_scale']
                    moe_weights['down_proj_weight_scale'] = [
                        dp_ws[i, :, :] for i in range(num_expert)
                    ]

                # Module-level globals for NVFP4 loaders
                for src_key in [
                        'gate_up_proj_weight_scale_2',
                        'down_proj_weight_scale_2',
                        'gate_up_proj_input_scale',
                        'down_proj_input_scale',
                ]:
                    if src_key in module_weights:
                        moe_weights[src_key] = module_weights[src_key]

                module.load_weights(weights=[moe_weights])
            elif hasattr(module, "load_weights"):
                if 'qkv' in name:
                    # For qkv_proj
                    q_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'q_proj'), weights)
                    k_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'k_proj'), weights)
                    v_weight_bias = filter_weights(
                        name.replace('qkv_proj', 'v_proj'), weights)
                    module.load_weights(
                        weights=[q_weight_bias, k_weight_bias, v_weight_bias])
                else:
                    # For o_proj, sinks.
                    module.load_weights(weights=[module_weights])
            else:
                # Load four LN weights (attn.norm, mlp.norm, input_layernorm, post_attention_layernorm).
                if 'next_layer_layernorm' in name:
                    continue

                for n, p in module._parameters.items():
                    if p is not None:
                        p.data.copy_(module_weights[n][:])
