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
from tensorrt_llm.logger import logger

from ..attention.attention import Attention
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import (PositionalEmbeddingParams,
                                            PredefinedAttentionMask, RopeParams)
from ..distributed import (AllReduce, AllReduceFusionOp, AllReduceParams,
                           MoEAllReduce, allgather)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from ..moe.fused_moe import (MoEWeightLoadingMode, RenormalizeMoeRoutingMethod,
                             SwigluBiasActivation, TritonFusedMoE, create_moe,
                             is_moe_weight_owner)
from ..moe.fused_moe.quantization import MXFP4AsNVFP4TRTLLMGenFusedMoEMethod
# isort: on
from ..modules.fused_ops.rmsnorm_fp4_quant import (
    add_rmsnorm_fp4_quant, add_rmsnorm_fp4_quant_available)
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import Fp4QuantizedTensor
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, filter_weights, register_auto_model

# Use TinyGEMM when the number of tokens is not larger than this threshold
MIN_LATENCY_TINYGEMM_NUM_TOKENS = 128

# A/B escape hatch: restore separate RMSNorm and NVFP4 input quantization.
# Read once at import so the environment is never queried in the hot path.
_GPT_OSS_DISABLE_FUSED_MLP_NVFP4_INPUT = os.environ.get(
    "TLLM_GPT_OSS_DISABLE_FUSED_MLP_NVFP4_INPUT", "0") == "1"

if _GPT_OSS_DISABLE_FUSED_MLP_NVFP4_INPUT:
    logger.info_once(
        "GPT-OSS fused RMSNorm and NVFP4 input quantization disabled by "
        "TLLM_GPT_OSS_DISABLE_FUSED_MLP_NVFP4_INPUT=1",
        key="gpt_oss_fused_mlp_nvfp4_input_disabled",
    )

# Opt-in MXFP8 W8A8 for the checkpoint's bf16 dense layers, read once at
# import (GptOssForCausalLM._maybe_requantize_dense_to_mxfp8). Default off.
_MXFP8_ATTN_PROJ = os.environ.get("TLLM_MXFP8_ATTN_PROJ", "0") == "1"
_MXFP8_LM_HEAD = os.environ.get("TLLM_MXFP8_LM_HEAD", "0")
_printed: set = set()


def _print_once(key: str, message: str) -> None:
    """One line per process and key; printed, since workers log at "error"."""
    if key not in _printed:
        _printed.add(key)
        print(message, flush=True)


# GPT-OSS all-reduce fusions for the attention-TP path, each default off and read once at import:
# - TLLM_GPT_OSS_MOE_FINALIZE_AR=1: the MoE finalize (top-k weighted unpermute) runs inside the post-MoE all-reduce
#   (MoEAllReduce.finalize_allreduce_match_unfused), bit-exact to finalize + AR + residual + RMSNorm.
# - TLLM_GPT_OSS_AR_MXFP8_QUANT=1: the MoE's MXFP8 input quantization runs inside the pre-MoE all-reduce
#   (AllReduce.residual_rms_norm_mxfp8), bit-exact to AR + residual + RMSNorm + mxfp8_quantize.
# Both apply to one-shot sized steps (<= 128 tokens) on the MXFP4 TRTLLM-Gen MoE with MXFP8 activations; every other
# step, layer and configuration keeps the unfused path.
_GPT_OSS_MOE_FINALIZE_AR = os.environ.get("TLLM_GPT_OSS_MOE_FINALIZE_AR", "0") == "1"
_GPT_OSS_AR_MXFP8_QUANT = os.environ.get("TLLM_GPT_OSS_AR_MXFP8_QUANT", "0") == "1"
_GPT_OSS_FUSED_AR_MAX_TOKENS = 128
_gpt_oss_fusion_reported = set()


def _report_gpt_oss_fusion_once(name: str, detail: str) -> None:
    if name in _gpt_oss_fusion_reported:
        return
    _gpt_oss_fusion_reported.add(name)
    # Printed rather than logged: worker processes keep the logger at "error".
    print(f"{name} engaged: {detail} pid={os.getpid()}",
          file=sys.stderr,
          flush=True)


def _moe_input_hidden_alignment(impl) -> int:
    """The column alignment the MoE pads its MXFP8 input to (its quant method's input_hidden_alignment).

    The impl creates ``quant_method`` only in create_weights, after the decoder layers are built (and not at all
    before weight loading when model_config.skip_create_weights_in_init is set), so fall back to the method the
    impl will create; for the MXFP4 TRTLLM-Gen leaf that is a class constant (512).
    """
    method = getattr(impl, "quant_method", None)
    if method is None:
        method = impl._get_quant_method()
    return int(method.input_hidden_alignment)


def _fused_mlp_nvfp4_input_enabled(tp_size: int, enable_attn_dp: bool,
                                   sm_version: int) -> bool:
    # Attention-DP ranks normalize local tokens; only attention TP needs the
    # existing fused all-reduce + RMSNorm path instead.
    return (not _GPT_OSS_DISABLE_FUSED_MLP_NVFP4_INPUT
            and (tp_size == 1 or enable_attn_dp) and 100 <= sm_version < 120
            and add_rmsnorm_fp4_quant_available())


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

        # Every target layer after the first may reuse the decoder info the layer before it built in the same
        # pure-generation step (TLLM_REUSE_DECODER_INFO, see fmha/flashinfer_trtllm_gen.py); the EAGLE3 draft
        # layers are not GPT-OSS AttentionBlocks and always build their own.
        self.attn.reuse_decoder_info_from_previous_layer = layer_idx > 0

        # Only apply sliding window to every other layer
        self.sliding_window = pretrained_config.sliding_window if layer_idx % 2 == 0 else None

        self.sinks = Parameter(
            torch.empty(pretrained_config.num_attention_heads // self.tp_size,
                        dtype=torch.float32))
        self.norm = RMSNorm(hidden_size=pretrained_config.hidden_size,
                            eps=pretrained_config.rms_norm_eps,
                            dtype=pretrained_config.torch_dtype)

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
        use_tinygemm = (get_sm_version() in [90, 100, 103]
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
        x: torch.Tensor | Fp4QuantizedTensor,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None,
        do_finalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, Fp4QuantizedTensor):
            t = x.unquantized_hidden_states
            if t is None:
                raise ValueError(
                    "GPT-OSS MoE requires the BF16 view of a prequantized input"
                )
            orig_shape = t.shape
            hidden_dim = orig_shape[-1]
            t = t.view(-1, hidden_dim)
            expert_input = x
        else:
            orig_shape = x.shape
            hidden_dim = orig_shape[-1]
            t = x.view(-1, hidden_dim)
            expert_input = t

        g = self.compute_gate_output(t, lora_params=lora_params)

        # When attention_dp is not enabled, don't pass those parameters
        if isinstance(expert_input, Fp4QuantizedTensor):
            expert_output = self.experts(x=expert_input,
                                         router_logits=g,
                                         output_dtype=t.dtype,
                                         do_finalize=do_finalize)
        else:
            expert_output = self.experts(x=expert_input,
                                         router_logits=g,
                                         do_finalize=do_finalize)

        if not do_finalize:
            # [fc2_output, expert_weights, expanded_idx_to_permuted_idx] for a fused finalize + all-reduce.
            return expert_output, residual
        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def forward_attn_dp(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The router and output shape use the logical BF16 view, not the
        # packed/padded FP4 shape passed to the experts.
        t = x.unquantized_hidden_states if isinstance(x,
                                                      Fp4QuantizedTensor) else x
        if t is None:
            raise ValueError(
                "GPT-OSS MoE requires the BF16 view of a prequantized input")
        orig_shape = t.shape
        hidden_dim = orig_shape[-1]
        t = t.view(-1, hidden_dim)

        # Get attention_dp parameters
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if self.mapping.tp_size > 1 and all_rank_num_tokens is not None:
            if (isinstance(self.experts, (TritonFusedMoE))):
                t = allgather(t, self.mapping, dim=0, sizes=all_rank_num_tokens)

        g = self.compute_gate_output(t, lora_params=lora_params)

        # Let the MoE handle dispatch internally. Prequantized inputs must
        # retain their scales and explicitly request the BF16 output dtype.
        expert_input = t
        expert_kwargs = {}
        if isinstance(x, Fp4QuantizedTensor):
            expert_input = x
            expert_kwargs["output_dtype"] = t.dtype
        expert_output = self.experts(x=expert_input,
                                     router_logits=g,
                                     all_rank_num_tokens=all_rank_num_tokens,
                                     use_dp_padding=False,
                                     **expert_kwargs)

        expert_output = expert_output.view(orig_shape)
        return expert_output, residual

    def forward(
        self,
        x: torch.Tensor | Fp4QuantizedTensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        lora_params: Optional[dict] = None,
        do_finalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.enable_attention_dp:
            assert do_finalize, "attention DP keeps the finalized MoE output"
            return self.forward_attn_dp(x, attn_metadata, residual, lora_params)
        else:
            return self.forward_normal(x,
                                       residual,
                                       lora_params,
                                       do_finalize=do_finalize)


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
        sm_version = get_sm_version()
        self.enable_fused_mlp_nvfp4_input = \
            _fused_mlp_nvfp4_input_enabled(mapping.tp_size,
                                           self.enable_attn_dp, sm_version)

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
        self.moe_allreduce = None
        if self.is_tp:
            self.allreduce = AllReduce(
                mapping=config.mapping,
                strategy=config.allreduce_strategy,
                dtype=config.pretrained_config.torch_dtype)
            if _GPT_OSS_MOE_FINALIZE_AR:
                self.moe_allreduce = MoEAllReduce(mapping=config.mapping)
        # Whether the all-reduce fusions can apply to this layer at all, decided here (not in the compiled
        # forward). Steps above _GPT_OSS_FUSED_AR_MAX_TOKENS and EAGLE3 capture layers still take the unfused path.
        self._fused_ar_eligible = self._fusions_eligible()
        self._mxfp8_alignment = (_moe_input_hidden_alignment(
            self._mxfp8_moe_impl()) if self._fused_ar_eligible else None)
        if self._fused_ar_eligible:
            if _GPT_OSS_AR_MXFP8_QUANT:
                _report_gpt_oss_fusion_once(
                    "TLLM_GPT_OSS_AR_MXFP8_QUANT",
                    f"pre-MoE all-reduce writes the MXFP8 MoE input (<= {_GPT_OSS_FUSED_AR_MAX_TOKENS} tokens,"
                    f" alignment {self._mxfp8_alignment})")
            if _GPT_OSS_MOE_FINALIZE_AR:
                _report_gpt_oss_fusion_once(
                    "TLLM_GPT_OSS_MOE_FINALIZE_AR",
                    f"MoE finalize runs in the post-MoE all-reduce (<= {_GPT_OSS_FUSED_AR_MAX_TOKENS} tokens,"
                    " not on EAGLE3 capture layers)")
        elif (_GPT_OSS_MOE_FINALIZE_AR or _GPT_OSS_AR_MXFP8_QUANT) and self.is_tp:
            _report_gpt_oss_fusion_once(
                "TLLM_GPT_OSS_FUSED_AR_INELIGIBLE",
                "flags set but this layer keeps the unfused all-reduce (MoE leaf, MoE comm, MNNVL/symm-mem AR or"
                " multi-node TP)")

    def _mxfp8_moe_impl(self):
        """The MoE implementation when it is the MXFP4 TRTLLM-Gen leaf with MXFP8 activations and no MoE comm."""
        from ..moe.fused_moe.trtllm_gen.trtllm_w4a8_mxfp4_mxfp8 import \
            TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl
        experts = self.mlp.experts
        impl = getattr(experts, "backend", experts)
        if not isinstance(impl, TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl):
            return None
        if getattr(experts, "comm", None) is not None or getattr(
                experts, "reduce_results", False):
            return None
        # TLLM_MXFP4_MOE_NVFP4_ACT swaps this leaf to NVFP4 activations; the fused all-reduces write MXFP8 input and
        # assume the MXFP8 runner's un-finalized rows, so neither may run on that worker.
        if getattr(impl, "mxfp4_as_nvfp4", False):
            if "TLLM_GPT_OSS_FUSED_AR_REFUSED" not in _gpt_oss_fusion_reported:
                _gpt_oss_fusion_reported.add("TLLM_GPT_OSS_FUSED_AR_REFUSED")
                # Printed rather than logged: worker processes keep the logger at "error".
                print(
                    "TLLM_GPT_OSS_MOE_FINALIZE_AR / TLLM_GPT_OSS_AR_MXFP8_QUANT refused: TLLM_MXFP4_MOE_NVFP4_ACT=1 "
                    f"runs this MoE on NVFP4 activations; the unfused all-reduce is kept pid={os.getpid()}",
                    file=sys.stderr,
                    flush=True)
            return None
        return impl

    def _fusions_eligible(self) -> bool:
        ar = self.allreduce
        return bool((_GPT_OSS_MOE_FINALIZE_AR or _GPT_OSS_AR_MXFP8_QUANT)
                    and ar is not None and self._mxfp8_moe_impl() is not None
                    # The unfused all-reduce must be the custom one-shot/two-shot kernel path, not MNNVL or
                    # symmetric-memory, for the fused kernels to reproduce it.
                    and getattr(ar, "mnnvl_allreduce", None) is None
                    and getattr(ar, "symm_mem_allreduce", None) is None
                    and not self.mapping.is_multi_node())

    def _try_fused_mlp_nvfp4_input(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        lora_params: Optional[dict],
        all_rank_num_tokens: Optional[list[int]] = None,
    ) -> Optional[tuple[Fp4QuantizedTensor, torch.Tensor]]:
        """Fuse the local GPT-OSS MLP norm and NVFP4 input quantization."""
        if (not self.enable_fused_mlp_nvfp4_input or bool(lora_params)
                or x.dtype != torch.bfloat16 or x.dim() != 2
                or not x.is_contiguous() or residual.shape != x.shape
                or residual.dtype != x.dtype or not residual.is_contiguous()):
            return None

        # The MoE scheduler pads ordinary tensors when any DP rank is empty.
        # Use the shared count list so all ranks take that fallback together.
        if x.shape[0] == 0 or (all_rank_num_tokens is not None
                               and 0 in all_rank_num_tokens):
            return None

        get_quant_params = getattr(self.mlp.experts,
                                   "get_nvfp4_input_quant_params", None)
        if get_quant_params is None:
            return None
        quant_params = get_quant_params()
        if quant_params is None:
            return None
        input_scale, padded_hidden_size = quant_params
        hidden_size = x.shape[-1]
        norm_weight = self.post_attention_layernorm.weight
        if (hidden_size % 16 != 0 or padded_hidden_size < hidden_size
                or padded_hidden_size % 16 != 0
                or input_scale.dtype != torch.float32
                or input_scale.device != x.device or input_scale.numel() != 1
                or norm_weight.shape != (hidden_size, )
                or norm_weight.dtype != x.dtype
                or norm_weight.device != x.device
                or not norm_weight.is_contiguous()):
            return None
        if (padded_hidden_size != hidden_size
                and not hasattr(torch.ops.trtllm, "pad_nvfp4_activation")):
            return None

        fp4, sf, norm_out = add_rmsnorm_fp4_quant(
            x,
            residual,
            norm_weight,
            self.post_attention_layernorm.variance_epsilon,
            input_scale,
            padded_hidden_size,
        )
        return Fp4QuantizedTensor(
            fp4,
            sf,
            is_sf_swizzled=False,
            unquantized_hidden_states=norm_out,
        ), residual

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
        fused_mlp_input = self._try_fused_mlp_nvfp4_input(
            x, residual, lora_params,
            attn_metadata.all_rank_num_tokens if self.enable_attn_dp else None)
        if fused_mlp_input is None:
            x, residual = self.post_attention_layernorm(x, residual)
        else:
            x, residual = fused_mlp_input

        x, residual = self.mlp(x,
                               attn_metadata,
                               residual,
                               lora_params=lora_params)

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

        fusions_apply = (self._fused_ar_eligible
                         and x.shape[0] <= _GPT_OSS_FUSED_AR_MAX_TOKENS
                         and not lora_params)

        mxfp8_out = None
        if _GPT_OSS_AR_MXFP8_QUANT and fusions_apply:
            mxfp8_out = self.allreduce.residual_rms_norm_mxfp8(
                x,
                residual=residual,
                norm_weight=self.post_attention_layernorm.weight,
                eps=self.post_attention_layernorm.variance_epsilon,
                alignment=self._mxfp8_alignment,
            )
        if mxfp8_out is not None:
            norm_out, fp8_out, sf_out, residual = mxfp8_out
            # The prequantized-activation carrier the MoE path accepts (also through moe_custom_op); the
            # MXFP8 leaf passes E4M3 data with row-major UE8M0 scales straight through (fp4_block_scale.py).
            x = Fp4QuantizedTensor(fp8_out,
                                   sf_out.view(fp8_out.shape[0], -1),
                                   is_sf_swizzled=False,
                                   unquantized_hidden_states=norm_out)
        else:
            x, residual = self.allreduce(
                x,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ))

        is_capture_layer = (spec_metadata is not None
                            and spec_metadata.is_layer_capture(self.layer_idx))
        f2_strategy = (self.allreduce.thop_strategy_for_fusion()
                       if self.moe_allreduce is not None and fusions_apply
                       and not is_capture_layer else None)
        if f2_strategy is not None:
            (fc2_output, expert_weights,
             expanded_idx_to_permuted_idx), residual = self.mlp(
                 x,
                 attn_metadata,
                 residual,
                 lora_params=lora_params,
                 do_finalize=False)
            return self.moe_allreduce.finalize_allreduce_match_unfused(
                fc2_output,
                residual=residual,
                norm_weight=self.next_layer_layernorm.weight,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                expert_scale_factor=expert_weights,
                eps=self.next_layer_layernorm.variance_epsilon,
                strategy=f2_strategy,
            )

        x, residual = self.mlp(x,
                               attn_metadata,
                               residual,
                               lora_params=lora_params)

        if is_capture_layer:
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

        residual = None
        for block in self.block:
            hidden_states, residual = block(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
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
        self._maybe_requantize_dense_to_mxfp8()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def _maybe_requantize_dense_to_mxfp8(self) -> None:
        """Opt-in: run the checkpoint's bf16 dense layers as MXFP8 W8A8.

        TLLM_MXFP8_ATTN_PROJ=1: every attention qkv_proj and o_proj.
        TLLM_MXFP8_LM_HEAD=1: the target lm_head and the EAGLE3 one-model
        drafter's lm_head (when it has its own; a drafter that shares the
        target's gets the requantized one).
        Both apply only on single-GPU ranks (mapping.world_size == 1: TP1
        aggregated and TP1 prefill workers); TLLM_MXFP8_LM_HEAD=all also
        allows any rank whose lm_head holds the whole vocabulary. The engine
        quantizes the stored bf16 weights at load and derives every scale
        itself; nothing else is read from disk.
        """
        from ..modules.embedding import LMHead
        from ..modules.linear import requantize_linear_to_mxfp8

        mapping = self.model_config.mapping
        single_gpu = mapping.world_size == 1

        if _MXFP8_ATTN_PROJ:
            if not single_gpu:
                _print_once(
                    "mxfp8-attn-off", "[TLLM mxfp8-attn-proj] "
                    "TLLM_MXFP8_ATTN_PROJ=1 but qkv_proj/o_proj stay bf16: "
                    f"world_size {mapping.world_size} (single-GPU ranks only)")
            else:
                refused, done = [], 0
                for block in self.model.block:
                    for name in ("qkv_proj", "o_proj"):
                        reason = requantize_linear_to_mxfp8(
                            getattr(block.attn, name))
                        if reason is None:
                            done += 1
                        else:
                            refused.append(f"{name}: {reason}")
                _print_once(
                    "mxfp8-attn", f"[TLLM mxfp8-attn-proj] {done} of "
                    f"{2 * len(self.model.block)} GPT-OSS qkv_proj/o_proj "
                    "requantized from the checkpoint's bf16 to MXFP8 at load "
                    "(TLLM_MXFP8_ATTN_PROJ=1)" +
                    (f"; kept bf16: {sorted(set(refused))}" if refused else ""))

        if _MXFP8_LM_HEAD in ("1", "all"):
            heads = [("target", self.lm_head)]
            draft = getattr(self, "draft_model", None)
            spec_config = getattr(self.model_config, "spec_config", None)
            if (draft is not None and spec_config is not None
                    and spec_config.spec_dec_mode.is_eagle3_one_model()
                    and isinstance(getattr(draft, "lm_head", None), LMHead)
                    and draft.lm_head is not self.lm_head):
                heads.append(("draft", draft.lm_head))
            for role, head in heads:
                reason = None
                if not single_gpu and _MXFP8_LM_HEAD != "all":
                    reason = (f"world_size {mapping.world_size} (single-GPU "
                              "ranks only; TLLM_MXFP8_LM_HEAD=all widens it)")
                elif not (head.tp_mode is None or head.tp_size == 1):
                    reason = "the lm_head is vocabulary-sharded"
                elif head.padding_size or head.enable_lm_head_tp_in_adp:
                    reason = "padded vocabulary or lm_head TP in ADP"
                else:
                    reason = requantize_linear_to_mxfp8(head)
                _print_once(
                    f"mxfp8-lm-head-{role}", f"[TLLM mxfp8-lm-head] {role} "
                    f"lm_head [{head.out_features}, {head.in_features}] " +
                    ("requantized from the checkpoint's bf16 to MXFP8 at load "
                     f"(TLLM_MXFP8_LM_HEAD={_MXFP8_LM_HEAD})"
                     if reason is None else f"stays bf16: {reason}"))

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

                    if getattr(module, "mxfp4_as_nvfp4", False):
                        # TLLM_MXFP4_MOE_NVFP4_ACT: the backend repacks these
                        # MXFP4 tensors to NVFP4; its static FC1 input scale is
                        # the bound of this layer's RMSNorm output.
                        norm_key = name.replace(
                            "mlp.experts",
                            "post_attention_layernorm") + ".weight"
                        moe_weights["gate_up_proj_input_scale"] = (
                            MXFP4AsNVFP4TRTLLMGenFusedMoEMethod.
                            input_scale_for_amax(
                                MXFP4AsNVFP4TRTLLMGenFusedMoEMethod.
                                rmsnorm_output_amax_bound(weights[norm_key])))

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
