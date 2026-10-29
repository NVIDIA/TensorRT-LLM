# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# --------------------------------------------------
# Portions of this code were derived from DeepSeek‑V3:
#   https://github.com/deepseek-ai/DeepSeek-V3
#
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------

"""GLM-5.2 decoder modules and the checkpoint-compatible prefill path."""

import copy
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._ipc_utils import can_access_peer
from tensorrt_llm._torch.attention.attention import (
    maybe_allgather_for_helix_cp,
    maybe_slice_for_helix_cp,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    MoEAllReduce,
    MoEAllReduceParams,
)
from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.moe.fused_moe import (
    DEFAULT_MOE_ACTIVATION,
    DeepSeekV3MoeRoutingMethod,
    MoEWeightLoadingMode,
    SwigluActivation,
    create_moe,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f, is_sm_120f
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# isort: off
from tensorrt_llm._torch.moe.fused_moe.routing import Deepseekv3RoutingImpl

# isort: on
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, EagerFusionConfig
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import (
    Linear,
    TensorParallelMode,
    WeightsLoadingConfig,
    is_static_nvfp4_input_eligible,
)
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.peft.lora.layer import LoraLayer
from tensorrt_llm._torch.speculative import SpecMetadata
from tensorrt_llm._torch.utils import AuxStreamType, EventType, Fp4QuantizedTensor


@torch.compile(dynamic=True)
def moe_reduce_add_shared_output(routed_output, shared_output, out=None):
    routed_reduced = torch.sum(routed_output, dim=1, keepdim=False)
    if out is not None:
        torch.add(shared_output, routed_reduced, out=out)
        return out
    # In-place add to avoid allocating a temporary tensor, reducing peak memory
    return shared_output.add_(routed_reduced)


def _static_nvfp4_input_scale(linear):
    """Return `linear`'s calibrated NVFP4 input_scale if it is eligible to be
    folded into a producing RMSNorm's fused NVFP4 quantize, else None.

    Eligibility is the shared `is_static_nvfp4_input_eligible` predicate, also
    used by MLA._resolve_qa_fused_scale, so every fold site agrees on what
    "static-NVFP4 eligible" means."""
    if is_static_nvfp4_input_eligible(linear):
        return linear.input_scale
    return None


class Glm52Linear(Linear):
    """
    A wrapper around Linear because we may optionally use min-latency kernels depending on input shapes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,  # COLUMN parallel only
        quant_config: Optional[QuantConfig] = None,
        weights_loading_config: Optional[WeightsLoadingConfig] = None,
        reduce_output: bool = True,  # ROW parallel only
        skip_create_weights_in_init: bool = False,
        use_custom_cublas_mm: bool = False,
        use_cute_dsl_bf16_gemm: bool = False,
        use_cute_dsl_blockscaling_mm: bool = False,
        lora: Optional[LoraLayer] = None,
        locality_domain_policy: Optional[LocalityDomainPolicy] = None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            quant_config=quant_config,
            weights_loading_config=weights_loading_config,
            reduce_output=reduce_output,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=use_custom_cublas_mm,
            use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
            enable_locality_domain_bf16_linear=True,
            locality_domain_policy=locality_domain_policy,
            lora=lora,
            use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
        )

    def apply_linear(
        self,
        input,
        bias,
        lora_params: Optional[dict] | None = None,
        layer_idx: Optional[int] | None = None,
    ):
        if self._locality_domain_weight_shards is not None:
            return super().apply_linear(input, bias, lora_params, layer_idx)
        num_tokens = input.shape[0]
        has_any_quant = self.has_any_quant
        # The fused op has no LoRA inputs; let Linear merge active adapters.
        if (
            not has_any_quant
            and (self.lora is None or not lora_params)
            and 1 <= num_tokens <= 16
            and not is_sm_120f()
        ):
            output = torch.ops.trtllm.dsv3_fused_a_gemm_op(input, self.weight.t(), bias, None)
        else:
            output = super().apply_linear(input, bias, lora_params, layer_idx)
        return output


class Glm52Attention(MLA):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream_dict: Optional[Dict[AuxStreamType, torch.cuda.Stream]] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ):
        config = model_config.pretrained_config
        predicted_tokens_per_seq = (
            model_config.spec_config.tokens_per_gen_step
            if model_config.spec_config is not None
            else 1
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            qk_rope_head_dim=config.qk_rope_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            v_head_dim=config.v_head_dim,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.yarn,
                rope=RopeParams.from_config(config),
                is_neox=False,
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            reduce_output=reduce_output,
            enable_locality_domain_bf16_linear=True,
        )

        self.kv_a_proj_with_mqa = Glm52Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim + self.q_lora_rank,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            use_custom_cublas_mm=True,
            use_cute_dsl_bf16_gemm=model_config.use_cute_dsl_bf16_gemm,
            locality_domain_policy=model_config.locality_domain_policy,
        )


class Glm52Gate(Linear):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        dtype: Optional[torch.dtype] = None,
        fuse_routing_kernel: bool = True,
        apply_routing: bool = False,
        moe_backend: str = "CUTLASS",
        use_cute_dsl_bf16_gemm: bool = False,
        locality_domain_policy: Optional[LocalityDomainPolicy] = None,
    ):
        super().__init__(
            hidden_size,
            num_experts,
            bias=False,
            dtype=dtype,
            reduce_output=False,
            use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
            enable_locality_domain_bf16_linear=True,
            locality_domain_policy=locality_domain_policy,
        )
        self._skip_layerwise_quant_config = True
        self.moe_backend = moe_backend
        if moe_backend == "TRTLLM":
            bias_dtype = torch.bfloat16
        else:
            bias_dtype = torch.float32
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((num_experts), dtype=bias_dtype), requires_grad=False
        )

        assert not apply_routing, "Glm52Gate routing is called inside MoE"

        # NOTE: e_score_correction_bias belongs in this gate class but is required by the routing impl.
        self.routing_impl = Deepseekv3RoutingImpl(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            is_fused=fuse_routing_kernel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._locality_domain_weight_shards is not None:
            logits = self._run_bf16_linear_locality_domain(
                hidden_states, None, output_dtype=torch.float32
            )
        elif self.use_cute_dsl_bf16_gemm and is_sm_100f() and self.weight.dtype == torch.bfloat16:
            input_2d = hidden_states.view(-1, hidden_states.shape[-1])
            m, k = input_2d.shape
            n = self.weight.shape[0]
            output = torch.empty(m, n, dtype=torch.float32, device=hidden_states.device)
            bf16_gemm_op = (
                torch.ops.trtllm.cute_dsl_bf16_gemm_rubin
                if get_sm_version() == 107
                else torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell
            )
            bf16_gemm_op(input_2d.contiguous(), self.weight, output)
            logits = output.view(*hidden_states.shape[:-1], n)
        else:
            logits = torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_states, self.weight.t(), bias=None, out_dtype=torch.float32
            )
        return logits

    def load_weights(self, weights: List[Dict], allow_partial_loading: bool = False):
        assert len(weights) == 1
        w = weights[0].get("weight")
        bias = weights[0].get("e_score_correction_bias")
        if not allow_partial_loading:
            assert w is not None and bias is not None, (
                "Glm52Gate expects 'weight' and 'e_score_correction_bias' "
                "when partial loading is disabled"
            )
        if w is not None:
            self._weights_transformed = False
            self.weight.copy_(w[:])
        if bias is not None:
            self.e_score_correction_bias.copy_(bias[:].to(self.e_score_correction_bias.dtype))

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        return DeepSeekV3MoeRoutingMethod(
            top_k=self.routing_impl.top_k,
            n_group=self.routing_impl.n_group,
            topk_group=self.routing_impl.topk_group,
            routed_scaling_factor=self.routing_impl.routed_scaling_factor,
            is_fused=self.routing_impl.is_fused,
            # Pass a callable to fetch the tensor from Glm52Gate at runtime, ensuring it is on the correct device
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
        )

    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # topk routing
        return self.routing_method.apply(logits)

    def get_experts_per_token(self):
        return self.routing_method.top_k


class Glm52MoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        dtype: Optional[torch.dtype] = None,
        model_config: ModelConfig = ModelConfig(),
        override_quant_config: Optional[QuantConfig] = None,
        layer_idx: Optional[int] = None,
        swiglu_limit_scalar: Optional[float] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.top_k = top_k
        self.use_dp = model_config.mapping.enable_attention_dp
        self.use_cute_dsl_blockscaling_mm = model_config.use_cute_dsl_blockscaling_mm
        gate_cls = Glm52Gate
        if hasattr(model_config.pretrained_config, "gate_cls"):
            gate_cls = model_config.pretrained_config.gate_cls
        self.gate = gate_cls(
            hidden_size,
            num_experts,
            top_k=top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            dtype=dtype,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=model_config.moe_backend,
            use_cute_dsl_bf16_gemm=model_config.use_cute_dsl_bf16_gemm,
            locality_domain_policy=model_config.locality_domain_policy,
        )
        # For MIXED_PRECISION, resolve the per-expert quant config (e.g. W4A8_AWQ)
        # instead of using the ambiguous global MIXED_PRECISION config.
        # For other cases (e.g. nvfp4, unquantized MTP layers), use
        # override_quant_config as-is — it already encodes exclusions like MTP.
        if (
            override_quant_config is not None
            and override_quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
        ):
            expert_quant_config = self._get_experts_quant_config(model_config, layer_idx)
        else:
            expert_quant_config = override_quant_config
        moe_activation = DEFAULT_MOE_ACTIVATION
        if swiglu_limit_scalar is not None:
            moe_activation = SwigluActivation(clamp=swiglu_limit_scalar)
        self.experts = create_moe(
            num_experts=num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=False,  # In both low‑latency and attention‑DP modes, FusedMoE skips the in‑op all‑reduce.
            model_config=model_config,
            override_quant_config=expert_quant_config,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
            # DS-R1 W4A8 is only supported through custom quantization script from
            # examples/quantization/quantize_mixed_precision_moe.py
            weight_loading_mode=self._expert_weight_loading_mode(expert_quant_config),
            activation=moe_activation,
        )

        self.mapping = model_config.mapping

        shared_quant_config = self._get_shared_experts_quant_config(model_config, layer_idx)
        # Shared experts overlap routed MoE on another stream. Disable locality
        # domains for every precision so both paths do not contend for the same
        # global partition streams. Only thaw the private copy of ModelConfig.
        shared_model_config = copy.copy(model_config)
        shared_model_config._frozen = False
        shared_model_config.quant_config = shared_quant_config
        shared_model_config.locality_domain_policy = LocalityDomainPolicy(enabled=False)
        shared_model_config._frozen = model_config._frozen

        # For shared experts, use the block size implied by their quant config.
        block_size = 1
        if (
            shared_quant_config is not None
            and shared_quant_config.quant_algo is not None
            and shared_quant_config.group_size is not None
        ):
            block_size = shared_quant_config.group_size

        self.shared_tp_size, self.shared_output_scale = self._compute_shared_expert_tp_size(
            shared_expert_intermediate_size, block_size
        )

        self.shared_experts = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=shared_model_config,
            overridden_tp_size=self.shared_tp_size,
            reduce_output=False,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            swiglu_limit=swiglu_limit_scalar,
        )
        self.shared_experts_use_fp4 = (
            shared_quant_config is not None and shared_quant_config.layer_quant_mode.has_nvfp4()
        )

        self.allreduce = None
        if not self.use_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping, strategy=model_config.allreduce_strategy
            )
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

    def _compute_shared_expert_tp_size(
        self, intermediate_size: int, block_size: int
    ) -> tuple[int, float | None]:
        """
        In the case of Deepseek-R1, the TP size of MLP is capped by intermediate_size // block_size.
        For example, when the intermediate_size is 2048 and block scaling size is 128,
        TP sizes are limited to {1, 2, 4, 8, 16} because of 2048/128 = 16.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. In the case of Deepseek FP8 recipe,
                it's 128. For NVFP4, it's 16.

        Returns:
            tuple[int, float | None]: A tuple containing (shared_tp_size, shared_output_scale).
                - shared_tp_size: The computed TP size.
                - shared_output_scale: The output scale factor, or None if not needed.
        """

        assert intermediate_size % block_size == 0, (
            "intermediate_size must be divisible by block_size."
        )

        shared_output_scale = None
        # The block scale size is 128, which requires shared_expert_intermediate_size to be divisible by 128.
        if self.use_dp:
            # If using attention DP, the shared experts also use DP instead of TP.
            shared_tp_size = 1
        elif (
            hasattr(self.experts, "num_fused_shared_expert")
            and self.experts.num_fused_shared_expert > 0
        ):
            shared_tp_size = self.mapping.moe_tp_size
        else:
            # The block scale size (128) restricts the supported TP sizes to 1, 2, 4, 8, and 16.
            # The math.gcd operation ensures that shared_tp_size falls in the supported TP sizes.
            shared_tp_size = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )
            # If shared_tp_size was overridden, scale down the shared-expert output
            # accordingly before all-reduce.
            if shared_tp_size != self.mapping.tp_size:
                shared_output_scale = shared_tp_size / self.mapping.tp_size

        return shared_tp_size, shared_output_scale

    @staticmethod
    def _get_experts_quant_config(model_config, layer_idx: int) -> QuantConfig:
        if getattr(model_config, "quant_config_dict", None) is None:
            return model_config.quant_config
        return model_config.quant_config_dict.get(
            f"model.layers.{layer_idx}.mlp.experts", model_config.quant_config
        )

    @staticmethod
    def _expert_weight_loading_mode(
        expert_quant_config: Optional[QuantConfig],
    ) -> MoEWeightLoadingMode:
        # W4A8_CUSTOM only for the int4-weight-per-group experts produced by
        # examples/quantization/quantize_mixed_precision_moe.py; everything else
        # (including the unquantized case, where there is no expert quant config)
        # uses VANILLA. Guard None: override_quant_config is Optional, so the
        # resolved expert config can be None on an unquantized layer.
        if (
            expert_quant_config is not None
            and expert_quant_config.layer_quant_mode.is_int4_weight_only_per_group()
        ):
            return MoEWeightLoadingMode.W4A8_CUSTOM
        return MoEWeightLoadingMode.VANILLA

    @staticmethod
    def _get_shared_experts_quant_config(model_config, layer_idx: int) -> QuantConfig:
        # Prefer explicit per-layer quant config if provided.
        if getattr(model_config, "quant_config_dict", None) is not None:
            base_name = f"model.layers.{layer_idx}.mlp.shared_experts"
            qcfg = model_config.quant_config_dict.get(base_name, None)
            if qcfg is not None:
                return qcfg
            for name, qcfg in model_config.quant_config_dict.items():
                if name.startswith(base_name + "."):
                    return qcfg

        quant_config = model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return quant_config

        base_name = f"model.layers.{layer_idx}.mlp.shared_experts"
        candidates = [
            base_name,
            f"{base_name}.gate_up_proj",
            f"{base_name}.down_proj",
            f"{base_name}.gate_proj",
            f"{base_name}.up_proj",
        ]
        if any(quant_config.is_module_excluded_from_quantization(name) for name in candidates):
            return QuantConfig(
                quant_algo=None, kv_cache_quant_algo=quant_config.kv_cache_quant_algo
            )
        return quant_config

    def compute_routed_output(
        self, hidden_states, hidden_states_fp4, all_rank_num_tokens, do_finalize
    ):
        # max-throughput
        use_dp_padding = False
        # Add DP padding on SM120 for context comm performance
        # TODO: Move this model-agonostic part to MoE
        if self.use_dp and self.mapping.tp_size > 1 and is_sm_120f():
            use_dp_padding = True
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0])
            )

        router_logits = self.gate(hidden_states)

        routed_output = self.experts(
            hidden_states_fp4 if hidden_states_fp4 is not None else hidden_states,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=hidden_states.dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Optional[Fp4QuantizedTensor] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
    ) -> torch.Tensor:
        if not do_finalize:
            assert not self.use_dp

        def _compute_shared_output():
            shared_input = (
                hidden_states_fp4
                if (hidden_states_fp4 is not None and self.shared_experts_use_fp4)
                else hidden_states
            )
            shared_output = self.shared_experts(shared_input)
            if self.shared_output_scale is not None:
                shared_output *= self.shared_output_scale
            return shared_output

        def _compute_routed_output():
            routed_output = self.compute_routed_output(
                hidden_states, hidden_states_fp4, all_rank_num_tokens, do_finalize
            )
            return routed_output

        # NOTE: define compiled helpers at module scope to avoid defining decorators inside compiled frames

        if self.shared_experts is not None:
            routed_output, shared_output = maybe_execute_in_parallel(
                _compute_routed_output,
                _compute_shared_output,
                self.event_dict[EventType.Main],
                self.event_dict[EventType.MoeShared],
                self.aux_stream,
                disable_on_compile=True,
            )
        else:
            # Shared experts have been fused into the routed experts (see post_load_weights);
            # routed_output already contains the shared expert contribution.
            shared_output = None
            routed_output = _compute_routed_output()

        if not do_finalize:
            return [shared_output, *routed_output]
        else:
            if shared_output is None:
                final_hidden_states = routed_output
            elif not isinstance(shared_output, torch.Tensor):
                final_hidden_states = shared_output + routed_output
                if not self.use_dp and self.mapping.tp_size > 1:
                    final_hidden_states = self.allreduce(
                        final_hidden_states, all_reduce_params=final_all_reduce_params
                    )
                return final_hidden_states
            else:
                output_tensor = None
                if not self.use_dp and self.mapping.tp_size > 1:
                    w, actual_kind = torch.ops.trtllm.allocate_output(
                        shared_output, self.allreduce.output_buffer_kind, self.mapping.tp_group
                    )
                    if actual_kind == int(BufferKind.NCCL_WINDOW):
                        output_tensor = w
                if routed_output.dim() == 3:
                    assert shared_output.numel() * self.top_k == routed_output.numel(), (
                        "unmatched tensor shape"
                    )
                    final_hidden_states = moe_reduce_add_shared_output(
                        routed_output, shared_output, out=output_tensor
                    )
                else:
                    assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
                    if output_tensor is not None:
                        final_hidden_states = torch.add(
                            shared_output, routed_output, out=output_tensor
                        )
                    else:
                        # In-place add to avoid allocating a temporary tensor, reducing peak memory
                        final_hidden_states = shared_output.add_(routed_output)

            if not self.use_dp and self.mapping.tp_size > 1:
                final_hidden_states = self.allreduce(
                    final_hidden_states, all_reduce_params=final_all_reduce_params
                )
            return final_hidden_states


class Glm52DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        is_separate_draft_engine: bool = False,
        mapping_with_cp: Optional[Mapping] = None,
    ):
        super().__init__()
        self.model_config = model_config
        self.layer_idx = layer_idx
        self.mapping_with_cp = mapping_with_cp
        self.config = model_config.pretrained_config
        config = self.config

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.mapping = model_config.mapping
        mapping = self.mapping
        self.enable_attention_dp = mapping.enable_attention_dp
        self.mlp_tp_size = mapping.tp_size
        self.is_p2p_supported = can_access_peer(mapping)

        layer_idx_for_attention = layer_idx
        if is_separate_draft_engine:
            # KVCacheManager only support 1 layer for separate draft engine
            layer_idx_for_attention = layer_idx - model_config.pretrained_config.num_hidden_layers

        # When enable_attention_dp is True, TP reduction is skipped since each DP rank
        # works on different batch elements. However, with CP > 1, attention is split
        # across CP ranks for the SAME batch element, so reduction is still needed
        # within the CP group.
        needs_tp_reduce = not self.enable_attention_dp and self.mapping.tp_size > 1
        needs_cp_reduce = mapping_with_cp is not None and mapping_with_cp.has_cp_helix()
        self.self_attn = Glm52Attention(
            model_config,
            layer_idx=layer_idx_for_attention,
            aux_stream_dict=aux_stream_dict,
            mapping_with_cp=mapping_with_cp,
            reduce_output=needs_tp_reduce or needs_cp_reduce,
        )

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get("TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        quant_config = self._get_decoder_layer_quant_config(model_config, layer_idx)
        # For MIXED_PRECISION, the global quant_algo doesn't map to a single
        # QuantMode.  Per-module configs (e.g. expert W4A8_AWQ vs attention
        # FP8_BLOCK_SCALES) are resolved individually where needed, so we
        # conservatively set layer-level flags here.
        self.is_nvfp4 = (
            quant_config.quant_algo != QuantAlgo.MIXED_PRECISION
            and quant_config.layer_quant_mode.has_nvfp4()
        )

        self.allreduce = None
        self.moe_allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
                dtype=config.torch_dtype,
            )
            self.moe_allreduce = MoEAllReduce(self.mapping)

        has_tp = mapping.has_tp()
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
            self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION

            self.mlp = Glm52MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                shared_expert_intermediate_size=self.moe_intermediate_size
                * self.num_shared_experts,
                dtype=config.torch_dtype,
                model_config=model_config,
                override_quant_config=quant_config,
                aux_stream_dict=aux_stream_dict,
                layer_idx=layer_idx,
            )
        else:
            block_size = 1
            if quant_config and quant_config.group_size is not None:
                block_size = quant_config.group_size
            self.mlp_tp_size = self._compute_mlp_tp_size(config.intermediate_size, block_size)

            has_mlp_tp = self.mlp_tp_size > 1
            self.fusion_config.PRE_MLP_FUSION = self.enable_fusion and has_mlp_tp and self.is_nvfp4
            self.fusion_config.POST_MLP_FUSION = self.enable_fusion and has_mlp_tp

            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                overridden_tp_size=self.mlp_tp_size,
                reduce_output=has_mlp_tp,
                use_cute_dsl_blockscaling_mm=model_config.use_cute_dsl_blockscaling_mm,
                enable_locality_domain_bf16_linear=True,
            )

        # On NVFP4 models, construct the boundary/prologue norms as NVFP4 norms.
        # post_load_weights attaches each norm's .nvfp4_scale (the consuming
        # Linear's static input_scale); the fused (add+)RMSNorm+NVFP4-quant then
        # fires automatically from that attribute (RMSNorm.forward) only when a
        # scale was attached, kernel chosen by the size gate. input_layernorm
        # serves both layer 0's residual-less prologue and, as the previous
        # layer's next_layer_layernorm, the layer-boundary add+RMSNorm edge.
        nvfp4_norm = "nvfp4" if self.is_nvfp4 else None
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            quantize_type=nvfp4_norm,
        )

        # When enable_attention_dp is True, we normally skip attention all-reduce since each
        # DP rank works on different batch elements. However, with CP > 1, attention is split
        # across CP ranks for the SAME batch element, so all-reduce is still needed.
        has_cp = mapping_with_cp is not None and mapping_with_cp.cp_size > 1
        can_skip_for_attention_dp = self.enable_attention_dp and not has_cp
        self.disable_attn_allreduce = (
            self.fusion_config.PRE_MOE_FUSION
            or self.fusion_config.PRE_MLP_FUSION
            or self.mapping.tp_size == 1
            or can_skip_for_attention_dp
        )

        # Dense (GatedMLP) layers fold post_attention_layernorm -> NVFP4
        # gate_up_proj; MoE layers route expert-input quant differently and
        # never get an .nvfp4_scale attached, so the fused path stays inert for
        # them even when constructed as an NVFP4 norm.
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            quantize_type=nvfp4_norm,
        )
        self.next_layer_layernorm: RMSNorm = None

    def _get_decoder_layer_quant_config(
        self, model_config: ModelConfig[PretrainedConfig], layer_idx: int
    ):
        """
        The MTP layer in the nvfp4 checkpoint is unquantized. Because the TRTLLM
        moe_backend only supports fp8/fp4 quantization, we need to override
        the quant_config for the MTP layer.
        """
        quant_config = model_config.quant_config

        layer_name = f"model.layers.{layer_idx}"
        if quant_config.is_module_excluded_from_quantization(layer_name):
            return QuantConfig(
                quant_algo=None,
                kv_cache_quant_algo=quant_config.kv_cache_quant_algo,
            )
        else:
            return model_config.quant_config

    def _compute_mlp_tp_size(self, intermediate_size: int, block_size: int) -> int:
        """
        For DeepSeek‑R1, MLP TP size is limited by intermediate_size // block_size.
        Without the fused post-MLP all-reduce it is also capped at gpus_per_node to
        avoid an expensive inter‑node allreduce.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. In the case of Deepseek FP8 recipe,
                it's 128. For NVFP4, it's 16.

        Returns:
            int: The computed tp_size.
        """

        assert intermediate_size % block_size == 0, (
            "intermediate_size must be divisible by block_size."
        )
        if self.enable_attention_dp:
            # If using attention DP, the MLP also uses DP instead of TP.
            mlp_tp_size = 1
        else:
            # The two math.gcd operations ensure that mlp_tp_size falls in the candidate TP sizes.
            tp = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )

            if tp > self.mapping.gpus_per_node and not self.enable_fusion:
                # Without the fused post-MLP all-reduce the MLP reduces within
                # its own group, so a node-local MLP TP stays correct.
                mlp_tp_size = math.gcd(
                    tp,
                    self.mapping.gpus_per_node,
                )  # Avoid costly inter-node TP
            else:
                # The fused post-MLP all-reduce sums over every rank of the TP
                # group, so the MLP keeps the full TP: a smaller MLP TP would
                # count each partial sum tp_size / mlp_tp_size times.
                mlp_tp_size = tp
        return mlp_tp_size

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            if self.input_layernorm.nvfp4_scale is not None:
                # Layer-0 prologue: the residual-less input_layernorm folds the
                # NVFP4 kv_a_proj input-quant via its attached .nvfp4_scale
                # (RMSNorm.forward), returning an Fp4QuantizedTensor.
                # return_norm_out stashes the BF16 view DSA's pre_indexer_proj
                # needs on that tensor's unquantized_hidden_states.
                hidden_states = self.input_layernorm(hidden_states, return_norm_out=True)[0]
            else:
                hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )
        residual = maybe_slice_for_helix_cp(
            residual, attn_metadata, self.mapping_with_cp, self.layer_idx
        )
        if isinstance(self.mlp, Glm52MoE):
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MOE_FUSION = False
            return self.forward_MoE(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MLP_FUSION = False
            assert isinstance(self.mlp, GatedMLP)
            return self.forward_mlp(
                hidden_states=hidden_states,
                residual=residual,
                spec_metadata=spec_metadata,
            )

    def forward_MoE(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _run_MoE(hidden_states, hidden_states_fp4, do_finalize):
            return self.mlp(
                hidden_states,
                hidden_states_fp4,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(
                    enable_allreduce=not (
                        self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                    )
                ),
                do_finalize=do_finalize,
            )

        if self.fusion_config.PRE_MOE_FUSION:
            # moe_backend can be either CUTLASS or TRTLLM here
            # TODO: unify the two min-latency MoE backends by enabling quant fusion
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Note: this fusion pattern is only supported for single-node TRTLLM-nvfp4 backend now
        do_finalize = self.mapping.is_multi_node() or (
            not (
                self.fusion_config.POST_MOE_FUSION
                and hidden_states.shape[0] <= self.moe_allreduce.max_token
                and self.model_config.moe_backend == "TRTLLM"
                and self.mlp.experts.has_nvfp4
                and self.is_p2p_supported
            )
        )

        hidden_states = _run_MoE(hidden_states, hidden_states_fp4=None, do_finalize=do_finalize)

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                        trigger_completion_at_end=False,
                    ),
                )
            else:
                assert len(hidden_states) == 4, "hidden_states must have 4 elements"

                shared_output = hidden_states[0]
                fc2_output = hidden_states[1]
                expert_scale_factor = hidden_states[2]
                expanded_idx_to_permuted_idx = hidden_states[3]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=shared_output,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params
                )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual

    def _apply_next_layer_layernorm(self, hidden_states, residual):
        """Apply the next layer's input_layernorm at a layer boundary on the
        attention-DP / no-allreduce path. When that norm has an .nvfp4_scale
        attached, RMSNorm.forward folds the next layer's NVFP4 kv_a_proj
        input-quant into the same kernel and returns (Fp4QuantizedTensor,
        residual_out) with the BF16 view stashed for DSA's pre_indexer_proj
        (return_norm_out); otherwise it applies the plain add+RMSNorm. Shared by
        forward_MoE and forward_mlp so the boundary fold stays identical for MoE
        and dense layers."""
        if self.next_layer_layernorm is None:
            return hidden_states, residual
        if self.next_layer_layernorm.nvfp4_scale is not None:
            return self.next_layer_layernorm(hidden_states, residual, return_norm_out=True)
        return self.next_layer_layernorm(hidden_states, residual)

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_config.PRE_MLP_FUSION:
            if self.mlp.gate_up_proj.has_nvfp4:
                act_fp4, act_sf, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        scale=self.mlp.gate_up_proj.input_scale,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ),
                )
                hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
            else:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.post_attention_layernorm.weight,
                        eps=self.post_attention_layernorm.variance_epsilon,
                    ),
                )
        elif self.post_attention_layernorm.nvfp4_scale is not None:
            # Attention-DP path (no allreduce): post_attention_layernorm folds
            # the dense MLP's NVFP4 gate_up_proj input-quant into one kernel via
            # its attached .nvfp4_scale (RMSNorm.forward), mirroring the
            # PRE_MLP_FUSION quant but without the allreduce.
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            # No fusion
            # We need to add twoshot allreduce here to avoid modifying MLA logic
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(
            hidden_states,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MLP_FUSION or self.mlp_tp_size == 1)
            ),
        )

        if self.fusion_config.POST_MLP_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                ),
            )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            hidden_states, residual = self._apply_next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual


class Glm52Model(DecoderModel):
    def __init__(
        self, model_config: ModelConfig[PretrainedConfig], mapping_with_cp: Optional[Mapping] = None
    ):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.mapping_with_cp = mapping_with_cp
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList(
            [
                Glm52DecoderLayer(
                    model_config, layer_idx, self.aux_stream_dict, mapping_with_cp=mapping_with_cp
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None

        for idx, decoder_layer in enumerate(self.layers[: self.num_hidden_layers]):
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        hidden_states = maybe_allgather_for_helix_cp(
            hidden_states, attn_metadata, self.mapping_with_cp
        )
        return hidden_states


def setup_aliases(model) -> None:
    for idx, layer in enumerate(model.model.layers[: model.config.num_hidden_layers]):
        if idx == model.config.num_hidden_layers - 1:
            layer.next_layer_layernorm = model.model.norm
        else:
            next_layer = model.model.layers[idx + 1]
            layer.next_layer_layernorm = next_layer.input_layernorm

        # Attach each norm's .nvfp4_scale (the consuming Linear's static
        # input_scale) so RMSNorm.forward folds (add+)RMSNorm + NVFP4
        # input-quant into one kernel on the attention-DP path. The fold is
        # self-gating: _static_nvfp4_input_scale returns None (no fold)
        # unless that Linear is NVFP4 with a static (calibrated) input_scale,
        # so this is safe to run unconditionally on every layer.
        #
        # input_layernorm -> this layer's kv_a_proj covers BOTH the layer-0
        # residual-less prologue AND the layer-boundary add+RMSNorm edge (the
        # previous layer's next_layer_layernorm IS this input_layernorm).
        self_attn = getattr(layer, "self_attn", None)
        layer.input_layernorm.nvfp4_scale = _static_nvfp4_input_scale(
            getattr(self_attn, "kv_a_proj_with_mqa", None)
        )

        # Intra-layer dense-MLP fold: post_attention_layernorm -> this
        # layer's NVFP4 gate_up_proj. Only dense (GatedMLP) layers — MoE
        # layers route their expert input quant differently and get no
        # scale, so their fused path stays inert.
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, GatedMLP):
            layer.post_attention_layernorm.nvfp4_scale = _static_nvfp4_input_scale(
                getattr(mlp, "gate_up_proj", None)
            )

        # Note: merge shared expert into FusedMoe module
        if idx >= model.config.first_k_dense_replace and idx % model.config.moe_layer_freq == 0:
            if (
                hasattr(layer.mlp.experts, "num_fused_shared_expert")
                and layer.mlp.experts.num_fused_shared_expert > 0
            ):
                layer.mlp.experts.fuse_shared_expert(layer.mlp.shared_experts)
                layer.mlp.shared_experts = None

    # Also process MTP layers if present
    if model.draft_model is not None and hasattr(model.draft_model, "mtp_layers"):
        for layer in model.model.layers[model.config.num_hidden_layers :]:
            # MTP layers also have MoE, need to fuse shared experts
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                if (
                    hasattr(layer.mlp.experts, "num_fused_shared_expert")
                    and layer.mlp.experts.num_fused_shared_expert > 0
                ):
                    layer.mlp.experts.fuse_shared_expert(layer.mlp.shared_experts)
                    layer.mlp.shared_experts = None
