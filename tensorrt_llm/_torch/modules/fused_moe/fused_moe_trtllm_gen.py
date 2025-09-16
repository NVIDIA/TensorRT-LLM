from typing import Dict, List, Optional, Union

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version

from ...custom_ops.trtllm_gen_custom_ops import \
    fp4_block_scale_fake_output_without_finalize
from ...model_config import ModelConfig
from ...utils import Fp4QuantizedTensor, next_positive_power_of_2
from .interface import MoE, MoEWeightLoadingMode
from .quantization import (DeepSeekFP8BlockScalesFusedMoEMethod,
                           NVFP4TRTLLMGenFusedMoEMethod,
                           W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
                           W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
                           W4A16MXFP4TRTLLMGenFusedMoEMethod)
from .routing import BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod


class TRTLLMGenFusedMoE(MoE):
    """
    Fused Mixture of Experts (MoE) Layer with performance tuning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.

    MoE torch custom op:
        Only support min-latency mode now (SM100 Blackwell only).
        Quant: fp8 block scales quant and nvfp4 quant and w4a16_mxfp4 quant
            FusedMoE Op: routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    FusedMoE module:
        min-latency mode:
            dynamic quant + FusedMoe Op
            equals to: dynamic quant + routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
    AttentionDP should be turned off for min-latency mode.

    When we have redundant expert, we have more weight slots than `num_experts`, in that case, we separate the concepts of expert and slot.
    Expert is the concept from model's perspective while slot is the concept from model engine's perspective.
    There should be at lease `num_experts` slots in the model engine. More than that is OK, in that case, some experts may have multiple replicas.
    """

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        layer_idx: Optional[int] = None,
        bias: bool = False,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            layer_idx=layer_idx,
        )

        sm_version = get_sm_version()
        if sm_version >= 120:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE does not support SM120 and above.")

        assert not self.smart_router, "Smart router is not supported in TRTLLMGenFusedMoE."

        self.num_slots = self.num_experts
        self.expert_size_per_partition = self.num_experts // self.ep_size
        self.initial_global_assignments = [
            (ep_rank * self.num_experts // self.ep_size + local_slot_id) %
            self.num_experts for ep_rank in range(self.ep_size)
            for local_slot_id in range(self.expert_size_per_partition)
        ]
        self.slot_start = self.ep_rank * self.expert_size_per_partition
        self.slot_end = self.slot_start + self.expert_size_per_partition
        self.initial_local_expert_ids = self.initial_global_assignments[
            self.slot_start:self.slot_end]
        assert len(
            self.initial_local_expert_ids) == self.expert_size_per_partition

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _check_configs(self):
        assert self.has_deepseek_fp8_block_scales \
            or self.has_nvfp4 or self.has_w4a16_mxfp4 \
            or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_fp8 and w4a8_mxfp4_mxfp8 dtypes."

        if self.bias or self.swiglu_alpha is not None or self.swiglu_beta is not None or self.swiglu_limit is not None:
            assert self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, "TRTLLMGenFusedMoE only supports mxfp4 quantization with bias, swiglu_alpha, swiglu_beta and swiglu_limit."

    def _get_tile_tokens_dim(self, x: torch.Tensor):
        top_k = self.routing_method.top_k
        # Number of tokens in the input tensor.
        num_tokens = x.shape[0]
        # Factor to account for the imbalance of the experts.
        # factor equals to the max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
        # 1.0 means perfect expert distribution.
        # > 1.0 means some experts have more tokens than the perfect distribution.
        # < 1.0 does not make sense.
        imbalance_factor = 1.3
        # Calculate the number of tokens per expert assuming perfect distribution.
        num_tokens_per_expert = (num_tokens * top_k) // self.num_experts
        # Apply the imbalance factor.
        num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
        # And pad the number to the next power of 2.
        tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
        # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
        tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

        return tile_tokens_dim

    def _get_quant_method(self):
        if self.quant_config is not None:
            if self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a16_mxfp4():
                return W4A16MXFP4TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                return W4A8MXFP4FP8TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8():
                return W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
            else:
                raise NotImplementedError(
                    f"Unsupported quantization method by TRTLLMGenFusedMoE: {self.quant_config.quant_mode}"
                )
        else:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE doesn't support fp16/bf16/fp32 MoE.")

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True
        self._check_configs()

        # TODO: FIX this.
        if (self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8
                or self.has_w4a8_mxfp4_mxfp8) and not self.bias:
            self.w3_w1_bias = nn.Parameter(torch.zeros(
                (self.w3_w1_weight.shape[0], self.w3_w1_weight.shape[1]),
                dtype=torch.float32),
                                           requires_grad=False)
            self.register_parameter("w3_w1_bias", self.w3_w1_bias)
            self.w2_bias = nn.Parameter(torch.zeros(
                (self.w2_weight.shape[0], self.w2_weight.shape[1]),
                dtype=torch.float32),
                                        requires_grad=False)
            self.register_parameter("w2_bias", self.w2_bias)

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created

        assert len(weights) == 1
        weights = weights[0]

        self.quant_method.load_weights(self, weights, self.weight_loading_mode)

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:

        assert x.dtype == torch.bfloat16

        # DeepSeekV3 style routing
        if isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod):
            top_k = self.routing_method.routing_impl.top_k
            routing_bias = self.routing_method.e_score_correction_bias
            n_group = self.routing_method.routing_impl.n_group
            topk_group = self.routing_method.routing_impl.topk_group
            routed_scaling_factor = self.routing_method.routing_impl.routed_scaling_factor
        else:
            top_k = self.routing_method.top_k
            routing_bias = None
            n_group = None
            topk_group = None
            routed_scaling_factor = None

        # TODO: since routing kernel is integrated into moe_runner for fp8,
        #       here we just route the I/Os for moe_runner
        if self.has_deepseek_fp8_block_scales:
            assert do_finalize, "fp8_block_scale_moe_runner does not support do_finalize=False"
            x_val, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x)

            final_hidden_states = torch.ops.trtllm.fp8_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x_val,
                x_scale,
                self.w3_w1_weight,
                self.w3_w1_weight_scaling_factor,
                self.w2_weight,
                self.w2_weight_scaling_factor,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
            )
        elif self.has_nvfp4:
            scale_factor_use_ue8m0 = False
            is_scale_factor_swizzled = False  # use linear layout here
            hidden_states_fp4, hidden_states_scale_linear_fp4 = (
                torch.ops.trtllm.fp4_quantize(
                    x,
                    self.fc31_input_scale,
                    self.scaling_vector_size,
                    scale_factor_use_ue8m0,
                    is_scale_factor_swizzled,
                ))

            outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
                router_logits,
                routing_bias,
                hidden_states_fp4,
                hidden_states_scale_linear_fp4.view(torch.float8_e4m3fn),
                self.w3_w1_weight,
                self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                self.w2_weight,
                self.w2_weight_scale.view(torch.float8_e4m3fn),
                self.fc31_scale_c.data,
                self.fc31_alpha.data,
                self.fc2_alpha.data,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                do_finalize=do_finalize,
            )

            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return outputs
            else:
                final_hidden_states = outputs[0]
        elif self.has_w4a16_mxfp4:
            assert x.dtype == torch.bfloat16

            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))
            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2
            final_hidden_states = torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self._get_tile_tokens_dim(x),
                self.routing_method.routing_method_type,
                0,  # act_type
            )
            final_hidden_states = final_hidden_states[:, :self.
                                                      hidden_size].contiguous()
        elif self.has_w4a8_mxfp4_fp8:
            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))
            x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                x, self.fc31_input_gate_dequant[0])
            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2

            final_hidden_states = torch.ops.trtllm.e4m3_mxe2m1_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.fc31_input_dequant,  # output1_scales_scalar
                self.fc31_input_gate_dequant,  # output1_scales_gate_scalar
                self.fc2_input_dequant,  # output2_scales_scalar
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self._get_tile_tokens_dim(x),
                self.routing_method.routing_method_type,
                0,  # act_type
            )
            final_hidden_states = final_hidden_states[:, :self.
                                                      hidden_size].contiguous()
        elif self.has_w4a8_mxfp4_mxfp8:
            # TRTLLM-Gen uses linear SF layout for the mxfp8 input.
            mxfp8_x, sf = torch.ops.trtllm.mxfp8_quantize(
                x, False, alignment=self.quant_method.weight_alignment)
            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2

            final_hidden_states = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
                router_logits,
                routing_bias,
                mxfp8_x,
                sf,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.hidden_size,
                self.
                slot_start,  # local_expert_start;  use ep_rank if stride!=1
                self.expert_size_per_partition,  # local_expert_size
                routed_scaling_factor,
                self._get_tile_tokens_dim(x),
                self.routing_method.routing_method_type,
                0,  # act_type
            )
        else:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4 and w4a8_mxfp4_fp8 dtypes."
            )

        final_hidden_states = self.reducescatter_or_allreduce(
            final_hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        if use_dp_padding:
            rank = self.mapping.tp_rank
            final_hidden_states = final_hidden_states[:
                                                      all_rank_num_tokens[rank]]
        return final_hidden_states

    def forward_fake(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if do_finalize:
            # TRTLLMGenFusedMoE only supports bfloat16 output
            return super().forward_fake(x,
                                        router_logits,
                                        do_finalize=do_finalize,
                                        output_dtype=torch.bfloat16,
                                        all_rank_num_tokens=all_rank_num_tokens,
                                        use_dp_padding=use_dp_padding,
                                        **kwargs)
        else:
            is_deepseek_v3_routing = isinstance(self.routing_method,
                                                DeepSeekV3MoeRoutingMethod)
            top_k = self.routing_method.routing_impl.top_k if is_deepseek_v3_routing else self.routing_method.top_k
            routing_bias = self.routing_method.e_score_correction_bias if is_deepseek_v3_routing else None
            return fp4_block_scale_fake_output_without_finalize(
                x,
                self.num_experts,
                top_k,
                routing_bias,
            )
