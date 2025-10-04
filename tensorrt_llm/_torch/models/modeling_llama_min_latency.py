from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from transformers import LlamaConfig

from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a)

from ...models.modeling_utils import QuantConfig
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PredefinedAttentionMask
from ..model_config import ModelConfig
from ..modules.fused_moe import (BaseMoeRoutingMethod, CutlassFusedMoE,
                                 FusedMoEQuantScalesFP8,
                                 Llama4RenormalizeMoeRoutingMethod,
                                 MoEWeightLoadingMode)
from ..modules.gated_mlp import GatedMLP, swiglu
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..speculative import SpecMetadata
from ..utils import Fp4QuantizedTensor
from .modeling_llama import Llama4Attention, Llama4DecoderLayer, Llama4MoE

# Perf heuristics thresholds.
# Use routing gemv kernels when num_tokens <= 8.
MIN_LATENCY_ROUTING_GEMM_NUM_TOKENS = 8
# Use QKV gemv kernel with fused attn scaling when num_tokens <= 8.
MIN_LATENCY_QKV_GEMM_ATTN_SCALING_NUM_TOKENS = 8
# Use QKV gemv kernel when num_tokens <= 4.
MIN_LATENCY_QKV_GEMM_NUM_TOKENS = 4
# Use FC2 trtllm-gen kernel when num_tokens <= 16.
MIN_LATENCY_FC2_NUM_TOKENS = 16
# Use FC13 gemv kernel with fused swiglu when num_tokens < 4.
MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_GEMV = 4
# Use FC13 trtllm-gen kernel with fused swiglu when 4 <= num_tokens <= 16.
MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_TRTLLM_GEN = 16
# Use min-latency MoE kernels when num_tokens <= 8.
MIN_LATENCY_FUSED_MOE_NUM_TOKENS = 8


class Llama4MinLatencyLinear(Linear):
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
        gather_output: bool = False,
        quant_config: Optional[QuantConfig] = None,
        weights_loading_config: Optional[WeightsLoadingConfig] = None,
        reduce_output: bool = True,
        skip_create_weights_in_init: bool = False,
        use_custom_cublas_mm: bool = False,
        enable_fused_gemm_swiglu: bool = False,
        enable_fused_gemm_attn_scaling: bool = False,
        enable_trtllm_gen: bool = False,
    ):
        # First, initialize the base class.
        super().__init__(
            in_features,
            out_features,
            bias,
            dtype,
            mapping,
            tensor_parallel_mode,
            gather_output,
            quant_config,
            weights_loading_config,
            reduce_output,
            skip_create_weights_in_init,
            use_custom_cublas_mm,
        )

        # Set min-latency specific attributes.
        self.enable_fused_gemm_swiglu = enable_fused_gemm_swiglu
        self.enable_fused_gemm_attn_scaling = enable_fused_gemm_attn_scaling
        self.enable_trtllm_gen = enable_trtllm_gen
        self.position_ids = None

    def load_weights(self, weights: List[Dict]):

        super().load_weights(weights)

        # After loading weights, calculate the combined scale (input_scale * weight_scale) for special kernels and
        # trtllm-gen kernels.
        if self.has_fp8_qdq:
            if self.weight_scale.device != self.input_scale.device:
                self.weight_scale = torch.nn.Parameter(
                    self.weight_scale.to(self.input_scale.device))
            self.combined_scale = self.input_scale * self.weight_scale

            # If this is gate_up_proj + swiglu and trtllm-gen kernels will be used, we need to reorder the weights
            # for trtllm-gen gemm+swiglu kernels.
            if self.enable_trtllm_gen and self.enable_fused_gemm_swiglu:
                reordered_weight = self.weight.reshape(2,
                                                       self.out_features // 2,
                                                       self.in_features)
                reordered_weight = torch.cat(
                    [reordered_weight[1], reordered_weight[0]], dim=0)
                reordered_weight = reorder_rows_for_gated_act_gemm(
                    reordered_weight)
                self.trtllm_gen_weight = shuffle_matrix_a(
                    reordered_weight.view(torch.uint8),
                    128).view(torch.float8_e4m3fn)

            # Otherwise, if trtllm-gen kernels will be used, shuffle the weights.
            elif self.enable_trtllm_gen:
                self.trtllm_gen_weight = shuffle_matrix_a(
                    self.weight.view(torch.uint8),
                    128).view(torch.float8_e4m3fn)

    # Override apply_linear instead of forward so that we can reuse the AllReduce/AllGather logic in the parent class.
    def apply_linear(
        self,
        input,
        bias,
        lora_params: Optional[dict] | None = None,
        layer_idx: Optional[int] | None = None,
    ) -> torch.Tensor:

        # Quantize the input if it is not already in float8_e4m3fn.
        # We cannot do this when enable_fused_gemm_swiglu is True and num_tokens > 16 because the default path
        # does not support FP8 input.
        if self.has_fp8_qdq \
            and input.dtype != torch.float8_e4m3fn:
            input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                input, self.input_scale)

        # Use special BF16-input BF16-output gemm kernel for routing gemm when num_tokens <= 8.
        if input.shape[0] <= MIN_LATENCY_ROUTING_GEMM_NUM_TOKENS \
            and self.in_features == 5120 \
            and self.out_features == 128 \
            and input.dtype == torch.bfloat16 \
            and self.weight.dtype == torch.bfloat16 \
            and not self.has_any_quant:
            return torch.ops.trtllm.llama4_bf16_bf16_gemm(input, self.weight)

        # Use special FP8-input BF16-output gemm kernel for QKV-gemm.
        if input.shape[0] <= MIN_LATENCY_QKV_GEMM_ATTN_SCALING_NUM_TOKENS \
            and self.in_features == 5120 \
            and self.out_features == 896 \
            and input.dtype == torch.float8_e4m3fn:

            # Check if we should fuse attn_scaling into QKV-gemm.
            should_fuse_attn_scaling = self.enable_fused_gemm_attn_scaling \
                and self.position_ids is not None

            # When num_tokens < 4, we use the special QKV-gemm kernel.
            # Or if attn_scaling is needed, we also use the special QKV-gemm kernel.
            if input.shape[
                    0] <= MIN_LATENCY_QKV_GEMM_NUM_TOKENS or should_fuse_attn_scaling:
                return torch.ops.trtllm.llama4_fp8_bf16_gemm(
                    input,
                    self.weight,
                    self.combined_scale,
                    self.position_ids,
                )
            # Otherwise, we use the trtllm-gen QKV-gemm kernel.
            else:
                if not hasattr(self, "trtllm_gen_weight"):
                    raise ValueError('Expect trtllm_gen_weight to be set')
                return torch.ops.trtllm.fp8_per_tensor_scaling_tllmg_gemm(
                    input,
                    self.trtllm_gen_weight,
                    global_scale=self.combined_scale,
                    out_dtype=torch.bfloat16,
                    low_latency_kernel=True,
                    gated_silu=False,
                )

        # Use trtllm-gen FP8-input BF16-output gemm kernel for FC2.
        if input.shape[0] <= MIN_LATENCY_FC2_NUM_TOKENS \
            and self.in_features % 512 == 0 \
            and self.out_features == 5120 \
            and input.dtype == torch.float8_e4m3fn:
            if not hasattr(self, "trtllm_gen_weight"):
                raise ValueError('Expect trtllm_gen_weight to be set')
            return torch.ops.trtllm.fp8_per_tensor_scaling_tllmg_gemm(
                input,
                self.trtllm_gen_weight,
                global_scale=self.combined_scale,
                out_dtype=torch.bfloat16,
                low_latency_kernel=True,
                gated_silu=False,
            )

        # Use special FP8-input FP8-output gemm+swiglu kernel for FC13+swiglu.
        if self.enable_fused_gemm_swiglu and self.has_fp8_qdq:
            # When num_tokens < 4, we use the special gemm+swiglu kernel.
            if input.shape[
                    0] < MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_GEMV:
                if not hasattr(self, "inv_output_scale"):
                    raise ValueError('Expect inv_output_scale to be set')
                return torch.ops.trtllm.llama4_fp8_fp8_gemm_swiglu(
                    input,
                    self.weight,
                    self.combined_scale,
                    self.inv_output_scale,
                )
            # When 4 <= num_tokens <= 16, we use the trtllm-gen gemm+swiglu kernel.
            elif MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_GEMV <= input.shape[0] \
                and input.shape[0] <= MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_TRTLLM_GEN:
                if not hasattr(self, "trtllm_gen_global_scale"):
                    raise ValueError('Expect trtllm_gen_global_scale to be set')
                if not hasattr(self, "trtllm_gen_weight"):
                    raise ValueError('Expect trtllm_gen_weight to be set')
                return torch.ops.trtllm.fp8_per_tensor_scaling_tllmg_gemm(
                    input,
                    self.trtllm_gen_weight,
                    global_scale=self.trtllm_gen_global_scale,
                    global_scale_gate=self.combined_scale,
                    out_dtype=torch.float8_e4m3fn,
                    low_latency_kernel=True,
                    gated_silu=True,
                )

        # If special gemm+swiglu kernel is not used and enable_fused_gemm_swiglu is True, we need to apply swiglu
        # manually.
        if self.enable_fused_gemm_swiglu:
            intermediate = super().apply_linear(input, bias, lora_params,
                                                layer_idx)
            return swiglu(intermediate)

        # Otherwise, call the default apply_linear method.
        return super().apply_linear(input, bias, lora_params, layer_idx)

    # Set the position_ids for the next call to apply_linear.
    def set_position_ids(self, position_ids: Optional[torch.LongTensor] = None):
        self.position_ids = position_ids


class Llama4MinLatencyGatedMLP(GatedMLP):
    """
    A wrapper around GatedMLP so that we can pre-compute some scales needed by the special kernels.
    """

    def __init__(self,
                 *,
                 hidden_size: int,
                 intermediate_size: int,
                 bias: bool,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
                 dtype: Optional[torch.dtype] = None,
                 config: Optional[ModelConfig] = None,
                 overridden_tp_size: Optional[int] = None,
                 reduce_output: bool = True,
                 layer_idx: Optional[int] = None):

        # First, initialize the base class.
        super().__init__(hidden_size=hidden_size,
                         intermediate_size=intermediate_size,
                         bias=bias,
                         activation=activation,
                         dtype=dtype,
                         config=config,
                         overridden_tp_size=overridden_tp_size,
                         reduce_output=reduce_output,
                         layer_idx=layer_idx)

        # Override gate_up_proj and down_proj with Llama4Linear if we want to use special kernels.
        self.enable_fused_gemm_swiglu = False
        if self.hidden_size == 5120 \
            and (self.intermediate_size == 16384 or self.intermediate_size == 8192) \
            and self.mapping.tp_size == 8 \
            and config.quant_config.quant_mode.has_fp8_qdq():

            self.enable_fused_gemm_swiglu = True
            self.gate_up_proj = Llama4MinLatencyLinear(
                self.hidden_size,
                self.intermediate_size * 2,
                bias=bias,
                dtype=dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_GATE_UP_LINEAR),
                quant_config=config.get_quant_config(),
                reduce_output=reduce_output,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                enable_fused_gemm_swiglu=True,
                enable_trtllm_gen=True,
            )

            self.down_proj = Llama4MinLatencyLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=bias,
                dtype=dtype,
                mapping=self.mapping,
                tensor_parallel_mode=TensorParallelMode.ROW,
                quant_config=config.get_quant_config(),
                reduce_output=reduce_output,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                enable_trtllm_gen=True,
            )

    # After loading both gate_up_proj and down_proj, we need to set the scales needed by the special kernels and by
    # the trtllm-gen gemm+swiglu kernel.
    def post_load_weights(self):
        if self.gate_up_proj.has_fp8_qdq:
            # For the special gemm+swiglu kernel, we need to set the inverse of the output scale, which is the inverse
            # of down_proj's combined input scale.
            self.gate_up_proj.inv_output_scale = 1.0 / self.down_proj.input_scale
            # For the trtllm-gen gemm+swiglu kernel, we need to set the global scale, which is gate_up_proj's
            # combined input scale times inv_output_scale.
            self.gate_up_proj.trtllm_gen_global_scale = self.gate_up_proj.combined_scale * self.gate_up_proj.inv_output_scale

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        # When gemm+swiglu is fused, we need to temporarily disable the activation function to avoid reapplying swiglu.
        if self.enable_fused_gemm_swiglu:
            orig_activation = self.activation
            self.activation = None

        # Call the parent's forward method.
        output = super().forward(
            x,
            all_rank_num_tokens,
            final_all_reduce_params,
            lora_params,
        )

        # Restore the original activation function.
        if self.enable_fused_gemm_swiglu:
            self.activation = orig_activation

        return output


class Llama4MinLatencyAttention(Llama4Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
        use_qk_norm: bool = False,
        nope_layer: bool = False,
        attn_temperature_tuning: bool = True,
        aux_stream: Optional[torch.cuda.Stream] = None,
        attention_chunk_size: Optional[int] = None,
    ):

        # First, initialize the base class.
        super().__init__(
            model_config=model_config,
            layer_idx=layer_idx,
            use_qk_norm=use_qk_norm,
            nope_layer=nope_layer,
            attn_temperature_tuning=attn_temperature_tuning,
            aux_stream=aux_stream,
            attention_chunk_size=attention_chunk_size,
        )

        # Then, enable min-latency QKV gemm when the sizes are as expected.
        config = model_config.pretrained_config
        tp_size = model_config.mapping.tp_size
        self.enable_min_latency_qkv = False
        self.enable_fused_gemm_attn_scaling = False

        if config.hidden_size == 5120 \
            and config.num_attention_heads == 40 \
            and config.num_key_value_heads == 8 \
            and tp_size == 8:
            self.enable_min_latency_qkv = True

            # Decide whether to fuse attn_scaling into QKV gemm.
            self.enable_fused_gemm_attn_scaling = self.attn_temperature_tuning \
                and self.floor_scale == 8192.0 \
                and self.attn_scale == 0.1

            # When min-latency QKV gemm is enabled, override qkv_proj.
            self.qkv_proj = Llama4MinLatencyLinear(
                self.hidden_size,
                tp_size * self.q_size + 2 * tp_size * self.kv_size,
                bias=config.attention_bias,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR),
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                enable_fused_gemm_attn_scaling=self.
                enable_fused_gemm_attn_scaling,
                enable_trtllm_gen=True,
            )

    def _forward_nope(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        # If we are going to use min-latency gemm+attn_scaling kernel, pass position_ids to QKV gemm and set
        # skip_attn_scaling to True.
        skip_attn_scaling = False
        if self.enable_min_latency_qkv \
            and self.enable_fused_gemm_attn_scaling \
            and position_ids is not None \
            and hidden_states.shape[0] <= MIN_LATENCY_QKV_GEMM_ATTN_SCALING_NUM_TOKENS:
            self.qkv_proj.set_position_ids(position_ids)
            skip_attn_scaling = True

        return super()._forward_nope(position_ids, hidden_states, attn_metadata,
                                     attention_mask, all_reduce_params,
                                     skip_attn_scaling)


class Llama4MinLatencyFusedMoE(CutlassFusedMoE):

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
        aux_stream: torch.cuda.Stream = torch.cuda.Stream(),
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
    ):

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream=aux_stream,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        # Enable min-latency mode for Llama4 Maverick TP8 EP1.
        self.enable_min_latency_fused_moe = False
        if num_experts == 128 \
            and hidden_size == 5120 \
            and intermediate_size == 8192 \
            and model_config.quant_config is not None \
            and model_config.quant_config.quant_mode.has_fp8_qdq() \
            and model_config.mapping.moe_tp_size == 8 \
            and model_config.mapping.moe_ep_size == 1 \
            and routing_method.top_k == 1 \
            and apply_router_weight_on_input:
            self.enable_min_latency_fused_moe = True

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        x_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Use special min-latency MoE kernels when num_tokens <= 8.
        if self.enable_min_latency_fused_moe \
            and x.dtype == torch.float8_e4m3fn \
            and x.shape[0] <= MIN_LATENCY_FUSED_MOE_NUM_TOKENS:
            assert hasattr(self, "min_latency_quant_scales"
                           ), "Expect min_latency_quant_scales to be set"

            return torch.ops.trtllm.llama4_moe_tp8ep1_min_latency(
                x, router_logits, self.w3_w1_weight, self.w2_weight,
                self.min_latency_quant_scales)

        # Default MoE implementation does not support FP8 input, so use high-precision one instead.
        if x.dtype == torch.float8_e4m3fn:
            if x_high is not None:
                x = x_high
            else:
                raise ValueError(
                    "x_high is required when x.dtype is float8_e4m3fn in Llama4FusedMoE fallback path!"
                )

        return super().forward(x,
                               router_logits,
                               do_finalize=True,
                               output_dtype=output_dtype)


class Llama4MinLatencyMoE(Llama4MoE):

    def __init__(
            self,
            *,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            shared_expert_intermediate_size: int,
            aux_stream: torch.cuda.Stream,
            dtype: Optional[torch.dtype] = None,
            tune_max_num_tokens: int = 8192,
            model_config: ModelConfig = ModelConfig(),
    ):

        # First, initialize the base class.
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            aux_stream=aux_stream,
            dtype=dtype,
            tune_max_num_tokens=tune_max_num_tokens,
            model_config=model_config,
        )

        # Then, override modules with min-latency versions.
        self.shared_expert = Llama4MinLatencyGatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            reduce_output=False)

        self.experts = Llama4MinLatencyFusedMoE(
            routing_method=Llama4RenormalizeMoeRoutingMethod(top_k),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=
            False,  # In both low latency and max-throughput scenarios, FusedMoE needs not to do allreduce inside op.
            weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            model_config=model_config,
            apply_router_weight_on_input=True,
        )

        self.router = Llama4MinLatencyLinear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=model_config.pretrained_config.torch_dtype,
            quant_config=None)

    def post_load_weights(self):
        # Set min-latency quant scales for routed experts if we plan to use min-latency MoE kernels.
        # This is because the routed experts' input scale is after the score multiplication, so we must use the
        # pre-score scaling input scale, which happens to be shared expert's input scale.
        if self.experts.enable_min_latency_fused_moe and hasattr(
                self.shared_expert.gate_up_proj, "input_scale"):
            pre_score_scaling_input_scale = self.shared_expert.gate_up_proj.input_scale
            self.experts.min_latency_quant_scales = FusedMoEQuantScalesFP8(
                fc1_dequant=self.experts.fc31_dequant.data /
                self.experts.fc31_input_dequant.data *
                pre_score_scaling_input_scale,
                fc2_quant=self.experts.fc2_quant,
                fc2_dequant=self.experts.fc2_dequant,
                fc1_input_dequant=pre_score_scaling_input_scale,
            )

    def compute_routed_output(
            self,
            hidden_states,
            all_rank_num_tokens,
            hidden_states_high: Optional[torch.Tensor] = None):
        # Use high precision hidden states for routing gemm if it is provided.
        hidden_states_routing = hidden_states_high if hidden_states_high is not None else hidden_states
        router_logits = self.router.forward(hidden_states_routing)
        routed_output = self.experts.forward(
            hidden_states,
            router_logits,
            x_high=hidden_states_high,
        )

        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        # Optional input for routing gemm if experts and routing gemm require
        # different precisions for input hidden states.
        hidden_states_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        fn0 = lambda: self.shared_expert(hidden_states)
        fn1 = lambda: self.compute_routed_output(
            hidden_states, all_rank_num_tokens, hidden_states_high)
        shared_output, routed_output = maybe_execute_in_parallel(
            fn0, fn1, self.moe_event[0], self.moe_event[1], self.aux_stream)

        assert shared_output.size() == routed_output.size(
        ), f'unmatched tensor shape'
        final_hidden_states = shared_output + routed_output
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            final_hidden_states = self.all_reduce(
                final_hidden_states, all_reduce_params=final_all_reduce_params)

        return final_hidden_states


class Llama4MinLatencyDecoderLayer(Llama4DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # First, initialize the base class.
        super().__init__(model_config, layer_idx, aux_stream)

        # Then, override modules with min-latency versions.
        config = model_config.pretrained_config
        nope_layer = config.no_rope_layers[layer_idx] == 0
        attention_chunk_size = getattr(config, "attention_chunk_size",
                                       None) if not nope_layer else None
        self.self_attn = Llama4MinLatencyAttention(
            model_config,
            layer_idx=layer_idx,
            use_qk_norm=getattr(config, "use_qk_norm", False),
            nope_layer=nope_layer,
            attn_temperature_tuning=config.attn_temperature_tuning > 0,
            aux_stream=aux_stream,
            attention_chunk_size=attention_chunk_size)

        self.fusion_config.PRE_MLP_FUSION = False
        self.fusion_config.POST_MLP_FUSION = False
        self.fusion_config.PRE_MOE_FUSION = False
        self.fusion_config.POST_MOE_FUSION = False

        if self.is_mlp_layer:
            self.feed_forward = Llama4MinLatencyGatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                # Llama4 has no mlp_bias field.
                bias=getattr(config, "mlp_bias", False),
                dtype=config.torch_dtype,
                config=model_config,
                overridden_tp_size=1 if self.enable_attention_dp else None,
                layer_idx=layer_idx)

            self.fusion_config.PRE_MLP_FUSION = model_config.mapping.has_tp()
            self.fusion_config.POST_MLP_FUSION = model_config.mapping.has_tp()
        else:
            self.feed_forward = Llama4MinLatencyMoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                shared_expert_intermediate_size=config.intermediate_size,
                model_config=model_config,
                aux_stream=aux_stream,
                dtype=config.torch_dtype)

            self.fusion_config.PRE_MOE_FUSION = model_config.mapping.has_tp()
            self.fusion_config.POST_MOE_FUSION = model_config.mapping.has_tp()

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:

        num_tokens = hidden_states.shape[0]

        use_fp8_allreduce = num_tokens <= 128 and self.is_fp8_quant
        use_fp4_allreduce = num_tokens <= 128 and self.is_nvfp4

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.PRE_MOE_FUSION
                or self.fusion_config.PRE_MLP_FUSION
                or self.mapping.tp_size == 1 or self.enable_attention_dp)),
            **kwargs,
        )

        # Reserved if we need both high-precision and low-precision hidden states.
        hidden_states_high = None
        if self.fusion_config.PRE_MLP_FUSION and use_fp8_allreduce:
            hidden_states, residual = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=self.feed_forward.gate_up_proj.input_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        elif self.fusion_config.PRE_MLP_FUSION and use_fp4_allreduce:
            act_fp4, act_sf, residual = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=self.feed_forward.gate_up_proj.input_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
            hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
        elif self.fusion_config.PRE_MOE_FUSION and use_fp8_allreduce:
            # For pre-MoE all reduce in FP8 mode, we must output two variants of
            # the tensor: one in high-precision (BF16) and one in FP8.
            # This is because routing gemm requires BF16 input while shared
            # expert and routed expert require FP8 input.
            hidden_states_high, hidden_states, residual = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=self.feed_forward.shared_expert.gate_up_proj.
                    input_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        elif self.fusion_config.PRE_MOE_FUSION or self.fusion_config.PRE_MLP_FUSION:
            hidden_states, residual = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        else:
            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # In eagle3 mode, we capture the value in the boundary of decoder layer.
        # If fusing rms in the next layer, the value is not correct. Thus, if
        # this layer will be captured, we should not fuse the rms in the next
        # layer.
        if spec_metadata is not None:
            if spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MOE_FUSION = False
                self.fusion_config.POST_MLP_FUSION = False

        if self.is_mlp_layer:
            hidden_states = self.feed_forward(
                hidden_states,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                    self.fusion_config.POST_MLP_FUSION
                    or self.mapping.tp_size == 1 or self.enable_attention_dp)),
            )
        else:
            hidden_states = self.feed_forward(
                hidden_states,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION
                    or self.mapping.tp_size == 1 or self.enable_attention_dp)),
                hidden_states_high=hidden_states_high,
            )

        if spec_metadata is not None:
            # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
            # PyExecutor will extract these from the model engine's spec metadata.
            # They will be passed to the draft model engine on the first draft iteration.
            # TODO: can we support multiple model outputs instead?
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        needs_post_allreduce = self.fusion_config.POST_MOE_FUSION \
            or self.fusion_config.POST_MLP_FUSION
        if needs_post_allreduce and self.next_layer_layernorm is not None:
            if use_fp8_allreduce and self.next_attn is not None \
                and hasattr(elf.next_attn.qkv_proj, 'input_scale'):
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        scale=self.next_attn.qkv_proj.input_scale,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            elif use_fp4_allreduce and self.next_attn is not None \
                and hasattr(self.next_attn.qkv_proj, 'input_scale'):
                act_fp4, act_sf, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.
                        RESIDUAL_RMS_NORM_QUANT_NVFP4,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        scale=self.next_attn.qkv_proj.input_scale,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            else:
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
        elif self.next_layer_layernorm:
            hidden_states, residual = self.next_layer_layernorm(
                hidden_states, residual)
        elif needs_post_allreduce:
            hidden_states, residual = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.NONE,
                    residual=residual,
                ))

        return hidden_states, residual
