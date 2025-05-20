import copy
from collections.abc import Callable
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL.Image import Image
from torch import nn
from torch.nn import functional as F
from transformers import (AutoProcessor, Llama4Config, Llama4VisionModel,
                          LlamaConfig)
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector

from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, MoEAllReduce)
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.lora_manager import HfLoraLoader
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import split_matrix_tp
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a)

from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...models.modeling_utils import QuantConfig
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention, QkNormType
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (BaseMoeRoutingMethod, FusedMoE,
                                 FusedMoEQuantScalesFP8,
                                 Llama4RenormalizeMoeRoutingMethod,
                                 MoEWeightLoadingMode)
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata, get_spec_worker
from ..utils import Fp4QuantizedTensor
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, MissingLayer,
                             register_auto_model)

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


# A helper function to get the num_tokens from the input tensor.
def get_num_tokens(x: torch.Tensor) -> int:
    return x.fp4_tensor.shape[0] if isinstance(
        x, Fp4QuantizedTensor) else x.shape[0]


class Llama4Linear(Linear):
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
        post_load_weights_hook: Optional[Callable] = None,
    ):
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
        self.enable_fused_gemm_swiglu = enable_fused_gemm_swiglu
        self.enable_fused_gemm_attn_scaling = enable_fused_gemm_attn_scaling
        self.enable_trtllm_gen = enable_trtllm_gen
        self.post_load_weights_hook = post_load_weights_hook
        self.position_ids = None

    def load_weights(self, weights: List[Dict]):

        super().load_weights(weights)

        # After loading weights, calculate the combined scale (input_scale * weight_scale) for special kernels and
        # trtllm-gen kernels.
        if self.has_fp8_qdq:
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

        if self.post_load_weights_hook is not None:
            self.post_load_weights_hook(self)

    # Override apply_linear instead of forward so that we can reuse the AllReduce/AllGather logic in the parent class.
    def apply_linear(
        self,
        input,
        weight,
        bias,
        lora_params: Optional[dict] | None = None,
        layer_idx: Optional[int] | None = None,
    ) -> torch.Tensor:

        # Quantize the input if it is not already in float8_e4m3fn.
        # We cannot do this when enable_fused_gemm_swiglu is True and num_tokens > 16 because the default path
        # does not support FP8 input.
        if self.has_fp8_qdq \
            and input.dtype != torch.float8_e4m3fn \
            and not (self.enable_fused_gemm_swiglu \
                and get_num_tokens(input) > MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_TRTLLM_GEN):
            input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                input, self.input_scale)

        # Use special BF16-input BF16-output gemm kernel for routing gemm when num_tokens <= 8.
        if get_num_tokens(input) <= MIN_LATENCY_ROUTING_GEMM_NUM_TOKENS \
            and self.in_features == 5120 \
            and self.out_features == 128 \
            and input.dtype == torch.bfloat16 \
            and self.weight.dtype == torch.bfloat16 \
            and not self.has_any_quant:
            return torch.ops.trtllm.llama4_bf16_bf16_gemm(input, self.weight)

        # Use special FP8-input BF16-output gemm kernel for QKV-gemm.
        if get_num_tokens(input) <= MIN_LATENCY_QKV_GEMM_ATTN_SCALING_NUM_TOKENS \
            and self.in_features == 5120 \
            and self.out_features == 896 \
            and input.dtype == torch.float8_e4m3fn:

            # Check if we should fuse attn_scaling into QKV-gemm.
            should_fuse_attn_scaling = self.enable_fused_gemm_attn_scaling \
                and self.position_ids is not None

            # When num_tokens < 4, we use the special QKV-gemm kernel.
            # Or if attn_scaling is needed, we also use the special QKV-gemm kernel.
            if get_num_tokens(
                    input
            ) <= MIN_LATENCY_QKV_GEMM_NUM_TOKENS or should_fuse_attn_scaling:
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
        if get_num_tokens(input) <= MIN_LATENCY_FC2_NUM_TOKENS \
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
        if self.enable_fused_gemm_swiglu and input.dtype == torch.float8_e4m3fn:
            # When num_tokens < 4, we use the special gemm+swiglu kernel.
            if get_num_tokens(
                    input) < MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_GEMV:
                if not hasattr(self, "inv_output_scale"):
                    raise ValueError('Expect inv_output_scale to be set')
                return torch.ops.trtllm.llama4_fp8_fp8_gemm_swiglu(
                    input,
                    self.weight,
                    self.combined_scale,
                    self.inv_output_scale,
                )
            # When 4 <= num_tokens <= 16, we use the trtllm-gen gemm+swiglu kernel.
            elif MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_GEMV <= get_num_tokens(input) \
                and get_num_tokens(input) <= MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_TRTLLM_GEN:
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
            else:
                raise ValueError(
                    f"Gemm+SwiGLU cannot be fused when num_tokens > 16, got num_tokens = {get_num_tokens(input)}"
                )

        # Otherwise, call the default apply_linear method.
        return super().apply_linear(input, weight, bias, lora_params, layer_idx)

    # Set the position_ids for the next call to apply_linear.
    def set_position_ids(self, position_ids: Optional[torch.LongTensor] = None):
        self.position_ids = position_ids


class Llama4GatedMLP(GatedMLP):
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
        if self.hidden_size == 5120 \
            and (self.intermediate_size == 16384 or self.intermediate_size == 8192) \
            and self.mapping.tp_size == 8 \
            and config.quant_config.quant_mode.has_fp8_qdq():
            self.gate_up_proj = Llama4Linear(
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

            # After loading both gate_up_proj and down_proj, we need to set the scales needed by the special kernels and by
            # the trtllm-gen gemm+swiglu kernel.
            def post_load_weights_hook(gate_up_proj, down_proj):
                if gate_up_proj.has_fp8_qdq:
                    # For the special gemm+swiglu kernel, we need to set the inverse of the output scale, which is the inverse
                    # of down_proj's combined input scale.
                    gate_up_proj.inv_output_scale = 1.0 / down_proj.input_scale
                    # For the trtllm-gen gemm+swiglu kernel, we need to set the global scale, which is gate_up_proj's
                    # combined input scale times inv_output_scale.
                    gate_up_proj.trtllm_gen_global_scale = gate_up_proj.combined_scale * gate_up_proj.inv_output_scale

            self.down_proj = Llama4Linear(
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
                post_load_weights_hook=partial(post_load_weights_hook,
                                               self.gate_up_proj),
            )

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        cutlass_min_latency_mode: Optional[bool] = False,
        lora_params: Optional[dict] = None,
    ) -> torch.Tensor:
        # Use the special or trtllm-gen gemm+swiglu kernel for FC13+swiglu when num_tokens <= 16.
        # We cannot use the parent's forward method because it applies swiglu() after calling gate_up_proj.
        if self.gate_up_proj.has_fp8_qdq \
            and x.dtype == torch.float8_e4m3fn \
            and get_num_tokens(x) <= MIN_LATENCY_FC13_FUSED_GEMM_SWIGLU_NUM_TOKENS_TRTLLM_GEN \
            and lora_params is None:
            intermediate = self.gate_up_proj(x)
            return self.down_proj(intermediate,
                                  all_reduce_params=final_all_reduce_params)

        # Otherwise, use the default path.
        return super().forward(
            x,
            all_rank_num_tokens,
            final_all_reduce_params,
            cutlass_min_latency_mode,
            lora_params,
        )


class Llama4Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
        use_qk_norm: bool = False,
        nope_layer: bool = False,
        attn_temperature_tuning: bool = True,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        config = model_config.pretrained_config

        self.use_rope = not nope_layer
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams.from_config(config),
            is_neox=False,
        ) if self.use_rope else None

        super().__init__(hidden_size=config.hidden_size,
                         num_attention_heads=config.num_attention_heads,
                         num_key_value_heads=config.num_key_value_heads,
                         max_position_embeddings=config.max_position_embeddings,
                         bias=config.attention_bias,
                         pos_embd_params=pos_embd_params,
                         layer_idx=layer_idx,
                         dtype=config.torch_dtype,
                         config=model_config,
                         qk_norm_type=QkNormType.post_rope
                         if use_qk_norm else QkNormType.none)

        if self.use_rope and use_qk_norm:
            self.head_dim = config.hidden_size // config.num_attention_heads
            self.qk_norm = RMSNorm(hidden_size=self.head_dim,
                                   eps=1e-6,
                                   dtype=config.torch_dtype,
                                   has_weights=False)
            self.aux_stream = aux_stream
            self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.attn_temperature_tuning = attn_temperature_tuning and nope_layer
        self.floor_scale = getattr(config, "floor_scale", 8192.0)
        self.attn_scale = getattr(config, "attn_scale", 0.1)

        # Enable min-latency QKV gemm when the sizes are as expected.
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
            self.qkv_proj = Llama4Linear(
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

    def apply_qk_norm(self, q, k):

        def q_l2norm():
            return self.qk_norm(q.reshape(-1, self.head_dim)).reshape(
                -1, self.q_size)

        def k_l2norm():
            return self.qk_norm(k.reshape(-1, self.head_dim)).reshape(
                -1, self.kv_size)

        q, k = maybe_execute_in_parallel(
            q_l2norm,
            k_l2norm,
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        return q, k

    def _attention_scaling(self, q, position_ids):

        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def _forward_nope(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
    ):
        qkv = self.qkv_proj(hidden_states)

        q, k, v = qkv, None, None
        if self.attn_temperature_tuning:
            q, k, v = self.split_qkv(q, k, v)
            q = self._attention_scaling(q, position_ids)

        out_scale = None
        if self.o_proj.has_fp8_qdq or self.o_proj.has_nvfp4 or self.o_proj.has_fp8_block_scales:
            out_scale = self.o_proj.inv_input_scale

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.attn.forward(q,
                                        k,
                                        v,
                                        attn_metadata,
                                        out_scale=out_scale,
                                        attention_mask=attention_mask,
                                        mrope_config=mrope_config)

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)

        return attn_output

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        mrope_config: Optional[dict] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert lora_params is None, "LORA is not supported for Llama4Attention"

        num_tokens = hidden_states.fp4_tensor.size(0) if isinstance(
            hidden_states, Fp4QuantizedTensor) else hidden_states.size(0)

        # Set attn_temperature_tuning to False when min-latency QKV gemm is enabled
        # to avoid re-applying attn_scaling.
        # Also, pass the position_ids to QKV gemm because Linear module does not have position_ids as an argument.
        orig_attn_temperature_tuning = self.attn_temperature_tuning
        if self.enable_min_latency_qkv \
            and self.enable_fused_gemm_attn_scaling \
            and position_ids is not None \
            and num_tokens <= MIN_LATENCY_QKV_GEMM_ATTN_SCALING_NUM_TOKENS:
            self.attn_temperature_tuning = False
            self.qkv_proj.set_position_ids(position_ids)

        if self.use_rope:
            outputs = super().forward(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                attention_mask=attention_mask,
                mrope_config=mrope_config,
                all_reduce_params=all_reduce_params,
                lora_params=lora_params,
                **kwargs,
            )
        else:
            outputs = self._forward_nope(position_ids, hidden_states,
                                         attn_metadata, attention_mask,
                                         mrope_config, all_reduce_params)

        # Restore attn_temperature_tuning.
        self.attn_temperature_tuning = orig_attn_temperature_tuning

        return outputs


class LlamaAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
        )


class Llama4FusedMoE(FusedMoE):

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
        post_load_weights_hook: Optional[Callable] = None,
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

        self.post_load_weights_hook = post_load_weights_hook

        # Enable min-latency mode for Llama4 Maverick TP8 EP1.
        self.enable_min_latency_fused_moe = False
        if num_experts == 128 \
            and hidden_size == 5120 \
            and intermediate_size == 8192 \
            and model_config.quant_config.quant_mode.has_fp8_qdq() \
            and model_config.mapping.moe_tp_size == 8 \
            and model_config.mapping.moe_ep_size == 1 \
            and routing_method.top_k == 1 \
            and apply_router_weight_on_input:
            self.enable_min_latency_fused_moe = True

    def load_weights(self, weights: List[Dict]):
        super().load_weights(weights)

        if self.post_load_weights_hook:
            self.post_load_weights_hook(self)

    def forward(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        cutlass_min_latency_mode: bool = False,
        output_dtype: Optional[torch.dtype] = None,
        x_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Use special min-latency MoE kernels when num_tokens <= 8.
        if self.enable_min_latency_fused_moe \
            and x.dtype == torch.float8_e4m3fn \
            and get_num_tokens(x) <= MIN_LATENCY_FUSED_MOE_NUM_TOKENS:
            assert hasattr(self, "min_latency_quant_scales"
                           ), "Expect min_latency_quant_scales to be set"

            return torch.ops.trtllm.llama4_moe_tp8ep1_min_latency(
                x, router_logits, self.w3_w1_weight, self.w2_weight,
                self.min_latency_quant_scales)

        # Default MoE implementation does not support FP8 input, so use high-precision one instead.
        if x_high is not None and x.dtype == torch.float8_e4m3fn:
            x = x_high

        return super().forward(x, router_logits, cutlass_min_latency_mode,
                               output_dtype)


class Llama4MoE(nn.Module):

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
        from tensorrt_llm._torch.distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.top_k = top_k

        self.shared_expert = Llama4GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            reduce_output=False)

        def post_load_weights_hook(shared_expert, experts):
            # Set min-latency quant scales for routed experts if we plan to use min-latency MoE kernels.
            # This is because the routed experts' input scale is after the score multiplication, so we must use the
            # pre-score scaling input scale, which happens to be shared expert's input scale.
            if experts.enable_min_latency_fused_moe and hasattr(
                    shared_expert.gate_up_proj, "input_scale"):
                pre_score_scaling_input_scale = shared_expert.gate_up_proj.input_scale
                experts.min_latency_quant_scales = FusedMoEQuantScalesFP8(
                    fc1_dequant=experts.fc31_dequant.data /
                    experts.fc31_input_dequant.data *
                    pre_score_scaling_input_scale,
                    fc2_quant=experts.fc2_quant,
                    fc2_dequant=experts.fc2_dequant,
                    fc1_input_dequant=pre_score_scaling_input_scale,
                )

        self.experts = Llama4FusedMoE(
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
            post_load_weights_hook=partial(post_load_weights_hook,
                                           self.shared_expert))

        self.router = Llama4Linear(hidden_size,
                                   num_experts,
                                   bias=False,
                                   dtype=config.torch_dtype,
                                   quant_config=None)

        self.mapping = model_config.mapping
        self.all_reduce = AllReduce(self.mapping)
        self.moe_event = [torch.cuda.Event(), torch.cuda.Event()]
        self.aux_stream = aux_stream

    def compute_routed_output(
            self,
            hidden_states,
            all_rank_num_tokens,
            cutlass_min_latency_mode,
            hidden_states_high: Optional[torch.Tensor] = None):
        # Use high precision hidden states for routing gemm if it is provided.
        hidden_states_routing = hidden_states_high if hidden_states_high is not None else hidden_states
        router_logits = self.router(hidden_states_routing)
        routed_output = self.experts(hidden_states, router_logits,
                                     cutlass_min_latency_mode,
                                     hidden_states_high)
        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        cutlass_min_latency_mode: Optional[bool] = False,
        # Optional input for routing gemm if experts and routing gemm require
        # different precisions for input hidden states.
        hidden_states_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Only enable multi-stream for cuda graph since switch stream has extra host overhead
        # This design is mainly for low latency use case. Need to improve for max throughput use case.
        fn0 = lambda: self.shared_expert(hidden_states)
        fn1 = lambda: self.compute_routed_output(
            hidden_states, all_rank_num_tokens, cutlass_min_latency_mode,
            hidden_states_high)
        shared_output, routed_output = maybe_execute_in_parallel(
            fn0, fn1, self.moe_event[0], self.moe_event[1], self.aux_stream)
        if cutlass_min_latency_mode:
            return [shared_output, *routed_output]

        assert shared_output.size() == routed_output.size(
        ), f'unmatched tensor shape'
        final_hidden_states = shared_output + routed_output
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            final_hidden_states = self.all_reduce(
                final_hidden_states, all_reduce_params=final_all_reduce_params)

        return final_hidden_states


class Llama4DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.is_quanted = model_config.quant_config and model_config.quant_config.quant_mode.has_any_quant(
        )
        self.is_fp8_quant = self.is_quanted and model_config.quant_config.quant_mode.has_fp8_qdq(
        )
        self.is_nvfp4 = self.is_quanted and model_config.quant_config.quant_mode.has_nvfp4(
        )
        self.tp_size = model_config.mapping.moe_tp_size
        self.ep_size = model_config.mapping.moe_ep_size
        self.num_experts = model_config.pretrained_config.num_local_experts
        self.topk = model_config.pretrained_config.num_experts_per_tok
        self.hidden_size = model_config.pretrained_config.hidden_size
        self.intermediate_size = model_config.pretrained_config.intermediate_size

        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        self.fusion_config = EagerFusionConfig()

        self.self_attn = Llama4Attention(
            model_config,
            layer_idx=layer_idx,
            use_qk_norm=getattr(config, "use_qk_norm", False),
            nope_layer=config.no_rope_layers[layer_idx] == 0,
            attn_temperature_tuning=config.attn_temperature_tuning > 0,
            aux_stream=aux_stream)

        self.is_mlp_layer = (layer_idx +
                             1) % config.interleave_moe_layer_step != 0

        if self.is_mlp_layer:
            self.feed_forward = Llama4GatedMLP(
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
            self.feed_forward = Llama4MoE(
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

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

        self.mapping = model_config.mapping
        self.all_reduce = AllReduce(self.mapping)
        self.next_layer_layernorm: RMSNorm = None
        self.next_attn: LlamaAttention = None

        self.moe_allreduce = MoEAllReduce(self.mapping)
        self.pre_mlp_quant_allreduce = AllReduce(self.mapping)
        self.post_quant_allreduce = AllReduce(self.mapping)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:

        num_tokens = get_num_tokens(hidden_states)

        # Temporarily disable min-latency mode for Llama4
        cutlass_min_latency_mode = False
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
            hidden_states, residual = self.pre_mlp_quant_allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=self.feed_forward.gate_up_proj.input_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
        elif self.fusion_config.PRE_MLP_FUSION and use_fp4_allreduce:
            act_fp4, act_sf, residual = self.pre_mlp_quant_allreduce(
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
            hidden_states_high, hidden_states, residual = self.pre_mlp_quant_allreduce(
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
                cutlass_min_latency_mode = False

        if self.is_mlp_layer:
            hidden_states = self.feed_forward(
                hidden_states,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION
                    or self.fusion_config.POST_MLP_FUSION
                    or self.mapping.tp_size == 1 or self.enable_attention_dp)),
                cutlass_min_latency_mode=cutlass_min_latency_mode,
            )
        else:
            hidden_states = self.feed_forward(
                hidden_states,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION
                    or self.fusion_config.POST_MLP_FUSION
                    or self.mapping.tp_size == 1 or self.enable_attention_dp)),
                cutlass_min_latency_mode=cutlass_min_latency_mode,
                hidden_states_high=hidden_states_high,
            )

        if spec_metadata is not None:
            # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
            # PyExecutor will extract these from the model engine's spec metadata.
            # They will be passed to the draft model engine on the first draft iteration.
            # TODO: can we support multiple model outputs instead?
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        if cutlass_min_latency_mode:
            shared_output = hidden_states[0]
            hidden_states_activated_experts = hidden_states[1]
            num_activated_experts_per_node = hidden_states[2]
            experts_to_token_score = hidden_states[3]
            hidden_states, residual = self.moe_allreduce(
                residual,
                self.next_layer_layernorm.weight,
                device_num_experts=num_activated_experts_per_node,
                scale_input=experts_to_token_score,
                active_experts_token_input=hidden_states_activated_experts,
                token_input=shared_output,
                eps=self.next_layer_layernorm.variance_epsilon,
            )
        elif (self.fusion_config.POST_MOE_FUSION
              or self.fusion_config.POST_MLP_FUSION
              ) and self.next_layer_layernorm is not None:
            # fp8 POST_MOE_FUSION case might have illegal memory access issue. Disable it temporarily.
            if (self.fusion_config.POST_MLP_FUSION
                    or self.fusion_config.POST_MOE_FUSION
                ) and use_fp8_allreduce and self.next_attn is not None:
                hidden_states, residual = self.post_quant_allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        scale=self.next_attn.qkv_proj.input_scale,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
            elif use_fp4_allreduce and self.next_attn is not None:
                act_fp4, act_sf, residual = self.post_quant_allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
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

        return hidden_states, residual


class LlamaDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = LlamaAttention(
            model_config,
            layer_idx=layer_idx,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states, **kwargs)
        if spec_metadata is not None:
            # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
            # PyExecutor will extract these from the model engine's spec metadata.
            # They will be passed to the draft model engine on the first draft iteration.
            # TODO: can we support multiple model outputs instead?
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)
        return hidden_states, residual


class Eagle3LlamaAttention(LlamaAttention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
    ):
        super().__init__(model_config, layer_idx)

        model_config = model_config or ModelConfig()
        config = model_config.pretrained_config

        tp_size = model_config.mapping.tp_size

        # Override the QKV projection. The number of input features
        # is twice as big for EAGLE3 draft models.
        self.qkv_proj = Linear(
            2 * self.hidden_size,
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
        )


class Eagle3LlamaDecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = Eagle3LlamaAttention(
            model_config,
            layer_idx=layer_idx,
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=inter_size,
            bias=getattr(config, "mlp_bias", False),
            dtype=config.torch_dtype,
            config=model_config,
        )
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.hidden_norm = RMSNorm(hidden_size=config.hidden_size,
                                   eps=config.rms_norm_eps,
                                   dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

    def forward(
        self,
        position_ids: torch.LongTensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
    ) -> torch.Tensor:
        residual = hidden_states

        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        # assert isinstance(spec_metadata, Eagle3SpecMetadata)
        # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
        # PyExecutor will extract these from the draft model engine's spec metadata.
        # They will be passed to the draft model engine on the next iteration.
        # TODO: can we support multiple model outputs instead?
        spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states,
                                                  residual)

        return hidden_states, residual


class Llama4Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.num_hidden_layers = config.num_hidden_layers
        self.aux_stream = torch.cuda.Stream()

        if self.model_config.mapping.enable_attention_dp:
            self.embed_tokens = Embedding(config.vocab_size,
                                          config.hidden_size,
                                          dtype=config.torch_dtype)
        else:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        self.layers = nn.ModuleList([
            Llama4DecoderLayer(
                model_config,
                layer_idx,
                self.aux_stream,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

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
                lora_params=lora_params,
            )

        return hidden_states


class LlamaModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.num_hidden_layers = config.num_hidden_layers

        vocab_size = config.vocab_size
        # TODO smor- we load manually only if there is a single lora dir, need to come up with a better solution
        self.has_custom_embed_tokens = False
        if hasattr(
                model_config,
                'lora_config') and model_config.lora_config is not None and len(
                    model_config.lora_config.lora_dir) == 1:
            lora_loader = HfLoraLoader(model_config.lora_config.lora_dir)
            if lora_loader.vocab_size != 0 and lora_loader.embed_tokens is not None:
                vocab_size = lora_loader.vocab_size
                weight = lora_loader.embed_tokens
                self.has_custom_embed_tokens = True

        self.embed_tokens = Embedding(
            vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            parallel_config=None
            if model_config.mapping.enable_attention_dp else ParallelConfig(
                tensor_parallel_rank=model_config.mapping.tp_rank,
                tensor_parallel_size=model_config.mapping.tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                pipeline_parallel_size=model_config.mapping.pp_size,
                parallel_rank=model_config.mapping.rank,
                gather_output=True,
                gpus_per_node=model_config.mapping.gpus_per_node,
            ),
        )

        if self.has_custom_embed_tokens:
            with torch.no_grad():
                if model_config.mapping.tp_size > 1:
                    weight = split_matrix_tp(
                        weight,
                        model_config.mapping.tp_size,
                        model_config.mapping.tp_rank,
                        dim=0)  # split by vocabulary dimension
                x = weight.to(self.embed_tokens.dtype)
                self.embed_tokens.weight.data.copy_(x)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

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
                lora_params=lora_params,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


@register_auto_model("LlamaForCausalLM")
class LlamaForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        super().__init__(LlamaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
        self.draft_model = None
        if model_config.spec_config is not None and model_config.spec_config.spec_dec_mode.is_eagle3_one_model(
        ):
            draft_config = ModelConfig.from_pretrained(
                model_config.spec_config.draft_model_path,
                trust_remote_code=True,
                attn_backend=model_config.attn_backend,
                moe_backend=model_config.moe_backend,
                mapping=model_config.mapping)
            draft_config.spec_config = model_config.spec_config
            draft_config.max_num_tokens = model_config.max_num_tokens
            draft_config.moe_max_num_tokens = model_config.moe_max_num_tokens
            draft_config.quant_config.kv_cache_quant_algo = \
                model_config.quant_config.kv_cache_quant_algo
            self.draft_model = Eagle3LlamaForCausalLM(
                draft_config, model_config.pretrained_config.num_hidden_layers)
            self.spec_worker = get_spec_worker(model_config.spec_config,
                                               model_config.mapping)

            # Set to True to enable delayed all-gather for LM head.
            # This means LM head will not perform all-gather on the output logits, so each rank will only have
            # a subset of the logits. The decoding part must be aware of that and apply all gather after top1 reduction.
            self.enable_lm_head_delayed_all_gather = True
            if self.enable_lm_head_delayed_all_gather:
                # Delay all-gather until after top1 reduction.
                self.lm_head.gather_output = False
                self.draft_model.lm_head.gather_output = False

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
        )

        if self.draft_model is not None:
            # get logits
            logits = self.logits_processor.forward(
                hidden_states[spec_metadata.gather_ids],
                self.lm_head,
                attn_metadata,
                True,
            )
            # get accepted tokens and next draft tokens
            return self.spec_worker(input_ids=input_ids,
                                    position_ids=position_ids,
                                    hidden_states=hidden_states,
                                    logits=logits,
                                    attn_metadata=attn_metadata,
                                    spec_metadata=spec_metadata,
                                    draft_model=self.draft_model,
                                    main_model_lm_head=self.lm_head)
        else:
            logits = self.logits_processor.forward(
                hidden_states,
                self.lm_head,
                attn_metadata,
                return_context_logits,
            )

        return logits

    def load_weights(self, weights: Dict):
        super().load_weights(weights, skip_modules=["draft_model"])

    def load_draft_weights(self, weights: Dict):
        self.draft_model.load_weights(weights)
        self.draft_model.load_weights_from_target_model(self)


class Llama4InputProcessor(InputProcessor):

    def __init__(self, model_path, model_config, tokenizer):
        self.processor = AutoProcessor.from_pretrained(model_path,
                                                       use_fast=True)
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.vocab_size = model_config.text_config.vocab_size
        self.image_token_index = model_config.image_token_index

        self.encoder = nn.ModuleDict({
            "vision_model":
            Llama4VisionModel(model_config.vision_config),
            "multi_modal_projector":
            Llama4MultiModalProjector(model_config)
        }).cuda()
        load_sharded_checkpoint(self.encoder, model_path, strict=False)

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get(
            "multi_modal_data")
        images, do_rescale = None, True

        if mm_data and mm_data.get("image"):
            images = mm_data["image"]
            img_type = type(mm_data["image"][0])
            do_rescale = (img_type == Image)
            assert all(isinstance(img, img_type) for img in mm_data["image"])

        truncate_kwargs = {}
        if sampling_params.truncate_prompt_tokens is not None:
            truncate_kwargs[
                "max_length"] = sampling_params.truncate_prompt_tokens
            truncate_kwargs["truncation"] = True

        # preprocess images and insert image tokens
        processed = self.processor(
            text=text_prompt,
            images=images,
            return_tensors="pt",
            device="cuda",
            do_rescale=do_rescale,
            add_special_tokens=sampling_params.add_special_tokens,
            **truncate_kwargs)
        if images:
            token_ids, pixel_values = processed["input_ids"].squeeze(
            ), processed["pixel_values"]
            mm_embeds = self.encoder.vision_model(
                pixel_values.float().cuda()).last_hidden_state.flatten(0, 1)
            mm_embeds = self.encoder.multi_modal_projector(mm_embeds)
            # for fuse_input_embeds
            token_ids[token_ids == self.image_token_index] = self.vocab_size + 1
            return token_ids.tolist(), {"mm_embedding": mm_embeds}
        else:
            return processed["input_ids"].squeeze().tolist(), {}


@register_auto_model("Llama4ForConditionalGeneration")
@register_input_processor(Llama4InputProcessor)
class Llama4ForConditionalGeneration(DecoderModelForCausalLM[Llama4Model,
                                                             Llama4Config]):

    def __init__(
        self,
        model_config: ModelConfig[Llama4Config],
    ):
        # TODO: figure out a better way to handle multimodality.
        model_config = copy.copy(model_config)
        architectures = model_config.pretrained_config.architectures
        model_config.pretrained_config = model_config.pretrained_config.text_config
        model_config.pretrained_config.architectures = architectures
        super().__init__(Llama4Model(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

        self.is_eagle3_one_model = model_config.spec_config is not None and model_config.spec_config.spec_dec_mode.is_eagle3_one_model(
        )
        self.draft_model = None
        if self.is_eagle3_one_model:
            draft_config = ModelConfig.from_pretrained(
                model_config.spec_config.draft_model_path,
                trust_remote_code=True,
                attn_backend=model_config.attn_backend,
                moe_backend=model_config.moe_backend,
                mapping=model_config.mapping)
            draft_config.spec_config = model_config.spec_config
            draft_config.max_num_tokens = model_config.max_num_tokens
            draft_config.moe_max_num_tokens = model_config.moe_max_num_tokens
            draft_config.quant_config.kv_cache_quant_algo = \
                model_config.quant_config.kv_cache_quant_algo
            self.draft_model = Eagle3LlamaForCausalLM(
                draft_config, model_config.pretrained_config.num_hidden_layers)
            self.spec_worker = get_spec_worker(model_config.spec_config,
                                               model_config.mapping)

            # Set to True to enable delayed all-gather for LM head.
            # This means LM head will not perform all-gather on the output logits, so each rank will only have
            # a subset of the logits. The decoding part must be aware of that and apply all gather after top1 reduction.
            self.enable_lm_head_delayed_all_gather = True
            if self.enable_lm_head_delayed_all_gather:
                # Delay all-gather until after top1 reduction.
                self.lm_head.gather_output = False
                self.draft_model.lm_head.gather_output = False

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_eagle3_one_model:
            hidden_states = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                spec_metadata=spec_metadata,
            )

            if self.draft_model is not None:
                # get logits
                logits = self.logits_processor.forward(
                    hidden_states[spec_metadata.gather_ids],
                    self.lm_head,
                    attn_metadata,
                    True,
                )
                # get accepted tokens and next draft tokens
                return self.spec_worker(input_ids=input_ids,
                                        position_ids=position_ids,
                                        hidden_states=hidden_states,
                                        logits=logits,
                                        attn_metadata=attn_metadata,
                                        spec_metadata=spec_metadata,
                                        draft_model=self.draft_model,
                                        main_model_lm_head=self.lm_head)
            else:
                logits = self.logits_processor.forward(
                    hidden_states,
                    self.lm_head,
                    attn_metadata,
                    return_context_logits,
                )

            return logits
        else:
            mm_embed = kwargs.get("multi_modal_data", [])
            input_ids, inputs_embeds = fuse_input_embeds(
                self.model.embed_tokens, input_ids, mm_embed)
            logits = super().forward(
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                spec_metadata=spec_metadata,
                return_context_logits=return_context_logits)
            return logits

    def infer_max_seq_len(self):
        # TODO: implement chunked attention to support 10M context length
        return 8192

    def load_weights(self, weights: Dict):
        new_weights = {}
        for key, tensor in weights.items():
            if key.startswith("language_model."):
                new_key = key[len("language_model."):]
                new_weights[new_key] = tensor
            else:
                new_weights[key] = tensor

        super().load_weights(new_weights, skip_modules=["draft_model"])

        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            elif not isinstance(self.model.layers[idx + 1], MissingLayer):
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
                layer.next_attn = self.model.layers[idx + 1].self_attn

    def load_draft_weights(self, weights: Dict):
        self.draft_model.load_weights(weights)
        self.draft_model.load_weights_from_target_model(self)


@register_auto_model("MistralForCausalLM")
class MistralForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        # to support MistralConfig
        if not hasattr(model_config.pretrained_config, 'attention_bias'):
            model_config.pretrained_config.attention_bias = False
        if not hasattr(model_config.pretrained_config, 'rope_scaling'):
            model_config.pretrained_config.rope_scaling = None
        if not hasattr(model_config.pretrained_config, 'mlp_bias'):
            model_config.pretrained_config.mlp_bias = False

        super().__init__(LlamaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)


class Eagle3LlamaDraftModel(DecoderModel):

    def __init__(self,
                 model_config: ModelConfig[LlamaConfig],
                 start_layer_idx: int = 0) -> None:
        super().__init__(model_config)

        config = model_config.pretrained_config
        self.spec_config = model_config.spec_config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.mapping = model_config.mapping

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = Linear(self.hidden_size_in * 3,
                         config.hidden_size,
                         bias=getattr(config, "bias", False),
                         dtype=config.torch_dtype)

        self.midlayer = Eagle3LlamaDecoderLayer(model_config, start_layer_idx)

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

        if config.vocab_size != config.draft_vocab_size:
            self.d2t = nn.Parameter(torch.empty((config.draft_vocab_size, ),
                                                dtype=torch.int64),
                                    requires_grad=False)

        if self.hidden_size_in != config.hidden_size:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )
        else:
            # Shared with target model.
            self.embed_tokens = None

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.embed_tokens is not None

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids).to(self.dtype)

        assert hidden_states is not None

        # NOTE: If hidden states from the target model have to be concatenated,
        # we expect that to happen outside the model definition. This helps us
        # avoid data-dependent control flow and gives us better CUDA graph
        # coverage.
        hidden_states, residual = self.midlayer(position_ids=position_ids,
                                                embeds=inputs_embeds,
                                                hidden_states=hidden_states,
                                                attn_metadata=attn_metadata,
                                                spec_metadata=spec_metadata)

        hidden_states, hidden_states_to_save = self.norm(
            hidden_states, residual)
        if self.spec_config.spec_dec_mode.is_eagle3():
            spec_metadata.maybe_capture_hidden_states(1, hidden_states_to_save)
        return hidden_states, hidden_states_to_save


@register_auto_model("EAGLE3LlamaForCausalLM")
class Eagle3LlamaForCausalLM(DecoderModelForCausalLM[Eagle3LlamaDraftModel,
                                                     LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        start_layer_idx: int = 0,
    ):
        super().__init__(
            Eagle3LlamaDraftModel(model_config, start_layer_idx),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.draft_vocab_size)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        output, _ = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            hidden_states=hidden_states,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict):
        new_weights = {}
        for k, v in weights.items():
            new_k = "model." + k if 'lm_head' not in k else k
            new_weights[new_k] = v

        super().load_weights(new_weights)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens

    # TODO: should input/position IDs be included in this? Keeping it implicit
    # for now since the shapes/dtypes are the same across all models we have.
    def get_warmup_extra_inputs(self, batch_size: int,
                                num_tokens: int) -> Dict[str, Any]:

        hidden_states = torch.empty(batch_size * num_tokens,
                                    self.model.hidden_size_in,
                                    dtype=self.model.dtype,
                                    device='cuda')

        return {'hidden_states': hidden_states}

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Hack for eagle3. We might need to run a matmul to reduce
        the dimensionality of the hidden states on the first pass
        through the draft model. Shape dependent control flow will
        not work with CUDA graphs. So we have hoisted this logic out
        of the forward pass - the pyexecutor will call this function
        before running forward when applicable.
        """
        hidden_states = hidden_states.to(self.model.dtype)

        expected_hidden_size = self.model.hidden_size
        if hidden_states.shape[-1] != expected_hidden_size:
            hidden_states = self.model.fc(hidden_states)

        return hidden_states
