import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL.Image import Image
from torch import nn
from transformers import (AutoProcessor, Llama4Config, Llama4VisionModel,
                          LlamaConfig)
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.models.llama4.modeling_llama4 import Llama4MultiModalProjector

from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, MoEAllReduce)
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.lora_manager import HfLoraLoader
from tensorrt_llm.models.convert_utils import split_matrix_tp

from ...inputs import (ExtraProcessedInputs, InputProcessor, TextPrompt,
                       register_input_processor)
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention, QkNormType
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (FusedMoE, Llama4RenormalizeMoeRoutingMethod,
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
                             EagerFusionConfig, register_auto_model)


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
                         if use_qk_norm else QkNormType.none,
                         is_llama4=True)

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
        num_tokens = hidden_states.fp4_tensor.size(0) if isinstance(
            hidden_states, Fp4QuantizedTensor) else hidden_states.size(0)
        if self.attn_temperature_tuning and self.qkv_proj.use_llama4_qkv and num_tokens <= 8:
            assert position_ids is not None, "attn_temperature_tuning requires position_ids"
            assert self.floor_scale == 8192.0 and self.attn_scale == 0.1, "floor_scale and attn_scale should be 8192.0 and 0.1"
            qkv = self.qkv_proj(hidden_states, position_ids=position_ids)
        else:
            qkv = self.qkv_proj(hidden_states)

        q, k, v = qkv, None, None
        if self.attn_temperature_tuning and (not self.qkv_proj.use_llama4_qkv
                                             or num_tokens > 8):
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
        if self.use_rope:
            return super().forward(
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
            return self._forward_nope(position_ids, hidden_states,
                                      attn_metadata, attention_mask,
                                      mrope_config, all_reduce_params)


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

        self.shared_expert = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            reduce_output=False,
            is_llama4=True)

        self.experts = FusedMoE(
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
            # In Llama4 TP8 EP1 min-latency mode, we fuse the sigmoid score scaling into the FC31 kernel, so
            # we must use the input scale before the score scaling as the input scale for the MoE layer.
            # Therefore, pass a reference to the shared expert to the MoE layer so that it can access the input scale
            # before the score scaling.
            shared_expert=self.shared_expert)

        self.router = Linear(hidden_size,
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
            llama4_tp8ep1_min_latency_mode,
            hidden_states_high: Optional[torch.Tensor] = None):
        if self.enable_attention_dp and self.mapping.tp_size > 1:
            # Use padding here to keep the behavior unchanged
            use_dp_padding = True
            max_num_token_across_dp_ranks = max(all_rank_num_tokens)
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0,
                 max_num_token_across_dp_ranks - hidden_states.shape[0]))
            if hidden_states_high is not None:
                hidden_states_high = torch.nn.functional.pad(
                    hidden_states_high,
                    (0, 0, 0, max_num_token_across_dp_ranks -
                     hidden_states_high.shape[0]))
        # Use high precision hidden states for routing gemm if it is provided.
        hidden_states_routing = hidden_states_high if hidden_states_high is not None else hidden_states
        router_logits = self.router.llama4_router_forward(hidden_states_routing)
        routed_output = self.experts(hidden_states,
                                     router_logits,
                                     cutlass_min_latency_mode,
                                     llama4_tp8ep1_min_latency_mode,
                                     all_rank_num_tokens=all_rank_num_tokens,
                                     use_dp_padding=use_dp_padding)
        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        cutlass_min_latency_mode: Optional[bool] = False,
        llama4_tp8ep1_min_latency_mode: Optional[bool] = False,
        # Optional input for routing gemm if experts and routing gemm require
        # different precisions for input hidden states.
        hidden_states_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # If we do not use llama4_tp8ep1_min_latency_mode, pass the high-precision hidden states into shared/routed
        # experts when the hidden_states are in fp8 because the default path does not work with fp8 input.
        if not llama4_tp8ep1_min_latency_mode and hidden_states.dtype == torch.float8_e4m3fn and hidden_states_high is not None:
            hidden_states_experts = hidden_states_high
        else:
            hidden_states_experts = hidden_states

        # Only enable multi-stream for cuda graph since switch stream has extra host overhead
        # This design is mainly for low latency use case. Need to improve for max throughput use case.
        fn0 = lambda: self.shared_expert(hidden_states)
        fn1 = lambda: self.compute_routed_output(
            hidden_states_experts, all_rank_num_tokens,
            cutlass_min_latency_mode, llama4_tp8ep1_min_latency_mode,
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
            self.feed_forward = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                # Llama4 has no mlp_bias field.
                bias=getattr(config, "mlp_bias", False),
                dtype=config.torch_dtype,
                config=model_config,
                overridden_tp_size=1 if self.enable_attention_dp else None,
                layer_idx=layer_idx,
                is_llama4=True)

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
        # Only enable min-latency mode on Blackwell
        # TODO: Remove it after we fix crash on Hopper
        major, minor = torch.cuda.get_device_capability()
        is_blackwell = (major * 10 + minor) >= 100
        num_tokens = hidden_states.fp4_tensor.size(0) if isinstance(
            hidden_states, Fp4QuantizedTensor) else hidden_states.size(0)
        llama4_tp8ep1_min_latency_mode = is_blackwell and self.is_fp8_quant and self.tp_size == 8 and self.ep_size == 1 and self.num_experts == 128 and self.topk == 1 and num_tokens <= 8 and self.hidden_size == 5120 and self.intermediate_size == 8192
        # cutlass_min_latency_mode = hidden_states.size(
        #     0
        # ) <= 128 and self.fusion_config.POST_MOE_FUSION and is_blackwell and self.is_quanted
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
                llama4_tp8ep1_min_latency_mode=llama4_tp8ep1_min_latency_mode,
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
                llama4_tp8ep1_min_latency_mode=llama4_tp8ep1_min_latency_mode,
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
        pipeline_interface: Optional[PipelineInterface] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self._supports_pp and self.pp_size > 1:
            output = self.model(
                input_ids=input_ids,
                attn_metadata=attn_metadata,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pipeline_interface=pipeline_interface,
                spec_metadata=spec_metadata,
            )

            # No need to compute logits for non-last PP ranks
            if self.pp_rank < self.pp_size - 1:
                return output
            else:
                hidden_states = output
        else:
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
        pipeline_interface: Optional[PipelineInterface] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_eagle3_one_model:
            if self._supports_pp and self.pp_size > 1:
                output = self.model(
                    input_ids=input_ids,
                    attn_metadata=attn_metadata,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    pipeline_interface=pipeline_interface,
                    spec_metadata=spec_metadata,
                )

                # No need to compute logits for non-last PP ranks
                if self.pp_rank < self.pp_size - 1:
                    return output
                else:
                    hidden_states = output
            else:
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
                return_context_logits=return_context_logits,
                pipeline_interface=pipeline_interface)
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
