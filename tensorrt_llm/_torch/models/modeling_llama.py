import copy
import os
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
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import HfLoraLoader
from tensorrt_llm.models.convert_utils import split_matrix_tp

from ...inputs import (ExtraProcessedInputs, InputProcessor,
                       MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (Llama4RenormalizeMoeRoutingMethod,
                                 MoEWeightLoadingMode, create_moe)
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import Fp4QuantizedTensor
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, register_auto_model)

DISAGG = os.getenv('TLLM_MULTIMODAL_DISAGGREGATED', '0') == '1'


class Llama4Attention(Attention):

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
        config = model_config.pretrained_config

        self.use_rope = not nope_layer
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=RopeParams.from_config(config),
            is_neox=False,
        ) if self.use_rope else None
        self.use_qk_norm = use_qk_norm

        if model_config.attn_backend != "TRTLLM":
            # TODO: support chunked attention for other backends.
            # This is safe to do because we limit seqlen to 8k for
            # non TRTLLM backends.
            attention_chunk_size = None
        elif get_sm_version() <= 90 and model_config.spec_config is not None:
            # pre-Blackwell spec-dec kernel does not support
            attention_chunk_size = None
        else:
            # Disable chunked attention when max_seq_len is smaller than attention_chunk_size
            # TODO: Remove this after all attention kernels in TRTLLM backend support chunked attention
            if attention_chunk_size and model_config.max_seq_len and model_config.max_seq_len < attention_chunk_size:
                attention_chunk_size = None

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            rope_fusion=not self.
            use_qk_norm,  # Llama4 uses qk_norm after RoPE, so it is not possible to fuse RoPE into the attention OP with qk_norm.
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            attention_chunk_size=attention_chunk_size,
        )

        if self.use_qk_norm:
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

    def apply_rope(self, q: torch.Tensor, k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor], position_ids: torch.Tensor):
        q, k, v = self.split_qkv(q, k, v)
        if position_ids is not None:
            q, k, v = super().apply_rope(q, k, v, position_ids)
        # Llama4 applies QK norm after RoPE.
        if self.use_qk_norm:
            q, k = self.apply_qk_norm(q, k)

        return q, k, v

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
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
        all_reduce_params: Optional[AllReduceParams] = None,
        skip_attn_scaling: bool = False,
    ):

        qkv = self.qkv_proj(hidden_states)

        q, k, v = qkv, None, None
        if self.attn_temperature_tuning and not skip_attn_scaling:
            q, k, v = self.split_qkv(q, k, v)
            q = self._attention_scaling(q, position_ids)

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.forward_impl(q=q,
                                        k=k,
                                        v=v,
                                        attn_metadata=attn_metadata,
                                        attention_mask=attention_mask,
                                        attention_window_size=None,
                                        attention_mask_data=None,
                                        mrope_config=None,
                                        attention_sinks=None)

        if isinstance(attn_output, tuple):
            attn_output = Fp4QuantizedTensor(attn_output[0], attn_output[1])

        attn_output = self.o_proj(attn_output,
                                  all_reduce_params=all_reduce_params)

        return attn_output

    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.
        CAUSAL,
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
                all_reduce_params=all_reduce_params,
                lora_params=lora_params,
                **kwargs,
            )
        else:
            return self._forward_nope(position_ids=position_ids,
                                      hidden_states=hidden_states,
                                      attn_metadata=attn_metadata,
                                      attention_mask=attention_mask,
                                      all_reduce_params=all_reduce_params)


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
        layer_idx: Optional[int] = None,
    ):
        from tensorrt_llm._torch.distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.top_k = top_k

        # Create shared_expert before experts because in min-latency mode the experts depend on the scaling factors of
        # shared_expert.
        self.shared_expert = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=1 if self.enable_attention_dp else None,
            reduce_output=False)

        self.experts = create_moe(
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
            layer_idx=layer_idx)

        self.router = Linear(
            hidden_size,
            num_experts,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=None,
            force_dynamic_quantization=model_config.force_dynamic_quantization)

        self.mapping = model_config.mapping
        self.all_reduce = AllReduce(
            mapping=model_config.mapping,
            strategy=model_config.allreduce_strategy,
        )
        self.moe_event = [torch.cuda.Event(), torch.cuda.Event()]
        self.aux_stream = aux_stream

    def compute_routed_output(self, hidden_states, all_rank_num_tokens,
                              cutlass_min_latency_mode):
        router_logits = self.router(hidden_states)
        routed_output = self.experts(hidden_states,
                                     router_logits,
                                     do_finalize=not cutlass_min_latency_mode,
                                     all_rank_num_tokens=all_rank_num_tokens,
                                     use_dp_padding=False)
        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        cutlass_min_latency_mode: Optional[bool] = False,
    ) -> torch.Tensor:
        # Only enable multi-stream for cuda graph since switch stream has extra host overhead
        # This design is mainly for low latency use case. Need to improve for max throughput use case.
        fn0 = lambda: self.shared_expert(hidden_states)
        fn1 = lambda: self.compute_routed_output(
            hidden_states, all_rank_num_tokens, cutlass_min_latency_mode)
        shared_output, routed_output = maybe_execute_in_parallel(
            fn0, fn1, self.moe_event[0], self.moe_event[1], self.aux_stream)
        if cutlass_min_latency_mode:
            return [shared_output, *routed_output]

        assert shared_output.size() == routed_output.size(
        ), f'unmatched tensor shape'
        final_hidden_states = shared_output + routed_output
        if not self.enable_attention_dp and self.mapping.has_tp():
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

        self.enable_attention_dp = model_config.mapping.enable_attention_dp

        self.fusion_config = EagerFusionConfig()
        # self.fusion_config.PRE_MOE_FUSION = model_config.mapping.has_tp(
        # )

        nope_layer = config.no_rope_layers[layer_idx] == 0
        attention_chunk_size = getattr(config, "attention_chunk_size",
                                       None) if not nope_layer else None

        self.self_attn = Llama4Attention(
            model_config,
            layer_idx=layer_idx,
            use_qk_norm=getattr(config, "use_qk_norm", False),
            nope_layer=nope_layer,
            attn_temperature_tuning=config.attn_temperature_tuning > 0,
            aux_stream=aux_stream,
            attention_chunk_size=attention_chunk_size)

        self.is_mlp_layer = (layer_idx +
                             1) % config.interleave_moe_layer_step != 0

        self.enable_fusion = os.environ.get(
            "TRTLLM_LLAMA_EAGER_FUSION_DISABLED", "0") == "0"

        # MLP layer supports pre and post AR + Res + RMSNorm + NVFP4/FP8
        # MOE layer supports pre AR + Res + RMSNorm
        # MOE layer supports post AR + Res + RMSNorm + QUANT + NVFP4/FP8
        self.pre_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
        self.post_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

        # # Determine the pre and post feed forward fusion op based on the quant mode
        if self.is_nvfp4:
            self.pre_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
            self.post_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
        elif self.is_fp8_quant:
            self.pre_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
            self.post_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8

        if not self.is_mlp_layer:
            self.pre_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

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
            )
            # TODO(TRTLLM-7809): Fix fusion with PP>1
            self.fusion_config.PRE_MLP_FUSION = model_config.mapping.has_tp(
            ) and not self.enable_attention_dp and self.enable_fusion and not model_config.mapping.has_pp(
            )
            self.fusion_config.POST_MLP_FUSION = self.fusion_config.PRE_MLP_FUSION

        else:
            self.feed_forward = Llama4MoE(
                num_experts=config.num_local_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                shared_expert_intermediate_size=config.intermediate_size,
                model_config=model_config,
                aux_stream=aux_stream,
                dtype=config.torch_dtype,
                layer_idx=layer_idx)

            self.fusion_config.PRE_MOE_FUSION = model_config.mapping.has_tp(
            ) and not self.enable_attention_dp and self.enable_fusion and not model_config.mapping.has_pp(
            )
            self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

        self.mapping = model_config.mapping
        self.all_reduce = AllReduce(mapping=model_config.mapping,
                                    strategy=model_config.allreduce_strategy)
        self.next_layer_layernorm: RMSNorm = None
        self.next_attn: LlamaAttention = None

        self.moe_allreduce = MoEAllReduce(self.mapping)

        self.disable_attn_allreduce = (self.fusion_config.PRE_MOE_FUSION
                                       or self.fusion_config.PRE_MLP_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)
        self.disable_feed_forward_allreduce = (
            self.fusion_config.POST_MOE_FUSION
            or self.fusion_config.POST_MLP_FUSION or self.mapping.tp_size == 1
            or self.enable_attention_dp)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: Union[torch.Tensor, Fp4QuantizedTensor],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Only enable min-latency mode on Blackwell
        # TODO: Remove it after we fix crash on Hopper
        # major, minor = torch.cuda.get_device_capability()
        # is_blackwell = (major * 10 + minor) >= 100
        # cutlass_min_latency_mode = hidden_states.size(
        #     0
        # ) <= 128 and self.fusion_config.POST_MOE_FUSION and is_blackwell and self.is_quanted

        # Temporarily disable min-latency mode for Llama4
        cutlass_min_latency_mode = False

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )

        if self.fusion_config.PRE_MLP_FUSION or self.fusion_config.PRE_MOE_FUSION:
            if self.is_mlp_layer and (self.is_nvfp4 or self.is_fp8_quant):
                scale = self.feed_forward.gate_up_proj.input_scale
            else:
                scale = None
            allreduce_output = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=self.pre_feed_forward_fusion_op,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))

            if self.is_mlp_layer and self.is_nvfp4:
                act_fp4, act_sf, residual = allreduce_output
                hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
            else:
                hidden_states, residual = allreduce_output
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # disable fusion for layers captured by spec_metadata
        if spec_metadata is not None:
            if spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MLP_FUSION = False
                self.fusion_config.POST_MOE_FUSION = False
                self.disable_feed_forward_allreduce = self.mapping.tp_size == 1 or self.enable_attention_dp

        hidden_states = self.feed_forward(
            hidden_states,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_feed_forward_allreduce),
            cutlass_min_latency_mode=cutlass_min_latency_mode,
        )

        if spec_metadata is not None:
            # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
            # PyExecutor will extract these from the model engine's spec metadata.
            # They will be passed to the draft model engine on the first draft iteration.
            # TODO: can we support multiple model outputs instead?
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        if (self.fusion_config.POST_MOE_FUSION
                or self.fusion_config.POST_MLP_FUSION):
            # If there is no extra layernorm, do another pure allreduce because
            # the allreduce in feed-forward module has been disabled.
            if self.next_layer_layernorm is None:
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=None,
                        residual=residual,
                    ))
            else:
                # The next layernorm exists but it could be the last decoder layer.
                # Adjust the scale and fusion pattern.
                if not (self.next_attn is not None and (self.is_nvfp4
                                                   or self.is_fp8_quant)) \
                or not hasattr(self.next_attn.qkv_proj, 'input_scale'):
                    scale = None
                    self.post_feed_forward_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
                else:
                    scale = self.next_attn.qkv_proj.input_scale

                # TODO: MIN_LATENCY_MODE is hardcoded to False
                if cutlass_min_latency_mode:
                    shared_output = hidden_states[0]
                    hidden_states_activated_experts = hidden_states[1]
                    num_activated_experts_per_node = hidden_states[2]
                    experts_to_token_score = hidden_states[3]

                    allreduce_output = self.moe_allreduce(
                        residual,
                        self.next_layer_layernorm.weight,
                        device_num_experts=num_activated_experts_per_node,
                        scale_input=experts_to_token_score,
                        active_experts_token_input=
                        hidden_states_activated_experts,
                        token_input=shared_output,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    )
                else:
                    allreduce_output = self.all_reduce(
                        hidden_states,
                        all_reduce_params=AllReduceParams(
                            fusion_op=self.post_feed_forward_fusion_op,
                            residual=residual,
                            norm_weight=self.next_layer_layernorm.weight,
                            scale=scale,
                            eps=self.next_layer_layernorm.variance_epsilon,
                        ))

                # Unpack the allreduce output
                if self.next_attn is not None and self.is_nvfp4:
                    act_fp4, act_sf, residual = allreduce_output
                    hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
                else:
                    hidden_states, residual = allreduce_output
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
        self.mapping = model_config.mapping
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.is_quanted = model_config.quant_config and model_config.quant_config.quant_mode.has_any_quant(
        )
        self.is_fp8_quant = self.is_quanted and model_config.quant_config.quant_mode.has_fp8_qdq(
        )
        self.is_nvfp4 = self.is_quanted and model_config.quant_config.quant_mode.has_nvfp4(
        )

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

        self.all_reduce = AllReduce(mapping=model_config.mapping)

        self.next_layer_layernorm: RMSNorm = None
        self.next_attn: LlamaAttention = None

        self.attention_mask = PredefinedAttentionMask.CAUSAL
        # If the model is being used as an encoder model (prefill only) we use a full attention mask
        if not model_config.is_generation:
            self.attention_mask = PredefinedAttentionMask.FULL

        self.enable_fusion = os.environ.get(
            "TRTLLM_LLAMA_EAGER_FUSION_DISABLED", "0") == "0"
        # Disable fusion for small models due to accuracy issues
        self.enable_fusion &= config.hidden_size > 4096

        self.PRE_MLP_FUSION = self.mapping.has_tp(
        ) and not self.enable_attention_dp and self.enable_fusion
        self.POST_MLP_FUSION = self.mapping.has_tp() and self.enable_fusion

        if self.is_nvfp4:
            self.pre_mlp_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
            self.post_mlp_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
        elif self.is_fp8_quant:
            self.pre_mlp_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
            self.post_mlp_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8
        else:
            self.pre_mlp_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
            self.post_mlp_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM

        self.disable_attn_allreduce = (self.PRE_MLP_FUSION
                                       or self.mapping.tp_size == 1
                                       or self.enable_attention_dp)
        self.disable_mlp_allreduce = (self.POST_MLP_FUSION
                                      or self.mapping.tp_size == 1
                                      or self.enable_attention_dp)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=self.attention_mask,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )
        # Fully Connected
        if self.PRE_MLP_FUSION:
            if self.is_nvfp4 or self.is_fp8_quant:
                scale = self.mlp.gate_up_proj.input_scale
            else:
                scale = None

            all_reduce_output = self.all_reduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=self.pre_mlp_fusion_op,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ))
            if self.is_nvfp4:
                act_fp4, act_sf, residual = all_reduce_output
                hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
            else:
                hidden_states, residual = all_reduce_output
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # disable fusion for layers captured by spec_metadata
        if spec_metadata is not None:
            # how to know if is_layer_capture exists, if not do not call
            if hasattr(spec_metadata,
                       "is_layer_capture") and spec_metadata.is_layer_capture(
                           self.layer_idx):
                self.POST_MLP_FUSION = False
                self.disable_mlp_allreduce = self.mapping.tp_size == 1 or self.enable_attention_dp

        hidden_states = self.mlp(
            hidden_states,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not self.disable_mlp_allreduce),
            **kwargs,
        )

        if spec_metadata is not None:
            # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
            # PyExecutor will extract these from the model engine's spec metadata.
            # They will be passed to the draft model engine on the first draft iteration.
            # TODO: can we support multiple model outputs instead?

            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        if self.POST_MLP_FUSION:
            # If there is no extra layernorm, do another pure allreduce.
            if self.next_layer_layernorm is None:
                hidden_states, residual = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=None,
                        residual=residual,
                    ))
            else:
                # The next layernorm exists but it could be the last decoder layer.
                # Adjust the scale and fusion pattern.

                if not (self.next_attn is not None and (self.is_nvfp4
                                                   or self.is_fp8_quant)) \
                or not hasattr(self.next_attn.qkv_proj, 'input_scale'):
                    scale = None
                    self.post_mlp_fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
                else:
                    scale = self.next_attn.qkv_proj.input_scale

                all_reduce_output = self.all_reduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=self.post_mlp_fusion_op,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        scale=scale,
                        eps=self.next_layer_layernorm.variance_epsilon,
                    ))
                if self.next_attn is not None and self.is_nvfp4:
                    act_fp4, act_sf, residual = all_reduce_output
                    hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
                else:
                    hidden_states, residual = all_reduce_output
        elif self.next_layer_layernorm:
            hidden_states, residual = self.next_layer_layernorm(
                hidden_states, residual)

        return hidden_states, residual


class Llama4Model(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.num_hidden_layers = config.num_hidden_layers
        self.aux_stream = torch.cuda.Stream()
        self.mapping = model_config.mapping
        self.preload_weight_modules = []

        if self.model_config.mapping.enable_attention_dp:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        else:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        # If enable_min_latency is True, we will use min-latency mode.
        DecoderLayerClass = Llama4DecoderLayer
        if model_config.enable_min_latency:
            from .modeling_llama_min_latency import Llama4MinLatencyDecoderLayer
            DecoderLayerClass = Llama4MinLatencyDecoderLayer
            self.preload_weight_modules = ["gate_up_proj"]

        self.layers = nn.ModuleList([
            DecoderLayerClass(
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
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
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

        for idx, decoder_layer in enumerate(
                self.layers[:self.num_hidden_layers]):
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
        self.num_hidden_layers = config.num_hidden_layers

        vocab_size = config.vocab_size
        # TODO smor- we load manually only if there is a single lora dir, need to come up with a better solution
        self.has_custom_embed_tokens = False
        if hasattr(
                model_config,
                'lora_config') and model_config.lora_config is not None and len(
                    model_config.lora_config.lora_dir) == 1:
            # Only check for custom vocab in HF LoRA, not NeMo
            if model_config.lora_config.lora_ckpt_source == "hf":
                lora_loader = HfLoraLoader(model_config.lora_config.lora_dir)
                if lora_loader.vocab_size != 0 and lora_loader.embed_tokens is not None:
                    vocab_size = lora_loader.vocab_size
                    weight = lora_loader.embed_tokens
                    self.has_custom_embed_tokens = True

        if self.model_config.mapping.enable_attention_dp:
            self.embed_tokens = Embedding(
                vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        else:
            self.embed_tokens = Embedding(
                vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        if self.has_custom_embed_tokens:
            with torch.no_grad():
                if model_config.mapping.has_tp():
                    weight = split_matrix_tp(
                        weight,
                        model_config.mapping.tp_size,
                        model_config.mapping.tp_rank,
                        dim=0)  # split by vocabulary dimension
                x = weight.to(self.embed_tokens.dtype)
                self.embed_tokens.weight.data.copy_(x)

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(model_config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params=None,
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


@register_auto_model("LlamaForCausalLM")
class LlamaForCausalLM(SpecDecOneEngineForCausalLM[LlamaModel, LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        super().__init__(LlamaModel(model_config), model_config)

    def post_load_weights(self):
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
                layer.next_attn = self.model.layers[idx + 1].self_attn


class Llama4VisionEncoder(nn.Module):

    def __init__(self, model_config: ModelConfig[Llama4Config], *args,
                 **kwargs):
        super().__init__()
        self.pretrained_config = model_config.pretrained_config
        # TODO: use config.mapping.get_local_rank() instead
        self.device = f"cuda:{torch.cuda.current_device()}"

        self.dtype = self.pretrained_config.text_config.torch_dtype

    def load_weights(self, weights: Dict):
        module_dict = nn.ModuleDict({
            "vision_model":
            Llama4VisionModel(self.pretrained_config.vision_config),
            "multi_modal_projector":
            Llama4MultiModalProjector(self.pretrained_config),
        })

        # If the named params are present in the weights, load them directly.
        param_names = [name for name, _ in module_dict.named_parameters()]
        if all(name in weights for name in param_names):
            vision_encoder_weights = {
                name: weights[name]
                for name in param_names
            }
            module_dict.load_state_dict(vision_encoder_weights)

        # Otherwise, load the weights from the checkpoint.
        else:
            load_sharded_checkpoint(module_dict,
                                    self.pretrained_config._name_or_path,
                                    strict=False)

        self.vision_model = module_dict["vision_model"].to(self.device)
        self.mm_projector = module_dict["multi_modal_projector"].to(self.device)

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):
        pixel_values = [
            multimodal_param.multimodal_data["image"]["pixel_values"]
            for multimodal_param in multimodal_params
        ]
        pixel_values = torch.cat(pixel_values,
                                 dim=0).to(self.device).to(torch.float32)
        image_features = self.vision_model(
            pixel_values).last_hidden_state.flatten(0, 1)
        image_features = self.mm_projector(image_features)
        return [image_features]


class Llama4InputProcessor(InputProcessor):

    def __init__(self,
                 model_path,
                 model_config,
                 tokenizer,
                 trust_remote_code: bool = True):
        self.use_fast = True
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=self.use_fast)
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.vocab_size = model_config.text_config.vocab_size
        self.image_token_index = model_config.image_token_index
        self.fake_image_token = self.processor.fake_image_token
        self.image_token = self.processor.img_patch_token
        self.image_token_start_index = self.model_config.boi_token_index
        self.image_token_end_index = self.model_config.eoi_token_index

    def attach_multimodal_embeddings(
        self, inputs: TextPrompt, multimodal_embedding: Dict[str,
                                                             List[Dict[str,
                                                                       Any]]],
        sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        """
        Attach pre-processed multimodal embeddings into text token stream for Llama4 model.

        This method skips vision processing and works with externally provided embeddings.
        It replaces/expands image placeholders in the text with appropriate tokens and prepares
        the embeddings for model forward pass.

        Args:
            inputs: Text prompt containing image placeholders
            multimodal_embedding: Dictionary containing pre-processed image embedding data with special token information.
                                  Consider adding metadata fields (e.g., model_type, model_name, version) for validation.
        Returns:
            Tuple of (token_ids, extra_processed_inputs) where:
            - token_ids: List of processed token IDs with image placeholders
            - extra_processed_inputs: Optional dictionary containing multimodal embeddings
        """
        text_prompt = inputs.get("prompt")
        if not text_prompt:
            raise ValueError("Text prompt is required but not provided")

        if not isinstance(multimodal_embedding, dict):
            raise ValueError("multimodal_embedding must be a dictionary")

        if 'image' not in multimodal_embedding:
            raise ValueError(
                "Only image modality is supported for external multimodal embedding"
            )

        mm_embedding_info = multimodal_embedding['image']
        if not mm_embedding_info or not isinstance(mm_embedding_info[0], dict):
            raise ValueError(
                "Llama4 image embedding must contain special token information")

        # Extract embedding components
        try:
            mm_embeddings = [
                mm_embedding['mm_embeddings']
                for mm_embedding in mm_embedding_info
            ]
            mm_embedding_special_tokens = [
                mm_embedding['image_special_tokens']
                for mm_embedding in mm_embedding_info
            ]
            mm_embedding_special_offsets = [
                mm_embedding['image_special_token_offsets']
                for mm_embedding in mm_embedding_info
            ]
        except KeyError as e:
            raise ValueError(
                f"Missing required key in multimodal embedding: {e}")

        # Validate embedding dimensions
        model_hidden_size = self.model_config.text_config.hidden_size
        for i, embedding in enumerate(mm_embeddings):
            if embedding.shape[-1] != model_hidden_size:
                raise ValueError(
                    f"Multimodal embedding {i} hidden size {embedding.shape[-1]} "
                    f"must match model hidden size {model_hidden_size}")

        # Count image placeholders (number of images) in the prompt
        total_placeholders = text_prompt.count(self.fake_image_token)
        if total_placeholders == 0:
            raise ValueError(
                "No image placeholders found in the prompt, but multimodal embedding was provided"
            )

        if total_placeholders != len(mm_embeddings):
            raise ValueError(
                f"Number of image placeholders ({total_placeholders}) "
                f"does not match number of embeddings ({len(mm_embeddings)})")

        # Process prompt with image embeddings
        prompt_splits = text_prompt.split(self.fake_image_token)
        new_prompt_parts = []

        for local_image_index, split_part in enumerate(prompt_splits):
            new_prompt_parts.append(split_part)

            if local_image_index < total_placeholders:
                # Calculate total tokens for this image
                num_tokens = len(mm_embeddings[local_image_index]) + len(
                    mm_embedding_special_tokens[local_image_index])

                # Create image token sequence
                image_tokens = [self.image_token] * num_tokens

                # Replace special tokens with actual decoded tokens
                for offset, token_id in zip(
                        mm_embedding_special_offsets[local_image_index],
                        mm_embedding_special_tokens[local_image_index]):
                    if offset < 0 or offset >= len(image_tokens):
                        raise ValueError(
                            f"Image special token offset {offset} is out of range with the total image tokens length {len(image_tokens)}"
                        )
                    if offset < len(image_tokens):
                        image_tokens[offset] = self.tokenizer.decode([token_id])

                # Join tokens without spaces
                image_str = "".join(image_tokens)
                new_prompt_parts.append(image_str)

        # Combine all parts and tokenize
        processed_text = "".join(new_prompt_parts)
        kwargs = {}
        if sampling_params.truncate_prompt_tokens is not None:
            kwargs = dict(truncation=True,
                          max_length=sampling_params.truncate_prompt_tokens)
        text_inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            add_special_tokens=sampling_params.add_special_tokens,
            **kwargs)
        token_ids = text_inputs.input_ids.squeeze()

        # Replace image token indices with out-of-vocabulary tokens
        token_ids[token_ids == self.image_token_index] = self.vocab_size + 1
        # Concatenate all multimodal embeddings
        multimodal_data = {}
        multimodal_data["multimodal_embedding"] = torch.cat(mm_embeddings,
                                                            dim=0)
        return token_ids.tolist(), {"multimodal_data": multimodal_data}

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
            token_ids = processed["input_ids"].squeeze()
            # for fuse_input_embeds
            token_ids[token_ids == self.image_token_index] = self.vocab_size + 1

            multimodal_data = {}
            multimodal_data["image"] = {
                "pixel_values": processed["pixel_values"],
            }
            return token_ids.tolist(), {"multimodal_data": multimodal_data}
        else:
            return processed["input_ids"].squeeze().tolist(), {}


@register_auto_model("Llama4ForConditionalGeneration")
@register_input_processor(
    Llama4InputProcessor,
    model_type="llama4",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<|image|>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ))
class Llama4ForConditionalGeneration(SpecDecOneEngineForCausalLM[Llama4Model,
                                                                 Llama4Config]):

    def __init__(
        self,
        model_config: ModelConfig[Llama4Config],
    ):
        # Keep a reference to the full config (with vision) before switching to text-only
        full_model_config = model_config

        # TODO: figure out a better way to handle multimodality.
        model_config = copy.copy(model_config)
        architectures = model_config.pretrained_config.architectures
        model_config.pretrained_config = model_config.pretrained_config.text_config
        model_config.pretrained_config.architectures = architectures
        super().__init__(Llama4Model(model_config), model_config)
        self.preload_weight_modules = self.model.preload_weight_modules

        if not DISAGG:
            self.mm_encoder = Llama4VisionEncoder(full_model_config)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        if len(multimodal_params) > 0:
            if not DISAGG:
                mm_embeds = self.mm_encoder.forward(multimodal_params)
            else:
                mm_embeds = [
                    multimodal_param.multimodal_data["multimodal_embedding"]
                    for multimodal_param in multimodal_params
                ]

        input_ids, inputs_embeds = fuse_input_embeds(self.model.embed_tokens,
                                                     input_ids, mm_embeds,
                                                     **kwargs)
        return super().forward(attn_metadata,
                               input_ids,
                               position_ids,
                               inputs_embeds,
                               spec_metadata=spec_metadata,
                               return_context_logits=return_context_logits)

    def infer_max_seq_len(self):
        if self.model_config.attn_backend.upper() != 'TRTLLM':
            logger.warning(
                f"Attention backend {self.model_config.attn_backend} "
                "does not support chunked attention. Sequence length "
                "will be limited to 8192.")
            return 8192

        return super().infer_max_seq_len()

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        if not DISAGG:
            self.mm_encoder.load_weights(weights)

        # Temporarily detach mm_encoder so the TRT-LLM loader doesn't try to load it
        had_mm_encoder = hasattr(self, "mm_encoder")
        saved_mm_encoder = getattr(self, "mm_encoder", None)
        if had_mm_encoder:
            delattr(self, "mm_encoder")
        try:
            super().load_weights(weights, weight_mapper)
        finally:
            if had_mm_encoder:
                self.mm_encoder = saved_mm_encoder

    def post_load_weights(self):
        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm
                layer.next_attn = self.model.layers[idx + 1].self_attn


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
