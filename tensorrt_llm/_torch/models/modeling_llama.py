import copy
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from torch import nn
from transformers import Llama4Config, LlamaConfig, AutoModel, AutoProcessor, Llama4VisionModel, Llama4MultiModalProjector
from safetensors import safe_open
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, DeepseekAllReduce,
                                             ParallelConfig, TensorParallelMode)
from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import (FusedMoE, Llama4RenormalizeMoeRoutingMethod,
                                 MoEWeightLoadingMode)
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, WeightMode, WeightsLoadingConfig
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from ..speculative import Eagle3SpecMetadata, SpecMetadata
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             EagerFusionConfig, register_auto_model, support_pp)
from .modeling_multimodal_utils import fuse_input_embeds

from ...inputs import (
    ExtraProcessedInputs,
    InputProcessor,
    TextPrompt,
    register_input_processor,
)
from ...sampling_params import SamplingParams

from PIL.Image import Image


class LlamaRotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        config: LlamaConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            config,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            device=device,
            rope_type="default" if config.rope_scaling is None else "llama3")


class LlamaAttention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
        use_qk_norm: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
        use_gptj_style_rope: bool = False,
    ):
        config = model_config.pretrained_config

        # TODO: do we need to use NoPE in any versions of LLaMA4?
        nope_layer_interval = None
        if nope_layer_interval:
            is_nope_layer = (layer_idx + 1) % nope_layer_interval == 0
        else:
            is_nope_layer = False

        use_rope = not is_nope_layer
        if use_rope and model_config.fuse_pos_embd:
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj
                if use_gptj_style_rope else PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )
        else:
            pos_embd_params = None

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            rotary_emb=LlamaRotaryEmbedding(config),
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            use_qk_norm=use_qk_norm and not is_nope_layer,
            aux_stream=aux_stream,
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
        self.top_k = top_k
        self.experts = FusedMoE(
            routing_method=Llama4RenormalizeMoeRoutingMethod(top_k),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=
            False,  # In both low latency and max-throughput scenarios, FusedMoE needs not to do allreduce inside op.
            weight_loading_mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
            model_config=model_config)

        self.shared_expert = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            is_expert=True)

        self.router = Linear(hidden_size,
                             num_experts,
                             bias=False,
                             dtype=config.torch_dtype,
                             quant_config=None)

        self.parallel_config = ParallelConfig(
            tensor_parallel_rank=model_config.mapping.tp_rank,
            tensor_parallel_size=model_config.mapping.tp_size,
            gpus_per_node=model_config.mapping.gpus_per_node)
        self.all_reduce = AllReduce(self.parallel_config)
        self.moe_event = [torch.cuda.Event(), torch.cuda.Event()]
        self.aux_stream = aux_stream

    def compute_routed_output(self, hidden_states, all_rank_num_tokens,
                              min_latency_mode):
        router_logits = self.router(hidden_states)
        routed_output = self.experts(hidden_states, router_logits,
                                     min_latency_mode)
        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        min_latency_mode: Optional[bool] = False,
    ) -> torch.Tensor:
        # Only enable multi-stream for cuda graph since switch stream has extra host overhead
        # This design is mainly for low latency use case. Need to improve for max throughput use case.
        do_multi_stream = torch.cuda.is_current_stream_capturing()
        if do_multi_stream:
            self.moe_event[0].record()
        shared_output = self.shared_expert(hidden_states)
        if do_multi_stream:
            with torch.cuda.stream(self.aux_stream):
                self.moe_event[0].wait()
                routed_output = self.compute_routed_output(
                    hidden_states, all_rank_num_tokens, min_latency_mode)
                self.moe_event[1].record()
            self.moe_event[1].wait()
        else:
            routed_output = self.compute_routed_output(hidden_states,
                                                       all_rank_num_tokens,
                                                       min_latency_mode)
        if min_latency_mode:
            return [shared_output, *routed_output]

        assert shared_output.size() == routed_output.size(
        ), f'unmatched tensor shape'
        final_hidden_states = shared_output + routed_output
        if self.parallel_config.tensor_parallel_size > 1:
            final_hidden_states = self.all_reduce(
                final_hidden_states, all_reduce_params=final_all_reduce_params)

        return final_hidden_states


class LlamaDecoderLayer(DecoderLayer):

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
        self.fusion_config = EagerFusionConfig()
        # self.fusion_config.PRE_MOE_FUSION = model_config.mapping.has_tp(
        # )
        # TODO: re-enable these fusions
        self.fusion_config.PRE_MOE_FUSION = False
        self.fusion_config.POST_MLP_FUSION = False

        self.is_llama4 = config.model_type == "llama4_text"
        self.self_attn = LlamaAttention(
            model_config,
            layer_idx=layer_idx,
            use_qk_norm=getattr(config, "use_qk_norm", False),
            aux_stream=aux_stream,
            use_gptj_style_rope=self.is_llama4,
        )

        if self.is_llama4:
            is_mlp_layer = (layer_idx +
                            1) % config.interleave_moe_layer_step != 0
        else:
            # llama3 doesn't have config.interleave_moe_layer_step in its config.
            is_mlp_layer = True

        if is_mlp_layer:
            if self.is_llama4:
                inter_size = config.intermediate_size_mlp
            else:
                inter_size = config.intermediate_size

            mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=inter_size,
                # Llama4 has no mlp_bias field.
                bias=getattr(config, "mlp_bias", False),
                dtype=config.torch_dtype,
                config=model_config,
            )

            if self.is_llama4:
                self.feed_forward = mlp
            else:
                self.mlp = mlp

            # self.fusion_config.POST_MLP_FUSION = model_config.mapping.has_tp(
            # )
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

            # self.fusion_config.POST_MOE_FUSION = model_config.mapping.has_tp(
            # )

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)

        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)

        self.parallel_config = ParallelConfig(
            tensor_parallel_rank=model_config.mapping.tp_rank,
            tensor_parallel_size=model_config.mapping.tp_size,
            gpus_per_node=model_config.mapping.gpus_per_node)
        self.all_reduce = AllReduce(self.parallel_config)
        self.next_layer_layernorm: RMSNorm = None

        self.deepseek_allreduce = DeepseekAllReduce(self.parallel_config)

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:

        # Only enable min-latency mode on Blackwell
        # TODO: Remove it after we fix crash on Hopper
        major, minor = torch.cuda.get_device_capability()
        is_blackwell = (major * 10 + minor) >= 100
        min_latency_mode = hidden_states.size(
            0
        ) <= 128 and self.fusion_config.POST_MOE_FUSION and is_blackwell and self.is_quanted

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.PRE_MOE_FUSION
                or self.parallel_config.tensor_parallel_size == 1)),
            **kwargs,
        )

        if self.fusion_config.PRE_MOE_FUSION:
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

        feed_forward = self.feed_forward if self.is_llama4 else self.mlp
        hidden_states = feed_forward(
            hidden_states,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(enable_allreduce=not (
                self.fusion_config.POST_MOE_FUSION
                or self.fusion_config.POST_MLP_FUSION
                or self.parallel_config.tensor_parallel_size == 1)),
            min_latency_mode=min_latency_mode,
        )
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        if self.fusion_config.POST_MOE_FUSION or self.fusion_config.POST_MLP_FUSION:
            if min_latency_mode:
                shared_output = hidden_states[0]
                hidden_states_activated_experts = hidden_states[1]
                num_activated_experts_per_node = hidden_states[2]
                experts_to_token_score = hidden_states[3]
                activated_expert_global_ids = hidden_states[4]
                hidden_states, residual = self.deepseek_allreduce(
                    hidden_states_activated_experts,  # not used
                    [
                        residual, self.next_layer_layernorm.weight,
                        num_activated_experts_per_node, experts_to_token_score,
                        hidden_states_activated_experts, shared_output,
                        activated_expert_global_ids
                    ],
                    self.next_layer_layernorm.variance_epsilon,
                    AllReduceFusionOp.MOE_ALLREDUCE_RESIDUAL_RMS_NORM,
                )
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
        tp_rank = model_config.mapping.tp_rank
        gpus_per_node = model_config.mapping.gpus_per_node

        # Override the QKV projection. The number of input features
        # is twice as big for EAGLE3 draft models.
        self.qkv_proj = Linear(
            2 * self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=config.attention_bias,
            dtype=config.torch_dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_size=tp_size,
                tensor_parallel_rank=tp_rank,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gpus_per_node=gpus_per_node,
                pipeline_parallel_size=model_config.mapping.pp_size,
                parallel_rank=model_config.mapping.rank),
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=model_config.get_quant_config(),
            skip_create_weights=model_config.skip_create_weights,
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

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
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
        return hidden_states, residual


@support_pp
class LlamaModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.padding_idx = config.pad_token_id
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            parallel_config=ParallelConfig(
                tensor_parallel_rank=model_config.mapping.tp_rank,
                tensor_parallel_size=model_config.mapping.tp_size,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                pipeline_parallel_size=model_config.mapping.pp_size,
                parallel_rank=model_config.mapping.rank,
                gather_output=True,
                gpus_per_node=model_config.mapping.gpus_per_node,
            ),
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
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
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        residual = hidden_states
        hidden_states = self.layers[0].input_layernorm(hidden_states)
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(position_ids=position_ids,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    residual=residual,
                                                    spec_metadata=spec_metadata)

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

    def load_weights(self, weights: Dict):
        super().load_weights(weights)

        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm


@register_auto_model("Llama4ForCausalLM")
class Llama4ForCausalLM(DecoderModelForCausalLM[LlamaModel, Llama4Config]):
    def __init__(
        self,
        model_config: ModelConfig[Llama4Config],
    ):
        model_config = copy.copy(model_config)
        model_config.pretrained_config = model_config.pretrained_config.text_config
        super().__init__(LlamaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict):
        new_weights = {}
        for key, tensor in weights.items():
            if key.startswith("language_model."):
                new_key = key[len("language_model."):]
                new_weights[new_key] = tensor

        super().load_weights(new_weights)

        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm


class Llama4InputProcessor(InputProcessor):
    def __init__(self, model_path, model_config, tokenizer):
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.vocab_size = model_config.text_config.vocab_size
        self.image_token_index = model_config.image_token_index

        self.vision_model = Llama4VisionModel.from_pretrained(model_path).cuda()
        self.multi_modal_projector = Llama4MultiModalProjector(model_config).cuda()
        state_dict = self.multi_modal_projector.state_dict()
        
        with safe_open(Path(model_path) / "model.safetensors", framework="pt") as f:
            for k in state_dict.keys():
                state_dict[k] = f.get_tensor("multi_modal_projector." + k)

        self.multi_modal_projector.load_state_dict(state_dict)

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data = inputs.get("prompt"), inputs.get("multi_modal_data")
        if not mm_data or not mm_data.get("image"):
            # text only
            return token_ids, {}
        
        # preprocess images and insert image tokens
        img_type = type(mm_data["image"][0])
        assert all(isinstance(img, img_type) for img in mm_data["image"])

        processed = self.processor(
            text=text_prompt, images=mm_data["image"],
            return_tensors="pt", device="cuda",
            do_rescale=(img_type == Image)
        )
        token_ids, pixel_values = processed["input_ids"].squeeze(), processed["pixel_values"]
        mm_embeds = self.vision_model(pixel_values.float().cuda()).last_hidden_state.flatten(0, 1)
        mm_embeds = self.multi_modal_projector(mm_embeds)
        # for fuse_input_embeds
        token_ids[token_ids == self.image_token_index] = self.vocab_size + 1
        return token_ids.tolist(), {"prompt_tuning_config": [mm_embeds, None, None]}



@register_auto_model("Llama4ForConditionalGeneration")
@register_input_processor(Llama4InputProcessor)
class Llama4ForConditionalGeneration(Llama4ForCausalLM):
    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        mm_embed = kwargs.get("multi_modal_data", [])
        input_ids, inputs_embeds = fuse_input_embeds(self.model.embed_tokens, input_ids, mm_embed)
        logits = super().forward(
            attn_metadata, input_ids, position_ids, inputs_embeds, return_context_logits
        )
        return logits


@register_auto_model("MistralForCausalLM")
class MistralForCausalLM(DecoderModelForCausalLM[LlamaModel, LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        super().__init__(LlamaModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict):
        super().load_weights(weights)

        for idx, layer in enumerate(
                self.model.layers[:self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[
                    idx + 1].input_layernorm


class Eagle3LlamaDraftModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[LlamaConfig]) -> None:
        super().__init__(model_config)

        config = model_config.pretrained_config
        self.dtype = config.torch_dtype

        self.fc = Linear(config.hidden_size * 3,
                         config.hidden_size,
                         bias=False,
                         dtype=config.torch_dtype)

        self.midlayer = Eagle3LlamaDecoderLayer(model_config, 0)

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

        self.d2t = nn.Parameter(torch.empty((config.draft_vocab_size, ),
                                            dtype=torch.int64),
                                requires_grad=False)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        embed_tokens: torch.nn.Module,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = embed_tokens(input_ids).to(self.dtype)

        assert hidden_states is not None and len(hidden_states) > 0

        if len(hidden_states) > 1:
            hidden_states = torch.cat(hidden_states, dim=-1)
            hidden_states = self.fc(hidden_states.to(self.dtype))
        else:
            hidden_states = hidden_states[0].to(self.dtype)

        hidden_states, residual = self.midlayer(position_ids=position_ids,
                                                embeds=inputs_embeds,
                                                hidden_states=hidden_states,
                                                attn_metadata=attn_metadata)

        hidden_states, hidden_states_to_save = self.norm(
            hidden_states, residual)
        assert isinstance(spec_metadata, Eagle3SpecMetadata)
        spec_metadata.hidden_states.append(hidden_states_to_save)
        return hidden_states


@register_auto_model("EAGLE3LlamaForCausalLM")
class Eagle3LlamaForCausalLM(DecoderModelForCausalLM[Eagle3LlamaDraftModel,
                                                     LlamaConfig]):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
    ):
        super().__init__(
            Eagle3LlamaDraftModel(model_config),
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
        if "embed_tokens" not in kwargs:
            raise ValueError(
                "EAGLE3 checkpoints do not include embed_tokens. "
                "The embed_tokens module from the target model therefore needs to "
                "be passed explicitly via extra_model_inputs. NOTE: we can "
                "get rid of this hack by providing our own custom checkpoint "
                "format that includes embed_tokens.")

        output = self.model(
            input_ids=input_ids,
            embed_tokens=kwargs['embed_tokens'],
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
            new_k = "model." + k if 'lm_head' not in k and "embed_tokens" not in k else k
            new_weights[new_k] = v

        super().load_weights(new_weights)
