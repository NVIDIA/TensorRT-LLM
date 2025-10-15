from typing import Dict, Generic, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig, PretrainedConfig

from ...functional import PositionEmbeddingType
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig, TConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import moe_load_balancer_set_repeated_for_next_layer
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..modules.rms_norm import RMSNorm
from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
from ..speculative import SpecMetadata, get_spec_worker
from ..utils import AuxStreamType
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM, TModel,
                             register_auto_model)


class Eagle3Attention(Attention):

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

        tp_size = model_config.mapping.tp_size
        if model_config.mapping.enable_attention_dp:
            tp_size = 1
        # Override the QKV projection. The number of input features
        # is twice as big for EAGLE3 draft models.
        self.qkv_proj = Linear(
            2 * self.hidden_size,
            tp_size * self.q_size + 2 * tp_size * self.kv_size,
            bias=config.attention_bias,
            dtype=config.torch_dtype,
            mapping=self.qkv_proj.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_QKV_LINEAR),
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.
            skip_create_weights_in_init,
        )


class Eagle3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: LlamaConfig,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.self_attn = Eagle3Attention(model_config, layer_idx)

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
            overridden_tp_size=1
            if model_config.mapping.enable_attention_dp else None,
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

        # We save the hidden states in the spec metadata here. In _prepare_draft_tokens,
        # PyExecutor will extract these from the draft model engine's spec metadata.
        # They will be passed to the draft model engine on the next iteration.
        # TODO: can we support multiple model outputs instead?
        spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states,
                                                  residual)
        return hidden_states, residual


class Eagle3DraftModel(DecoderModel):

    def __init__(
        self,
        model_config: LlamaConfig,
        start_layer_idx: int = 0,
    ) -> None:
        super().__init__(model_config)

        config = model_config.pretrained_config
        self.spec_config = model_config.spec_config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.mapping = model_config.mapping
        self.num_layers = model_config.pretrained_config.num_hidden_layers

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        if self.spec_config.num_capture_layers > 1:
            self.fc = Linear(self.hidden_size_in *
                             self.spec_config.num_capture_layers,
                             config.hidden_size,
                             bias=getattr(config, "bias", False),
                             dtype=config.torch_dtype)

        if self.num_layers > 1:
            self.midlayer = nn.ModuleList([
                Eagle3DecoderLayer(model_config, start_layer_idx + i)
                for i in range(self.num_layers)
            ])
        else:
            self.midlayer = Eagle3DecoderLayer(model_config, start_layer_idx)

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

        if config.draft_vocab_size is not None and config.vocab_size != config.draft_vocab_size:
            self.d2t = nn.Parameter(torch.empty((config.draft_vocab_size, ),
                                                dtype=torch.int32),
                                    requires_grad=False)

        if self.hidden_size_in != config.hidden_size:
            if model_config.mapping.enable_attention_dp:
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
        residual = None
        if self.num_layers > 1:
            for layer in self.midlayer:
                if residual is not None:
                    hidden_states = hidden_states + residual
                hidden_states, residual = layer(position_ids=position_ids,
                                                embeds=inputs_embeds,
                                                hidden_states=hidden_states,
                                                attn_metadata=attn_metadata,
                                                spec_metadata=spec_metadata)
        else:
            hidden_states, residual = self.midlayer(position_ids=position_ids,
                                                    embeds=inputs_embeds,
                                                    hidden_states=hidden_states,
                                                    attn_metadata=attn_metadata,
                                                    spec_metadata=spec_metadata)

        hidden_states, hidden_states_to_save = self.norm(
            hidden_states, residual)
        return hidden_states, hidden_states_to_save


# We use Llama3 as the base architecture for EAGLE3 draft layers
@register_auto_model("EAGLE3LlamaForCausalLM")
class Eagle3ForCausalLM(DecoderModelForCausalLM[Eagle3DraftModel, LlamaConfig]):

    def __init__(
        self,
        model_config: LlamaConfig,
        start_layer_idx: int = 0,
    ):
        draft_vocab_size = model_config.pretrained_config.vocab_size
        if model_config.pretrained_config.draft_vocab_size is not None:
            draft_vocab_size = model_config.pretrained_config.draft_vocab_size
        super().__init__(Eagle3DraftModel(model_config, start_layer_idx),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=draft_vocab_size)
        self.load_lm_head_from_target = True

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
        hidden_states = self.apply_eagle3_fc(spec_metadata.get_hidden_states())
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

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        new_weights = {}
        for k, v in weights.items():
            if 'lm_head' not in k:
                new_k = "model." + k
            else:
                self.load_lm_head_from_target = False
                new_k = k
            new_weights[new_k] = v
        if self.load_lm_head_from_target:
            super().load_weights(weights=new_weights,
                                 weight_mapper=weight_mapper,
                                 skip_modules=['lm_head'])
        else:
            super().load_weights(weights=new_weights,
                                 weight_mapper=weight_mapper)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens
        if self.load_lm_head_from_target:
            self.lm_head = target_model.lm_head

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


class MTPForCausalLM(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        start_layer_idx: int = 0,
        lm_head: nn.Module = None,
        model: nn.Module = None,
    ):
        super().__init__()
        # Import here to avoid circular import
        from .modeling_deepseekv3 import DeepseekV3MTP

        spec_dec_mode = model_config.spec_config.spec_dec_mode
        assert spec_dec_mode.is_mtp_one_model()
        mtp_num_layers = 1 if spec_dec_mode.is_mtp_eagle_one_model(
        ) else model_config.spec_config.num_nextn_predict_layers

        moe_load_balancer_set_repeated_for_next_layer(
            model_config.spec_config.num_nextn_predict_layers // mtp_num_layers)

        self.mtp_layers = nn.ModuleList([
            DeepseekV3MTP(model_config, layer_idx + start_layer_idx,
                          model.aux_stream_dict)
            for layer_idx in range(mtp_num_layers)
        ])
        self.lm_head = lm_head
        self.embed_tokens = model.embed_tokens


class MTPDraftModel(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int, aux_stream_dict: Dict[AuxStreamType,
                                                       torch.cuda.Stream]):
        super().__init__()
        # Import here to avoid circular import
        from .modeling_deepseekv3 import DeepseekV3MTP

        mtp_layer = DeepseekV3MTP(model_config,
                                  layer_idx,
                                  aux_stream_dict,
                                  is_separate_draft_engine=True)
        setattr(self, f"layers.{layer_idx}", mtp_layer)
        self.layers = mtp_layer
        self.layer_idx = layer_idx
        self.config = model_config.pretrained_config
        self.embed_tokens = Embedding(
            self.config.vocab_size,
            self.config.hidden_size,
            dtype=self.config.torch_dtype,
        )

    def __repr__(self):
        """Custom string representation to display layer index"""
        return f"(layers): ({self.layer_idx}): {repr(self.layers)}"

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.layers(
            input_ids,
            position_ids,
            hidden_states,
            embed_tokens=self.embed_tokens,
            attn_metadata=attn_metadata,
            all_rank_num_tokens=all_rank_num_tokens,
        )

        return hidden_states


@register_auto_model("MTPDraftModelForCausalLM")
class MTPDraftModelForCausalLM(DecoderModelForCausalLM[MTPDraftModel,
                                                       PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        self.model_config = model_config
        aux_stream_list = [torch.cuda.Stream() for _ in range(2)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
        }
        super().__init__(
            MTPDraftModel(self.model_config,
                          self.model_config.pretrained_config.num_hidden_layers,
                          self.aux_stream_dict),
            config=self.model_config,
            hidden_size=self.model_config.pretrained_config.hidden_size,
            vocab_size=self.model_config.pretrained_config.vocab_size)

    def load_weights(self, weights: Dict):
        # Import here to avoid circular import
        from .modeling_deepseekv3 import DeepseekV3WeightLoader
        weight_loader = DeepseekV3WeightLoader(self, is_draft_model=True)
        weight_loader.load_weights(weights)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens
        self.lm_head = target_model.lm_head

    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: torch.IntTensor = None,
                position_ids: torch.IntTensor = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                return_context_logits: bool = False,
                spec_metadata: Optional[SpecMetadata] = None,
                hidden_states: torch.Tensor = None,
                **kwargs) -> torch.Tensor:

        hidden_states = spec_metadata.get_hidden_states()
        output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
            **kwargs)
        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )


def get_draft_model(model_config, draft_config, lm_head, model):
    assert getattr(model_config, 'spec_config', None) != None
    spec_dec_mode = model_config.spec_config.spec_dec_mode
    if spec_dec_mode.is_eagle3_one_model():
        return Eagle3ForCausalLM(
            draft_config, model_config.pretrained_config.num_hidden_layers)
    elif spec_dec_mode.is_mtp_one_model():
        return MTPForCausalLM(model_config,
                              model_config.pretrained_config.num_hidden_layers,
                              lm_head, model)
    elif spec_dec_mode.is_mtp_eagle():
        return MTPDraftModelForCausalLM(model_config)
    else:
        raise NotImplementedError(
            f"get_draft_model does not support speculative decoding mode {spec_dec_mode}."
        )


class SpecDecOneEngineForCausalLM(DecoderModelForCausalLM[TModel, TConfig],
                                  Generic[TModel, TConfig]):

    def __init__(self, model: TModel, model_config: ModelConfig[TConfig]):
        super().__init__(model,
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)
        self.draft_model = None
        spec_config = getattr(model_config, 'spec_config', None)
        if spec_config and spec_config.spec_dec_mode.use_one_engine():
            draft_config = None
            if spec_config.spec_dec_mode.is_eagle3_one_model():
                draft_config = ModelConfig.from_pretrained(
                    model_config.spec_config.speculative_model_dir,
                    trust_remote_code=True,
                    attn_backend=model_config.attn_backend,
                    moe_backend=model_config.moe_backend,
                    mapping=model_config.mapping,
                    spec_config=model_config.spec_config,
                    max_num_tokens=model_config.max_num_tokens,
                    moe_max_num_tokens=model_config.moe_max_num_tokens)
                draft_config.quant_config.kv_cache_quant_algo = \
                model_config.quant_config.kv_cache_quant_algo

            self.draft_model = get_draft_model(model_config, draft_config,
                                               self.lm_head, self.model)
            self.spec_worker = get_spec_worker(model_config.spec_config,
                                               model_config,
                                               model_config.mapping)

            if draft_config is not None:
                for key, value in draft_config.extra_attrs.items():
                    assert key in ('attn_layers', 'mla_layers')
                    assert key in model_config.extra_attrs
                    model_config.extra_attrs[key].update(value)
        self.layer_idx = -1

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
            **kwargs,
        )

        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states)
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[:attn_metadata.num_tokens]

        if self.draft_model is not None:
            # get logits
            logits = self.logits_processor.forward(
                hidden_states[spec_metadata.gather_ids],
                self.lm_head,
                attn_metadata,
                True,
            )
            mtp_input_ids = input_ids
            mtp_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if input_ids is not None:
                    # Slice along the first dimension
                    mtp_input_ids = input_ids[:attn_metadata.num_tokens]
                if position_ids is not None:
                    # Slice along the last dimension
                    mtp_position_ids = position_ids[:, :attn_metadata.
                                                    num_tokens]

            # get accepted tokens and next draft tokens
            return self.spec_worker(input_ids=mtp_input_ids,
                                    position_ids=mtp_position_ids,
                                    hidden_states=hidden_states,
                                    logits=logits,
                                    attn_metadata=attn_metadata,
                                    spec_metadata=spec_metadata,
                                    draft_model=self.draft_model)
        else:
            logits = self.logits_processor.forward(
                hidden_states,
                self.lm_head,
                attn_metadata,
                return_context_logits,
            )

        return logits

    def load_weights(self,
                     weights: Dict,
                     weight_mapper: Optional[BaseWeightMapper] = None):
        super().load_weights(weights=weights,
                             weight_mapper=weight_mapper,
                             skip_modules=["draft_model"])

    def load_draft_weights(self,
                           weights: Dict,
                           weight_mapper: Optional[BaseWeightMapper] = None):
        self.draft_model.load_weights(weights=weights,
                                      weight_mapper=weight_mapper)
        self.draft_model.load_weights_from_target_model(self)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False
