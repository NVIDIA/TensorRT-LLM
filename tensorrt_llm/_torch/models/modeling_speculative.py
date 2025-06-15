from typing import Any, Dict, Generic, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from tensorrt_llm.functional import PositionEmbeddingType

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig, TConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata, get_spec_worker
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

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = Linear(self.hidden_size_in * 3,
                         config.hidden_size,
                         bias=getattr(config, "bias", False),
                         dtype=config.torch_dtype)

        self.midlayer = Eagle3DecoderLayer(model_config, start_layer_idx)

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)

        if config.draft_vocab_size is not None and config.vocab_size != config.draft_vocab_size:
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

    def load_weights(self, weights: Dict):
        new_weights = {}
        for k, v in weights.items():
            if 'lm_head' not in k:
                new_k = "model." + k
            else:
                self.load_lm_head_from_target = False
                new_k = k
            new_weights[new_k] = v
        if self.load_lm_head_from_target:
            super().load_weights(new_weights, skip_modules=['lm_head'])
        else:
            super().load_weights(new_weights)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens
        if self.load_lm_head_from_target:
            self.lm_head = target_model.lm_head

    # TODO: should input/position IDs be included in this? Keeping it implicit
    # for now since the shapes/dtypes are the same across all models we have.
    def get_warmup_extra_inputs(self, batch_size: int,
                                num_tokens: int) -> Dict[str, Any]:

        hidden_states = torch.empty(batch_size * num_tokens,
                                    self.model.hidden_size,
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


def get_draft_model(model_config, draft_config):
    assert getattr(model_config, 'spec_config', None) != None
    spec_dec_mode = model_config.spec_config.spec_dec_mode
    if spec_dec_mode.is_eagle3_one_model():
        return Eagle3ForCausalLM(
            draft_config, model_config.pretrained_config.num_hidden_layers)
    else:
        raise NotImplemented(
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
        if getattr(
                model_config, 'spec_config', None
        ) and model_config.spec_config.spec_dec_mode.use_one_engine():
            draft_config = ModelConfig.from_pretrained(
                model_config.spec_config.draft_model_path,
                trust_remote_code=True,
                attn_backend=model_config.attn_backend,
                moe_backend=model_config.moe_backend,
                mapping=model_config.mapping,
                spec_config=model_config.spec_config,
                max_num_tokens=model_config.max_num_tokens,
                moe_max_num_tokens=model_config.moe_max_num_tokens)

            draft_config.quant_config.kv_cache_quant_algo = \
                model_config.quant_config.kv_cache_quant_algo

            self.draft_model = get_draft_model(model_config, draft_config)
            self.spec_worker = get_spec_worker(model_config.spec_config,
                                               model_config.mapping)

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
                                    draft_model=self.draft_model)
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
