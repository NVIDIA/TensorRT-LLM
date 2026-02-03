from typing import Dict, Generic, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig, PretrainedConfig

from tensorrt_llm.logger import logger

from ...functional import PositionEmbeddingType
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig, TConfig
from ..modules.attention import MLA, Attention
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


def _ensure_draft_vocab_size(config: PretrainedConfig) -> None:
    if hasattr(config,
               "draft_vocab_size") and config.draft_vocab_size is not None:
        return

    logger.warning(
        "Missing 'draft_vocab_size' in pretrained config; defaulting to 'vocab_size'. "
        "Set 'draft_vocab_size' explicitly if the draft head uses a different vocabulary."
    )
    config.draft_vocab_size = config.vocab_size


class Eagle3Attention(Attention):

    def __init__(
        self,
        model_config: ModelConfig[LlamaConfig],
        layer_idx: Optional[int] = None,
        next_layer_regular: bool = False,
    ):
        config = model_config.pretrained_config
        self._next_layer_regular = next_layer_regular
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
        if not self._next_layer_regular:
            qkv_shard_indices_mapping = {
                "q": (0, self.q_size),
                "k": (self.q_size, self.kv_size),
                "v": (self.q_size + self.kv_size, self.kv_size),
            }
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
                fused_weight_shard_indices_mapping=qkv_shard_indices_mapping,
            )


class Eagle3MLAttention(MLA):
    """
    MLA (Multi-head Latent Attention) for Eagle3 draft model (e.g., DeepSeekV3).
    The first layer takes concatenated [embeds, hidden_states] as input (2x hidden_size),
    while subsequent layers take regular hidden_states (1x hidden_size).
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        next_layer_regular: bool = False,
    ):
        config = model_config.pretrained_config
        self._next_layer_regular = next_layer_regular

        predicted_tokens_per_seq = (
            model_config.spec_config.max_total_draft_tokens +
            1 if model_config.spec_config is not None else 1)

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
            aux_stream=aux_stream,
        )

        # Override the kv_a_proj_with_mqa projection for first layer.
        # The number of input features is twice as big for EAGLE3 draft models.
        if not self._next_layer_regular:
            quant_config = model_config.get_quant_config()
            # For Eagle3, first layer takes [embeds, hidden_states] concatenated
            self.kv_a_proj_with_mqa = Linear(
                2 * config.hidden_size,  # Double input size
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                dtype=config.torch_dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                use_custom_cublas_mm=True,
            )


class Eagle3DecoderLayer(DecoderLayer):
    """
    Unified decoder layer for Eagle3 speculative decoding.
    Supports both standard attention (Llama-style) and MLA (DeepSeekV3-style).
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int = 0,
        is_first_layer: bool = True,
        use_mla: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        config = model_config.pretrained_config
        eagle_config = config.eagle_config if hasattr(config,
                                                      "eagle_config") else {}
        self.layer_idx = layer_idx
        self._next_layer_regular = (eagle_config.get("next_layer_regular", True)
                                    and not is_first_layer) or eagle_config.get(
                                        "eh_proj_before_attn", False)

        # Select attention type based on config
        if use_mla:
            self.self_attn = Eagle3MLAttention(
                model_config,
                layer_idx,
                aux_stream=aux_stream,
                next_layer_regular=self._next_layer_regular,
            )
        else:
            self.self_attn = Eagle3Attention(model_config, layer_idx,
                                             self._next_layer_regular)

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

        if not self._next_layer_regular:
            self.input_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )

        self.hidden_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        if not self._next_layer_regular:
            embeds = self.input_layernorm(embeds)
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
    """
    Unified Eagle3 draft model supporting both standard attention (Llama-style)
    and MLA attention (DeepSeekV3-style).
    """

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        start_layer_idx: int = 0,
        use_mla: bool = False,
    ) -> None:
        super().__init__(model_config)

        config = model_config.pretrained_config
        eagle_config = config.eagle_config if hasattr(config,
                                                      "eagle_config") else {}
        self.spec_config = model_config.spec_config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.mapping = model_config.mapping
        self.num_layers = model_config.pretrained_config.num_hidden_layers
        self._eh_proj_before_attn = eagle_config.get("eh_proj_before_attn",
                                                     False)
        self._use_mla = use_mla

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self._return_hidden_post_norm = eagle_config.get(
            "return_hidden_post_norm", False)

        # Create auxiliary CUDA stream for MLA operations (only needed for MLA)
        self.aux_stream = torch.cuda.Stream() if use_mla else None

        if self.spec_config.num_capture_layers > 1:
            self.fc = Linear(
                self.hidden_size_in * self.spec_config.num_capture_layers,
                config.hidden_size,
                bias=getattr(config, "bias", False),
                dtype=config.torch_dtype,
            )

        if self.num_layers > 1:
            self.midlayer = nn.ModuleList([
                Eagle3DecoderLayer(
                    model_config,
                    start_layer_idx + i,
                    is_first_layer=(i == 0),
                    use_mla=use_mla,
                    aux_stream=self.aux_stream,
                ) for i in range(self.num_layers)
            ])
        else:
            self.midlayer = Eagle3DecoderLayer(
                model_config,
                start_layer_idx,
                use_mla=use_mla,
                aux_stream=self.aux_stream,
            )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        if (config.draft_vocab_size is not None
                and config.vocab_size != config.draft_vocab_size):
            self.d2t = nn.Parameter(
                torch.empty((config.draft_vocab_size, ), dtype=torch.int32),
                requires_grad=False,
            )

        if self._eh_proj_before_attn:
            self.enorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
            self.eh_proj = nn.Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=eagle_config.get("eh_proj_bias", False),
                dtype=config.torch_dtype,
            )

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
        # ideally, we expect that to happen outside the model definition. This
        # helps us avoid data-dependent control flow and gives us better CUDA
        # graph coverage.
        if self._eh_proj_before_attn:
            input_embeds = self.enorm(inputs_embeds)
            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
            hidden_states = self.eh_proj(hidden_states)

        residual = None
        if self.num_layers > 1:
            for layer in self.midlayer:
                if residual is not None:
                    hidden_states = hidden_states + residual
                hidden_states, residual = layer(
                    position_ids=position_ids,
                    embeds=inputs_embeds,
                    hidden_states=hidden_states,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                )
        else:
            hidden_states, residual = self.midlayer(
                position_ids=position_ids,
                embeds=inputs_embeds,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
            )

        hidden_states, hidden_states_to_save = self.norm(
            hidden_states, residual)
        if self._return_hidden_post_norm:
            return hidden_states, hidden_states
        return hidden_states, hidden_states_to_save


# We use Llama3 as the base architecture for EAGLE3 draft layers
@register_auto_model("EAGLE3LlamaForCausalLM")
@register_auto_model("Eagle3DeepSeekV3ForCausalLM")
class Eagle3ForCausalLM(DecoderModelForCausalLM[Eagle3DraftModel,
                                                PretrainedConfig]):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        start_layer_idx: int = 0,
    ):
        config = model_config.pretrained_config
        _ensure_draft_vocab_size(config)

        # Determine if we should use MLA attention based on config
        # MLA is used for DeepSeekV3-style models that have kv_lora_rank
        config = model_config.pretrained_config
        self._use_mla = hasattr(config, 'kv_lora_rank') and config.kv_lora_rank

        draft_model = Eagle3DraftModel(
            model_config,
            start_layer_idx,
            use_mla=self._use_mla,
        )

        super().__init__(
            draft_model,
            config=model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.draft_vocab_size,
        )
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

        if self._use_mla:
            # Use DeepseekV3WeightLoader for proper MLA weight handling
            from .modeling_deepseekv3 import DeepseekV3WeightLoader
            weight_loader = DeepseekV3WeightLoader(self, is_draft_model=False)
            if self.load_lm_head_from_target:
                weight_loader.load_weights(new_weights,
                                           skip_modules=['lm_head'])
            else:
                weight_loader.load_weights(new_weights)
        else:
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


class MistralLarge3DraftModel(DecoderModel):

    def __init__(
        self,
        model_config: ModelConfig,
        start_layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ) -> None:
        super().__init__(model_config)

        from .modeling_deepseekv3 import DeepseekV3DecoderLayer
        config = model_config.pretrained_config
        self.spec_config = model_config.spec_config
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.mapping = model_config.mapping
        self.num_layers = model_config.pretrained_config.num_hidden_layers

        self.fc = Linear(
            self.hidden_size * 2,
            config.hidden_size,
            bias=getattr(config, "bias", False),
            dtype=config.torch_dtype,
            quant_config=model_config.get_quant_config(),
        )
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(model_config, start_layer_idx,
                                   aux_stream_dict)
        ])

        self.norm = RMSNorm(hidden_size=config.hidden_size,
                            eps=config.rms_norm_eps,
                            dtype=config.torch_dtype)
        self.embed_tokens = None

    def post_load_weights(self):
        self.layers[0].next_layer_layernorm = self.norm

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        spec_metadata: SpecMetadata | None = None,
        hidden_states: torch.Tensor | None = None,
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
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        hidden_states, residual = self.layers[0](position_ids=position_ids,
                                                 hidden_states=hidden_states,
                                                 attn_metadata=attn_metadata,
                                                 residual=None,
                                                 spec_metadata=spec_metadata)

        return hidden_states, hidden_states


# We use MistralLarge3 as the base architecture for EAGLE3 draft layers
# NOTE: Class name says "Eagle" not "Eagle3" to match checkpoint naming (e.g., "Mistral-Large-3-675B-Instruct-2512-Eagle")
@register_auto_model("MistralLarge3EagleForCausalLM")
class MistralLarge3EagleForCausalLM(DecoderModelForCausalLM):

    def __init__(
        self,
        model_config: ModelConfig,
        start_layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        draft_vocab_size = model_config.pretrained_config.vocab_size
        super().__init__(MistralLarge3DraftModel(model_config, start_layer_idx,
                                                 aux_stream_dict),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=draft_vocab_size)
        self.load_lm_head_from_target = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata: SpecMetadata | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = spec_metadata.get_hidden_states()
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

    def load_weights(self, weights: Dict, *args, **kwargs):
        from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import \
            MistralLarge3WeightMapper
        params_map = kwargs.get("params_map")
        weight_mapper = MistralLarge3WeightMapper()
        if params_map is None:
            params_map = weight_mapper.mistral_llm_mapping

        llm_weights = weight_mapper.rename_by_params_map(weights=weights,
                                                         params_map=params_map)
        from .modeling_deepseekv3 import DeepseekV3WeightLoader
        weight_loader = DeepseekV3WeightLoader(self, is_draft_model=False)
        weight_loader.load_weights(llm_weights, skip_modules=['lm_head'])

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        if self.model.embed_tokens is None:
            self.model.embed_tokens = target_model.model.embed_tokens
        if self.load_lm_head_from_target:
            self.lm_head = target_model.lm_head

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(self.model.dtype)
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
        model_type = model_config.pretrained_config.model_type
        mtp_layer = None
        match model_type:
            case "glm4_moe":
                from .modeling_glm import Glm4MTP
                mtp_layer = Glm4MTP
            case "deepseek_v3" | "deepseek_v32":
                from .modeling_deepseekv3 import DeepseekV3MTP
                mtp_layer = DeepseekV3MTP
            case "exaone_moe":
                from .modeling_exaone_moe import ExaoneMoeMTP
                mtp_layer = ExaoneMoeMTP
            case "nemotron_h":
                from .modeling_nemotron_h import NemotronHMTP
                mtp_layer = NemotronHMTP
            case _:
                raise ValueError(
                    f"Model type {model_type} not supported for MTP")

        spec_dec_mode = model_config.spec_config.spec_dec_mode
        assert spec_dec_mode.is_mtp_one_model()
        mtp_num_layers = 1 if spec_dec_mode.is_mtp_eagle_one_model(
        ) else model_config.spec_config.num_nextn_predict_layers

        moe_load_balancer_set_repeated_for_next_layer(
            model_config.spec_config.num_nextn_predict_layers // mtp_num_layers)

        self.mtp_layers = nn.ModuleList([
            mtp_layer(model_config, layer_idx + start_layer_idx,
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
        model_type = model_config.pretrained_config.model_type
        if model_type == "glm4_moe":
            from .modeling_glm import Glm4MTP
            mtp_layer = Glm4MTP(model_config,
                                layer_idx,
                                aux_stream_dict,
                                is_separate_draft_engine=True)
        elif model_type in ["deepseek_v3", "deepseek_v32"]:
            from .modeling_deepseekv3 import DeepseekV3MTP
            mtp_layer = DeepseekV3MTP(model_config,
                                      layer_idx,
                                      aux_stream_dict,
                                      is_separate_draft_engine=True)
        elif model_type in ["exaone_moe"]:
            from .modeling_exaone_moe import ExaoneMoeMTP
            mtp_layer = ExaoneMoeMTP(model_config, layer_idx, aux_stream_dict)

        elif model_type == "nemotron_h":
            from .modeling_nemotron_h import NemotronHMTP
            mtp_layer = NemotronHMTP(model_config,
                                     layer_idx,
                                     aux_stream_dict,
                                     is_separate_draft_engine=False)
        else:
            raise ValueError(
                f"MTPDraftModel does not support model_type: {model_type}")
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
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.layers(
            input_ids,
            position_ids,
            hidden_states,
            embed_tokens=self.embed_tokens,
            attn_metadata=attn_metadata,
            all_rank_num_tokens=all_rank_num_tokens,
            spec_metadata=spec_metadata,
        )

        return hidden_states


@register_auto_model("MTPDraftModelForCausalLM")
class MTPDraftModelForCausalLM(DecoderModelForCausalLM[MTPDraftModel,
                                                       PretrainedConfig]):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        self.model_config = model_config
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
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
        model_type = self.model_config.pretrained_config.model_type
        match model_type:
            case "glm4_moe":
                from .modeling_glm import Glm4WeightLoader
                weight_loader = Glm4WeightLoader(self, is_draft_model=True)
            case "deepseek_v3" | "deepseek_v32":
                from .modeling_deepseekv3 import DeepseekV3WeightLoader
                weight_loader = DeepseekV3WeightLoader(self,
                                                       is_draft_model=True)
            case "exaone_moe":
                raise ValueError(
                    f"Model type {model_type} not supported for MTP for two engine mode. Please use one engine mode instead."
                )
            case _:
                raise ValueError(
                    f"Model type {model_type} not supported for MTP")
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
            spec_metadata=spec_metadata,
            **kwargs)
        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )


def get_draft_model(model_config, draft_config, lm_head, model):
    assert getattr(model_config, 'spec_config', None) is not None
    spec_dec_mode = model_config.spec_config.spec_dec_mode
    if spec_dec_mode.is_eagle3_one_model():
        if model_config.spec_config.eagle3_model_arch == "llama3":
            # Eagle3ForCausalLM handles both Llama3 and DeepSeekV3 architectures
            return Eagle3ForCausalLM(
                draft_config, model_config.pretrained_config.num_hidden_layers)
        elif model_config.spec_config.eagle3_model_arch == "mistral_large3":
            return MistralLarge3EagleForCausalLM(
                draft_config, model_config.pretrained_config.num_hidden_layers,
                model.aux_stream_dict)
        else:
            raise ValueError(
                f"Unsupported eagle3 model architecture: {spec_dec_mode.eagle3_model_arch}"
            )

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
        self.draft_config = None
        spec_config = getattr(model_config, 'spec_config', None)
        if spec_config and spec_config.spec_dec_mode.use_one_engine():
            if spec_config.spec_dec_mode.is_eagle3_one_model():
                if spec_config.eagle3_model_arch == "mistral_large3":
                    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import \
                        MistralConfigLoader
                    self.draft_config = MistralConfigLoader().load(
                        spec_config.speculative_model,
                        mapping=model_config.mapping,
                        moe_backend=model_config.moe_backend,
                        moe_max_num_tokens=model_config.moe_max_num_tokens,
                        max_num_tokens=model_config.max_num_tokens,
                        moe_load_balancer=model_config.moe_load_balancer,
                        skip_create_weights_in_init=True,
                    )
                    self.draft_config.extra_attrs = model_config.extra_attrs
                elif spec_config.eagle3_model_arch == "llama3":
                    self.draft_config = ModelConfig.from_pretrained(
                        model_config.spec_config.speculative_model,
                        trust_remote_code=True,
                        attn_backend=model_config.attn_backend,
                        moe_backend=model_config.moe_backend,
                        mapping=model_config.mapping,
                        spec_config=model_config.spec_config,
                        max_num_tokens=model_config.max_num_tokens,
                        moe_max_num_tokens=model_config.moe_max_num_tokens)
                else:
                    raise ValueError(
                        f"Unsupported eagle3 model architecture for draft model: {spec_config.eagle3_model_arch}"
                    )
                self.draft_config.quant_config.kv_cache_quant_algo = \
                model_config.quant_config.kv_cache_quant_algo

            self.draft_model = get_draft_model(model_config, self.draft_config,
                                               self.lm_head, self.model)
            self.spec_worker = get_spec_worker(model_config.spec_config,
                                               model_config,
                                               model_config.mapping)
            self.epilogue.append(self.draft_model)
            self.epilogue.append(self.spec_worker)

            if self.draft_config is not None and model_config.spec_config.eagle3_model_arch == "llama3":
                for key, value in self.draft_config.extra_attrs.items():
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
                     weight_mapper: Optional[BaseWeightMapper] = None,
                     params_map: Optional[Dict[str, str]] = None,
                     allow_partial_loading: bool = False):
        super().load_weights(weights=weights,
                             weight_mapper=weight_mapper,
                             skip_modules=["draft_model"],
                             params_map=params_map,
                             allow_partial_loading=allow_partial_loading)

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
