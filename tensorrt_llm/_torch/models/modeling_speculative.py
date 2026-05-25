import inspect
from dataclasses import replace
from typing import Dict, Generic, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, PretrainedConfig

from tensorrt_llm.logger import logger

from ...functional import PositionEmbeddingType, RotaryScalingType
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
from ..modules.rotary_embedding import RotaryEmbedding

try:
    from ..custom_ops import \
        flashinfer_apply_rope_with_cos_sin_cache_inplace as _flashinfer_rope
except ImportError:
    _flashinfer_rope = None
from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
from ..speculative import (SpecMetadata, get_spec_worker,
                           should_use_separate_draft_kv_cache)
from ..utils import AuxStreamType
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM, TModel,
                             get_model_architecture, register_auto_model)


def _ensure_draft_vocab_size(config: PretrainedConfig) -> None:
    if hasattr(config,
               "draft_vocab_size") and config.draft_vocab_size is not None:
        return

    logger.warning(
        "Missing 'draft_vocab_size' in pretrained config; defaulting to 'vocab_size'. "
        "Set 'draft_vocab_size' explicitly if the draft head uses a different vocabulary."
    )
    config.draft_vocab_size = config.vocab_size


def _slice_spec_position_ids(position_ids: Optional[torch.Tensor],
                             num_tokens: int) -> Optional[torch.Tensor]:
    """Slice speculative position IDs along the token dimension."""
    if position_ids is None:
        return None
    return position_ids[..., :num_tokens]


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

        predicted_tokens_per_seq = (model_config.spec_config.tokens_per_gen_step
                                    if model_config.spec_config is not None else
                                    1)

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
        self._norm_before_fc = eagle_config.get("norm_before_fc", False)
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
                quant_config=model_config.get_quant_config(),
            )
        if self._norm_before_fc:
            self.input_norm = RMSNorm(
                hidden_size=self.hidden_size_in *
                self.spec_config.num_capture_layers,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
        else:
            self.input_norm = None

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
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # When ``all_rank_num_tokens`` is supplied the caller wants this draft
        # forward to run with a different attention-DP token distribution
        # (e.g. the worker's per-step value); restore the original on exit so
        # the next call sees the same attn_metadata it had on entry.
        previous_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        try:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                assert self.embed_tokens is not None
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
        finally:
            if all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = previous_all_rank_num_tokens


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
        # Remap weight names: some Eagle3 checkpoints use "layers.X.*" naming convention
        # while the model expects "midlayer.*" naming. Handle both formats.
        import re
        remapped_weights = {}
        # Access num_layers from the inner draft model (self.model is Eagle3DraftModel)
        num_layers = self.model.num_layers
        for k, v in weights.items():
            new_k = k
            # For single-layer models: "layers.0.*" -> "midlayer.*"
            # For multi-layer models: "layers.X.*" -> "midlayer.X.*"
            if num_layers == 1:
                # Single layer: layers.0.foo -> midlayer.foo
                new_k = re.sub(r'^layers\.0\.', 'midlayer.', new_k)
            else:
                # Multi-layer: layers.X.foo -> midlayer.X.foo
                new_k = re.sub(r'^layers\.(\d+)\.', r'midlayer.\1.', new_k)
            remapped_weights[new_k] = v

        new_weights = {}
        for k, v in remapped_weights.items():
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
            if self.model._norm_before_fc:
                hidden_states = self.model.input_norm(hidden_states)
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            assert self.embed_tokens is not None
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


class PARDForCausalLM(nn.Module):
    """Draft model wrapper for PARD (Parallel Draft) speculative decoding.

    See PARDWorker for the full algorithm description.
    """

    def __init__(self, draft_config):
        super().__init__()
        DraftModelClass, _ = get_model_architecture(
            draft_config.pretrained_config)

        # Remove spec_config to prevent recursive spec-dec initialization
        draft_config_no_spec = replace(draft_config, spec_config=None)

        # Weights will be loaded later by ModelLoader.load_draft_weights()
        self.draft_model_full = DraftModelClass(draft_config_no_spec)
        self.model = self.draft_model_full.model
        self.lm_head = self.draft_model_full.lm_head

        # Required by weight mappers
        self.model_config = draft_config_no_spec
        self.config = draft_config_no_spec.pretrained_config

        # Fall back: pard_token -> mask_token_id -> vocab_size
        pretrained_config = draft_config.pretrained_config
        self.mask_token_id = getattr(
            pretrained_config, 'pard_token',
            getattr(pretrained_config, 'mask_token_id',
                    pretrained_config.vocab_size))
        logger.info(
            f"PARD draft model initialized with mask_token_id: {self.mask_token_id}"
        )

        self.logits_processor = None  # Set by caller after construction

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load weights into the PARD draft model."""
        self.draft_model_full.load_weights(weights=weights,
                                           weight_mapper=weight_mapper,
                                           **kwargs)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_out = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )

        return hidden_states_out, hidden_states_out


class DFlashForCausalLM(nn.Module):
    """Draft model wrapper for DFlash speculative decoding.

    DFlash uses cross-attention where Q comes from noise/query tokens and K/V
    come from the concatenation of target hidden states and noise hidden states.
    The target_hidden stays CONSTANT across all layers (no input_layernorm applied).

    Reference: https://arxiv.org/pdf/2602.06036
    """

    def __init__(self, draft_config):
        super().__init__()

        # DFlash draft models may use custom architecture names (e.g. "DFlashDraftModel")
        # that are not registered in MODEL_CLASS_MAPPING. Fall back to model_type-based
        # architecture name (e.g. "qwen3" -> "Qwen3ForCausalLM").
        pretrained_cfg = draft_config.pretrained_config
        try:
            DraftModelClass, _ = get_model_architecture(pretrained_cfg)
        except RuntimeError:
            model_type = pretrained_cfg.model_type
            arch_name = "".join(w.capitalize()
                                for w in model_type.split("_")) + "ForCausalLM"
            logger.info(
                f"DFlash: architecture {pretrained_cfg.architectures} not found, "
                f"falling back to {arch_name} based on model_type={model_type}")
            original_archs = pretrained_cfg.architectures
            try:
                pretrained_cfg.architectures = [arch_name]
                DraftModelClass, _ = get_model_architecture(pretrained_cfg)
            finally:
                pretrained_cfg.architectures = original_archs

        # Remove spec_config to prevent recursive spec-dec initialization
        draft_config_no_spec = replace(draft_config, spec_config=None)

        # Weights will be loaded later by ModelLoader.load_draft_weights()
        self.draft_model_full = DraftModelClass(draft_config_no_spec)
        self.model = self.draft_model_full.model
        self.lm_head = self.draft_model_full.lm_head

        # Required by weight mappers
        self.model_config = draft_config_no_spec
        self.config = draft_config_no_spec.pretrained_config

        # Get mask_token_id from dflash_config
        pretrained_config = draft_config.pretrained_config
        dflash_config = getattr(pretrained_config, 'dflash_config', {})
        self.mask_token_id = dflash_config.get(
            'mask_token_id',
            getattr(pretrained_config, 'mask_token_id',
                    pretrained_config.vocab_size))

        self.target_layer_ids = dflash_config.get('target_layer_ids', None)
        self.block_size = getattr(pretrained_config, 'block_size', None)
        logger.info(
            f"DFlash draft model initialized with mask_token_id: {self.mask_token_id}, "
            f"target_layer_ids: {self.target_layer_ids}, block_size: {self.block_size}"
        )

        self.logits_processor = None  # Set by caller after construction

        # RoPE - lazily initialized from draft model's attention module
        self._rope_initialized = False
        self._rotary_cos_sin = None
        self._is_neox = True

        self._cos_sin_cache_fp32 = None
        self._rope_dummy_q = None

        # Lazy-built after weights load (see _build_fused_kv_buffers).
        self._fused_kv_weight = None
        self._fused_kv_bias = None
        self._k_norm_stacked = None
        self._k_norm_eps = None
        self._num_attn_layers = 0
        self._head_dim = 0
        self._num_kv_heads = 0
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False

    def _init_rope(self):
        """Initialize RoPE from the draft model's attention configuration.

        Reuses the existing RotaryEmbedding infrastructure which correctly
        handles all RoPE variants (standard, YaRN, scaled, etc.).
        """
        attn0 = self.model.layers[0].self_attn

        if attn0.rotary_emb is not None:
            self._rotary_cos_sin = attn0.rotary_emb.rotary_cos_sin
            self._is_neox = attn0.rotary_emb.is_neox
        elif attn0.pos_embd_params is not None:
            rope_emb = RotaryEmbedding(
                attn0.pos_embd_params.rope,
                head_dim=attn0.head_dim,
                is_neox=attn0.pos_embd_params.is_neox,
            )
            self._rotary_cos_sin = rope_emb.rotary_cos_sin
            self._is_neox = rope_emb.is_neox
        else:
            # Fallback: basic NeoX-style RoPE
            config = self.config
            head_dim = getattr(config, 'head_dim',
                               config.hidden_size // config.num_attention_heads)
            rope_theta = getattr(config, 'rope_theta', 1000000.0)
            max_pos = getattr(config, 'max_position_embeddings', 32768)

            inv_freq = 1.0 / (rope_theta**(torch.arange(
                0, head_dim, 2, dtype=torch.float32, device='cuda') / head_dim))
            positions = torch.arange(max_pos,
                                     dtype=torch.float32,
                                     device='cuda')
            freqs = torch.outer(positions, inv_freq)
            rope_cos = freqs.cos().to(config.torch_dtype)
            rope_sin = freqs.sin().to(config.torch_dtype)
            # [max_pos, 2, rot_dim//2] to match RotaryEmbedding format
            self._rotary_cos_sin = torch.stack([rope_cos, rope_sin], dim=1)
            self._is_neox = True

        self._rope_initialized = True

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load weights into the DFlash draft model.

        DFlash checkpoints differ from standard HF format:
        - Layer weights lack the 'model.' prefix (e.g., 'layers.0...' not 'model.layers.0...')
        - Extra DFlash-specific weights: 'fc.weight', 'hidden_norm.weight'
        - Missing embed_tokens and lm_head (shared with target model)
        """
        # Remap: add 'model.' prefix where needed, and extract DFlash-specific weights
        remapped = {}
        for key, value in weights.items():
            if key in ('fc.weight', 'hidden_norm.weight'):
                # DFlash-specific projection weights - store directly
                remapped[key] = value
            elif key == 'norm.weight':
                remapped['model.norm.weight'] = value
            elif not key.startswith('model.'):
                remapped[f'model.{key}'] = value
            else:
                remapped[key] = value

        # Load DFlash-specific weights directly
        if 'fc.weight' in remapped:
            self.fc = nn.Linear(remapped['fc.weight'].shape[1],
                                remapped['fc.weight'].shape[0],
                                bias=False,
                                device='cuda',
                                dtype=remapped['fc.weight'].dtype)
            self.fc.weight.data.copy_(remapped['fc.weight'])
            del remapped['fc.weight']

        if 'hidden_norm.weight' in remapped:
            rms_norm_eps = getattr(self.config, 'rms_norm_eps', 1e-6)
            self.hidden_norm = nn.RMSNorm(
                remapped['hidden_norm.weight'].shape[0],
                eps=rms_norm_eps,
                device='cuda',
                elementwise_affine=True,
                dtype=remapped['hidden_norm.weight'].dtype)
            self.hidden_norm.weight.data.copy_(remapped['hidden_norm.weight'])
            del remapped['hidden_norm.weight']

        # Load remaining weights into the draft model.
        # DFlash checkpoints don't include embed_tokens or lm_head, so allow partial loading
        # since those modules won't find matching weights.
        self.draft_model_full.load_weights(weights=remapped,
                                           weight_mapper=weight_mapper,
                                           allow_partial_loading=True)

    def load_weights_from_target_model(self,
                                       target_model: torch.nn.Module) -> None:
        """Share embed_tokens and lm_head from the target model."""
        self.draft_model_full.model.embed_tokens = target_model.model.embed_tokens
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head

    def precompute_context_kv(
        self,
        projected_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-norm / post-RoPE K and V for ALL drafter layers in one fused GEMM.

        Args:
            projected_hidden: [N, hidden_size], already fc + hidden_norm'd.
            positions:        [N] int32/64, RoPE positions for each entry.
        Returns:
            k: [N, L, nkv, hd]  post k_norm and RoPE
            v: [N, L, nkv, hd]  post split only
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        N = projected_hidden.shape[0]
        L = self._num_attn_layers
        nkv = self._num_kv_heads
        hd = self._head_dim
        weight_dtype = self._fused_kv_weight.dtype
        if projected_hidden.dtype != weight_dtype:
            projected_hidden = projected_hidden.to(weight_dtype)

        kv_flat = F.linear(projected_hidden, self._fused_kv_weight,
                           self._fused_kv_bias)
        # Per-layer layout [L0_K|L0_V|L1_K|L1_V|...] keeps K and V contiguous
        # after the select() splits — no extra copy required.
        kv = kv_flat.view(N, L, 2, nkv, hd)
        k = kv[:, :, 0].contiguous()
        v = kv[:, :, 1].contiguous()

        if self._k_norm_stacked is not None:
            # Fuse L per-layer RMSNorms into one. k is [N, L, nkv, hd];
            # each layer has its own weight ([L, hd]) but shares eps.
            k = F.rms_norm(k, (hd, ), eps=self._k_norm_eps)
            k = k * self._k_norm_stacked.view(1, L, 1, hd)

        self._fused_rope_inplace(k.view(N * L, nkv * hd), positions, N, L)
        return k, v

    def _get_cos_sin_cache(self) -> torch.Tensor:
        """Return the flashinfer-style cos/sin cache for the drafter.

        Shape [max_positions, head_dim], fp32 — flashinfer's
        apply_rope_with_cos_sin_cache_inplace requires fp32 regardless of
        the query/key dtype.
        """
        if self._cos_sin_cache_fp32 is not None:
            return self._cos_sin_cache_fp32
        if not self._rope_initialized:
            self._init_rope()
        max_pos = self._rotary_cos_sin.shape[0]
        self._cos_sin_cache_fp32 = self._rotary_cos_sin.view(max_pos, -1).to(
            torch.float32).contiguous()
        return self._cos_sin_cache_fp32

    def _fused_rope_inplace(
        self,
        k_flat: torch.Tensor,
        positions: torch.Tensor,
        N: int,
        L: int,
    ) -> None:
        """In-place fused RoPE over [N*L, nkv*hd] K values.

        Layout of k_flat: row (i*L + l) holds layer l of position i, so
        positions must be repeat_interleaved by L to match.
        """
        positions_int32 = positions.view(-1).to(torch.int32)
        if L > 1:
            positions_int32 = positions_int32.repeat_interleave(L)

        if _flashinfer_rope is not None:
            # flashinfer requires a non-None query tensor; pass a single-head
            # scratch so the extra rotate is negligible.
            need_rows = k_flat.shape[0]
            dummy_q = self._rope_dummy_q
            if (dummy_q is None or dummy_q.dtype != k_flat.dtype
                    or dummy_q.shape[0] < need_rows):
                dummy_q = k_flat.new_empty(need_rows, self._head_dim)
                self._rope_dummy_q = dummy_q
            _flashinfer_rope(
                positions_int32,
                dummy_q[:need_rows],
                k_flat,
                self._head_dim,
                self._get_cos_sin_cache(),
                self._is_neox,
            )
            return

        # Pure-PyTorch fallback (older environments without flashinfer).
        cos, sin = self._get_rope_cos_sin(positions_int32.view(1, -1),
                                          dtype=k_flat.dtype)
        k_roped = RotaryEmbedding.apply_rotary_pos_emb(
            k_flat.view(k_flat.shape[0], -1, self._head_dim),
            cos.squeeze(0),
            sin.squeeze(0),
            unsqueeze_dim=1,
            is_neox=self._is_neox,
        )
        k_flat.copy_(k_roped.view_as(k_flat))

    def _build_fused_kv_buffers(self) -> None:
        """Stack per-layer KV projection + k_norm weights for a single fused GEMM.

        Must run after weights are loaded.
        """
        if self._fused_kv_weight is not None:
            return
        layers_attn = [layer.self_attn for layer in self.model.layers]
        attn0 = layers_attn[0]
        q_size = attn0.q_size
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_kv_heads = attn0.num_key_value_heads
        for a in layers_attn[1:]:
            assert (a.kv_size == kv_size and a.head_dim == head_dim
                    and a.num_key_value_heads == num_kv_heads), (
                        "DFlash fused KV requires all drafter layers to share "
                        "kv_size / head_dim / num_kv_heads.")

        has_k_norm = [hasattr(a, 'k_norm') for a in layers_attn]
        assert all(has_k_norm) or not any(has_k_norm), (
            "DFlash fused KV requires either all or no drafter layers to have k_norm."
        )

        kv_weights = [
            a.qkv_proj.weight[q_size:q_size + 2 * kv_size] for a in layers_attn
        ]
        fused_kv_weight = torch.cat(kv_weights, dim=0).contiguous()
        if attn0.qkv_proj.bias is not None:
            kv_biases = [
                a.qkv_proj.bias[q_size:q_size + 2 * kv_size]
                for a in layers_attn
            ]
            self._fused_kv_bias = torch.cat(kv_biases, dim=0).contiguous()
        else:
            self._fused_kv_bias = None

        if all(has_k_norm):
            k_norm0 = layers_attn[0].k_norm
            eps = k_norm0.variance_epsilon
            eps_set = {a.k_norm.variance_epsilon for a in layers_attn}
            assert len(eps_set) == 1, (
                f"DFlash fused k_norm requires all drafter layers to share "
                f"variance_epsilon; got {sorted(eps_set)}.")
            self._k_norm_stacked = torch.stack(
                [a.k_norm.weight.data for a in layers_attn])
            self._k_norm_eps = eps
        else:
            self._k_norm_stacked = None
            self._k_norm_eps = None
        self._num_attn_layers = len(layers_attn)
        self._head_dim = head_dim
        self._num_kv_heads = num_kv_heads
        self._fused_kv_weight = fused_kv_weight

        # fused_qk_norm_rope derives YaRN / partial-rotary frequencies on
        # the fly, which can disagree with precompute_context_kv's cached
        # cos/sin. Only enable it when the drafter uses plain RoPE.
        self._has_qk_norm = (all(has_k_norm)
                             and all(hasattr(a, 'q_norm') for a in layers_attn))
        rope_params = getattr(getattr(attn0, 'pos_embd_params', None), 'rope',
                              None)
        scale_type = getattr(rope_params, 'scale_type', None)
        partial_rotary_factor = getattr(
            getattr(attn0, 'pretrained_config', None), 'partial_rotary_factor',
            1.0)
        self._use_fused_qk_norm_rope = (self._has_qk_norm
                                        and hasattr(attn0, 'apply_qk_norm_rope')
                                        and rope_params is not None
                                        and scale_type
                                        in (None, RotaryScalingType.none)
                                        and partial_rotary_factor == 1.0)

        logger.debug(
            f"DFlash: fused KV weights built for {self._num_attn_layers} layers "
            f"(fused_kv_weight shape={tuple(self._fused_kv_weight.shape)})")

    def _get_rope_cos_sin(self, positions, dtype=None):
        """Get cos/sin for given positions, suitable for apply_rotary_pos_emb.

        Args:
            positions: [B, seq_len]
            dtype: target dtype for cos/sin (default: keep original)
        Returns:
            rope_cos: [B, seq, rot_dim//2] (broadcastable with unsqueeze_dim=1)
            rope_sin: [B, seq, rot_dim//2]
        """
        if not self._rope_initialized:
            self._init_rope()

        # rotary_cos_sin: [max_pos, 2, rot_dim//2]
        rope_cache = self._rotary_cos_sin[positions]  # [B, seq, 2, rot_dim//2]
        rope_cos = rope_cache[..., 0, :]  # [B, seq, rot_dim//2]
        rope_sin = rope_cache[..., 1, :]
        if dtype is not None:
            rope_cos = rope_cos.to(dtype)
            rope_sin = rope_sin.to(dtype)
        return rope_cos, rope_sin

    def dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache: torch.Tensor,
        ctx_v_cache: torch.Tensor,
        ctx_cache_batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash draft forward with cross-attention over a pooled K/V buffer.

        All shapes are fixed so the forward is CUDA-graph compatible.

        Args:
            noise_embedding: [B, block_size, hidden_size]
            query_positions: [B, block_size]
            num_ctx_per_req: [B] — per-batch context length in the pool
            ctx_k_cache: [pool_batch, L, max_ctx+block_size, nkv, hd]
            ctx_v_cache: [pool_batch, L, max_ctx+block_size, nkv, hd]
            ctx_cache_batch_idx: [B] — slot index into the pool per batch entry
        Returns:
            [B * block_size, hidden_size]
        """
        from flash_attn import flash_attn_with_kvcache

        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()

        layer0 = self.model.layers[0]
        attn0 = layer0.self_attn
        q_size = attn0.q_size
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_heads_per_rank = attn0.num_heads
        num_kv_heads_per_rank = attn0.num_key_value_heads

        has_qk_norm = self._has_qk_norm
        is_bf16 = noise_embedding.dtype == torch.bfloat16
        use_fused_qk_norm_rope = self._use_fused_qk_norm_rope and is_bf16
        use_fused_rope = (_flashinfer_rope is not None and has_qk_norm
                          and is_bf16 and not use_fused_qk_norm_rope)

        B = noise_embedding.shape[0]
        block_size = noise_embedding.shape[1]

        hidden_states = noise_embedding  # [B, block_size, hidden]

        # Precompute RoPE cos/sin for the pure-PyTorch fallback path only.
        # The fused flashinfer path reads self._get_cos_sin_cache() inline.
        rope_dtype = hidden_states.dtype
        if not use_fused_rope:
            q_rope_cos, q_rope_sin = self._get_rope_cos_sin(query_positions,
                                                            dtype=rope_dtype)
        _rope = RotaryEmbedding.apply_rotary_pos_emb

        # cache_seqlens (BEFORE append). flash_attn appends block_size
        # k/v at cache_seqlens[i]..+block_size for batch i.
        cache_seqlens_i32 = num_ctx_per_req[:B].to(torch.int32)
        cache_batch_idx_i32 = ctx_cache_batch_idx.to(torch.int32)

        # Flatten query positions once for the fused QK-norm-RoPE kernel.
        query_positions_flat_i32 = query_positions.reshape(-1).to(torch.int32)

        residual = None

        for layer_idx, layer in enumerate(self.model.layers):
            attn_mod = layer.self_attn

            # Apply input_layernorm (flatten to 2D for norm, reshape back)
            hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if residual is None:
                residual = hidden_states.clone()
                hs_normed_flat = layer.input_layernorm(hs_flat)
            else:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed_flat, res_flat = layer.input_layernorm(
                    hs_flat, res_flat)
                residual = res_flat.reshape(B, block_size, -1)

            # QKV projection on normed query tokens (2D)
            qkv_query = attn_mod.qkv_proj(hs_normed_flat)  # [B*blk, qkv_size]

            if use_fused_qk_norm_rope:
                # One kernel does q_norm + k_norm + RoPE in-place on qkv.
                # Only safe when the drafter's rope params don't use YaRN /
                # long-rope / partial-rotary — otherwise fall back to the
                # shared-cache path below.
                attn_mod.apply_qk_norm_rope(qkv_query, query_positions_flat_i32)
                q_all_2d = qkv_query[:, :q_size]
                k_noise_2d = qkv_query[:, q_size:q_size + kv_size]
                v_noise_2d = qkv_query[:, q_size + kv_size:]
                Q_bshd = q_all_2d.reshape(B, block_size, num_heads_per_rank,
                                          head_dim)
                k_noise_bshd = k_noise_2d.reshape(B, block_size,
                                                  num_kv_heads_per_rank,
                                                  head_dim)
                v_noise_bshd = v_noise_2d.reshape(B, block_size,
                                                  num_kv_heads_per_rank,
                                                  head_dim)
            elif use_fused_rope:
                # Per-head RMSNorm on q/k (returns new contiguous tensors),
                # then flashinfer in-place RoPE sharing the same cos/sin cache
                # as precompute_context_kv.
                q = attn_mod.q_norm(qkv_query[:, :q_size].reshape(
                    -1, head_dim)).view(-1, q_size)
                k = attn_mod.k_norm(qkv_query[:,
                                              q_size:q_size + kv_size].reshape(
                                                  -1,
                                                  head_dim)).view(-1, kv_size)
                _flashinfer_rope(
                    query_positions_flat_i32,
                    q,
                    k,
                    head_dim,
                    self._get_cos_sin_cache(),
                    self._is_neox,
                )
                Q_bshd = q.view(B, block_size, num_heads_per_rank, head_dim)
                k_noise_bshd = k.view(B, block_size, num_kv_heads_per_rank,
                                      head_dim)
                v_noise_bshd = qkv_query[:, q_size + kv_size:].reshape(
                    B, block_size, num_kv_heads_per_rank, head_dim)
            else:
                qkv_query_3d = qkv_query.reshape(B, block_size, -1)
                q_all = qkv_query_3d[..., :q_size]
                k_noise_all = qkv_query_3d[..., q_size:q_size + kv_size]
                v_noise_all = qkv_query_3d[..., q_size + kv_size:]
                if has_qk_norm:
                    q_for_rope = attn_mod.q_norm(q_all.reshape(
                        -1, head_dim)).reshape(B, block_size, q_size)
                    k_noise_for_rope = attn_mod.k_norm(
                        k_noise_all.reshape(-1, head_dim)).reshape(
                            B, block_size, kv_size)
                else:
                    q_for_rope = q_all
                    k_noise_for_rope = k_noise_all
                Q = _rope(q_for_rope.reshape(B, block_size, num_heads_per_rank,
                                             head_dim).transpose(1, 2),
                          q_rope_cos,
                          q_rope_sin,
                          unsqueeze_dim=1,
                          is_neox=self._is_neox)
                k_noise_rope = _rope(k_noise_for_rope.reshape(
                    B, block_size, num_kv_heads_per_rank,
                    head_dim).transpose(1, 2),
                                     q_rope_cos,
                                     q_rope_sin,
                                     unsqueeze_dim=1,
                                     is_neox=self._is_neox)
                Q_bshd = Q.transpose(1, 2)
                k_noise_bshd = k_noise_rope.transpose(1, 2)
                v_noise_bshd = v_noise_all.reshape(B, block_size,
                                                   num_kv_heads_per_rank,
                                                   head_dim)

            # Per-layer view into the pooled ctx cache.
            # [pool_batch, max_ctx+block, nkv, hd]; flash_attn dereferences
            # each batch via cache_batch_idx, no gather.
            layer_k_cache = ctx_k_cache[:, layer_idx]
            layer_v_cache = ctx_v_cache[:, layer_idx]

            # flash_attn appends k_noise/v_noise in-place at
            # cache_seqlens[i]..+block_size for each batch i.
            out = flash_attn_with_kvcache(
                q=Q_bshd,
                k_cache=layer_k_cache,
                v_cache=layer_v_cache,
                k=k_noise_bshd,
                v=v_noise_bshd,
                cache_seqlens=cache_seqlens_i32,
                cache_batch_idx=cache_batch_idx_i32,
                causal=False,
            )
            attn_output = out.reshape(B * block_size, q_size)

            # o_proj (flat 2D, handles all-reduce internally)
            hidden_out = attn_mod.o_proj(attn_output)

            # Post-attention layernorm + MLP (flat 2D)
            res_flat = residual.reshape(-1, residual.shape[-1])
            hidden_out, res_flat = layer.post_attention_layernorm(
                hidden_out, res_flat)
            hidden_out = layer.mlp(hidden_out)

            hidden_states = hidden_out.reshape(B, block_size, -1)
            residual = res_flat.reshape(B, block_size, -1)

        # Final norm
        hidden_states_out, _ = self.model.norm(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            residual.reshape(-1, residual.shape[-1]))
        return hidden_states_out

    def forward(
        self,
        attn_metadata,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_out = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )

        return hidden_states_out, hidden_states_out


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
            case "deepseek_v3" | "deepseek_v32" | "glm_moe_dsa":
                from .modeling_deepseekv3 import DeepseekV3MTP
                mtp_layer = DeepseekV3MTP
            case "exaone_moe":
                from .modeling_exaone_moe import ExaoneMoeMTP
                mtp_layer = ExaoneMoeMTP
            case "nemotron_h" | "nemotron_h_puzzle":
                from .modeling_nemotron_h import NemotronHMTP
                mtp_layer = NemotronHMTP
            case "qwen3_next" | "qwen3_5_text" | "qwen3_5_moe_text":
                from .modeling_qwen3_next import Qwen3NextMTP
                mtp_layer = Qwen3NextMTP
            case _:
                raise ValueError(
                    f"Model type {model_type} not supported for MTP")

        spec_dec_mode = model_config.spec_config.spec_dec_mode
        assert spec_dec_mode.is_mtp_one_model()
        checkpoint_mtp_num_layers = model_config.pretrained_config.num_nextn_predict_layers
        if spec_dec_mode.is_mtp_eagle_one_model():
            mtp_num_layers = 1
            mtp_repeat_count = model_config.spec_config.max_draft_len
        else:
            mtp_num_layers = min(model_config.spec_config.max_draft_len,
                                 checkpoint_mtp_num_layers)
            mtp_repeat_count = 1

        moe_load_balancer_set_repeated_for_next_layer(mtp_repeat_count)

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
        elif model_type in ["deepseek_v3", "deepseek_v32", "glm_moe_dsa"]:
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
        elif model_type == "qwen3_next":
            from .modeling_qwen3_next import Qwen3NextMTP
            mtp_layer = Qwen3NextMTP(model_config, layer_idx, aux_stream_dict)
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
            case "deepseek_v3" | "deepseek_v32" | "glm_moe_dsa":
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
    elif spec_dec_mode.is_pard():
        return PARDForCausalLM(draft_config)
    elif spec_dec_mode.is_dflash():
        return DFlashForCausalLM(draft_config)
    elif spec_dec_mode.is_draft_target_one_model():
        return AutoModelForCausalLM.from_config(draft_config)
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
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        spec_config = getattr(model_config, 'spec_config', None)
        self.spec_config = spec_config
        if spec_config and spec_config.spec_dec_mode.use_one_engine():
            # Only create draft_model for modes MTP, Eagle3 (not SA)
            if not spec_config.spec_dec_mode.is_sa():
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
                    self.draft_config.extra_attrs = model_config.extra_attrs

                elif spec_config.spec_dec_mode.is_external_drafter():
                    self.draft_config = ModelConfig.from_pretrained(
                        model_config.spec_config.speculative_model,
                        trust_remote_code=True,
                        attn_backend=model_config.attn_backend,
                        moe_backend=model_config.moe_backend,
                        mapping=model_config.mapping,
                        spec_config=None,  # Avoid recursive spec-dec
                        max_num_tokens=model_config.max_num_tokens,
                        moe_max_num_tokens=model_config.moe_max_num_tokens)
                    self.draft_config.quant_config.kv_cache_quant_algo = \
                        model_config.quant_config.kv_cache_quant_algo
                    self.draft_config.extra_attrs = model_config.extra_attrs

                self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(
                    spec_config)

                self.draft_model = get_draft_model(model_config,
                                                   self.draft_config,
                                                   self.lm_head, self.model)
                if self.draft_model is not None:
                    self.epilogue.append(self.draft_model)
                if (spec_config.spec_dec_mode.is_parallel_draft()
                    ) and self.draft_model is not None:
                    self.draft_model.logits_processor = self.logits_processor

            # spec_worker is created for all one-engine modes (MTP, Eagle3, SA)
            self.spec_worker = get_spec_worker(
                model_config.spec_config,
                model_config,
                model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache)
            if self.spec_worker is not None:
                self.epilogue.append(self.spec_worker)
        self.layer_idx = -1

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
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

        if self.spec_worker is not None:
            # get logits
            logits = self.logits_processor.forward(
                hidden_states[spec_metadata.gather_ids],
                self.lm_head,
                attn_metadata,
                True,
            )

            spec_input_ids = input_ids
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if input_ids is not None:
                    # Slice along the first dimension
                    spec_input_ids = input_ids[:attn_metadata.num_tokens]
                if position_ids is not None:
                    spec_position_ids = _slice_spec_position_ids(
                        position_ids, attn_metadata.num_tokens)

            # get accepted tokens and next draft tokens
            return self.spec_worker(input_ids=spec_input_ids,
                                    position_ids=spec_position_ids,
                                    hidden_states=hidden_states,
                                    logits=logits,
                                    attn_metadata=attn_metadata,
                                    spec_metadata=spec_metadata,
                                    draft_model=self.draft_model,
                                    resource_manager=resource_manager)
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
        args = inspect.getfullargspec(self.draft_model.load_weights).args
        if "weight_mapper" in args:
            self.draft_model.load_weights(weights=weights,
                                          weight_mapper=weight_mapper)
        else:
            self.draft_model.load_weights(weights=weights)

        if self.spec_config and (
                not self.spec_config.spec_dec_mode.is_external_drafter()
                or self.spec_config.spec_dec_mode.is_dflash()):
            self.draft_model.load_weights_from_target_model(self)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False
