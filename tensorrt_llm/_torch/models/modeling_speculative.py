# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
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
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import moe_load_balancer_set_repeated_for_next_layer
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import (Linear, TensorParallelMode, WeightMode,
                              WeightsLoadingConfig)
from ..modules.mla import MLA
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding

try:
    from ..custom_ops import \
        flashinfer_apply_rope_with_cos_sin_cache_inplace as _flashinfer_rope
except ImportError:
    _flashinfer_rope = None
from ..pyexecutor.config_utils import (_is_sliding_attention_layer,
                                       get_layer_attention_window)
from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
from ..speculative import (SpecMetadata, get_spec_worker,
                           should_use_separate_draft_kv_cache)
from ..speculative.dflash_attention import (get_dflash_flash_attention,
                                            get_dflash_trtllm_gen_ops)
from ..utils import AuxStreamType
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM, TModel,
                             get_model_architecture, register_auto_model)

_SPECULATIVE_POSITION_HEADROOM = "_speculative_position_headroom"


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
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
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
            aux_stream_dict=aux_stream_dict,
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
                aux_stream_dict={AuxStreamType.Attention: aux_stream},
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
            "return_hidden_post_norm", False) or getattr(
                config, "norm_output", False)

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

        self._use_fc_norm = getattr(config, "fc_norm", False)
        if self._use_fc_norm:
            self.fc_norm = nn.ModuleList([
                RMSNorm(
                    hidden_size=self.hidden_size_in,
                    eps=config.rms_norm_eps,
                    dtype=config.torch_dtype,
                ) for _ in range(self.spec_config.num_capture_layers)
            ])
        else:
            self.fc_norm = None

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
            if self.model.fc_norm is not None:
                chunks = hidden_states.chunk(len(self.model.fc_norm), dim=-1)
                hidden_states = torch.cat([
                    norm(chunk)
                    for norm, chunk in zip(self.model.fc_norm, chunks)
                ],
                                          dim=-1)
            elif self.model._norm_before_fc:
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
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            assert self.embed_tokens is not None
            inputs_embeds = self.embed_tokens(input_ids).to(self.dtype)

        assert hidden_states is not None

        previous_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        if all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

        try:
            # NOTE: If hidden states from the target model have to be concatenated,
            # we expect that to happen outside the model definition. This helps us
            # avoid data-dependent control flow and gives us better CUDA graph
            # coverage.
            residual = None
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            hidden_states, residual = self.layers[0](
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=None,
                spec_metadata=spec_metadata)
        finally:
            if all_rank_num_tokens is not None:
                attn_metadata.all_rank_num_tokens = previous_all_rank_num_tokens

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
        draft_config_no_spec = replace(draft_config,
                                       spec_config=None,
                                       lm_head_gather_output=False)

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


def dspark_layer_window_size(use_swa: bool, swa_window: int, layer_types,
                             layer_idx: int) -> tuple[int, int]:
    """flash-attn ``window_size`` for one draft layer of the block decode.

    DSpark drafters (deepseek-ai/DeepSpec) run the draft block through HF
    attention with ``sliding_window`` set on 'sliding_attention' layers and
    is_causal=False. HF's flash path
    (transformers/modeling_flash_attention_utils.py) translates that to
    ``window_size = (sliding_window - 1, sliding_window - 1)``, i.e. each
    query attends keys within ``swa_window - 1`` KV-index distance on both
    sides. In the DFlash pool layout KV index == token position, so this
    limits draft queries to the most recent ``swa_window`` context tokens
    plus the (nearby) draft block. Full-attention layers and non-dspark
    drafters keep flash-attn's default ``(-1, -1)`` (no window).
    """
    if not use_swa:
        return (-1, -1)
    if layer_types is not None and layer_idx < len(layer_types) and \
            layer_types[layer_idx] != 'sliding_attention':
        return (-1, -1)
    return (swa_window - 1, swa_window - 1)


def dspark_markov_step_bias(prev_tokens: torch.Tensor, markov_w1: torch.Tensor,
                            markov_w2: torch.Tensor) -> torch.Tensor:
    """Vanilla Markov head logit bias for one intra-block draft step.

    Reference: DeepSpec ``VanillaMarkov`` (deepspec/modeling/dspark/
    markov_head.py): ``bias = markov_w2(markov_w1(prev_token))`` where
    markov_w1 is nn.Embedding(vocab, rank) and markov_w2 is
    nn.Linear(rank, vocab, bias=False). With both weights stored
    [vocab, rank] this is ``markov_w1[prev] @ markov_w2.T``.

    Args:
        prev_tokens: [B] long, previous token per request (draft vocab).
        markov_w1: [vocab, rank].
        markov_w2: [vocab_or_shard, rank] (rows may be a TP vocab shard).
    Returns:
        [B, vocab_or_shard] bias in the markov weights' dtype.
    """
    return F.linear(F.embedding(prev_tokens, markov_w1), markov_w2)


def dspark_markov_chain_logits(
    base_logits: torch.Tensor,
    first_prev_tokens: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    argmax_fn=None,
) -> torch.Tensor:
    """Apply the vanilla Markov intra-block bias across a drafted block.

    Reference: DeepSpec ``VanillaMarkov.sample_block_tokens`` at
    temperature 0: for step i, ``logits_i += bias(prev_i)`` with
    ``prev_0`` = the anchor token (last accepted token, block slot 0) and
    ``prev_{i>0}`` = the greedy token from step i-1's *biased* logits.
    llama.cpp PR #25173 implements the same greedy chain.

    Args:
        base_logits: [B, K, vocab_or_shard] shared-lm_head logits.
        first_prev_tokens: [B] long, anchor token ids (draft vocab).
        markov_w1 / markov_w2: see :func:`dspark_markov_step_bias`.
        argmax_fn: callable([B, vocab_or_shard]) -> [B] token ids in the
            full draft vocab; defaults to plain argmax. Workers pass a
            TP-aware argmax when the draft logits are vocab-sharded.
    Returns:
        [B, K, vocab_or_shard] biased logits. Greedy per-position argmax of
        the result reproduces the reference sampled chain exactly.
    """
    K = base_logits.shape[1]
    if K == 0:
        return base_logits
    prev = first_prev_tokens.long()
    steps = []
    for i in range(K):
        bias = dspark_markov_step_bias(prev, markov_w1, markov_w2)
        step_logits = base_logits[:, i] + bias.to(base_logits.dtype)
        steps.append(step_logits)
        if argmax_fn is not None:
            prev = argmax_fn(step_logits).long()
        else:
            prev = torch.argmax(step_logits, dim=-1)
    return torch.stack(steps, dim=1)


class DFlashForCausalLM(nn.Module):
    """Draft model wrapper for DFlash speculative decoding.

    DFlash uses cross-attention where Q comes from noise/query tokens and K/V
    come from the concatenation of target hidden states and noise hidden states.
    The target_hidden stays CONSTANT across all layers (no input_layernorm applied).

    Reference: https://arxiv.org/pdf/2602.06036
    """

    def __init__(self,
                 draft_config,
                 *,
                 dflash_attention_backend: str = 'VANILLA'):
        """Build the draft model, resolving its architecture from the draft config
        (falling back to a model_type-derived name when the checkpoint uses a
        custom DFlash architecture label)."""
        super().__init__()

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
        draft_config_no_spec = replace(draft_config,
                                       spec_config=None,
                                       lm_head_gather_output=False)

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
        self.dflash_attention_backend = dflash_attention_backend
        if self.dflash_attention_backend == 'VANILLA':
            self._dflash_flash_attention = get_dflash_flash_attention()
        elif self.dflash_attention_backend == 'TRTLLM':
            self._dflash_trtllm_gen_ops = get_dflash_trtllm_gen_ops()
        else:
            raise ValueError(
                "DFlash attention backend must be VANILLA or TRTLLM, got "
                f"{self.dflash_attention_backend!r}.")
        self._dflash_trtllm_gen_workspace = None
        self._dflash_trtllm_gen_counters = None
        self.register_buffer("_dflash_batch_indices", None, persistent=False)
        self.register_buffer("_dflash_block_offsets", None, persistent=False)
        self._dflash_trtllm_gen_device = None
        self._dflash_trtllm_gen_sm_count = None
        logger.info(
            f"DFlash draft model initialized with mask_token_id: {self.mask_token_id}, "
            f"target_layer_ids: {self.target_layer_ids}, block_size: {self.block_size}, "
            f"attention_backend: {self.dflash_attention_backend}")

        # DSpark drafters (DFlash + low-rank Markov head + confidence head,
        # arXiv 2607.05147; reference: deepseek-ai/DeepSpec). The weights-
        # independent drafter-forward semantics ARE implemented here:
        #   - vanilla Markov intra-block logit bias (applied by DFlashWorker
        #     through apply_markov_chain_logits),
        #   - sliding-window attention on 'sliding_attention' draft layers
        #     during the block decode (use_swa / swa_window_size),
        #   - the shift_label output convention (hidden state at block slot j
        #     predicts draft token j+1; slot 0 holds the anchor token).
        # Confidence-scheduled verification is NOT implemented yet: the
        # confidence_proj weights are loaded (for the follow-up MR) but never
        # used, and drafting always proposes the full K tokens.
        self._dspark_shift_label = bool(dflash_config.get('shift_label', False))
        self._dspark_use_swa = bool(dflash_config.get('use_swa', False))
        self._dspark_swa_window = int(
            dflash_config.get('swa_window_size', 0) or 0)
        self._dspark_markov_rank = int(dflash_config.get('markov_rank', 0) or 0)
        self._dspark_markov_head_type = str(
            dflash_config.get('markov_head_type', 'vanilla')
            or 'vanilla').lower()
        self._dspark_use_confidence_head = bool(
            dflash_config.get('use_confidence_head', False))
        # Plain None placeholders rather than nn.Parameter/buffer: most
        # DFlash checkpoints don't ship these heads, and their shapes
        # ([vocab, rank]) are checkpoint-dependent, so nothing is
        # pre-allocated. load_weights() fills them in only when the
        # checkpoint ships them; consumers treat None as "head absent".
        self.markov_w1 = None  # [vocab, rank] (nn.Embedding weight layout)
        self.markov_w2 = None  # [vocab, rank] (nn.Linear(rank->vocab) weight)
        self.confidence_proj_weight = None  # loaded, unused (follow-up MR)
        self.confidence_proj_bias = None

        if self._dspark_markov_rank > 0 and \
                self._dspark_markov_head_type != 'vanilla':
            raise ValueError(
                f"DFlash dspark drafter declares markov_head_type="
                f"'{self._dspark_markov_head_type}'; only 'vanilla' is "
                "supported (gated/rnn heads need per-step hidden features).")
        if self._dspark_use_swa and self._dspark_swa_window < 1:
            raise ValueError(
                "DFlash dspark drafter sets use_swa but swa_window_size="
                f"{dflash_config.get('swa_window_size')} is invalid.")
        # causal=true is only invalid under the dspark convention. Legacy
        # DFlash drafter configs (e.g. Laguna) also carry a causal field;
        # their causality is handled by the legacy decode path
        # (_sliding_layers_causal), so don't reject them here.
        is_dspark = (str(dflash_config.get('projector_type', '')
                         or '').lower() == 'dspark' or self._dspark_shift_label
                     or self._dspark_use_swa or self._dspark_markov_rank > 0
                     or self._dspark_use_confidence_head)
        if is_dspark and dflash_config.get('causal'):
            raise ValueError(
                "DFlash dspark drafter sets causal=true; the block decode "
                "only supports the non-causal dspark convention.")
        # Per-layer flash-attn window for the block decode, resolved once.
        num_draft_layers = getattr(pretrained_config, 'num_hidden_layers', 0)
        layer_types = getattr(pretrained_config, 'layer_types', None)
        self._dspark_layer_windows = [
            dspark_layer_window_size(self._dspark_use_swa,
                                     self._dspark_swa_window, layer_types, i)
            for i in range(num_draft_layers)
        ]
        if self._dspark_use_confidence_head:
            logger.warning(
                "DFlash dspark drafter declares use_confidence_head; "
                "confidence-scheduled verification is not implemented yet "
                "(confidence_proj weights are loaded but unused, drafting "
                "always proposes the full K tokens).")

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
        self._num_heads = 0
        self._head_dim = 0
        self._num_kv_heads = 0
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False
        # Laguna-specific draft-layer behaviors, disabled by default so generic
        # DFlash drafters keep the original contract (no context input_layernorm,
        # non-causal block attention). Subclasses opt in.
        self._context_input_layernorm = False
        self._sliding_layers_causal = False
        self._warn_inferred_attention_windows()

    @staticmethod
    def _rope_signature(attn):
        """Return the effective RoPE configuration used by an attention layer."""
        if attn.rotary_emb is not None:
            return (
                attn.rotary_emb.rope_params,
                attn.rotary_emb.head_dim,
                attn.rotary_emb.is_neox,
            )
        if attn.pos_embd_params is not None:
            return (
                attn.pos_embd_params.rope,
                attn.head_dim,
                attn.pos_embd_params.is_neox,
            )
        return None

    def _validate_uniform_rope(self):
        """Check that all draft layers can safely share one RoPE cache."""
        if len(self.model.layers) == 0:
            raise ValueError("DFlash requires at least one draft model layer.")

        signatures = [
            self._rope_signature(layer.self_attn) for layer in self.model.layers
        ]

        mismatched_layers = [
            layer_idx
            for layer_idx, signature in enumerate(signatures[1:], start=1)
            if signature != signatures[0]
        ]
        if mismatched_layers:
            layer_types = getattr(self.config, 'layer_types', None)
            raise ValueError(
                "DFlash shares one RoPE cache across draft layers, but layers "
                f"{mismatched_layers} have a different effective RoPE "
                f"configuration from layer 0. layer_types={layer_types}.")

    def _init_rope(self):
        """Initialize RoPE from the draft model's attention configuration.

        Reuses the existing RotaryEmbedding infrastructure which correctly
        handles all RoPE variants (standard, YaRN, scaled, etc.).
        """
        # The flattened context-KV path shares layer 0's RoPE cache.
        self._validate_uniform_rope()
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

    def project_target_hidden(self,
                              hidden_states: torch.Tensor) -> torch.Tensor:
        """Project captured target hidden states into the draft hidden space.

        Generic DFlash: fc then hidden_norm. Subclasses (e.g. Laguna) may
        normalize the per-aux features first by overriding this method.
        """
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        return self.hidden_norm(self.fc(hidden_states))

    @property
    def has_markov_head(self) -> bool:
        return self._dspark_markov_rank > 0 and self.markov_w1 is not None

    def apply_markov_chain_logits(
            self,
            base_logits: torch.Tensor,
            first_prev_tokens: torch.Tensor,
            argmax_fn=None,
            vocab_slice: slice | None = None) -> torch.Tensor:
        """Apply the dspark vanilla-Markov intra-block bias to block logits.

        No-op (returns ``base_logits`` unchanged) for non-dspark drafters.
        See :func:`dspark_markov_chain_logits` for the semantics; when
        ``base_logits`` is a TP vocab shard, the caller must pass this
        rank's ``vocab_slice`` (to shard the markov_w2 rows identically)
        and an ``argmax_fn`` returning full-vocab token ids — DFlashWorker
        handles both.
        """
        if not self.has_markov_head:
            return base_logits
        markov_w2 = self.markov_w2 if vocab_slice is None else \
            self.markov_w2[vocab_slice]
        return dspark_markov_chain_logits(base_logits,
                                          first_prev_tokens,
                                          self.markov_w1,
                                          markov_w2,
                                          argmax_fn=argmax_fn)

    def _post_attention_gate(self, attn_output, gate_input, attn_mod, num_heads,
                             head_dim):
        """Hook applied to the block-attention output before o_proj.

        No-op for generic DFlash; overridden by drafters that gate (e.g. Laguna).
        """
        return attn_output

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load weights into the DFlash draft model.

        DFlash checkpoints differ from standard HF format:
        - Layer weights lack the 'model.' prefix (e.g., 'layers.0...' not 'model.layers.0...')
        - Extra DFlash-specific weights: 'fc.weight', 'hidden_norm.weight'
        - Missing embed_tokens and lm_head (shared with target model)
        """
        # Laguna DFlash checkpoints may ship a fused self_attn.qkv_proj; the draft
        # loader expects split q/k/v (a fused key is silently dropped otherwise).
        if any(k.endswith('self_attn.qkv_proj.weight') for k in weights):
            for attr in ('num_attention_heads_per_layer',
                         'num_key_value_heads_per_layer'):
                per_layer = getattr(self.config, attr, None)
                if per_layer is not None and len(set(per_layer)) > 1:
                    raise ValueError(
                        "DFlash load_weights() splits the fused qkv_proj using "
                        "the global head count, but the drafter has heterogeneous "
                        f"{attr} {sorted(set(per_layer))}; per-layer qkv splitting "
                        "is required for this checkpoint.")
            head_dim = getattr(
                self.config, 'head_dim',
                self.config.hidden_size // self.config.num_attention_heads)
            num_kv_heads = getattr(self.config, 'num_key_value_heads',
                                   self.config.num_attention_heads)
            q = self.config.num_attention_heads * head_dim
            kv = num_kv_heads * head_dim
            split = {}
            for k, v in weights.items():
                if k.endswith('self_attn.qkv_proj.weight'):
                    b = k[:-len('qkv_proj.weight')]
                    split[b + 'q_proj.weight'] = v[:q]
                    split[b + 'k_proj.weight'] = v[q:q + kv]
                    split[b + 'v_proj.weight'] = v[q + kv:]
                else:
                    split[k] = v
            weights = split

        # DSpark head weights: keep them out of the backbone remap (they'd
        # get a 'model.' prefix and be dropped by allow_partial_loading).
        # markov_w1/markov_w2 drive the intra-block logit bias; the
        # confidence_proj weights are loaded for the confidence-scheduling
        # follow-up MR but are not used yet.
        dspark_keys = ('markov_w1.weight', 'markov_w2.weight',
                       'confidence_proj.weight', 'confidence_proj.bias')
        dspark_weights = {k: weights[k] for k in dspark_keys if k in weights}
        if dspark_weights:
            weights = {
                k: v
                for k, v in weights.items() if k not in dspark_weights
            }
        if self._dspark_markov_rank > 0:
            vocab = self.config.vocab_size
            rank = self._dspark_markov_rank
            for k in ('markov_w1.weight', 'markov_w2.weight'):
                if k not in dspark_weights:
                    raise ValueError(
                        f"DFlash dspark drafter declares markov_rank="
                        f"{self._dspark_markov_rank} but the checkpoint is "
                        f"missing {k}.")
                if tuple(dspark_weights[k].shape) != (vocab, rank):
                    raise ValueError(
                        f"DFlash dspark {k} has shape "
                        f"{tuple(dspark_weights[k].shape)}, expected "
                        f"[vocab, markov_rank] = ({vocab}, {rank}).")
            self.markov_w1 = dspark_weights['markov_w1.weight'].to('cuda')
            self.markov_w2 = dspark_weights['markov_w2.weight'].to('cuda')
        if 'confidence_proj.weight' in dspark_weights:
            self.confidence_proj_weight = dspark_weights[
                'confidence_proj.weight'].to('cuda')
        if 'confidence_proj.bias' in dspark_weights:
            self.confidence_proj_bias = dspark_weights[
                'confidence_proj.bias'].to('cuda')

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
        if getattr(self, '_input_ln_eps', None) is not None:
            ph = projected_hidden.float()
            ph = ph * torch.rsqrt(
                ph.pow(2).mean(-1, keepdim=True) + self._input_ln_eps)
            projected_hidden = ph.to(weight_dtype)
        elif projected_hidden.dtype != weight_dtype:
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
        num_heads = attn0.num_heads
        num_kv_heads = attn0.num_key_value_heads
        # Head counts are read from layer 0 here and in dflash_forward; assert
        # uniformity (the target uses per-layer heads, the drafter does not).
        for a in layers_attn[1:]:
            assert (
                a.q_size == q_size and a.kv_size == kv_size
                and a.head_dim == head_dim and a.num_heads == num_heads
                and a.num_key_value_heads == num_kv_heads), (
                    "DFlash fused KV requires all drafter layers to share "
                    "q_size / kv_size / head_dim / num_heads / num_kv_heads.")

        has_k_norm = [hasattr(a, 'k_norm') for a in layers_attn]
        assert all(has_k_norm) or not any(has_k_norm), (
            "DFlash fused KV requires either all or no drafter layers to have k_norm."
        )

        kv_weights = [
            a.qkv_proj.weight[q_size:q_size + 2 * kv_size] for a in layers_attn
        ]
        # Fold each drafter layer's input_layernorm weight into its KV projection
        # so context K/V match the query path. vLLM laguna_dflash applies
        # layer.input_layernorm to context states before KV; RMSNorm gives
        # (x_hat * w) @ Wkv.T == x_hat @ (Wkv * w).T, and the shared 1/rms(x) is
        # applied to projected_hidden in precompute_context_kv.
        dlayers = self.model.layers
        if self._context_input_layernorm and all(
                hasattr(dl, 'input_layernorm') for dl in dlayers):
            eps_set = {
                getattr(dl.input_layernorm, 'variance_epsilon',
                        getattr(self.config, 'rms_norm_eps', 1e-6))
                for dl in dlayers
            }
            assert len(eps_set) == 1, (
                "DFlash fused context input_layernorm needs all drafter layers "
                f"to share variance_epsilon; got {sorted(eps_set)}")
            self._input_ln_eps = eps_set.pop()
            folded = []
            for w, dl in zip(kv_weights, dlayers):
                scale = dl.input_layernorm.weight.data
                if getattr(dl.input_layernorm, 'use_gemma', False):
                    scale = scale + 1
                folded.append(w * scale[None, :].to(w.dtype))
            kv_weights = folded
        else:
            self._input_ln_eps = None
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
        self._num_heads = num_heads
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

    def _warn_inferred_attention_windows(self) -> None:
        """Warn once at initialization when checkpoint metadata enables SWA."""
        if getattr(self.config, 'use_sliding_window', None) is not None:
            return

        num_hidden_layers = getattr(self.config, 'num_hidden_layers', None)
        if num_hidden_layers is None:
            num_hidden_layers = len(self.model.layers)
        layers_by_window = {}
        for layer_idx in range(num_hidden_layers):
            window = get_layer_attention_window(self.config, layer_idx)
            if window is not None:
                layers_by_window.setdefault(window, []).append(layer_idx)

        for window, layer_indices in layers_by_window.items():
            logger.warning(
                "DFlash inferred pooled-context sliding-window attention from "
                f"checkpoint config for draft layers {layer_indices}: "
                f"window={window}. Context attention is truncated to {window} "
                "tokens for these layers; if the drafter expects full context, "
                "acceptance rate may drop. Set use_sliding_window explicitly "
                "to confirm or disable windowing.")

    def _get_attention_mask_args(self, layer_idx):
        """Return FlashAttention causal and local-window arguments for a layer."""
        layer_types = getattr(self.config, 'layer_types', None)
        is_sliding_layer = False
        if layer_types:
            layer_type = layer_types[layer_idx % len(layer_types)]
            is_sliding_layer = _is_sliding_attention_layer(layer_type)

        sliding_window = get_layer_attention_window(self.config, layer_idx)
        is_sliding_layer = is_sliding_layer or sliding_window is not None
        if not is_sliding_layer:
            return False, (-1, -1)

        causal = self._sliding_layers_causal or sliding_window is not None
        if sliding_window is None:
            # Legacy drafters without an explicit window preserve their prior
            # non-windowed behavior.
            return causal, (-1, -1)
        # FlashAttention's bounds are inclusive: W tokens are current + W-1 left.
        return causal, (sliding_window - 1, 0)

    def _prepare_dflash_trtllm_gen_buffers(
        self,
        dtype: torch.dtype,
        device: torch.device,
        max_batch_size: int,
        block_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        trtllm_gen_ops = self._dflash_trtllm_gen_ops
        workspace_bytes = trtllm_gen_ops.get_workspace_size(
            dtype=dtype,
            num_tokens=max_batch_size * block_size,
            num_gen_tokens=max_batch_size * block_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            max_num_requests=max_batch_size,
            rotary_embedding_dim=0,
            fp8_context_fmha=False,
        )
        device = torch.device(device)
        is_capturing = torch.cuda.is_current_stream_capturing()
        if self._dflash_trtllm_gen_device != device:
            if is_capturing:
                raise RuntimeError(
                    "DFlash TRTLLM-Gen buffers must be prepared on the current "
                    "device before CUDA graph capture.")
            self._dflash_trtllm_gen_device = device
            self._dflash_trtllm_gen_sm_count = (
                torch.cuda.get_device_properties(device).multi_processor_count)

        workspace = self._dflash_trtllm_gen_workspace
        workspace_needs_allocation = (
            workspace is None or workspace.device != device
            or workspace.numel() * workspace.element_size() < workspace_bytes)
        if workspace_needs_allocation:
            if is_capturing:
                raise RuntimeError(
                    "The DFlash TRTLLM-Gen workspace must be allocated at the "
                    "required size before CUDA graph capture.")
            self._dflash_trtllm_gen_workspace = torch.empty(workspace_bytes,
                                                            dtype=torch.uint8,
                                                            device=device)

        sm_count = self._dflash_trtllm_gen_sm_count
        counter_bytes = trtllm_gen_ops.get_multi_ctas_kv_counter_size(
            num_heads, max_batch_size, sm_count)
        counters = self._dflash_trtllm_gen_counters
        counters_need_allocation = (counters is None
                                    or counters.device != device
                                    or counters.numel() *
                                    counters.element_size() < counter_bytes)
        if counters_need_allocation:
            if is_capturing:
                raise RuntimeError(
                    "The DFlash TRTLLM-Gen counter buffer must be allocated at "
                    "the required size before CUDA graph capture.")
            self._dflash_trtllm_gen_counters = torch.zeros(counter_bytes,
                                                           dtype=torch.uint8,
                                                           device=device)

        append_batch_indices = self._dflash_batch_indices
        block_offsets = self._dflash_block_offsets
        static_indices_need_allocation = (
            append_batch_indices is None or block_offsets is None
            or append_batch_indices.device != device
            or block_offsets.device != device
            or append_batch_indices.size(0) < max_batch_size
            or append_batch_indices.size(1) != block_size
            or block_offsets.numel() != block_size)
        if static_indices_need_allocation:
            if is_capturing:
                raise RuntimeError(
                    "DFlash TRTLLM-Gen index buffers must be allocated at the "
                    "required size before CUDA graph capture.")
            self._dflash_batch_indices = (torch.arange(
                max_batch_size, dtype=torch.int32,
                device=device).view(-1, 1).expand(-1, block_size).contiguous())
            self._dflash_block_offsets = torch.arange(block_size,
                                                      dtype=torch.int32,
                                                      device=device)

    def dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache: torch.Tensor,
        ctx_v_cache: torch.Tensor,
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache: Optional[torch.Tensor] = None,
        ctx_page_table: Optional[torch.Tensor] = None,
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
        if self.dflash_attention_backend == 'TRTLLM':
            if ctx_kv_cache is None or ctx_page_table is None:
                raise RuntimeError(
                    "DFlash TRTLLM-Gen requires a paged context cache and page table."
                )
            trtllm_gen_ops = self._dflash_trtllm_gen_ops
        elif self.dflash_attention_backend == 'VANILLA':
            flash_attention = self._dflash_flash_attention
        else:
            raise ValueError(
                "DFlash attention backend must be VANILLA or TRTLLM, got "
                f"{self.dflash_attention_backend!r}.")

        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()

        layer0 = self.model.layers[0]
        attn0 = layer0.self_attn
        q_size = attn0.q_size
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        # Uniformity across layers is asserted in _build_fused_kv_buffers (above).
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

        if self.dflash_attention_backend == 'TRTLLM':
            max_batch_size = ctx_page_table.size(0)
            self._prepare_dflash_trtllm_gen_buffers(
                hidden_states.dtype,
                hidden_states.device,
                max_batch_size,
                block_size,
                num_heads_per_rank,
                num_kv_heads_per_rank,
                head_dim,
            )
            block_tables = ctx_page_table.index_select(
                0, cache_batch_idx_i32.long())
            pages_per_slot = block_tables.size(1)
            page_size = ctx_kv_cache.size(-2)
            kv_indices = block_tables.flatten()
            kv_indptr = torch.arange(
                0,
                (B + 1) * pages_per_slot,
                pages_per_slot,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            seq_lens_after = cache_seqlens_i32 + block_size
            kv_last_page_len = ((seq_lens_after - 1) % page_size) + 1
            batch_indices = self._dflash_batch_indices
            append_batch_indices = batch_indices[:B].reshape(-1)
            append_positions = (
                cache_seqlens_i32.view(-1, 1) +
                self._dflash_block_offsets).reshape(-1).contiguous()

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
            causal, window_size = self._get_attention_mask_args(layer_idx)
            dspark_window = (self._dspark_layer_windows[layer_idx] if layer_idx
                             < len(self._dspark_layer_windows) else (-1, -1))
            if dspark_window != (-1, -1):
                window_size = dspark_window
            if self.dflash_attention_backend == 'TRTLLM':
                layer_cache = ctx_kv_cache[layer_idx]
                trtllm_gen_ops.append_paged_kv_cache(
                    append_key=k_noise_bshd.reshape(-1, num_kv_heads_per_rank,
                                                    head_dim).contiguous(),
                    append_value=v_noise_bshd.reshape(-1, num_kv_heads_per_rank,
                                                      head_dim).contiguous(),
                    batch_indices=append_batch_indices,
                    positions=append_positions,
                    paged_kv_cache=layer_cache,
                    kv_indices=kv_indices,
                    kv_indptr=kv_indptr,
                    kv_last_page_len=kv_last_page_len,
                    kv_layout="HND",
                )
                out = torch.empty_like(Q_bshd)
                q_flat = Q_bshd.reshape(-1, num_heads_per_rank, head_dim)
                out_flat = out.reshape(-1, num_heads_per_rank, head_dim)
                window_left = window_size[0]
                if causal:
                    trtllm_gen_ops.batch_decode_with_kv_cache(
                        query=q_flat,
                        kv_cache=(layer_cache[:, 0], layer_cache[:, 1]),
                        workspace_buffer=self._dflash_trtllm_gen_workspace,
                        block_tables=block_tables,
                        seq_lens=seq_lens_after,
                        max_seq_len=pages_per_slot * page_size,
                        bmm1_scale=head_dim**-0.5,
                        bmm2_scale=1.0,
                        window_left=window_left,
                        out=out_flat,
                        sinks=None,
                        enable_pdl=False,
                        kv_layout="HND",
                        backend="trtllm-gen",
                        q_len_per_req=block_size,
                        max_q_len=None,
                        cum_seq_lens_q=None,
                        kv_cache_sf=None,
                        uses_shared_paged_kv_idx=True,
                        bmm1_scale_log2=None,
                        multi_ctas_kv_counter_buffer=self.
                        _dflash_trtllm_gen_counters,
                    )
                else:
                    cum_seq_lens_q = torch.arange(
                        0,
                        (B + 1) * block_size,
                        block_size,
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )
                    cum_seq_lens_kv = torch.cat((
                        torch.zeros(1,
                                    dtype=torch.int32,
                                    device=hidden_states.device),
                        seq_lens_after.cumsum(0, dtype=torch.int32),
                    ))
                    trtllm_gen_ops.batch_context_with_kv_cache(
                        query=q_flat,
                        kv_cache=(layer_cache[:, 0], layer_cache[:, 1]),
                        workspace_buffer=self._dflash_trtllm_gen_workspace,
                        block_tables=block_tables,
                        seq_lens=seq_lens_after,
                        max_q_len=block_size,
                        max_kv_len=pages_per_slot * page_size,
                        bmm1_scale=head_dim**-0.5,
                        bmm2_scale=1.0,
                        batch_size=B,
                        cum_seq_lens_q=cum_seq_lens_q,
                        cum_seq_lens_kv=cum_seq_lens_kv,
                        window_left=window_left,
                        out=out_flat,
                        sinks=None,
                        enable_pdl=False,
                        kv_layout="HND",
                        kv_cache_sf=None,
                        uses_shared_paged_kv_idx=True,
                        causal=False,
                        multi_ctas_kv_counter_buffer=self.
                        _dflash_trtllm_gen_counters,
                    )
            else:  # VANILLA, validated before entering the layer loop.
                layer_k_cache = ctx_k_cache[:, layer_idx]
                layer_v_cache = ctx_v_cache[:, layer_idx]
                out = flash_attention(
                    q=Q_bshd,
                    k_cache=layer_k_cache,
                    v_cache=layer_v_cache,
                    k=k_noise_bshd,
                    v=v_noise_bshd,
                    cache_seqlens=cache_seqlens_i32,
                    cache_batch_idx=cache_batch_idx_i32,
                    causal=causal,
                    window_size=window_size,
                )
            attn_output = out.reshape(B * block_size, q_size)

            # Per-drafter post-attention gate (no-op for generic DFlash; Laguna
            # applies per-head softplus g_proj gating). gate input is the
            # input_layernorm output (the attention input).
            attn_output = self._post_attention_gate(attn_output, hs_normed_flat,
                                                    attn_mod,
                                                    num_heads_per_rank,
                                                    head_dim)

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
        """Run the draft model and return (hidden_states, hidden_states) for the
        speculative-decoding contract."""
        hidden_states_out = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )

        return hidden_states_out, hidden_states_out


class DFlashLagunaForCausalLM(DFlashForCausalLM):
    """Laguna DFlash drafter.

    The generic block decode lives in DFlashForCausalLM; this subclass supplies
    the Laguna draft-layer specifics: per-head g_proj softplus gating and the
    per-aux fc_norm applied to captured target features before fc.
    """

    @staticmethod
    def _normalize_config(config: PretrainedConfig) -> None:
        """Fill TRT-LLM Laguna defaults missing from dense DFlash drafts."""
        if getattr(config, "num_experts", None) is None:
            config.num_experts = 0
        if getattr(config, "mlp_layer_types", None) is None:
            config.mlp_layer_types = ["dense"] * config.num_hidden_layers
        if getattr(config, "block_size", None) is None:
            dflash_config = getattr(config, "dflash_config", {})
            if isinstance(dflash_config, dict):
                config.block_size = dflash_config.get("block_size", None)

    def __init__(self,
                 draft_config,
                 *,
                 dflash_attention_backend: str = 'VANILLA'):
        """Pin the Laguna draft-layer class and enable Laguna-specific behaviors
        (context input_layernorm, causal sliding blocks); reject non-per-head
        gating."""
        # The checkpoint labels itself with the vLLM name (model_type "llama");
        # remap to the Laguna architecture so TRT-LLM builds the Laguna layers.
        draft_config.pretrained_config.architectures = ["LagunaForCausalLM"]
        self._normalize_config(draft_config.pretrained_config)
        super().__init__(
            draft_config,
            dflash_attention_backend=dflash_attention_backend,
        )
        self._context_input_layernorm = True
        self._sliding_layers_causal = True
        gating = getattr(self.config, 'gating', True)
        if gating not in (True, 'per-head'):
            raise NotImplementedError(
                f"Laguna DFlash drafter supports per-head gating only, "
                f"got gating={gating!r}")

    def load_weights(self, weights, weight_mapper=None, **kwargs):
        """Build the per-aux ``fc_norm`` from the drafter's ``aux_hidden_norms.*``
        weights, then defer the remaining weights to the base loader."""
        aux_keys = sorted(
            (k for k in weights if k.startswith('aux_hidden_norms.')),
            key=lambda k: int(k.split('.')[1]))
        if not aux_keys:
            raise ValueError(
                "Laguna DFlash checkpoint is missing aux_hidden_norms.* weights"
            )
        weights = dict(weights)
        eps = getattr(self.config, 'rms_norm_eps', 1e-6)
        norms = []
        for k in aux_keys:
            w = weights.pop(k)
            norm = nn.RMSNorm(w.shape[0],
                              eps=eps,
                              device='cuda',
                              elementwise_affine=True,
                              dtype=w.dtype)
            norm.weight.data.copy_(w)
            norms.append(norm)
        self.fc_norm = nn.ModuleList(norms)
        super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)

    def project_target_hidden(self, hidden_states):
        """Project captured target features to the draft width: apply the per-aux
        ``fc_norm`` to each hidden chunk, then ``fc`` + ``hidden_norm``."""
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        fc_norm = getattr(self, 'fc_norm', None)
        if fc_norm is not None:
            chunks = hidden_states.chunk(len(fc_norm), dim=-1)
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(fc_norm, chunks)], dim=-1)
        return self.hidden_norm(self.fc(hidden_states))

    def _post_attention_gate(self, attn_output, gate_input, attn_mod, num_heads,
                             head_dim):
        """Apply Laguna's per-head softplus output gate (``g_proj``) to the
        attention output; a no-op when the layer has no ``g_proj``."""
        g_proj = getattr(attn_mod, 'g_proj', None)
        if g_proj is None:
            return attn_output
        gate = F.softplus(g_proj(gate_input).float()).to(attn_output.dtype)
        return (attn_output.unflatten(-1, (num_heads, head_dim)) *
                gate.unsqueeze(-1)).flatten(-2)


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
            case "inkling" | "inkling_text" | "inkling_mm_model":
                from .modeling_inkling import InklingMTPBlock
                mtp_layer = InklingMTPBlock
            case "step3p7" | "step3p5":
                from .modeling_step3p7 import Step3p7MTP
                mtp_layer = Step3p7MTP
            case "deepseek_v4":
                from .modeling_deepseekv4 import DeepseekV4MTP
                mtp_layer = DeepseekV4MTP
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


def external_drafter_config_kwargs(model_config, spec_config) -> dict:
    """`ModelConfig.from_pretrained` kwargs for a one-model external drafter.

    The drafter is a separate checkpoint, so it gets its own `ModelConfig`; the
    kwargs below are the execution-layout properties it must inherit from the
    target engine it runs inside.

    `moe_load_balancer` is propagated for DSpark ONLY. DSpark's draft stages are
    full DeepSeek-V4 blocks sharing the target's expert topology and layer-index
    namespace (`layer_idx = num_hidden_layers + stage_id`), so they can register
    into the target's EPLB manager. Other external drafters (PARD, DFlash,
    draft-target) are independent checkpoints whose expert topology and layer
    numbering need not match the target's, and whose EPLB configs would therefore
    be keyed against a different namespace -- do not generalize this without
    designing a per-drafter EPLB config domain and layer identity first.
    """
    kwargs = dict(
        trust_remote_code=True,
        attn_backend=model_config.attn_backend,
        moe_backend=model_config.moe_backend,
        mapping=model_config.mapping,
        spec_config=None,  # Avoid recursive spec-dec
        max_num_tokens=model_config.max_num_tokens,
        moe_max_num_tokens=model_config.moe_max_num_tokens,
    )
    if spec_config.spec_dec_mode.is_dspark():
        kwargs["moe_load_balancer"] = model_config.moe_load_balancer
    return kwargs


def get_draft_model(model_config, draft_config, lm_head, model):
    """Construct the draft model for the configured speculative-decoding mode
    (EAGLE3 / MTP / PARD / DFlash). The DFlash branch selects the Laguna drafter
    by detecting its architecture in the draft checkpoint's own config."""
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

    elif model_config.spec_config.uses_external_draft_model:
        if draft_config is None:
            raise ValueError(
                "MTP speculative decoding with an external draft model requires "
                "its model config.")
        return AutoModelForCausalLM.from_config(draft_config)
    elif spec_dec_mode.is_mtp_one_model():
        return MTPForCausalLM(model_config,
                              model_config.pretrained_config.num_hidden_layers,
                              lm_head, model)
    elif spec_dec_mode.is_mtp_eagle():
        return MTPDraftModelForCausalLM(model_config)
    elif spec_dec_mode.is_pard():
        return PARDForCausalLM(draft_config)
    elif spec_dec_mode.is_dflash():
        draft_arches = getattr(draft_config.pretrained_config, "architectures",
                               None) or []
        dflash_attention_backend = model_config.spec_config.attention_backend
        if any("Laguna" in arch for arch in draft_arches):
            return DFlashLagunaForCausalLM(
                draft_config,
                dflash_attention_backend=dflash_attention_backend,
            )
        return DFlashForCausalLM(
            draft_config,
            dflash_attention_backend=dflash_attention_backend,
        )
    elif spec_dec_mode.is_dspark():
        # Lazy import to avoid a cycle (modeling_dspark -> modeling_deepseekv4 ->
        # modeling_speculative). The DSpark draft reuses the target's aux streams.
        # The draft stage count (n_mtp_layers) is not in the HF config, so derive
        # it from the checkpoint's mtp.* namespace.
        from .modeling_dspark import (DSparkForCausalLM, count_dspark_stages,
                                      validate_dspark_eplb_layer_base)
        num_stages = count_dspark_stages(
            model_config.spec_config.speculative_model)
        validate_dspark_eplb_layer_base(model_config, draft_config)
        return DSparkForCausalLM(
            draft_config,
            getattr(model, "aux_stream_dict", None),
            num_stages=num_stages,
            block_size=model_config.spec_config.block_size,
        )
    elif spec_dec_mode.is_draft_target_one_model():
        # Keep the draft LM head vocab-sharded so greedy draft sampling uses the
        # lighter TP gather (see SpecWorkerBase.greedy_sample_draft_with_tp_gather).
        was_frozen = draft_config._frozen
        draft_config._frozen = False
        draft_config.lm_head_gather_output = False
        draft_config._frozen = was_frozen
        return AutoModelForCausalLM.from_config(draft_config)
    else:
        raise NotImplementedError(
            f"get_draft_model does not support speculative decoding mode {spec_dec_mode}."
        )


class SpecDecOneEngineForCausalLM(DecoderModelForCausalLM[TModel, TConfig],
                                  Generic[TModel, TConfig]):

    def __init__(self,
                 model: TModel,
                 model_config: ModelConfig[TConfig],
                 hidden_size: int | None = None,
                 vocab_size: int | None = None) -> None:
        # Composite configs (e.g. vision-language wrappers) may not expose
        # hidden_size/vocab_size at the top level; callers can pass the
        # text-config values explicitly.
        if hidden_size is None:
            hidden_size = model_config.pretrained_config.hidden_size
        if vocab_size is None:
            vocab_size = model_config.pretrained_config.vocab_size
        super().__init__(model,
                         config=model_config,
                         hidden_size=hidden_size,
                         vocab_size=vocab_size)
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

                elif spec_config.uses_external_draft_model:
                    self.draft_config = ModelConfig.from_pretrained(
                        spec_config.speculative_model,
                        trust_remote_code=True,
                        attn_backend=model_config.attn_backend,
                        moe_backend=model_config.moe_backend,
                        mapping=model_config.mapping,
                        spec_config=None,
                        max_num_tokens=model_config.max_num_tokens,
                        moe_max_num_tokens=model_config.moe_max_num_tokens)
                    self.draft_config.quant_config.kv_cache_quant_algo = \
                        model_config.quant_config.kv_cache_quant_algo
                    self.draft_config.extra_attrs = dict(
                        model_config.extra_attrs)
                    self.draft_config.extra_attrs[
                        _SPECULATIVE_POSITION_HEADROOM] = (
                            2 * spec_config.tokens_per_gen_step)

                elif spec_config.spec_dec_mode.is_external_drafter():
                    self.draft_config = ModelConfig.from_pretrained(
                        model_config.spec_config.speculative_model,
                        **external_drafter_config_kwargs(
                            model_config, spec_config))
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
                # Cache the static draft->target vocab map now that the draft
                # model is loaded, so workers read self._d2t instead of probing
                # draft_model.model.d2t on every forward.
                self.spec_worker.set_draft_model(self.draft_model)
                self.epilogue.append(self.spec_worker)
        self.layer_idx = -1

    def setup_aliases(self) -> None:
        if (self.draft_model is not None
                and getattr(self.draft_model, "shares_target_kv_cache", False)):
            self.draft_model.load_weights_from_target_model(self)

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

            # VLM wrappers (e.g. Qwen3VLModelBase) replace input_ids with
            # fused inputs_embeds; fall back to the pre-fusion token IDs
            # they forward via `orig_input_ids` so MTP / Eagle drafters
            # can still access the prompt tokens.
            spec_input_ids = input_ids if input_ids is not None else kwargs.get(
                "orig_input_ids")
            spec_position_ids = position_ids
            if attn_metadata.padded_num_tokens is not None:
                if spec_input_ids is not None:
                    # Slice along the first dimension
                    spec_input_ids = spec_input_ids[:attn_metadata.num_tokens]
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

    def mtp_head_module_names(self) -> List[str]:
        """Names of the MTP heads under every alias they are reachable by.

        One-model MTP registers the same head objects twice: under
        ``draft_model.mtp_layers.{h}`` and, after the target model extends its
        layer list, under ``model.layers.{num_hidden_layers + h}``. A load that
        wants to leave the heads untouched has to exclude both aliases.
        """
        mtp_layers = getattr(self.draft_model, "mtp_layers", None)
        if not mtp_layers:
            return []
        head_ids = {id(layer) for layer in mtp_layers}
        return [
            name for name, module in self.named_modules(remove_duplicate=False)
            if name and id(module) in head_ids
        ]

    def load_weights(self,
                     weights: Dict,
                     weight_mapper: Optional[BaseWeightMapper] = None,
                     params_map: Optional[Dict[str, str]] = None,
                     allow_partial_loading: bool = False):
        from tensorrt_llm._torch.speculative.utils import (
            filter_mtp_checkpoint_weights, uses_mtp_head_checkpoint)

        skip_modules = ["draft_model"]
        if uses_mtp_head_checkpoint(self.spec_config):
            # The heads come from speculative_model in a second pass
            # (load_draft_weights), so exclude them here. They must be
            # *skipped* rather than tolerated via allow_partial_loading:
            # partial loading suppresses process_weights_after_loading() on
            # every quantized Linear/MoE it touches, which would leave the
            # target model's quant scales (NVFP4 alphas, MoE input scales)
            # uninitialized.
            weights = filter_mtp_checkpoint_weights(weights)
            skip_modules.extend(self.mtp_head_module_names())
        super().load_weights(weights=weights,
                             weight_mapper=weight_mapper,
                             skip_modules=skip_modules,
                             params_map=params_map,
                             allow_partial_loading=allow_partial_loading)

    def load_draft_weights(self,
                           weights: Dict,
                           weight_mapper: Optional[BaseWeightMapper] = None):
        from tensorrt_llm._torch.models.modeling_utils import \
            _load_weights_impl_v2
        from tensorrt_llm._torch.speculative.utils import (
            remap_preprocessed_mtp_weights_for_draft_model,
            select_mtp_checkpoint_weights,
            skip_modules_for_separate_mtp_checkpoint, uses_mtp_head_checkpoint)

        if uses_mtp_head_checkpoint(self.spec_config):
            # Load MTP heads into draft_model only, and verify every non-shared
            # MTP parameter has a matching tensor. The previous parent-model
            # load used allow_partial_loading=True, which silently left MTP
            # modules at random init when keys did not bind.
            n_total = len(weights)
            weights = select_mtp_checkpoint_weights(weights)
            if not weights:
                raise ValueError(
                    "speculative_model was set for MTP but no 'mtp.*' weights "
                    f"were found in {self.spec_config.speculative_model!r}. "
                    "Expected keys like 'mtp.layers.0.*'.")
            n_dropped = n_total - len(weights)
            if n_dropped:
                logger.warning(
                    "Ignoring %d non-mtp.* tensors from speculative_model while "
                    "loading MTP heads (kept %d mtp.* tensors).", n_dropped,
                    len(weights))
            if weight_mapper is None:
                raise ValueError(
                    "weight_mapper is required to load separate MTP heads")
            weights = weight_mapper.preprocess_weights(weights)
            num_hidden_layers = self.config.num_hidden_layers
            num_mtp_layers = len(self.draft_model.mtp_layers)
            weights = remap_preprocessed_mtp_weights_for_draft_model(
                weights,
                num_hidden_layers=num_hidden_layers,
                num_mtp_layers=num_mtp_layers,
            )

            # Skip optional modules (e.g. shared_head) only when absent from
            # this checkpoint; architectures that ship those tensors still load
            # them under allow_partial_loading=False.
            _load_weights_impl_v2(
                self.draft_model,
                weights,
                weight_mapper,
                skip_modules=skip_modules_for_separate_mtp_checkpoint(weights),
                allow_partial_loading=False,
            )
            return

        args = inspect.getfullargspec(self.draft_model.load_weights).args
        if "weight_mapper" in args:
            self.draft_model.load_weights(weights=weights,
                                          weight_mapper=weight_mapper)
        else:
            self.draft_model.load_weights(weights=weights)

        if self.spec_config and (
                not self.spec_config.spec_dec_mode.is_external_drafter()
                or self.spec_config.spec_dec_mode.is_dflash()
                or self.spec_config.spec_dec_mode.is_dspark()):
            self.draft_model.load_weights_from_target_model(self)

    def set_guided_decoder(self,
                           guided_decoder: CapturableGuidedDecoder) -> bool:
        if hasattr(self.spec_worker, "set_guided_decoder"):
            return self.spec_worker.set_guided_decoder(guided_decoder)
        return False
