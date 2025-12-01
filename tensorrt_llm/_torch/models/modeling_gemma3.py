import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import Gemma3TextConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata, FlashInferAttentionMetadata
from ..attention_backend.interface import (AttentionMask, CustomAttentionMask,
                                           PositionalEmbeddingParams,
                                           PredefinedAttentionMask, RopeParams)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


class Gemma3TextScaledWordEmbedding(Embedding):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
    ):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
        )
        self.embed_scale = torch.sqrt(torch.tensor(hidden_size)).to(self.dtype)

    @torch.inference_mode()
    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Gemma3Attention(QKNormRoPEAttention):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: Optional[int] = None,
        is_sliding: bool = False,
    ):
        self.is_sliding = is_sliding
        config = model_config.pretrained_config
        rope_params = RopeParams.from_config(config)
        self.attention_window_size = None
        if is_sliding:
            rope_params.theta = config.rope_local_base_freq
            rope_params.scale_type = RotaryScalingType.none
            rope_params.scale = 1.0
            self.attention_window_size = config.sliding_window
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )
        q_scaling = math.sqrt(config.query_pre_attn_scalar) / math.sqrt(
            config.head_dim)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            q_scaling=q_scaling,
        )

    @torch.inference_mode()
    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        attention_mask_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        if attention_mask_data is not None:
            assert isinstance(
                attn_metadata, FlashInferAttentionMetadata
            ), "Only FlashInfer backend supports custom attention mask currently."
            assert attention_mask == CustomAttentionMask.CUSTOM
        return super().forward(position_ids=position_ids,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata,
                               attention_mask=attention_mask,
                               attention_window_size=self.attention_window_size,
                               attention_mask_data=attention_mask_data,
                               **kwargs)


# This function is written to be compatible with TRTLLM's GatedMLP class.
def pytorch_gelu_tanh(gate_x: torch.Tensor) -> torch.Tensor:
    gate, x = gate_x.chunk(2, dim=-1)
    return nn.functional.gelu(gate, approximate="tanh") * x


class Gemma3DecoderLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        is_sliding = (config.layer_types[layer_idx] == "sliding_attention")
        self.self_attn = Gemma3Attention(
            model_config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
        )

        self.mlp = GatedMLP(hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            bias=False,
                            activation=pytorch_gelu_tanh,
                            dtype=config.torch_dtype,
                            config=model_config,
                            layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                       eps=config.rms_norm_eps,
                                       dtype=config.torch_dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                eps=config.rms_norm_eps,
                                                dtype=config.torch_dtype)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size=config.hidden_size,
                                                 eps=config.rms_norm_eps,
                                                 dtype=config.torch_dtype)
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype)

    @torch.inference_mode()
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=CustomAttentionMask.CUSTOM if attention_mask_data
            is not None else PredefinedAttentionMask.CAUSAL,
            attention_mask_data=attention_mask_data,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states,
                                 lora_params=kwargs.get("lora_params", None))
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3TextModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[Gemma3TextConfig]):
        super().__init__(model_config)
        config = self.model_config
        self.hidden_size = config.pretrained_config.hidden_size

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.pretrained_config.vocab_size,
            config.pretrained_config.hidden_size,
            dtype=config.pretrained_config.torch_dtype,
            mapping=config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList([
            Gemma3DecoderLayer(
                model_config,
                layer_idx,
            ) for layer_idx in range(config.pretrained_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=config.pretrained_config.hidden_size,
                            eps=config.pretrained_config.rms_norm_eps,
                            dtype=config.pretrained_config.torch_dtype)

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        local_attention_mask_data: Optional[torch.Tensor] = None,
        global_attention_mask_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.to(self.dtype)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                attention_mask_data=local_attention_mask_data
                if decoder_layer.self_attn.is_sliding else
                global_attention_mask_data,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("Gemma3ForCausalLM")
class Gemma3ForCausalLM(DecoderModelForCausalLM[Gemma3TextModel,
                                                Gemma3TextConfig]):

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
    ):
        super().__init__(Gemma3TextModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    def _get_token_type_mask(self, image_token_mask: torch.BoolTensor):
        device = image_token_mask.device
        sequence_length = len(image_token_mask)

        # Create a list of token type ids. 0 for text tokens, 1 for all image tokens (regardless of which image they belong to).
        token_type_ids = torch.zeros(sequence_length,
                                     dtype=torch.int32,
                                     device=device)
        token_type_ids[image_token_mask] = 1

        # There could be image tokens from multiple images where those corresponding to the
        # same image are contiguous. We assign a unique id to each contiguous blob of image tokens now.

        # Pad with zero at the start to detect changes.
        padded = torch.cat((torch.tensor([0], device=device), token_type_ids))

        # Identify where blobs start (0->1 transitions)
        starts = (padded[1:] > padded[:-1]).int()

        # Cumulative sum of starts gives a unique id for each blob. Note that
        # this assigns a unique id to the zeros separating the blobs.
        blob_ids = torch.cumsum(starts, dim=0)

        # Mask out zeros (positions where token_type_ids == 0).
        token_type_ids *= blob_ids

        # Create a mask where each blob is a unique id.
        token_type_mask = token_type_ids.unsqueeze(
            0) == token_type_ids.unsqueeze(1)

        # If text token, do not change anything.
        token_type_mask = torch.where(token_type_ids == 0, False,
                                      token_type_mask)

        return token_type_mask

    def get_context_mask(
        self,
        image_token_mask: torch.BoolTensor,
        effective_sliding_window: Optional[int] = None,
    ):
        """
        Returns an attention mask such that text tokens attend to each other in causal fashion while image
        tokens attend in causal fashion as well as to all other image tokens in a bidirectional manner.
        Args:
            image_token_mask: A boolean tensor of shape (sequence_length,) where True indicates an image token.
            effective_sliding_window: The effective sliding window size for the attention mask. Default is None, which means no sliding window.
            For Gemma3, this is the sliding window size from config (e.g. 512 for 1B model).
        Returns:
            A boolean attention mask of shape (sequence_length, sequence_length).
        """
        device = image_token_mask.device
        sequence_length = len(image_token_mask)
        if effective_sliding_window is None or effective_sliding_window >= sequence_length:
            causal_mask = torch.arange(
                sequence_length, device=device).unsqueeze(0) <= torch.arange(
                    sequence_length, device=device).unsqueeze(1)
        else:
            attention_mask_1 = (torch.arange(sequence_length,
                                             device=device).unsqueeze(0)
                                <= torch.arange(sequence_length,
                                                device=device).unsqueeze(1))
            attention_mask_2 = (
                torch.arange(sequence_length, device=device).unsqueeze(0)
                > torch.arange(sequence_length, device=device).unsqueeze(1) -
                effective_sliding_window)
            causal_mask = attention_mask_1 & attention_mask_2

        # Apply a bidirectional mask for image tokens.
        token_type_mask = self._get_token_type_mask(image_token_mask)
        causal_mask = causal_mask.masked_fill(token_type_mask, True)
        return causal_mask

    # ASSUMPTIONS:
    # 1) Chunked prefill is disabled to avoid chunking image tokens as they need bidirectional attention.
    # 2) KV cache reuse is disabled to avoid partially matched image tokens (entire image must be reused to get things correct).
    def get_flashinfer_attention_mask(
            self,
            image_token_mask: torch.BoolTensor,
            attn_metadata: AttentionMetadata,
            effective_sliding_window: Optional[int] = None) -> torch.Tensor:
        """
        This is specifically needed for context phase requests. Currently, we don't create custom mask for generation requests because FlashInfer backend
        doesn't use it anyway and there's nothing special we need to do for generation requests.
        - This function will only be called for a batch when there's at least one context request in the batch with image tokens.
        - In context phase, each sample's input_ids may have a mix of image tokens and text tokens where tokens corresponding to an image
        appear as a contiguous blob. Example: torch.IntTensor([2, 3, 4, 5, img_idx, img_idx, img_idx, ..., img_idx, 100])
        - While the text tokens attend to other tokens in a causal fashion, image tokens attend to others in a causal fashion and well as
        attend to other image tokens in a bidirectional manner. Hence, the need for custom masking.
        Args:
            image_token_mask: A boolean tensor of shape (len(input_ids),) where True indicates an image token. This corresponds to concatenated
            list of tokens for all samples in the batch.
            attn_metadata: The attention metadata for the batch.
            effective_sliding_window: The effective sliding window size for the attention mask. Default is None, which means no sliding window.
            For Gemma3, this is the sliding window size from config (e.g. 512 for 1B model).
        Returns:
            A flattened boolean mask of shape (sum(q_len[i] * k_len[i] for i in range(batch_size)).
        """

        assert isinstance(
            attn_metadata, FlashInferAttentionMetadata
        ), "Only FlashInfer backend supports custom mask currently."
        num_contexts = attn_metadata.num_contexts
        assert num_contexts > 0, "There should be at least one context request in the batch for custom mask."

        qo_indptr = attn_metadata.qo_indptr[:num_contexts + 1]
        cached_token_lens = attn_metadata.cached_token_lens[:num_contexts]
        assert (cached_token_lens == 0).all(
        ), "cached_token_lens should be 0 for context requests since chunked prefill and kv cache reuse must be disabled."

        # Create masks for context requests.
        context_mask_list = []
        for i in range(num_contexts):
            mask_i = self.get_context_mask(
                image_token_mask=image_token_mask[qo_indptr[i]:qo_indptr[i +
                                                                         1]],
                effective_sliding_window=effective_sliding_window,
            )
            context_mask_list.append(mask_i.flatten())
        return torch.cat(context_mask_list, dim=0).contiguous()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        image_token_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        local_attention_mask_data = None
        global_attention_mask_data = None
        if image_token_mask is not None:
            global_attention_mask_data = self.get_flashinfer_attention_mask(
                image_token_mask=image_token_mask,
                attn_metadata=attn_metadata,
                effective_sliding_window=None,
            )
            local_attention_mask_data = self.get_flashinfer_attention_mask(
                image_token_mask=image_token_mask,
                attn_metadata=attn_metadata,
                effective_sliding_window=self.config.sliding_window,
            )

        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            local_attention_mask_data=local_attention_mask_data,
            global_attention_mask_data=global_attention_mask_data,
            **kwargs,
        )

        return self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        super().load_weights(weights, weight_mapper)
