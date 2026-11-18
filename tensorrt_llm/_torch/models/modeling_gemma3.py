# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch import nn
from transformers import Gemma3TextConfig

from tensorrt_llm._torch.attention.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping

from ..attention.backends import (AttentionMetadata,
                                  FlashInferAttentionMetadata,
                                  TrtllmAttentionMetadata)
from ..attention.backends.interface import (AttentionMask, CustomAttentionMask,
                                            PositionalEmbeddingParams,
                                            PredefinedAttentionMask, RopeParams)
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..utils import inference_mode_unless_compiling
from .modeling_utils import (DecoderModel, DecoderModelForCausalLM,
                             register_auto_model)


@triton.jit
def _write_mask(
    output,
    blob_ids,
    query_length,
    prefix_length,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key_length = prefix_length + query_length
    valid = col < key_length
    offsets = row.to(tl.int64) * key_length + col
    allowed = col <= prefix_length + row
    if WINDOW is not None:
        allowed &= col > prefix_length + row - WINDOW
    query_blob = tl.load(blob_ids + row)
    key_blob = tl.load(blob_ids + col - prefix_length,
                       valid & (col >= prefix_length),
                       other=0)
    allowed |= (query_blob != 0) & (query_blob == key_blob)
    tl.store(output + offsets, allowed, valid)


def build_gemma_attention_mask(
    token_types: torch.Tensor,
    context_lengths: list[int],
    prefix_lengths: list[int],
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Build flattened bool masks for packed Gemma context requests.

    CUDA graph capture requires prior warmup. Request lengths, prefix lengths,
    and the sliding window are host-side launch parameters and must remain
    fixed during replay; token-type values may change in place. This function
    does not modify token_types.

    Args:
        token_types: One-dimensional CUDA bool image flags (Gemma3) or integer
            modality IDs (Gemma4: 0=text, 1=image, 2=video, 3=audio). Contiguous
            runs of the same vision type attend bidirectionally. A strided
            input is supported; trailing generation tokens are ignored.
        context_lengths: Number of new tokens for each context request.
        prefix_lengths: Cached token counts in the same request order. Chunk
            boundaries must not split a vision blob, as in the model caller.
        sliding_window: Exclusive lower attention-window bound, or None.

    Returns:
        Contiguous mask with sum(q * (p + q)) elements. Prefix columns apply
        causal/window rules; vision blobs override those rules only among
        new tokens. Intermediate storage is linear in the context length.
    """
    mask_sizes = [
        q * (p + q)
        for q, p in zip(context_lengths, prefix_lengths, strict=True)
    ]
    output = torch.empty(sum(mask_sizes),
                         dtype=torch.bool,
                         device=token_types.device)
    token_offset = 0
    mask_offset = 0
    for q, p, size in zip(context_lengths,
                          prefix_lengths,
                          mask_sizes,
                          strict=True):
        types = token_types[token_offset:token_offset + q]
        if size:
            vision = (types == 1) | (types == 2)
            previous = torch.cat((types.new_zeros(1), types[:-1]))
            starts = (types != previous) & vision
            blobs = torch.cumsum(starts, dim=0, dtype=torch.int32)
            blobs.masked_fill_(~vision, 0)
            _write_mask[(q, triton.cdiv(
                p + q, 1024))](output[mask_offset:mask_offset + size], blobs, q,
                               p, sliding_window, 1024)
        token_offset += q
        mask_offset += size
    return output


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

    @inference_mode_unless_compiling
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
            # transformers 5.x moved rope_local_base_freq into
            # rope_parameters["sliding_attention"]["rope_theta"]
            local_freq = getattr(config, 'rope_local_base_freq', None)
            if local_freq is None:
                rp = getattr(config, 'rope_parameters', None) or {}
                local_freq = rp.get('sliding_attention', {}).get('rope_theta')
            if local_freq is None:
                raise ValueError(
                    "Gemma3 sliding attention requires a local RoPE base "
                    "frequency, but neither `config.rope_local_base_freq` "
                    "(transformers 4.x) nor "
                    "`config.rope_parameters['sliding_attention']['rope_theta']` "
                    "(transformers 5.x) is set on the model config.")
            rope_params.theta = local_freq
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

    @inference_mode_unless_compiling
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
                attn_metadata,
                (FlashInferAttentionMetadata, TrtllmAttentionMetadata),
            ), "Only FlashInfer and TRTLLM backends support custom attention masks."
            assert attention_mask == CustomAttentionMask.CUSTOM
        return super().forward(position_ids=position_ids,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata,
                               attention_mask=attention_mask,
                               attention_window_size=self.attention_window_size,
                               attention_mask_data=attention_mask_data,
                               **kwargs)


# This function is written to be compatible with TRTLLM's GatedMLP class.
def gelu_tanh(gate_x: torch.Tensor) -> torch.Tensor:
    if IS_FLASHINFER_AVAILABLE:
        return torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(gate_x)

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
                            activation=gelu_tanh,
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

    @inference_mode_unless_compiling
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

    @inference_mode_unless_compiling
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

    @classmethod
    def get_preferred_transceiver_runtime(
        cls,
        pretrained_config: Any = None,
    ) -> Optional[Literal["CPP", "PYTHON"]]:
        return "PYTHON"

    def __init__(
        self,
        model_config: ModelConfig[Gemma3TextConfig],
    ):
        super().__init__(Gemma3TextModel(model_config),
                         config=model_config,
                         hidden_size=model_config.pretrained_config.hidden_size,
                         vocab_size=model_config.pretrained_config.vocab_size)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls,
                                               pretrained_config: Any = None
                                               ) -> Literal["V2"]:
        """Prefer KV cache manager V2 for Gemma3's VSWA layout.

        V2 sizes the sliding-window and full-attention pools independently
        instead of statically dividing memory between them.
        """
        return "V2"

    def get_context_mask(
        self,
        image_token_mask: torch.BoolTensor,
        effective_sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Build causal/window attention with bidirectional image blobs."""
        length = len(image_token_mask)
        return build_gemma_attention_mask(image_token_mask, [length], [0],
                                          effective_sliding_window).view(
                                              length, length)

    # Image blobs cannot be split by chunked prefill or partial prefix reuse.
    def get_attention_mask(
            self,
            image_token_mask: torch.BoolTensor,
            attn_metadata: AttentionMetadata,
            effective_sliding_window: Optional[int] = None) -> torch.Tensor:
        """
        This is specifically needed for context phase requests. Generation
        requests use the regular causal attention semantics and do not need a
        custom mask.
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

        num_contexts = attn_metadata.num_contexts
        assert num_contexts > 0, "There should be at least one context request in the batch for custom mask."

        context_lens = attn_metadata.seq_lens[:num_contexts].tolist()
        cached_token_lens = (
            attn_metadata.kv_cache_params.
            num_cached_tokens_per_seq[:num_contexts]
            if attn_metadata.kv_cache_params is not None and
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq is not None
            else [0] * num_contexts)
        assert not any(cached_token_lens), (
            "cached_token_lens should be 0 for context requests since chunked prefill "
            "and kv cache reuse must be disabled.")

        # Create masks for context requests.
        return build_gemma_attention_mask(image_token_mask, context_lens,
                                          cached_token_lens,
                                          effective_sliding_window)

    @inference_mode_unless_compiling
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
            global_attention_mask_data = self.get_attention_mask(
                image_token_mask=image_token_mask,
                attn_metadata=attn_metadata,
                effective_sliding_window=None,
            )
            local_attention_mask_data = self.get_attention_mask(
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
