# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import tensorrt as trt

# from tensorrt_llm.bindings import PluginManager  # Not needed for VoRA
from tensorrt_llm.functional import (Tensor, concat, constant, expand_dims_like,
                                      view, select, shape, slice, 
                                      unsqueeze, where)
from tensorrt_llm.layers import Attention, AttentionMaskType, MLP
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.model_weights_loader import ModelWeightsLoader
from tensorrt_llm.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.models.qwen.model import QWenDecoderLayer, QWenModel
from tensorrt_llm.module import Module, ModuleList

from .config import VoRAConfig
from .vision_embedding import build_vision_embedding


class VoRAAttention(Attention):
    """VoRA-specific attention that supports hybrid attention masks.
    
    Vision tokens use bidirectional attention while text tokens use causal attention.
    """
    
    def __init__(self, config: VoRAConfig, layer_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.use_hybrid_attention = config.vision_attention_mask == "bidirectional"
        
    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                vision_token_mask: Optional[Tensor] = None,
                past_key_value: Optional[Tensor] = None,
                sequence_length: Optional[Tensor] = None,
                past_key_value_length: Optional[Tensor] = None,
                context: Optional[Tensor] = None,
                context_mask: Optional[Tensor] = None,
                cache_indirection: Optional[Tensor] = None,
                kv_cache_params: Optional[Tensor] = None,
                kv_quant_scale: Optional[Tensor] = None,
                kv_dequant_scale: Optional[Tensor] = None,
                encoder_output: Optional[Tensor] = None):
        
        # If we have vision_token_mask and hybrid attention is enabled,
        # we need to create a custom attention mask
        if self.use_hybrid_attention and vision_token_mask is not None:
            # This will be handled by custom plugin in production
            # For now, use standard attention with provided mask
            pass
            
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            past_key_value_length=past_key_value_length,
            context=context,
            context_mask=context_mask,
            cache_indirection=cache_indirection,
            kv_cache_params=kv_cache_params,
            kv_quant_scale=kv_quant_scale,
            kv_dequant_scale=kv_dequant_scale,
            encoder_output=encoder_output
        )


class VoRADecoderLayer(QWenDecoderLayer):
    """VoRA decoder layer with hybrid attention support."""
    
    def __init__(self, config: VoRAConfig, layer_idx: int):
        # Initialize parent but we'll replace attention
        super().__init__(config, layer_idx)
        
        # Replace attention with VoRA-specific attention
        self.attention = VoRAAttention(
            config=config,
            layer_idx=layer_idx,
            local_layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,  # Default, will be overridden for vision
            bias=config.attn_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode
        )
    
    def forward(self,
                hidden_states,
                attention_mask=None,
                vision_token_mask=None,
                use_cache=False,
                kv_cache_params=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                medusa_position_offsets=None,
                medusa_packed_mask=None,
                cache_indirection=None,
                encoder_output=None):
        
        # Self-attention with potential hybrid mask
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attention_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            vision_token_mask=vision_token_mask,
            past_key_value=past_key_value,
            sequence_length=sequence_length,
            past_key_value_length=past_key_value_length,
            cache_indirection=cache_indirection,
            kv_cache_params=kv_cache_params,
            encoder_output=encoder_output
        )
        
        if use_cache:
            attention_output, presents = attention_output
        
        hidden_states = residual + attention_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return (hidden_states, presents)
        return hidden_states


class VoRAModel(QWenModel):
    """VoRA transformer model with vision embedding support."""
    
    def __init__(self, config: VoRAConfig):
        super().__init__(config)
        
        # Override layers with VoRA-specific layers
        self.layers = ModuleList(
            [VoRADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Add vision embedding module
        self.vision_embedding = build_vision_embedding(
            config, 
            config.hidden_size,
            dtype=config.dtype
        )
    
    def forward(self,
                input_ids=None,
                vision_input=None,
                vision_token_mask=None,
                position_ids=None,
                use_cache=False,
                attention_mask=None,
                kv_cache_params=None,
                past_key_value=None,
                sequence_length=None,
                past_key_value_length=None,
                medusa_position_offsets=None,
                medusa_packed_mask=None,
                cache_indirection=None,
                hidden_states=None,
                encoder_output=None):
        
        # If we have vision input, process it first
        if vision_input is not None:
            vision_embeds = self.vision_embedding(vision_input)
            # vision_embeds shape: [batch_size, num_patches, hidden_size]
            
            # Get text embeddings
            if hidden_states is None:
                hidden_states = self.vocab_embedding(input_ids)
            
            # Concatenate vision and text embeddings
            # Assuming vision comes first in the sequence
            hidden_states = concat([vision_embeds, hidden_states], dim=1)
        elif hidden_states is None:
            hidden_states = self.vocab_embedding(input_ids)
        
        # Continue with standard forward pass
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                vision_token_mask=vision_token_mask,
                use_cache=use_cache,
                kv_cache_params=kv_cache_params,
                past_key_value=past_key_value[layer_idx] if past_key_value else None,
                sequence_length=sequence_length,
                past_key_value_length=past_key_value_length,
                medusa_position_offsets=medusa_position_offsets,
                medusa_packed_mask=medusa_packed_mask,
                cache_indirection=cache_indirection,
                encoder_output=encoder_output
            )
            
            if use_cache:
                hidden_states, present = hidden_states
                if past_key_value is None:
                    past_key_value = []
                past_key_value.append(present)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        if use_cache:
            return (hidden_states, past_key_value)
        return hidden_states


class VoRAForCausalLM(DecoderModelForCausalLM):
    """VoRA model for causal language modeling with vision support."""
    
    config_class = VoRAConfig
    
    def __init__(self, config: VoRAConfig):
        self.check_config(config)
        transformer = VoRAModel(config)
        lm_head = config.lm_head_cls(config)
        super().__init__(config, transformer, lm_head)
    
    def check_config(self, config):
        config.set_if_not_exist('rotary_base', 1000000.0)
        config.set_if_not_exist('rotary_scaling', None)
        config.set_if_not_exist('attn_bias', True)
        config.set_if_not_exist('moe', None)
    
    @classmethod
    def from_hugging_face(cls,
                         hf_model_dir: str,
                         dtype: str = 'auto',
                         mapping: Optional[Mapping] = None,
                         quant_config: Optional[dict] = None,
                         **kwargs):
        """Load VoRA model from HuggingFace checkpoint."""
        from transformers import AutoConfig
        import os
        
        # Load config
        config = VoRAConfig.from_hugging_face(
            hf_model_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs
        )
        
        # Create model
        model = cls(config)
        
        # Define weight mapping for VoRA
        # VoRA weights have "llm." prefix and vision_embedding weights
        custom_dict = {
            # LLM weights mapping
            "transformer": "llm.model",
            "vocab_embedding": "llm.model.embed_tokens",
            "ln_f": "llm.model.norm",
            "lm_head": "llm.lm_head",
            "attention.qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
            "attention.dense": "self_attn.o_proj",
            "mlp.fc": "mlp.up_proj",
            "mlp.gate": "mlp.gate_proj", 
            "mlp.proj": "mlp.down_proj",
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            
            # Vision embedding weights mapping
            "vision_embedding.patchifier.proj": "vision_embedding.patchifier.proj",
            "vision_embedding.patchifier.norm": "vision_embedding.patchifier.norm",
            "vision_embedding.pos_embed": "vision_embedding.pos_embed",
            "vision_embedding.out_proj": "vision_embedding.out_proj",
        }
        
        # Load weights
        loader = ModelWeightsLoader(hf_model_dir, custom_dict)
        loader.generate_tllm_weights(model)
        
        return model
    
    def prepare_inputs(self, *args, **kwargs):
        """Prepare inputs for VoRA model, handling both text and vision inputs."""
        # Get base inputs
        inputs = super().prepare_inputs(*args, **kwargs)
        
        # Add vision-specific inputs if provided
        if 'vision_input' in kwargs:
            inputs['vision_input'] = kwargs['vision_input']
        if 'vision_token_mask' in kwargs:
            inputs['vision_token_mask'] = kwargs['vision_token_mask']
            
        return inputs