# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv
)
from tensorrt_llm.logger import logger


class VoRAAttention(Qwen2Attention):
    """Custom attention module for VoRA that properly handles 4D attention masks."""
    
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        logger.info(f"Initializing VoRAAttention for layer {layer_idx}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with VoRA 4D attention mask support."""
        
        # Store original shape for proper output reshaping
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Debug logging
        if attention_mask is not None:
            logger.debug(f"[VoRAAttention] Input attention_mask shape: {attention_mask.shape}")
            logger.debug(f"[VoRAAttention] Input attention_mask ndim: {attention_mask.ndim}")
            logger.debug(f"[VoRAAttention] hidden_states shape: {hidden_states.shape}")
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle past key values for generation
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        
        # Apply attention mask - VoRA's 4D attention mask handling
        if attention_mask is not None:
            logger.debug(f"[VoRAAttention] Attention mask shape: {attention_mask.shape}")
            logger.debug(f"[VoRAAttention] Attention weights shape: {attn_weights.shape}")
            
            # Handle VoRA's 4D attention mask format
            if attention_mask.ndim == 4:
                # VoRA's 4D mask: [batch_size, 1, seq_len, seq_len]
                # During generation, we need to extract the relevant portion
                bs, heads, mask_seq_len, mask_tgt_len = attention_mask.shape
                attn_seq_len = attn_weights.shape[-1]  # key sequence length
                query_len = attn_weights.shape[-2]    # query sequence length
                
                logger.debug(f"[VoRAAttention] Mask dimensions: bs={bs}, heads={heads}, mask_seq_len={mask_seq_len}, mask_tgt_len={mask_tgt_len}")
                logger.debug(f"[VoRAAttention] Attention weights dimensions: query_len={query_len}, attn_seq_len={attn_seq_len}")
                
                # During generation, we need to handle the case where query_len might be 1 (generating one token)
                # but the mask is for the full sequence
                if query_len == 1 and mask_seq_len > 1:
                    # Extract the mask for the current position (last row for causal generation)
                    attention_mask = attention_mask[:, :, -1:, :]
                    logger.debug(f"[VoRAAttention] Extracted mask for single token generation: {attention_mask.shape}")
                elif mask_seq_len != query_len or mask_tgt_len != attn_seq_len:
                    # Extract the relevant portion of the mask
                    # For VoRA, we typically need the last query_len rows and last attn_seq_len columns
                    attention_mask = attention_mask[:, :, -query_len:, -attn_seq_len:]
                    logger.debug(f"[VoRAAttention] Extracted mask portion: {attention_mask.shape}")
                
                # Expand mask for all attention heads if needed
                if attention_mask.shape[1] == 1 and self.config.num_attention_heads > 1:
                    attention_mask = attention_mask.expand(-1, self.config.num_attention_heads, -1, -1)
                    logger.debug(f"[VoRAAttention] Expanded mask for all heads: {attention_mask.shape}")
                
                # Final shape verification
                if attention_mask.shape[-2:] != attn_weights.shape[-2:]:
                    logger.error(f"[VoRAAttention] Shape mismatch after processing: mask={attention_mask.shape}, weights={attn_weights.shape}")
                    # As a last resort, try to make it work
                    if attn_weights.shape[-2] == 1:
                        attention_mask = attention_mask[:, :, -1:, :attn_weights.shape[-1]]
                
                # Apply the mask (VoRA's mask already contains the correct values)
                attn_weights = attn_weights + attention_mask
                
            elif attention_mask.ndim == 2:
                # Standard 2D mask: [batch_size, seq_len] - convert to 4D
                from transformers.modeling_utils import _prepare_4d_causal_attention_mask
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_len),
                    hidden_states,
                    past_key_value.get_seq_length() if past_key_value is not None else 0,
                )
                attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape attention output
        # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Get the actual sequence length from the attention output
        actual_batch_size, actual_seq_len = attn_output.shape[:2]
        
        # Merge heads
        attn_output = attn_output.reshape(actual_batch_size, actual_seq_len, self.config.num_attention_heads * self.head_dim)
        
        # Final output projection
        attn_output = self.o_proj(attn_output)
        
        logger.debug(f"[VoRAAttention] Output shape: {attn_output.shape}")
        
        return attn_output, attn_weights


def replace_attention_with_vora(model):
    """Replace all Qwen2Attention modules with VoRAAttention."""
    
    replaced_count = 0
    
    # Iterate through all layers
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'self_attn'):
            # Get the config from the original attention
            config = layer.self_attn.config if hasattr(layer.self_attn, 'config') else model.config
            
            # Create new VoRA attention
            vora_attn = VoRAAttention(config, i)
            
            # Copy weights from original attention
            vora_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data.clone()
            vora_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data.clone()
            vora_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data.clone()
            vora_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data.clone()
            
            if hasattr(layer.self_attn.q_proj, 'bias') and layer.self_attn.q_proj.bias is not None:
                vora_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data.clone()
                vora_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data.clone()
                vora_attn.v_proj.bias.data = layer.self_attn.v_proj.bias.data.clone()
            
            # Replace the attention module
            layer.self_attn = vora_attn
            replaced_count += 1
    
    logger.info(f"Replaced {replaced_count} attention layers with VoRAAttention")
    return model