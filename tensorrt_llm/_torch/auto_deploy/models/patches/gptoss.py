import types
from typing import Callable, Dict, Optional

import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM


def gpt_oss_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """GPT OSS Attention forward function rewritten to wrap attention as a custom op."""
    from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

    # Add new parameters
    sliding_window = getattr(self, "sliding_window", -1)  # Default to -1 if not present

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Apply Q, K, V projections (same as original)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Use original rope implementation
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle KV cache properly
    if past_key_value is not None:
        # Update KV cache - check if it has update method (modern cache objects)
        if hasattr(past_key_value, "update"):
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            # Handle legacy tuple-based cache
            if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                past_key, past_value = past_key_value
                key_states = torch.cat([past_key, key_states], dim=2)
                value_states = torch.cat([past_value, value_states], dim=2)

    # Convert from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim]
    query_states = query_states.transpose(1, 2).contiguous()
    key_states = key_states.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    # Get sinks parameter from model if available
    sinks = None
    if hasattr(self, "sinks"):
        # If sinks is a model parameter, use it directly
        sinks = self.sinks

    # Use custom op to capture attention. This layout is bsnd (batch, seq, num_heads, head_dim)
    attn_output = torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=True,
        scale=self.scaling,
        sinks=sinks,
        sliding_window=sliding_window,
    )

    # Reshape back to original input shape
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, past_key_value


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, Callable] = {
    "GptOssAttention": gpt_oss_attention,
}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES.keys():
            # Replace the forward method
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


AutoModelForCausalLM.from_config = get_model_from_config_patched
