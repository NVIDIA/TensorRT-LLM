import types
import warnings
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache


# The model is in
# /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/autodeploy_data/
# hf_home/modules/transformers_modules/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/
# modeling_deepseek.py
@torch.inference_mode()
def deepseek_v3_attention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.IntTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """DeepSeekV3Attention forward function rewritten to wrap MultiheadLatentAttention as a custom op.

    TODO: explain what I'm doing here


    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    assert self.q_lora_rank is not None, "q_lora_rank must be set"

    # x * W_DQ (i.e. q down projection)
    q_normed_dn = self.q_a_layernorm(self.q_a_proj(hidden_states))  # (bsz, q_len, self.q_lora_rank)

    wq_b = self.q_b_proj.weight  # (self.num_heads * self.q_head_dim, self.q_lora_rank)

    # c_KV = x * W_DKV (i.e. kv down projection)
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [bsz, q_len, 512 + 64]
    # Separates the compressed kv into the low-rank part and the positional encoding part
    compressed_kv, k_pe = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )  # compressed_kv ~ [bsz, q_len, 512 ], k_pe ~ [bsz, q_len, 64]
    compressed_kv = self.kv_a_layernorm(compressed_kv)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

    cos_cache, sin_cache = self.rotary_emb.cos_cached, self.rotary_emb.sin_cached

    wkv_b = self.kv_b_proj.weight  # [128 * 256, 512]
    wo_proj = self.o_proj.weight

    # Use custom op to capture mla. This does not handle KV cache
    # as passing transformers Cache into a custom op is throwing an error.
    # Is not an issue, because we intend to replace mla op with our implementation further along the pipeline
    args = (
        q_normed_dn,
        compressed_kv,
        k_pe,
        sin_cache,
        cos_cache,
        wkv_b,  # CONSTANTS
        wq_b,
        None,  # w_uq_ukv
        wo_proj,
        None,  # w_uv_o
        position_ids,  # METADATA
        self.softmax_scale,  # CONSTANTS
    )

    attn_output = torch.ops.auto_deploy.torch_deepseek_mla_no_cache(*args)
    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


# This patched module matches exactly with HF generate
@torch.inference_mode()
def deepseek_v3_moe_exact(self, hidden_states):
    """DeepSeekV3MoE forward function rewritten to enable torch export.

    This custom implementation matches exactly with the deepseek implementation. There are
    some errors in the output tensors when the index_add based implementation is used, leading
    to some mismatch in the outputs for some prompts. This ensures exact match between HF output
    without custom patch and with custom patch.
    """
    identity = hidden_states
    batch_size, sequence_length, hidden_dim = hidden_states.shape

    selected_experts, routing_weights, *_ = self.gate(hidden_states)

    hidden_states = hidden_states.view(-1, hidden_dim)
    idxs = torch.argsort(selected_experts.view(-1), stable=True)

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_classes=self.experts_per_rank
    ).permute(2, 1, 0)
    outputs = []
    for expert_idx in range(len(self.experts)):
        expert_layer = self.experts[expert_idx]
        _, top_x = torch.where(expert_mask[expert_idx])
        # Sort the top_xs and idx
        sorted, _ = torch.sort(top_x)
        tokens_for_this_expert = hidden_states[None, sorted].reshape(-1, hidden_dim)
        expert_out = expert_layer(tokens_for_this_expert)
        outputs.append(expert_out)

    outs = torch.cat(outputs, dim=0)
    # Wrap torch.zeros() in a custom op to fix meta device issue during inference.
    new_x = torch.zeros(
        (*selected_experts.view(-1).shape, hidden_dim),
        device=selected_experts.device,
        dtype=outs.dtype,
    )
    new_x[idxs] = outs
    final_hidden_states = (
        new_x.view(*selected_experts.shape, -1)
        .type(routing_weights.dtype)
        .mul_(routing_weights.unsqueeze(-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    if self.config.n_shared_experts is not None:
        final_hidden_states = final_hidden_states + self.shared_experts(identity)

    return final_hidden_states.to(hidden_states.dtype)


@torch.inference_mode()
def deepseek_v3_moe(self, hidden_states):
    """DeepSeekV3MoE forward function rewritten in Mixtral style to enable torch export."""

    selected_experts, routing_weights, *_ = self.gate(hidden_states)
    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],
        w2_weight=[expert.down_proj.weight for expert in self.experts],
        w3_weight=[expert.up_proj.weight for expert in self.experts],
    )

    if self.config.n_shared_experts is not None:
        final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)

    return final_hidden_states.to(hidden_states.dtype)


def deepseek_v3_rope(self, x, seq_len=None):
    """DeepSeekV3 Rotary Embedding forward function rewritten to enable torch export.
    We return the full cached cos and sin values, instead of slicing them based on seq_len as this
    would cause an issue during the generate phase (when seq_len=1 from input_ids). We also move the cos
    sin buffers to appropriate device to enable export.
    """

    return (
        self.cos_cached.to(dtype=x.dtype).to(device=x.device),
        self.sin_cached.to(dtype=x.dtype).to(device=x.device),
    )


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "DeepseekV3MoE": deepseek_v3_moe,
    "DeepseekV2MoE": deepseek_v3_moe,
    "DeepseekV3RotaryEmbedding": deepseek_v3_rope,
    "DeepseekV3YarnRotaryEmbedding": deepseek_v3_rope,
    "DeepseekV2RotaryEmbedding": deepseek_v3_rope,
    "DeepseekV2YarnRotaryEmbedding": deepseek_v3_rope,
    "DeepseekV3Attention": deepseek_v3_attention,
}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES.keys():
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
