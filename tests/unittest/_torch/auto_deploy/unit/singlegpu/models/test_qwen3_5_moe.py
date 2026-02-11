"""Tests for the custom Qwen3.5 MoE model implementation.

Validates that the prefill-only custom model (with autodeploy custom ops)
produces numerically equivalent results to the HuggingFace reference
implementation (pure-PyTorch fallback paths) for each major block:
  1. GatedDeltaNet (linear attention)
  2. Attention (full attention with gating)
  3. SparseMoeBlock (routed + shared experts)
  4. DecoderLayer (composed token mixer + channel mixer)

Additionally validates vision tower and multimodal wrapper via functional
comparison tests (reference re-implementations on the same weights):
  5. VisionAttention reference comparison
  6. VisionBlock reference comparison
  7. VisionModel reference comparison (single-image and multi-image)
  8. Multimodal forward reference comparison
  9. position_embeddings passthrough equivalence
 10. get_rope_index correctness
 11. torch.export with cos/sin extra inputs
"""

import pytest
import torch
import torch.nn.functional as F

# Register autodeploy custom ops (torch_moe, torch_causal_conv1d, etc.)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeConfig,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeExperts,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeModel,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeTopKRouter,
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionBlock,
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _make_small_config(**overrides) -> Qwen3_5MoeTextConfig:
    """Create a small Qwen3.5 MoE config for fast unit testing.

    Uses 4 layers (3 linear_attention + 1 full_attention) with minimal dimensions.
    """
    defaults = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=16,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 2, 2],
        },
        # GDN params
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        # MoE params
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        # layer_types: default pattern gives 3 linear + 1 full attention for 4 layers
    )
    defaults.update(overrides)
    return Qwen3_5MoeTextConfig(**defaults)


def _init_block_weights(module, std=0.02):
    """Initialize weights for standalone block-level testing.

    When modules are created directly (not via PreTrainedModel._from_config),
    the _init_weights callback never fires, leaving torch.empty() params
    uninitialized (containing NaN/garbage). This function initializes them.
    """
    for m in module.modules():
        if isinstance(m, Qwen3_5MoeExperts):
            m.gate_up_proj.data.normal_(mean=0.0, std=std)
            m.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(m, Qwen3_5MoeTopKRouter):
            # Use randn for non-trivial routing
            m.weight.data.normal_(mean=0.0, std=1.0)


# =============================================================================
# HF Reference Implementations
# =============================================================================
# Pure-PyTorch "torch fallback" functions copied from the HF reference
# modeling_qwen3_5_moe.py. These have zero external dependencies and are used
# to validate that our autodeploy custom-ops based forward passes produce
# equivalent results.


class OriginalQwen3_5MoeRMSNorm(torch.nn.Module):
    """Original HF Qwen3.5 MoE RMSNorm with (1 + weight) scaling, zero-init.

    Used as a reference to validate the modified Qwen3_5MoeRMSNorm (which uses a
    load-time pre-hook to fold the +1 offset into the weight parameter).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def ref_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm. Copied from HF modeling_qwen3_5_moe.py l2norm."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def ref_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    """Chunked gated delta rule. Copied from HF torch_chunk_gated_delta_rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = ref_l2norm(query, dim=-1, eps=1e-6)
        key = ref_l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def ref_repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """KV head repetition for GQA. Copied from HF repeat_kv."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def ref_eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_key_value_groups: int,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager matmul-based attention. Adapted from HF eager_attention_forward.

    All tensors in bnsd layout: [B, N, S, D].
    Returns: [B, S, N, D] (transposed, ready for reshape).
    """
    key_states = ref_repeat_kv(key, num_key_value_groups)
    value_states = ref_repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


# =============================================================================
# Reference Forward Functions
# =============================================================================
# These functions take our custom modules (with their weights) and re-run the
# forward pass using the HF reference logic (no custom ops). This lets us
# compare the two code paths on identical weights + inputs.


def ref_gdn_forward(module, hidden_states):
    """HF-style GatedDeltaNet forward (torch fallback path, prefill-only, no cache).

    Uses module.conv1d directly (nn.Conv1d forward) instead of torch_causal_conv1d,
    and ref_chunk_gated_delta_rule (with internal l2norm) instead of
    torch_l2norm + torch_gated_delta_rule.
    """
    batch_size, seq_len, _ = hidden_states.shape

    # Projections
    mixed_qkv = module.in_proj_qkv(hidden_states)  # [B, S, conv_dim]
    z = module.in_proj_z(hidden_states)  # [B, S, value_dim]
    z = z.reshape(batch_size, seq_len, -1, module.head_v_dim)
    b = module.in_proj_b(hidden_states)  # [B, S, num_v_heads]
    a = module.in_proj_a(hidden_states)  # [B, S, num_v_heads]

    # Conv1d (HF torch fallback: transpose -> nn.Conv1d -> truncate -> silu -> transpose)
    mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, conv_dim, S]
    mixed_qkv = F.silu(module.conv1d(mixed_qkv)[:, :, :seq_len])  # [B, conv_dim, S]
    mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S, conv_dim]

    # Split Q/K/V
    query, key, value = torch.split(
        mixed_qkv,
        [module.key_dim, module.key_dim, module.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)

    # Gating
    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)

    # GQA repeat
    if module.num_v_heads // module.num_k_heads > 1:
        query = query.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)
        key = key.repeat_interleave(module.num_v_heads // module.num_k_heads, dim=2)

    # Gated delta rule (HF torch fallback -- l2norm applied internally)
    core_attn_out, _ = ref_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    # Gated RMSNorm
    core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
    z = z.reshape(-1, module.head_v_dim)
    core_attn_out = module.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return module.out_proj(core_attn_out)


def ref_attention_forward(module, hidden_states, position_embeddings):
    """HF-style Attention forward (eager path, bnsd layout, prefill-only, no cache).

    Uses ref_eager_attention_forward (matmul SDPA) instead of torch_attention,
    and operates in bnsd layout with unsqueeze_dim=1 for RoPE.
    """
    input_shape = hidden_states.shape[:-1]  # (B, S)
    head_dim = module.head_dim
    num_heads = module.num_heads
    num_kv_heads = module.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    scaling = head_dim**-0.5
    hidden_shape = (*input_shape, -1, head_dim)

    # Q projection with gating (HF style: view then transpose to bnsd)
    query_states, gate = torch.chunk(
        module.q_proj(hidden_states).view(*input_shape, -1, head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)  # (B, S, N*D)

    # Q/K norms then transpose to bnsd: [B, S, N, D] -> [B, N, S, D]
    query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # RoPE (unsqueeze_dim=1 for bnsd layout)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    # Additive causal mask: 0 for attended, -inf for masked
    q_len = input_shape[-1]
    causal_mask = torch.full((q_len, q_len), float("-inf"), device=hidden_states.device)
    causal_mask = causal_mask.triu(diagonal=1).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    # Eager attention (HF matmul-based SDPA in bnsd layout)
    attn_output = ref_eager_attention_forward(
        query_states,
        key_states,
        value_states,
        num_key_value_groups=num_kv_groups,
        scaling=scaling,
        attention_mask=causal_mask,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return module.o_proj(attn_output)


def ref_moe_forward(module, hidden_states):
    """HF-style SparseMoeBlock forward (manual expert dispatch loop).

    Uses the HF Qwen3_5MoeExperts.forward dispatch logic instead of torch_moe.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    # Router (HF-style: softmax -> topk -> renormalize)
    router_logits = F.linear(hidden_states_flat, module.gate.weight)
    routing_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(routing_probs, module.gate.top_k, dim=-1)
    router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(hidden_states_flat.dtype)

    # Expert dispatch (HF loop from Qwen3_5MoeExperts.forward)
    gate_up_proj = module.experts.gate_up_proj  # [E, 2*I, H]
    down_proj = module.experts.down_proj  # [E, H, I]
    num_experts = module.experts.num_experts

    final_hidden_states = torch.zeros_like(hidden_states_flat)
    with torch.no_grad():
        expert_mask = F.one_hot(router_indices, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states_flat[token_idx]
        gate, up = F.linear(current_state, gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = F.silu(gate) * up
        current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
        current_hidden_states = current_hidden_states * router_top_value[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )

    # Shared expert with sigmoid gating
    shared_expert_output = module.shared_expert(hidden_states_flat)
    shared_expert_output = (
        F.sigmoid(module.shared_expert_gate(hidden_states_flat)) * shared_expert_output
    )
    final_hidden_states = final_hidden_states + shared_expert_output

    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


def ref_decoder_layer_forward(module, hidden_states, position_embeddings):
    """HF-style DecoderLayer forward using reference implementations.

    Layernorms are identical (pure PyTorch), so we use the module's norms directly.
    Only the attention/GDN and MoE blocks use the reference code paths.
    """
    residual = hidden_states
    hidden_states = module.input_layernorm(hidden_states)

    if module.layer_type == "linear_attention":
        hidden_states = ref_gdn_forward(module.linear_attn, hidden_states)
    elif module.layer_type == "full_attention":
        hidden_states = ref_attention_forward(module.self_attn, hidden_states, position_embeddings)

    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = module.post_attention_layernorm(hidden_states)
    hidden_states = ref_moe_forward(module.mlp, hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


# =============================================================================
# Vision Reference Forward Functions
# =============================================================================
# These re-implement the HF vision forward logic using our module's weights
# in pure PyTorch. They validate that our simplified vision port produces
# numerically equivalent results.


def ref_vision_attention_forward(module, hidden_states, cu_seqlens, position_embeddings):
    """HF-style VisionAttention forward (eager path, non-causal).

    Re-implements the attention using explicit matmul SDPA, operating on the
    same weights as the module. This validates the QKV projection, RoPE
    application, per-chunk variable-length attention, and output projection.
    """
    seq_length = hidden_states.shape[0]

    # QKV projection and reshape (same as HF: reshape to (S, 3, H, D), permute, unbind)
    query_states, key_states, value_states = (
        F.linear(hidden_states, module.qkv.weight, module.qkv.bias)
        .reshape(seq_length, 3, module.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )

    # Apply vision RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    # Transpose to (1, H, S, D) for batched attention
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    # Split by variable-length sequences via cu_seqlens
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    q_splits = torch.split(query_states, lengths, dim=2)
    k_splits = torch.split(key_states, lengths, dim=2)
    v_splits = torch.split(value_states, lengths, dim=2)

    # Per-chunk non-causal attention (HF eager path)
    scaling = module.scaling
    attn_outputs = []
    for q, k, v in zip(q_splits, k_splits, v_splits):
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_outputs.append(torch.matmul(attn_weights, v))

    attn_output = torch.cat(attn_outputs, dim=2)  # (1, H, total_S, D)
    attn_output = attn_output.squeeze(0).transpose(0, 1)  # (S, H, D)
    attn_output = attn_output.reshape(seq_length, -1).contiguous()

    # Output projection
    attn_output = F.linear(attn_output, module.proj.weight, module.proj.bias)
    return attn_output


def ref_vision_block_forward(module, hidden_states, cu_seqlens, position_embeddings):
    """HF-style VisionBlock forward using reference attention.

    Composes: hidden_states + ref_attn(norm1(hidden_states)) + mlp(norm2(...))
    LayerNorm and MLP are identical to our implementation (pure PyTorch, no
    custom ops), so we use them directly from the module.
    """
    hidden_states = hidden_states + ref_vision_attention_forward(
        module.attn, module.norm1(hidden_states), cu_seqlens, position_embeddings
    )
    hidden_states = hidden_states + module.mlp(module.norm2(hidden_states))
    return hidden_states


def ref_vision_model_forward(module, hidden_states, grid_thw):
    """HF-style VisionModel forward using reference block implementations.

    Re-implements the full pipeline:
      1. PatchEmbed (same, no simplification)
      2. fast_pos_embed_interpolate (same, no simplification)
      3. rot_pos_emb -> (cos, sin)
      4. cu_seqlens from grid_thw
      5. Loop blocks using ref_vision_block_forward
      6. Merger (same, no simplification)

    This validates that the pipeline composition is correct and that the
    intermediate tensor shapes flow correctly through all stages.
    """
    # 1. Patch embedding
    hidden_states = module.patch_embed(hidden_states)

    # 2. Learned positional embeddings (bilinear interpolation)
    pos_embeds = module.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    # 3. Rotary position embeddings for vision
    rotary_pos_emb = module.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # 4. Compute cu_seqlens from grid_thw
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # 5. Loop through vision blocks using reference implementation
    for blk in module.blocks:
        hidden_states = ref_vision_block_forward(
            blk, hidden_states, cu_seqlens, position_embeddings
        )

    # 6. Patch merger
    merged_hidden_states = module.merger(hidden_states)
    return hidden_states, merged_hidden_states


def ref_multimodal_forward(model, input_ids, pixel_values=None, image_grid_thw=None):
    """HF-style multimodal forward using reference sub-pipelines.

    Re-implements the Qwen3_5MoeModel.forward step by step:
      1. embed_tokens(input_ids) -> inputs_embeds
      2. vision model on pixel_values -> image_embeds
      3. masked_scatter image_embeds into inputs_embeds
      4. get_rope_index -> position_ids (3, B, S)
      5. rotary_emb -> (cos, sin)
      6. language_model(inputs_embeds, position_embeddings)

    This validates the multimodal wrapper orchestration: embedding merge,
    mRoPE position ID computation, and correct forwarding to the LM.
    """
    # 1. Embed text tokens
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # 2-3. Vision encoding + embedding merge
    if pixel_values is not None and image_grid_thw is not None:
        vision_output = model.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
        image_embeds_list = list(torch.split(image_embeds, split_sizes))
        image_embeds = torch.cat(image_embeds_list, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask = (
            (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # 4. Compute mRoPE position IDs
    position_ids, _ = model.get_rope_index(input_ids, image_grid_thw)

    # 5. Compute cos/sin from rotary embedding
    position_embeddings = model.rotary_emb(inputs_embeds, position_ids)

    # 6. Call language model with pre-computed position embeddings
    return model.language_model(
        inputs_embeds=inputs_embeds,
        position_embeddings=position_embeddings,
    )


# =============================================================================
# HF Reference Comparison Tests
# =============================================================================


@torch.no_grad()
def test_gdn_matches_hf_reference():
    """Compare GatedDeltaNet custom-ops forward against HF reference torch fallback."""
    config = _make_small_config()
    torch.manual_seed(42)
    gdn = Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
    gdn.eval()

    B, S = 2, 8
    hidden_states = torch.randn(B, S, config.hidden_size)

    our_output = gdn(hidden_states)
    ref_output = ref_gdn_forward(gdn, hidden_states)

    torch.testing.assert_close(
        our_output,
        ref_output,
        rtol=1e-4,
        atol=1e-4,
        msg="GatedDeltaNet custom-ops forward should match HF reference",
    )


@torch.no_grad()
def test_attention_matches_hf_reference():
    """Compare Attention custom-ops forward against HF reference eager attention."""
    config = _make_small_config()
    torch.manual_seed(42)
    # Layer 3 is full_attention in the default 4-layer config
    attn = Qwen3_5MoeAttention(config, layer_idx=3)
    attn.eval()

    rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config)

    B, S = 2, 8
    hidden_states = torch.randn(B, S, config.hidden_size)
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    position_ids = position_ids[None, ...].expand(3, B, -1)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    our_output = attn(hidden_states, position_embeddings)
    ref_output = ref_attention_forward(attn, hidden_states, position_embeddings)

    torch.testing.assert_close(
        our_output,
        ref_output,
        rtol=1e-5,
        atol=1e-5,
        msg="Attention custom-ops forward should match HF reference",
    )


@torch.no_grad()
def test_moe_matches_hf_reference():
    """Compare SparseMoeBlock torch_moe forward against HF reference expert dispatch."""
    config = _make_small_config()
    torch.manual_seed(42)
    moe = Qwen3_5MoeSparseMoeBlock(config)
    moe.eval()
    _init_block_weights(moe)

    B, S = 2, 8
    hidden_states = torch.randn(B, S, config.hidden_size)

    our_output = moe(hidden_states)
    ref_output = ref_moe_forward(moe, hidden_states)

    torch.testing.assert_close(
        our_output,
        ref_output,
        rtol=1e-5,
        atol=1e-5,
        msg="SparseMoeBlock torch_moe forward should match HF reference",
    )


@torch.no_grad()
def test_decoder_layer_matches_hf_reference():
    """Compare DecoderLayer forward against HF reference for both layer types."""
    config = _make_small_config()
    rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config)

    B, S = 2, 8
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    position_ids = position_ids[None, ...].expand(3, B, -1)

    # Test one linear_attention layer (idx 0) and one full_attention layer (idx 3)
    for layer_idx in [0, 3]:
        torch.manual_seed(42)
        layer = Qwen3_5MoeDecoderLayer(config, layer_idx)
        layer.eval()
        _init_block_weights(layer)

        hidden_states = torch.randn(B, S, config.hidden_size)
        position_embeddings = rotary_emb(hidden_states, position_ids)

        our_output = layer(hidden_states, position_embeddings)
        ref_output = ref_decoder_layer_forward(layer, hidden_states, position_embeddings)

        torch.testing.assert_close(
            our_output,
            ref_output,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                f"DecoderLayer {layer_idx} ({config.layer_types[layer_idx]}) "
                f"custom-ops forward should match HF reference"
            ),
        )


# =============================================================================
# RMSNorm Load Hook Correctness Test
# =============================================================================


@torch.no_grad()
def test_rmsnorm_load_hook_matches_original():
    """Verify modified Qwen3_5MoeRMSNorm (load hook + simplified forward)
    produces identical output to the original (1 + weight) formulation."""
    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

    dim = 64
    eps = 1e-6

    # Create original (HF-style) norm with learned weights
    torch.manual_seed(42)
    original = OriginalQwen3_5MoeRMSNorm(dim, eps)
    original.weight.data.normal_(mean=0.0, std=0.1)  # non-trivial weights

    # Create modified norm with load hook, load original's state dict
    modified = Qwen3_5MoeRMSNorm(dim, eps)
    modified.load_state_dict(original.state_dict())

    # The load hook should have added 1.0:
    # modified.weight == original.weight + 1.0
    torch.testing.assert_close(
        modified.weight.data,
        original.weight.data + 1.0,
        msg="Load hook should offset weight by +1.0",
    )

    # Forward pass: both should produce identical outputs
    x = torch.randn(2, 8, dim)
    out_original = original(x)
    out_modified = modified(x)

    torch.testing.assert_close(
        out_original,
        out_modified,
        rtol=1e-5,
        atol=1e-5,
        msg="Modified RMSNorm with load hook should match original (1+weight) RMSNorm",
    )


# =============================================================================
# Vision & Multimodal Config Helpers
# =============================================================================


def _make_small_vision_config(**overrides) -> Qwen3_5MoeVisionConfig:
    """Small vision config for fast tests."""
    defaults = dict(
        depth=2,
        hidden_size=32,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=64,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,  # 4x4 grid
        initializer_range=0.02,
    )
    defaults.update(overrides)
    return Qwen3_5MoeVisionConfig(**defaults)


def _make_small_composite_config(**overrides) -> Qwen3_5MoeConfig:
    """Small composite config (text + vision) for fast tests."""
    text_config = _make_small_config()
    vision_config = _make_small_vision_config(out_hidden_size=text_config.hidden_size)
    defaults = dict(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
    )
    defaults.update(overrides)
    return Qwen3_5MoeConfig(**defaults)


# =============================================================================
# Vision Reference Comparison Tests
# =============================================================================


def _make_vision_test_inputs(vision_config, grid_thw):
    """Create test inputs for vision modules: pixel_values, cu_seqlens, position_embeddings."""
    total_patches = int(torch.prod(grid_thw, dim=1).sum().item())
    t_ps, ps = vision_config.temporal_patch_size, vision_config.patch_size
    pixel_values = torch.randn(total_patches, vision_config.in_channels * t_ps * ps * ps)
    return pixel_values


@torch.no_grad()
def test_vision_attention_matches_reference():
    """Compare VisionAttention forward against HF reference eager attention."""
    vision_config = _make_small_vision_config()
    torch.manual_seed(42)
    attn = Qwen3_5MoeVisionAttention(vision_config)
    attn.eval()

    # 2 variable-length sequences: 6 tokens and 4 tokens
    S = 10
    hidden_states = torch.randn(S, vision_config.hidden_size)
    cu_seqlens = torch.tensor([0, 6, 10], dtype=torch.int32)

    # Compute vision RoPE position embeddings
    head_dim = vision_config.hidden_size // vision_config.num_heads
    rotary_dim = head_dim // 2
    freqs = torch.randn(S, rotary_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    our_output = attn(hidden_states, cu_seqlens, position_embeddings)
    ref_output = ref_vision_attention_forward(attn, hidden_states, cu_seqlens, position_embeddings)

    torch.testing.assert_close(
        our_output,
        ref_output,
        rtol=1e-5,
        atol=1e-5,
        msg="VisionAttention forward should match HF reference eager attention",
    )


@torch.no_grad()
def test_vision_block_matches_reference():
    """Compare VisionBlock forward against HF reference (ref attention + MLP)."""
    vision_config = _make_small_vision_config()
    torch.manual_seed(42)
    block = Qwen3_5MoeVisionBlock(vision_config)
    block.eval()

    S = 10
    hidden_states = torch.randn(S, vision_config.hidden_size)
    cu_seqlens = torch.tensor([0, 6, 10], dtype=torch.int32)

    head_dim = vision_config.hidden_size // vision_config.num_heads
    rotary_dim = head_dim // 2
    freqs = torch.randn(S, rotary_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    our_output = block(hidden_states, cu_seqlens, position_embeddings)
    ref_output = ref_vision_block_forward(block, hidden_states, cu_seqlens, position_embeddings)

    torch.testing.assert_close(
        our_output,
        ref_output,
        rtol=1e-5,
        atol=1e-5,
        msg="VisionBlock forward should match HF reference",
    )


@torch.no_grad()
def test_vision_model_matches_reference():
    """Compare VisionModel forward against HF reference for a single image."""
    vision_config = _make_small_vision_config()
    torch.manual_seed(42)
    vision_model = Qwen3_5MoeVisionModel(vision_config)
    vision_model.eval()

    # Single image: T=1, H=4, W=4 -> 16 patches
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    pixel_values = _make_vision_test_inputs(vision_config, grid_thw)

    our_output = vision_model(pixel_values, grid_thw)
    ref_last_hidden, ref_pooler = ref_vision_model_forward(vision_model, pixel_values, grid_thw)

    torch.testing.assert_close(
        our_output.last_hidden_state,
        ref_last_hidden,
        rtol=1e-5,
        atol=1e-5,
        msg="VisionModel last_hidden_state should match HF reference",
    )
    torch.testing.assert_close(
        our_output.pooler_output,
        ref_pooler,
        rtol=1e-5,
        atol=1e-5,
        msg="VisionModel pooler_output should match HF reference",
    )


@torch.no_grad()
def test_vision_model_multi_image_matches_reference():
    """Compare VisionModel forward against HF reference for multiple images."""
    vision_config = _make_small_vision_config()
    torch.manual_seed(42)
    vision_model = Qwen3_5MoeVisionModel(vision_config)
    vision_model.eval()

    # 2 images of different sizes: T=1,H=4,W=4 (16 patches) + T=1,H=2,W=4 (8 patches)
    grid_thw = torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.long)
    pixel_values = _make_vision_test_inputs(vision_config, grid_thw)

    our_output = vision_model(pixel_values, grid_thw)
    ref_last_hidden, ref_pooler = ref_vision_model_forward(vision_model, pixel_values, grid_thw)

    torch.testing.assert_close(
        our_output.last_hidden_state,
        ref_last_hidden,
        rtol=1e-5,
        atol=1e-5,
        msg="VisionModel multi-image last_hidden_state should match HF reference",
    )
    torch.testing.assert_close(
        our_output.pooler_output,
        ref_pooler,
        rtol=1e-5,
        atol=1e-5,
        msg="VisionModel multi-image pooler_output should match HF reference",
    )


# =============================================================================
# Position Embeddings Passthrough Tests
# =============================================================================


@torch.no_grad()
def test_position_embeddings_passthrough():
    """Test that passing pre-computed position_embeddings produces the same output
    as computing them internally from position_ids."""
    config = _make_small_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeTextModel(config)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    # Path 1: Let the model compute position_embeddings internally
    output_internal = model(input_ids=input_ids, position_ids=position_ids)

    # Path 2: Compute position_embeddings externally and pass them in
    inputs_embeds = model.embed_tokens(input_ids)
    position_ids_3d = position_ids[None, ...].expand(3, B, -1)
    cos, sin = model.rotary_emb(inputs_embeds, position_ids_3d)
    output_external = model(input_ids=input_ids, position_embeddings=(cos, sin))

    torch.testing.assert_close(
        output_internal.last_hidden_state,
        output_external.last_hidden_state,
        rtol=1e-5,
        atol=1e-5,
        msg="External position_embeddings should produce identical output to internal computation",
    )


@torch.no_grad()
def test_rope_cos_sin_kwargs():
    """Test that passing rope_cos/rope_sin as separate kwargs produces the same
    output as using position_embeddings tuple or internal computation."""
    config = _make_small_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeTextModel(config)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    # Reference: internal computation
    output_ref = model(input_ids=input_ids, position_ids=position_ids)

    # Compute cos/sin externally
    inputs_embeds = model.embed_tokens(input_ids)
    position_ids_3d = position_ids[None, ...].expand(3, B, -1)
    cos, sin = model.rotary_emb(inputs_embeds, position_ids_3d)

    # Test: pass as separate kwargs (export-friendly path)
    output_kwargs = model(input_ids=input_ids, rope_cos=cos, rope_sin=sin)

    torch.testing.assert_close(
        output_ref.last_hidden_state,
        output_kwargs.last_hidden_state,
        rtol=1e-5,
        atol=1e-5,
        msg="rope_cos/rope_sin kwargs should produce identical output to internal computation",
    )


@torch.no_grad()
def test_causal_lm_position_embeddings():
    """Test that Qwen3_5MoeForCausalLM correctly passes position_embeddings through."""
    config = _make_small_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeForCausalLM(config)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    # Reference output using position_ids
    output_ref = model(input_ids=input_ids, position_ids=position_ids)

    # Compute cos/sin externally
    inputs_embeds = model.model.embed_tokens(input_ids)
    position_ids_3d = position_ids[None, ...].expand(3, B, -1)
    cos, sin = model.model.rotary_emb(inputs_embeds, position_ids_3d)

    # Test: pass as position_embeddings tuple
    output_tuple = model(input_ids=input_ids, position_embeddings=(cos, sin))

    # Test: pass as separate rope_cos/rope_sin
    output_kwargs = model(input_ids=input_ids, rope_cos=cos, rope_sin=sin)

    torch.testing.assert_close(
        output_ref.logits,
        output_tuple.logits,
        rtol=1e-5,
        atol=1e-5,
        msg="CausalLM position_embeddings tuple path should match",
    )
    torch.testing.assert_close(
        output_ref.logits,
        output_kwargs.logits,
        rtol=1e-5,
        atol=1e-5,
        msg="CausalLM rope_cos/rope_sin path should match",
    )


# =============================================================================
# get_rope_index Tests
# =============================================================================


@torch.no_grad()
def test_get_rope_index_text_only():
    """Test get_rope_index for a text-only sequence (no vision tokens)."""
    config = _make_small_composite_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeModel(config)

    B, S = 2, 8
    input_ids = torch.randint(0, 50, (B, S))  # no special tokens

    position_ids, deltas = model.get_rope_index(input_ids)

    # For text-only: all 3 mRoPE dimensions should be identical
    assert position_ids.shape == (3, B, S), f"Expected (3, {B}, {S}), got {position_ids.shape}"
    torch.testing.assert_close(
        position_ids[0],
        position_ids[1],
        msg="Text-only: T and H position IDs should be identical",
    )
    torch.testing.assert_close(
        position_ids[0],
        position_ids[2],
        msg="Text-only: T and W position IDs should be identical",
    )
    # Should be 0, 1, 2, ..., S-1
    expected = torch.arange(S).unsqueeze(0).expand(B, -1)
    torch.testing.assert_close(
        position_ids[0],
        expected,
        msg="Text-only position IDs should be sequential",
    )
    # Deltas should be 0 for text-only
    assert (deltas == 0).all(), f"Expected zero deltas, got {deltas}"


@torch.no_grad()
def test_get_rope_index_with_image():
    """Test get_rope_index for a sequence containing image placeholders."""
    config = _make_small_composite_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeModel(config)

    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id

    # Build a sequence: [text, vision_start, image_token*4, text]
    # Grid: T=1, H=2, W=2 -> after spatial merge (2x2): 1 merged token
    # But position IDs are for pre-merge (1*1*1 = 1 spatial position)
    # Actually for the LLM: grid_h // merge = 2//2=1, grid_w // merge = 2//2=1
    # so 1*1*1 = 1 vision position in position_ids
    # Image tokens in input_ids: 1 token (1 merged vision token placeholder)
    input_ids = torch.tensor(
        [
            [
                1,
                2,
                3,  # 3 text tokens
                vision_start_token_id,
                image_token_id,  # 1 vision placeholder
                4,
                5,  # 2 text tokens
            ]
        ]
    )
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)

    position_ids, deltas = model.get_rope_index(input_ids, image_grid_thw=image_grid_thw)

    assert position_ids.shape == (3, 1, 7), f"Expected (3, 1, 7), got {position_ids.shape}"
    # Text before image: positions 0,1,2
    assert position_ids[0, 0, 0] == 0
    assert position_ids[0, 0, 1] == 1
    assert position_ids[0, 0, 2] == 2
    # vision_start_token: position 3
    assert position_ids[0, 0, 3] == 3
    # Image token: T dimension should use temporal index
    # The vision token position: st_idx = 4 (text_len=4 since ed=4), T=4, H=0, W=0
    # After image: trailing text should continue sequentially


@torch.no_grad()
def test_get_rope_index_with_attention_mask():
    """Test get_rope_index respects attention_mask for text-only."""
    config = _make_small_composite_config()
    model = Qwen3_5MoeModel(config)

    B, S = 1, 6
    input_ids = torch.randint(0, 50, (B, S))
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])  # last 2 tokens are padding

    position_ids, deltas = model.get_rope_index(input_ids, attention_mask=attention_mask)

    assert position_ids.shape == (3, B, S)
    # Non-padded positions: 0, 1, 2, 3
    assert position_ids[0, 0, 0] == 0
    assert position_ids[0, 0, 3] == 3
    # Padded positions should be 1 (masked fill value)
    assert position_ids[0, 0, 4] == 1
    assert position_ids[0, 0, 5] == 1


# =============================================================================
# Multimodal Reference Comparison Tests
# =============================================================================


@torch.no_grad()
def test_multimodal_forward_matches_reference():
    """Compare multimodal wrapper forward against step-by-step HF reference.

    Tests the full pipeline: vision encoding + embedding merge + mRoPE + LM.
    Runs the same input through both the wrapper's forward() and the
    ref_multimodal_forward() that re-implements each step explicitly.
    """
    config = _make_small_composite_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeForConditionalGeneration(config)
    model.eval()

    image_token_id = config.image_token_id
    vision_start_token_id = config.vision_start_token_id
    vc = config.vision_config

    # Vision: 1 image, T=1, H=4, W=4 -> 16 patches -> 4 merged tokens
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    num_patches = 1 * 4 * 4
    t_ps, ps = vc.temporal_patch_size, vc.patch_size
    pixel_values = torch.randn(num_patches, vc.in_channels * t_ps * ps * ps)

    merge_factor = vc.spatial_merge_size**2
    num_merged_tokens = num_patches // merge_factor  # 4

    # Build input_ids: [text, vision_start, image_token * num_merged_tokens, text]
    text_before = [1, 2, 3]
    vision_part = [vision_start_token_id] + [image_token_id] * num_merged_tokens
    text_after = [4, 5]
    input_ids = torch.tensor([text_before + vision_part + text_after])

    # Our wrapper forward
    our_output = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    # Reference: step-by-step re-implementation
    ref_output = ref_multimodal_forward(
        model.model, input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
    )

    torch.testing.assert_close(
        our_output.logits,
        ref_output.logits,
        rtol=1e-5,
        atol=1e-5,
        msg="Multimodal forward should match step-by-step HF reference",
    )


# =============================================================================
# torch.export Tests
# =============================================================================


@torch.no_grad()
def test_export_text_model_with_position_ids():
    """Test that the text model can be exported with standard position_ids."""
    config = _make_small_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeForCausalLM(config)
    model.eval()

    B, S = 2, 4
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)

    # Run normally first
    ref_output = model(input_ids=input_ids, position_ids=position_ids)

    # Export with Dim.AUTO to let torch.export infer dynamic vs static
    from torch.export import Dim

    dynamic_shapes = {
        "input_ids": {0: Dim.AUTO, 1: Dim.AUTO},
        "position_ids": {0: Dim.AUTO, 1: Dim.AUTO},
    }

    exported = torch.export.export(
        model,
        (),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # Run exported model
    export_output = exported.module()(input_ids=input_ids, position_ids=position_ids)
    torch.testing.assert_close(
        ref_output.logits,
        export_output.logits,
        rtol=1e-5,
        atol=1e-5,
        msg="Exported model should produce same output as eager model",
    )


@torch.no_grad()
def test_export_text_model_with_rope_cos_sin():
    """Test that the text model can be exported with rope_cos/rope_sin as extra inputs.

    This validates the Option 3 mRoPE export path: cos/sin are computed
    externally and passed as separate tensor kwargs.
    """
    config = _make_small_config()
    torch.manual_seed(42)
    model = Qwen3_5MoeForCausalLM(config)
    model.eval()

    B, S = 2, 4
    inputs_embeds = torch.randn(B, S, config.hidden_size)
    position_ids_3d = torch.arange(S).view(1, 1, -1).expand(3, B, -1)
    cos, sin = model.model.rotary_emb(inputs_embeds, position_ids_3d)

    # Reference: pass cos/sin via rope_cos/rope_sin
    ref_output = model(inputs_embeds=inputs_embeds, rope_cos=cos, rope_sin=sin)

    # Export with cos/sin as inputs, using Dim.AUTO
    from torch.export import Dim

    dynamic_shapes = {
        "inputs_embeds": {0: Dim.AUTO, 1: Dim.AUTO},
        "rope_cos": {0: Dim.AUTO, 1: Dim.AUTO},
        "rope_sin": {0: Dim.AUTO, 1: Dim.AUTO},
    }

    exported = torch.export.export(
        model,
        (),
        kwargs={"inputs_embeds": inputs_embeds, "rope_cos": cos, "rope_sin": sin},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    export_output = exported.module()(inputs_embeds=inputs_embeds, rope_cos=cos, rope_sin=sin)
    torch.testing.assert_close(
        ref_output.logits,
        export_output.logits,
        rtol=1e-5,
        atol=1e-5,
        msg="Exported model with rope_cos/rope_sin should match eager output",
    )

    # Also test with different batch/seq dims to verify dynamic shapes work
    B2, S2 = 3, 8
    inputs_embeds2 = torch.randn(B2, S2, config.hidden_size)
    position_ids_3d2 = torch.arange(S2).view(1, 1, -1).expand(3, B2, -1)
    cos2, sin2 = model.model.rotary_emb(inputs_embeds2, position_ids_3d2)

    ref_output2 = model(inputs_embeds=inputs_embeds2, rope_cos=cos2, rope_sin=sin2)
    export_output2 = exported.module()(inputs_embeds=inputs_embeds2, rope_cos=cos2, rope_sin=sin2)
    torch.testing.assert_close(
        ref_output2.logits,
        export_output2.logits,
        rtol=1e-5,
        atol=1e-5,
        msg="Dynamic shape export should work with different batch/seq dims",
    )
