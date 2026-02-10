"""Tests for the custom Qwen3.5 MoE model implementation.

Validates that the prefill-only custom model (with autodeploy custom ops)
produces numerically equivalent results to the HuggingFace reference
implementation (pure-PyTorch fallback paths) for each major block:
  1. GatedDeltaNet (linear attention)
  2. Attention (full attention with gating)
  3. SparseMoeBlock (routed + shared experts)
  4. DecoderLayer (composed token mixer + channel mixer)
"""

import pytest
import torch
import torch.nn.functional as F

# Register autodeploy custom ops (torch_moe, torch_causal_conv1d, etc.)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeExperts,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeTopKRouter,
    apply_rotary_pos_emb,
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
