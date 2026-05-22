# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical equivalence tests for GPT-OSS AutoDeploy custom model.

Reference is the HuggingFace GPT-OSS modeling code shipped in transformers
(``transformers.models.gpt_oss``). When that import is unavailable we skip
the affected tests rather than reproduce the math by hand.
"""

from typing import Optional, Tuple

import pytest
import torch
from torch.export import Dim
from transformers import GptOssConfig

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register canonical ops
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssExperts,
    GptOssForCausalLM,
    GptOssMLP,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
    GptOssTopKRouter,
)

# ---------------------------------------------------------------------------
# HF reference imports (skip tests if unavailable)
# ---------------------------------------------------------------------------


def _get_hf_classes():
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention as HFAttention
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer as HFLayer
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts as HFExperts
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM as HFForCausalLM
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP as HFMLP
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRMSNorm as HFRMSNorm
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding as HFRotary
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter as HFRouter

        return {
            "RMSNorm": HFRMSNorm,
            "Rotary": HFRotary,
            "Router": HFRouter,
            "Experts": HFExperts,
            "MLP": HFMLP,
            "Attention": HFAttention,
            "Layer": HFLayer,
            "ForCausalLM": HFForCausalLM,
        }
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_rmse_close(
    actual: torch.Tensor, expected: torch.Tensor, rmse_ratio_tol: float, msg: str = ""
) -> None:
    diff = actual.float() - expected.float()
    rmse_diff = torch.sqrt(torch.mean(diff**2))
    rmse_ref = torch.sqrt(torch.mean(expected.float() ** 2))
    ratio = (rmse_diff / rmse_ref.clamp(min=1e-12)).item()
    assert ratio < rmse_ratio_tol, (
        f"{msg}RMSE ratio {ratio:.6f} exceeds tolerance {rmse_ratio_tol}. "
        f"(rmse_diff={rmse_diff.item():.6f}, rmse_ref={rmse_ref.item():.6f})"
    )


def _device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _position_ids(batch: int, seq: int, device) -> torch.Tensor:
    return torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)


def _small_config(num_layers: int = 3) -> GptOssConfig:
    """Tiny but representative config covering both sliding and full attention.

    We use ``rope_type=default`` (no YaRN) here so equivalence comparisons
    against HF do not have to thread through the rope scaling code path.
    """
    return GptOssConfig(
        num_hidden_layers=num_layers,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=32,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=4,
        max_position_embeddings=64,
        rope_scaling={"rope_type": "default"},
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        attention_bias=True,
        attention_dropout=0.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        pad_token_id=0,
    )


def _rope_position_embeddings(
    config: GptOssConfig, B: int, S: int, device, dtype
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Return (ad_pos_emb, hf_pos_emb).

    AD pos_emb: (cos, sin) of shape ``[B, S, head_dim]`` (Llama-style duplicated).
    HF pos_emb: (cos, sin) of shape ``[B, S, head_dim/2]`` (GPT-OSS native).
    """
    head_dim = config.head_dim
    # AD-side
    rope = GptOssRotaryEmbedding(
        head_dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rope_scaling=config.rope_scaling,
    ).to(device=device)
    pos_ids = _position_ids(B, S, device)
    dummy = torch.zeros(1, device=device, dtype=dtype)
    ad_cos, ad_sin = rope(dummy, pos_ids)  # [B, S, head_dim]

    # HF-side: cos/sin are half-size; equal to the front half of AD's duplicated cache.
    hf_cos = ad_cos[..., : head_dim // 2]
    hf_sin = ad_sin[..., : head_dim // 2]
    return (ad_cos, ad_sin), (hf_cos, hf_sin)


# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------


def test_rmsnorm_equivalence():
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    H = 64
    ref = hf["RMSNorm"](H, eps=1e-5).to(device=device, dtype=dtype).eval()
    ad = GptOssRMSNorm(H, eps=1e-5).to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(2, 8, H, device=device, dtype=dtype)
    with torch.no_grad():
        torch.testing.assert_close(ad(x), ref(x), rtol=1e-3, atol=1e-3)


def test_rotary_embedding_equivalence():
    """Our pre-cached RoPE table matches HF's on-the-fly computation."""
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8

    ad_pos_emb, hf_pos_emb = _rope_position_embeddings(config, B, S, device, dtype)
    ad_cos, ad_sin = ad_pos_emb
    hf_cos_half, hf_sin_half = hf_pos_emb

    # Build HF rotary the way the HF model does and ask it for cos/sin.
    hf_rope = hf["Rotary"](config=config).to(device=device).eval()
    pos_ids = _position_ids(B, S, device)
    dummy = torch.zeros(1, device=device, dtype=dtype)
    with torch.no_grad():
        hf_cos_full, hf_sin_full = hf_rope(dummy, pos_ids)  # [B, S, head_dim/2]
    # The AD cache uses cat((freqs, freqs)); HF returns just freqs. Compare first half.
    torch.testing.assert_close(
        ad_cos[..., : config.head_dim // 2], hf_cos_full, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        ad_sin[..., : config.head_dim // 2], hf_sin_full, rtol=1e-3, atol=1e-3
    )
    # Second half should equal the first half by construction (Llama-style duplication).
    torch.testing.assert_close(
        ad_cos[..., config.head_dim // 2 :], hf_cos_half, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        ad_sin[..., config.head_dim // 2 :], hf_sin_half, rtol=1e-3, atol=1e-3
    )


def test_router_equivalence():
    """torch_moe_router scatter output matches HF's router_scores."""
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()

    ref = hf["Router"](config).to(device=device, dtype=dtype).eval()
    # Initialise so weights aren't garbage from torch.empty.
    torch.nn.init.normal_(ref.weight, std=0.02)
    torch.nn.init.normal_(ref.bias, std=0.02)
    ad = GptOssTopKRouter(config).to(device=device, dtype=dtype).eval()
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(2, 8, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        ref_scores, _ = ref(x)  # [B*S, E]
        ad_scores = ad(x)  # [B*S, E]
    assert_rmse_close(ad_scores, ref_scores, rmse_ratio_tol=1e-3, msg="Router: ")


def _init_experts_(ref_experts, ad_experts):
    """Initialise both experts modules with matching random weights."""
    torch.nn.init.normal_(ref_experts.gate_up_proj, std=0.02)
    torch.nn.init.normal_(ref_experts.gate_up_proj_bias, std=0.02)
    torch.nn.init.normal_(ref_experts.down_proj, std=0.02)
    torch.nn.init.normal_(ref_experts.down_proj_bias, std=0.02)
    ad_experts.load_state_dict(ref_experts.state_dict())


def test_experts_equivalence():
    """torch_moe_dense_mlp matches HF GptOssExperts inference path."""
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8
    T = B * S
    E = config.num_local_experts

    ref = hf["Experts"](config).to(device=device, dtype=dtype).eval()
    ad = GptOssExperts(config).to(device=device, dtype=dtype).eval()
    _init_experts_(ref, ad)

    # Top-k routing weights with the same structure produced by GptOssTopKRouter.
    router_logits = torch.randn(T, E, device=device, dtype=dtype)
    top_v, top_idx = torch.topk(router_logits, config.num_experts_per_tok, dim=-1)
    top_v = torch.softmax(top_v, dim=1, dtype=top_v.dtype)
    routing_weights = torch.zeros_like(router_logits).scatter_(1, top_idx, top_v)

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        # HF GptOssExperts is in inference mode here; it picks the dense bmm branch on CUDA.
        # On CPU it picks the sparse branch — both are mathematically equivalent.
        ref_out = ref(x, router_indices=top_idx, routing_weights=routing_weights)
        ad_out = ad(x, routing_weights)
        ad_out = ad_out.view(B, S, -1)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.02, msg="Experts: ")


def test_mlp_equivalence():
    """Full MoE block (router + experts) matches HF GptOssMLP."""
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()
    B, S = 2, 8

    ref = hf["MLP"](config).to(device=device, dtype=dtype).eval()
    ad = GptOssMLP(config).to(device=device, dtype=dtype).eval()
    # Random init for the ref, then copy across.
    torch.nn.init.normal_(ref.router.weight, std=0.02)
    torch.nn.init.normal_(ref.router.bias, std=0.02)
    torch.nn.init.normal_(ref.experts.gate_up_proj, std=0.02)
    torch.nn.init.normal_(ref.experts.gate_up_proj_bias, std=0.02)
    torch.nn.init.normal_(ref.experts.down_proj, std=0.02)
    torch.nn.init.normal_(ref.experts.down_proj_bias, std=0.02)
    ad.load_state_dict(ref.state_dict())

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        ref_out, _ = ref(x)
        ad_out = ad(x)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.02, msg="MoE block: ")


# ---------------------------------------------------------------------------
# Attention test
# ---------------------------------------------------------------------------


def _init_attention_(ref: torch.nn.Module, ad: torch.nn.Module):
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        torch.nn.init.normal_(getattr(ref, proj).weight, std=0.02)
        if getattr(ref, proj).bias is not None:
            torch.nn.init.normal_(getattr(ref, proj).bias, std=0.02)
    torch.nn.init.normal_(ref.sinks, std=0.02)
    ad.load_state_dict(ref.state_dict())


def _hf_attention_forward(
    hf_attn,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Run the HF GptOssAttention forward in eager mode without a mask.

    The HF attention requires an ``_attn_implementation == 'eager'`` attribute on
    the config (we set it on the test config). We pass ``attention_mask=None`` —
    the eager kernel applies its own causal handling via the sinks normaliser.
    """
    # GPT-OSS HF eager kernel does not apply a causal mask itself when ``attention_mask`` is None.
    # Build a [b,1,s,s] additive mask with -inf on the upper triangle so prefill is causal.
    bsz, q_len, _ = hidden_states.shape
    # We are only ever passing q_len tokens with no past, so s_q == s_k == q_len.
    causal = torch.triu(
        torch.full(
            (q_len, q_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype
        ),
        diagonal=1,
    ).view(1, 1, q_len, q_len)
    out, _ = hf_attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=causal,
    )
    return out


def test_attention_equivalence_full():
    """GQA attention with sinks (full attention layer)."""
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()
    config._attn_implementation = "eager"
    # Use full-attention layer index (config alternates: 0=sliding, 1=full).
    layer_idx = 1
    B, S = 2, 8

    ref = hf["Attention"](config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    ad = GptOssAttention(config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    _init_attention_(ref, ad)
    assert ad.sliding_window is None  # this layer is "full_attention"

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    ad_pos, hf_pos = _rope_position_embeddings(config, B, S, device, dtype)

    with torch.no_grad():
        ad_out = ad(x, position_embeddings=ad_pos)
        ref_out = _hf_attention_forward(ref, x, hf_pos[0], hf_pos[1])
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Attention (full): ")


def test_attention_equivalence_sliding():
    """GQA attention with sinks AND sliding window (sliding layer)."""
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()
    config._attn_implementation = "eager"
    layer_idx = 0  # sliding_attention
    B, S = 2, 8

    ref = hf["Attention"](config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    ad = GptOssAttention(config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    _init_attention_(ref, ad)
    assert ad.sliding_window == config.sliding_window

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    ad_pos, hf_pos = _rope_position_embeddings(config, B, S, device, dtype)

    # Build the HF sliding-window mask: causal AND within window.
    q_pos = torch.arange(S, device=device).unsqueeze(1)
    k_pos = torch.arange(S, device=device).unsqueeze(0)
    diff = q_pos - k_pos
    allowed = (diff >= 0) & (diff < config.sliding_window)
    mask = torch.zeros((1, 1, S, S), device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed.view(1, 1, S, S), float("-inf"))

    with torch.no_grad():
        ad_out = ad(x, position_embeddings=ad_pos)
        ref_out, _ = ref(
            hidden_states=x,
            position_embeddings=hf_pos,
            attention_mask=mask,
        )
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.10, msg="Attention (sliding): ")


# ---------------------------------------------------------------------------
# Decoder layer tests
# ---------------------------------------------------------------------------


def _init_decoder_layer_(ref, ad):
    """Init reference layer with random weights and copy to AD."""
    _init_attention_(ref.self_attn, ad.self_attn)
    # HF GptOssMLP === GptOssTopKRouter + GptOssExperts (same names as ours).
    torch.nn.init.normal_(ref.mlp.router.weight, std=0.02)
    torch.nn.init.normal_(ref.mlp.router.bias, std=0.02)
    torch.nn.init.normal_(ref.mlp.experts.gate_up_proj, std=0.02)
    torch.nn.init.normal_(ref.mlp.experts.gate_up_proj_bias, std=0.02)
    torch.nn.init.normal_(ref.mlp.experts.down_proj, std=0.02)
    torch.nn.init.normal_(ref.mlp.experts.down_proj_bias, std=0.02)
    torch.nn.init.normal_(ref.input_layernorm.weight, std=0.02)
    torch.nn.init.normal_(ref.post_attention_layernorm.weight, std=0.02)
    ad.load_state_dict(ref.state_dict())


def _run_hf_decoder(hf_layer, x: torch.Tensor, hf_pos, sliding_window: Optional[int]):
    """Run HF decoder with explicit causal (and optional sliding) mask."""
    bsz, q_len, _ = x.shape
    if sliding_window is None:
        causal = torch.triu(
            torch.full((q_len, q_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        ).view(1, 1, q_len, q_len)
    else:
        q_pos = torch.arange(q_len, device=x.device).unsqueeze(1)
        k_pos = torch.arange(q_len, device=x.device).unsqueeze(0)
        diff = q_pos - k_pos
        allowed = (diff >= 0) & (diff < sliding_window)
        causal = torch.zeros((1, 1, q_len, q_len), device=x.device, dtype=x.dtype)
        causal = causal.masked_fill(~allowed.view(1, 1, q_len, q_len), float("-inf"))
    out = hf_layer(
        hidden_states=x,
        attention_mask=causal,
        position_embeddings=hf_pos,
    )
    if isinstance(out, tuple):
        out = out[0]
    return out


def test_decoder_layer_equivalence_full():
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()
    config._attn_implementation = "eager"
    layer_idx = 1  # full_attention
    B, S = 2, 8

    ref = hf["Layer"](config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    ad = GptOssDecoderLayer(config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    _init_decoder_layer_(ref, ad)

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    ad_pos, hf_pos = _rope_position_embeddings(config, B, S, device, dtype)

    with torch.no_grad():
        ad_out = ad(x, position_embeddings=ad_pos)
        ref_out = _run_hf_decoder(ref, x, hf_pos, sliding_window=None)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="Decoder (full): ")


def test_decoder_layer_equivalence_sliding():
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    config = _small_config()
    config._attn_implementation = "eager"
    layer_idx = 0  # sliding_attention
    B, S = 2, 8

    ref = hf["Layer"](config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    ad = GptOssDecoderLayer(config, layer_idx=layer_idx).to(device=device, dtype=dtype).eval()
    _init_decoder_layer_(ref, ad)

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    ad_pos, hf_pos = _rope_position_embeddings(config, B, S, device, dtype)

    with torch.no_grad():
        ad_out = ad(x, position_embeddings=ad_pos)
        ref_out = _run_hf_decoder(ref, x, hf_pos, sliding_window=config.sliding_window)
    assert_rmse_close(ad_out, ref_out, rmse_ratio_tol=0.05, msg="Decoder (sliding): ")


# ---------------------------------------------------------------------------
# Full model test
# ---------------------------------------------------------------------------


def _transfer_hf_to_ad_full_model(hf_model, ad_model: GptOssForCausalLM):
    """HF and AD use the same parameter names; load_state_dict directly."""
    sd = hf_model.state_dict()
    missing, unexpected = ad_model.load_state_dict(sd, strict=False)
    # The AD-side rotary buffer (``model.rotary_emb._ad_*_cached``) is non-persistent
    # so it's expected to be missing; HF's ``rotary_emb.inv_freq`` is non-persistent too.
    assert not unexpected, f"Unexpected keys: {unexpected[:10]}"


def test_full_model_equivalence():
    hf = _get_hf_classes()
    if hf is None:
        pytest.skip("transformers.models.gpt_oss not available")
    device, dtype = _device_and_dtype()
    # Use 3 layers so we exercise both sliding and full attention plus an extra.
    config = _small_config(num_layers=3)
    config._attn_implementation = "eager"

    ref = hf["ForCausalLM"](config).to(device=device, dtype=dtype).eval()
    # Random init (default _init_weights leaves stacked params with empty().normal_()
    # which already runs in HFPreTrainedModel; just ensure deterministic seed).
    torch.manual_seed(0)
    for _, p in ref.named_parameters():
        if p.dim() > 0:
            torch.nn.init.normal_(p, std=0.02)

    ad = GptOssForCausalLM(config).to(device=device, dtype=dtype).eval()
    _transfer_hf_to_ad_full_model(ref, ad)

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    with torch.no_grad():
        ref_out = ref(input_ids=input_ids, position_ids=pos_ids, use_cache=False)
        ad_out = ad(input_ids=input_ids, position_ids=pos_ids)

    assert ad_out.logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(ad_out.logits).all()
    assert_rmse_close(ad_out.logits, ref_out.logits, rmse_ratio_tol=0.05, msg="Full model: ")


# ---------------------------------------------------------------------------
# Export test
# ---------------------------------------------------------------------------


def test_export():
    """Model can be exported with torch.export and produces correct output."""
    device = "cpu"
    dtype = torch.float32

    config = _small_config(num_layers=2)
    model = GptOssForCausalLM(config).to(device=device, dtype=dtype).eval()
    torch.manual_seed(0)
    for _, p in model.named_parameters():
        if p.dim() > 0:
            torch.nn.init.normal_(p, std=0.02)

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos_ids = _position_ids(B, S, device)

    dynamic_shapes = {
        "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    }

    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": pos_ids},
        dynamic_shapes=dynamic_shapes,
    )

    with torch.no_grad():
        pre_export_out = model(input_ids=input_ids, position_ids=pos_ids)
        exported_out = gm(input_ids, position_ids=pos_ids)

    logits = (
        exported_out[0]
        if isinstance(exported_out, tuple)
        else getattr(exported_out, "logits", exported_out)
    )
    assert torch.isfinite(logits).all(), "Export produced non-finite values"
    torch.testing.assert_close(logits, pre_export_out.logits, rtol=1e-3, atol=1e-3)

    B2, S2 = 1, 4
    ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    pos2 = _position_ids(B2, S2, device)
    with torch.no_grad():
        out2 = gm(ids2, position_ids=pos2)
    logits2 = out2[0] if isinstance(out2, tuple) else getattr(out2, "logits", out2)
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()
