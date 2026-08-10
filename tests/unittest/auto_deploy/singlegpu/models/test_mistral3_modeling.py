# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical tests for the AutoDeploy Mistral3/Mistral4 custom model path.

The ``HfMistral4*`` classes below are minimal standalone reference classes copied from the public
HF mainline Mistral4 source and trimmed to the prefill-only paths exercised in these tests:
https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/mistral4/modeling_mistral4.py
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.export import Dim
from transformers import Mistral3Config

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import (
    Mistral3ForConditionalGenerationAD,
    Mistral4Attention,
    Mistral4DecoderLayer,
    Mistral4ForCausalLM,
    Mistral4MoE,
    Mistral4RMSNorm,
    Mistral4TextConfig,
)
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.auto_deploy.utils.node_utils import get_all_layer_subgraphs


def assert_rmse_close(
    actual: torch.Tensor, expected: torch.Tensor, rmse_ratio_tol: float, msg: str = ""
) -> None:
    actual = actual.float()
    expected = expected.float()
    rmse = torch.sqrt(torch.mean((actual - expected) ** 2))
    denom = torch.sqrt(torch.mean(expected**2)).clamp_min(1e-8)
    ratio = (rmse / denom).item()
    assert ratio <= rmse_ratio_tol, f"{msg}rmse_ratio={ratio:.6f} > {rmse_ratio_tol:.6f}"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _small_text_config() -> Mistral4TextConfig:
    return Mistral4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_head_dim=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        first_k_dense_replace=0,
        moe_layer_freq=1,
        max_position_embeddings=128,
        rope_parameters={
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 8.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.1,
        },
        pad_token_id=0,
    )


def _small_outer_config() -> Mistral3Config:
    return Mistral3Config(
        text_config=_small_text_config().to_dict(),
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 64,
            "intermediate_size": 128,
            "patch_size": 14,
            "image_size": 28,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "head_dim": 16,
            "num_channels": 3,
            "hidden_act": "silu",
        },
        image_token_index=10,
        spatial_merge_size=2,
        vision_feature_layer=-1,
        projector_hidden_act="gelu",
        multimodal_projector_bias=False,
    )


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(1234)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class HfMistral4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(input_dtype)


class HfMistral4RotaryEmbedding(nn.Module):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        rs = config.rope_scaling
        self.dim = config.qk_rope_head_dim
        self.base = rs.get("rope_theta", 10000.0)
        self.scale = rs.get("factor", 1.0)
        self.beta_fast = rs.get("beta_fast", 32.0)
        self.beta_slow = rs.get("beta_slow", 1.0)
        self.mscale = rs.get("mscale", 1.0)
        self.mscale_all_dim = rs.get("mscale_all_dim", 1.0)
        self.original_max_position_embeddings = rs.get("original_max_position_embeddings", 8192)
        self.max_position_embeddings = config.max_position_embeddings
        self._build_cache()

    @staticmethod
    def _find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    @classmethod
    def _find_correction_range(cls, low_rot, high_rot, dim, base, max_position_embeddings):
        low = math.floor(cls._find_correction_dim(low_rot, dim, base, max_position_embeddings))
        high = math.ceil(cls._find_correction_dim(high_rot, dim, base, max_position_embeddings))
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _get_mscale(scale, mscale):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    def _build_cache(self):
        dim = self.dim
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (
            self.scale * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        low, high = self._find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        mask = 1.0 - torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3),
            0,
            1,
        )
        inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        mscale = self._get_mscale(self.scale, self.mscale) / self._get_mscale(
            self.scale, self.mscale_all_dim
        )
        self.register_buffer("cos_cached", emb.cos() * mscale, persistent=False)
        self.register_buffer("sin_cached", emb.sin() * mscale, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached.to(dtype=x.dtype, device=x.device)[position_ids]
        sin = self.sin_cached.to(dtype=x.dtype, device=x.device)[position_ids]
        return cos, sin


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class HfMistral4MLP(nn.Module):
    def __init__(self, config: Mistral4TextConfig, intermediate_size: int | None = None):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HfMistral4MoEGate(nn.Module):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(torch.empty((config.n_routed_experts, config.hidden_size)))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(config.n_routed_experts), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(
            hidden_states.view(-1, hidden_states.shape[-1]),
            self.weight,
            self.e_score_correction_bias,
        )
        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weight = torch.softmax(topk_logits, dim=-1)
        return topk_idx, topk_weight


class HfMistral4MoE(nn.Module):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        self.gate = HfMistral4MoEGate(config)
        self.experts = nn.ModuleList(
            [
                HfMistral4MLP(config, config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.shared_experts = (
            HfMistral4MLP(config, config.moe_intermediate_size * config.n_shared_experts)
            if config.n_shared_experts
            else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        topk_idx, topk_weight = self.gate(hidden_states)
        output = torch.zeros_like(hidden_states_flat)
        for token_idx in range(hidden_states_flat.shape[0]):
            token = hidden_states_flat[token_idx : token_idx + 1]
            for slot in range(topk_idx.shape[1]):
                expert_idx = int(topk_idx[token_idx, slot])
                expert_output = self.experts[expert_idx](token)
                output[token_idx] += topk_weight[token_idx, slot].to(
                    output.dtype
                ) * expert_output.squeeze(0)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states_flat)
        return output.view_as(hidden_states)


class HfMistral4Attention(nn.Module):
    def __init__(self, config: Mistral4TextConfig, layer_idx: int):
        super().__init__()
        del layer_idx
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = HfMistral4RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = HfMistral4RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.rotary_emb = HfMistral4RotaryEmbedding(config)
        self.softmax_scale = self.q_head_dim ** (-0.5)
        mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0.0)
        if mscale_all_dim:
            yarn_scale = HfMistral4RotaryEmbedding._get_mscale(
                config.rope_scaling["factor"], mscale_all_dim
            )
            self.softmax_scale = self.softmax_scale * yarn_scale * yarn_scale

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(batch_size, seq_len, 1, self.qk_rope_head_dim)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q_pe, k_pe = _apply_rope(q_pe, k_pe, cos, sin)

        kv_expanded = F.linear(compressed_kv, self.kv_b_proj.weight)
        kv_expanded = kv_expanded.view(
            batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q = torch.cat([q_nope, q_pe], dim=-1).permute(0, 2, 1, 3).float()
        k = (
            torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)
            .permute(0, 2, 1, 3)
            .float()
        )
        v = v.permute(0, 2, 1, 3).float()

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v).permute(0, 2, 1, 3).to(hidden_states.dtype)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(output)


class HfMistral4DecoderLayer(nn.Module):
    def __init__(self, config: Mistral4TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = HfMistral4Attention(config, layer_idx)
        self.input_layernorm = HfMistral4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HfMistral4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = HfMistral4MoE(config)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class HfMistral4Model(nn.Module):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HfMistral4DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = HfMistral4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, inputs_embeds=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        return self.norm(hidden_states)


class HfMistral4ForCausalLM(nn.Module):
    def __init__(self, config: Mistral4TextConfig):
        super().__init__()
        self.model = HfMistral4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, inputs_embeds=None):
        hidden_states = self.model(input_ids, position_ids, inputs_embeds=inputs_embeds)
        return F.linear(hidden_states, self.lm_head.weight).float()


def _fused_expert_checkpoint_from_moe(moe_module: Mistral4MoE) -> dict[str, torch.Tensor]:
    gate_up = []
    down = []
    for expert in moe_module.experts:
        gate_up.append(torch.cat([expert.gate_proj.weight, expert.up_proj.weight], dim=0))
        down.append(expert.down_proj.weight)
    return {
        "experts.gate_up_proj": torch.stack(gate_up, dim=0),
        "experts.down_proj": torch.stack(down, dim=0),
    }


def _fused_expert_scale_checkpoint_from_moe(moe_module: Mistral4MoE) -> dict[str, torch.Tensor]:
    num_experts = len(moe_module.experts)
    return {
        "experts.gate_up_proj_scale_inv": torch.arange(
            1, num_experts + 1, dtype=torch.float32
        ).view(num_experts, 1, 1),
        "experts.down_proj_scale_inv": torch.arange(
            101, 101 + num_experts, dtype=torch.float32
        ).view(num_experts, 1, 1),
        "experts.gate_up_proj_activation_scale": torch.arange(
            201, 201 + num_experts, dtype=torch.float32
        ),
        "experts.down_proj_activation_scale": torch.arange(
            301, 301 + num_experts, dtype=torch.float32
        ),
    }


def _expand_fused_moe_checkpoint(fused_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expanded = {}
    gate_up = fused_state["experts.gate_up_proj"]
    down = fused_state["experts.down_proj"]
    intermediate_size = gate_up.shape[1] // 2
    for idx in range(gate_up.shape[0]):
        expanded[f"experts.{idx}.gate_proj.weight"] = gate_up[idx, :intermediate_size]
        expanded[f"experts.{idx}.up_proj.weight"] = gate_up[idx, intermediate_size:]
        expanded[f"experts.{idx}.down_proj.weight"] = down[idx]
    return expanded


def _load_reference_block(reference: nn.Module, ad_block: nn.Module) -> None:
    state_dict = ad_block.state_dict()
    if hasattr(ad_block, "experts"):
        fused = _fused_expert_checkpoint_from_moe(ad_block)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("experts.")}
        state_dict.update(_expand_fused_moe_checkpoint(fused))
    reference.load_state_dict(state_dict, strict=True)


def test_rmsnorm_equivalence():
    device = _device()
    dtype = torch.bfloat16
    ad_module = Mistral4RMSNorm(64).to(device=device, dtype=dtype)
    ref_module = HfMistral4RMSNorm(64).to(device=device, dtype=dtype)
    _load_reference_block(ref_module, ad_module)
    hidden_states = torch.randn(2, 5, 64, device=device, dtype=dtype)
    actual = ad_module(hidden_states)
    expected = ref_module(hidden_states)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_attention_equivalence():
    device = _device()
    dtype = torch.bfloat16
    config = _small_text_config()
    ad_module = Mistral4Attention(config, layer_idx=0).to(device=device, dtype=dtype)
    ref_module = HfMistral4Attention(config, layer_idx=0).to(device=device, dtype=dtype)
    _load_reference_block(ref_module, ad_module)
    hidden_states = torch.randn(2, 6, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(6, device=device).unsqueeze(0).expand(2, -1)
    actual = ad_module(hidden_states, position_ids)
    expected = ref_module(hidden_states, position_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.10, msg="Attention: ")


def test_moe_equivalence_and_converter():
    device = _device()
    dtype = torch.bfloat16
    config = _small_text_config()
    ad_module = Mistral4MoE(config).to(device=device, dtype=dtype)
    ref_module = HfMistral4MoE(config).to(device=device, dtype=dtype)
    _load_reference_block(ref_module, ad_module)
    hidden_states = torch.randn(2, 4, config.hidden_size, device=device, dtype=dtype)
    actual = ad_module(hidden_states)
    expected = ref_module(hidden_states)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.02, msg="MoE: ")


def test_moe_fused_checkpoint_hook_expands_static_fp8_scales():
    config = _small_text_config()
    moe = Mistral4MoE(config)

    state_dict = {}
    state_dict.update(_fused_expert_checkpoint_from_moe(moe))
    state_dict.update(_fused_expert_scale_checkpoint_from_moe(moe))

    moe._load_experts_from_fused_checkpoint(state_dict, "")

    for idx in range(config.n_routed_experts):
        gate_scale = state_dict[f"experts.{idx}.gate_proj.weight_scale"]
        up_scale = state_dict[f"experts.{idx}.up_proj.weight_scale"]
        down_scale = state_dict[f"experts.{idx}.down_proj.weight_scale"]
        gate_input_scale = state_dict[f"experts.{idx}.gate_proj.input_scale"]
        up_input_scale = state_dict[f"experts.{idx}.up_proj.input_scale"]
        down_input_scale = state_dict[f"experts.{idx}.down_proj.input_scale"]

        assert gate_scale.shape == torch.Size([])
        assert up_scale.shape == torch.Size([])
        assert down_scale.shape == torch.Size([])
        torch.testing.assert_close(gate_scale, torch.tensor(idx + 1, dtype=torch.float32))
        torch.testing.assert_close(up_scale, torch.tensor(idx + 1, dtype=torch.float32))
        torch.testing.assert_close(down_scale, torch.tensor(101 + idx, dtype=torch.float32))
        torch.testing.assert_close(gate_input_scale, torch.tensor(201 + idx, dtype=torch.float32))
        torch.testing.assert_close(up_input_scale, torch.tensor(201 + idx, dtype=torch.float32))
        torch.testing.assert_close(down_input_scale, torch.tensor(301 + idx, dtype=torch.float32))

    assert "experts.gate_up_proj_scale_inv" not in state_dict
    assert "experts.down_proj_scale_inv" not in state_dict
    assert "experts.gate_up_proj_activation_scale" not in state_dict
    assert "experts.down_proj_activation_scale" not in state_dict


def test_moe_fused_checkpoint_hook_uses_owned_expert_ids():
    config = _small_text_config()
    moe = Mistral4MoE(config)

    original_state_dict = moe.state_dict

    def local_only_state_dict(*args, **kwargs):
        full = original_state_dict(*args, **kwargs)
        keep = {}
        for key, value in full.items():
            if not key.startswith("experts."):
                keep[key] = value
                continue
            parts = key.split(".", 3)
            if len(parts) >= 4 and parts[1] in {"1", "3"}:
                keep[key] = value
        return keep

    moe.state_dict = local_only_state_dict

    state_dict = {}
    state_dict.update(_fused_expert_checkpoint_from_moe(moe))
    state_dict.update(_fused_expert_scale_checkpoint_from_moe(moe))

    moe._load_experts_from_fused_checkpoint(state_dict, "")

    assert "experts.1.gate_proj.weight" in state_dict
    assert "experts.3.up_proj.weight" in state_dict
    assert "experts.1.down_proj.weight_scale" in state_dict
    assert "experts.3.down_proj.input_scale" in state_dict
    assert "experts.0.gate_proj.weight" not in state_dict
    assert "experts.2.up_proj.weight" not in state_dict
    assert "experts.4.down_proj.weight_scale" not in state_dict


def test_attention_kv_b_proj_load_hook_dequantizes_absorbed_fp8_weight():
    device = _device()
    config = _small_text_config()
    attention = Mistral4Attention(config, layer_idx=0).to(device)

    dequantized = torch.tensor(
        [[-0.5, 0.25], [0.75, -0.125]],
        dtype=torch.float32,
        device=device,
    )
    scale = torch.tensor(0.25, dtype=torch.float32, device=device)
    quantized = (dequantized / scale).to(torch.float8_e4m3fn)
    state_dict = {
        "kv_b_proj.weight": quantized,
        "kv_b_proj.weight_scale_inv": scale,
        "kv_b_proj.activation_scale": torch.tensor(3.0, dtype=torch.float32, device=device),
    }

    attention._load_absorbed_kv_b_proj_from_fp8_checkpoint(state_dict, "")

    assert "kv_b_proj.weight_scale_inv" not in state_dict
    assert "kv_b_proj.activation_scale" not in state_dict
    assert state_dict["kv_b_proj.weight"].dtype == attention.kv_b_proj.weight.dtype
    torch.testing.assert_close(
        state_dict["kv_b_proj.weight"].float(),
        dequantized,
        rtol=0,
        atol=1e-3,
    )


def test_decoder_layer_equivalence():
    device = _device()
    dtype = torch.bfloat16
    config = _small_text_config()
    ad_module = Mistral4DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype)
    ref_module = HfMistral4DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype)
    _load_reference_block(ref_module, ad_module)
    hidden_states = torch.randn(2, 5, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(5, device=device).unsqueeze(0).expand(2, -1)
    actual = ad_module(hidden_states, position_ids)
    expected = ref_module(hidden_states, position_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.05, msg="Decoder layer: ")


def test_full_model_equivalence_cpu():
    device = "cpu"
    dtype = torch.float32
    config = _small_text_config()
    ad_model = Mistral4ForCausalLM(config).to(device=device, dtype=dtype)
    ref_model = HfMistral4ForCausalLM(config).to(device=device, dtype=dtype)
    ref_model.load_state_dict(ad_model.state_dict(), strict=True)
    input_ids = torch.randint(0, config.vocab_size, (2, 5), device=device)
    position_ids = torch.arange(5, device=device).unsqueeze(0).expand(2, -1)
    actual = ad_model(input_ids=input_ids, position_ids=position_ids).logits
    expected = ref_model(input_ids, position_ids)
    assert_rmse_close(actual, expected, rmse_ratio_tol=0.05, msg="Full model: ")


def test_top_level_wrapper_inputs_embeds_path():
    device = _device()
    dtype = torch.bfloat16
    config = _small_outer_config()
    model = Mistral3ForConditionalGenerationAD(config).to(device=device, dtype=dtype)
    inputs_embeds = torch.randn(2, 5, config.text_config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(5, device=device).unsqueeze(0).expand(2, -1)
    wrapper_logits = model(inputs_embeds=inputs_embeds, position_ids=position_ids).logits
    ref_logits = model.language_model(inputs_embeds=inputs_embeds, position_ids=position_ids).logits
    torch.testing.assert_close(wrapper_logits, ref_logits, atol=1e-3, rtol=1e-3)


def test_mistral3_wrappers_do_not_forward_none_inputs_embeds():
    class RecordingModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, **kwargs):
            self.kwargs = kwargs
            hidden_states = torch.zeros(1, 2, 4)
            return type("Out", (), {"last_hidden_state": hidden_states, "logits": hidden_states})()

    text_cfg = _small_text_config()
    causal_lm = Mistral4ForCausalLM(text_cfg)
    causal_lm.model = RecordingModule()
    causal_lm.lm_head = nn.Identity()
    _ = causal_lm(
        input_ids=torch.ones(1, 2, dtype=torch.long), position_ids=torch.arange(2).view(1, 2)
    )
    assert "inputs_embeds" not in causal_lm.model.kwargs

    outer = Mistral3ForConditionalGenerationAD(_small_outer_config())
    outer.language_model = RecordingModule()
    _ = outer(input_ids=torch.ones(1, 2, dtype=torch.long), position_ids=torch.arange(2).view(1, 2))
    assert "inputs_embeds" not in outer.language_model.kwargs


def test_export_mistral4_text_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for export coverage.")

    device = "cuda"
    dtype = torch.bfloat16
    config = _small_text_config()
    model = Mistral4ForCausalLM(config).to(device=device, dtype=dtype)
    input_ids = torch.randint(0, config.vocab_size, (2, 5), device=device)
    position_ids = torch.arange(5, device=device).unsqueeze(0).expand(2, -1)

    gm = torch_export_to_gm(
        model,
        args=(input_ids, position_ids),
        dynamic_shapes=(
            {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        ),
        clone=True,
    )

    out = gm(input_ids, position_ids).logits
    assert torch.isfinite(out).all()

    input_ids_2 = torch.randint(0, config.vocab_size, (1, 7), device=device)
    position_ids_2 = torch.arange(7, device=device).unsqueeze(0)
    out_2 = gm(input_ids_2, position_ids_2).logits
    assert out_2.shape[:2] == (1, 7)
    assert torch.isfinite(out_2).all()


def test_exported_mistral4_graph_has_valid_layer_subgraphs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for export coverage.")

    device = "cuda"
    dtype = torch.bfloat16
    config = _small_text_config()
    config.num_hidden_layers = 1
    model = Mistral4ForCausalLM(config).to(device=device, dtype=dtype)
    input_ids = torch.randint(0, config.vocab_size, (1, 4), device=device)
    position_ids = torch.arange(4, device=device).unsqueeze(0)

    gm = torch_export_to_gm(model, args=(input_ids, position_ids), clone=True)
    layer_subgraphs, _ = get_all_layer_subgraphs(gm)

    layer_types = [layer.layer_type.name for layer in layer_subgraphs]
    assert "MLA" in layer_types
    assert "MLP" in layer_types


def test_registration():
    assert (
        AutoModelForCausalLMFactory._custom_model_mapping["Mistral4TextConfig"]
        == Mistral4ForCausalLM
    )
    assert (
        AutoModelForCausalLMFactory._custom_model_mapping["Mistral3Config"]
        == Mistral3ForConditionalGenerationAD
    )
