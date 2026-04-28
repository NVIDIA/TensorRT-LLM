# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Slimmed down PyTorch DeepSeek V4 model implementation for auto_deploy export.

Source:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py

This implementation differs from the original in the following ways:
* Simplified for prefill-only inference (no KV-cache mutation or decode state)
* Uses auto_deploy custom ops for export compatibility (torch_attention, torch_moe,
  torch_rmsnorm). Interleaved RoPE is inlined in plain PyTorch on real cos/sin tables
  because the complex-freqs canonical op produces ``aten.imag`` nodes during export
  that break follow-on shape propagation in AD transforms.
* Sparse attention is expressed as ``torch_attention`` plus an explicit top-k mask
* Compressed-KV token path is implemented for prefill only
* No MTP blocks
* Weight FP4/FP8 quantization is handled by AD transforms at load time; activation
  fake-quantization in the attention KV/indexer paths is preserved because it is
  part of the reference inference math.

Key architectural features preserved:
* Hyper-Connections (HC): hc_mult copies of hidden state between blocks with learnable
  Sinkhorn-based mixing (hc_pre / hc_post).
* MLA-style attention with single KV head (MQA), low-rank Q (wq_a -> q_norm -> wq_b),
  grouped low-rank O projection (wo_a grouped einsum -> wo_b), learnable attn sinks,
  parameterless per-head RMS on Q, interleaved complex RoPE on the last rope_head_dim.
* Learned KV compression and the ratio-4 Indexer used by DeepSeek sparse attention.
* MoE with per-layer routing: first num_hash_layers use token_id -> expert_id lookup;
  remaining layers use sqrtsoftplus scoring with bias-shifted top-k selection.
* YaRN RoPE scaling; two rope tables when compress_ratios[layer] != 0 vs 0.
"""

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (  # noqa: F401 -- register op
    deepseek_v4_sparse_attention,
)
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType


class DeepseekV4Config(PretrainedConfig):
    """Bundled configuration for DeepSeek V4.

    V4 is not in transformers 4.57.3; the HF repo ships its own config via
    trust_remote_code. We also register a local class here so the model loads
    even when remote code is not trusted.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1024,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        sliding_window: int = 128,
        hidden_act: str = "silu",
        # MoE
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        num_hash_layers: int = 3,
        moe_intermediate_size: int = 2048,
        scoring_func: str = "sqrtsoftplus",
        routed_scaling_factor: float = 1.5,
        norm_topk_prob: bool = True,
        swiglu_limit: float = 10.0,
        # RoPE
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        compress_rope_theta: float = 160000.0,
        rope_scaling: Optional[dict] = None,
        # Compression (used only to decide which rope table a layer uses here)
        compress_ratios: Optional[List[int]] = None,
        # HC
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        # Index (unused in prefill-only forward; kept to accept HF config fields)
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        # Other
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.sliding_window = sliding_window
        self.hidden_act = hidden_act
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hash_layers = num_hash_layers
        self.moe_intermediate_size = moe_intermediate_size
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.swiglu_limit = swiglu_limit
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.rope_scaling = rope_scaling
        self.compress_ratios = (
            list(compress_ratios) if compress_ratios is not None else [0] * num_hidden_layers
        )
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


try:
    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
except (TypeError, ValueError):
    pass


_CHECKPOINT_KEY_REPLACEMENTS = (
    (r"^embed\.weight$", "model.embed_tokens.weight"),
    (r"^head\.weight$", "lm_head.weight"),
    (r"^norm\.weight$", "model.norm.weight"),
    (r"^hc_head_", "model.hc_head_"),
    (r"^layers\.", "model.layers."),
    (r"\.ffn\.experts\.(\d+)\.w1\.", r".ffn.experts.\1.gate_proj."),
    (r"\.ffn\.experts\.(\d+)\.w3\.", r".ffn.experts.\1.up_proj."),
    (r"\.ffn\.experts\.(\d+)\.w2\.", r".ffn.experts.\1.down_proj."),
    (r"\.ffn\.shared_experts\.w1\.", ".ffn.shared_experts.gate_proj."),
    (r"\.ffn\.shared_experts\.w3\.", ".ffn.shared_experts.up_proj."),
    (r"\.ffn\.shared_experts\.w2\.", ".ffn.shared_experts.down_proj."),
    (r"\.scale$", ".weight_scale_inv"),
)


def _rename_deepseek_v4_checkpoint_key(key: str) -> str:
    for pattern, replacement in _CHECKPOINT_KEY_REPLACEMENTS:
        key = re.sub(pattern, replacement, key)
    return key


def _unstack_deepseek_v4_expert_tensors(state_dict: dict[str, torch.Tensor]) -> None:
    """Convert stacked MoE checkpoint tensors to the per-expert ModuleList layout."""
    direct_names = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    for key in list(state_dict.keys()):
        direct_match = re.match(
            r"^(?P<prefix>(?:model\.)?layers\.\d+\.ffn\.)experts\.(?P<which>w[123])\.weight$",
            key,
        )
        if direct_match is not None:
            stacked = state_dict.pop(key)
            target_name = direct_names[direct_match.group("which")]
            for expert_idx in range(stacked.shape[0]):
                state_dict[
                    f"{direct_match.group('prefix')}experts.{expert_idx}.{target_name}.weight"
                ] = stacked[expert_idx]
            continue

        gate_up_match = re.match(
            r"^(?P<prefix>(?:model\.)?layers\.\d+\.ffn\.)experts\.gate_up_proj(?:\.weight)?$",
            key,
        )
        if gate_up_match is not None:
            stacked = state_dict.pop(key)
            gate, up = stacked.chunk(2, dim=1)
            for expert_idx in range(stacked.shape[0]):
                prefix = gate_up_match.group("prefix")
                state_dict[f"{prefix}experts.{expert_idx}.gate_proj.weight"] = gate[expert_idx]
                state_dict[f"{prefix}experts.{expert_idx}.up_proj.weight"] = up[expert_idx]
            continue

        down_match = re.match(
            r"^(?P<prefix>(?:model\.)?layers\.\d+\.ffn\.)experts\.down_proj(?:\.weight)?$",
            key,
        )
        if down_match is not None:
            stacked = state_dict.pop(key)
            for expert_idx in range(stacked.shape[0]):
                state_dict[f"{down_match.group('prefix')}experts.{expert_idx}.down_proj.weight"] = (
                    stacked[expert_idx]
                )


def _dequantize_deepseek_v4_wo_a(state_dict: dict[str, torch.Tensor]) -> None:
    """Dequantize grouped ``wo_a`` FP8 weights the same way HF ``convert.py`` does."""
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return

    for weight_key in list(state_dict.keys()):
        if not re.match(r"^(?:model\.)?layers\.\d+\.attn\.wo_a\.weight$", weight_key):
            continue
        scale_key = weight_key.removesuffix(".weight") + ".weight_scale_inv"
        if scale_key not in state_dict:
            continue
        weight = state_dict[weight_key]
        scale = state_dict[scale_key]
        if weight.dtype != fp8_dtype:
            continue

        if weight.shape[0] % 128 != 0 or weight.shape[1] % 128 != 0:
            continue

        weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
        weight = weight.float() * scale[:, None, :, None].float()
        state_dict[weight_key] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
        del state_dict[scale_key]


def _remap_deepseek_v4_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> None:
    _unstack_deepseek_v4_expert_tensors(state_dict)
    for key in list(state_dict.keys()):
        new_key = _rename_deepseek_v4_checkpoint_key(key)
        if new_key != key:
            state_dict[new_key] = state_dict.pop(key)
    _unstack_deepseek_v4_expert_tensors(state_dict)
    _dequantize_deepseek_v4_wo_a(state_dict)


class DeepseekV4RMSNorm(nn.Module):
    """Standard weighted RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        ).to(hidden_states.dtype)


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int
) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def _build_freqs_cis(
    rope_head_dim: int,
    max_seq_len: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real-valued rotary (cos, sin) tables with optional YaRN scaling.

    Returns ``(cos, sin)``, each of shape ``[max_seq_len, rope_head_dim // 2]``,
    encoding ``cos(pos * inv_freq[k])`` / ``sin(...)``. We avoid complex tensors in
    the exported graph because ``torch.export`` decomposes complex indexing into
    ``aten.imag`` / ``aten.real`` nodes that break follow-on shape propagation.
    """
    freqs = 1.0 / (base ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim))
    if original_seq_len > 0:
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rope_head_dim, base, original_seq_len
        )
        smooth = 1 - _yarn_linear_ramp_mask(low, high, rope_head_dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def _apply_interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Interleaved (complex-style) RoPE on the last dim of ``x``, in pure real ops.

    ``x``: ``[..., rope_head_dim]`` with channel pairs ``[a0, b0, a1, b1, ...]``.
    ``cos``/``sin``: ``[B, S, rope_head_dim // 2]``; expanded to the broadcast layout
    expected against ``x[..., ::2]`` / ``x[..., 1::2]`` by the caller.
    ``inverse=True`` negates sin, i.e. multiplies by the complex conjugate.
    """
    if inverse:
        sin = -sin
    a = x[..., 0::2]
    b = x[..., 1::2]
    out_a = a * cos - b * sin
    out_b = a * sin + b * cos
    return torch.stack([out_a, out_b], dim=-1).flatten(-2).type_as(x)


def _ceil_pow2_scale(amax: torch.Tensor, max_value: float, min_value: float) -> torch.Tensor:
    amax = amax.clamp_min(min_value)
    return torch.pow(2.0, torch.ceil(torch.log2(amax / max_value)))


def _fake_fp8_act_quant(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Reference in-place ``act_quant`` approximation for DeepSeek V4 KV tensors."""
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = _ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 448.0, 1.0e-4)
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _fake_fp4_act_quant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Reference in-place ``fp4_act_quant`` approximation for DeepSeek V4 indexer tensors."""
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(*x_float.shape[:-1], dim // block_size, block_size)
    scale = _ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 6.0, 6.0 * 2.0**-126)
    normalized = torch.clamp(grouped / scale, -6.0, 6.0)
    levels = normalized.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    level_idx = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1)
    quant = levels[level_idx] * normalized.sign()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Hadamard rotation used before FP4 simulation in the DeepSeek V4 indexer."""
    dim = x.shape[-1]
    if dim <= 1:
        return x
    if dim & (dim - 1):
        raise ValueError(f"Hadamard rotation requires power-of-two dimension, got {dim}.")

    orig_shape = x.shape
    y = x.float()
    width = 1
    while width < dim:
        y = y.reshape(*y.shape[:-1], dim // (2 * width), 2, width)
        left = y[..., 0, :]
        right = y[..., 1, :]
        y = torch.cat([left + right, left - right], dim=-1).reshape(orig_shape)
        width *= 2
    return (y * (dim**-0.5)).to(x.dtype)


class DeepseekV4RotaryEmbedding(nn.Module):
    """Shared rotary embedding: precomputes two (cos, sin) table pairs (real floats).

    * base: theta=rope_theta, no YaRN — used by layers with compress_ratio == 0
    * compress: theta=compress_rope_theta with YaRN — used by layers with compress_ratio != 0

    ``forward(x)`` returns all four cached tables. The model forward slices them
    by position_ids once, and each attention layer picks the appropriate pair
    based on its compress_ratio. The ``_ad_`` prefix keeps the buffers on meta
    during AD's lift_to_meta pass.
    """

    def __init__(
        self,
        rope_head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        compress_rope_theta: float,
        rope_scaling: Optional[dict],
    ):
        super().__init__()
        self.rope_head_dim = rope_head_dim
        self.max_position_embeddings = max_position_embeddings

        factor = 1.0
        original_seq_len = 0
        beta_fast = 32
        beta_slow = 1
        if rope_scaling is not None and isinstance(rope_scaling, dict):
            factor = float(rope_scaling.get("factor", 1.0))
            original_seq_len = int(rope_scaling.get("original_max_position_embeddings", 0))
            beta_fast = int(rope_scaling.get("beta_fast", 32))
            beta_slow = int(rope_scaling.get("beta_slow", 1))

        cos_base, sin_base = _build_freqs_cis(
            rope_head_dim,
            max_position_embeddings,
            rope_theta,
            0,
            1.0,
            beta_fast,
            beta_slow,
        )
        cos_compress, sin_compress = _build_freqs_cis(
            rope_head_dim,
            max_position_embeddings,
            compress_rope_theta,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
        )
        self.register_buffer("_ad_cos_base", cos_base, persistent=False)
        self.register_buffer("_ad_sin_base", sin_base, persistent=False)
        self.register_buffer("_ad_cos_compress", cos_compress, persistent=False)
        self.register_buffer("_ad_sin_compress", sin_compress, persistent=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del x
        return (
            self._ad_cos_base,
            self._ad_sin_base,
            self._ad_cos_compress,
            self._ad_sin_compress,
        )


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference-faithful Sinkhorn splitting of HC mix logits.

    mixes: [..., (2 + hc) * hc] — concatenation of [pre_hc, post_hc, comb_hc*hc] logits.
    Returns ``pre: [..., hc]``, ``post: [..., hc]``, ``comb: [..., hc, hc]``.
    """
    hc = hc_mult
    pre_logits = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
    post_logits = mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    comb_logits = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]

    pre = torch.sigmoid(pre_logits) + eps
    post = 2.0 * torch.sigmoid(post_logits)

    comb = comb_logits.view(*comb_logits.shape[:-1], hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class DeepseekV4MLP(nn.Module):
    """SwiGLU expert / shared expert with optional clamp (matches V4 Expert).

    w1 = gate_proj, w3 = up_proj, w2 = down_proj. swiglu_limit > 0 clamps the
    (gate, up) activations before silu*up — matches the reference Expert forward.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class DeepseekV4MoEGate(nn.Module):
    """Routing with two modes (layer-dependent):

    * Hash routing (layer_idx < num_hash_layers): selected_experts come from
      ``tid2eid[input_ids]`` and routing weights come from ``sqrtsoftplus(x @ W)``
      gathered at those indices. No bias, no topk.
    * Score routing (otherwise): selected_experts come from topk of
      ``sqrtsoftplus(x @ W) + bias``; routing weights are gathered from the
      unbiased sqrtsoftplus scores, normalized (when norm_topk_prob), scaled by
      ``routed_scaling_factor``.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.hash_routing = layer_idx < config.num_hash_layers
        self.vocab_size = config.vocab_size

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )
        if self.hash_routing:
            self.register_buffer(
                "tid2eid",
                torch.zeros(self.vocab_size, self.top_k, dtype=torch.int32),
                persistent=True,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.zeros(self.n_routed_experts, dtype=torch.float32))

    def forward(
        self, hidden_states_flat: torch.Tensor, input_ids_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Keep the GEMM input in model dtype so FP8 fused linears can quantize it,
        # then promote scores for routing math.
        router_logits = F.linear(hidden_states_flat, self.weight).float()
        # sqrtsoftplus: sqrt(softplus(x))
        scores = F.softplus(router_logits).sqrt()
        unbiased_scores = scores

        if self.hash_routing:
            selected_experts = self.tid2eid[input_ids_flat].to(torch.int64)
            # Clamp defensively — the checkpoint should always emit expert ids in
            # [0, n_routed_experts), but out-of-range sentinels (e.g. -1 in a
            # stale slot) would fail ``gather`` with a CUDA assert.
            selected_experts = selected_experts.clamp(0, self.n_routed_experts - 1)
        else:
            biased_scores = scores + self.bias
            selected_experts = biased_scores.topk(self.top_k, dim=-1)[1]

        routing_weights = unbiased_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return selected_experts, routing_weights


class DeepseekV4MoE(nn.Module):
    """Routed MoE via torch_moe custom op + 1 shared expert."""

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.experts = nn.ModuleList(
            [
                DeepseekV4MLP(
                    config.hidden_size,
                    config.moe_intermediate_size,
                    swiglu_limit=config.swiglu_limit,
                )
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV4MoEGate(config, layer_idx)
        # Shared expert: no swiglu_limit (matches reference)
        shared_inter = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepseekV4MLP(config.hidden_size, shared_inter, swiglu_limit=0.0)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.view(-1, orig_shape[-1])
        selected_experts, routing_weights = self.gate(flat, input_ids.reshape(-1))
        routed = torch.ops.auto_deploy.torch_moe(
            flat,
            selected_experts,
            routing_weights,
            w1_weight=[e.gate_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[e.up_proj.weight for e in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
            swiglu_limit=self.swiglu_limit,
        )
        return routed.view(*orig_shape) + self.shared_experts(hidden_states)


def _overlap_transform(tensor: torch.Tensor, head_dim: int, value: float = 0.0) -> torch.Tensor:
    """HF DeepSeek V4 overlap transform used by ratio-4 compression."""
    bsz, _, ratio, _ = tensor.shape
    previous = tensor[:, :, :, :head_dim]
    current = tensor[:, :, :, head_dim:]
    prefix = tensor.new_full((bsz, 1, ratio, head_dim), value)
    previous = torch.cat([prefix, previous[:, :-1]], dim=1)
    return torch.cat([previous, current], dim=2)


def _window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Prefill ``start_pos == 0`` version of HF ``get_window_topk_idxs``.

    The window axis is kept at static ``window_size`` for export. Entries outside
    the real causal window are marked ``-1`` and ignored by the sparse mask.
    """
    base = torch.arange(seqlen, device=device).unsqueeze(1)
    matrix = base - window_size + 1 + torch.arange(window_size, device=device)
    matrix = torch.where((matrix < 0) | (matrix > base), -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _compress_topk_idxs(
    ratio: int,
    bsz: int,
    seqlen: int,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Prefill ``start_pos == 0`` version of HF ``get_compress_topk_idxs``.

    The compressed-candidate axis is padded to ``seqlen`` to avoid specializing
    dynamic export shapes on ``seqlen // ratio``.
    """
    matrix = torch.arange(seqlen, device=device).repeat(seqlen, 1)
    valid_count = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
    mask = matrix >= valid_count
    matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _build_sparse_attn_mask(topk_idxs: torch.Tensor, total_kv_len: int) -> torch.Tensor:
    """Convert HF top-k index lists into an additive attention mask.

    ``topk_idxs`` is ``[B, S, K]`` and may contain ``-1`` for invalid slots.
    The returned mask is ``[B, 1, S, total_kv_len]`` with 0 for selected
    positions and ``-inf`` elsewhere.
    """
    valid = topk_idxs >= 0
    kv_positions = torch.arange(total_kv_len, device=topk_idxs.device)
    selected = topk_idxs.unsqueeze(-1) == kv_positions.view(1, 1, 1, -1)
    selected = (selected & valid.unsqueeze(-1)).any(dim=2)
    mask = topk_idxs.new_zeros(selected.shape, dtype=torch.float32)
    mask = mask.masked_fill(~selected, -10000.0)
    return mask.unsqueeze(1)


def _manual_attention_with_sinks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Reference-style attention for compressed sparse layers.

    Generic KV-cache insertion assumes Q/K/V share the same token axis. DeepSeek V4
    compressed sparse attention appends learned compressed KV tokens, so those
    layers stay as plain PyTorch math instead of ``torch_attention``.
    """
    bsz, seqlen, n_heads, head_dim = q.shape
    kv_len = k.shape[1]
    q_bh = q.transpose(1, 2).float()
    k_bh = k.expand(bsz, kv_len, n_heads, head_dim).transpose(1, 2).float()
    v_bh = v.expand(bsz, kv_len, n_heads, head_dim).transpose(1, 2).float()
    scores = torch.matmul(q_bh, k_bh.transpose(-2, -1)) * scale
    scores = scores + attn_mask.float()

    sink_logits = sinks.float().reshape(1, n_heads, 1, 1).expand(bsz, n_heads, seqlen, 1)
    logits_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink_logits)
    exp_scores = torch.exp(scores - logits_max)
    exp_sinks = torch.exp(sink_logits - logits_max)
    attn = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + exp_sinks)
    out = torch.matmul(attn, v_bh)
    return out.transpose(1, 2).contiguous().to(q.dtype)


def _ratio_chunk_indices(
    seqlen: int, ratio: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return padded ratio chunk token indices and their validity mask."""
    chunk_ids = torch.arange(seqlen, device=device).unsqueeze(1)
    token_offsets = torch.arange(ratio, device=device)
    indices = chunk_ids * ratio + token_offsets
    valid = indices < seqlen
    indices = torch.where(valid, indices, torch.zeros_like(indices))
    return indices, valid


class DeepseekV4Compressor(nn.Module):
    """Prefill-only learned KV compressor from HF DeepSeek V4.

    The decode-time rolling state buffers are intentionally omitted.  This
    module implements the ``start_pos == 0`` path, which is the path needed for
    AutoDeploy export/reference equivalence.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.rotate = rotate
        self.overlap = compress_ratio == 4
        coff = 1 + int(self.overlap)

        self.ape = nn.Parameter(
            torch.zeros(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        self.wkv = nn.Linear(self.hidden_size, coff * self.head_dim, bias=False)
        self.wgate = nn.Linear(self.hidden_size, coff * self.head_dim, bias=False)
        self.norm = DeepseekV4RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        ratio = self.compress_ratio
        chunk_indices, chunk_valid = _ratio_chunk_indices(seqlen, ratio, hidden_states.device)
        flat_indices = chunk_indices.reshape(-1)

        kv_all = self.wkv(hidden_states).float()
        score_all = self.wgate(hidden_states).float()
        kv = kv_all[:, flat_indices].view(bsz, seqlen, ratio, -1)
        score = score_all[:, flat_indices].view(bsz, seqlen, ratio, -1) + self.ape
        score = torch.where(
            chunk_valid.view(1, seqlen, ratio, 1),
            score,
            score.new_full((), -1.0e20),
        )
        if self.overlap:
            kv = _overlap_transform(kv, self.head_dim, 0.0)
            score = _overlap_transform(score, self.head_dim, -1.0e20)

        compressed = (kv * score.softmax(dim=2)).sum(dim=2)
        compressed = self.norm(compressed.to(hidden_states.dtype))

        rd = self.rope_head_dim
        chunk_start = torch.arange(seqlen, device=hidden_states.device) * ratio
        chunk_start = torch.where(chunk_start < seqlen, chunk_start, torch.zeros_like(chunk_start))
        cos_comp = cos[:, chunk_start]
        sin_comp = sin[:, chunk_start]
        nope, pe = torch.split(compressed, [self.head_dim - rd, rd], dim=-1)
        pe = _apply_interleaved_rope(pe, cos_comp, sin_comp)
        compressed = torch.cat([nope, pe], dim=-1)
        if self.rotate:
            return _fake_fp4_act_quant(_hadamard_rotate(compressed), block_size=32)

        nope, pe = torch.split(compressed, [self.head_dim - rd, rd], dim=-1)
        nope = _fake_fp8_act_quant(nope, block_size=64)
        return torch.cat([nope, pe], dim=-1)


class DeepseekV4Indexer(nn.Module):
    """Ratio-4 compressed-token indexer from HF DeepSeek V4, prefill path only."""

    def __init__(self, config: DeepseekV4Config, compress_ratio: int):
        super().__init__()
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.compress_ratio = compress_ratio

        self.wq_b = nn.Linear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.compressor = DeepseekV4Compressor(
            config, compress_ratio, self.index_head_dim, rotate=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        ratio = self.compress_ratio

        rd = self.rope_head_dim
        q = self.wq_b(q_lora).view(bsz, seqlen, self.index_n_heads, self.index_head_dim)
        q_nope, q_pe = torch.split(q, [self.index_head_dim - rd, rd], dim=-1)
        q_pe = _apply_interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
        q = torch.cat([q_nope, q_pe], dim=-1)
        q = _fake_fp4_act_quant(_hadamard_rotate(q), block_size=32)

        index_k = self.compressor(hidden_states, cos, sin)
        weights = self.weights_proj(hidden_states).float() * (
            self.softmax_scale * self.index_n_heads**-0.5
        )

        index_score = torch.einsum("bshd,btd->bsht", q, index_k).float()
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)

        num_compressed = seqlen // ratio
        if self.index_topk == 0:
            return torch.empty(bsz, seqlen, 0, device=hidden_states.device, dtype=torch.int64)
        if num_compressed == 0:
            return torch.full(
                (bsz, seqlen, self.index_topk),
                -1,
                device=hidden_states.device,
                dtype=torch.int64,
            )

        index_score = index_score[..., :num_compressed]
        compressed_positions = torch.arange(num_compressed, device=hidden_states.device)
        valid_count = torch.arange(1, seqlen + 1, device=hidden_states.device).unsqueeze(1) // ratio
        future_mask = compressed_positions.unsqueeze(0) >= valid_count
        index_score = index_score.masked_fill(future_mask.unsqueeze(0), -1.0e20)

        k = min(self.index_topk, num_compressed)
        topk_idxs = index_score.topk(k, dim=-1).indices
        invalid = topk_idxs >= valid_count.unsqueeze(0)
        topk_idxs = torch.where(invalid, -1, topk_idxs + offset)
        if k < self.index_topk:
            pad = torch.full(
                (bsz, seqlen, self.index_topk - k),
                -1,
                device=hidden_states.device,
                dtype=topk_idxs.dtype,
            )
            topk_idxs = torch.cat([topk_idxs, pad], dim=-1)
        return topk_idxs.to(torch.int64)


class DeepseekV4Attention(nn.Module):
    """MLA-variant with single KV head, sliding/compressed sparse attention.

    * Q: wq_a -> q_norm -> wq_b -> reshape [B,S,H,D] -> parameterless per-head rsqrt -> RoPE(last rd)
    * KV: wkv -> kv_norm -> RoPE(last rd) ; a single [B,S,D] tensor used as both K and V
    * Attention:
      - compress_ratio == 0: ``torch_attention`` with sliding window + sinks
      - compress_ratio != 0: compressed KV tokens are appended and selected with
        explicit sparse top-k masks before ``torch_attention``
    * Post: inverse RoPE on output's last rd dims (undoes K-side rotation baked into V)
    * O: grouped low-rank projection via einsum (wo_a) then wo_b

    Decode-time compression state is dropped in this prefill-only port.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.n_groups = config.o_groups
        self.window_size = config.sliding_window
        self.compress_ratio = int(config.compress_ratios[layer_idx])
        self.rms_eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5

        # Group must divide n_heads evenly.
        assert self.n_heads % self.n_groups == 0
        self.group_head_width = (self.n_heads * self.head_dim) // self.n_groups

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, eps=self.rms_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=self.rms_eps)
        # Grouped low-rank O: wo_a stored as [n_groups * o_lora_rank, group_head_width]
        # so the einsum ``bsgd,grd->bsgr`` is a plain linear whose weight is reshaped.
        self.wo_a = nn.Linear(self.group_head_width, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        if self.compress_ratio != 0:
            self.compressor = DeepseekV4Compressor(config, self.compress_ratio, self.head_dim)
            self.indexer = (
                DeepseekV4Indexer(config, self.compress_ratio) if self.compress_ratio == 4 else None
            )
        else:
            self.compressor = None
            self.indexer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_base: torch.Tensor,
        sin_base: torch.Tensor,
        cos_compress: torch.Tensor,
        sin_compress: torch.Tensor,
        cos_compress_table: Optional[torch.Tensor] = None,
        sin_compress_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape

        # Q: low-rank + per-head parameterless RMS + interleaved RoPE on last rd dims.
        qr = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(qr).view(bsz, seqlen, self.n_heads, self.head_dim)
        # Parameterless per-head RMS (reference: q *= rsqrt(q.square().mean(-1, keepdim=True) + eps))
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_eps).to(q.dtype)

        # KV: single head, shared for both K and V; RoPE on last rd dims.
        kv = self.kv_norm(self.wkv(hidden_states)).view(bsz, seqlen, 1, self.head_dim)

        # RoPE: interleaved (complex-style) in pure real ops. Broadcast cos/sin with the
        # [B, S, H, rope_head_dim] layout by adding a singleton head dim: [B, S, 1, rd/2].
        if self.compress_ratio != 0:
            cos = cos_compress.unsqueeze(2)
            sin = sin_compress.unsqueeze(2)
        else:
            cos = cos_base.unsqueeze(2)
            sin = sin_base.unsqueeze(2)
        q_nope, q_pe = torch.split(q, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_nope, kv_pe = torch.split(kv, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_interleaved_rope(q_pe, cos, sin)
        kv_pe = _apply_interleaved_rope(kv_pe, cos, sin)
        kv_nope = _fake_fp8_act_quant(kv_nope, block_size=64)
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = torch.cat([kv_nope, kv_pe], dim=-1)

        if self.compress_ratio != 0:
            # Sparse compressed attention: emit a dedicated custom op so the KV-cache
            # transform can recognise and replace it with a cached variant. The op
            # owns the Compressor/Indexer math internally; we forward their weights.
            assert cos_compress_table is not None
            assert sin_compress_table is not None
            cos_flat = cos.squeeze(2)
            sin_flat = sin.squeeze(2)
            if self.indexer is not None:
                indexer_wq_b = self.indexer.wq_b.weight
                indexer_weights_proj = self.indexer.weights_proj.weight
                indexer_compressor_wkv = self.indexer.compressor.wkv.weight
                indexer_compressor_wgate = self.indexer.compressor.wgate.weight
                indexer_compressor_ape = self.indexer.compressor.ape
                indexer_compressor_norm_weight = self.indexer.compressor.norm.weight
            else:
                indexer_wq_b = None
                indexer_weights_proj = None
                indexer_compressor_wkv = None
                indexer_compressor_wgate = None
                indexer_compressor_ape = None
                indexer_compressor_norm_weight = None
            attn_out = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attn(
                q,
                kv,
                hidden_states,
                qr,
                cos_flat,
                sin_flat,
                cos_compress_table,
                sin_compress_table,
                self.attn_sink,
                self.compressor.wkv.weight,
                self.compressor.wgate.weight,
                self.compressor.ape,
                self.compressor.norm.weight,
                indexer_wq_b,
                indexer_weights_proj,
                indexer_compressor_wkv,
                indexer_compressor_wgate,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                self.softmax_scale,
                self.window_size,
                self.compress_ratio,
                self.rope_head_dim,
                self.head_dim,
                self.indexer.index_n_heads if self.indexer is not None else 0,
                self.indexer.index_head_dim if self.indexer is not None else 0,
                self.indexer.index_topk if self.indexer is not None else 0,
                self.rms_eps,
                self.layer_idx,
                "mha_sparse",
            )
        else:
            # Attention: sliding window + learnable sinks, K == V (same tensor).
            attn_out = torch.ops.auto_deploy.torch_attention(
                q,
                kv,
                kv,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.softmax_scale,
                sinks=self.attn_sink,
                sliding_window=self.window_size,
                layout="bsnd",
                layer_idx=self.layer_idx,
                layer_type="mha",
            )

        # Inverse RoPE on the output's last rd dims to undo the V-side rotation.
        attn_nope, attn_pe = torch.split(attn_out, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        attn_pe = _apply_interleaved_rope(attn_pe, cos, sin, inverse=True)
        attn_out = torch.cat([attn_nope, attn_pe], dim=-1)

        # Grouped low-rank O: view as [B, S, n_groups, group_head_width] then einsum.
        attn_out = attn_out.reshape(bsz, seqlen, self.n_groups, self.group_head_width)
        # wo_a weight is [n_groups * o_lora_rank, group_head_width]; view as [n_groups, o_lora_rank, group_head_width]
        wo_a_w = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, self.group_head_width)
        o = torch.einsum("bsgd,grd->bsgr", attn_out, wo_a_w)
        return self.wo_b(o.reshape(bsz, seqlen, self.n_groups * self.o_lora_rank))


class DeepseekV4Block(nn.Module):
    """Transformer block with Hyper-Connections residual wiring.

    Input / output: [B, S, hc_mult, D]. Each sub-layer does:
      residual = x
      y, post, comb = hc_pre(x, hc_*_fn, hc_*_scale, hc_*_base)  # y: [B,S,D]
      y = sublayer_norm(y)
      y = sublayer(y)
      x = hc_post(y, residual, post, comb)  # back to [B,S,hc_mult,D]
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = DeepseekV4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        # Float32 HC mixing params match reference semantics. Weights are overwritten by
        # the checkpoint; zeros here mean untrained runs behave as a neutral residual-ish
        # mixer (sigmoid(0)=0.5, softmax over uniform = 1/hc) instead of producing NaN.
        self.hc_attn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.ones(3, dtype=torch.float32))

    def _hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, S, hc_mult, D] -> flatten trailing two dims, normalize, mix.
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(2)
        x_norm = x_flat.float()
        rsqrt = torch.rsqrt(x_norm.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.matmul(x_norm, hc_fn.float().transpose(0, 1)) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x_norm.view(shape), dim=2)
        return y.to(dtype), post, comb

    def _hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B,S,D], residual: [B,S,hc,D], post: [B,S,hc], comb: [B,S,hc,hc]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        cos_base: torch.Tensor,
        sin_base: torch.Tensor,
        cos_compress: torch.Tensor,
        sin_compress: torch.Tensor,
        cos_compress_table: Optional[torch.Tensor] = None,
        sin_compress_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention sub-layer
        residual = hidden_states
        x, post, comb = self._hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(
            x,
            cos_base,
            sin_base,
            cos_compress,
            sin_compress,
            cos_compress_table,
            sin_compress_table,
        )
        hidden_states = self._hc_post(x, residual, post, comb)

        # FFN (MoE) sub-layer
        residual = hidden_states
        x, post, comb = self._hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        hidden_states = self._hc_post(x, residual, post, comb)
        return hidden_states


@dataclass
class DeepseekV4Output(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class DeepseekV4CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4Block"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DeepseekV4Model(DeepseekV4PreTrainedModel):
    """Embedding + HC expansion + N blocks + HC collapse (head) + final RMSNorm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [DeepseekV4Block(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(
            rope_head_dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            compress_rope_theta=config.compress_rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # HC collapse parameters at the model head (sigmoid * scale + base, no Sinkhorn).
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.zeros(config.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.zeros(config.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _hc_head_collapse(self, x: torch.Tensor) -> torch.Tensor:
        """Collapse [B,S,hc,D] -> [B,S,D] using learned per-copy gates.

        pre[i,j] = sigmoid(mixes[i,j] * hc_scale + hc_base[j]) + hc_eps, then sum over hc.
        """
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(2)
        x_norm = x_flat.float()
        rsqrt = torch.rsqrt(x_norm.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.matmul(x_norm, self.hc_head_fn.float().transpose(0, 1)) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_norm.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeepseekV4Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        assert position_ids is not None, "position_ids must be provided"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Hash routing needs input_ids; if only embeds are given, use zeros (hash routing will
        # select expert 0 for all tokens, producing incorrect but export-friendly behavior).
        if input_ids is None:
            input_ids = torch.zeros(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )

        # Expand to hc_mult copies once at the start of the stack.
        h = inputs_embeds.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

        # Slice RoPE tables once per forward; each layer picks between base and
        # compressed tables by compress_ratio.
        cos_base_table, sin_base_table, cos_compress_table, sin_compress_table = self.rotary_emb(
            inputs_embeds
        )
        cos_base = cos_base_table[position_ids]
        sin_base = sin_base_table[position_ids]
        cos_compress = cos_compress_table[position_ids]
        sin_compress = sin_compress_table[position_ids]

        for layer in self.layers:
            h = layer(
                h,
                input_ids,
                cos_base,
                sin_base,
                cos_compress,
                sin_compress,
                cos_compress_table,
                sin_compress_table,
            )

        # HC head collapse -> final RMSNorm
        h = self._hc_head_collapse(h)
        h = self.norm(h)
        return DeepseekV4Output(last_hidden_state=h)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    """DeepSeek V4 with LM head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV4Config, **kwargs):
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._register_load_state_dict_pre_hook(self._load_hf_checkpoint_keys)
        self.post_init()

    @staticmethod
    def _load_hf_checkpoint_keys(state_dict, prefix, *args):
        del args
        if prefix == "":
            _remap_deepseek_v4_checkpoint_keys(state_dict)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> DeepseekV4CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return DeepseekV4CausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("DeepseekV4Config", DeepseekV4ForCausalLM)
