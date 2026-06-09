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

"""Semantic DeepSeek V4 model for AutoDeploy export.

The implementation keeps the DeepSeek reference hierarchy
``embed`` / ``layers`` / ``norm`` / ``head`` so checkpoint-facing names remain
auditable. It intentionally omits Triton kernels, CUDA graph, and MTP modules.
Sparse attention uses the narrow source-op boundary: this model forms Q/KV, raw
compressor projections, and top-k indices; the source/cache ops own
compressed-row construction plus attention over those rows and the sink term.
"""

import math
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ... import custom_ops  # noqa: F401 -- register custom ops
from ...utils.quantization_utils import fake_fp4_act_quant as _fake_fp4_act_quant
from ...utils.quantization_utils import fake_fp8_act_quant as _fake_fp8_act_quant
from ...utils.quantization_utils import hadamard_rotate as _hadamard_rotate
from ..factory import ModelFactoryRegistry
from ..hf import AutoModelForCausalLMFactory
from ..quant_checkpoint_layout import (
    FineGrainedFP8CheckpointLayout,
    PackedMXFP4ExpertLayout,
    QuantCheckpointLayoutRegistry,
    QuantizedCheckpointLayout,
    QuantizedCheckpointLayoutError,
    load_packed_mxfp4_expert_tensors,
)


@dataclass
class DeepseekV4CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


def _normalize_deepseek_v4_rope_scaling(
    rope_scaling: Optional[Mapping[str, Any]],
    max_position_embeddings: int,
) -> dict[str, Any]:
    """Return a YaRN config accepted by current transformers validators."""

    normalized = (
        dict(rope_scaling)
        if rope_scaling is not None
        else {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": max_position_embeddings,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }
    )
    rope_type = normalized.get("rope_type", normalized.get("type", "yarn"))
    normalized["rope_type"] = rope_type
    normalized["type"] = normalized.get("type", rope_type)
    normalized["factor"] = float(normalized.get("factor", 1.0))
    normalized["beta_fast"] = float(normalized.get("beta_fast", 32.0))
    normalized["beta_slow"] = float(normalized.get("beta_slow", 1.0))
    original_seq_len = int(normalized.get("original_max_position_embeddings", 0) or 0)
    if original_seq_len <= 0:
        original_seq_len = max_position_embeddings
    normalized["original_max_position_embeddings"] = original_seq_len
    return normalized


class DeepseekV4Config(PretrainedConfig):
    """Minimal local config for DeepSeek V4 when transformers lacks one."""

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
        q_lora_rank: int = 1024,
        qk_rope_head_dim: int = 64,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        sliding_window: int = 128,
        compress_ratios: Optional[Tuple[int, ...]] = None,
        compress_rope_theta: float = 160000.0,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 512,
        moe_intermediate_size: int = 2048,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        num_hash_layers: int = 3,
        scoring_func: str = "sqrtsoftplus",
        routed_scaling_factor: float = 1.5,
        norm_topk_prob: bool = True,
        swiglu_limit: float = 10.0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        rms_norm_eps: float = 1e-6,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        ad_rope_cache_len: Optional[int] = None,
        ad_compress_max_seq_len: Optional[int] = None,
        skip_mtp: bool = True,
        use_cache: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs,
    ) -> None:
        ratios = tuple(int(ratio) for ratio in (compress_ratios or (0,) * num_hidden_layers))
        if len(ratios) < num_hidden_layers:
            ratios = ratios + (0,) * (num_hidden_layers - len(ratios))

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.sliding_window = sliding_window
        self.compress_ratios = ratios[:num_hidden_layers]
        self.compress_rope_theta = compress_rope_theta
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hash_layers = num_hash_layers
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.swiglu_limit = swiglu_limit
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = _normalize_deepseek_v4_rope_scaling(
            rope_scaling, max_position_embeddings
        )
        self.rms_norm_eps = rms_norm_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.ad_rope_cache_len = ad_rope_cache_len or min(max_position_embeddings, 4096)
        self.ad_compress_max_seq_len = ad_compress_max_seq_len or self.ad_rope_cache_len
        self.skip_mtp = skip_mtp
        self.use_cache = use_cache
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
    AutoConfig.register(DeepseekV4Config.model_type, DeepseekV4Config, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register(DeepseekV4Config.model_type, DeepseekV4Config)
    except ValueError:
        pass


_DEEPSEEK_V4_FP8_TARGET_PATTERNS = (
    r"layers\.\d+\.(?:"
    r"attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)|"
    r"attn\.indexer\.wq_b|"
    r"ffn\.shared_experts\.w[123]"
    r")",
)
_DEEPSEEK_V4_EXCLUDED_MODULES = (
    "embed",
    "head",
    "*.ffn.gate",
    "*.attn.compressor",
    "*.attn.indexer.compressor",
    "*.attn.indexer.weights_proj",
    "*.norm",
    "*.hc_*",
    "*.attn_sink",
    "mtp.*",
)

_DEEPSEEK_V4_MXFP4_EXPERT_RE = re.compile(
    r"layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>w[123])\.(?P<kind>weight|scale)"
)
_DEEPSEEK_V4_MXFP4_KEY_TEMPLATE = "layers.{layer}.ffn.experts.{expert}.{projection}.{kind}"
_DEEPSEEK_V4_MXFP4_LAYER_NAME_RE = re.compile(r"(?:^|[._])layers[._](?P<layer>\d+)[._]ffn[._]")
_DEEPSEEK_V4_MXFP4_PROJECTIONS = ("w1", "w2", "w3")
_DEEPSEEK_V4_MXFP4_GATE_UP_ORDER = ("w3", "w1")
_DEEPSEEK_V4_MXFP4_DOWN_PROJECTION = "w2"

DeepseekV4PackedMxfp4ExpertsCheckpointLayout = PackedMXFP4ExpertLayout


def build_deepseek_v4_packed_mxfp4_experts_layout() -> PackedMXFP4ExpertLayout:
    return PackedMXFP4ExpertLayout(
        key_pattern=_DEEPSEEK_V4_MXFP4_EXPERT_RE,
        key_template=_DEEPSEEK_V4_MXFP4_KEY_TEMPLATE,
        layer_name_pattern=_DEEPSEEK_V4_MXFP4_LAYER_NAME_RE,
        projections=_DEEPSEEK_V4_MXFP4_PROJECTIONS,
        gate_up_order=_DEEPSEEK_V4_MXFP4_GATE_UP_ORDER,
        down_projection=_DEEPSEEK_V4_MXFP4_DOWN_PROJECTION,
    )


def _deepseek_v4_scale_fmt(qconf: Mapping[str, object]) -> str:
    scale_fmt = qconf.get("scale_fmt")
    if not isinstance(scale_fmt, str) or not scale_fmt:
        raise QuantizedCheckpointLayoutError("DeepSeek V4 quantization_config requires scale_fmt.")
    scale_fmt = scale_fmt.lower()
    if scale_fmt != "ue8m0":
        raise QuantizedCheckpointLayoutError(
            "DeepSeek V4 quantized checkpoint layout supports "
            f"scale_fmt='ue8m0', got '{scale_fmt}'."
        )
    return scale_fmt


def _deepseek_v4_weight_block_size(qconf: Mapping[str, object]) -> tuple[int, int]:
    block_size = qconf.get("weight_block_size")
    if not isinstance(block_size, Sequence) or isinstance(block_size, str):
        raise QuantizedCheckpointLayoutError(
            "DeepSeek V4 quantization_config requires weight_block_size "
            "as a two-element integer sequence."
        )
    try:
        parsed = tuple(_deepseek_v4_positive_int("weight_block_size", dim) for dim in block_size)
    except TypeError as error:
        raise QuantizedCheckpointLayoutError(
            "DeepSeek V4 quantization_config requires weight_block_size "
            f"as a two-element integer sequence, got {block_size}."
        ) from error
    if len(parsed) != 2:
        raise QuantizedCheckpointLayoutError(
            "DeepSeek V4 quantization_config requires weight_block_size "
            f"with two dimensions, got {block_size}."
        )
    return parsed


def _deepseek_v4_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} should contain integers, got {value}.")
    parsed = operator.index(value)
    if parsed <= 0:
        raise QuantizedCheckpointLayoutError(
            f"DeepSeek V4 quantization_config {name} values should be positive, got {value}."
        )
    return parsed


@QuantCheckpointLayoutRegistry.register("deepseek_v4")
def _build_deepseek_v4_checkpoint_layout(
    config: Mapping[str, object],
) -> QuantizedCheckpointLayout | None:
    qconf = config.get("quantization_config")
    if not isinstance(qconf, Mapping):
        return None
    if str(qconf.get("quant_method", "")).lower() != "fp8":
        return None
    scale_fmt = _deepseek_v4_scale_fmt(qconf)
    weight_block_size = _deepseek_v4_weight_block_size(qconf)
    expert_layout = build_deepseek_v4_packed_mxfp4_experts_layout()

    return QuantizedCheckpointLayout(
        finegrained_fp8=FineGrainedFP8CheckpointLayout(
            weight_name_patterns=tuple(
                pattern + r"\.weight$" for pattern in _DEEPSEEK_V4_FP8_TARGET_PATTERNS
            ),
            exclude_patterns=_DEEPSEEK_V4_EXCLUDED_MODULES,
            weight_block_size=weight_block_size,
            scale_suffix="scale",
            runtime_scale_name="weight_scale_inv",
            scale_fmt=scale_fmt,
        ),
        checkpoint_consumers=(expert_layout,),
        extra_quant_config={
            "expert_quant_method": expert_layout.quant_method,
            "expert_block_size": expert_layout.expert_block_size,
        },
    )


_EXPERT_ALIAS_TO_WEIGHT = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


def _rename_deepseek_v4_checkpoint_key(key: str) -> str:
    if key.startswith("model."):
        key = key.removeprefix("model.")
    if key == "embed_tokens.weight":
        return "embed.weight"
    if key == "lm_head.weight":
        return "head.weight"

    def expert_alias(match: re.Match) -> str:
        return f"{match.group(1)}{_EXPERT_ALIAS_TO_WEIGHT[match.group(2)]}"

    key = re.sub(
        r"(\.ffn\.experts\.\d+\.)(gate_proj|up_proj|down_proj)(?=\.)",
        expert_alias,
        key,
    )
    return re.sub(
        r"(\.ffn\.shared_experts\.)(gate_proj|up_proj|down_proj)(?=\.)",
        expert_alias,
        key,
    )


def _remap_deepseek_v4_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> None:
    removals = []
    updates = {}
    for key, tensor in list(state_dict.items()):
        new_key = _rename_deepseek_v4_checkpoint_key(key)
        if new_key.startswith("mtp.0."):
            removals.append(key)
            continue

        if new_key != key:
            removals.append(key)
            updates[new_key] = tensor

    for key in removals:
        state_dict.pop(key, None)
    for key, tensor in updates.items():
        state_dict.setdefault(key, tensor)


def _linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    tp_mode: str = "none",
    layer_type: str = "unknown",
    tp_min_local_shape: int = 1,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_linear_simple(
        x,
        weight,
        bias,
        tp_mode=tp_mode,
        layer_type=layer_type,
        tp_min_local_shape=tp_min_local_shape,
    )


def _linear_module(
    x: torch.Tensor,
    module: nn.Linear,
    tp_mode: str = "none",
    layer_type: str = "unknown",
    tp_min_local_shape: int = 1,
) -> torch.Tensor:
    return _linear(
        x,
        module.weight,
        module.bias,
        tp_mode=tp_mode,
        layer_type=layer_type,
        tp_min_local_shape=tp_min_local_shape,
    )


def _rope_scaling_value(config: PretrainedConfig, key: str, default):
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    return rope_scaling.get(key, default)


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def _build_rope_tables(
    rope_head_dim: int,
    max_seq_len: int,
    base: float,
    original_seq_len: int,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (base ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32) / rope_head_dim))
    if original_seq_len > 0:
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, rope_head_dim, base, original_seq_len
        )
        smooth = 1 - _yarn_linear_ramp_mask(low, high, rope_head_dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, freqs)
    return freqs.cos(), freqs.sin()


def _apply_interleaved_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    if inverse:
        sin = -sin
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


def _window_topk_idxs(
    window_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(seq_len, device=device).unsqueeze(1)
    key_positions = query_positions - window_size + 1 + torch.arange(window_size, device=device)
    key_positions = torch.where(
        (key_positions < 0) | (key_positions > query_positions),
        -1,
        key_positions,
    )
    return key_positions.unsqueeze(0).expand(batch_size, -1, -1)


def _compress_topk_idxs(
    ratio: int,
    batch_size: int,
    seq_len: int,
    offset: int,
    max_compressed_len: int,
    device: torch.device,
) -> torch.Tensor:
    compressed_positions = torch.arange(max_compressed_len, device=device)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // ratio
    topk_idxs = compressed_positions.unsqueeze(0).expand(seq_len, -1)
    topk_idxs = torch.where(topk_idxs < valid_lengths, topk_idxs + offset, -1)
    return topk_idxs.unsqueeze(0).expand(batch_size, -1, -1)


def _overlap_transform(tensor: torch.Tensor, head_dim: int, value: float) -> torch.Tensor:
    batch_size, compressed_len, ratio, _ = tensor.shape
    previous = tensor[:, :, :, :head_dim]
    current = tensor[:, :, :, head_dim:]
    prefix = tensor.new_full((batch_size, 1, ratio, head_dim), value)
    previous = torch.cat((prefix, previous[:, :-1]), dim=1)
    return torch.cat((previous, current), dim=2)


def _sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_compress_table: torch.Tensor,
    sin_compress_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    softmax_scale: float,
    layer_idx: int,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rope_dim: Optional[int],
    rms_norm_eps: float,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        cos_compress_table,
        sin_compress_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        softmax_scale,
        enable_sharding=True,
        layer_idx=layer_idx,
        layer_type="mla",
        window_size=window_size,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
        head_dim=kv.shape[-1],
        rope_dim=rope_dim,
        rms_norm_eps=rms_norm_eps,
    )


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    post_logits = mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    comb_logits = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]

    pre = torch.sigmoid(pre_logits) + eps
    post = 2.0 * torch.sigmoid(post_logits)
    comb = comb_logits.view(*comb_logits.shape[:-1], hc_mult, hc_mult)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.eps)


class DeepseekV4RotaryEmbedding(nn.Module):
    def __init__(self, config: DeepseekV4Config) -> None:
        super().__init__()
        factor = float(_rope_scaling_value(config, "factor", 1.0))
        original_seq_len = int(_rope_scaling_value(config, "original_max_position_embeddings", 0))
        beta_fast = int(_rope_scaling_value(config, "beta_fast", 32))
        beta_slow = int(_rope_scaling_value(config, "beta_slow", 1))

        cos_base, sin_base = _build_rope_tables(
            config.qk_rope_head_dim,
            config.ad_rope_cache_len,
            config.rope_theta,
            0,
            1.0,
            beta_fast,
            beta_slow,
        )
        cos_compress, sin_compress = _build_rope_tables(
            config.qk_rope_head_dim,
            config.ad_rope_cache_len,
            config.compress_rope_theta,
            original_seq_len,
            factor,
            beta_fast,
            beta_slow,
        )
        self.register_buffer("_ad_cos_base", cos_base, persistent=False)
        self.register_buffer("_ad_sin_base", sin_base, persistent=False)
        self.register_buffer("_ad_cos_compress", cos_compress, persistent=False)
        self.register_buffer("_ad_sin_compress", sin_compress, persistent=False)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self._ad_cos_base,
            self._ad_sin_base,
            self._ad_cos_compress,
            self._ad_sin_compress,
        )


class DeepseekV4MLP(nn.Module):
    """SwiGLU body with checkpoint names w1=gate, w3=up, w2=down."""

    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = _linear_module(x, self.w1, tp_mode="colwise", layer_type="moe")
        up = _linear_module(x, self.w3, tp_mode="colwise", layer_type="moe")
        if self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        hidden = (F.silu(gate.float()) * up.float()).to(self.w2.weight.dtype)
        down = _linear_module(hidden, self.w2, tp_mode="rowwise", layer_type="moe")
        down = torch.ops.auto_deploy.all_reduce(down, layer_type="moe")
        return down.to(x.dtype)


class DeepseekV4MoEGate(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.score_func = config.scoring_func
        self.hash_routing = layer_idx < config.num_hash_layers
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.float32)
        )
        if self.hash_routing:
            self.register_buffer(
                "tid2eid",
                torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
                persistent=True,
            )
            self.register_parameter("bias", None)
        else:
            self.register_buffer("tid2eid", None, persistent=False)
            self.bias = nn.Parameter(torch.zeros(config.n_routed_experts, dtype=torch.float32))

    def forward(
        self,
        hidden_states_flat: torch.Tensor,
        input_ids_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func != "sqrtsoftplus":
            raise ValueError(f"Unsupported DeepSeek V4 scoring_func: {self.score_func}")

        router_logits = _linear(hidden_states_flat.to(self.weight.dtype), self.weight).float()
        scores = F.softplus(router_logits).sqrt()

        if self.hash_routing:
            selected_experts = self.tid2eid[input_ids_flat.to(torch.long)].to(torch.int64)
        else:
            selected_experts = (scores + self.bias).topk(self.top_k, dim=-1).indices

        routing_weights = scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * self.routed_scaling_factor
        return selected_experts, routing_weights


class DeepseekV4MoE(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.n_routed_experts = config.n_routed_experts
        self.swiglu_limit = config.swiglu_limit
        self.gate = DeepseekV4MoEGate(config, layer_idx)
        self.experts = nn.Module()
        self._register_mxfp4_runtime_buffers()
        self._register_load_state_dict_pre_hook(self._load_mxfp4_checkpoint_experts)
        shared_intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepseekV4MLP(
            config.hidden_size,
            shared_intermediate_size,
            swiglu_limit=0.0,
        )

    def _register_mxfp4_runtime_buffers(self) -> None:
        layout = build_deepseek_v4_packed_mxfp4_experts_layout()
        block_size = layout.expert_block_size
        packed_block_width = block_size // layout.packed_values_per_byte
        if self.hidden_size % block_size != 0 or self.intermediate_size % block_size != 0:
            raise ValueError(
                "DeepSeek V4 MXFP4 experts require hidden_size and "
                f"moe_intermediate_size to be divisible by {block_size}."
            )
        self.experts.register_buffer(
            "gate_up_proj_blocks",
            torch.zeros(
                self.n_routed_experts,
                2 * self.intermediate_size,
                self.hidden_size // block_size,
                packed_block_width,
                dtype=torch.uint8,
            ),
            persistent=True,
        )
        self.experts.register_buffer(
            "gate_up_proj_scales",
            torch.zeros(
                self.n_routed_experts,
                2 * self.intermediate_size,
                self.hidden_size // block_size,
                dtype=torch.uint8,
            ),
            persistent=True,
        )
        self.experts.register_buffer(
            "down_proj_blocks",
            torch.zeros(
                self.n_routed_experts,
                self.hidden_size,
                self.intermediate_size // block_size,
                packed_block_width,
                dtype=torch.uint8,
            ),
            persistent=True,
        )
        self.experts.register_buffer(
            "down_proj_scales",
            torch.zeros(
                self.n_routed_experts,
                self.hidden_size,
                self.intermediate_size // block_size,
                dtype=torch.uint8,
            ),
            persistent=True,
        )
        self.experts.register_buffer(
            "gate_up_proj_bias",
            torch.zeros(
                self.n_routed_experts,
                2 * self.intermediate_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.experts.register_buffer(
            "down_proj_bias",
            torch.zeros(self.n_routed_experts, self.hidden_size, dtype=torch.float32),
            persistent=False,
        )

    def _load_mxfp4_checkpoint_experts(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *args,
    ) -> None:
        del args
        layout = build_deepseek_v4_packed_mxfp4_experts_layout()
        target_names = {
            "gate_up_blocks": f"{prefix}experts.gate_up_proj_blocks",
            "gate_up_scales": f"{prefix}experts.gate_up_proj_scales",
            "down_blocks": f"{prefix}experts.down_proj_blocks",
            "down_scales": f"{prefix}experts.down_proj_scales",
        }
        packed = load_packed_mxfp4_expert_tensors(
            state_dict,
            "",
            checkpoint_layout=layout,
            target_names=target_names,
            layer=self.layer_idx,
            num_experts=self.n_routed_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
        )
        if packed is None:
            return

        self._resize_mxfp4_runtime_buffer("gate_up_proj_blocks", packed.gate_up_blocks)
        self._resize_mxfp4_runtime_buffer("gate_up_proj_scales", packed.gate_up_scales)
        self._resize_mxfp4_runtime_buffer("down_proj_blocks", packed.down_blocks)
        self._resize_mxfp4_runtime_buffer("down_proj_scales", packed.down_scales)

    def _resize_mxfp4_runtime_buffer(self, name: str, loaded: torch.Tensor) -> None:
        current = getattr(self.experts, name)
        if current.shape == loaded.shape and current.dtype == loaded.dtype:
            return
        setattr(
            self.experts,
            name,
            torch.empty(loaded.shape, dtype=loaded.dtype, device=current.device),
        )

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, original_shape[-1])
        selected_experts, routing_weights = self.gate(hidden_states_flat, input_ids.reshape(-1))
        routed = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing(
            hidden_states_flat,
            selected_experts,
            routing_weights.to(hidden_states_flat.dtype),
            self.experts.gate_up_proj_blocks,
            self.experts.gate_up_proj_bias,
            self.experts.gate_up_proj_scales,
            1.0,
            float(self.swiglu_limit),
            self.experts.down_proj_blocks,
            self.experts.down_proj_bias,
            self.experts.down_proj_scales,
            "up_gate",
            "deepseek",
            "moe",
        )
        return routed.view(*original_shape).to(hidden_states.dtype) + self.shared_experts(
            hidden_states
        )


class DeepseekV4Compressor(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
    ) -> None:
        super().__init__()
        assert compress_ratio > 0, "DeepSeek V4 compressor requires a positive ratio"
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.rotate = rotate
        self.overlap = compress_ratio == 4
        channels = 2 if self.overlap else 1
        self.max_compressed_len = max(
            1,
            (int(config.ad_compress_max_seq_len) + compress_ratio - 1) // compress_ratio,
        )
        self.max_compressed_tokens = self.max_compressed_len * compress_ratio

        self.ape = nn.Parameter(
            torch.zeros(compress_ratio, channels * self.head_dim, dtype=torch.float32)
        )
        self.wkv = nn.Linear(self.hidden_size, channels * self.head_dim, bias=False)
        self.wgate = nn.Linear(self.hidden_size, channels * self.head_dim, bias=False)
        self.norm = DeepseekV4RMSNorm(self.head_dim, config.rms_norm_eps)

    def project(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        kv_all = _linear_module(hidden_states, self.wkv, layer_type="mla").float()
        score_all = _linear_module(hidden_states, self.wgate, layer_type="mla").float()
        return kv_all, score_all

    def compress_projected(
        self,
        kv_all: torch.Tensor,
        score_all: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        position_ids: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = kv_all.shape
        ratio = self.compress_ratio
        assert seq_len <= self.max_compressed_tokens, (
            "DeepSeek V4 compressor sequence length exceeds ad_compress_max_seq_len"
        )

        row_offsets = torch.arange(self.max_compressed_len, device=kv_all.device)
        token_offsets = torch.arange(ratio, device=kv_all.device)
        gather_idxs = row_offsets.unsqueeze(1) * ratio + token_offsets
        valid = gather_idxs < seq_len
        gather_idxs = torch.where(valid, gather_idxs, torch.zeros_like(gather_idxs))
        flat_idxs = gather_idxs.reshape(-1)

        kv = kv_all[:, flat_idxs].view(batch_size, self.max_compressed_len, ratio, -1)
        score = score_all[:, flat_idxs].view(batch_size, self.max_compressed_len, ratio, -1)
        score = score + self.ape
        score = torch.where(
            valid.view(1, self.max_compressed_len, ratio, 1),
            score,
            score.new_full((), -1.0e20),
        )
        if self.overlap:
            kv = _overlap_transform(kv, self.head_dim, 0.0)
            score = _overlap_transform(score, self.head_dim, -1.0e20)

        compressed = (kv * score.softmax(dim=2)).sum(dim=2).to(output_dtype)
        compressed = self.norm(compressed)

        row_start = row_offsets * ratio
        row_start = torch.minimum(row_start, torch.full_like(row_start, seq_len - 1))
        compressed_position_ids = position_ids[:, row_start]
        cos = cos_table[compressed_position_ids]
        sin = sin_table[compressed_position_ids]

        nope, pe = torch.split(
            compressed,
            [self.head_dim - self.rope_head_dim, self.rope_head_dim],
            dim=-1,
        )
        pe = _apply_interleaved_rope(pe, cos, sin)
        compressed = torch.cat((nope, pe), dim=-1)
        if self.rotate:
            return _fake_fp4_act_quant(_hadamard_rotate(compressed), block_size=32)

        nope, pe = torch.split(
            compressed,
            [self.head_dim - self.rope_head_dim, self.rope_head_dim],
            dim=-1,
        )
        nope = _fake_fp8_act_quant(nope, block_size=64)
        return torch.cat((nope, pe), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        kv_all, score_all = self.project(hidden_states)
        return self.compress_projected(
            kv_all,
            score_all,
            cos_table,
            sin_table,
            position_ids,
            hidden_states.dtype,
        )


class DeepseekV4Indexer(nn.Module):
    def __init__(self, config: DeepseekV4Config, compress_ratio: int) -> None:
        super().__init__()
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = nn.Linear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False
        )
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.compressor = DeepseekV4Compressor(
            config,
            compress_ratio,
            config.index_head_dim,
            rotate=True,
        )

    def project(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        q = _linear_module(
            q_lora,
            self.wq_b,
            tp_mode="colwise",
            layer_type="mla",
            tp_min_local_shape=self.index_head_dim,
        )
        q = torch.ops.auto_deploy.view(
            q,
            [batch_size, seq_len, self.index_n_heads, self.index_head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        q_nope, q_pe = torch.split(
            q,
            [self.index_head_dim - self.rope_head_dim, self.rope_head_dim],
            dim=-1,
        )
        q_pe = _apply_interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
        q = _fake_fp4_act_quant(_hadamard_rotate(torch.cat((q_nope, q_pe), dim=-1)), block_size=32)

        weights = _linear_module(
            hidden_states,
            self.weights_proj,
            tp_mode="colwise",
            layer_type="mla",
        )
        weights = weights.float() * (self.softmax_scale * self.index_n_heads**-0.5)
        compressor_kv, compressor_gate = self.compressor.project(hidden_states)
        return q, weights, compressor_kv, compressor_gate

    def select_topk(
        self,
        q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        seq_len: int,
        offset: int,
    ) -> torch.Tensor:
        index_score = torch.matmul(
            q.transpose(1, 2),
            index_k.transpose(1, 2).unsqueeze(1),
        ).transpose(1, 2)
        index_score = index_score.float()
        index_score = (index_score.relu() * weights.unsqueeze(-1)).sum(dim=2)
        index_score = torch.ops.auto_deploy.all_reduce(index_score, layer_type="mla")

        batch_size = q.shape[0]
        compressed_positions = torch.arange(
            self.compressor.max_compressed_len,
            device=q.device,
        )
        valid_lengths = torch.arange(1, seq_len + 1, device=q.device).unsqueeze(1)
        valid_lengths = valid_lengths // self.compress_ratio
        index_score = index_score.masked_fill(
            (compressed_positions.unsqueeze(0) >= valid_lengths).unsqueeze(0),
            -1.0e20,
        )

        if self.index_topk == 0:
            return torch.empty(batch_size, seq_len, 0, device=q.device, dtype=torch.int64)

        topk_count = min(self.index_topk, self.compressor.max_compressed_len)
        topk_idxs = index_score.topk(topk_count, dim=-1).indices
        invalid = topk_idxs >= valid_lengths.unsqueeze(0)
        topk_idxs = torch.where(invalid, -1, topk_idxs + offset)
        if topk_count < self.index_topk:
            pad = torch.full(
                (batch_size, seq_len, self.index_topk - topk_count),
                -1,
                device=q.device,
                dtype=topk_idxs.dtype,
            )
            topk_idxs = torch.cat((topk_idxs, pad), dim=-1)
        return topk_idxs.to(torch.int64)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        position_ids: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        q, weights, compressor_kv, compressor_gate = self.project(hidden_states, q_lora, cos, sin)
        index_k = self.compressor.compress_projected(
            compressor_kv,
            compressor_gate,
            cos_table,
            sin_table,
            position_ids,
            hidden_states.dtype,
        )
        return self.select_topk(q, index_k, weights, hidden_states.shape[1], offset)


class DeepseekV4Attention(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
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
        assert config.num_key_value_heads == 1, "DeepSeek V4 semantic model expects one KV head"
        assert self.n_heads % self.n_groups == 0, "o_groups must divide num_attention_heads"
        self.group_head_width = (self.n_heads * self.head_dim) // self.n_groups

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(self.q_lora_rank, self.rms_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, self.rms_eps)
        self.wo_a = nn.Linear(self.group_head_width, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        if self.compress_ratio:
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
        position_embeddings: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        cos_base_table, sin_base_table, cos_compress_table, sin_compress_table = position_embeddings
        cos_base = cos_base_table[position_ids]
        sin_base = sin_base_table[position_ids]
        cos_compress = cos_compress_table[position_ids]
        sin_compress = sin_compress_table[position_ids]
        cos = cos_compress if self.compress_ratio else cos_base
        sin = sin_compress if self.compress_ratio else sin_base

        q_lora = _linear_module(hidden_states, self.wq_a, layer_type="mla")
        q_lora = self.q_norm(q_lora)
        q = _linear_module(
            q_lora,
            self.wq_b,
            tp_mode="colwise",
            layer_type="mla",
            tp_min_local_shape=self.head_dim,
        )
        q = torch.ops.auto_deploy.view(
            q,
            [batch_size, seq_len, self.n_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + self.rms_eps).to(q.dtype)

        kv = _linear_module(hidden_states, self.wkv, layer_type="mla")
        kv = self.kv_norm(kv)

        q_nope, q_pe = torch.split(q, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        kv_nope, kv_pe = torch.split(kv, [self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_pe = _apply_interleaved_rope(q_pe, cos.unsqueeze(2), sin.unsqueeze(2))
        kv_pe = _apply_interleaved_rope(kv_pe, cos, sin)
        kv_nope = _fake_fp8_act_quant(kv_nope, block_size=64)
        q = torch.cat((q_nope, q_pe), dim=-1)
        kv = torch.cat((kv_nope, kv_pe), dim=-1)

        empty_compressor_state = kv.new_empty(batch_size, seq_len, 0)
        empty_compressor_ape = kv.new_empty(0, 0)
        empty_compressor_norm_weight = kv.new_empty(0)
        empty_rope_table = kv.new_empty(0, 0)
        empty_indexer_q = q.new_empty(batch_size, seq_len, 0, 0)
        empty_indexer_weights = q.new_empty(batch_size, seq_len, 0)
        if self.compress_ratio:
            window_idxs = _window_topk_idxs(
                self.window_size, batch_size, seq_len, hidden_states.device
            )
            compressor_kv, compressor_gate = self.compressor.project(hidden_states)
            if self.indexer is not None:
                (
                    indexer_q,
                    indexer_weights,
                    indexer_compressor_kv,
                    indexer_compressor_gate,
                ) = self.indexer.project(
                    hidden_states,
                    q_lora,
                    cos_compress,
                    sin_compress,
                )
                index_k = self.indexer.compressor.compress_projected(
                    indexer_compressor_kv,
                    indexer_compressor_gate,
                    cos_compress_table,
                    sin_compress_table,
                    position_ids,
                    hidden_states.dtype,
                )
                compressed_idxs = self.indexer.select_topk(
                    indexer_q,
                    index_k,
                    indexer_weights,
                    seq_len,
                    seq_len,
                )
                indexer_compressor_ape = self.indexer.compressor.ape
                indexer_compressor_norm_weight = self.indexer.compressor.norm.weight
            else:
                indexer_q = empty_indexer_q
                indexer_weights = empty_indexer_weights
                indexer_compressor_kv = empty_compressor_state
                indexer_compressor_gate = empty_compressor_state
                indexer_compressor_ape = empty_compressor_ape
                indexer_compressor_norm_weight = empty_compressor_norm_weight
                compressed_idxs = _compress_topk_idxs(
                    self.compress_ratio,
                    batch_size,
                    seq_len,
                    seq_len,
                    self.compressor.max_compressed_len,
                    hidden_states.device,
                )
            topk_idxs = torch.cat((window_idxs, compressed_idxs), dim=-1).to(torch.int64)
            attn_output = _sparse_attention(
                q,
                kv,
                self.attn_sink,
                topk_idxs,
                compressor_kv,
                compressor_gate,
                self.compressor.ape,
                self.compressor.norm.weight,
                cos_compress_table,
                sin_compress_table,
                position_ids,
                indexer_q,
                indexer_weights,
                indexer_compressor_kv,
                indexer_compressor_gate,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                self.softmax_scale,
                self.layer_idx,
                self.window_size,
                self.compress_ratio,
                self.compressor.max_compressed_len,
                self.rope_head_dim,
                self.rms_eps,
            )
        else:
            topk_idxs = _window_topk_idxs(
                self.window_size, batch_size, seq_len, hidden_states.device
            ).to(torch.int64)
            attn_output = _sparse_attention(
                q,
                kv,
                self.attn_sink,
                topk_idxs,
                empty_compressor_state,
                empty_compressor_state,
                empty_compressor_ape,
                empty_compressor_norm_weight,
                empty_rope_table,
                empty_rope_table,
                position_ids,
                empty_indexer_q,
                empty_indexer_weights,
                empty_compressor_state,
                empty_compressor_state,
                empty_compressor_ape,
                empty_compressor_norm_weight,
                self.softmax_scale,
                self.layer_idx,
                self.window_size,
                self.compress_ratio,
                None,
                None,
                self.rms_eps,
            )

        out_nope, out_pe = torch.split(
            attn_output,
            [self.nope_head_dim, self.rope_head_dim],
            dim=-1,
        )
        out_pe = _apply_interleaved_rope(out_pe, cos.unsqueeze(2), sin.unsqueeze(2), inverse=True)
        attn_output = torch.cat((out_nope, out_pe), dim=-1)
        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [batch_size, seq_len, self.n_groups, self.group_head_width],
            tp_scaled_dim=2,
            layer_type="mla",
        )
        wo_a = torch.ops.auto_deploy.view(
            self.wo_a.weight,
            [self.n_groups, self.o_lora_rank, self.group_head_width],
            tp_scaled_dim=0,
            layer_type="mla",
            tp_min_local_shape=self.o_lora_rank,
        )
        attn_output = torch.ops.auto_deploy.torch_grouped_linear(
            attn_output,
            wo_a,
            None,
            tp_mode="colwise",
            layer_type="mla",
            tp_min_local_shape=self.o_lora_rank,
        )
        attn_output = _linear_module(attn_output, self.wo_b, tp_mode="rowwise", layer_type="mla")
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mla")
        return attn_output


class DeepseekV4Block(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4Attention(config, layer_idx)
        self.ffn = DeepseekV4MoE(config, layer_idx)
        self.attn_norm = DeepseekV4RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = DeepseekV4RMSNorm(config.hidden_size, self.norm_eps)

        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self._reset_hc_parameters()

    def _reset_hc_parameters(self) -> None:
        nn.init.normal_(self.hc_attn_fn, mean=0.0, std=0.02)
        nn.init.normal_(self.hc_ffn_fn, mean=0.0, std=0.02)
        nn.init.zeros_(self.hc_attn_base)
        nn.init.zeros_(self.hc_ffn_base)
        nn.init.ones_(self.hc_attn_scale)
        nn.init.ones_(self.hc_ffn_scale)

    def _hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = _linear(flat, hc_fn) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = torch.sum(pre.unsqueeze(-1) * flat.view(original_shape), dim=2)
        return y.to(x.dtype), post, comb

    @staticmethod
    def _hc_post(
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        y = post.unsqueeze(-1) * x.unsqueeze(-2)
        y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        position_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(hidden_states, position_embeddings, position_ids)
        hidden_states = self._hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states, input_ids)
        return self._hc_post(hidden_states, residual, post, comb)


class DeepseekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = ""
    _no_split_modules = ["DeepseekV4Block"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\.0\."]
    supports_gradient_checkpointing = False

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.register_load_state_dict_pre_hook(self._remap_load_state_hook)

    @staticmethod
    def _remap_load_state_hook(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        *args,
    ) -> None:
        del args
        if prefix:
            return
        _remap_deepseek_v4_checkpoint_keys(state_dict)

    def _init_weights(self, module: nn.Module) -> None:
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, DeepseekV4RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, DeepseekV4MoEGate):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DeepseekV4Compressor):
            module.ape.data.normal_(mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            module.attn_sink.data.zero_()
        elif isinstance(module, DeepseekV4Block):
            module.hc_attn_fn.data.normal_(mean=0.0, std=std)
            module.hc_ffn_fn.data.normal_(mean=0.0, std=std)
            module.hc_attn_base.data.zero_()
            module.hc_ffn_base.data.zero_()
            module.hc_attn_scale.data.fill_(1.0)
            module.hc_ffn_scale.data.fill_(1.0)
        elif isinstance(module, DeepseekV4ForCausalLM):
            module.hc_head_fn.data.normal_(mean=0.0, std=std)
            module.hc_head_base.data.zero_()
            module.hc_head_scale.data.fill_(1.0)


class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    def __init__(self, config: DeepseekV4Config, **kwargs) -> None:
        kwargs.pop("skip_mtp", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected DeepSeek V4 model kwarg(s): {unexpected}")
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_mult = config.hc_mult
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(torch.empty(config.hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(config.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, new_embeddings):
        self.embed = new_embeddings

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def get_decoder(self):
        return self

    def _hc_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        original_shape = hidden_states.shape
        flat = hidden_states.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.config.rms_norm_eps)
        mixes = _linear(flat, self.hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.config.hc_eps
        hidden_states = torch.sum(pre.unsqueeze(-1) * flat.view(original_shape), dim=2)
        return hidden_states.to(original_dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> DeepseekV4CausalLMOutput:
        assert input_ids is not None, "input_ids is required"
        assert position_ids is not None, "position_ids is required"

        hidden_states = self.embed(input_ids)
        position_embeddings = self.rotary_emb()
        hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, position_ids, input_ids)
        hidden_states = self._hc_head(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = _linear(hidden_states.float(), self.head.weight.float(), None).float()
        return DeepseekV4CausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("DeepseekV4Config", DeepseekV4ForCausalLM)


@ModelFactoryRegistry.register("DeepseekV4AutoModelForCausalLM")
class DeepseekV4AutoModelForCausalLMFactory(AutoModelForCausalLMFactory):
    """DeepSeek V4 factory for export sizing and config overrides."""

    def _get_model_config(self) -> Tuple[PretrainedConfig, Dict[str, Any]]:
        model_config, unused_kwargs = super()._get_model_config()
        if (
            "ad_rope_cache_len" in self.model_kwargs
            and "ad_compress_max_seq_len" not in self.model_kwargs
            and hasattr(model_config, "ad_rope_cache_len")
            and hasattr(model_config, "ad_compress_max_seq_len")
        ):
            model_config.ad_compress_max_seq_len = model_config.ad_rope_cache_len
        return model_config, unused_kwargs

    def get_example_inputs(self) -> Dict[str, torch.Tensor]:
        model_config, _ = self._get_model_config()
        ratios = tuple(getattr(model_config, "compress_ratios", ()))
        num_layers = int(getattr(model_config, "num_hidden_layers", len(ratios)))
        active_ratios = [int(ratio) for ratio in ratios[:num_layers] if int(ratio) > 0]
        export_seq_len = max(4, 2 * max(active_ratios, default=0))
        export_cap = min(
            self.max_seq_len,
            int(getattr(model_config, "ad_rope_cache_len", self.max_seq_len)),
            int(getattr(model_config, "ad_compress_max_seq_len", self.max_seq_len)),
        )
        export_seq_len = max(1, min(export_cap, export_seq_len))
        input_ids = torch.ones(2, export_seq_len, dtype=torch.long)
        position_ids = (
            torch.arange(export_seq_len, dtype=torch.long).unsqueeze(0).expand_as(input_ids)
        )
        return {"input_ids": input_ids, "position_ids": position_ids}


DeepseekV4AutoModelForCausalLMFactory.register_custom_model_cls(
    "DeepseekV4Config",
    DeepseekV4ForCausalLM,
)
