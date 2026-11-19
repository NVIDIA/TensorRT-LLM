# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""H3's text-only Qwen3-VL encoder on a tensor-parallel subgroup."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig

from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog.attention.fused_qk_norm_rope import fused_qk_norm_rope
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._utils import get_hf_rope_theta
from tensorrt_llm.mapping import Mapping, MpiTopology

from ..components.linear import Linear
from ..packing import MINIMAX_H3_TEXT_ENCODER_LAYER
from ..parallel.tp import shard_bounds
from .ops.text_attention import text_causal_attention


class _MiniMaxH3TextEncoderMapping(MpiTopology):
    """Mapping for the text-encoder-only TP subgroup."""

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        return object.__new__(cls)

    def __init__(self, *args, tp_group_pg=None, **kwargs) -> None:
        self._tp_group_pg = tp_group_pg
        self._use_torch_distributed_allreduce = True
        super().__init__(*args, **kwargs)

    @property
    def tp_group_pg(self):
        if self._tp_group_pg is None:
            raise RuntimeError("MiniMax-H3 text encoder TP process group is not initialized.")
        return self._tp_group_pg

    @property
    def tp_group_name(self) -> str:
        return self.tp_group_pg.group_name


def _norm(x, residual, weight, eps):
    if IS_FLASHINFER_AVAILABLE:
        if residual is None:
            return flashinfer_rmsnorm(x, weight, eps), x
        return flashinfer_fused_add_rmsnorm(x, residual, weight, eps)
    total = x.float() if residual is None else x.float() + residual.float()
    normalized = total * torch.rsqrt(total.square().mean(dim=-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype), total.to(x.dtype)


def _rope_scaling(config, rope, rotary_dim, theta):
    if rope.get("rope_type") != "yarn":
        return 1.0, 0.0, 0.0, 1.0

    max_pos = rope.get("original_max_position_embeddings", config.max_position_embeddings)
    factor = (
        config.max_position_embeddings / max_pos
        if "original_max_position_embeddings" in rope
        else rope.get("factor", 1.0)
    )

    def correction_dim(rotations):
        return rotary_dim * math.log(max_pos / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = correction_dim(rope.get("beta_fast") or 32)
    high = correction_dim(rope.get("beta_slow") or 1)
    if rope.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, rotary_dim - 1)
    attention_factor = rope.get("attention_factor")
    if attention_factor is None:

        def mscale(value):
            return 1.0 + 0.1 * value * math.log(factor) if factor > 1 else 1.0

        scale, all_dim = rope.get("mscale"), rope.get("mscale_all_dim")
        attention_factor = mscale(scale) / mscale(all_dim) if scale and all_dim else mscale(1)
    return factor, low, high, attention_factor


class _TextBlock(nn.Module):
    def __init__(self, config, mapping, device):
        super().__init__()
        size, rank = mapping.tp_size, mapping.tp_rank
        hidden = config.hidden_size
        n_q, n_kv = config.num_attention_heads, config.num_key_value_heads
        self.dim = getattr(config, "head_dim", None) or hidden // n_q
        assert n_q % size == 0
        assert max(n_kv, size) % min(n_kv, size) == 0
        self.n_q = n_q // size
        self.n_kv = max(1, n_kv // size)
        self.eps = config.rms_norm_eps
        self.theta = get_hf_rope_theta(config)
        self.rotary_dim = int(self.dim * getattr(config, "partial_rotary_factor", 1.0))
        rope = (
            getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None) or {}
        )
        self.rope_scaling = _rope_scaling(config, rope, self.rotary_dim, self.theta)
        self.sections = rope.get("mrope_section") if rope.get("mrope_interleaved") else None
        self.q_width, self.kv_width = self.n_q * self.dim, self.n_kv * self.dim
        self.q_start = rank * self.q_width
        # When TP exceeds the KV head count, consecutive ranks share one KV head.
        self.kv_start = rank * n_kv // size * self.dim
        self.ff_slice = slice(*shard_bounds(config.intermediate_size, size, rank))
        local_ffn = self.ff_slice.stop - self.ff_slice.start
        attn_bias = bool(getattr(config, "attention_bias", False))
        mlp_bias = bool(getattr(config, "mlp_bias", False))

        def projection(in_features, out_features, bias):
            return Linear(in_features, out_features, bias=bias, dtype=torch.bfloat16).to(device)

        self.qkv_proj = projection(hidden, self.q_width + 2 * self.kv_width, attn_bias)
        self.o_proj = projection(self.q_width, hidden, attn_bias and rank == 0)
        self.gate_up_proj = projection(hidden, 2 * local_ffn, mlp_bias)
        self.down_proj = projection(local_ffn, hidden, mlp_bias and rank == 0)

        def gain(width):
            return nn.Parameter(
                torch.ones(width, device=device, dtype=torch.bfloat16), requires_grad=False
            )

        self.input_norm = gain(hidden)
        self.q_norm = gain(self.dim)
        self.k_norm = gain(self.dim)
        self.post_attn_norm = gain(hidden)

    def forward(self, x, residual, positions, offsets, reduce):
        x, residual = _norm(x, residual, self.input_norm, self.eps)
        qkv = fused_qk_norm_rope(
            self.qkv_proj(x),
            self.n_q,
            self.n_kv,
            self.n_kv,
            self.dim,
            self.rotary_dim,
            self.eps,
            self.q_norm,
            self.k_norm,
            self.theta,
            True,
            positions,
            *self.rope_scaling,
            use_mrope=self.sections is not None,
            mrope_section1=self.sections[1] if self.sections is not None else 0,
            mrope_section2=self.sections[2] if self.sections is not None else 0,
        )
        q, k, v = (
            part.reshape(x.shape[0], heads, self.dim).contiguous()
            for part, heads in zip(
                qkv.split((self.q_width, self.kv_width, self.kv_width), dim=-1),
                (self.n_q, self.n_kv, self.n_kv),
                strict=True,
            )
        )
        x = reduce(self.o_proj(text_causal_attention(q, k, v, offsets).flatten(1)))
        x, residual = _norm(x, residual, self.post_attn_norm, self.eps)
        x = self.down_proj(torch.ops.trtllm.silu_and_mul(self.gate_up_proj(x)))
        return reduce(x), residual

    def load_weights(self, weights, prefix):
        def get(key):
            return weights[prefix + key]

        q_rows = slice(self.q_start, self.q_start + self.q_width)
        kv_rows = slice(self.kv_start, self.kv_start + self.kv_width)
        for suffix in ("weight", "bias"):
            qkv = getattr(self.qkv_proj, suffix)
            if qkv is not None:
                for target, name, rows in zip(
                    qkv.split((self.q_width, self.kv_width, self.kv_width)),
                    ("q", "k", "v"),
                    (q_rows, kv_rows, kv_rows),
                    strict=True,
                ):
                    target.copy_(get(f"self_attn.{name}_proj.{suffix}")[rows])
            gate_up = getattr(self.gate_up_proj, suffix)
            if gate_up is not None:
                for target, name in zip(gate_up.chunk(2), ("gate", "up"), strict=True):
                    target.copy_(get(f"mlp.{name}_proj.{suffix}")[self.ff_slice])
        self.o_proj.weight.copy_(get("self_attn.o_proj.weight")[:, q_rows])
        self.down_proj.weight.copy_(get("mlp.down_proj.weight")[:, self.ff_slice])
        for module, name in ((self.o_proj, "self_attn.o_proj"), (self.down_proj, "mlp.down_proj")):
            if module.bias is not None:
                module.bias.copy_(get(name + ".bias")[:])
        self.input_norm.copy_(get("input_layernorm.weight")[:])
        self.q_norm.copy_(get("self_attn.q_norm.weight")[:])
        self.k_norm.copy_(get("self_attn.k_norm.weight")[:])
        self.post_attn_norm.copy_(get("post_attention_layernorm.weight")[:])


class MiniMaxH3TensorParallelTextEncoder(nn.Module):
    """Encode one prompt through the first target_layer Qwen3-VL text blocks."""

    def __init__(
        self,
        text_config: Any,
        mapping: Mapping,
        device: torch.device,
        *,
        target_layer: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
        max_num_tokens: int | None = None,
    ) -> None:
        super().__init__()
        if text_config.num_hidden_layers <= target_layer:
            raise ValueError(
                "MiniMax-H3 requires the unnormalized hidden state after Qwen3-VL "
                f"layer {target_layer}, but the encoder has {text_config.num_hidden_layers} layers."
            )
        self.target_layer = target_layer
        self.mapping = mapping
        self.device = device
        self.dtype = torch.bfloat16
        self.vocab_size = text_config.vocab_size
        local_vocab = math.ceil(self.vocab_size / mapping.tp_size)
        self.vocab_start = mapping.tp_rank * local_vocab
        self.vocab_stop = min(self.vocab_start + local_vocab, self.vocab_size)
        self.embedding = nn.Parameter(
            torch.empty(local_vocab, text_config.hidden_size, dtype=self.dtype, device=device),
            requires_grad=False,
        )
        self.reduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL, dtype=self.dtype)
        self.layers = nn.ModuleList(
            _TextBlock(text_config, mapping, device) for _ in range(target_layer)
        )
        self.eval()

    @classmethod
    def from_pretrained_config(
        cls,
        checkpoint_dir: str | Path,
        mapping: Mapping,
        device: torch.device,
        *,
        target_layer: int = MINIMAX_H3_TEXT_ENCODER_LAYER,
    ) -> MiniMaxH3TensorParallelTextEncoder:
        config = AutoConfig.from_pretrained(checkpoint_dir, subfolder="text_encoder")
        if not hasattr(config, "text_config"):
            raise ValueError("MiniMax-H3 text_encoder config must contain text_config.")
        return cls(config.text_config, mapping, device, target_layer=target_layer)

    @torch.no_grad()
    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        weights = {
            key.removeprefix("model.").removeprefix("language_model."): value
            for key, value in weights.items()
        }
        rows = weights["embed_tokens.weight"][self.vocab_start : self.vocab_stop]
        self.embedding[: len(rows)].copy_(rows)
        self.embedding[len(rows) :].zero_()
        for i, layer in enumerate(self.layers):
            layer.load_weights(weights, f"layers.{i}.")

    @torch.inference_mode()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        valid = (input_ids >= self.vocab_start) & (input_ids < self.vocab_stop)
        ids = (input_ids - self.vocab_start).masked_fill(~valid, 0)
        x = self.reduce(F.embedding(ids, self.embedding).masked_fill_(~valid.unsqueeze(-1), 0))
        seq_len = len(input_ids)
        offsets = torch.tensor([0, seq_len], device=self.device, dtype=torch.int32)
        positions = torch.arange(seq_len, device=self.device, dtype=torch.int32)
        if self.layers and self.layers[0].sections is not None:
            positions = positions.expand(3, -1).contiguous()
        residual = None
        for block in self.layers:
            x, residual = block(x, residual, positions, offsets, self.reduce)
        if residual is not None:
            x = x + residual
        return x.unsqueeze(0)
