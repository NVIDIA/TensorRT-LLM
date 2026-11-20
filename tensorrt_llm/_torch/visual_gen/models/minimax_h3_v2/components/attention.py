# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

from enum import Enum
from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.attention.backends.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

from ..ops.rope import apply_minimax_h3_qk_norm_rope_bf16, apply_minimax_h3_rotary_emb
from ..ops.vsa import H3VSAMetadata, h3_vsa_attention
from .dense_attention import create_h3_attention
from .linear import Linear, shard_bounds
from .norm import PerHeadRMSNorm
from .ulysses_attention import wrap_parallel_attention


class QKVMode(str, Enum):
    FUSE_QKV = "fuse_qkv"
    SEPARATE_QKV = "separate"


def _head_bounds(num_kv_heads: int, tp_size: int, rank: int, ulysses_size: int) -> tuple[int, int]:
    """Partition KV heads across TP ranks in complete Ulysses groups."""
    assert num_kv_heads % ulysses_size == 0
    groups = num_kv_heads // ulysses_size
    assert groups >= tp_size
    start, end = shard_bounds(groups, tp_size, rank)
    return start * ulysses_size, end * ulysses_size


def _dense_attention_model_config(model_config: DiffusionModelConfig) -> DiffusionModelConfig:
    """Return a shallow config copy with VSA disabled for non-video attention."""

    sparse_config = getattr(model_config.attention, "sparse_attention_config", None)
    if sparse_config is None or getattr(sparse_config, "algorithm", None) != "vsa":
        return model_config

    dense_config = model_config.model_copy(deep=False)
    dense_config.attention = model_config.attention.model_copy(deep=True)
    dense_config.attention.sparse_attention_config = None
    return dense_config


class MiniMaxH3RotaryPosEmbed(nn.Module):
    """Three-axis RoPE over packed ``(time, height, width)`` coordinates."""

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                "position_ids must have shape [sequence_length, 3], got "
                f"{list(position_ids.shape)}."
            )
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3Attention(nn.Module):
    """Full non-causal attention over H3's packed text, video and audio rows."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm_eps: float,
        model_config: DiffusionModelConfig,
        layer_idx: int,
        module_name: str,
    ) -> None:
        super().__init__()
        self.head_dim = attention_head_dim
        self.qk_norm = True
        self.fuse_qk_norm_rope = True
        mapping = model_config.mapping
        vgm = model_config.visual_gen_mapping
        u = vgm.ulysses_size if vgm is not None else 1
        tp = mapping.tp_size if mapping else 1
        start, end = _head_bounds(num_attention_heads, tp, mapping.tp_rank if mapping else 0, u)
        self.local_num_attention_heads = self.local_num_key_value_heads = end - start
        self.local_q_dim = self.local_kv_dim = (end - start) * attention_head_dim
        self.use_async_ulysses = bool(u > 1 and model_config.parallel.async_ulysses)
        self.qkv_mode = QKVMode.SEPARATE_QKV if self.use_async_ulysses else QKVMode.FUSE_QKV
        dim = num_attention_heads * attention_head_dim
        shard = (start * attention_head_dim, end * attention_head_dim)
        column = "column" if tp > 1 else None
        linear_args = dict(
            bias=False,
            dtype=model_config.torch_dtype,
            mapping=mapping,
            quant_config=model_config.quant_config,
            force_dynamic_quantization=model_config.force_dynamic_quantization,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
        )
        if self.use_async_ulysses:
            for name in ("to_q", "to_k", "to_v"):
                setattr(
                    self,
                    name,
                    Linear(
                        hidden_size,
                        dim,
                        **linear_args,
                        tensor_parallel_mode=column,
                        reduce_output=False,
                        override_tp_sharding=shard,
                    ),
                )
        else:
            self.qkv_proj = Linear(
                hidden_size,
                3 * dim,
                **linear_args,
                tensor_parallel_mode=column,
                reduce_output=False,
                override_tp_sharding={name: shard for name in ("q", "k", "v")},
                fused_weight_shard_indices_mapping={
                    name: (i * self.local_q_dim, self.local_q_dim)
                    for i, name in enumerate(("q", "k", "v"))
                },
            )
        self.norm_q = PerHeadRMSNorm(
            hidden_size=attention_head_dim,
            eps=qk_norm_eps,
            dtype=model_config.torch_dtype,
            has_weights=True,
        )
        self.norm_k = PerHeadRMSNorm(
            hidden_size=attention_head_dim,
            eps=qk_norm_eps,
            dtype=model_config.torch_dtype,
            has_weights=True,
        )
        self.to_out = nn.ModuleList(
            [
                Linear(
                    dim,
                    hidden_size,
                    **linear_args,
                    tensor_parallel_mode="row" if tp > 1 else None,
                    reduce_output=tp > 1,
                    allreduce_strategy=model_config.allreduce_strategy,
                    override_tp_sharding=shard,
                )
            ]
        )
        dense = _dense_attention_model_config(model_config)
        inner = create_h3_attention(
            dense,
            num_heads=self.local_num_attention_heads // u,
            head_dim=attention_head_dim,
            layer_idx=layer_idx,
            module_name=module_name,
        )
        self.attn = wrap_parallel_attention(
            inner, visual_gen_mapping=vgm, async_ulysses=self.use_async_ulysses, config=model_config
        )

    def get_qkv(self, x):
        if self.use_async_ulysses:
            return self.to_q(x), self.to_k(x), self.to_v(x)
        return self.qkv_proj(x).split(self.local_q_dim, dim=-1)

    def _attn_impl(self, q, k, v, **kwargs):
        batch, seq = q.shape[:2]
        shape = (batch, seq, self.local_num_attention_heads, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        hnd = self.attn.preferred_layout == AttentionTensorLayout.HND
        if hnd:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = self.attn.forward(
            q=q, k=k, v=v, batch_size=batch, seq_len=seq, seq_len_kv=seq, **kwargs
        )
        if hnd:
            out = out.transpose(1, 2)
        return out.flatten(2)

    def _use_fused_qk_norm_rope(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        if not self.fuse_qk_norm_rope or not self.qk_norm or self.qkv_mode != QKVMode.FUSE_QKV:
            return False
        if self.local_num_attention_heads != self.local_num_key_value_heads or self.head_dim != 128:
            return False
        for norm in (self.norm_q, self.norm_k):
            if norm.allreduce is not None or norm.use_gemma or norm.weight.dtype != torch.bfloat16:
                return False
        cos, sin = rotary_emb
        if (
            not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.ndim != 3
            or cos.shape != (hidden_states.shape[1], cos.shape[-1])
            or sin.shape != cos.shape
            or cos.shape[-1] % 32
            or cos.shape[-1] > self.head_dim
        ):
            return False
        if torch.is_grad_enabled() and any(
            t.requires_grad
            for t in (hidden_states, cos, sin, self.norm_q.weight, self.norm_k.weight)
        ):
            return False
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
        vsa_metadata: H3VSAMetadata | None = None,
    ) -> torch.Tensor:
        if self.use_async_ulysses:
            return self.forward_async(
                hidden_states,
                rotary_emb,
                key_padding_mask=key_padding_mask,
                timestep=timestep,
                valid_seq_len=valid_seq_len,
            )
        query, key, value = self._project_qkv(hidden_states, rotary_emb)
        hidden_states = self._attn_impl(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            discard_padded_queries=True,
            valid_seq_len=valid_seq_len,
            timestep=timestep,
        )
        return self.to_out[0](hidden_states)

    def _project_qkv(self, hidden_states, rotary_emb):
        batch_size, sequence_length = hidden_states.shape[:2]
        if rotary_emb is not None and self._use_fused_qk_norm_rope(hidden_states, rotary_emb):
            qkv = self.qkv_proj(hidden_states)
            value = qkv.split([self.local_q_dim, self.local_kv_dim, self.local_kv_dim], dim=-1)[2]
            cos, sin = rotary_emb
            query, key = apply_minimax_h3_qk_norm_rope_bf16(
                qkv,
                self.norm_q.weight,
                self.norm_k.weight,
                cos,
                sin,
                self.norm_q.variance_epsilon,
                self.local_num_attention_heads,
                self.head_dim,
            )
        else:
            query, key, value = self.get_qkv(hidden_states)
            query = query.view(
                batch_size, sequence_length, self.local_num_attention_heads, self.head_dim
            )
            key = key.view(
                batch_size, sequence_length, self.local_num_key_value_heads, self.head_dim
            )
            query, key = self.norm_q(query), self.norm_k(key)
            if rotary_emb is not None:
                query = apply_minimax_h3_rotary_emb(query, *rotary_emb)
                key = apply_minimax_h3_rotary_emb(key, *rotary_emb)
            query = query.flatten(2)
            key = key.flatten(2)
        return query, key, value

    def forward_async(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        valid_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        head_dim = self.head_dim

        def project(linear: Linear, heads: int, norm: PerHeadRMSNorm) -> torch.Tensor:
            states = norm(linear(hidden_states).view(batch_size, sequence_length, heads, head_dim))
            if rotary_emb is not None:
                states = apply_minimax_h3_rotary_emb(states, *rotary_emb)
            return states

        def compute_q() -> torch.Tensor:
            return project(self.to_q, self.local_num_attention_heads, self.norm_q)

        def compute_k() -> torch.Tensor:
            return project(self.to_k, self.local_num_key_value_heads, self.norm_k)

        def compute_v() -> torch.Tensor:
            return self.to_v(hidden_states).view(
                batch_size, sequence_length, self.local_num_key_value_heads, head_dim
            )

        out_4d = self.attn.forward_async(
            compute_q,
            compute_k,
            compute_v,
            key_padding_mask=key_padding_mask,
            discard_padded_queries=True,
            valid_seq_len=valid_seq_len,
            timestep=timestep,
        )
        batch_size, local_seq_len = out_4d.shape[:2]
        return self.to_out[0](out_4d.reshape(batch_size, local_seq_len, -1))


class MiniMaxH3VSAAttention(MiniMaxH3Attention):
    """Apply a checkpoint's learned compression gate to the mixed-modality VSA output."""

    def __init__(
        self, *, model_config, hidden_size, num_attention_heads, attention_head_dim, **kwargs
    ):
        mapping = model_config.visual_gen_mapping or model_config.mapping
        if mapping.world_size != 1:
            raise ValueError("MiniMax-H3 VSA currently supports a single GPU.")
        super().__init__(
            model_config=model_config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            **kwargs,
        )
        self._vsa_backend = model_config.attention.backend
        if self._vsa_backend not in ("TRTLLM", "CUTEDSL"):
            raise ValueError("MiniMax-H3 VSA requires the TRTLLM or CUTEDSL backend.")
        self.to_gate_compress = Linear(
            hidden_size,
            num_attention_heads * attention_head_dim,
            bias=False,
            dtype=model_config.torch_dtype,
            mapping=model_config.mapping,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
        )

    def forward(
        self,
        hidden_states,
        rotary_emb=None,
        key_padding_mask=None,
        timestep=None,
        valid_seq_len=None,
        vsa_metadata: H3VSAMetadata | None = None,
    ):
        if vsa_metadata is None:
            raise RuntimeError("MiniMax-H3 VSA requires packed-sequence metadata.")
        q, k, v = self._project_qkv(hidden_states, rotary_emb)
        shape = (*hidden_states.shape[:2], self.local_num_attention_heads, self.head_dim)
        gate = self.to_gate_compress(hidden_states).view(shape)
        out = h3_vsa_attention(
            q.view(shape),
            k.view(shape),
            v.view(shape),
            gate,
            vsa_metadata,
            self._attn_impl,
            backend=self._vsa_backend,
        )
        return self.to_out[0](out.flatten(2))
