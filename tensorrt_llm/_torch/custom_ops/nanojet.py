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

"""PyTorch custom operators backed by NanoJet kernels."""

import math

import torch

from ..nanojet_utils import is_nanojet_available
from ..utils import get_model_extra_attrs
from .fast_custom_op import fast_custom_op

_REGISTERED = False


def _model_cache(name: str) -> dict:
    attrs = get_model_extra_attrs()
    if attrs is None:
        raise RuntimeError("NanoJet custom ops require model extra attrs")
    return attrs.setdefault(name, {})


def _scalar(scale: torch.Tensor) -> float:
    if scale.numel() != 1:
        raise RuntimeError(f"NanoJet requires a scalar scale, got shape {tuple(scale.shape)}")
    key = (scale.data_ptr(), scale._version, scale.dtype, scale.device)
    cache = _model_cache("nanojet_scalar_cache")
    if key not in cache:
        value = float(scale.detach())
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"NanoJet requires a positive finite scale, got {value}")
        cache[key] = value
    return cache[key]


def _inverse_scale(scale: torch.Tensor) -> float:
    return 1.0 / _scalar(scale)


def register_nanojet_ops() -> bool:
    """Register NanoJet custom ops on demand."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if not is_nanojet_available():
        return False

    from nanojet_kernels import ops

    from ..modules.attention import extract_extra_attrs

    @fast_custom_op("trtllm::nanojet_rmsnorm_fp8", mutates_args=())
    def nanojet_rmsnorm_fp8(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        shape = hidden_states.shape
        output = ops.unified_rmsnorm(
            hidden_states.reshape(-1, shape[-1]),
            weight,
            eps=eps,
            out_dtype=torch.float8_e4m3fn,
            fp8_scale=_inverse_scale(output_scale),
        )
        return output.view(shape)

    @nanojet_rmsnorm_fp8.register_fake
    def _nanojet_rmsnorm_fp8_fake(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        return torch.empty_like(hidden_states, dtype=torch.float8_e4m3fn)

    @fast_custom_op("trtllm::nanojet_fused_qkv_gemm_norm_rope", mutates_args=())
    def nanojet_fused_qkv_gemm_norm_rope(
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        query_norm_weight: torch.Tensor,
        key_norm_weight: torch.Tensor,
        position_ids: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        eps: float,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
    ) -> torch.Tensor:
        shape = hidden_states.shape
        activation = hidden_states.reshape(-1, shape[-1])
        if activation.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "nanojet_fused_qkv_gemm_norm_rope requires an FP8 activation, "
                f"got {activation.dtype}"
            )
        if num_heads_k != num_heads_v:
            raise RuntimeError("NanoJet QKV fusion requires equal K and V head counts")
        query_size = num_heads_q * head_dim
        key_value_size = num_heads_k * head_dim
        if qkv_weight.shape != (query_size + 2 * key_value_size, activation.shape[-1]):
            raise RuntimeError("NanoJet QKV weight shape does not match the attention layout")
        positions = position_ids.reshape(-1)
        if positions.numel() != activation.shape[0]:
            raise RuntimeError(
                "nanojet_fused_qkv_gemm_norm_rope requires one position ID per token"
            )
        attrs = get_model_extra_attrs()
        if attrs is None or "nanojet_rope_table" not in attrs:
            raise RuntimeError("NanoJet QKV RoPE table was not initialized")

        output = torch.empty(
            activation.shape[0],
            query_size + 2 * key_value_size,
            dtype=torch.bfloat16,
            device=activation.device,
        )
        dequant_scale = _scalar(input_scale) * _scalar(weight_scale)
        ops.fused_qkv_gemm_norm_rope(
            output,
            activation,
            qkv_weight,
            dequant_scale,
            dequant_scale,
            dequant_scale,
            query_size,
            key_value_size,
            query_norm_weight,
            key_norm_weight,
            eps,
            attrs["nanojet_rope_table"],
            positions,
        )
        return output.view(*shape[:-1], output.shape[-1])

    @nanojet_fused_qkv_gemm_norm_rope.register_fake
    def _nanojet_fused_qkv_gemm_norm_rope_fake(
        hidden_states: torch.Tensor,
        qkv_weight: torch.Tensor,
        query_norm_weight: torch.Tensor,
        key_norm_weight: torch.Tensor,
        position_ids: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        eps: float,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
    ) -> torch.Tensor:
        output_size = (num_heads_q + num_heads_k + num_heads_v) * head_dim
        return hidden_states.new_empty(
            (*hidden_states.shape[:-1], output_size),
            dtype=torch.bfloat16,
        )

    @fast_custom_op("trtllm::nanojet_attention_fp8", mutates_args=())
    def nanojet_attention_fp8(
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        attention_mask: str,
        attention_window_size: int | None,
        attention_mask_data: torch.Tensor | None,
        attention_sinks: torch.Tensor | None,
        relative_attention_bias: torch.Tensor | None,
        relative_attention_max_distance: int,
        layer_idx: str,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        metadata, attn_layer = extract_extra_attrs(layer_idx, "attn")
        if (
            metadata.num_generations != 0
            or metadata.num_contexts != metadata.num_seqs
            or metadata.kv_cache_manager is not None
        ):
            raise RuntimeError("NanoJet attention is prefill-only and does not use a KV cache")
        if metadata.multi_item_part_lens is not None:
            raise RuntimeError("NanoJet attention does not support multi-item scoring")
        if attention_mask not in ("causal", "full") or attention_mask_data is not None:
            raise RuntimeError("NanoJet attention does not support custom attention masks")
        assert attention_sinks is None or attention_sinks.numel() == 0, (
            "NanoJet attention does not support attention sinks"
        )
        if relative_attention_bias is not None or relative_attention_max_distance != 0:
            raise RuntimeError("NanoJet attention does not support relative attention bias")

        num_heads = attn_layer.num_heads
        num_kv_heads = attn_layer.num_key_value_heads
        head_dim = attn_layer.head_dim
        if k is None and v is None:
            q, k, v = q.split(
                [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim],
                dim=-1,
            )
        elif k is None or v is None:
            raise RuntimeError("NanoJet attention requires either fused QKV or separate Q, K, V")

        q = q.reshape(-1, num_heads, head_dim)
        k = k.reshape(-1, num_kv_heads, head_dim)
        v = v.reshape(-1, num_kv_heads, head_dim)
        output = torch.empty_like(q, dtype=torch.float8_e4m3fn)
        max_seq_len = metadata.max_seq_len
        cu_q_seqlens = metadata.cu_q_seqlens
        if cu_q_seqlens is None:
            cu_q_seqlens = metadata.mla_prepare_ctx_cu_seqlens()
        if cu_q_seqlens is None:
            raise RuntimeError("NanoJet attention requires context sequence lengths")
        ops.flash_attention(
            q,
            k,
            v,
            out_tensor=output,
            cu_seqlens_q=cu_q_seqlens,
            cu_seqlens_k=cu_q_seqlens,
            max_seqlen_q=max_seq_len,
            max_seqlen_k=max_seq_len,
            softmax_scale=1.0 / (math.sqrt(head_dim) * attn_layer.q_scaling),
            causal=attention_mask == "causal",
            window_size_left=(-1 if attention_window_size is None else attention_window_size),
            output_scale=_inverse_scale(output_scale),
        )
        return output.view(q.shape[0], num_heads * head_dim)

    @nanojet_attention_fp8.register_fake
    def _nanojet_attention_fp8_fake(
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        attention_mask: str,
        attention_window_size: int | None,
        attention_mask_data: torch.Tensor | None,
        attention_sinks: torch.Tensor | None,
        relative_attention_bias: torch.Tensor | None,
        relative_attention_max_distance: int,
        layer_idx: str,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        _, attn_layer = extract_extra_attrs(layer_idx, "attn")
        return q.new_empty(
            (q.shape[0], attn_layer.num_heads * attn_layer.head_dim),
            dtype=torch.float8_e4m3fn,
        )

    @fast_custom_op("trtllm::nanojet_swiglu_gemm_fp8", mutates_args=())
    def nanojet_swiglu_gemm_fp8(
        hidden_states: torch.Tensor,
        gate_up_weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        shape = hidden_states.shape
        activation = hidden_states.reshape(-1, shape[-1])
        if activation.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"nanojet_swiglu_gemm_fp8 requires an FP8 activation, got {activation.dtype}"
            )
        output = ops.swiglu(
            activation,
            gate_up_weight,
            _scalar(input_scale),
            _scalar(weight_scale),
            _scalar(weight_scale),
            _inverse_scale(output_scale),
        )
        return output.view(*shape[:-1], output.shape[-1])

    @nanojet_swiglu_gemm_fp8.register_fake
    def _nanojet_swiglu_gemm_fp8_fake(
        hidden_states: torch.Tensor,
        gate_up_weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_scale: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states.new_empty(
            (*hidden_states.shape[:-1], gate_up_weight.shape[0] // 2),
            dtype=torch.float8_e4m3fn,
        )

    @fast_custom_op("trtllm::nanojet_gemm_fp8_add_", mutates_args=("residual",))
    def nanojet_gemm_fp8_add_(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        activation = hidden_states.reshape(-1, hidden_states.shape[-1])
        accumulator = residual.view(-1, residual.shape[-1])
        if activation.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"nanojet_gemm_fp8_add_ requires an FP8 activation, got {activation.dtype}"
            )
        if activation.shape[-1] != weight.shape[-1]:
            raise RuntimeError(
                "nanojet_gemm_fp8_add_ activation and weight reduction dimensions differ"
            )
        if accumulator.shape != (activation.shape[0], weight.shape[0]):
            raise RuntimeError(
                "nanojet_gemm_fp8_add_ residual shape does not match the GEMM output"
            )
        ops.gemm_fp8_add(
            accumulator,
            activation,
            weight,
            _scalar(input_scale),
            _scalar(weight_scale),
        )

    @nanojet_gemm_fp8_add_.register_fake
    def _nanojet_gemm_fp8_add_fake(
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        return None

    _REGISTERED = True
    return True


__all__ = ["register_nanojet_ops"]
