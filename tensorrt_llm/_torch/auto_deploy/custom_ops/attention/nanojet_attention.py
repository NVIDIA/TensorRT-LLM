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

"""nanojet FlashAttention-3 as an auto-deploy attention backend. Prefill only.

Quantizes to e4m3 in the kernel epilogue when handed an ``output_scale``, so ``o_proj`` can
read the result directly.
"""

import math
from typing import List, Optional

import torch
from torch.fx import Node

from ....nanojet_utils import is_nanojet_available
from ...utils.nanojet_graph import per_tensor_scale
from ..attention_interface import AttentionRegistry, BatchInfo, Constant, MHACallable
from .torch_backend_attention import TorchBackendAttention

_REGISTERED = False

# Where ``fuse_nanojet_attn_quant_fp8`` records the scale the reader quantizes by.
NANOJET_ATTENTION_INPUT_SCALE = "nanojet_attention_input_scale"


def register() -> bool:
    """Define the ops, importing nanojet only now. Idempotent; returns availability."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if not is_nanojet_available():
        return False
    _REGISTERED = True

    from nanojet_kernels import ops

    @torch.library.custom_op("auto_deploy::nanojet_attention", mutates_args=())
    def nanojet_mha_with_cache(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        # STANDARD METADATA
        batch_info_host: torch.Tensor,
        seq_len: torch.Tensor,
        input_pos: torch.Tensor,
        slot_idx: torch.Tensor,
        cu_seqlen: torch.Tensor,
        # EXTRA METADATA
        # CONSTANTS
        scale: Optional[float],
        sinks: Optional[torch.Tensor] = None,
        sliding_window_size: Optional[int] = None,
        logit_cap: Optional[float] = None,
        read_cache_only: bool = False,
        output_scale: Optional[float] = None,
        custom_attn_mask: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Varlen FlashAttention over the Q/K/V in hand. Prefill only."""
        batch_info = BatchInfo(batch_info_host)
        num_seq, num_extend, num_decode = batch_info.get_num_sequences()
        num_total_tokens, _, _ = batch_info.get_num_tokens()
        if num_decode:
            raise RuntimeError(
                f"nanojet attention backend is prefill-only, but this batch has {num_decode} "
                "decode sequence(s). It keeps no KV cache. Use attn_backend='trtllm' or "
                "'flashinfer' for generation."
            )
        if read_cache_only:
            raise RuntimeError(
                "nanojet attention backend was asked to read from a KV cache (shared-KV "
                "layer), which it does not keep. Use attn_backend='trtllm' or 'flashinfer'."
            )
        if custom_attn_mask is not None and custom_attn_mask.numel() > 0:
            raise RuntimeError(
                "nanojet attention backend does not support a non-empty custom_attn_mask"
            )
        assert sinks is None or sinks.numel() == 0, "nanojet attention does not support sinks"
        assert logit_cap is None or logit_cap <= 0.0, (
            "nanojet attention does not support logit_cap"
        )
        if num_extend:
            raise RuntimeError(
                f"nanojet attention backend received {num_extend} continuation chunk(s) "
                "(extend request: cached context plus new tokens). The keys and values for the "
                "earlier part of the sequence live in a KV cache it does not keep, so attending "
                "over only this chunk would be silently wrong. Disable chunked prefill, or use "
                "attn_backend='trtllm' or 'flashinfer'."
            )

        batch, seq = q.shape[:2]
        qk_head_dim = q.shape[-1]
        v_head_dim = v.shape[-1]
        num_heads = q.shape[2] if q.ndim == 4 else q.shape[2] // qk_head_dim
        num_kv_heads = k.shape[2] if k.ndim == 4 else k.shape[2] // qk_head_dim
        output_shape = (
            (batch, seq, num_heads * v_head_dim)
            if q.ndim == 3
            else (batch, seq, num_heads, v_head_dim)
        )
        bs_view = (batch, seq) if seq == 1 else (batch * seq,)

        q = q.reshape(*bs_view, num_heads, qk_head_dim)
        k = k.reshape(*bs_view, num_kv_heads, qk_head_dim)
        v = v.reshape(*bs_view, num_kv_heads, v_head_dim)
        if scale is None:
            scale = 1.0 / math.sqrt(qk_head_dim)

        result_dtype = torch.float8_e4m3fn if output_scale is not None else q.dtype
        result = torch.empty(*bs_view, num_heads, v_head_dim, dtype=result_dtype, device=q.device)
        cumulative = cu_seqlen[: num_seq + 1].to(torch.int32)
        max_seqlen = int(seq)
        ops.flash_attention(
            q,
            k,
            v,
            out_tensor=result,
            cu_seqlens_q=cumulative,
            cu_seqlens_k=cumulative,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scale,
            causal=True,
            window_size_left=-1 if sliding_window_size is None else sliding_window_size,
            output_scale=1.0 if output_scale is None else output_scale,
        )
        if out is not None:
            out_flat = out.view(*bs_view, num_heads, v_head_dim)
            out_flat[:num_total_tokens].copy_(result[:num_total_tokens])
            if num_total_tokens < out_flat.shape[0]:
                out_flat[num_total_tokens:].zero_()
            return out.new_empty(0)
        if num_total_tokens < result.shape[0]:
            result[num_total_tokens:].zero_()
        return result.view(*output_shape)

    @nanojet_mha_with_cache.register_fake
    def _(
        q,
        k,
        v,
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        scale,
        sinks=None,
        sliding_window_size=None,
        logit_cap=None,
        read_cache_only=False,
        output_scale=None,
        custom_attn_mask=None,
        out=None,
    ):
        v_head_dim = v.shape[-1]
        dtype = torch.float8_e4m3fn if output_scale is not None else q.dtype
        if out is not None:
            return out.new_empty(0)
        if q.ndim == 3:
            return torch.empty(*q.shape[:3], dtype=dtype, device=q.device)
        return torch.empty(*q.shape[:3], v_head_dim, dtype=dtype, device=q.device)

    return True


@AttentionRegistry.register("nanojet")
class NanojetAttention(TorchBackendAttention):
    """Varlen FlashAttention-3 over an unpaged cache.

    Everything except the kernel itself — layout, cache shape, metadata, constants —
    matches the torch backend, so those are inherited rather than restated.
    """

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        if not register():
            raise RuntimeError(
                "attention backend 'nanojet' was selected but nanojet is not importable"
            )
        return torch.ops.auto_deploy.nanojet_attention.default

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Quantize in the epilogue by the scale ``fuse_nanojet_attn_quant_fp8`` recorded."""
        scale = source_attn_node.meta.get(NANOJET_ATTENTION_INPUT_SCALE)
        output_scale = (
            None
            if scale is None
            else 1.0 / per_tensor_scale(source_attn_node.graph.owning_module, scale)
        )
        return list(super().get_constants(source_attn_node)) + [output_scale]

    @classmethod
    def get_cache_initializers(cls, source_attn_node, cache_config):
        """No caches. Prefill has every key and value already."""
        return {}
