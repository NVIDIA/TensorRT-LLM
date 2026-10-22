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
"""Dense per-tensor FP8 context attention on trtllm-gen kernels (``trtllm::fmha_dense_fp8``)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

from .interface import Fmha, FmhaPhase

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

LOG2E = math.log2(math.e)


def dense_fp8_supported(
    head_dim: int, num_heads: int, out_dtype: torch.dtype, fp16_softmax: bool
) -> bool:
    """Whether a trtllm-gen dense FP8 kernel serves this head layout on the current device."""
    probe = getattr(torch.ops.trtllm, "fmha_dense_fp8_is_supported", None)
    if probe is None:
        return False
    return bool(probe(head_dim, num_heads, out_dtype, fp16_softmax))


def _cu_seqlens(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len


def _bmm1_scales(scale_softmax_log2: torch.Tensor) -> torch.Tensor:
    """``[1/log2(e), scale_softmax_log2]`` as the fp32 [2] device tensor the kernels read."""
    inv_log2e = torch.full((1,), 1.0 / LOG2E, dtype=torch.float32, device=scale_softmax_log2.device)
    return torch.cat([inv_log2e, scale_softmax_log2.float().reshape(1)])


class TrtllmGenDenseFp8Fmha(Fmha):
    """Per-tensor FP8 dense context attention without a KV cache (visual generation).

    BF16 Q, K and V are quantized per tensor; the Q and K dequant scales fold into the base-2
    softmax scale and the V dequant scale into the output scale, as in the cuDNN FP8 backend.
    Serves layers that carry a per-tensor FP8 ``quant_attention_config`` (visual generation),
    so LLM layers never select it.
    """

    supports_per_tensor_fp8 = True

    @staticmethod
    def _per_tensor_fp8_config(attn: "TrtllmAttention") -> bool:
        cfg = getattr(attn, "quant_attention_config", None)
        return (
            cfg is not None
            and cfg.qk_dtype == "fp8"
            and cfg.v_dtype == "fp8"
            and (cfg.q_block_size, cfg.k_block_size, cfg.v_block_size) == (0, 0, 0)
        )

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        if not cls._per_tensor_fp8_config(attn):
            return False
        if attn.is_mla_enable or attn.num_heads != attn.num_kv_heads:
            logger.debug("TrtllmGenDenseFp8Fmha is unavailable: MHA without MLA is required.")
            return False
        if not dense_fp8_supported(attn.head_dim, attn.num_heads, torch.bfloat16, False):
            logger.debug(
                "TrtllmGenDenseFp8Fmha is unavailable: no trtllm-gen dense FP8 kernel for "
                f"head_dim={attn.head_dim}, num_heads={attn.num_heads} on this device."
            )
            return False
        return True

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(q, k, v, metadata, forward_args)
        if not supported:
            logger.debug(f"TrtllmGenDenseFp8Fmha does not support request: {reason}")
        return supported

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> tuple[bool, str]:
        attn = self.attn
        if meta.kv_cache_manager is not None:
            return False, "only cache-free attention is supported."
        if k is None or v is None or fwd.is_fused_qkv:
            return False, "separate Q, K and V are required."
        if meta.is_cross:
            return False, "cross attention is not supported."
        if fwd.attention_input_type == AttentionInputType.generation_only or int(
            meta.num_generations
        ):
            return False, "only the context phase is supported."
        num_contexts = int(meta.num_contexts)
        if num_contexts <= 0:
            return False, "the request contains no context requests."
        if not (fwd.qk_fp8_per_tensor and fwd.v_fp8_per_tensor):
            return False, "only the per-tensor FP8 Q/K/V recipe is supported."
        if (attn.attention_chunk_size or 0) != 0:
            return False, "chunked context attention is not supported."
        try:
            mask_type = AttentionMaskType(fwd.mask_type)
        except (AttributeError, TypeError, ValueError):
            return False, "the attention mask is not dense."
        if mask_type != AttentionMaskType.padding:
            return False, f"attention mask type {mask_type} is not supported."
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            return False, f"BF16 Q, K and V are required, got {q.dtype}/{k.dtype}/{v.dtype}."
        if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
            return False, "attention inputs must be rank 2."
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            return False, "the attention inputs must be contiguous."
        width = attn.num_heads * attn.head_dim
        if q.shape[1] != width or k.shape[1] != width or v.shape[1] != width:
            return False, "Q, K and V widths do not match the layer's heads and head dimension."
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            return False, "equal Q and K/V lengths are required."
        output = fwd.output
        if output is None or output.dtype != torch.bfloat16 or not output.is_contiguous():
            return False, "a contiguous BF16 output tensor is required."
        if fwd.output_sf is not None or fwd.out_scale is not None:
            return False, "quantized attention output is not supported."
        seq_len = int(meta.max_seq_len)
        host_lens = meta.prompt_lens_cpu_runtime
        if host_lens is None or host_lens.numel() < num_contexts:
            return False, "host context lengths are required."
        if (
            seq_len <= 0
            or q.shape[0] != num_contexts * seq_len
            or not bool((host_lens[:num_contexts] == seq_len).all())
        ):
            return False, "uniform sequence lengths are required."
        window = fwd.attention_window_size
        if isinstance(window, int) and 0 < window < seq_len:
            return False, "sliding-window attention is not supported."
        return True, ""

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        attn = self.attn
        batch_size = int(metadata.num_contexts)
        seq_len = int(metadata.max_seq_len)
        num_heads, head_dim = attn.num_heads, attn.head_dim
        tokens = q.shape[0]
        q8, q_descale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(q)
        k8, k_descale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(k)
        v8, v_descale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(v)
        sm_scale = 1.0 / (math.sqrt(head_dim) * attn.q_scaling)
        scale_softmax_log2 = q_descale.float() * k_descale.float() * (sm_scale * LOG2E)
        out = torch.ops.trtllm.fmha_dense_fp8(
            q8.view(tokens, num_heads, head_dim),
            k8.view(tokens, num_heads, head_dim),
            v8.view(tokens, num_heads, head_dim),
            _cu_seqlens(batch_size, seq_len, q.device),
            seq_len,
            _bmm1_scales(scale_softmax_log2),
            v_descale.float().reshape(1),
            False,
            torch.bfloat16,
        )
        forward_args.output.view(tokens, num_heads, head_dim).copy_(out)

    def run_prequantized(
        self,
        q8: torch.Tensor,
        k8: torch.Tensor,
        v8: torch.Tensor,
        v_descale: torch.Tensor,
        batch_size: int,
        seq_len: int,
        out_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attention on e4m3 Q/K/V [B*S, H, D] whose Q/K scales fold the softmax scale.

        Q and K are quantized as ``x * sqrt(softmax_scale * log2(e))`` (no other descale), so
        the FP16-softmax kernels run with a unit softmax scale. Returns BF16 [B*S, H, D], or
        e4m3 quantized with ``out_scale`` (the dequant scale of the consumer's input) when given.
        """
        device = q8.device
        out_dtype = torch.bfloat16 if out_scale is None else torch.float8_e4m3fn
        scale_out = v_descale.float().reshape(1)
        if out_scale is not None:
            scale_out = scale_out / out_scale.float().reshape(1)
        return torch.ops.trtllm.fmha_dense_fp8(
            q8,
            k8,
            v8,
            _cu_seqlens(batch_size, seq_len, device),
            seq_len,
            None,
            scale_out,
            True,
            out_dtype,
        )
