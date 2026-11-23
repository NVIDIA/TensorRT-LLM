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
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

from .interface import Fmha, FmhaPhase

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

LOG2E = math.log2(math.e)


def is_per_tensor_fp8(cfg: Optional["QuantAttentionConfig"]) -> bool:
    """Whether a ``QuantAttentionConfig`` is the per-tensor FP8 recipe (e4m3 Q/K/V, one scale each)."""
    return (
        cfg is not None
        and not cfg.vc_attention
        and cfg.qk_dtype == "fp8"
        and cfg.v_dtype == "fp8"
        and (cfg.q_block_size, cfg.k_block_size, cfg.v_block_size) == (0, 0, 0)
    )


# Deferred output rescale of the pre-quantized (FP16-softmax) path: the kernels skip the output
# accumulator correction until the row max has grown by more than 2^1. With any positive
# threshold the kernels quantize P with a scale of 1.75 instead of 448 (P stays below
# 1.75 * 2^1 < 448), so small probabilities lose precision: the deferral is lossy and only
# runs when a caller opts in (defer_prequantized_rescale).
PREQUANTIZED_SKIP_CORRECTION_LOG2 = 1.0


def dense_fp8_supported(
    head_dim: int,
    num_heads: int,
    out_dtype: torch.dtype,
    fp16_softmax: bool,
    skip_correction_threshold: float = 0.0,
) -> bool:
    """Whether a trtllm-gen dense FP8 kernel serves this head layout on the current device."""
    return bool(
        torch.ops.trtllm.fmha_dense_fp8_is_supported(
            head_dim, num_heads, out_dtype, fp16_softmax, skip_correction_threshold
        )
    )


def _cu_seqlens(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Cumulative sequence offsets [B + 1] for B sequences of ``seq_len`` tokens."""
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
    # Devices the dense FP8 recipe is validated on; ``AttentionConfig`` accepts the recipe for
    # TRTLLM on the same SMs.
    SUPPORTED_SMS = frozenset({107})

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        # log2 threshold of the deferred output rescale on the pre-quantized path; 0 rescales the
        # output on every row-max growth.
        self._prequantized_skip_correction = 0.0

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        """Whether the layer carries the per-tensor FP8 recipe and a kernel serves its heads."""
        if not is_per_tensor_fp8(getattr(attn, "quant_attention_config", None)):
            return False
        if get_sm_version() not in cls.SUPPORTED_SMS:
            logger.debug("TrtllmGenDenseFp8Fmha is unavailable: SM107 is required.")
            return False
        if attn.sparse_params is not None:
            logger.debug("TrtllmGenDenseFp8Fmha is unavailable: sparse attention is not supported.")
            return False
        if attn.is_mla_enable or attn.num_heads != attn.num_kv_heads:
            logger.debug("TrtllmGenDenseFp8Fmha is unavailable: MHA without MLA is required.")
            return False
        if not dense_fp8_supported(
            attn.head_dim, attn.num_heads, torch.bfloat16, fp16_softmax=False
        ):
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
        """Whether this request runs here; logs the reason at debug level when it does not."""
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
        """(supported, reason) for cache-free, uniform-length per-tensor FP8 context attention."""
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
        mask_type = AttentionMaskType(fwd.mask_type)
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
        """Quantize BF16 Q/K/V per tensor and run the dense FP8 kernel into ``forward_args.output``."""
        attn = self.attn
        batch_size = int(metadata.num_contexts)
        seq_len = int(metadata.max_seq_len)
        # Library selection is cached without sequence lengths, so a ragged batch can reach
        # this path on a cache hit. Re-check before the uniform-length kernel call.
        host_lens = metadata.prompt_lens_cpu_runtime[:batch_size]
        if q.shape[0] != batch_size * seq_len or not bool((host_lens == seq_len).all()):
            raise RuntimeError("trtllm-gen dense FP8 attention requires uniform sequence lengths.")
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

    def supports_prequantized_fp8(self, fp8_output: bool = False) -> bool:
        """Whether a kernel takes e4m3 Q/K/V with the softmax scale folded into Q and K, and
        with ``fp8_output`` writes an e4m3 output."""
        attn = self.attn
        out_dtype = torch.float8_e4m3fn if fp8_output else torch.bfloat16
        return dense_fp8_supported(attn.head_dim, attn.num_heads, out_dtype, fp16_softmax=True)

    def defer_prequantized_rescale(self, enabled: bool) -> bool:
        """Defer the pre-quantized path's output rescale up to a 2^1 row-max growth (lossy, see
        PREQUANTIZED_SKIP_CORRECTION_LOG2); returns whether it is deferred. Needs a kernel that
        takes the threshold for both BF16 and e4m3 output."""
        attn = self.attn
        deferred = enabled and all(
            dense_fp8_supported(
                attn.head_dim,
                attn.num_heads,
                dtype,
                fp16_softmax=True,
                skip_correction_threshold=PREQUANTIZED_SKIP_CORRECTION_LOG2,
            )
            for dtype in (torch.bfloat16, torch.float8_e4m3fn)
        )
        self._prequantized_skip_correction = PREQUANTIZED_SKIP_CORRECTION_LOG2 if deferred else 0.0
        return deferred

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
        """Attention on e4m3 [T, H, D] Q/K/V; BF16 output, or e4m3 when ``out_scale`` is given."""
        out_dtype = torch.bfloat16 if out_scale is None else torch.float8_e4m3fn
        scale_out = v_descale.float().reshape(1)
        if out_scale is not None:
            scale_out = scale_out / out_scale.float().reshape(1)
        return torch.ops.trtllm.fmha_dense_fp8(
            q8,
            k8,
            v8,
            _cu_seqlens(batch_size, seq_len, q8.device),
            seq_len,
            None,
            scale_out,
            True,
            out_dtype,
            self._prequantized_skip_correction,
        )
