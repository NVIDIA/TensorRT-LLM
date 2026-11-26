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
"""Dense per-tensor FP8 self-attention on trtllm-gen kernels (``trtllm::fmha_dense_fp8``).

The VisualGen TRTLLM backend runs the per-tensor FP8 recipe on Rubin (SM107) through this class:
BF16 Q/K/V quantized per tensor, or Q/K/V an FP8 producer already quantized with the softmax
scale folded into Q and K. Cache-free, unmasked, equal Q and KV lengths, MHA.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

LOG2E = math.log2(math.e)

# Devices the dense FP8 recipe is validated on; ``AttentionConfig`` accepts the recipe for TRTLLM
# on the same SMs.
SUPPORTED_SMS = frozenset({107})


def is_per_tensor_fp8(cfg: Optional["QuantAttentionConfig"]) -> bool:
    """Whether a ``QuantAttentionConfig`` is the per-tensor FP8 recipe (e4m3 Q/K/V, one scale each)."""
    return (
        cfg is not None
        and not cfg.vc_attention
        and cfg.qk_dtype == "fp8"
        and cfg.v_dtype == "fp8"
        and (cfg.q_block_size, cfg.k_block_size, cfg.v_block_size) == (0, 0, 0)
    )


def dense_fp8_supported(
    head_dim: int, num_heads: int, out_dtype: torch.dtype, fp16_softmax: bool
) -> bool:
    """Whether a trtllm-gen dense FP8 kernel serves this head layout on the current device."""
    return bool(
        torch.ops.trtllm.fmha_dense_fp8_is_supported(head_dim, num_heads, out_dtype, fp16_softmax)
    )


def dense_fp8_serves(
    quant_attention_config: Optional["QuantAttentionConfig"], has_sparse_attention: bool
) -> bool:
    """Whether the TRTLLM backend runs this recipe's self-attention on the dense FP8 kernels:
    the per-tensor FP8 recipe, no sparse attention, on a supported SM."""
    return (
        is_per_tensor_fp8(quant_attention_config)
        and not has_sparse_attention
        and get_sm_version() in SUPPORTED_SMS
    )


def _cu_seqlens(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Cumulative sequence offsets [B + 1] for B sequences of ``seq_len`` tokens."""
    return torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len


def _bmm1_scales(scale_softmax_log2: torch.Tensor) -> torch.Tensor:
    """``[1/log2(e), scale_softmax_log2]`` as the fp32 [2] device tensor the kernels read."""
    inv_log2e = torch.full((1,), 1.0 / LOG2E, dtype=torch.float32, device=scale_softmax_log2.device)
    return torch.cat([inv_log2e, scale_softmax_log2.float().reshape(1)])


class DenseFp8Attention:
    """Per-tensor FP8 dense self-attention of one layer on the trtllm-gen dense FP8 kernels."""

    def __init__(self, num_heads: int, head_dim: int, q_scaling: float = 1.0):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_scaling = q_scaling

    @classmethod
    def create(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        quant_attention_config: Optional["QuantAttentionConfig"],
        has_sparse_attention: bool,
        q_scaling: float = 1.0,
    ) -> Optional["DenseFp8Attention"]:
        """The dense FP8 path for a layer, or None when it does not serve the layer: the
        per-tensor FP8 recipe on SM107, no sparse attention, MHA, and a kernel for the heads."""
        if not dense_fp8_serves(quant_attention_config, has_sparse_attention):
            return None
        if num_heads != num_kv_heads:
            logger.debug("trtllm-gen dense FP8 attention requires MHA.")
            return None
        if not dense_fp8_supported(head_dim, num_heads, torch.bfloat16, False):
            logger.debug(
                "No trtllm-gen dense FP8 kernel for "
                f"head_dim={head_dim}, num_heads={num_heads} on this device."
            )
            return None
        return cls(num_heads, head_dim, q_scaling)

    def run(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """BF16 Q/K/V [B*S, H*D] (B sequences of ``seq_len`` tokens) to BF16 [B*S, H, D].

        Q, K and V are quantized per tensor; the Q and K dequant scales fold into the base-2
        softmax scale and the V dequant scale into the output scale, as in the cuDNN FP8 backend.
        """
        tokens = batch_size * seq_len
        heads = (tokens, self.num_heads, self.head_dim)
        q8, q_descale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(q)
        k8, k_descale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(k)
        v8, v_descale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(v)
        sm_scale = 1.0 / (math.sqrt(self.head_dim) * self.q_scaling)
        scale_softmax_log2 = q_descale.float() * k_descale.float() * (sm_scale * LOG2E)
        return torch.ops.trtllm.fmha_dense_fp8(
            q8.view(heads),
            k8.view(heads),
            v8.view(heads),
            _cu_seqlens(batch_size, seq_len, q.device),
            seq_len,
            _bmm1_scales(scale_softmax_log2),
            v_descale.float().reshape(1),
            False,
            torch.bfloat16,
        )

    def supports_prequantized(self, fp8_output: bool = False) -> bool:
        """Whether a kernel takes e4m3 Q/K/V with the softmax scale folded into Q and K, and
        with ``fp8_output`` writes an e4m3 output."""
        out_dtype = torch.float8_e4m3fn if fp8_output else torch.bfloat16
        return dense_fp8_supported(self.head_dim, self.num_heads, out_dtype, True)

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
        """Attention on e4m3 [T, H, D] Q/K/V whose Q and K carry ``x * sqrt(softmax_scale *
        log2(e))``, so the FP16-softmax kernels run with a unit softmax scale. BF16 output, or
        e4m3 quantized with ``out_scale`` (the dequant scale of the consumer's input) when given."""
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
        )
