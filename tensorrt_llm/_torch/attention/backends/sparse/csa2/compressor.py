# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 pooling using the DeepSeek-V4 paged compressor kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm


@dataclass(frozen=True)
class CSA2CompressionBatch:
    """Native paged compression metadata, including state across chunks.

    All metadata tensors are CUDA int32. State pools are FP32; KV and score
    page tables are independent. Output slots include runtime padding when
    used with a CUDA Graph. The caller masks padded rows before publication.
    """

    kv_state: torch.Tensor
    score_state: torch.Tensor
    kv_page_table: torch.Tensor
    score_page_table: torch.Tensor
    kv_lengths: torch.Tensor
    start_positions: torch.Tensor
    cu_seq_lengths: torch.Tensor
    cu_compressed_lengths: torch.Tensor
    output_rows: int
    page_size: int
    max_compressed_per_request: int


class CSA2Compressor(nn.Module):
    """Return normalized pre-RoPE main latents for main KV and index K.

    Ratio 2 uses the existing non-overlap prefill kernel for both chunked
    prefill and multi-token decode; like the V4 compressor it projects values
    and gates with the checkpoint's fused ``wkv_gate`` weight. Ratio 1 has
    neither a gate nor state. The learned absolute-position bias used by V4 is
    absent from CSA2.

    The projections are BF16 ``Linear`` layers producing BF16 rows; the pooling
    kernel promotes them to FP32 and keeps its state and online softmax in
    FP32. The reference promotes a pooling compressor's projections themselves
    to FP32 (model.py:446-448); the only deviation is one BF16 rounding of the
    rows, measured on full GSM8K and GPQA Diamond as indistinguishable from
    run-to-run noise.
    """

    def __init__(self, hidden_size: int, head_dim: int, ratio: int, eps: float) -> None:
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError("CSA2 compressor ratio must be 1 or 2")
        if head_dim not in (128, 512):
            raise ValueError("The native compressor supports head dimensions 128 and 512")
        self.ratio = ratio
        self.head_dim = head_dim
        self.eps = eps
        if ratio == 2:
            self.wkv_gate = Linear(hidden_size, 2 * head_dim, bias=False, dtype=torch.bfloat16)
        else:
            self.wkv = Linear(hidden_size, head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(hidden_size=head_dim, eps=eps, dtype=torch.bfloat16)
        # The existing native ABI accepts APE; a constant zero buffer preserves
        # the ABI without inventing a missing checkpoint weight.
        self.register_buffer(
            "zero_ape", torch.zeros(ratio, head_dim, dtype=torch.float32), persistent=False
        )

    def forward(self, x: torch.Tensor, batch: CSA2CompressionBatch | None = None) -> torch.Tensor:
        if self.ratio == 1:
            latent = self.wkv(x)
        else:
            if batch is None:
                raise ValueError("Ratio-2 compression requires paged state metadata")
            if batch.kv_state.dtype != torch.float32 or batch.score_state.dtype != torch.float32:
                raise ValueError("CSA2 partial compression state must remain FP32")
            kv_score = self.wkv_gate(x)
            # Rows without a complete window are not written by the kernel and
            # must read as zero (chain rewind re-derives state from them).
            latent = torch.zeros((batch.output_rows, self.head_dim), device=x.device, dtype=x.dtype)
            torch.ops.trtllm.compressor_prefill_reduction(
                kv_score,
                self.zero_ape,
                batch.kv_state,
                batch.score_state,
                batch.kv_page_table,
                batch.score_page_table,
                latent,
                batch.kv_lengths,
                batch.start_positions,
                batch.cu_seq_lengths,
                batch.cu_compressed_lengths,
                batch.kv_lengths.shape[0],
                batch.page_size,
                self.head_dim,
                self.ratio,
                batch.max_compressed_per_request,
            )
        # A batch without a completed window has no latent rows to normalize.
        return self.norm(latent) if latent.shape[0] else latent
