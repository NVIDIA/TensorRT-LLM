# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 pooling using the DeepSeek-V4 paged compressor kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm._torch.modules.linear import _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE, Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

from .projection import _pdl

# Rows up to which a step's state update, pooling, RMSNorm and index-K projection run as one launch
# (decode-sized steps, where the four separate launches cost more than their work).
COMPRESS_DECODE_MAX_ROWS = 64
# Index-K outputs per program: a row's projection spreads over N / 16 programs.
_N_BLOCK = 16
# Rows up to which a decode-graph step's BF16 index-source projections run on the small-M warp-MMA
# kernel instead of cuBLAS split-K GEMMs.
LOWM_MAX_ROWS = 64


def bf16_projection(linear: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """``linear(x)`` of a bias-free BF16 projection; in decode graphs, steps of up to
    ``LOWM_MAX_ROWS`` rows run the small-M kernel (FP32 accumulation rounded to BF16 once, in
    another order than cuBLAS)."""
    weight = linear.weight
    if (
        _FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
        and getattr(linear, "bias", None) is None
        and x.dim() == 2
        and 0 < x.shape[0] <= LOWM_MAX_ROWS
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    ):
        from tensorrt_llm._torch.modules.low_m_gemm import bf16_lowm_linear, lowm_fp8_mma_fits

        if lowm_fp8_mma_fits(x.shape[0], *weight.shape):
            return bf16_lowm_linear(x, weight, torch.bfloat16)
    return linear(x)


@triton.jit
def _compress_project_kernel(
    X,
    x_stride,
    KV_STATE,
    SCORE_STATE,
    KV_PAGES,
    SCORE_PAGES,
    KV_LENS,
    STARTS,
    CU_SEQ,
    CU_COMP,
    batch,
    max_blocks,
    page_size,
    NORM_W,
    eps,
    WK,
    LATENT,
    INDEX_K,
    D: tl.constexpr,
    N: tl.constexpr,
    RATIO: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    """Program ``(row, block)``: block ``block`` of ``INDEX_K[row] = LATENT[row] @ WK.T`` for the
    normalized pooled window (or, at ratio 1, the normalized row) of compressed row ``row``, zero
    without a completed window. Block 0 also writes ``LATENT[row]`` and, at ratio 2, keeps source
    row ``row`` as FP32 paged state, as the native kernel does."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = tl.arange(0, D)
    outputs = block * N_BLOCK + tl.arange(0, N_BLOCK)
    # The weights do not depend on earlier kernels, so they load under the dependent wait.
    w = tl.load(WK + outputs[:, None] * D + cols[None, :]).to(tl.float32)
    weight = tl.load(NORM_W + cols).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    if RATIO == 1:
        complete = True
    else:
        requests = tl.arange(0, BATCH_BLOCK)
        live = requests < batch
        if (block == 0) & (row < tl.load(CU_SEQ + batch)):
            request = tl.sum((tl.load(CU_SEQ + 1 + requests, live, 2147483647) <= row).to(tl.int32))
            position = tl.load(STARTS + request) + row - tl.load(CU_SEQ + request)
            kv_page = tl.load(KV_PAGES + request * max_blocks + position // page_size)
            score_page = tl.load(SCORE_PAGES + request * max_blocks + position // page_size)
            kv_row = kv_page.to(tl.int64) * page_size + position % page_size
            score_row = score_page.to(tl.int64) * page_size + position % page_size
            tl.store(
                KV_STATE + kv_row * D + cols, tl.load(X + row * x_stride + cols).to(tl.float32)
            )
            # + 0.0: the native kernel adds the (zero) APE before storing.
            score = tl.load(X + row * x_stride + D + cols).to(tl.float32) + 0.0
            tl.store(SCORE_STATE + score_row * D + cols, score)
        owner = tl.sum((tl.load(CU_COMP + 1 + requests, live, 2147483647) <= row).to(tl.int32))
        owner = tl.minimum(owner, batch - 1)
        start = tl.load(STARTS + owner)
        window = start // 2 + row - tl.load(CU_COMP + owner)
        complete = (row < tl.load(CU_COMP + batch)) & (window < tl.load(KV_LENS + owner) // 2)
    if complete:
        if RATIO == 1:
            pooled = tl.load(X + row * x_stride + cols).to(tl.float32)
        else:
            first = window * 2
            base = tl.load(CU_SEQ + owner) - start
            # The native kernel folds the window's new positions in first, then a cached one.
            cached = first < start
            a = tl.where(cached, first + 1, first)
            b = tl.where(cached, first, first + 1)
            k_a = tl.load(X + (base + a) * x_stride + cols).to(tl.float32)
            s_a = tl.load(X + (base + a) * x_stride + D + cols).to(tl.float32)
            if cached:
                kv_page = tl.load(KV_PAGES + owner * max_blocks + b // page_size)
                score_page = tl.load(SCORE_PAGES + owner * max_blocks + b // page_size)
                k_b = tl.load(
                    KV_STATE + (kv_page.to(tl.int64) * page_size + b % page_size) * D + cols
                )
                s_b = tl.load(
                    SCORE_STATE + (score_page.to(tl.int64) * page_size + b % page_size) * D + cols
                )
            else:
                k_b = tl.load(X + (base + b) * x_stride + cols).to(tl.float32)
                s_b = tl.load(X + (base + b) * x_stride + D + cols).to(tl.float32)
            top = tl.maximum(s_a, s_b)
            e_a = tl.exp(s_a - top)
            e_b = tl.exp(s_b - top)
            pooled = (tl.fma(k_a, e_a, k_b * e_b) / (e_a + e_b)).to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / D + eps)
        normalized = ((pooled * inv) * weight).to(tl.bfloat16)
        if block == 0:
            tl.store(LATENT + row * D + cols, normalized)
        projected = tl.sum(w * normalized.to(tl.float32)[None, :], 1)
        tl.store(INDEX_K + row * N + outputs, projected.to(tl.bfloat16))
    else:
        if block == 0:
            tl.store(LATENT + row * D + cols, tl.zeros((D,), tl.bfloat16))
        tl.store(INDEX_K + row * N + outputs, tl.zeros((N_BLOCK,), tl.bfloat16))


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
            latent = bf16_projection(self.wkv, x)
        else:
            if batch is None:
                raise ValueError("Ratio-2 compression requires paged state metadata")
            if batch.kv_state.dtype != torch.float32 or batch.score_state.dtype != torch.float32:
                raise ValueError("CSA2 partial compression state must remain FP32")
            kv_score = bf16_projection(self.wkv_gate, x)
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

    def project(
        self, x: torch.Tensor, batch: CSA2CompressionBatch | None, index_wk: nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(latent, index_wk(latent))`` for :meth:`forward`'s ``latent``; a decode-sized step runs
        the state update, pooling, RMSNorm and index-K projection in one launch."""
        if self._fuses(x, batch, index_wk.weight):
            source = bf16_projection(self.wkv if self.ratio == 1 else self.wkv_gate, x)
            return self._compress_project(source, batch, index_wk.weight)
        latent = self(x, batch)
        return latent, bf16_projection(index_wk, latent)

    def _fuses(self, x, batch, index_weight) -> bool:
        rows = x.shape[0] if x.dim() == 2 else 0
        if not (
            x.is_cuda
            and 0 < rows <= COMPRESS_DECODE_MAX_ROWS
            and index_weight.dtype == self.norm.weight.dtype == torch.bfloat16
            and index_weight.is_contiguous()
            and index_weight.shape[1] == self.head_dim
            and index_weight.shape[0] % _N_BLOCK == 0
            and self.norm.weight.is_contiguous()
        ):
            return False
        if self.ratio == 1:
            return True
        tables = (batch.kv_page_table, batch.score_page_table)
        return (
            batch.output_rows == rows
            and batch.kv_state.dtype == batch.score_state.dtype == torch.float32
            and batch.kv_state.is_contiguous()
            and batch.score_state.is_contiguous()
            and all(t.dtype == torch.int32 and t.dim() == 2 and t.is_contiguous() for t in tables)
            and batch.kv_page_table.shape == batch.score_page_table.shape
            and batch.kv_lengths.shape[0] < 1024
        )

    def _compress_project(self, source, batch, index_weight):
        rows = source.shape[0]
        latent = torch.empty((rows, self.head_dim), dtype=source.dtype, device=source.device)
        index_k = torch.empty(
            (rows, index_weight.shape[0]), dtype=source.dtype, device=source.device
        )
        if self.ratio == 1:
            # Ratio 1 reads no state, page tables or request offsets.
            paged = (source,) * 8
            batch_size, max_blocks, page_size = 0, 0, 1
        else:
            paged = (
                batch.kv_state,
                batch.score_state,
                batch.kv_page_table,
                batch.score_page_table,
                batch.kv_lengths,
                batch.start_positions,
                batch.cu_seq_lengths,
                batch.cu_compressed_lengths,
            )
            batch_size = batch.kv_lengths.shape[0]
            max_blocks, page_size = batch.kv_page_table.shape[1], batch.page_size
        pdl = _pdl()
        _compress_project_kernel[(rows, index_weight.shape[0] // _N_BLOCK)](
            source,
            source.stride(0),
            *paged,
            batch_size,
            max_blocks,
            page_size,
            self.norm.weight,
            self.norm.variance_epsilon,
            index_weight,
            latent,
            index_k,
            self.head_dim,
            index_weight.shape[0],
            self.ratio,
            triton.next_power_of_2(batch_size + 1),
            _N_BLOCK,
            pdl,
            num_warps=4,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
        return latent, index_k
