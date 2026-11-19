# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE prologue in one launch: routing top-16 of 896 experts + MXFP8 quantization of the
routed experts' BF16 input (SM100/SM103).

``k31_moe_prologue(logits, bias, scale, x)`` returns ``(topk_ids int32 [M, 16], topk_weights
bf16 [M, 16], x_fp8 e4m3 [M, H], x_sf uint8 [M, H / 32])`` for ``1 <= M <= 65536`` rows: what the
routed-expert scheduler gets from ``kimi_k3_noaux_tc_mxfp8_quant`` (up to 64 rows) or from
``noaux_tc`` (the routing-tail kernel up to 2048 rows) + the BF16 cast + ``mxfp8_quantize``
(above), bit for bit, from one grid:

* CTAs ``[0, ceil(M / 8))`` route 8 tokens each with the routing-tail kernel's selection code
  (:class:`RouteWideKernel`, top-k only; one warp per token).
* CTAs after them quantize 2 rows each (8 above 512 rows), one 32-element block per thread and step,
  with ``mxfp8_quantize``'s arithmetic (UE8M0 scale ``ceil`` of ``amax * rcp.approx(448)``,
  E4M3 round to nearest even, linear row-major scales).

The two halves are independent, so routing and quantization overlap instead of running back to
back; either half can also run alone (``parts``), so a caller can route as soon as the logits exist
and quantize once the input does. Both wait on the previous grid (programmatic dependent launch)
before reading their input, and every CTA releases the dependents after its stores. The weights are
rounded like the path they replace: once from the FP64 expression up to 64 rows (the fused C++ op; FP32 arithmetic
except at BF16 rounding midpoints, :data:`WEIGHTS_BF16_F64_FAST`), the BF16 cast of ``noaux_tc``'s
FP32 weight above.
"""

from __future__ import annotations

import functools
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64

from .route_tail import (
    _WIDE_BIAS_WORDS,
    NUM_EXPERTS,
    TOP_K,
    WEIGHTS_BF16,
    WEIGHTS_BF16_F64_FAST,
    WEIGHTS_RUNNER,
    RouteWideKernel,
    _check_inputs,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _ld_nc_v4_f32,
    _mxfp8_quant_block,
    _rcp_approx_ftz,
    _st_global_b16,
    _st_global_b32,
    _sts_v4_f32,
)

#: Rows quantized per CTA: 2 (224 blocks of 32 for 256 threads) up to ``_QUANT_WIDE_FROM`` rows, 8 above.
_QUANT_ROWS = 2
_QUANT_ROWS_WIDE = 8
_QUANT_WIDE_FROM = 513
#: Largest row count (a prefill chunk of the MoE; top-k only, so no sort table bounds it).
MAX_ROWS = 65536
#: Tokens routed per CTA (one per warp).
_ROUTE_TOKENS = 8
_NUM_WARPS = 8
#: Up to this many rows the weights follow ``kimi_k3_noaux_tc_mxfp8_quant`` (FP64 rounded once).
FUSED_C_MAX_ROWS = 64
#: The halves of the grid a launch runs: routing + quantization, routing only, quantization only.
PARTS = ("all", "route", "quant")


class MoePrologueKernel(RouteWideKernel):
    """Top-16 (no sort) on ``ceil(M / 8)`` CTAs + MXFP8 quantization of ``[M, cols]`` on
    ``ceil(M / quant_rows)`` more CTAs of the same grid (``parts``: both halves, or one alone)."""

    def __init__(
        self,
        cols: int,
        weights: str,
        use_pdl: bool = True,
        quant_rows: int = _QUANT_ROWS,
        parts: str = "all",
    ):
        super().__init__(_ROUTE_TOKENS, _NUM_WARPS, use_pdl, weights)
        if cols % 32:
            raise ValueError(f"cols={cols} must be a multiple of 32")
        self.cols = cols
        self.blocks_per_row = cols // 32
        self.quant_rows = quant_rows
        assert parts in PARTS, parts
        self.parts = parts

    @cute.jit
    def __call__(
        self,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
        x_addr: Int64,
        xq_addr: Int64,
        xsf_addr: Int64,
        stream: cuda.CUstream,
    ):
        scratch_words = self.topk_words

        @cute.struct
        class SharedStorage:
            scratch: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, scratch_words], 16]

        self.shared_storage = SharedStorage
        route_ctas = (m + (self.tokens - 1)) // self.tokens
        quant_ctas = (m + (self.quant_rows - 1)) // self.quant_rows
        if cutlass.const_expr(self.parts == "route"):
            quant_ctas = 0
        if cutlass.const_expr(self.parts == "quant"):
            route_ctas = 0
        self.kernel(
            m,
            route_ctas,
            logits_addr,
            bias_addr,
            ids_addr,
            vals_addr,
            scale_bits,
            x_addr,
            xq_addr,
            xsf_addr,
        ).launch(
            grid=(route_ctas + quant_ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        route_ctas: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
        x_addr: Int64,
        xq_addr: Int64,
        xsf_addr: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        if cta < route_ctas:
            storage = utils.SmemAllocator().allocate(self.shared_storage)
            scratch = storage.scratch.data_ptr().toint()
            bias_smem = scratch
            cand_all = bias_smem + _WIDE_BIAS_WORDS * 4
            top_all = cand_all + self.cand_words * 4
            team = warp
            cand_smem = cand_all + team * (self.region_slots * 16)
            top_smem = top_all + team * (TOP_K * 8)
            t_begin = cta * self.tokens
            t_end = t_begin + self.tokens
            if t_end > m:
                t_end = m
            # The bias is a weight: stage it before waiting on the previous grid.
            for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
                v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
                _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
            cute.arch.barrier()
            for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
                expert, weight = self.select_token(
                    token,
                    lane,
                    logits_addr,
                    bias_smem,
                    scale_bits,
                    cand_smem,
                    top_smem,
                )
                if lane < TOP_K:
                    out = token * TOP_K + lane
                    _st_global_b32(ids_addr + Int64(out * 4), expert)
                    _st_global_b16(vals_addr + Int64(out * 2), weight)
        else:
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_wait()
            rcp448 = _rcp_approx_ftz(Float32(448.0))
            r_begin = (cta - route_ctas) * self.quant_rows
            r_end = r_begin + self.quant_rows
            if r_end > m:
                r_end = m
            n_blocks = (r_end - r_begin) * self.blocks_per_row
            for b in cutlass.range(tidx, n_blocks, self.threads, unroll=1):
                row = r_begin + b // self.blocks_per_row
                blk = b - (b // self.blocks_per_row) * self.blocks_per_row
                elem = Int64(row) * Int64(self.cols) + Int64(blk * 32)
                _mxfp8_quant_block(
                    x_addr + elem * Int64(2),
                    xq_addr + elem,
                    xsf_addr + Int64(row) * Int64(self.blocks_per_row) + Int64(blk),
                    rcp448,
                )
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()


@functools.cache
def _compiled(cols: int, weights: str, use_pdl: bool, quant_rows: int, parts: str = "all"):
    kernel = MoePrologueKernel(cols, weights, use_pdl, quant_rows, parts)
    dummy = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(64),
        *([dummy] * 4),
        Int64(0),
        *([dummy] * 3),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_moe_prologue(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    use_pdl: bool = True,
    weights: str | None = None,
    parts: str = "all",
) -> None:
    """Launch on the current stream into caller-provided outputs (inputs not re-validated).
    ``weights`` overrides the weights mode (a ``route_tail`` ``WEIGHTS_*`` value). ``parts``: ``"route"``
    writes only ``topk_ids`` / ``topk_weights`` (``x`` gives the shape; its data is not read), ``"quant"`` only
    ``x_fp8`` / ``x_sf`` (the logits are not read)."""
    m, cols = x.shape
    if weights is None:
        weights = WEIGHTS_BF16_F64_FAST if m <= FUSED_C_MAX_ROWS else WEIGHTS_BF16
    quant_rows = _QUANT_ROWS if m < _QUANT_WIDE_FROM else _QUANT_ROWS_WIDE
    compiled = _compiled(cols, weights, use_pdl, quant_rows, parts)
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    compiled(
        Int32(m),
        Int64(logits.data_ptr()),
        Int64(e_score_correction_bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(scale_bits),
        Int64(x.data_ptr()),
        Int64(x_fp8.data_ptr()),
        Int64(x_sf.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )


def k31_moe_prologue(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    x: torch.Tensor,
    runner_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-16 of FP32 router logits ``[M, 896]`` with BF16 weights + MXFP8 of BF16 ``x [M, H]``.

    ``runner_weights``: the weights of the TRTLLM-Gen runner's in-kernel routing (``WEIGHTS_RUNNER``; unit
    scale) instead of the separated-routing path's.

    Returns ``(topk_ids int32 [M, 16], topk_weights bf16 [M, 16], x_fp8 [M, H], x_sf uint8 [M, H / 32])``.
    """
    if runner_weights and float(routed_scaling_factor) != 1.0:
        raise ValueError(
            "runner_weights reproduces the in-kernel routing at routed_scaling_factor 1.0 only"
        )
    _check_inputs(logits, e_score_correction_bias)
    m = logits.shape[0]
    if not 1 <= m <= MAX_ROWS:
        raise ValueError(f"rows={m} must be in [1, {MAX_ROWS}]")
    if (
        x.dim() != 2
        or x.shape[0] != m
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[1] % 32
        or x.data_ptr() % 16
    ):
        raise ValueError("x must be a contiguous, 16-byte aligned BF16 [M, H] with H % 32 == 0")
    dev = logits.device
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=dev)
    topk_weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=dev)
    x_fp8 = torch.empty((m, x.shape[1]), dtype=torch.float8_e4m3fn, device=dev)
    x_sf = torch.empty((m, x.shape[1] // 32), dtype=torch.uint8, device=dev)
    run_moe_prologue(
        logits,
        e_score_correction_bias,
        routed_scaling_factor,
        x,
        topk_ids,
        topk_weights,
        x_fp8,
        x_sf,
        weights=WEIGHTS_RUNNER if runner_weights else None,
    )
    return topk_ids, topk_weights, x_fp8, x_sf


def k31_moe_route(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    runner_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The routing half of :func:`k31_moe_prologue` (same launch code, the routing CTAs only): its ``(topk_ids,
    topk_weights)`` of the same logits."""
    if runner_weights and float(routed_scaling_factor) != 1.0:
        raise ValueError(
            "runner_weights reproduces the in-kernel routing at routed_scaling_factor 1.0 only"
        )
    _check_inputs(logits, e_score_correction_bias)
    m = logits.shape[0]
    if not 1 <= m <= MAX_ROWS:
        raise ValueError(f"rows={m} must be in [1, {MAX_ROWS}]")
    dev = logits.device
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=dev)
    topk_weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=dev)
    # the quantization CTAs are not launched: the logits only give the row count in place of x
    run_moe_prologue(logits, e_score_correction_bias, routed_scaling_factor, logits, topk_ids, topk_weights,
                         logits, logits, weights=WEIGHTS_RUNNER if runner_weights else None,
                         parts="route")  # fmt: skip
    return topk_ids, topk_weights


def k31_moe_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The quantization half of :func:`k31_moe_prologue` (same launch code, the quantization CTAs only):
    ``(x_fp8, x_sf)`` of BF16 ``x [M, H]``."""
    m = x.shape[0]
    if (
        x.dim() != 2
        or not 1 <= m <= MAX_ROWS
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[1] % 32
        or x.data_ptr() % 16
    ):
        raise ValueError(
            "x must be a contiguous, 16-byte aligned BF16 [M, H] with H % 32 == 0, M in range"
        )
    x_fp8 = torch.empty((m, x.shape[1]), dtype=torch.float8_e4m3fn, device=x.device)
    x_sf = torch.empty((m, x.shape[1] // 32), dtype=torch.uint8, device=x.device)
    # the routing CTAs are not launched: the logits / bias / id / weight pointers are not read
    run_moe_prologue(x, x, 1.0, x, x, x, x_fp8, x_sf, parts="quant")
    return x_fp8, x_sf


__all__ = [
    "FUSED_C_MAX_ROWS",
    "MAX_ROWS",
    "MoePrologueKernel",
    "k31_moe_prologue",
    "k31_moe_quant",
    "k31_moe_route",
    "run_moe_prologue",
]
