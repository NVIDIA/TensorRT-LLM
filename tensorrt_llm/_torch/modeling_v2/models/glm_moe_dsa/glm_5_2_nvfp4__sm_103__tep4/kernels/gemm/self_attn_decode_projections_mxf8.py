# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 decode-row (M <= 64) projections kv_a, q_b, o_proj on MXF8 weights.

The context path (``self_attn_prefill_projections_mxf8``) already keeps each trunk
layer's kv_a and o_proj weights (and the FP8 route's q_b) resident as MXF8.  At decode
sizes those projections are weight-read bound, so running them on the same e4m3 bytes
reads half the weight traffic of the BF16 cells.  One launch pair per call:

1. ``self_attn_prefill_act_quant_mxf8`` in its PDL mode: quantizes the decode rows into a
   persistent 64-row scratch (e4m3 + blocked e8m0 scales; scale rows past the batch read
   code 127) after ``griddepcontrol.wait``, having released its dependent at entry;
2. ``_projection_tcgen05_mxf8_template``: the block-scaled GEMM, launched as that quant's
   programmatic dependent, streams its weight stages (and prefetches the rest of its
   weight slice into L2) before its own wait, then reads the activation.

decode act-quant fusion (``gemm_only``): when the producer already wrote the activation into this
launcher's scratch (``xq`` / ``sfa``) -- the decode q_a RMSNorm's MXF8 copy for q_b, the
V-B GEMM's MXF8 side output for o_proj -- only the GEMM launches, as that producer's
programmatic dependent.  The producer writes the bytes the quant would.

Numerics: the same MXF8 operands as the context path's GEMM (PROJ-FP8 stage 0: real-prompt
78-layer forward ppl 3.54 vs 3.59 BF16); FP32 accumulation, BF16 output.

Pattern: glm-decode-proj-mxf8
Device: SM100+ (Blackwell)
Shape: rows 1..64; (N, K) = kv_a (2624, 6144), q_b (4096, 2048), o_proj (6144, 4096)
"""

from __future__ import annotations

import torch

from ..common import self_attn_prefill_act_quant_mxf8 as _quant
from . import _projection_tcgen05_mxf8_template as _tpl
from . import self_attn_prefill_projections_mxf8 as _ctx

ROWS_MAX = _tpl.ROWS_MAX
QUANT_PASSES = 4  # 64 rows per quant CTA: one row tile covers every decode bucket

CONFIGS: dict[str, _tpl.Mxf8ProjectionConfig] = {
    "kv_a_proj_with_mqa": _tpl.Mxf8ProjectionConfig(
        "self_attn.kv_a_proj_with_mqa",
        2_624,
        6_144,
        mma_n=64,
        stages=5,
        split_k=3,
        kernel_name_prefix="glm52_dec_proj_mxf8_kv_a_proj_with_mqa",
    ),
    "q_b_proj": _tpl.Mxf8ProjectionConfig(
        "self_attn.q_b_proj",
        4_096,
        2_048,
        mma_n=64,
        stages=4,
        split_k=2,
        kernel_name_prefix="glm52_dec_proj_mxf8_q_b_proj",
    ),
    "o_proj": _tpl.Mxf8ProjectionConfig(
        "self_attn.o_proj",
        6_144,
        4_096,
        mma_n=64,
        stages=6,
        split_k=1,
        kernel_name_prefix="glm52_dec_proj_mxf8_o_proj",
    ),
}
QuantizedWeight = _ctx.QuantizedWeight
quantize_weight = _ctx.quantize_weight

#: MTP3 / MTP5 decode rows (32 sequences x widths 4 / 6, and the ragged counts of fewer
#: sequences): ``WideDecodeMxf8Projection`` serves rows ``WIDE_ROWS_MIN..WIDE_ROWS_MAX``.
WIDE_ROWS_MIN = ROWS_MAX + 1
WIDE_ROWS_MAX = 192
#: Per op, the wide launcher's template kernel for rows up to its ``rows_max`` (128): one CTA
#: per 64 output columns, the weight streamed before the PDL wait, split-K 1 (the K order of the
#: context MXF8 GEMM those rows took before, same operands: its output bitwise).
WIDE_CONFIGS: dict[str, _tpl.Mxf8ProjectionConfig] = {
    "o_proj": _tpl.Mxf8ProjectionConfig(
        "self_attn.o_proj",
        6_144,
        4_096,
        mma_n=64,
        stages=8,
        split_k=1,
        a_box_rows=128,
        k_sub=1,
        rows_max=128,
        kernel_name_prefix="glm52_dec_proj_mxf8_o_proj_w128",
    ),
}
#: Per op, above the template's rows (129..192): the context MXF8 GEMM's (ABI op, tile)
#: instance compiled with the PDL weights-first prologue (``compile_gemm(pdl=True)``). Its
#: 2-CTA M = 256 MMA splits the rows over a CTA pair, where the template's second 128-row
#: block would have every CTA read 192 activation rows per 64 weight columns (bound by the
#: tcgen05 operand reads: B300 ncu, 192 rows, 15.3 us against 12.0 us standalone).
WIDE_CTX: dict[str, tuple[str, str]] = {"o_proj": ("o_proj", "256x128")}


class DecodeMxf8Projection:
    """One op's decode launcher: ``__call__(x [rows, K] bf16, w, out [rows, N] bf16)`` with
    ``rows <= 64`` and ``w`` a ``QuantizedWeight`` of the ``[N, K]`` weight.  The activation
    scratch is owned here (allocated once, so CUDA-graph capture allocates nothing) and is
    reused by every layer: each launch pair is ordered after the previous one by the chain's
    PDL waits.  ``hits`` counts launches."""

    rows_min = 1
    rows_max = ROWS_MAX

    def __init__(
        self,
        op: str,
        device_index: int = 0,
        config: _tpl.Mxf8ProjectionConfig | None = None,
    ):
        if op not in CONFIGS:
            raise ValueError(f"unknown decode mxf8 projection {op!r}; known: {tuple(CONFIGS)}")
        self.op = op
        self.config = config or CONFIGS[op]
        self.n, self.k = self.config.n, self.config.k
        self.device_index = device_index
        dev = torch.device("cuda", device_index)
        self._quant = _quant.compile_quant(device_index, passes=QUANT_PASSES, pdl=True)
        if self._quant.rows_per_cta != ROWS_MAX:
            raise ValueError("decode mxf8: the quant must cover 64 rows per CTA")
        self._gemm = _tpl.compile_projection(self.config)
        self.xq = torch.zeros(ROWS_MAX, self.k, dtype=torch.uint8, device=dev)
        # Scale rows 64..127 of the padded atom are never written by a 64-row quant tile:
        # code 127 (scale 1) once, as the quant kernel writes for rows past R.
        self.sfa = torch.full(
            (_quant.sf_bytes(ROWS_MAX, self.k),), 127, dtype=torch.uint8, device=dev
        )
        self.hits = 0
        # launches whose activation a producer wrote (``gemm_only``, decode act-quant fusion)
        self.pre_hits = 0

    def check(self, x: torch.Tensor, w: QuantizedWeight, out: torch.Tensor) -> None:
        rows = int(x.shape[0])
        if not 1 <= rows <= ROWS_MAX:
            raise ValueError(f"{self.op} decode mxf8: rows {rows} not in [1, {ROWS_MAX}]")
        if (
            x.dtype is not torch.bfloat16
            or x.shape[1] != self.k
            or x.stride(1) != 1
            or x.stride(0) % 8
            or x.data_ptr() % 16
        ):
            raise ValueError(
                f"{self.op} decode mxf8: x must be bf16 [rows, {self.k}], pitch % 8, 16B aligned"
            )
        if (
            out.dtype is not torch.bfloat16
            or tuple(out.shape) != (rows, self.n)
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"{self.op} decode mxf8: out must be contiguous bf16 [{rows}, {self.n}]"
            )
        if tuple(w.q.shape) != (self.n, self.k) or w.q.data_ptr() % 128 or w.sf.data_ptr() % 128:
            raise ValueError(
                f"{self.op} decode mxf8: weight must be e4m3 [{self.n}, {self.k}], 128B aligned"
            )

    def __call__(self, x: torch.Tensor, w: QuantizedWeight, out: torch.Tensor) -> torch.Tensor:
        rows = int(x.shape[0])
        self._quant.fn(
            x,
            self.xq[:rows],
            self.sfa,
            rows,
            self.k // 128,
            x.stride(0),
            1,
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        self._gemm(self.xq, w.q, self.sfa, w.sf, out, rows)
        self.hits += 1
        return out

    def gemm_only(self, w: QuantizedWeight, out: torch.Tensor, rows: int) -> torch.Tensor:
        """The GEMM alone on the activation a producer already wrote into ``xq`` / ``sfa``
        (rows ``< rows``; decode act-quant fusion).  The caller guarantees the producer is this launch's
        stream predecessor and wrote exactly those rows for this call."""
        if not 1 <= rows <= ROWS_MAX:
            raise ValueError(f"{self.op} decode mxf8: rows {rows} not in [1, {ROWS_MAX}]")
        if (
            out.dtype is not torch.bfloat16
            or tuple(out.shape) != (rows, self.n)
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"{self.op} decode mxf8: out must be contiguous bf16 [{rows}, {self.n}]"
            )
        self._gemm(self.xq, w.q, self.sfa, w.sf, out, rows)
        self.hits += 1
        self.pre_hits += 1
        return out


class WideDecodeMxf8Projection:
    """One op's launcher for the MTP3 / MTP5 decode rows (``WIDE_ROWS_MIN..WIDE_ROWS_MAX``):
    ``__call__(x [rows, K] bf16, w, out [rows, N] bf16)`` (the PDL quant into the operand
    scratch, then the GEMM) and ``gemm_only(w, out, rows)`` (the operand a producer wrote:
    the V-B GEMM's MXF8 side output). The ``[192, K]`` scratch ``xq`` / ``sfa`` is owned
    here (allocated once; a producer bakes its addresses in). Rows up to the template
    config's ``rows_max`` (``WIDE_CONFIGS``, 128) run its kernel on the scratch's first
    ``rows_max`` rows (rows past the call's are computed from stale bytes, never stored);
    rows above it the context MXF8 GEMM's PDL instance (``WIDE_CTX``) on the call's rows.
    Both are the context MXF8 GEMM's K order on the same operands: each row's output is
    that GEMM's, bitwise, whatever the instance or row count. ``upper`` (bench / tests):
    another template config holding 192 rows in place of the context instance.
    ``hits`` / ``pre_hits`` count launches."""

    rows_min = WIDE_ROWS_MIN
    rows_max = WIDE_ROWS_MAX

    def __init__(
        self,
        op: str,
        device_index: int = 0,
        config: _tpl.Mxf8ProjectionConfig | None = None,
        upper: _tpl.Mxf8ProjectionConfig | None = None,
    ):
        if op not in WIDE_CONFIGS:
            raise ValueError(
                f"unknown wide decode mxf8 projection {op!r}; known: {tuple(WIDE_CONFIGS)}"
            )
        self.op = op
        self.config = config or WIDE_CONFIGS[op]
        if not self.config.wide or self.config.rows_max >= self.rows_max:
            raise ValueError(
                f"{op} wide decode mxf8: the template holds 65..{self.rows_max - 1} rows"
            )
        if upper is not None and (
            upper.rows_max != self.rows_max or (upper.n, upper.k) != (self.config.n, self.config.k)
        ):
            raise ValueError(
                f"{op} wide decode mxf8: the upper template holds {self.rows_max} rows"
            )
        self.n, self.k = self.config.n, self.config.k
        self.device_index = device_index
        dev = torch.device("cuda", device_index)
        self._quant = _quant.compile_quant(device_index, passes=QUANT_PASSES, pdl=True)
        self._tpl = _tpl.compile_projection(self.config)
        self._upper_tpl = None if upper is None else _tpl.compile_projection(upper)
        self._ctx = None
        if upper is None:
            ctx_op, tile = WIDE_CTX[op]
            if (_ctx.ABIS[ctx_op].out_features, _ctx.ABIS[ctx_op].in_features) != (self.n, self.k):
                raise ValueError(f"{op} wide decode mxf8: {ctx_op} is not [{self.n}, {self.k}]")
            self._ctx = _ctx.compile_gemm(ctx_op, tile, device_index, pdl=True)
        self.xq = torch.zeros(self.rows_max, self.k, dtype=torch.uint8, device=dev)
        # Scale rows past the last quant tile (192..255 of the padded atoms) are never
        # written: code 127 (scale 1) once, as the quant kernel writes for rows past R.
        self.sfa = torch.full(
            (_quant.sf_bytes(self.rows_max, self.k),), 127, dtype=torch.uint8, device=dev
        )
        self.hits = 0
        self.pre_hits = 0

    def _gemm(self, w: QuantizedWeight, out: torch.Tensor, rows: int) -> None:
        r = self.config.rows_max
        if rows <= r:
            # The template compiled for the first r rows of the scratch reads exactly those.
            self._tpl(self.xq[:r], w.q, self.sfa, w.sf, out, rows)
        elif self._upper_tpl is not None:
            self._upper_tpl(self.xq, w.q, self.sfa, w.sf, out, rows)
        else:
            self._ctx(
                self.xq[:rows],
                w.q,
                self.sfa,
                w.sf,
                out,
                rows,
                self.n,
                self.k,
                torch.cuda.current_stream(out.device).cuda_stream,
            )

    def _check_rows(self, rows: int) -> None:
        if not self.rows_min <= rows <= self.rows_max:
            raise ValueError(
                f"{self.op} wide decode mxf8: rows {rows} not in [{self.rows_min}, {self.rows_max}]"
            )

    def _check_out(self, out: torch.Tensor, rows: int) -> None:
        if (
            out.dtype is not torch.bfloat16
            or tuple(out.shape) != (rows, self.n)
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"{self.op} wide decode mxf8: out must be contiguous bf16 [{rows}, {self.n}]"
            )

    def check(self, x: torch.Tensor, w: QuantizedWeight, out: torch.Tensor) -> None:
        rows = int(x.shape[0])
        self._check_rows(rows)
        if (
            x.dtype is not torch.bfloat16
            or x.shape[1] != self.k
            or x.stride(1) != 1
            or x.stride(0) % 8
            or x.data_ptr() % 16
        ):
            raise ValueError(
                f"{self.op} wide decode mxf8: x must be bf16 [rows, {self.k}], pitch % 8, 16B aligned"
            )
        self._check_out(out, rows)
        if tuple(w.q.shape) != (self.n, self.k) or w.q.data_ptr() % 128 or w.sf.data_ptr() % 128:
            raise ValueError(
                f"{self.op} wide decode mxf8: weight must be e4m3 [{self.n}, {self.k}], 128B aligned"
            )

    def __call__(self, x: torch.Tensor, w: QuantizedWeight, out: torch.Tensor) -> torch.Tensor:
        rows = int(x.shape[0])
        self._check_rows(rows)
        self._quant.fn(
            x,
            self.xq[:rows],
            self.sfa,
            rows,
            self.k // 128,
            x.stride(0),
            -(-rows // self._quant.rows_per_cta),
            torch.cuda.current_stream(x.device).cuda_stream,
        )
        self._gemm(w, out, rows)
        self.hits += 1
        return out

    def gemm_only(self, w: QuantizedWeight, out: torch.Tensor, rows: int) -> torch.Tensor:
        """The GEMM alone on the activation a producer already wrote into ``xq`` / ``sfa``
        (rows ``< rows``). The caller guarantees the producer is this launch's stream
        predecessor and wrote exactly those rows for this call."""
        self._check_rows(rows)
        self._check_out(out, rows)
        self._gemm(w, out, rows)
        self.hits += 1
        self.pre_hits += 1
        return out


def build(op: str, device_index: int = 0) -> DecodeMxf8Projection:
    return DecodeMxf8Projection(op, device_index)


def build_wide(op: str, device_index: int = 0) -> WideDecodeMxf8Projection:
    return WideDecodeMxf8Projection(op, device_index)


def _emulate(
    xq: torch.Tensor, sfa: torch.Tensor, w: QuantizedWeight, rows: int, n: int, k: int
) -> torch.Tensor:
    a = _quant.dequantize(xq[:rows], sfa, rows, k)
    b = _quant.dequantize(w.q, w.sf, n, k)
    return a @ b.t()
