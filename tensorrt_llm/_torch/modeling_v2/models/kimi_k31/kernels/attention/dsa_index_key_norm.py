# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 DSA indexer's index-key LayerNorm (CuTe DSL), with one fixed reduction order.

``y = LayerNorm(x) * weight + bias`` over each 128-wide bf16 row ``x`` (fp32 statistics, fp32
``weight`` / ``bias``, bf16 output), computed with exactly the instructions of the Triton kernel
``torch.compile`` builds for the indexer's ``LayerNorm`` module (a persistent reduction) at the
``XBLOCK=8, num_warps=2`` configuration, so every output bit equals that kernel's. A row is held by
16 lanes, lane ``c`` holding columns ``8c .. 8c + 7`` as fp32 ``x0 .. x7``:

1. ``s = ((x0 + x2) + (x4 + x6)) + ((x1 + x3) + (x5 + x7))``, then four butterfly adds over the
   half warp (lane offsets 8, 4, 2, 1): the row sum;
2. ``mean = sum * 2**-7`` (exact);
3. ``d = x - mean``; the same tree over ``d * d`` gives ``var_sum``;
4. ``rstd = rsqrt.approx.ftz(fma(var_sum, 2**-7, eps))``;
5. ``y = fma(weight, d * rstd, bias)``, rounded to bf16 (nearest even).

Every operation rounds to nearest (no contraction other than the two fmas above), so the result
does not depend on the launch configuration, the GPU or the run. ``index_key_norm_row`` is the
per-row device function, ``IndexKeyNormKernel`` the launch.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Values per row (the indexer head dim).
HEAD_DIM = 128
#: Lanes per row: 16 bytes (8 bf16 values) each.
_LANES_PER_ROW = 16
#: 1 / HEAD_DIM as a PTX fp32 literal (2**-7: the mean and variance scalings are exact).
_INV_DIM = "0f3C000000"


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _row_sum(src: str, dst: str) -> list:
    """PTX: ``dst`` = the row sum of the 8 fp32 registers ``<src>0 .. <src>7`` of each of the 16
    lanes of a row, in the reference kernel's order."""
    return [
        f"add.rn.f32 a0, {src}0, {src}2;",
        f"add.rn.f32 a1, {src}1, {src}3;",
        f"add.rn.f32 a2, {src}4, {src}6;",
        f"add.rn.f32 a3, {src}5, {src}7;",
        "add.rn.f32 a0, a0, a2;",
        "add.rn.f32 a1, a1, a3;",
        f"add.rn.f32 {dst}, a0, a1;",
    ] + [
        # XOR offsets below 16 stay inside the row's half warp.
        line
        for offset in (8, 4, 2, 1)
        for line in (
            f"shfl.sync.bfly.b32 t, {dst}, {offset}, 0x1f, 0xffffffff;",
            f"add.rn.f32 {dst}, {dst}, t;",
        )
    ]


def _norm_row_ptx() -> str:
    lines = [
        ".reg .b32 r;",
        ".reg .f32 x<8>, d<8>, w<8>, b<8>, a<4>, t, s, v, rs;",
    ]
    for i in range(4):
        # Element 2i is the low half of input word i (little endian), 2i + 1 the high half.
        lines += [
            f"shl.b32 r, ${4 + i}, 16;",
            f"mov.b32 x{2 * i}, r;",
            f"and.b32 r, ${4 + i}, 0xFFFF0000;",
            f"mov.b32 x{2 * i + 1}, r;",
        ]
    lines += [
        "ld.global.nc.v4.f32 {w0, w1, w2, w3}, [$8];",
        "ld.global.nc.v4.f32 {w4, w5, w6, w7}, [$8+16];",
        "ld.global.nc.v4.f32 {b0, b1, b2, b3}, [$9];",
        "ld.global.nc.v4.f32 {b4, b5, b6, b7}, [$9+16];",
    ]
    lines += _row_sum("x", "s")
    lines.append(f"mul.rn.f32 s, s, {_INV_DIM};")
    lines += [f"sub.rn.f32 d{i}, x{i}, s;" for i in range(8)]
    lines += [f"mul.rn.f32 x{i}, d{i}, d{i};" for i in range(8)]
    lines += _row_sum("x", "v")
    lines += [f"fma.rn.f32 v, v, {_INV_DIM}, $10;", "rsqrt.approx.ftz.f32 rs, v;"]
    lines += [f"mul.rn.f32 d{i}, d{i}, rs;" for i in range(8)]
    lines += [f"fma.rn.f32 d{i}, w{i}, d{i}, b{i};" for i in range(8)]
    # Upper half = first source: element 2j + 1 above element 2j.
    lines += [f"cvt.rn.bf16x2.f32 ${j}, d{2 * j + 1}, d{2 * j};" for j in range(4)]
    return "{\n" + "\n".join(lines) + "\n}"


_NORM_ROW_PTX = _norm_row_ptx()


@dsl_user_op
def index_key_norm_row(words, weight_addr, bias_addr, eps, *, loc=None, ip=None):
    """The normalized row as four bf16x2 words, from ``words`` (this lane's 8 bf16 values,
    columns ``8 * lane ..``, of a row held by 16 consecutive lanes) and this lane's 8 fp32 weights
    and biases at ``weight_addr`` / ``bias_addr`` (16-byte aligned). All 32 lanes of the warp
    must execute it (the row sums shuffle)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(w, loc=loc, ip=ip) for w in words]
        + [
            _lv(weight_addr, loc=loc, ip=ip),
            _lv(bias_addr, loc=loc, ip=ip),
            _fv(eps, loc=loc, ip=ip),
        ],
        _NORM_ROW_PTX,
        "=r,=r,=r,=r,r,r,r,r,l,l,f",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _ld_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_v4_b32(addr, words, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in words],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


class IndexKeyNormKernel:
    """``out[t] = index-key LayerNorm(x[t])`` for ``t < num_tokens`` (see the module docstring).

    ``x`` bf16 rows of 128 (unit inner stride, 16-byte aligned rows, any row stride), ``out``
    bf16 ``[>= T, 128]`` contiguous, ``weight`` / ``bias`` fp32 ``[128]`` (16-byte aligned). A CTA
    holds ``rows_per_cta`` rows (16 lanes each). A plain launch (no PDL).
    """

    def __init__(self, rows_per_cta: int):
        self.rows = rows_per_cta
        self.threads = rows_per_cta * _LANES_PER_ROW

    @cute.jit
    def __call__(
        self,
        x: Int64,
        x_row_stride: Int64,
        weight: Int64,
        bias: Int64,
        out: cute.Tensor,
        num_tokens: Int32,
        eps: Float32,
        stream,
    ):
        self.kernel(x, x_row_stride, weight, bias, out, num_tokens, eps).launch(
            grid=((num_tokens + (self.rows - 1)) // self.rows, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        x_row_stride: Int64,
        weight: Int64,
        bias: Int64,
        out: cute.Tensor,
        num_tokens: Int32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        row = tidx // _LANES_PER_ROW
        lane = tidx - row * _LANES_PER_ROW
        out_ptr = out.iterator.toint()
        token = bid * self.rows + row
        # Rows past the step still normalize (the shuffles need every lane) a valid row, and store
        # nothing.
        src_token = token
        if token >= num_tokens:
            src_token = num_tokens - 1
        t = Int64(src_token)
        words = _ld_v4_b32(x + t * x_row_stride * 2 + Int64(lane * 16))
        normed = index_key_norm_row(words, weight + Int64(lane * 32), bias + Int64(lane * 32), eps)
        if token < num_tokens:
            _st_v4_b32(out_ptr + t * (HEAD_DIM * 2) + Int64(lane * 16), normed)


_COMPILED: dict = {}


def _compiled(rows_per_cta: int):
    compiled = _COMPILED.get(rows_per_cta)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "index_key_norm compiles on its first call for a configuration; make that call "
                "before CUDA graph capture"
            )
        a = Int64(1 << 20)
        out_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (cute.sym_int(), HEAD_DIM), stride_order=(1, 0)
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            IndexKeyNormKernel(rows_per_cta),
            *([a] * 4),
            out_fake,
            Int32(1),
            Float32(1e-6),
            stream,
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[rows_per_cta] = compiled
    return compiled


def rows_supported(x: torch.Tensor) -> bool:
    """bf16 rows of 128 with a unit inner stride, each 16-byte aligned."""
    return (
        x.dim() == 2
        and x.shape[-1] == HEAD_DIM
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
    )


def params_supported(weight: torch.Tensor, bias: torch.Tensor) -> bool:
    """fp32 ``[128]`` weight and bias, contiguous and 16-byte aligned."""
    return all(
        p.dtype == torch.float32
        and p.numel() == HEAD_DIM
        and p.is_contiguous()
        and p.data_ptr() % 16 == 0
        for p in (weight, bias)
    )


def index_key_norm_into(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    num_tokens: int,
    rows_per_cta: int = 16,
) -> None:
    """Launch on the current stream: ``out[:num_tokens]`` (bf16 ``[>= T, 128]``, contiguous) from
    ``x[:num_tokens]`` (see :func:`rows_supported`), ``weight`` / ``bias`` (see
    :func:`params_supported`). Compiles once per configuration on the first call, which must not
    be under CUDA graph capture."""
    if num_tokens == 0:
        return
    if not (rows_supported(x) and params_supported(weight, bias)):
        raise ValueError(
            "index_key_norm takes bf16 rows of 128 (unit inner stride, 16-byte aligned rows) and "
            "fp32 [128] weight and bias (contiguous, 16-byte aligned)"
        )
    if not (out.is_contiguous() and out.dtype == torch.bfloat16 and out.data_ptr() % 16 == 0):
        raise ValueError("index_key_norm writes a contiguous, 16-byte aligned bf16 output")
    _compiled(rows_per_cta)(
        x.data_ptr(),
        x.stride(0),
        weight.data_ptr(),
        bias.data_ptr(),
        out,
        num_tokens,
        eps,
    )


__all__ = [
    "HEAD_DIM",
    "IndexKeyNormKernel",
    "index_key_norm_into",
    "index_key_norm_row",
    "params_supported",
    "rows_supported",
]
