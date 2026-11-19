# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 block verification and its bonus draw in one launch (CuTe DSL).

The kernel computes what ``ops/spec/dflash2_verify.py``'s ``dflash2_block_verify`` computes in
torch, one request at a time, with the reference's float32 operations in the reference's order
wherever that order is deterministic:

* the prefix ratios ``r_i`` (``prefix * p / q``, clamped at 1), the proposal mass of each draft
  and the sparse overlap ``sum_c min(r_i p_i(c), q_i(c))``: the candidate sums in the tree torch's
  reduction kernel uses for a row of ``k <= 32`` values (``block_width = last_pow2(k)`` lanes,
  ``x_t + x_{t + width}`` per lane, then a shuffle-down tree with halving offsets);
* ``h_i = R_i / (R_i + 1 - r_i)`` and ``tau = max {i : eta_i < h_i}``;
* the bonus row ``max(r_tau p_tau - q_tau, 0)`` (``p_g`` when every draft is accepted), its
  cumulative sum and the inverse-CDF search: the cumulative sum is torch's own scan of a
  ``[n, V]`` tensor (``scan_innermost_dim``: chunks of ``2 * num_threads_x`` columns, a Sklansky
  scan inside each chunk with the running total added to its first element), so every cumulative
  value, the total and the drawn index are bitwise the reference's; rows without mass take the
  first maximum of ``p_tau`` (NaN first), as ``torch.argmax``.

* ``sum_v p_i(v)`` of the interior rows (in ``R_i``): torch's reduction kernel splits it over a
  configuration that depends on the batch and the device (``setReduceConfig``: vectorized loads,
  ``block_width x block_height`` threads, ``ctas_per_output`` CTAs from the SM count); the kernel
  derives the same configuration (:func:`torch_row_sum_plan`) and its CTAs are torch's CTAs: the
  same four running sums per thread, the same x / y reductions, the same final combination, so
  ``h_i`` and ``tau`` are bitwise the reference's. This holds whenever every torch CTA reads whole
  1024-column scan chunks (``V`` a multiple of 2048, the batch's scan chunk 1024 columns): every
  Kimi K3.1 shape. Other shapes (small test vocabularies) sum each row in the kernel's own fixed
  order, where ``h_i`` can differ by float32 ulps.

Grid: one CTA per (request ``b``, target row ``i``, slice of the row): torch's CTAs of the row
sum (512 threads, ``ctas_per_output`` slices, each reading 2048-column blocks strided by the slice
count), or for other shapes 256 threads over :func:`slice_plan`'s contiguous chunk ranges. A CTA
reads its columns once, coalesced: it adds up ``p_i`` and, for the case ``tau = i``, writes the
bonus weights' per-chunk scan terms (each chunk's first element and the pairwise sums of its
blocks ``[2^j, 2^(j+1))``, which are all the Sklansky scan of a chunk needs besides the incoming
total). The last CTA of a request to finish (a per-request arrival counter) takes ``tau``, runs
the chunk totals of row ``tau`` in order, finds the chunk holding the draw and rescans that one
chunk; only a row without mass is read again, for its first maximum. That CTA resets the
counter, so the workspace is reused by every launch (CUDA-graph safe).
"""

from __future__ import annotations

import functools
from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

THREADS = 256
WARPS = THREADS // 32
#: Threads of a CTA that reproduces torch's reduction kernel (its MAX_NUM_THREADS for float).
TORCH_THREADS = 512
#: Largest draft count (block width - 1) and candidate count a launch serves.
MAX_DRAFTS = 15
MAX_CANDIDATES = 32
#: Shared-memory cap of the bonus row's per-chunk scan terms (floats).
MAX_SPINE_FLOATS = 8192
#: Most CTAs one target row is split over.
MAX_SLICES = 32


def _last_pow2(x: int) -> int:
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


def torch_scan_log_threads_x(num_rows: int, row_size: int) -> int:
    """``log2(num_threads_x)`` of torch's innermost-dimension scan of a ``[num_rows, row_size]``
    tensor (``get_log_num_threads_x_inner_scan``); a chunk is ``2 << result`` columns."""
    lx = 0
    while (1 << lx) < row_size:
        lx += 1
    ly = 0
    while (1 << ly) < num_rows:
        ly += 1
    log_x = int((9 + lx - ly) / 2)  # C++ integer division truncates toward zero
    return min(max(log_x, 4), 9)


# ---------------------------------------------------------------------------
# Inline PTX helpers (explicit .rn float operations: never contracted into FMAs)
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, side_effects=True, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=side_effects,
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


def _f32op(text, a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            text,
            "=f,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return _f32op("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _sub(a, b, *, loc=None, ip=None):
    return _f32op("sub.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    return _f32op("mul.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _div(a, b, *, loc=None, ip=None):
    return _f32op("div.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _min_nan(a, b, *, loc=None, ip=None):
    """``torch.minimum`` / ``clamp(max=)``: NaN if either operand is NaN."""
    return _f32op("min.NaN.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _clamp0(a, *, loc=None, ip=None):
    """``clamp(min=0)``: negatives become 0, NaN stays NaN."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip)],
            "max.NaN.f32 $0, $1, 0f00000000;",
            "=f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


def _fcmp(op, a, b, *, loc=None, ip=None):
    """1 where ``a op b`` (ordered: 0 when either is NaN), else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp." + op + ".f32 p, $1, $2;\nselp.b32 $0, 1, 0, p;\n}",
            "=r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lt(a, b, *, loc=None, ip=None):
    return _fcmp("lt", a, b, loc=loc, ip=ip)


@dsl_user_op
def _gt(a, b, *, loc=None, ip=None):
    return _fcmp("gt", a, b, loc=loc, ip=ip)


@dsl_user_op
def _le(a, b, *, loc=None, ip=None):
    return _fcmp("le", a, b, loc=loc, ip=ip)


@dsl_user_op
def _self(cond, a, b, *, loc=None, ip=None):
    """``a`` where ``cond != 0``, else ``b`` (float32)."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(cond, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}",
            "=f,r,f,f",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _seli(cond, a, b, *, loc=None, ip=None):
    """``a`` where ``cond != 0``, else ``b`` (int32)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(cond, loc=loc, ip=ip), _iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.b32 $0, $2, $3, p;\n}",
            "=r,r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ilt(a, b, *, loc=None, ip=None):
    """1 if ``a < b`` (signed int32), else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.lt.s32 p, $1, $2;\nselp.b32 $0, 1, 0, p;\n}",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clamp_index(v, hi, *, loc=None, ip=None):
    """``min(max(v, 0), hi)`` of an int64, as int32."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(v, loc=loc, ip=ip), _lv(hi, loc=loc, ip=ip)],
            "{\n.reg .s64 t;\nmax.s64 t, $1, 0;\nmin.s64 t, t, $2;\ncvt.u32.u64 $0, t;\n}",
            "=r,l,l",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _argmax_wins(va, ia, vb, ib, *, loc=None, ip=None):
    """1 if ``(va, ia)`` beats ``(vb, ib)`` in ``torch.argmax`` (NaN above all, then the larger
    value, ties to the lower index), else 0."""
    return Int32(
        _asm(
            T.i32(),
            [
                _fv(va, loc=loc, ip=ip),
                _iv(ia, loc=loc, ip=ip),
                _fv(vb, loc=loc, ip=ip),
                _iv(ib, loc=loc, ip=ip),
            ],
            "{\n.reg .pred na, nb, il, eq, gt, t, w, x;\n"
            "setp.nan.f32 na, $1, $1;\nsetp.nan.f32 nb, $3, $3;\n"
            "setp.lt.s32 il, $2, $4;\nsetp.eq.f32 eq, $1, $3;\nsetp.gt.f32 gt, $1, $3;\n"
            "and.pred t, eq, il;\nor.pred t, t, gt;\nnot.pred nb, nb;\n"
            "and.pred w, nb, t;\nor.pred x, nb, il;\n@na mov.pred w, x;\n"
            "selp.b32 $0, 1, 0, w;\n}",
            "=r,f,r,f,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_down_f32(v, offset, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(v, loc=loc, ip=ip), _iv(offset, loc=loc, ip=ip)],
            "shfl.sync.down.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ieq(a, b, *, loc=None, ip=None):
    """1 if ``a == b`` (int32), else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.eq.s32 p, $1, $2;\nselp.b32 $0, 1, 0, p;\n}",
            "=r,r,r",
            False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx_f32(v, src, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(v, loc=loc, ip=ip), _iv(src, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx_i32(v, src, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip), _iv(src, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_xor_f32(v, mask, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(v, loc=loc, ip=ip), _iv(mask, loc=loc, ip=ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_xor_i32(v, mask, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(v, loc=loc, ip=ip), _iv(mask, loc=loc, ip=ip)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_f32_pred(addr, ok, *, loc=None, ip=None):
    """The float at ``addr`` if ``ok != 0``, else 0 (no access)."""
    return Float32(
        _asm(
            T.f32(),
            [_lv(addr, loc=loc, ip=ip), _iv(ok, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\nmov.f32 $0, 0f00000000;\n"
            "@p ld.global.nc.f32 $0, [$1];\n}",
            "=f,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_v4_f32_pred(addr, ok, *, loc=None, ip=None):
    """The 4 floats at ``addr`` (16-byte aligned) if ``ok != 0``, else zeros (no access)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip), _iv(ok, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $5, 0;\nmov.f32 $0, 0f00000000;\n"
        "mov.f32 $1, 0f00000000;\nmov.f32 $2, 0f00000000;\nmov.f32 $3, 0f00000000;\n"
        "@p ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];\n}",
        "=f,=f,=f,=f,l,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), res, [j], loc=loc, ip=ip)) for j in range(4))


@dsl_user_op
def _ld_nc_i32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_i64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.b64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_cg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_cg_i32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.cg.b32 $0, [$1];",
            "=r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(v, loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_i32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_i64(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _lv(v, loc=loc, ip=ip)],
        "st.global.b64 [$0], $1;",
        "l,l,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_add_global(addr, v, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fence_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ld_s_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_s_i32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "ld.shared.b32 $0, [$1];",
            "=r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_s_f32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(v, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_s_i32(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_add_shared(addr, v, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _iv(v, loc=loc, ip=ip)],
        "red.shared.add.u32 [$0], $1;",
        "r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _splitmix_uniform(seed, offset, step, *, loc=None, ip=None):
    """``dflash2_uniforms``' draw ``step - 1`` of the row keyed by ``(seed, offset)``: the top 24
    bits of ``mix(mix(mix(seed) ^ offset) + step * gamma)`` times ``2**-24`` (SplitMix64, 64-bit
    wrapping arithmetic, logical shifts)."""
    mix = (
        "shr.b64 t, z, 30;\nxor.b64 z, z, t;\nmul.lo.u64 z, z, 0xBF58476D1CE4E5B9;\n"
        "shr.b64 t, z, 27;\nxor.b64 z, z, t;\nmul.lo.u64 z, z, 0x94D049BB133111EB;\n"
        "shr.b64 t, z, 31;\nxor.b64 z, z, t;\n"
    )
    return Float32(
        _asm(
            T.f32(),
            [_lv(seed, loc=loc, ip=ip), _lv(offset, loc=loc, ip=ip), _lv(step, loc=loc, ip=ip)],
            "{\n.reg .b64 z, t;\n.reg .f32 f;\nmov.b64 z, $1;\n"
            + mix
            + "xor.b64 z, z, $2;\n"
            + mix
            + "mul.lo.u64 t, $3, 0x9E3779B97F4A7C15;\nadd.u64 z, z, t;\n"
            + mix
            + "shr.b64 z, z, 40;\ncvt.rn.f32.u64 f, z;\nmul.rn.f32 $0, f, 0f33800000;\n}",
            "=f,l,l,l",
            False,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class DFlash2BlockVerifyKernel:
    """``(tau, bonus)`` of every request; see the module docstring.

    Args:
        drafts: g, the drafted tokens per request (block width - 1).
        top_k: k, the candidates per slot (``<= 32``).
        vocab: V, the target vocabulary.
        log_threads_x: torch's scan geometry for the batch (:func:`torch_scan_log_threads_x`).
        draft_i64, cand_i64: integer widths of the drafted tokens and candidate ids.

    The acceptance and bonus uniforms are derived from per-request ``(seed, offset)`` as
    ``dflash2_uniforms`` does.
    """

    def __init__(
        self,
        drafts,
        top_k,
        vocab,
        log_threads_x,
        draft_i64,
        cand_i64,
        torch_xy=None,
    ):
        if not 1 <= drafts <= MAX_DRAFTS:
            raise ValueError(f"1..{MAX_DRAFTS} drafted tokens, got {drafts}")
        if not 1 <= top_k <= MAX_CANDIDATES:
            raise ValueError(f"1..{MAX_CANDIDATES} candidates per slot, got {top_k}")
        self.g = drafts
        self.k = top_k
        self.vocab = vocab
        self.log_x = log_threads_x
        self.chunk = 2 << log_threads_x  # C columns per scan chunk
        self.log_chunk = log_threads_x + 1  # log2(C)
        self.per_lane = self.chunk // 32  # E consecutive columns per lane
        self.log_lane = self.log_chunk - 5  # log2(E)
        self.nchunks = (vocab + self.chunk - 1) // self.chunk
        self.terms = self.log_chunk + 1  # scan terms per chunk: x0, U_0 .. U_{log C - 1}
        if self.nchunks * self.terms > MAX_SPINE_FLOATS:
            raise ValueError(
                f"vocab {vocab} in {self.chunk}-column chunks needs "
                f"{self.nchunks * self.terms} scan terms; at most {MAX_SPINE_FLOATS}"
            )
        # Chunks of >= 128 columns over a vocabulary of whole 16-byte vectors are read
        # coalesced: lane l holds columns [128 m + 4 l, 128 m + 4 l + 4) of each 128-column
        # segment m. Smaller chunks: lane l holds columns [E l, E l + E), scalar loads.
        self.coalesced = vocab % 4 == 0 and self.chunk >= 128
        self.segs = self.chunk // 128
        self.draft_i64 = draft_i64
        self.cand_i64 = cand_i64
        # torch_xy = (block_width, block_height) of torch's reduction of the interior rows'
        # vocabulary sums: the CTA then is torch's CTA (512 threads, one per torch thread) and
        # sums in torch's order; None keeps the kernel's own order (256 threads, slices).
        self.torch_order = torch_xy is not None
        if self.torch_order:
            self.tbw, self.tbh = torch_xy
            if self.tbw * self.tbh != TORCH_THREADS or self.tbw < 32 or self.chunk != 1024:
                raise ValueError(f"unsupported torch reduction geometry {torch_xy}")
        self.threads = TORCH_THREADS if self.torch_order else THREADS
        # Workspace floats per chunk: the scan terms, or (torch order) x0, U_0 .. U_6 and the
        # eight 128-column segment totals, from which the finishing CTA forms U_7 .. U_9.
        self.part_terms = 16 if self.torch_order else self.terms
        self.warps = self.threads // 32
        # torch's reduction of a k-wide row: block_width lanes, then halving shuffle offsets.
        self.bw = min(_last_pow2(top_k), 32)

    # ----- small traced helpers ------------------------------------------------------------

    def _load_index(self, addr, wide: bool):
        if cutlass.const_expr(wide):
            v = _ld_nc_i64(addr)
        else:
            v = Int64(_ld_nc_i32(addr))
        return _clamp_index(v, Int64(self.vocab - 1))

    def _lane_sum(self, x, lane):
        """torch's sum of a k-wide row held one value per lane (lanes >= k hold 0); valid in
        lane 0."""
        v = _add(Float32(0.0), x)
        if cutlass.const_expr(self.bw < 32):
            second = _shfl_down_f32(v, self.bw)
            has_second = _ilt(lane + self.bw, Int32(self.k))
            v = _add(v, _self(has_second, second, Float32(0.0)))
        else:
            v = _add(v, Float32(0.0))
        v = _add(v, Float32(0.0))
        v = _add(v, Float32(0.0))
        off = self.bw // 2
        while off >= 1:
            v = _add(v, _shfl_down_f32(v, off))
            off //= 2
        return v

    @staticmethod
    def _pairwise_levels(values):
        """``[values, pairwise sums, sums of 4, ...]`` down to one value (traced statically)."""
        levels = [list(values)]
        while len(levels[-1]) > 1:
            prev = levels[-1]
            levels.append([_add(prev[2 * t], prev[2 * t + 1]) for t in range(len(prev) // 2)])
        return levels

    def _coin(self, seeds, ss_b, offsets, so_b, b64, s):
        """Uniform ``s`` of request ``b`` (``s == g`` is the bonus draw's)."""
        seed = _ld_nc_i64(seeds + b64 * ss_b * 8)
        offset = _ld_nc_i64(offsets + b64 * so_b * 8)
        return _splitmix_uniform(seed, offset, Int64(s + 1))

    def _sanitize_q(self, q):
        ok = _le(Float32(0.0), q) * _le(q, Float32(1.0))
        return _self(ok, q, Float32(0.0))

    def _load_chunk(self, buf, row_probs, ch, lane, ok):
        """The lane's ``E`` columns of chunk ``ch`` into ``buf`` (0 past the vocabulary, all 0
        when ``ok == 0``)."""
        V, C, E = self.vocab, self.chunk, self.per_lane
        if cutlass.const_expr(self.coalesced):
            for m in range(self.segs):
                col = ch * C + 128 * m + 4 * lane
                okm = ok * _ilt(col, Int32(V))
                vals = _ld_nc_v4_f32_pred(row_probs + Int64(col) * 4, okm)
                for c in range(4):
                    buf[4 * m + c] = vals[c]
        else:
            first = ch * C + lane * E
            for e in range(E):
                oke = ok * _ilt(first + e, Int32(V))
                buf[e] = _ld_nc_f32_pred(row_probs + Int64(first + e) * 4, oke)

    @cute.jit
    def _stream_torch_order(
        self,
        row_probs: Int64,
        row_part: Int64,
        scale: Float32,
        sl: Int32,
        vec_step: Int32,
        tid: Int32,
        warp: Int32,
        lane: Int32,
        has_s: Int32,
        cid_s: Int32,
        ovl_s: Int32,
        tred_s: Int32,
        slice_out: Int64,
    ):
        """One torch CTA of the row's vocabulary sum (``sl`` = its ``blockIdx.y``): thread ``t``
        is torch's thread ``(t % bw, t // bw)``, reading 16-byte vector ``512 sl + j vec_step +
        t`` at step ``j`` into four running sums, then torch's x and y reductions; the CTA value
        goes to ``slice_out``. Each step's 512 vectors are two whole scan chunks (warp ``w``:
        segment ``w % 8`` of chunk ``w // 8``): the chunk's first-segment warp writes x0 and
        U_0 .. U_6, every warp its segment total."""
        k, V, C = self.k, self.vocab, self.chunk
        nvec = V // 4
        a0 = Float32(0.0)
        a1 = Float32(0.0)
        a2 = Float32(0.0)
        a3 = Float32(0.0)
        first = sl * 512
        nx = _ld_nc_v4_f32_pred(row_probs + Int64((first + tid) * 4) * 4, _ilt(first, Int32(nvec)))
        n0, n1, n2, n3 = nx
        for base in cutlass.range(first, nvec, vec_step, unroll=1):
            col = (base + tid) * 4
            vals = (n0, n1, n2, n3)
            # The next step's vector travels while this one is reduced.
            nb = base + vec_step
            nx = _ld_nc_v4_f32_pred(row_probs + Int64((nb + tid) * 4) * 4, _ilt(nb, Int32(nvec)))
            n0, n1, n2, n3 = nx
            a0 = _add(a0, vals[0])
            a1 = _add(a1, vals[1])
            a2 = _add(a2, vals[2])
            a3 = _add(a3, vals[3])
            w0 = _mul(scale, vals[0])
            w1 = _mul(scale, vals[1])
            w2 = _mul(scale, vals[2])
            w3 = _mul(scale, vals[3])
            ch = (base * 4) // C + warp // 8
            if _ld_s_i32(has_s + 4 * ch) != 0:
                for j in cutlass.range_constexpr(k):
                    rel = _ld_s_i32(cid_s + 4 * j) - col
                    oj = _ld_s_f32(ovl_s + 4 * j)
                    w0 = _self(_ieq(rel, Int32(0)), _sub(w0, oj), w0)
                    w1 = _self(_ieq(rel, Int32(1)), _sub(w1, oj), w1)
                    w2 = _self(_ieq(rel, Int32(2)), _sub(w2, oj), w2)
                    w3 = _self(_ieq(rel, Int32(3)), _sub(w3, oj), w3)
            c0 = _clamp0(w0)
            c1 = _clamp0(w1)
            c2 = _clamp0(w2)
            c3 = _clamp0(w3)
            a23 = _add(c2, c3)
            v = _add(_add(c0, c1), a23)
            chunk_part = row_part + Int64(ch) * self.part_terms * 4
            first_seg = _ieq(warp % 8, Int32(0))
            if first_seg != 0:
                if lane == 0:
                    _st_f32(chunk_part, c0)
                    _st_f32(chunk_part + 4, c1)
                    _st_f32(chunk_part + 8, a23)
            for m in cutlass.range_constexpr(5):
                if first_seg != 0:
                    if lane == (1 << m):
                        _st_f32(chunk_part + 4 * (3 + m), v)
                v = _add(v, _shfl_down_f32(v, 1 << m))
            if lane == 0:
                _st_f32(chunk_part + 4 * (8 + warp % 8), v)
        cute.arch.barrier()
        # torch: combine the four running sums, block_x_reduce, block_y_reduce.
        value = _add(_add(_add(a0, a1), a2), a3)
        bw, bh = self.tbw, self.tbh
        x = tid % bw
        y = tid // bw
        x_offs = [bw >> (i + 1) for i in range(8) if (bw >> (i + 1)) >= 32]
        y_offs = [bh >> (i + 1) for i in range(8) if (bh >> (i + 1)) >= 1]
        if cutlass.const_expr(bw > 32):
            _st_s_f32(tred_s + 4 * tid, value)
            for i in cutlass.range_constexpr(len(x_offs)):
                cute.arch.barrier()
                if x < x_offs[i]:
                    value = _add(value, _ld_s_f32(tred_s + 4 * (tid + x_offs[i])))
                    _st_s_f32(tred_s + 4 * tid, value)
        cute.arch.barrier()
        for m in cutlass.range_constexpr(5):
            value = _add(value, _shfl_down_f32(value, 16 >> m))
        cute.arch.barrier()
        _st_s_f32(tred_s + 4 * tid, value)
        for i in cutlass.range_constexpr(len(y_offs)):
            cute.arch.barrier()
            if y < y_offs[i]:
                value = _add(value, _ld_s_f32(tred_s + 4 * (tid + y_offs[i] * bw)))
                _st_s_f32(tred_s + 4 * tid, value)
        if tid == 0:
            _st_f32(slice_out, value)

    # ----- launch ---------------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        probs: Int64,
        sp_b: Int64,
        sp_i: Int64,
        drafts: Int64,
        sd_b: Int64,
        sd_s: Int64,
        cands: Int64,
        sc_b: Int64,
        sc_s: Int64,
        sc_j: Int64,
        qs: Int64,
        sq_b: Int64,
        sq_s: Int64,
        sq_j: Int64,
        seeds: Int64,
        ss_b: Int64,
        offsets: Int64,
        so_b: Int64,
        tau_t: cute.Tensor,
        bonus_out: Int64,
        part: Int64,
        sl_f: Int64,
        ov_f: Int64,
        counters: Int64,
        num_reqs: Int32,
        slices: Int32,
        per_slice: Int32,
        stream,
    ):
        @cute.struct
        class SharedStorage:
            r_s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 16], 16]
            # Unused; keeps the shared-memory offsets of the fields below.
            acc_s: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 16], 16]
            cand_all: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 16 * 32], 16]
            q_all: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 16 * 32], 16]
            cid_s: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32], 16]
            ovl_s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 32], 16]
            red_f: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 16], 16]
            red_v: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 16], 16]
            red_i: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 16], 16]
            tred_s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, TORCH_THREADS], 16]
            misc_i: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 16], 16]
            misc_f: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 16], 16]
            has_s: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.nchunks], 16]
            spine_s: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, (self.nchunks + 1) * self.terms], 16
            ]
            ends_s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.nchunks], 16]
            buf_s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.chunk], 16]
            wbuf_s: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.chunk], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            probs,
            sp_b,
            sp_i,
            drafts,
            sd_b,
            sd_s,
            cands,
            sc_b,
            sc_s,
            sc_j,
            qs,
            sq_b,
            sq_s,
            sq_j,
            seeds,
            ss_b,
            offsets,
            so_b,
            tau_t,
            bonus_out,
            part,
            sl_f,
            ov_f,
            counters,
            slices,
            per_slice,
        ).launch(
            grid=(num_reqs, self.g + 1, slices),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    # ----- kernel ---------------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        probs: Int64,
        sp_b: Int64,
        sp_i: Int64,
        drafts: Int64,
        sd_b: Int64,
        sd_s: Int64,
        cands: Int64,
        sc_b: Int64,
        sc_s: Int64,
        sc_j: Int64,
        qs: Int64,
        sq_b: Int64,
        sq_s: Int64,
        sq_j: Int64,
        seeds: Int64,
        ss_b: Int64,
        offsets: Int64,
        so_b: Int64,
        tau_t: cute.Tensor,
        bonus_out: Int64,
        part: Int64,
        sl_f: Int64,
        ov_f: Int64,
        counters: Int64,
        slices: Int32,
        per_slice: Int32,
    ):
        g, k, V = self.g, self.k, self.vocab
        C, E, LE = self.chunk, self.per_lane, self.log_lane
        nchunks, terms = self.nchunks, self.terms
        cand_bytes = 8 if self.cand_i64 else 4
        draft_bytes = 8 if self.draft_i64 else 4

        tau_out = tau_t.iterator.toint()
        tid, _, _ = cute.arch.thread_idx()
        req, row, sl = cute.arch.block_idx()
        warp = tid // 32
        lane = tid % 32
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        r_s = storage.r_s.data_ptr().toint()
        cand_all = storage.cand_all.data_ptr().toint()
        q_all = storage.q_all.data_ptr().toint()
        cid_s = storage.cid_s.data_ptr().toint()
        ovl_s = storage.ovl_s.data_ptr().toint()
        red_f = storage.red_f.data_ptr().toint()
        red_v = storage.red_v.data_ptr().toint()
        red_i = storage.red_i.data_ptr().toint()
        misc_i = storage.misc_i.data_ptr().toint()
        misc_f = storage.misc_f.data_ptr().toint()
        has_s = storage.has_s.data_ptr().toint()
        spine_s = storage.spine_s.data_ptr().toint()
        ends_s = storage.ends_s.data_ptr().toint()
        buf_s = storage.buf_s.data_ptr().toint()
        wbuf_s = storage.wbuf_s.data_ptr().toint()
        tred_s = storage.tred_s.data_ptr().toint()
        NT, NW = self.threads, self.warps

        b64 = Int64(req)
        req_probs = probs + b64 * sp_b * 4
        draft_row = drafts + b64 * sd_b * draft_bytes
        cand_row = cands + b64 * sc_b * cand_bytes
        q_row = qs + b64 * sq_b * 4
        rows_total = Int64(g + 1)
        row_probs = req_probs + Int64(row) * sp_i * 4
        slot = row
        if slot > g - 1:
            slot = Int32(g - 1)

        # This warp's first chunk of the slice is in flight during the prologue.
        c_begin = sl * per_slice + warp
        c_end = sl * per_slice + per_slice
        if c_end > nchunks:
            c_end = Int32(nchunks)
        if cutlass.const_expr(self.torch_order):
            c_end = c_begin
        x = cute.make_rmem_tensor((E,), Float32)
        w = cute.make_rmem_tensor((E,), Float32)
        if cutlass.const_expr(not self.torch_order):
            self._load_chunk(x, row_probs, c_begin, lane, _ilt(c_begin, c_end))

        for c0 in cutlass.range(tid, nchunks, NT, unroll=1):
            _st_s_i32(has_s + 4 * c0, Int32(0))
        cute.arch.barrier()

        # ---- 1. warp 0: proposal mass and target probability of every draft, the prefix
        # ratios, and this row's candidate overlaps (the bonus weights for tau = row) ---------
        if warp == 0:
            in_k = _ilt(lane, Int32(k))
            lane_k = _seli(in_k, lane, Int32(k - 1))
            lane_g = _seli(_ilt(lane, Int32(g)), lane, Int32(g - 1))
            # Lane s holds draft s; every lane < k holds candidate `lane` of every slot.
            d_lane = self._load_index(
                draft_row + Int64(lane_g) * sd_s * draft_bytes, self.draft_i64
            )
            cs = []
            qvs = []
            for s in cutlass.range_constexpr(g):
                c = self._load_index(
                    cand_row + (Int64(s) * sc_s + Int64(lane_k) * sc_j) * cand_bytes,
                    self.cand_i64,
                )
                qv = self._sanitize_q(
                    _ld_nc_f32(q_row + (Int64(s) * sq_s + Int64(lane_k) * sq_j) * 4)
                )
                _st_s_i32(cand_all + 4 * (32 * s + lane), c)
                _st_s_f32(q_all + 4 * (32 * s + lane), qv)
                cs.append(c)
                qvs.append(qv)
            pd_lane = _ld_nc_f32(req_probs + (Int64(lane_g) * sp_i + Int64(d_lane)) * 4)
            c_row = _ld_s_i32(cand_all + 4 * (32 * slot + lane))
            q_row_v = _ld_s_f32(q_all + 4 * (32 * slot + lane))
            p_row_c = _ld_nc_f32(row_probs + Int64(c_row) * 4)
            # Every lane runs the (tiny) chain, so every lane holds the ratios.
            prefix = Float32(1.0)
            scale = Float32(1.0)
            for s in cutlass.range_constexpr(g):
                d_s = _shfl_idx_i32(d_lane, s)
                mass = _self(in_k * _ieq(cs[s], d_s), qvs[s], Float32(0.0))
                qd = self._lane_sum(mass, lane)
                qd = _shfl_idx_f32(qd, 0)
                pd = _shfl_idx_f32(pd_lane, s)
                q_ok = _gt(qd, Float32(0.0)) * _le(qd, Float32(1.0))
                ratio = _min_nan(_div(_mul(prefix, pd), qd), Float32(1.0))
                prefix = _self(q_ok, ratio, Float32(0.0))
                if cutlass.const_expr(s + 1 <= g - 1):
                    scale = _self(_ieq(row, Int32(s + 1)), prefix, scale)
                if lane == 0:
                    _st_s_f32(r_s + 4 * (s + 1), prefix)
            o = _min_nan(_mul(scale, p_row_c), q_row_v)
            o = _self(in_k * (Int32(1) - _ieq(row, Int32(g))), o, Float32(0.0))
            if lane < k:
                _st_s_i32(cid_s + 4 * lane, c_row)
                _st_s_f32(ovl_s + 4 * lane, o)
                _st_s_i32(has_s + 4 * (c_row // C), Int32(1))
            ovl_sum = self._lane_sum(o, lane)
            if lane == 0:
                _st_s_f32(r_s, Float32(1.0))
                _st_s_f32(misc_f + 12, scale)
                if sl == 0:
                    _st_f32(ov_f + (b64 * rows_total + Int64(row)) * 4, ovl_sum)
        cute.arch.barrier()

        # ---- 2. stream this slice of the row: sum, per-chunk scan terms --------------------
        scale = _ld_s_f32(misc_f + 12)
        row_part = part + ((b64 * rows_total + Int64(row)) * nchunks) * self.part_terms * 4
        s0 = Float32(0.0)
        s1 = Float32(0.0)
        s2 = Float32(0.0)
        s3 = Float32(0.0)
        if cutlass.const_expr(self.torch_order):
            self._stream_torch_order(
                row_probs,
                row_part,
                scale,
                sl,
                per_slice,
                tid,
                warp,
                lane,
                has_s,
                cid_s,
                ovl_s,
                tred_s,
                sl_f + ((b64 * rows_total + Int64(row)) * MAX_SLICES + Int64(sl)) * 4,
            )
        for ch in cutlass.range(c_begin, c_end, WARPS, unroll=1):
            sums = [s0, s1, s2, s3]
            for e in cutlass.range_constexpr(E):
                # Padding columns read as 0: the sum is unchanged.
                sums[e % 4] = _add(sums[e % 4], x[e])
                w[e] = _mul(scale, x[e])
            s0, s1, s2, s3 = sums
            # The next chunk travels while this one is patched, summed and stored.
            nxt = ch + WARPS
            self._load_chunk(x, row_probs, nxt, lane, _ilt(nxt, c_end))
            chunk_part = row_part + Int64(ch) * terms * 4
            if cutlass.const_expr(self.coalesced):
                if _ld_s_i32(has_s + 4 * ch) != 0:
                    for j in cutlass.range_constexpr(k):
                        rel = _ld_s_i32(cid_s + 4 * j) - ch * C
                        if rel >= 0:
                            if rel < C:
                                if lane == ((rel >> 2) & 31):
                                    slot_e = ((rel >> 7) << 2) | (rel & 3)
                                    oj = _ld_s_f32(ovl_s + 4 * j)
                                    for e in cutlass.range_constexpr(E):
                                        if slot_e == e:
                                            w[e] = _sub(w[e], oj)
                cw = [_clamp0(w[e]) for e in range(E)]
                # Blocks of 4 in the lane, of 8..128 across lanes (shuffle tree per segment),
                # then of 256.. across the segments in lane 0: the same pairwise sums.
                seg_tot = []
                for m in cutlass.range_constexpr(self.segs):
                    a23 = _add(cw[4 * m + 2], cw[4 * m + 3])
                    v = _add(_add(cw[4 * m], cw[4 * m + 1]), a23)
                    if cutlass.const_expr(m == 0):
                        if lane == 0:
                            _st_f32(chunk_part, cw[0])
                            _st_f32(chunk_part + 4, cw[1])
                            _st_f32(chunk_part + 8, a23)
                    for j in cutlass.range_constexpr(5):
                        if cutlass.const_expr(m == 0):
                            if lane == (1 << j):
                                _st_f32(chunk_part + 4 * (3 + j), v)
                        v = _add(v, _shfl_down_f32(v, 1 << j))
                    seg_tot.append(v)
                if lane == 0:
                    seg_levels = self._pairwise_levels(seg_tot)
                    for lv in cutlass.range_constexpr(len(seg_levels) - 1):
                        _st_f32(chunk_part + 4 * (8 + lv), seg_levels[lv][1])
            else:
                first = ch * C + lane * E
                if _ld_s_i32(has_s + 4 * ch) != 0:
                    for j in cutlass.range_constexpr(k):
                        rel = _ld_s_i32(cid_s + 4 * j) - first
                        if rel >= 0:
                            if rel < E:
                                oj = _ld_s_f32(ovl_s + 4 * j)
                                for e in cutlass.range_constexpr(E):
                                    if rel == e:
                                        w[e] = _sub(w[e], oj)
                levels = self._pairwise_levels([_clamp0(w[e]) for e in range(E)])
                if lane == 0:
                    _st_f32(chunk_part, levels[0][0])
                    for lv in cutlass.range_constexpr(LE):
                        _st_f32(chunk_part + 4 * (1 + lv), levels[lv][1])
                v = levels[LE][0]
                for m in cutlass.range_constexpr(5):
                    if lane == (1 << m):
                        _st_f32(chunk_part + 4 * (1 + LE + m), v)
                    v = _add(v, _shfl_down_f32(v, 1 << m))

        if cutlass.const_expr(not self.torch_order):
            # Slice sum in the kernel's own fixed order.
            total_p = _add(_add(s0, s1), _add(s2, s3))
            for m in cutlass.range_constexpr(5):
                total_p = _add(total_p, _shfl_xor_f32(total_p, 1 << m))
            if lane == 0:
                _st_s_f32(red_f + 4 * warp, total_p)
            cute.arch.barrier()
            if tid == 0:
                row_sum = _ld_s_f32(red_f)
                for wi in cutlass.range_constexpr(1, NW):
                    row_sum = _add(row_sum, _ld_s_f32(red_f + 4 * wi))
                _st_f32(
                    sl_f + ((b64 * rows_total + Int64(row)) * MAX_SLICES + Int64(sl)) * 4, row_sum
                )

        # ---- 3. arrival: the last CTA of the request finishes it. The CTA barrier orders
        # every thread's stores before thread 0's release fence and counter update; the last
        # CTA's acquire fence comes before the barrier that releases its readers. -------------
        cute.arch.barrier()
        if tid == 0:
            _fence_gpu()
            old = _atom_add_global(counters + b64 * 4, Int32(1))
            last = Int32(0)
            if old == (g + 1) * slices - 1:
                last = Int32(1)
                _fence_gpu()
            _st_s_i32(misc_i, last)
        cute.arch.barrier()
        if _ld_s_i32(misc_i) != 0:
            # tau: every interior row's sum over its slices (lane p holds slice p), h, coins.
            if warp == 0:
                tau = Int32(0)
                in_sl = _ilt(lane, slices)
                lane_sl = _seli(in_sl, lane, Int32(0))
                parts = []
                ovs = []
                for s in cutlass.range_constexpr(1, g):
                    base = (b64 * rows_total + Int64(s)) * MAX_SLICES
                    parts.append(_ld_cg_f32(sl_f + (base + Int64(lane_sl)) * 4))
                    ovs.append(_ld_cg_f32(ov_f + (b64 * rows_total + Int64(s)) * 4))
                coins_l = []
                for s in cutlass.range_constexpr(1, g + 1):
                    coins_l.append(self._coin(seeds, ss_b, offsets, so_b, b64, s - 1))
                for s in cutlass.range_constexpr(1, g + 1):
                    r = _ld_s_f32(r_s + 4 * s)
                    h = r
                    if cutlass.const_expr(s < g):
                        row_sum = _self(in_sl, parts[s - 1], Float32(0.0))
                        if cutlass.const_expr(self.torch_order):
                            # torch's last CTA: thread p takes CTA p's value (0 + v), the
                            # other threads 0, then its x-reduction: shuffle-down tree.
                            row_sum = _add(Float32(0.0), row_sum)
                            for m in cutlass.range_constexpr(5):
                                row_sum = _add(row_sum, _shfl_down_f32(row_sum, 16 >> m))
                            row_sum = _shfl_idx_f32(row_sum, 0)
                        else:
                            for m in cutlass.range_constexpr(5):
                                row_sum = _add(row_sum, _shfl_xor_f32(row_sum, 1 << m))
                        # h = R / (R + 1 - r), R = max(r sum(p) - sum_c min(r p(c), q(c)), 0)
                        resid = _clamp0(_sub(_mul(r, row_sum), ovs[s - 1]))
                        interior = _gt(r, Float32(0.0)) * _lt(r, Float32(1.0))
                        h = _self(interior, _div(resid, _sub(_add(resid, Float32(1.0)), r)), r)
                    tau = _seli(_lt(coins_l[s - 1], h), Int32(s), tau)
                if lane == 0:
                    _st_s_i32(misc_i + 4, tau)
                    _st_s_i32(misc_i + 8, Int32(0))
                    _st_s_i32(misc_i + 12, Int32(0))
            cute.arch.barrier()
            bonus_row = _ld_s_i32(misc_i + 4)
            b_scale = Float32(1.0)
            if bonus_row >= 1:
                if bonus_row <= g - 1:
                    b_scale = _ld_s_f32(r_s + 4 * bonus_row)
            b_slot = bonus_row
            if b_slot > g - 1:
                b_slot = Int32(g - 1)
            b_probs = req_probs + Int64(bonus_row) * sp_i * 4
            if warp == 0:
                if lane < k:
                    c = _ld_s_i32(cand_all + 4 * (32 * b_slot + lane))
                    qv = _ld_s_f32(q_all + 4 * (32 * b_slot + lane))
                    o = _min_nan(_mul(b_scale, _ld_nc_f32(b_probs + Int64(c) * 4)), qv)
                    if bonus_row == g:
                        o = Float32(0.0)
                    _st_s_i32(cid_s + 4 * lane, c)
                    _st_s_f32(ovl_s + 4 * lane, o)
            b_part = part + ((b64 * rows_total + Int64(bonus_row)) * nchunks) * self.part_terms * 4
            if cutlass.const_expr(self.torch_order):
                # U_7 = segment 1, U_8 = segments 2 + 3, U_9 = (4 + 5) + (6 + 7).
                for c2 in cutlass.range(tid, nchunks, NT, unroll=1):
                    src = b_part + Int64(c2) * 64
                    dst = spine_s + 4 * terms * c2
                    vals = [_ld_cg_f32(src + 4 * t) for t in range(16)]
                    for t in cutlass.range_constexpr(8):
                        _st_s_f32(dst + 4 * t, vals[t])
                    _st_s_f32(dst + 32, vals[9])
                    _st_s_f32(dst + 36, _add(vals[10], vals[11]))
                    _st_s_f32(
                        dst + 40,
                        _add(_add(vals[12], vals[13]), _add(vals[14], vals[15])),
                    )
            else:
                for t0 in cutlass.range(tid, nchunks * terms, NT, unroll=1):
                    _st_s_f32(spine_s + 4 * t0, _ld_cg_f32(b_part + Int64(t0) * 4))
            cute.arch.barrier()

            # Chunk totals in order: each chunk's first element plus the incoming total, then
            # its blocks [2^j, 2^(j+1)) one by one (torch's Sklansky scan of the chunk). The
            # next chunk's terms are loaded while this chunk's chain runs.
            if tid == 0:
                cur = cute.make_rmem_tensor((terms,), Float32)
                nx = cute.make_rmem_tensor((terms,), Float32)
                for t in cutlass.range_constexpr(terms):
                    cur[t] = _ld_s_f32(spine_s + 4 * t)
                carry = Float32(0.0)
                for ch in cutlass.range(nchunks, unroll=1):
                    base_n = spine_s + 4 * terms * (ch + 1)
                    for t in cutlass.range_constexpr(terms):
                        nx[t] = _ld_s_f32(base_n + 4 * t)
                    acc = _add(cur[0], carry)
                    for t in cutlass.range_constexpr(1, terms):
                        acc = _add(cur[t], acc)
                    _st_s_f32(ends_s + 4 * ch, acc)
                    carry = acc
                    for t in cutlass.range_constexpr(terms):
                        cur[t] = nx[t]
                u = self._coin(seeds, ss_b, offsets, so_b, b64, g)
                _st_s_f32(misc_f + 4, carry)
                _st_s_f32(misc_f + 8, _mul(u, carry))
            cute.arch.barrier()
            target = _ld_s_f32(misc_f + 8)
            # Chunks whose last cumulative value is <= target (the cumulative sum is monotone).
            below = Int32(0)
            for c1 in cutlass.range(tid, nchunks, NT, unroll=1):
                if _gt(_ld_s_f32(ends_s + 4 * c1), target) == 0:
                    below = below + 1
            for m in cutlass.range_constexpr(5):
                below = below + _shfl_xor_i32(below, 1 << m)
            if lane == 0:
                _atom_add_shared(misc_i + 8, below)
            cute.arch.barrier()
            cstar = _ld_s_i32(misc_i + 8)
            index = Int32(V - 1)
            if cstar < nchunks:
                carry_in = Float32(0.0)
                if cstar > 0:
                    carry_in = _ld_s_f32(ends_s + 4 * (cstar - 1))
                for e0 in cutlass.range_constexpr((C + NT - 1) // NT):
                    e = tid + e0 * NT
                    if e < C:
                        col = cstar * C + e
                        wv = Float32(0.0)
                        if col < V:
                            wv = _mul(b_scale, _ld_nc_f32(b_probs + Int64(col) * 4))
                            for j in cutlass.range_constexpr(k):
                                if _ld_s_i32(cid_s + 4 * j) == col:
                                    wv = _sub(wv, _ld_s_f32(ovl_s + 4 * j))
                            wv = _clamp0(wv)
                        _st_s_f32(wbuf_s + 4 * e, wv)
                        if e == 0:
                            wv = _add(wv, carry_in)
                        _st_s_f32(buf_s + 4 * e, wv)
                cute.arch.barrier()
                for m in cutlass.range_constexpr(self.log_x + 1):
                    s = 1 << m
                    for p0 in cutlass.range_constexpr((C // 2 + NT - 1) // NT):
                        t = tid + p0 * NT
                        if t < C // 2:
                            a = ((t >> m) << (m + 1)) | s
                            ti = a + (t % s)
                            si = a - 1
                            _st_s_f32(
                                buf_s + 4 * ti,
                                _add(_ld_s_f32(buf_s + 4 * ti), _ld_s_f32(buf_s + 4 * si)),
                            )
                    cute.arch.barrier()
                live = V - cstar * C
                if live > C:
                    live = Int32(C)
                cnt = Int32(0)
                for e0 in cutlass.range_constexpr((C + NT - 1) // NT):
                    e = tid + e0 * NT
                    if e < live:
                        if _gt(_ld_s_f32(buf_s + 4 * e), target) == 0:
                            cnt = cnt + 1
                for m in cutlass.range_constexpr(5):
                    cnt = cnt + _shfl_xor_i32(cnt, 1 << m)
                if lane == 0:
                    _atom_add_shared(misc_i + 12, cnt)
                cute.arch.barrier()
                index = cstar * C + _ld_s_i32(misc_i + 12)
            if tid == 0:
                wv = Float32(0.0)
                if cstar < nchunks:
                    wv = _ld_s_f32(wbuf_s + 4 * (index - cstar * C))
                else:
                    # No chunk ends above the target: the reference's index V clamps to V - 1.
                    wv = _mul(b_scale, _ld_nc_f32(b_probs + Int64(index) * 4))
                    for j in cutlass.range_constexpr(k):
                        if _ld_s_i32(cid_s + 4 * j) == index:
                            wv = _sub(wv, _ld_s_f32(ovl_s + 4 * j))
                    wv = _clamp0(wv)
                total = _ld_s_f32(misc_f + 4)
                has_mass = _gt(total, Float32(0.0)) * _gt(wv, Float32(0.0))
                _st_s_i32(misc_i + 20, has_mass)
                _st_s_i32(misc_i + 24, index)
            cute.arch.barrier()
            bonus = _ld_s_i32(misc_i + 24)
            if _ld_s_i32(misc_i + 20) == 0:
                # No mass to draw from: the first maximum of p_tau (NaN first), torch.argmax.
                best_v = Float32(float("-inf"))
                best_i = Int32(2147483647)
                for col in cutlass.range(tid, V, NT, unroll=1):
                    pv = _ld_nc_f32(b_probs + Int64(col) * 4)
                    win = _argmax_wins(pv, col, best_v, best_i)
                    best_v = _self(win, pv, best_v)
                    best_i = _seli(win, col, best_i)
                for m in cutlass.range_constexpr(5):
                    ov = _shfl_xor_f32(best_v, 1 << m)
                    oi = _shfl_xor_i32(best_i, 1 << m)
                    win = _argmax_wins(ov, oi, best_v, best_i)
                    best_v = _self(win, ov, best_v)
                    best_i = _seli(win, oi, best_i)
                if lane == 0:
                    _st_s_f32(red_v + 4 * warp, best_v)
                    _st_s_i32(red_i + 4 * warp, best_i)
                cute.arch.barrier()
                if tid == 0:
                    bv = _ld_s_f32(red_v)
                    bi = _ld_s_i32(red_i)
                    for wi in cutlass.range_constexpr(1, NW):
                        ov = _ld_s_f32(red_v + 4 * wi)
                        oi = _ld_s_i32(red_i + 4 * wi)
                        win = _argmax_wins(ov, oi, bv, bi)
                        bv = _self(win, ov, bv)
                        bi = _seli(win, oi, bi)
                    _st_s_i32(misc_i + 24, bi)
                cute.arch.barrier()
                bonus = _ld_s_i32(misc_i + 24)
            if tid == 0:
                _st_i64(tau_out + b64 * 8, Int64(bonus_row))
                _st_i64(bonus_out + b64 * 8, Int64(bonus))
                _st_i32(counters + b64 * 4, Int32(0))


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _compiled(drafts, top_k, vocab, log_x, draft_i64, cand_i64, torch_xy):
    kernel = DFlash2BlockVerifyKernel(drafts, top_k, vocab, log_x, draft_i64, cand_i64, torch_xy)
    a = Int64(1 << 20)
    i = Int32(1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    tau_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),))
    return cute.compile(
        kernel,
        *([a] * 18),
        tau_fake,
        *([a] * 5),
        i,
        i,
        i,
        stream,
        options="--opt-level 2 --enable-tvm-ffi",
    )


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _max_threads_per_sm(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).max_threads_per_multi_processor


def torch_row_sum_plan(num_outputs: int, vocab: int, num_sms: int, max_threads_per_sm: int):
    """``(block_width, block_height, ctas_per_output)`` of torch's CUDA sum over the last dim of
    ``num_outputs`` contiguous float32 rows of ``vocab`` columns (``setReduceConfig``: vectorized
    input, block dimensions, the input split across warps and CTAs), when it is a configuration
    this kernel reproduces (every torch CTA reads whole 1024-column chunks); else None."""

    def cdiv(a, b):
        return -(-a // b)

    def last_pow2(x):
        return 1 << (x.bit_length() - 1)

    if vocab % 2048 or num_outputs < 1:
        return None
    max_threads = TORCH_THREADS
    dim0, dim1 = vocab // 4, num_outputs
    d0 = last_pow2(dim0) if dim0 < max_threads else max_threads
    d1 = last_pow2(dim1) if dim1 < max_threads else max_threads
    bw = min(d0, 32)
    bh = min(d1, max_threads // bw)
    bw = min(d0, max_threads // bh)
    step = bw
    if cdiv(vocab, step) < min(bh * 16, 256):
        return None  # torch would give each warp its own output
    step *= bh
    if bw * bh != max_threads or bw < 32:
        return None
    target = num_sms * (max_threads_per_sm // max_threads)
    cpo = 1
    if cdiv(vocab, step) >= 256 and num_outputs <= target:
        per_thread = cdiv(vocab, step)
        cpo = min(max(cdiv(target, num_outputs), cdiv(per_thread, 256)), cdiv(per_thread, 16))
    if not 1 <= cpo <= MAX_SLICES:
        return None
    return bw, bh, cpo


def slice_plan(num_reqs: int, drafts: int, nchunks: int, num_sms: int) -> Tuple[int, int]:
    """``(slices, chunks per slice)`` of each target row: enough CTAs for four per SM, every
    slice a multiple of the CTA's warps. A function of the shape only, so a shape's row sums
    (and the kernel's results) are the same on every call."""
    rows = num_reqs * (drafts + 1)
    want = max(1, -(-nchunks * rows // (4 * num_sms)))
    per_slice = min(nchunks, -(-want // WARPS) * WARPS)
    slices = -(-nchunks // per_slice)
    if slices > MAX_SLICES:
        slices = MAX_SLICES
        per_slice = -(-nchunks // slices)
    return slices, per_slice


#: Launch scratch per (device, power-of-two request capacity, drafts, chunks, terms), sliced to each call's
#: requests: a bounded set however the batch size varies. Entries are never replaced, so a captured graph keeps
#: a live buffer.
_WORKSPACES: Dict[Tuple, Tuple[torch.Tensor, ...]] = {}


def _workspace(device, num_reqs, drafts, nchunks, terms):
    capacity = 1 << (num_reqs - 1).bit_length()
    key = (device, capacity, drafts, nchunks, terms)
    ws = _WORKSPACES.get(key)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "dflash2_block_verify must run once eagerly for this shape before CUDA graph capture"
            )
        rows = capacity * (drafts + 1)
        ws = (
            torch.empty(rows * nchunks * terms, dtype=torch.float32, device=device),
            torch.empty(rows * MAX_SLICES, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
        )
        _WORKSPACES[key] = ws
    part, slice_sums, overflow, counters = ws
    rows = num_reqs * (drafts + 1)
    return (
        part[: rows * nchunks * terms],
        slice_sums[: rows * MAX_SLICES],
        overflow[:rows],
        counters[:num_reqs],
    )


def _launch(target_probs, draft_tokens, candidate_ids, q, seed, offset):
    n, rows, vocab = target_probs.shape
    g = rows - 1
    k = candidate_ids.shape[-1]
    device = target_probs.device
    tau = torch.empty(n, dtype=torch.int64, device=device)
    bonus = torch.empty(n, dtype=torch.int64, device=device)
    if n == 0:
        return tau, bonus
    log_x = torch_scan_log_threads_x(n, vocab)
    chunk = 2 << log_x
    nchunks = (vocab + chunk - 1) // chunk
    dev = device.index if device.index is not None else torch.cuda.current_device()
    plan = None
    if g >= 2 and chunk == 1024:
        plan = torch_row_sum_plan(n * (g - 1), vocab, _num_sms(dev), _max_threads_per_sm(dev))
    if plan is not None:
        torch_xy = plan[:2]
        slices, per_slice = plan[2], 512 * plan[2]
    else:
        torch_xy = None
        slices, per_slice = slice_plan(n, g, nchunks, _num_sms(dev))
    compiled = _compiled(
        g,
        k,
        vocab,
        log_x,
        draft_tokens.dtype == torch.int64,
        candidate_ids.dtype == torch.int64,
        torch_xy,
    )
    part, sl_f, ov_f, counters = _workspace(device, n, g, nchunks, 16)
    compiled(
        target_probs.data_ptr(),
        target_probs.stride(0),
        target_probs.stride(1),
        draft_tokens.data_ptr(),
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        candidate_ids.data_ptr(),
        candidate_ids.stride(0),
        candidate_ids.stride(1),
        candidate_ids.stride(2),
        q.data_ptr(),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        seed.data_ptr(),
        seed.stride(0),
        offset.data_ptr(),
        offset.stride(0),
        tau,
        bonus.data_ptr(),
        part.data_ptr(),
        sl_f.data_ptr(),
        ov_f.data_ptr(),
        counters.data_ptr(),
        n,
        slices,
        per_slice,
    )
    return tau, bonus


def dflash2_block_verify_seeded_cute(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
    q: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(tau [n] int64, bonus [n] int64)`` of ``dflash2_block_verify`` with the uniforms
    ``dflash2_uniforms(seed, offset, g + 1)`` (acceptance coins ``[:, :g]``, bonus ``[:, g]``)
    derived in the kernel.

    ``target_probs`` float32 ``[n, g + 1, V]`` with unit vocab stride and 16-byte aligned rows;
    ``draft_tokens`` int32/int64 ``[n, g]``, ``candidate_ids`` int32/int64 ``[n, g, k]``, ``q``
    float32 ``[n, g, k]``, ``seed`` / ``offset`` int64 ``[n]``, any strides. Compiles once per
    (g, k, V, scan geometry, dtypes) and allocates its workspace once per (n, g, V): the first
    call of a shape must be eager.
    """
    return _launch(target_probs, draft_tokens, candidate_ids, q, seed, offset)


__all__ = [
    "DFlash2BlockVerifyKernel",
    "dflash2_block_verify_seeded_cute",
    "slice_plan",
    "torch_row_sum_plan",
    "torch_scan_log_threads_x",
]
