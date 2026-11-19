# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 DSA indexer's FP8 prologue in one launch (CuTe DSL).

For the ``T`` tokens of a step, from the indexer query ``q`` (bf16 ``[T, H, 128]``, the wq_b
output), the normalized index key ``k`` (bf16 ``[T, 128]``) and the head weights ``w`` (bf16
``[T, H]``, a column view of the fused wk | weights_proj output), one launch writes:

1. ``q_fp8`` (E4M3 ``[T * H, 128]``) and ``q_scale`` (fp32 ``[T * H]``): every 128-wide row
   quantized with one scale, ``fused_cat_fp8``'s arithmetic: ``amax = max(max |x|, 1e-12)``,
   ``s = amax * fp32(1 / 448)`` (rounded up to a power of two with UE8M0),
   ``out = cvt.rn.satfinite.e4m3(x * rcp.approx.ftz(s))``;
2. ``k_fp8`` (E4M3 ``[T, 128]``) and ``k_scale`` (fp32 ``[T]``): the same for each key row;
3. ``weights`` (fp32 ``[T, H]``): ``(float(w) * q_scale) * weight_scale``, two round-to-nearest
   fp32 multiplies (no contraction, no flush to zero), as torch evaluates
   ``w * q_scale * weight_scale``;
4. the paged index-key cache (uint8 ``[pages, P, 1, 132]``, tokens contiguous within a page, any
   page stride): key ``t``'s 128 bytes at the flat byte offset ``slot_mapping_fp8[t]`` and its 4
   scale bytes at ``slot_mapping_scale[t]``, a flat offset ``f`` addressing byte
   ``(f // (P * 132)) * page_stride + f % (P * 132)``; a token with a negative key or scale slot
   is not written (``indexer_k_cache_scatter_op``'s rule);
5. optionally the MLA ``latent_cache = cat([compressed_kv, k_pe], -1)`` (bf16 ``[T, C + R]``), a
   copy.

Each value is computed with the instructions of the kernels it replaces, so every output byte
equals theirs.

Work split: a row of 128 values is read by 16 lanes (one 16-byte load each) and its absolute
maximum reduced with four butterfly shuffles inside the half warp. A CTA of ``R`` rows (``16 * R``
threads) holds either ``R`` key rows (the first ``ceil(T / R)`` CTAs) or ``R`` query rows of one
token (``T * H / R`` CTAs); with the latent copy, the last CTAs copy one 16-byte chunk of it per
thread. The kernel launches with PDL: every CTA waits on ``griddepcontrol.wait`` before its first
global read and then releases the dependent grid.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: Values per quantized row (the indexer head dim).
HEAD_DIM = 128
#: Bytes of one token's index-key slot: 128 E4M3 bytes; its fp32 scale sits after the page's keys.
BYTES_PER_TOKEN = HEAD_DIM + 4
#: Lanes per row: 16 bytes (8 bf16 values) each.
_LANES_PER_ROW = 16
#: ``fused_cat_fp8``'s constants as PTX fp32 literals: 1e-12 (amax floor) and fp32(1 / 448).
_MIN_AMAX = "0f2B8CBCCC"
_INV_E4M3_MAX = "0f3B124925"


# ---------------------------------------------------------------------------
# Inline PTX
# ---------------------------------------------------------------------------


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
def _ld_bf16_as_f32(addr, *, loc=None, ip=None):
    """The bf16 at ``addr`` widened to fp32 (exact)."""
    return Float32(
        _asm(
            T.f32(),
            [_lv(addr, loc=loc, ip=ip)],
            "{\n.reg .b32 r;\nld.global.u16 r, [$1];\nshl.b32 r, r, 16;\nmov.b32 $0, r;\n}",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_relaxed_s64(addr, *, loc=None, ip=None):
    """A strong (relaxed, GPU scope) load: ptxas keeps it where it is issued instead of sinking
    it into the branch that uses it, so it overlaps the row load."""
    return Int64(
        _asm(
            T.i64(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.s64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_v2_b32(addr, lo, hi, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(lo, loc=loc, ip=ip), _iv(hi, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


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


@dsl_user_op
def _st_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _f32_bits(value, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_fv(value, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _weight(w, scale, weight_scale, *, loc=None, ip=None):
    """``(w * scale) * weight_scale``: two IEEE fp32 multiplies, round to nearest, no flush."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(w, loc=loc, ip=ip), _fv(scale, loc=loc, ip=ip), _fv(weight_scale, loc=loc, ip=ip)],
            "{\n.reg .f32 x;\nmul.rn.f32 x, $1, $2;\nmul.rn.f32 $0, x, $3;\n}",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


def _quant_row_ptx() -> str:
    """PTX of :func:`_quant_row`: UE8M0 scales (each rounded up to a power of two)."""
    lines = [
        ".reg .b32 r, e, m;",
        ".reg .f32 f<8>, a, t, sc, inv;",
        ".reg .b16 h<4>;",
        ".reg .pred p;",
    ]
    for i in range(4):
        # Element 2i is the low half of input word i (little endian), 2i + 1 the high half.
        lines += [
            f"shl.b32 r, ${3 + i}, 16;",
            f"mov.b32 f{2 * i}, r;",
            f"and.b32 r, ${3 + i}, 0xFFFF0000;",
            f"mov.b32 f{2 * i + 1}, r;",
        ]
    lines.append("abs.f32 a, f0;")
    for i in range(1, 8):
        lines += [f"abs.f32 t, f{i};", "max.f32 a, a, t;"]
    for offset in (8, 4, 2, 1):
        # XOR offsets below 16 stay inside the row's half warp.
        lines += [f"shfl.sync.bfly.b32 t, a, {offset}, 0x1f, 0xffffffff;", "max.f32 a, a, t;"]
    lines += [f"max.f32 a, a, {_MIN_AMAX};", f"mul.rn.f32 sc, a, {_INV_E4M3_MAX};"]
    # Round up to a power of two: keep the exponent, add one if the mantissa is non-zero.
    lines += [
        "mov.b32 r, sc;",
        "and.b32 e, r, 0x7F800000;",
        "and.b32 m, r, 0x007FFFFF;",
        "setp.ne.u32 p, m, 0;",
        "@p add.u32 e, e, 0x00800000;",
        "mov.b32 sc, e;",
    ]
    lines.append("rcp.approx.ftz.f32 inv, sc;")
    for i in range(8):
        lines.append(f"mul.rn.f32 f{i}, f{i}, inv;")
    for j in range(4):
        # Upper byte = first source: element 2j + 1 above element 2j.
        lines.append(f"cvt.rn.satfinite.e4m3x2.f32 h{j}, f{2 * j + 1}, f{2 * j};")
    lines += ["mov.b32 $0, sc;", "mov.b32 $1, {h0, h1};", "mov.b32 $2, {h2, h3};"]
    return "{\n" + "\n".join(lines) + "\n}"


_QUANT_ROW_PTX = _quant_row_ptx()


@dsl_user_op
def _quant_row(words, *, loc=None, ip=None):
    """``(scale, lo, hi)`` of the 8 bf16 values in ``words`` (four 32-bit words) of a row held by
    16 consecutive lanes: the row's scale and this lane's 8 E4M3 bytes. All 32 lanes of the warp
    must execute it (the amax reduction shuffles)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.i32(), T.i32()]),
        [_iv(w, loc=loc, ip=ip) for w in words],
        _QUANT_ROW_PTX,
        "=f,=r,=r,r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class DsaFp8DecodePrologueKernel:
    """See the module docstring.

    Args:
        n_heads: H, query rows per token.
        tokens_per_block: P, tokens per index-key cache page.
        rows_per_cta: R, rows per CTA (divides ``n_heads``).
        cache_vec_bytes: Bytes per cache store, 8 when the cache base and page stride are 8-byte
            aligned, else 4.
        latent_dims: ``(kv_lora_rank, rope_dim)`` to also write ``latent_cache = cat([
            compressed_kv, k_pe], -1)`` (bf16, both multiples of 8), or None.
    """

    def __init__(
        self,
        n_heads: int,
        tokens_per_block: int,
        rows_per_cta: int,
        cache_vec_bytes: int,
        latent_dims: tuple[int, int] | None = None,
    ):
        if n_heads % rows_per_cta:
            raise ValueError(f"rows_per_cta={rows_per_cta} must divide n_heads={n_heads}")
        if cache_vec_bytes not in (4, 8):
            raise ValueError(f"cache_vec_bytes must be 4 or 8, got {cache_vec_bytes}")
        if latent_dims is not None and (latent_dims[0] % 8 or latent_dims[1] % 8):
            raise ValueError(f"latent dims must be multiples of 8, got {latent_dims}")
        self.n_heads = n_heads
        self.page_bytes = tokens_per_block * BYTES_PER_TOKEN
        self.rows = rows_per_cta
        self.threads = rows_per_cta * _LANES_PER_ROW
        self.cache_vec_bytes = cache_vec_bytes
        self.with_latent = latent_dims is not None
        # 16-byte chunks of a latent row and of its compressed_kv part.
        self.latent_chunks = 0 if latent_dims is None else (latent_dims[0] + latent_dims[1]) // 8
        self.ckv_chunks = 0 if latent_dims is None else latent_dims[0] // 8

    @cute.jit
    def __call__(
        self,
        q: Int64,
        q_row_stride: Int64,
        k: Int64,
        k_row_stride: Int64,
        w: Int64,
        w_row_stride: Int64,
        q_fp8: Int64,
        q_scale: Int64,
        weights: cute.Tensor,
        k_fp8: Int64,
        k_scale: Int64,
        cache: Int64,
        page_stride: Int64,
        slot_fp8: Int64,
        slot_scale: Int64,
        ckv: Int64,
        ckv_row_stride: Int64,
        k_pe: Int64,
        k_pe_row_stride: Int64,
        latent: Int64,
        num_tokens: Int32,
        weight_scale: Float32,
        stream,
    ):
        k_ctas = (num_tokens + (self.rows - 1)) // self.rows
        q_ctas = num_tokens * (self.n_heads // self.rows)
        l_ctas = (num_tokens * self.latent_chunks + (self.threads - 1)) // self.threads
        self.kernel(
            q,
            q_row_stride,
            k,
            k_row_stride,
            w,
            w_row_stride,
            q_fp8,
            q_scale,
            weights,
            k_fp8,
            k_scale,
            cache,
            page_stride,
            slot_fp8,
            slot_scale,
            ckv,
            ckv_row_stride,
            k_pe,
            k_pe_row_stride,
            latent,
            num_tokens,
            weight_scale,
        ).launch(
            grid=(k_ctas + q_ctas + l_ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        q: Int64,
        q_row_stride: Int64,
        k: Int64,
        k_row_stride: Int64,
        w: Int64,
        w_row_stride: Int64,
        q_fp8: Int64,
        q_scale: Int64,
        weights: cute.Tensor,
        k_fp8: Int64,
        k_scale: Int64,
        cache: Int64,
        page_stride: Int64,
        slot_fp8: Int64,
        slot_scale: Int64,
        ckv: Int64,
        ckv_row_stride: Int64,
        k_pe: Int64,
        k_pe_row_stride: Int64,
        latent: Int64,
        num_tokens: Int32,
        weight_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        row = tidx // _LANES_PER_ROW
        lane = tidx - row * _LANES_PER_ROW
        col_bytes = Int64(lane * 16)
        out_col = Int64(lane * 8)
        k_ctas = (num_tokens + (self.rows - 1)) // self.rows
        kq_ctas = k_ctas + num_tokens * (self.n_heads // self.rows)
        weights_out = weights.iterator.toint()

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        if bid < k_ctas:
            token = bid * self.rows + row
            # Rows past the step still quantize (the shuffles need every lane) a valid row, and
            # store nothing.
            src_token = token
            if token >= num_tokens:
                src_token = num_tokens - 1
            t = Int64(src_token)
            words = _ld_v4_b32(k + t * k_row_stride * 2 + col_bytes)
            fp8_slot = _ld_relaxed_s64(slot_fp8 + t * 8)
            scale_slot = _ld_relaxed_s64(slot_scale + t * 8)
            scale, lo, hi = _quant_row(words)
            if token < num_tokens:
                _st_v2_b32(k_fp8 + t * HEAD_DIM + out_col, lo, hi)
                if lane == 0:
                    _st_f32(k_scale + t * 4, scale)
                if fp8_slot >= Int64(0):
                    if scale_slot >= Int64(0):
                        page = fp8_slot // self.page_bytes
                        dst = cache + page * page_stride + (fp8_slot - page * self.page_bytes)
                        dst = dst + out_col
                        if cutlass.const_expr(self.cache_vec_bytes == 8):
                            _st_v2_b32(dst, lo, hi)
                        else:
                            _st_b32(dst, lo)
                            _st_b32(dst + 4, hi)
                        if lane == 0:
                            scale_page = scale_slot // self.page_bytes
                            scale_dst = (
                                cache
                                + scale_page * page_stride
                                + (scale_slot - scale_page * self.page_bytes)
                            )
                            _st_b32(scale_dst, _f32_bits(scale))
        elif bid < kq_ctas:
            q_row = (bid - k_ctas) * self.rows + row
            token = q_row // self.n_heads
            head = q_row - token * self.n_heads
            t = Int64(token)
            words = _ld_v4_b32(q + (t * q_row_stride + Int64(head * HEAD_DIM)) * 2 + col_bytes)
            w_value = _ld_bf16_as_f32(w + (t * w_row_stride + Int64(head)) * 2)
            scale, lo, hi = _quant_row(words)
            r = Int64(q_row)
            _st_v2_b32(q_fp8 + r * HEAD_DIM + out_col, lo, hi)
            if lane == 0:
                _st_f32(q_scale + r * 4, scale)
                _st_f32(weights_out + r * 4, _weight(w_value, scale, weight_scale))
        else:
            if cutlass.const_expr(self.with_latent):
                # One 16-byte chunk of cat([compressed_kv, k_pe]) per thread.
                chunk = (bid - kq_ctas) * self.threads + tidx
                if chunk < num_tokens * self.latent_chunks:
                    token = chunk // self.latent_chunks
                    j = chunk - token * self.latent_chunks
                    t = Int64(token)
                    src = ckv + t * ckv_row_stride * 2 + Int64(j * 16)
                    if j >= self.ckv_chunks:
                        src = k_pe + t * k_pe_row_stride * 2 + Int64((j - self.ckv_chunks) * 16)
                    v = _ld_v4_b32(src)
                    _st_v4_b32(latent + (t * self.latent_chunks + Int64(j)) * 16, v)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------

_COMPILED: dict = {}


def _compiled(
    n_heads: int,
    tokens_per_block: int,
    rows_per_cta: int,
    cache_vec_bytes: int,
    latent_dims: tuple[int, int] | None,
):
    key = (n_heads, tokens_per_block, rows_per_cta, cache_vec_bytes, latent_dims)
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "dsa_fp8_decode_prologue compiles on its first call for a configuration; make "
                "that call before CUDA graph capture"
            )
        kernel = DsaFp8DecodePrologueKernel(*key)
        a = Int64(1 << 20)
        weights_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), n_heads), stride_order=(1, 0)
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            kernel,
            *([a] * 8),
            weights_fake,
            *([a] * 11),
            Int32(1),
            Float32(1.0),
            stream,
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    return compiled


def supports_cache_layout(k_cache: torch.Tensor) -> bool:
    """Whether ``k_cache`` is a uint8 ``[pages, P, 1, 132]`` view with contiguous tokens within a
    page (any page stride, 4-byte aligned)."""
    return (
        k_cache.dim() == 4
        and k_cache.dtype == torch.uint8
        and k_cache.shape[2] == 1
        and k_cache.shape[3] == BYTES_PER_TOKEN
        and k_cache.stride(3) == 1
        and k_cache.stride(1) == BYTES_PER_TOKEN
        and k_cache.stride(0) % 4 == 0
        and k_cache.data_ptr() % 4 == 0
    )


def _rows_16b(t: torch.Tensor) -> bool:
    """bf16 rows with a unit inner stride, each 16-byte aligned."""
    return (
        t.dtype == torch.bfloat16
        and t.stride(-1) == 1
        and t.stride(0) % 8 == 0
        and t.data_ptr() % 16 == 0
    )


def dsa_fp8_decode_prologue_into(
    q: torch.Tensor,
    k: torch.Tensor,
    weights_bf16: torch.Tensor,
    weight_scale: float,
    num_tokens: int,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping_fp8: torch.Tensor,
    slot_mapping_scale: torch.Tensor,
    compressed_kv: torch.Tensor | None = None,
    k_pe: torch.Tensor | None = None,
    latent_cache: torch.Tensor | None = None,
    rows_per_cta: int = 16,
) -> None:
    """Launch the prologue on the current stream for the first ``num_tokens`` rows.

    Inputs: ``q`` bf16 ``[>= T, H, 128]`` (heads contiguous, 16-byte aligned rows), ``k`` bf16
    ``[>= T, 128]`` (unit inner stride, 16-byte aligned rows), ``weights_bf16`` bf16 ``[>= T, H]``
    (unit inner stride), ``k_cache`` as :func:`supports_cache_layout`, int64 slot mappings
    ``[>= T]``. Outputs, written in full: ``q_fp8`` E4M3 ``[T * H, 128]``, ``q_scale`` fp32
    ``[T * H, 1]``, ``weights`` fp32 ``[T, H]``, ``k_fp8`` E4M3 ``[T, 128]``, ``k_scale`` fp32
    ``[T, 1]``, all contiguous; the scales are UE8M0 (powers of two). With ``latent_cache``
    (bf16 ``[T, C + R]``, contiguous) also ``latent_cache = cat([compressed_kv[:T], k_pe[:T]],
    -1)`` from bf16 ``[>= T, C]`` and ``[>= T, R]`` rows (16-byte aligned, ``C`` and ``R``
    multiples of 8). Compiles once per configuration on the first call, which must not be under
    CUDA graph capture; later calls only launch.
    """
    if num_tokens == 0:
        return
    n_heads = q.shape[1]
    if not (
        q.dim() == 3
        and q.shape[2] == HEAD_DIM
        and q.stride(1) == HEAD_DIM
        and _rows_16b(q)
        and _rows_16b(k)
        and weights_bf16.dtype == torch.bfloat16
        and weights_bf16.stride(-1) == 1
    ):
        raise ValueError(
            "dsa_fp8_decode_prologue takes bf16 q [T, H, 128] with contiguous heads and k [T, 128] "
            "with 16-byte aligned rows, and bf16 weights with a unit inner stride"
        )
    if not supports_cache_layout(k_cache):
        raise ValueError(
            "dsa_fp8_decode_prologue takes a uint8 [pages, P, 1, 132] cache view with contiguous "
            "tokens within a page"
        )
    latent_dims = None
    ckv_ptr = ckv_stride = pe_ptr = pe_stride = latent_ptr = 0
    if latent_cache is not None:
        latent_dims = (compressed_kv.shape[-1], k_pe.shape[-1])
        if not (
            _rows_16b(compressed_kv)
            and _rows_16b(k_pe)
            and latent_dims[0] % 8 == 0
            and latent_dims[1] % 8 == 0
            and latent_cache.is_contiguous()
            and latent_cache.data_ptr() % 16 == 0
            and latent_cache.shape[-1] == latent_dims[0] + latent_dims[1]
        ):
            raise ValueError(
                "the latent_cache copy takes bf16 compressed_kv and k_pe rows (16-byte aligned, "
                "widths multiples of 8) and a contiguous latent_cache of their joint width"
            )
        ckv_ptr, ckv_stride = compressed_kv.data_ptr(), compressed_kv.stride(0)
        pe_ptr, pe_stride = k_pe.data_ptr(), k_pe.stride(0)
        latent_ptr = latent_cache.data_ptr()
    cache_vec_bytes = 8 if k_cache.stride(0) % 8 == 0 and k_cache.data_ptr() % 8 == 0 else 4
    compiled = _compiled(n_heads, k_cache.shape[1], rows_per_cta, cache_vec_bytes, latent_dims)
    compiled(
        q.data_ptr(),
        q.stride(0),
        k.data_ptr(),
        k.stride(0),
        weights_bf16.data_ptr(),
        weights_bf16.stride(0),
        q_fp8.data_ptr(),
        q_scale.data_ptr(),
        weights,
        k_fp8.data_ptr(),
        k_scale.data_ptr(),
        k_cache.data_ptr(),
        k_cache.stride(0),
        slot_mapping_fp8.data_ptr(),
        slot_mapping_scale.data_ptr(),
        ckv_ptr,
        ckv_stride,
        pe_ptr,
        pe_stride,
        latent_ptr,
        num_tokens,
        weight_scale,
    )


__all__ = [
    "BYTES_PER_TOKEN",
    "HEAD_DIM",
    "DsaFp8DecodePrologueKernel",
    "dsa_fp8_decode_prologue_into",
    "supports_cache_layout",
]
