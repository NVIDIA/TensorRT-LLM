# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM local (K3.1 MXFP8 projections): W8A16 weight operands.

An MXFP8 weight [N, K] (E4M3 codes, one UE8M0 exponent per 32 consecutive K) is kept
as its codes and scales only, in one byte buffer laid out K-block-major so that any
row range of one 64-wide K block is contiguous:

    codes  [K / 64][Np][64]  at byte 0
    scales [K / 64][Np][2]   at byte K * Np  (the two 32-groups of each 64-wide block)

``Np`` is N rounded up to ``ROW_PAD`` (padding rows are zero codes). This is a byte
permutation of the checkpoint codes and scales; nothing is rounded or re-encoded.

The W8A16 kernels convert each element to BF16 in registers before the BF16 MMA:
E4M3 -> F16 (exact) -> F32 (exact) times 2^(e - 127) (an exact power-of-two product in
F32, subnormals kept) -> BF16 (exact whenever the product is a BF16 value, which the
loader checks against the exact dequantization). The BF16 operand tiles are therefore
bitwise the tiles the BF16 kernels read from the exactly dequantized weight, and every
MMA, its K order and the split-K reduction are unchanged: outputs are bitwise equal.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from .primitives import asm

ROW_PAD = 128
# Bytes per 64-wide K block of one row: codes, scales.
CODE_BYTES, SCALE_BYTES = 64, 2


class W8Weight:
    """Host owner of one W8A16 weight buffer (see the module docstring)."""

    def __init__(self, buffer: torch.Tensor, n: int, k: int):
        if buffer.dtype != torch.uint8 or buffer.dim() != 1 or not buffer.is_contiguous():
            raise TypeError("W8A16 weight buffer must be contiguous uint8 bytes")
        if k % 64 or n <= 0:
            raise ValueError(f"W8A16 weight needs K % 64 == 0, got [{n}, {k}]")
        self.n, self.k = n, k
        self.n_pad = -(-n // ROW_PAD) * ROW_PAD
        if buffer.numel() != self.n_pad * k * 33 // 32:
            raise ValueError("W8A16 weight buffer size does not match its shape")
        if buffer.data_ptr() % 128:
            raise ValueError("W8A16 weight buffer must be 128-byte aligned")
        self.buffer = buffer

    @classmethod
    def from_checkpoint(cls, codes: torch.Tensor, scales: torch.Tensor) -> "W8Weight":
        """``codes`` E4M3 (or its bytes) [N, K], ``scales`` UE8M0 bytes [N, K / 32]."""
        if codes.dtype == torch.float8_e4m3fn:
            codes = codes.view(torch.uint8)
        n, k = codes.shape
        if scales.dtype != torch.uint8 or tuple(scales.shape) != (n, k // 32) or k % 64:
            raise ValueError("W8A16 needs UE8M0 bytes [N, K/32] and K % 64 == 0")
        n_pad = -(-n // ROW_PAD) * ROW_PAD
        blocks = k // 64
        buffer = torch.zeros(n_pad * k * 33 // 32, dtype=torch.uint8, device=codes.device)
        body = buffer[: n_pad * k].view(blocks, n_pad, 64)
        body[:, :n].copy_(codes.reshape(n, blocks, 64).transpose(0, 1))
        tail = buffer[n_pad * k :].view(blocks, n_pad, 2)
        tail[:, :n].copy_(scales.reshape(n, blocks, 2).transpose(0, 1))
        return cls(buffer, n, k)

    @property
    def shape(self):
        return torch.Size((self.n, self.k))

    @property
    def device(self):
        return self.buffer.device

    @property
    def address(self) -> int:
        return self.buffer.data_ptr()

    def codes(self) -> torch.Tensor:
        """The checkpoint-layout E4M3 bytes [N, K] (a copy; tests and audits)."""
        body = self.buffer[: self.n_pad * self.k].view(self.k // 64, self.n_pad, 64)
        return body[:, : self.n].transpose(0, 1).reshape(self.n, self.k).contiguous()

    def scales(self) -> torch.Tensor:
        """The checkpoint-layout UE8M0 bytes [N, K / 32] (a copy)."""
        tail = self.buffer[self.n_pad * self.k :].view(self.k // 64, self.n_pad, 2)
        return tail[:, : self.n].transpose(0, 1).reshape(self.n, self.k // 32).contiguous()


def is_w8(weight) -> bool:
    return isinstance(weight, W8Weight)


# ---------------------------------------------------------------------------- device
@asm(Float32, pure=True)
def scale_f32(byte):
    """2^(byte - 127) as FP32 (byte 0: the subnormal 2^-127)."""
    code = "{ .reg .b32 t, u; .reg .pred p; shl.b32 t, $1, 23; setp.eq.u32 p, $1, 0; "
    code += "selp.b32 u, 4194304, t, p; mov.b32 $0, u; }"
    return code, "=f,r", (Uint32(byte),)


@asm((Uint32,) * 4, pure=True)
def bf16x8(lo, hi, scale):
    """Eight E4M3 codes (``lo``: elements 0-3, ``hi``: 4-7, byte order) times
    ``scale`` as four BF16x2 words (element 2 j in the low half of word j)."""
    code = "{ .reg .b16 c<4>, l<8>; .reg .b32 h<4>; .reg .f32 f<8>; .reg .b64 s, x, y; "
    code += "mov.b32 {c0, c1}, $4; mov.b32 {c2, c3}, $5; mov.b64 s, {$6, $6}; "
    for i in range(4):
        code += f"cvt.rn.f16x2.e4m3x2 h{i}, c{i}; mov.b32 {{l{2 * i}, l{2 * i + 1}}}, h{i}; "
    for i in range(8):
        code += f"cvt.f32.f16 f{i}, l{i}; "
    for i in range(4):
        a, b = 2 * i, 2 * i + 1
        code += f"mov.b64 x, {{f{a}, f{b}}}; mul.rn.f32x2 y, x, s; mov.b64 {{f{a}, f{b}}}, y; "
    for i in range(4):
        code += f"cvt.rn.bf16x2.f32 ${i}, f{2 * i + 1}, f{2 * i}; "
    code += "}"
    return code, "=r,=r,=r,=r,r,r,f", (Uint32(lo), Uint32(hi), Float32(scale))


@asm((Uint32,) * 4)
def lds128(address):
    code = "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];"
    return code, "=r,=r,=r,=r,r,~{memory}", (Int32(address),)


@asm(Uint32)
def lds8(address):
    return "ld.shared.u8 $0, [$1];", "=r,r,~{memory}", (Int32(address),)


@asm()
def sts128(address, words):
    code = "st.shared.v4.b32 [$0], {$1, $2, $3, $4};"
    return code, "r,r,r,r,r,~{memory}", (Int32(address), *(Uint32(w) for w in words))


@asm()
def bulk_g2s(dst, src, barrier, size):
    """1-D bulk copy of ``size`` bytes (L2 evict_first) into this CTA's shared address
    ``dst``, completing ``size`` transaction bytes on ``barrier``."""
    code = "{ .reg .b64 p; createpolicy.fractional.L2::evict_first.b64 p, 1.0; "
    code += "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
    code += ".L2::cache_hint [$0], [$1], $3, [$2], p; }"
    return code, "r,l,r,r,~{memory}", (dst, Int64(src), barrier, Int32(size))


def code_offset(n_pad, kb, row):
    """Byte offset of (64-wide K block ``kb``, row ``row``)'s 64 codes."""
    return (kb * n_pad + row) * CODE_BYTES


def scale_offset(n_pad, k, kb, row):
    """Byte offset of (K block ``kb``, row ``row``)'s two scale bytes."""
    return k * n_pad + (kb * n_pad + row) * SCALE_BYTES


def swizzled(tile, row, chunk):
    """Shared address of 16-byte ``chunk`` of ``row`` in a K-major 128B-swizzled BF16
    [rows, 64] tile at shared address ``tile`` (the TMA SW128 box layout)."""
    return tile + row // 8 * 1024 + row % 8 * 128 + (chunk ^ row % 8) * 16


@cute.jit
def stage_rows(dst, barrier, base, n_pad: cutlass.Constexpr, k: cutlass.Constexpr,
               kb, row0, rows: cutlass.Constexpr, at_row: cutlass.Constexpr = 0,
               total: cutlass.Constexpr = 0):  # fmt: skip
    """Bulk-copy rows [row0, + rows) of K block ``kb`` into the staging block at shared
    ``dst`` (``total`` rows: codes [total][64], then scales [total][2]) at row
    ``at_row``. The caller arms ``barrier`` with the bytes of every part."""
    total_rows = total if total else rows
    src = Int64(base) + Int64(code_offset(n_pad, kb, row0))
    bulk_g2s(dst + at_row * CODE_BYTES, src, barrier, rows * CODE_BYTES)
    src = Int64(base) + Int64(scale_offset(n_pad, k, kb, row0))
    dst_scales = dst + total_rows * CODE_BYTES + at_row * SCALE_BYTES
    bulk_g2s(dst_scales, src, barrier, rows * SCALE_BYTES)


@cute.jit
def convert_block(staged, tile, rows: cutlass.Constexpr, threads: cutlass.Constexpr,
                  tid):  # fmt: skip
    """Staging block [rows][64] codes + [rows][2] scales at shared ``staged`` -> the
    BF16 [rows, 64] SW128 tile at shared ``tile``. Thread ``tid`` of ``threads``: one
    (row, 32-group) item at a time."""
    items: cutlass.Constexpr = rows * 2
    for i in cutlass.range_constexpr((items + threads - 1) // threads):
        item = tid + threads * i
        if cutlass.const_expr((i + 1) * threads <= items):
            convert_item(staged, tile, rows, item)
        else:
            if item < items:  # noqa: SIM102
                convert_item(staged, tile, rows, item)


@cute.jit
def convert_item(staged, tile, rows: cutlass.Constexpr, item):
    row, half = item // 2, item % 2
    scale = scale_f32(lds8(staged + rows * CODE_BYTES + row * SCALE_BYTES + half))
    for q in cutlass.range_constexpr(2):
        # rotate the two 16-byte reads by row so a warp's loads spread over the banks
        part = (q + row // 4) % 2
        words = lds128(staged + row * CODE_BYTES + half * 32 + part * 16)
        lo = bf16x8(words[0], words[1], scale)
        hi = bf16x8(words[2], words[3], scale)
        chunk = half * 4 + part * 2
        sts128(swizzled(tile, row, chunk), lo)
        sts128(swizzled(tile, row, chunk + 1), hi)
