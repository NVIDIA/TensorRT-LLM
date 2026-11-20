# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton building blocks for kernels that end in an NVFP4 quantize.

Every fused "something + NVFP4" kernel of this target (SwiGLU, the attention
output gate, the gated RMSNorm, the gemma add+RMSNorm) ends in the same tail:
a group-16 E4M3 block scale and packed E2M1 codes, bitwise equal to
``torch.ops.trtllm.fp4_quantize`` (``cvt_warp_fp16_to_fp4`` in
``quantization.cuh``, built without fast math). The helpers here are that tail's
instructions, written as inline PTX so Triton cannot substitute an
approximation of its own:

* ``_rcp_approx_ftz``: ``rcp.approx.ftz.f32`` -- the CUDA quantizer's reciprocal;
* ``_mul_rn``: ``mul.rn.f32`` without flush-to-zero (the quantizer keeps subnormals
  in its multiplies; only the reciprocal flushes);
* ``_cvt_e4m3_bits``: ``cvt.rn.satfinite.e4m3x2.f32`` of one value, low byte;
* ``_cvt_e2m1x8``: eight values to eight E2M1 nibbles in one 32-bit word, the
  quantizer's ``fp32_vec_to_e2m1`` sequence (even element in the low nibble);
* ``_sf_offset``: byte offset of block-scale column ``c`` of row ``m`` in the
  128x4-swizzled scale buffer (``get_sf_out_offset_128x4``, batch 0);
* ``_gdc_wait_with_memory_clobber``: ``griddepcontrol.wait`` that no load or store
  of the kernel can be scheduled above.

The arithmetic a kernel uses with them, per 16-element block with block maximum
``vmax`` and global scale ``s``::

    sf  = e4m3(s * (vmax * rcp(6)))
    out = vmax != 0 ? rcp(sf * rcp(s)) : 0
    q   = e2m1(x * out)

Private to the target: other entries import these names, nothing else does.
"""

import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]


@triton.jit
def _gdc_wait_with_memory_clobber():
    # griddepcontrol.wait with a memory clobber, so no load or store of this
    # kernel can be scheduled above it.
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _mul_rn(a, b):
    # quantization.cuh is built without --use_fast_math, so its fp32 multiplies
    # keep subnormals; only rcp.approx is .ftz.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _cvt_e4m3_bits(x):
    # Low byte = e4m3(x), round-to-nearest-even, satfinite.
    return tl.inline_asm_elementwise(
        "{\n.reg .b16 t;\ncvt.rn.satfinite.e4m3x2.f32 t, $2, $1;\ncvt.u32.u16 $0, t;\n}",
        "=r,f,f",
        [x, tl.zeros_like(x)],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_e2m1x8(e0, e1, e2, e3, e4, e5, e6, e7):
    # Same instruction sequence as fp32_vec_to_e2m1 in quantization.cuh.
    return tl.inline_asm_elementwise(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, $2, $1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, $4, $3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, $6, $5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, $8, $7;\n"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n"
        "}",
        "=r,f,f,f,f,f,f,f,f",
        [e0, e1, e2, e3, e4, e5, e6, e7],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sf_offset(m, c, NUM_K_TILES: tl.constexpr):
    # get_sf_out_offset_128x4 (quantization.cuh), batch 0.
    return ((m // 128) * NUM_K_TILES + c // 4) * 512 + (m % 32) * 16 + ((m % 128) // 32) * 4 + c % 4
