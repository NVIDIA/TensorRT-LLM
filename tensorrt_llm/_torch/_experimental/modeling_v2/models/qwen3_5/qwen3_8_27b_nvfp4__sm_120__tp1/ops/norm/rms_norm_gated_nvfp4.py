# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GDN gated RMS norm (silu gate, norm before gate) with the NVFP4 quantize of the
output projection's input fused into its epilogue.

One Triton kernel: the multi-row gated norm kernel's expressions (4 head rows a
program, 4 warps, so the same ``tl.sum`` reduction), ``y`` rounded to bf16 as the
unfused norm stores it, then group-16 NVFP4 with E4M3 scales in the 128x4-swizzled
layout -- bitwise the registered ``torch.ops.trtllm.rms_norm_gated_token_major`` op
followed by ``fp4_quantize`` (the quantize tail is ``_nvfp4_triton``'s). Programs past
the main ones zero the scale padding rows. Launched with PDL.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major,
)

from ..quantization._nvfp4_triton import (
    _cvt_e2m1x8,
    _cvt_e4m3_bits,
    _gdc_wait_with_memory_clobber,
    _mul_rn,
    _rcp_approx_ftz,
    _sf_offset,
)

#: The checkpoint's GDN epilogue: 48 value heads of 128, `rms_norm_eps` 1e-6.
_CKPT = dict(heads=48, width=128, eps=1e-6)
_SF_VEC = 16
#: Scale columns one padding program zeroes.
_PAD_COLS = 128
#: The multi-row gated norm kernel's launch: head rows a program, warps.
_ROWS, _NUM_WARPS = 4, 4


@triton.jit
def _rms_norm_gated_nvfp4_kernel(
    X,  # [M, N] head rows (M = num_tokens * HEADS_PER_TOK)
    W,  # [N] norm weight
    Z,  # gate, token-major [num_tokens, HEADS_PER_TOK, N] view
    FP4,  # int32 view of the uint8 [num_tokens, HEADS_PER_TOK * N / 2] output
    SF,  # uint8 swizzled SF, ceil128(num_tokens) * NUM_K_TILES * 4 bytes
    SCALE,  # 1-element fp32 global scale (6 / input_scale)
    stride_x_row,
    stride_z_tok,
    M,
    num_tokens,
    num_main_programs,
    eps,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS_PER_TOK: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    PAD_COLS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """The multi-row gated norm kernel's expressions, then NVFP4.

    The norm part is a copy of the multi-row kernel (same [ROWS, N] block, same
    num_warps, so the same tl.sum reduction) and y is rounded to BF16 as the unfused
    path stores it. The quantize tail is cvt_warp_fp16_to_fp4 as built. Programs past
    num_main_programs zero the SF padding rows [num_tokens, ceil128(num_tokens)).

    With USE_PDL every load/store stays behind griddepcontrol.wait, and the dependents
    (the W4A4 GEMM) are released only after this program's stores, as
    quantize_with_block_size does.
    """
    if USE_PDL:
        _gdc_wait_with_memory_clobber()
    pid = tl.program_id(0)
    NSF: tl.constexpr = N // 16
    if pid < num_main_programs:
        rows = pid * ROWS + tl.arange(0, ROWS)
        row_mask = rows < M
        cols = tl.arange(0, N)
        mask2d = row_mask[:, None]
        x_off = rows[:, None].to(tl.int64) * stride_x_row + cols[None, :]
        x = tl.load(X + x_off, mask=mask2d, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols).to(tl.float32)
        y = x * rstd[:, None] * w[None, :]
        tok = rows // HEADS_PER_TOK
        head = rows % HEADS_PER_TOK
        z_off = tok[:, None].to(tl.int64) * stride_z_tok + head[:, None] * N + cols[None, :]
        z = tl.load(Z + z_off, mask=mask2d, other=0.0).to(tl.float32)
        gate = tl.sigmoid(z)
        gate *= z
        y *= gate
        # The unfused path stores BF16 and fp4_quantize reloads it.
        y = y.to(tl.bfloat16).to(tl.float32)

        h2 = tl.reshape(y, [ROWS * NSF, 16])
        vmax = tl.max(tl.abs(h2), axis=1)
        s = tl.load(SCALE)
        s_v = tl.full([ROWS * NSF], 0.0, tl.float32) + s
        r6 = _rcp_approx_ftz(tl.full([ROWS * NSF], 6.0, tl.float32))
        sf_val = _mul_rn(s_v, _mul_rn(vmax, r6))
        sf_u8 = _cvt_e4m3_bits(sf_val).to(tl.uint8)
        sf_f = sf_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        out_scale = tl.where(vmax != 0.0, _rcp_approx_ftz(_mul_rn(sf_f, _rcp_approx_ftz(s_v))), 0.0)
        q = _mul_rn(h2, tl.broadcast_to(out_scale[:, None], [ROWS * NSF, 16]))
        q8 = tl.reshape(q, [ROWS * N // 8, 2, 2, 2])
        p, qq = tl.split(q8)
        p0, p1 = tl.split(p)
        q0, q1 = tl.split(qq)
        p00, p01 = tl.split(p0)
        p10, p11 = tl.split(p1)
        q00, q01 = tl.split(q0)
        q10, q11 = tl.split(q1)
        packed = _cvt_e2m1x8(p00, q00, p10, q10, p01, q01, p11, q11)

        # Head row r owns FP4 bytes [r * N / 2, (r + 1) * N / 2) of the
        # contiguous [num_tokens, HEADS_PER_TOK * N / 2] output.
        w_idx = tl.arange(0, ROWS * N // 8)
        w_off = pid.to(tl.int64) * (ROWS * N // 8) + w_idx
        tl.store(FP4 + w_off, packed, mask=w_off < M * (N // 8))

        sf_idx = tl.arange(0, ROWS * NSF)
        sf_row = pid * ROWS + sf_idx // NSF
        sf_tok = sf_row // HEADS_PER_TOK
        sf_col = (sf_row % HEADS_PER_TOK) * NSF + sf_idx % NSF
        sf_off = _sf_offset(sf_tok.to(tl.int64), sf_col, NUM_K_TILES)
        tl.store(SF + sf_off, sf_u8, mask=sf_row < M)
    else:
        # Zero SF padding rows [num_tokens, ceil128(num_tokens)) for one
        # PAD_COLS-wide column block (as quantize_with_block_size does).
        cb = pid - num_main_programs
        pcols = cb * PAD_COLS + tl.arange(0, PAD_COLS)
        m0 = (num_tokens // 128) * 128
        prow = m0 + tl.arange(0, 128).to(tl.int64)
        off = _sf_offset(prow[:, None], pcols[None, :], NUM_K_TILES)
        pmask = (prow >= num_tokens)[:, None] & (pcols < HEADS_PER_TOK * NSF)[None, :]
        tl.store(SF + off, tl.zeros([128, PAD_COLS], tl.uint8), mask=pmask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _launch(x, z, weight, eps, global_scale, pdl):
    M, N = x.shape
    num_tokens, heads, _ = z.shape
    weight = weight.contiguous()
    K = heads * N
    num_sf_cols = K // _SF_VEC
    num_k_tiles = (num_sf_cols + 3) // 4
    m_pad = (num_tokens + 127) // 128 * 128
    fp4 = torch.empty((num_tokens, K // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((m_pad * num_k_tiles * 4,), dtype=torch.uint8, device=x.device)
    num_main = triton.cdiv(M, _ROWS)
    num_pad = triton.cdiv(num_k_tiles * 4, _PAD_COLS) if num_tokens % 128 != 0 else 0
    if num_main + num_pad == 0:
        return fp4, sf
    with torch.cuda.device(x.device.index):
        _rms_norm_gated_nvfp4_kernel[(num_main + num_pad,)](
            x,
            weight,
            z,
            fp4.view(torch.int32),
            sf,
            global_scale,
            x.stride(0),
            z.stride(0),
            M,
            num_tokens,
            num_main,
            eps,
            N=N,
            ROWS=_ROWS,
            HEADS_PER_TOK=heads,
            NUM_K_TILES=num_k_tiles,
            PAD_COLS=_PAD_COLS,
            USE_PDL=pdl,
            num_warps=_NUM_WARPS,
            launch_pdl=pdl,
        )
    return fp4, sf


def _cell(why: str, tokens: int, global_scale: float) -> Cell:
    return Cell(why=why, spec=dict(tokens=tokens, global_scale=global_scale, **_CKPT))


class RmsNormGatedNvfp4(OpWrapper):
    """`fp4_quantize(rms_norm_gated_token_major(x, z, weight, eps), global_scale, 16)`:
    row `r` of `x` `[T * heads, N]` gated by `z[r // heads, r % heads]` (silu gate, norm
    before gate, plain `weight`), rounded to bf16, then NVFP4 per 16 elements of each
    token's `heads * N` row.

    Returns `(fp4, sf)`: `fp4` `[T, heads * N / 2]` uint8 (element `2i` in the low
    nibble of byte `i`) and `sf` the 1-D 128x4-swizzled e4m3 scale buffer,
    `ceil(T / 128) * 128 * ceil(heads * N / 64) * 4` bytes, its padding rows
    `[T, ceil128(T))` zeroed -- what `fp4_quantize(..., is_sf_swizzled_layout=True)`
    returns. Both new; no input is mutated.

    Fusion boundary: norm, weight, gate, bf16 rounding, NVFP4 quantize. The recurrence
    producing `x`, the projection producing `z`, and out_proj's W4A4 GEMM consuming
    `(fp4, sf)` are the caller's.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: tokens, global_scale, heads, width, eps. x is the contiguous [T * 48, 128]
    # recurrence output; z the [T, 48, 128] view of the in_proj_qkvz output's z columns
    # (token stride 16384).
    CELLS: tuple[Cell, ...] = (
        _cell("decode 33 rows: the first past the bf16 norm band; 95 SF padding rows", 33, 336.0),
        _cell("decode 128 rows (concurrency 128): no SF padding", 128, 100.0),
        _cell("mixed step 2048 + 124 rows: 4-row programs over 104256 head rows", 2172, 2.0),
    )

    note = """
    Metadata consumed: none.

    Call site (modeling): every step with more than 32 rows, ahead of out_proj's W4A4
    copy (`nvfp4_gemm` on `(fp4, sf)`); `global_scale` that copy's activation global
    scale (`6 / input_scale`, one fp32 element); `pdl` the engine's PDL switch
    (`get_env_enable_pdl()`).

    Launch: `cdiv(T * heads, 4)` main programs (4 head rows each, 4 warps) plus, when T
    is not a multiple of 128, `cdiv(ceil(heads * N / 64) * 4, 128)` programs zeroing
    the scale padding rows. With `pdl` every load and store waits on
    `griddepcontrol.wait`, and `launch_dependents` fires after the program's stores.

    Numerics: the norm is the multi-row kernel's (fp32, `tl.sum` over a 4 x N block,
    `1 / sqrt(var + eps)`), `y` rounded once to bf16; the quantize is
    `cvt_warp_fp16_to_fp4`'s (`rcp.approx.ftz`, non-ftz `mul.rn`, e4m3 / e2m1 RNE
    satfinite). FP4 and SF bytes bitwise equal to the registered
    `torch.ops.trtllm.rms_norm_gated_token_major` (4 rows a program at every row count;
    the catalog entry's own 8-row prefill launch from 32768 rows sums in another order)
    followed by `fp4_quantize`, padding included.

    Not checked by the op (is_valid covers it): the shape eligibility (N a power of two
    <= 256 and a multiple of 64, `heads * N / 16` a multiple of 4), x's unit column
    stride, z's packed (heads, N) block, `global_scale` one fp32 element (only `[0]` is
    read), weight length.
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        global_scale: torch.Tensor,
        pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch(x, z, weight, eps, global_scale, pdl)

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        global_scale: torch.Tensor,
        pdl: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The gated norm entry's fp32 reference (rounded once to bf16), then
        `fp4_quantize`'s native-torch NVFP4 with swizzled scales. `pdl` is ignored."""
        y = rms_norm_gated_token_major.reference(x, z, weight, eps, None, "silu")
        num_tokens, heads, n = z.shape
        return fp4_quantize.reference(
            y.view(num_tokens, heads * n), global_scale, _SF_VEC, False, True
        )

    def is_valid(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        global_scale: torch.Tensor,
        pdl: bool = True,
    ) -> None:
        M, N = x.shape
        num_tokens, heads, n_z = z.shape
        assert n_z == N and num_tokens * heads == M, "x [T * heads, N], z [T, heads, N]"
        assert x.dtype == z.dtype == torch.bfloat16, "bf16 x and z"
        assert x.stride(1) == 1, "x rows with unit column stride"
        assert z.stride(2) == 1 and z.stride(1) == N, "z's (heads, N) block contiguous per token"
        assert N <= 256 and N & (N - 1) == 0 and N % 64 == 0, "N a power of two <= 256, % 64"
        assert (heads * N // _SF_VEC) % 4 == 0, "whole 4-column scale tiles per token"
        assert weight.shape == (N,), "weight [N]: the kernel reads weight[0:N] unmasked"
        assert global_scale.dtype == torch.float32 and global_scale.numel() == 1, (
            "global_scale one fp32 element: the kernel loads [0] only"
        )


rms_norm_gated_nvfp4 = RmsNormGatedNvfp4()
