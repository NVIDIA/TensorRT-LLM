# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CSA2 output projection: inverse RoPE, the grouped O-A GEMM and O-B, with the activation quantized once.

`att` is the attention output `[M, G * D]` bf16 (every 512-channel head carries its interleaved RoPE tail rotated
forward). The engine's decode path (`csa2/projection.py`) is reproduced here:

* at most 16 rows of a decode-graph step (`flashinfer_mxfp8_decode_graph_capture`, the engine's `_lowm_o_a` gate),
  given O-B's per-row scales (`wo_b_scale`) and weights the low-M kernel streams well (a target with one or two groups
  per rank), run two low-M warp-MMA launches: the grouped O-A rotating the tails back as it loads
  (`fp8_grouped_lowm`, bf16 latent) and O-B (`mxfp8_lowm_linear`); exact FP8 weights, no quantized latent;
* else at most 16 rows run one Triton launch that rotates the
tails back as it loads, multiplies by the 128x128-block FP8
  O-A weight (`inverse_rope_block128_small_m`) and writes the latent already as MXFP8 for O-B;
* more rows quantize the rotated output to MXFP8 in one launch (`inverse_rope_mxfp8`) and run the grouped CuTe-DSL MXFP8
  GEMM (`grouped_output_mxfp8`), again with an MXFP8 latent;
* O-B is the autotuned / dense MXFP8 GEMM of `csa2_mxfp8_linear` over the already quantized latent.
  `native_eager` (tep8): outside decode graphs O-B is stock's native autotuned projection GEMM at every row count
  (`native_mxfp8_gemm`), as `csa2_mxfp8_linear(native_eager=True)`.

O-A is the checkpoint's grouped `wo_a` requantized to 128x128 power-of-two blocks (the engine's loader does the same).
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_mxfp8_linear import SMALL_M_ROWS, _dequant, _lowm_fits, _quantized_gemm
from .csa2_rope import csa2_rope_tail

_BLOCK = 128
_HEAD_DIM = 512
_NAN_ROWS = {}


def _nan_row(device: torch.device, columns: int) -> torch.Tensor:
    """One bf16 NaN row per (device, width), made once (the first call is an eager step, before any graph capture)."""
    key = (device.index, columns)
    row = _NAN_ROWS.get(key)
    if row is None:
        row = _NAN_ROWS[key] = torch.full(
            (1, columns), float("nan"), dtype=torch.bfloat16, device=device
        )
    return row


def csa2_o_a_mxfp8_scales(block_scale: torch.Tensor) -> torch.Tensor:
    """The fp32 `[G, R/128, D/128]` power-of-two block scales -> the CUTLASS-swizzled UE8M0 scales of the grouped MXFP8
    GEMM `[G, R*D/32]` (a setup helper: scale metadata only, the E4M3 weight bytes are untouched)."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import mxfp8_weight_scale

    out = mxfp8_weight_scale(block_scale)
    assert out is not None, "the O-A block scales must be exact powers of two"
    return out


def requantize_block128(weight: torch.Tensor):
    """bf16 `[R, C]` -> (E4M3 `[R, C]`, fp32 `[R/128, C/128]` power-of-two multipliers), `w ~= q * s` (a setup helper;
    the engine loader's requantization of the grouped `wo_a`)."""
    r, c = weight.shape
    tr, tc = -(-r // _BLOCK), -(-c // _BLOCK)
    w = torch.nn.functional.pad(weight.float(), (0, tc * _BLOCK - c, 0, tr * _BLOCK - r))
    amax = w.reshape(tr, _BLOCK, tc, _BLOCK).abs().amax(dim=(1, 3))
    ratio = amax / 448.0
    scale = torch.where(
        ratio > 0, torch.exp2(torch.ceil(torch.log2(ratio))), torch.ones_like(ratio)
    )
    expanded = scale.repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1)
    return (w / expanded)[:r, :c].to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


class Csa2OProjection(OpWrapper):
    """`out = (inverse_rope(att) grouped@ wo_a) @ wo_b.T`, bf16 `[M, N]`.

    `att` bf16 `[M, G*D]` (unit column stride), `pos` int32 `[M]`, `cos_sin` the engine's fp32 `[P, 64]` table (all cos,
    then all sin), `wo_a` E4M3 `[G, R, D]` with fp32 block scales `wo_a_scale` `[G, R/128, D/128]` and `wo_a_mx` their
    swizzled UE8M0 form (`csa2_o_a_mxfp8_scales`), `wo_b` E4M3 `[N, G*R]` with its swizzled scales `wo_b_sw` and `alpha`
    fp32 `[1]`. `wo_b_scale` (optional) is `wo_b`'s per-row 1x32 UE8M0
    scales `[N, G*R/32]`: with it small steps take the low-M pair.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 decode, six rows: the small-M O-A launch",
            spec=dict(m=6, g=8, r=1024, d=4096, n=5120),
        ),
        Cell(
            why="dep8 decode, 16 rows: the largest small-M step",
            spec=dict(m=16, g=8, r=1024, d=4096, n=5120),
        ),
        Cell(
            why="dep8 step of 160 rows: grouped MXFP8 GEMM, 128x64 tiles",
            spec=dict(m=160, g=8, r=1024, d=4096, n=5120),
        ),
        Cell(
            why="dep8 step of 300 rows: grouped MXFP8 GEMM, 2-CTA tiles",
            spec=dict(m=300, g=8, r=1024, d=4096, n=5120),
        ),
        Cell(why="tep8 decode, one group, six rows", spec=dict(m=6, g=1, r=1024, d=4096, n=5120)),
        Cell(why="tep8 verify, one group, 24 rows", spec=dict(m=24, g=1, r=1024, d=4096, n=5120)),
        Cell(
            why="tep8 decode, one group, six rows: the low-M pair",
            spec=dict(m=6, g=1, r=1024, d=4096, n=5120, lowm=True, graph=True),
        ),
        Cell(
            why="tep8 decode, one row: the low-M pair",
            spec=dict(m=1, g=1, r=1024, d=4096, n=5120, lowm=True, graph=True),
        ),
        Cell(
            why="tep8, 16 rows: the low-M pair at its edge",
            spec=dict(m=16, g=1, r=1024, d=4096, n=5120, lowm=True, graph=True),
        ),
        Cell(
            why=(
                "tep8 eager step, one group, six rows, O-B scales given: "
                "the small-M launch (low-M only in a decode graph)"
            ),
            spec=dict(m=6, g=1, r=1024, d=4096, n=5120, lowm=True),
        ),
        Cell(
            why="tep8 eager verify, 24 rows (`native_eager`): O-B on stock's native autotuned projection GEMM",
            spec=dict(m=24, g=1, r=1024, d=4096, n=5120, native=True),
        ),
        Cell(
            why="tep8 context chunk of 2048 rows (`native_eager`): O-B on stock's native autotuned projection GEMM",
            spec=dict(m=2048, g=1, r=1024, d=4096, n=5120, native=True),
        ),
        Cell(
            why="dep2 context chunk of 4096 rows: grouped MXFP8 GEMM",
            spec=dict(m=4096, g=8, r=1024, d=4096, n=5120),
        ),
        Cell(why="dep2 context step of 9500 rows", spec=dict(m=9500, g=8, r=1024, d=4096, n=5120)),
    )
    note = """
    The latent between O-A and O-B is quantized to MXFP8 (a 3-bit mantissa) and the activation of the large steps too,
    so the gate is a relative RMS error of 3% (5% above 16 rows, where the activation is quantized as well) against the
    unquantized product with the same (requantized) weights. The low-M pair keeps
    a bf16 latent (exact FP8 weights, fp32 accumulation): its gate is 1%.
    """

    def __call__(
        self,
        att: torch.Tensor,
        pos: torch.Tensor,
        cos_sin: torch.Tensor,
        wo_a: torch.Tensor,
        wo_a_scale: torch.Tensor,
        wo_a_mx: torch.Tensor,
        wo_b: torch.Tensor,
        wo_b_sw: torch.Tensor,
        alpha: torch.Tensor,
        wo_b_scale: torch.Tensor = None,
        att_mxfp8: tuple = None,
        out: torch.Tensor = None,
        native_eager: bool = False,
    ) -> torch.Tensor:
        """`att_mxfp8`: the O-A input as the sparse decode's combine already wrote it (E4M3 `[G, M, D]`, swizzled
        scales: `csa2_sparse_decode(mxfp8_out=...)`); `att` is then not read and may be None. `out`: a contiguous bf16
        `[M, N]` that receives the result (returned)."""
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import (
            grouped_output_mxfp8,
            inverse_rope_block128_small_m,
        )

        g, r, d = wo_a.shape
        if att_mxfp8 is not None:
            m = att_mxfp8[0].shape[1]
            if att is None:  # nothing reads it: one cached NaN row broadcast, as the engine passes
                att = _nan_row(wo_a.device, g * d).expand(m, g * d)
            got = grouped_output_mxfp8(
                att,
                pos,
                cos_sin,
                wo_a,
                wo_a_mx,
                alpha,
                quantize_output=True,
                prequantized=att_mxfp8,
            )
            if got is None:
                raise RuntimeError(
                    "the grouped MXFP8 O-A GEMM must be compiled by an eager step before a graph capture"
                )
            return _quantized_gemm(
                got[0], got[1], wo_b, wo_b_sw, alpha, out, native_eager=native_eager
            )
        m = att.shape[0]
        if (
            wo_b_scale is not None
            and _lowm_fits(m, g * r, d, wo_a)
            and _lowm_fits(m, wo_b.shape[0], g * r, wo_b)
        ):
            from tensorrt_llm._torch.modules.low_m_gemm import fp8_grouped_lowm, mxfp8_lowm_linear

            latent = fp8_grouped_lowm(
                att if att.is_contiguous() else att.contiguous(), wo_a.view(g * r, d),
                wo_a_scale.view(g * r // _BLOCK, d // _BLOCK), r,
                inverse_rope=(pos, cos_sin.view(-1, 2, 32), _HEAD_DIM),
            )  # fmt: skip
            y = mxfp8_lowm_linear(latent, wo_b, wo_b_scale)
            return y if out is None else out.copy_(y)
        if m <= SMALL_M_ROWS:
            data, scales = inverse_rope_block128_small_m(
                att, pos, cos_sin, wo_a, wo_a_scale, name="csa2_o_a", quantize_output=True
            )
        else:
            got = grouped_output_mxfp8(
                att, pos, cos_sin, wo_a, wo_a_mx, alpha, quantize_output=True
            )
            if got is None:
                raise RuntimeError(
                    "the grouped MXFP8 O-A GEMM must be compiled by an eager step before a graph capture"
                )
            data, scales = got
        return _quantized_gemm(data, scales, wo_b, wo_b_sw, alpha, out, native_eager=native_eager)

    def reference(
        self,
        att,
        pos,
        cos_sin,
        wo_a,
        wo_a_scale,
        wo_a_mx,
        wo_b,
        wo_b_sw,
        alpha,
        wo_b_scale=None,
        att_mxfp8=None,
        out=None,
        native_eager=False,
    ) -> torch.Tensor:
        """fp32 product of the inverse-rotated output with the dequantized O-A (block scales) and O-B weights
        (`wo_b_scale` the per-row 1x32 UE8M0 scales of `wo_b`; the test passes them)."""
        m = att.shape[0]
        g, r, d = wo_a.shape
        table = torch.stack([cos_sin[:, :32], cos_sin[:, 32:]], dim=-1)  # [P, 32, 2] (cos, sin)
        x = att.view(m, g * d // _HEAD_DIM, _HEAD_DIM)
        x = csa2_rope_tail.reference(x, table, pos, 64, True).reshape(m, g, d).float()
        w = wo_a.float() * wo_a_scale.repeat_interleave(_BLOCK, 1).repeat_interleave(_BLOCK, 2)
        latent = torch.einsum("mgd,grd->mgr", x, w).reshape(m, g * r)
        return (latent @ _dequant(wo_b, wo_b_scale).t()).to(torch.bfloat16)

    def is_valid(
        self,
        att,
        pos,
        cos_sin,
        wo_a,
        wo_a_scale,
        wo_a_mx,
        wo_b,
        wo_b_sw,
        alpha,
        wo_b_scale=None,
        att_mxfp8=None,
        out=None,
        native_eager=False,
    ) -> None:
        g, r, d = wo_a.shape
        if (
            att_mxfp8 is not None
        ):  # the O-A input as the sparse decode wrote it: `att` is not read (may be None)
            data, scales = att_mxfp8
            assert (
                data.dtype == torch.float8_e4m3fn
                and data.dim() == 3
                and data.shape[0] == g
                and data.shape[2] == d
            )
            assert scales.dtype == torch.uint8 and scales.numel() == g * (
                -(-data.shape[1] // 128) * 128
            ) * (d // 32), (
                "att_mxfp8 scales are the per-group swizzled UE8M0 tiles of csa2_sparse_decode(mxfp8_out=...)"
            )
            m = data.shape[1]
        else:
            assert att.dtype == torch.bfloat16 and att.stride(-1) == 1 and att.shape[1] == g * d
            m = att.shape[0]
        assert d % _HEAD_DIM == 0
        assert pos.dtype == torch.int32 and pos.shape == (m,) and pos.is_contiguous()
        assert cos_sin.dtype == torch.float32 and cos_sin.is_contiguous() and cos_sin.shape[1] == 64
        assert (
            wo_a.dtype == torch.float8_e4m3fn
            and wo_a.is_contiguous()
            and r % _BLOCK == 0
            and d % _BLOCK == 0
        )
        assert wo_a_scale.dtype == torch.float32 and wo_a_scale.shape == (
            g,
            r // _BLOCK,
            d // _BLOCK,
        )
        assert wo_a_mx.dtype == torch.uint8 and wo_a_mx.numel() == g * r * d // 32
        assert (
            wo_b.dtype == torch.float8_e4m3fn
            and wo_b.shape[1] == g * r
            and alpha.dtype == torch.float32
        )


class Csa2OProjectionFp8In(OpWrapper):
    """`out = (dequant(data, scale) grouped@ wo_a) @ wo_b.T`, bf16 `[M, N]`, from an O-A input that is already rotated
    back and quantized (`csa2_sparse_context_o_a_input`: E4M3 `data [G, M, D]`, fp32 per row and 128 channels `scale
    [G, D/128, pad4(M)]`), as stock's `_project_prequantized_output`: the grouped block-128 FP8 O-A GEMM writes the
    latent as MXFP8 for O-B (`grouped_output_block128_mxfp8`), then O-B as in `csa2_o_projection`. `wo_a` E4M3
    `[G, R, D]` with fp32 128x128 block scales `wo_a_scale`, `wo_b` E4M3
    `[N, G*R]` with its swizzled scales `wo_b_sw`, `alpha` fp32 `[1]`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="dep2 context step of 300 rows", spec=dict(m=300, g=8, r=1024, d=4096, n=5120)),
        Cell(
            why="dep2 context chunk of 4097 rows (scale rows padded)",
            spec=dict(m=4097, g=8, r=1024, d=4096, n=5120),
        ),
    )
    note = (
        "The latent is MXFP8 between O-A and O-B: gate 3% relative "
        "RMS against the fp32 product of the dequantized input."
    )

    def __call__(
        self, data, scale, wo_a, wo_a_scale, wo_b, wo_b_sw, alpha, out=None
    ) -> torch.Tensor:
        """`out`: a contiguous bf16 `[M, N]` that receives the result (returned), as stock's `out` of the O-B GEMM."""
        from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import (
            grouped_output_block128_mxfp8,
        )

        latent, latent_scale = grouped_output_block128_mxfp8(data, scale, wo_a, wo_a_scale)
        return _quantized_gemm(latent, latent_scale, wo_b, wo_b_sw, alpha, out)

    def reference(
        self, data, scale, wo_a, wo_a_scale, wo_b, wo_b_sw, alpha, wo_b_scale=None, out=None
    ) -> torch.Tensor:
        """fp32 product of the dequantized input with the dequantized O-A (block scales) and O-B weights (`wo_b_scale`
        the per-row 1x32 UE8M0 scales of `wo_b`; the test passes them)."""
        g, m, d = data.shape
        x = data.float() * scale[:, :, :m].permute(0, 2, 1).repeat_interleave(_BLOCK, 2)
        w = wo_a.float() * wo_a_scale.repeat_interleave(_BLOCK, 1).repeat_interleave(_BLOCK, 2)
        latent = torch.einsum("gmd,grd->mgr", x, w).reshape(m, -1)
        return (latent @ _dequant(wo_b, wo_b_scale).t()).to(torch.bfloat16)

    def is_valid(
        self, data, scale, wo_a, wo_a_scale, wo_b, wo_b_sw, alpha, wo_b_scale=None, out=None
    ) -> None:
        g, m, d = data.shape
        r = wo_a.shape[1]
        assert data.dtype == torch.float8_e4m3fn and data.is_contiguous() and m > 0
        assert (
            scale.dtype == torch.float32
            and scale.is_contiguous()
            and scale.shape == (g, d // _BLOCK, (m + 3) // 4 * 4)
        )
        assert (
            wo_a.dtype == torch.float8_e4m3fn and wo_a.shape == (g, r, d) and wo_a.is_contiguous()
        )
        assert wo_a_scale.dtype == torch.float32 and wo_a_scale.shape == (
            g,
            r // _BLOCK,
            d // _BLOCK,
        )
        assert wo_a_scale.is_contiguous() and r % _BLOCK == 0 and d % _BLOCK == 0
        assert (
            wo_b.dtype == torch.float8_e4m3fn
            and wo_b.shape[1] == g * r
            and alpha.dtype == torch.float32
        )


csa2_o_projection = Csa2OProjection()
csa2_o_projection_fp8_in = Csa2OProjectionFp8In()
