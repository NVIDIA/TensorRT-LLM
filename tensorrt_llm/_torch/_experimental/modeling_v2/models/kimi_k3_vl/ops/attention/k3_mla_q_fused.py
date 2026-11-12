# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 FP4 MLA decode query in one kernel: q_a RMSNorm, q_b, W_UK, RoPE, FP4 quantization."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

from ...kernels.k3r_mla_q import k3_mla_q_fused as _kernels

#: E2M1 magnitudes by code (bit 3 is the sign).
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: Smallest FP4 group scale the query quantization uses (2^-9).
_SCALE_FLOOR = 2.0**-9


def swizzled_sf_offset(row: torch.Tensor, col: torch.Tensor, cols: int) -> torch.Tensor:
    """Byte offset of scale ``(row, col)`` in the 128x4-swizzled layout of ``cols`` columns."""
    padded = (cols + 3) // 4 * 4
    return (
        col % 4
        + col // 4 * (4 * 128)
        + row % 32 * 16
        + row % 128 // 32 * 4
        + row // 128 * (128 * padded)
    )


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """FP32 values of 4-bit E2M1 codes."""
    table = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)
    mag = table[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """E2M1 codes of FP32 ``v``: round to nearest, ties to the even code, saturating at 6."""
    table = torch.tensor(_E2M1, dtype=torch.float32, device=v.device)
    mag = v.abs().clamp_max(6.0)
    lower = (torch.bucketize(mag, table, right=True) - 1).clamp(0, 7)
    upper = (lower + 1).clamp_max(7)
    d_lo = mag - table[lower]
    d_hi = table[upper] - mag
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (upper % 2 == 0) & (upper != lower))
    code = torch.where(take_hi, upper, lower)
    return (code | torch.where(v < 0, 8, 0)).to(torch.uint8)


def fp4_quantize_groups(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP4 groups of 16 (last dim 16): ``(codes uint8 [..., 16], scale byte uint8 [...], scale
    FP32 [...])`` with the scale ``e4m3(max(amax / 6, 2^-9))`` and codes ``e2m1(v / scale)``."""
    amax = v.abs().amax(-1)
    scale = (amax.double() / 6.0).clamp(_SCALE_FLOOR, 448.0).to(torch.float8_e4m3fn)
    scale_f32 = scale.float()
    codes = e2m1_round((v.double() / scale_f32.double().unsqueeze(-1)).float())
    return codes, scale.view(torch.uint8), scale_f32


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack codes (last dim even) two per byte, element 2k in the low nibble of byte k."""
    pairs = codes.view(*codes.shape[:-1], -1, 2).to(torch.int32)
    return (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_e2m1`."""
    lo = packed & 15
    hi = packed >> 4
    return torch.stack((lo, hi), -1).flatten(-2)


class K3MlaQFused(OpWrapper):
    """``k3_mla_q_fused``: the packed FP4 decode query of ``T <= 16`` tokens of 96 heads, written
    into the decode's Q buffers ``q_fp4`` (uint8 ``[rows, 320]``) and ``q_sf`` (swizzled E4M3
    scales). Returns None."""

    ARCHS = frozenset({Arch.SM_107})
    # Spec: `requests` x `gen_len` tokens (T <= 16), `norm` (the q_a RMSNorm fused: FlashInfer's
    # CuTe DSL norm is the one replaced). The rotary table is K3's identity table and the query
    # global scale the FP4 MLA backend's.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8: two requests of 8 tokens (T = 16, the 16-token compile), fused norm",
            spec=dict(requests=2, gen_len=8, norm=True),
        ),
        Cell(
            why="one token, q_a already normalized (FlashInfer's CUDA norm ran before)",
            spec=dict(requests=1, gen_len=1, norm=False),
        ),
    )

    note = """
    x: BF16 [T, 1536] q_a rows (raw with `norm = (weight, eps)`: the kernel applies the q_a
    RMSNorm with FlashInfer's arithmetic; else already normalized). w_qb: FP8 [96 * 192, 1536]
    with block exponents `w_qb_exp` (`block_exponents`); w_uk: BF16 [96, 512, 128].
    Token t of request t // gen_len owns row (t // gen_len) * rows_per_seq + (t % gen_len) * 96 +
    head: 256 bytes of E2M1 absorbed values (global scale `q_global_scale`), then 32 bytes of RoPE
    values and 32 of their residual; one E4M3 scale per 16 values in the 128x4 swizzled layout.
    A negative position skips the token's RoPE part.

    The q_b product follows the low-M FP8 GEMM (MXFP8 activation, 128x128 UE8M0 weight blocks,
    FP32 per 32-value group, the two K halves added) up to the MMA's internal order; W_UK is a
    BF16 MMA with FP32 accumulation; RoPE and the FP4 quantization repeat the FP4 MLA cache
    update's operations (`div.full` / `rcp.approx`). Bitwise the production chain in the kernel's
    own tests. Writes only the rows of the given tokens. A process cache holds one compiled
    kernel per (T padded to 8 / 16, use_pdl, norm). With `use_pdl` the dependent grid is released
    after the query stores.
    """

    def __call__(
        self,
        x: torch.Tensor,
        w_qb: torch.Tensor,
        w_qb_exp: torch.Tensor,
        w_uk: torch.Tensor,
        q_global_scale: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        q_fp4: torch.Tensor,
        q_sf: torch.Tensor,
        gen_len: int,
        rows_per_seq: int,
        use_pdl: bool,
        norm: tuple[torch.Tensor, float] | None,
    ) -> None:
        _kernels.k3_mla_q_fused(
            x,
            w_qb,
            w_qb_exp,
            w_uk,
            q_global_scale,
            cos_sin,
            positions,
            q_fp4,
            q_sf,
            gen_len=gen_len,
            rows_per_seq=rows_per_seq,
            use_pdl=use_pdl,
            norm=norm,
        )

    def reference(
        self,
        x: torch.Tensor,
        w_qb: torch.Tensor,
        w_qb_exp: torch.Tensor,
        w_uk: torch.Tensor,
        q_global_scale: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        q_fp4: torch.Tensor,
        q_sf: torch.Tensor,
        gen_len: int,
        rows_per_seq: int,
        use_pdl: bool,
        norm: tuple[torch.Tensor, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The post-state ``(q_fp4, q_sf)`` (``q_sf`` as uint8): the rows of the given tokens
        rewritten, every other byte as passed. FP64 q_b product of the MXFP8 activation
        (``mxfp8_quantize_reference``) with the block-scaled weight, rounded to BF16; FP64 W_UK
        absorption rounded to BF16; GPT-J RoPE in FP32 with the pair-``p`` table values
        ``cos_sin[pos, 2p]``, ``[2p + 1]``, rounded to BF16; then FP4 groups of 16 of the values
        times ``q_global_scale`` (exact division where the kernel uses ``div.full`` /
        ``rcp.approx``), the RoPE groups with their residual. ``use_pdl`` does not change the
        result; the cells use positions >= 0."""
        from ...kernels.k3r_gemm.k3_fp8w_lowm_mxfp8 import mxfp8_quantize_reference

        heads, nope, rope, lora = (
            _kernels.NUM_HEADS,
            _kernels.NOPE_DIM,
            _kernels.ROPE_DIM,
            _kernels.KV_LORA_RANK,
        )
        tokens = x.shape[0]
        out_q, out_sf = q_fp4.clone(), q_sf.clone().view(torch.uint8)
        if tokens == 0:
            return out_q, out_sf
        rows = x if norm is None else flashinfer_rmsnorm.reference(x, norm[0], norm[1])
        xq, xs = mxfp8_quantize_reference(rows.contiguous())
        a = xq.double() * torch.exp2(xs.double() - 127).repeat_interleave(32, -1)
        n, k = w_qb.shape
        w_scale = torch.exp2(w_qb_exp.double() - 127).repeat_interleave(128, 0)[:n]
        w = w_qb.double() * w_scale.repeat_interleave(128, 1)[:, :k]
        q = (a @ w.T).to(torch.bfloat16).view(tokens, heads, nope + rope)
        absorbed = torch.einsum("tjd,jcd->tjc", q[..., :nope].double(), w_uk.double())
        absorbed = absorbed.to(torch.bfloat16)
        table = cos_sin.view(-1, 2 * rope)[positions[:tokens].long()].view(tokens, 1, -1, 2)
        cos, sin = table[:, :, : rope // 2, 0], table[:, :, : rope // 2, 1]
        x0, x1 = q[..., nope:].float().view(tokens, heads, rope // 2, 2).unbind(-1)
        even = (x0 * cos).sub(x1 * sin)
        odd = (x1 * cos).add(x0 * sin)
        rotated = torch.stack((even, odd), -1).view(tokens, heads, rope).to(torch.bfloat16)
        gs = q_global_scale.float().view(())

        prefix = (absorbed.float() * gs).view(tokens, heads, lora // 16, 16)
        p_codes, p_sf, _ = fp4_quantize_groups(prefix)
        pe = (rotated.float() * gs).view(tokens, heads, rope // 16, 16)
        m_codes, m_sf, m_sf_f32 = fp4_quantize_groups(pe)
        main = e2m1_values(m_codes).double() * m_sf_f32.double().unsqueeze(-1)
        residual = (pe.double() - main).float()
        r_codes, r_sf, _ = fp4_quantize_groups(residual)

        rope_groups = rope // 16
        packed = torch.cat(
            (
                pack_e2m1(p_codes.flatten(-2)),
                torch.cat((pack_e2m1(m_codes), pack_e2m1(r_codes)), -1).flatten(-2),
            ),
            -1,
        )
        sf_bytes = torch.cat(
            (p_sf, torch.stack((m_sf, r_sf), -1).view(tokens, heads, 2 * rope_groups)), -1
        )
        t = torch.arange(tokens, device=x.device)
        seq = t // gen_len
        token_rows = seq * rows_per_seq + (t - seq * gen_len) * heads
        q_rows = (token_rows[:, None] + torch.arange(heads, device=x.device)[None, :]).flatten()
        out_q[q_rows] = packed.view(-1, packed.shape[-1])
        cols = sf_bytes.shape[-1]
        col = torch.arange(cols, device=x.device)
        offsets = swizzled_sf_offset(q_rows[:, None], col[None, :], cols)
        out_sf.view(-1)[offsets.flatten()] = sf_bytes.reshape(-1)
        return out_q, out_sf

    def is_valid(
        self,
        x: torch.Tensor,
        w_qb: torch.Tensor,
        w_qb_exp: torch.Tensor,
        w_uk: torch.Tensor,
        q_global_scale: torch.Tensor,
        cos_sin: torch.Tensor,
        positions: torch.Tensor,
        q_fp4: torch.Tensor,
        q_sf: torch.Tensor,
        gen_len: int,
        rows_per_seq: int,
        use_pdl: bool,
        norm: tuple[torch.Tensor, float] | None,
    ) -> None:
        # The launcher checks only positions, q_fp4 and cos_sin; the weights and x are read by
        # TMA / 16-byte loads in the geometry the kernel is built for, whatever their shapes.
        assert _kernels.can_run(x, w_qb, w_uk), "x [T <= 16, 1536] BF16, K3 q_b / W_UK weights"
        tokens = x.shape[0]
        assert tokens % gen_len == 0, "every request has gen_len tokens"
        assert rows_per_seq >= gen_len * _kernels.NUM_HEADS, "rows_per_seq below gen_len * 96"
        assert q_fp4.shape[0] >= tokens // gen_len * rows_per_seq, "q_fp4 lacks rows"


k3_mla_q_fused = K3MlaQFused()
