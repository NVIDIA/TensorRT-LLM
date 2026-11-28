# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack the three byte-row formats of the DeepSeek-V4.1 CSA2 cache pools (plain torch)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# kind -> (elements per row, block size, fp4?, scale kind)
_KINDS = {
    "swa": (512, 32, False, "e8m0"),  # 512 fp8 + 16 UE8M0 scales  = 528 bytes
    "main": (512, 16, True, "e4m3"),  # 256 fp4 + 32 E4M3 scales   = 288 bytes
    "index": (128, 32, True, "e8m0"),  # 64 fp4 + 4 UE8M0 scales    = 68 bytes
}
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # midpoints between neighbouring e2m1 magnitudes
_TIE_UP = (False, True, False, True, False, True, False)  # round-half-to-even picks the even code


def row_bytes(kind: str) -> int:
    n, blk, fp4, _ = _KINDS[kind]
    return (n // 2 if fp4 else n) + n // blk


def _ceil_log2(v: torch.Tensor) -> torch.Tensor:
    m, e = torch.frexp(v)
    return torch.where(m == 0.5, e - 1, e)


def _e2m1_code(a: torch.Tensor) -> torch.Tensor:
    code = torch.zeros_like(a, dtype=torch.uint8)
    for t, up in zip(_MID, _TIE_UP):
        code += ((a >= t) if up else (a > t)).to(torch.uint8)
    return code


def _ceil_log2_f64(v: torch.Tensor) -> torch.Tensor:
    """Smallest integer `e` with `2**e >= v`, from float64 `log2` then fixed by exact comparisons.

    `torch.log2` on a float64 CUDA tensor is not exact at powers of two (`log2(2**-6)` came back a
    hair above -6, so a plain `ceil` gave -5): an `amax` that is an exact power-of-two multiple of
    the scale divisor (7.0 / 448, 0.75 / 6) then got a scale twice too large. Checking `2**e`
    against `v` in float64, where both sides are exact, removes the dependence on `log2`'s last bit.
    """
    v = v.double()
    e = torch.ceil(torch.log2(v))
    e = torch.where(torch.exp2(e - 1.0) >= v, e - 1.0, e)
    e = torch.where(torch.exp2(e) < v, e + 1.0, e)
    return e.float()


def _kind_cells(rows_decode: int, rows_prefill: int) -> tuple[Cell, ...]:
    return (
        Cell(
            why="swa rows (the sliding-window pool) at the decode row count",
            spec=dict(kind="swa", rows=rows_decode, amp=1.0),
        ),
        Cell(
            why="swa rows over a full prefill chunk, large dynamic range",
            spec=dict(kind="swa", rows=rows_prefill, amp=8.0),
        ),
        Cell(
            why="main rows (compressed KV, fp4 + e4m3 scales) at the decode row count",
            spec=dict(kind="main", rows=rows_decode, amp=1.0),
        ),
        Cell(
            why="main rows over a prefill chunk, large dynamic range",
            spec=dict(kind="main", rows=rows_prefill, amp=8.0),
        ),
        Cell(
            why="index rows (fp4 + ue8m0 scales) at the decode row count",
            spec=dict(kind="index", rows=rows_decode, amp=1.0),
        ),
        Cell(
            why="index rows over a prefill chunk",
            spec=dict(kind="index", rows=rows_prefill, amp=8.0),
        ),
        Cell(
            why="all-zero rows: the scale clamps, not a division by zero",
            spec=dict(kind="index", rows=4, amp=0.0),
        ),
        Cell(
            why="tiny rows: below the smallest representable scale of the kind",
            spec=dict(kind="main", rows=4, amp=1e-6),
        ),
    )


class Csa2PackRows(OpWrapper):
    """Quantize `x[..., n]` to the byte row of `kind`; returns uint8 `[..., row_bytes(kind)]`.

    The three formats are the ones the DeepSeek-V4.1 checkpoint reference quantizes its
    cache rows to: fp8 + power-of-two scales (`swa`), fp4 + e4m3 scales (`main`), fp4 +
    power-of-two scales (`index`). The bytes are written to the engine's page pools by
    `csa2_scatter_rows*` and read back by `csa2_unpack_rows`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _kind_cells(1, 8192)
    note = """
    Plain torch, so any architecture computes the same bytes; sm_103 is where the cells
    were driven. fp32 and bf16 inputs are certified; fp16 is not. NaN / inf rows are not
    certified.
    """

    def __call__(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        n, blk, fp4, sk = _KINDS[kind]
        assert x.shape[-1] == n, (x.shape, n)
        xb = x.float().unflatten(-1, (n // blk, blk))
        amax = xb.abs().amax(dim=-1, keepdim=True)
        if not fp4:  # fp8 e4m3, power-of-two scale
            amax = amax.clamp(min=1e-4)
            e = _ceil_log2(amax / 448.0)
            s = torch.ldexp(torch.ones_like(amax), e)
            q = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).flatten(-2)
            sb = (e + 127).clamp(0, 254).to(torch.uint8).squeeze(-1)
            return torch.cat([q, sb], dim=-1)
        if sk == "e4m3":
            amax = amax.clamp(min=6 * 2.0**-9)
            s8 = (amax / 6.0).to(torch.float8_e4m3fn)
            s = s8.float()
            sb = s8.view(torch.uint8).squeeze(-1)
        else:
            amax = amax.clamp(min=6 * 2.0**-126)
            e = _ceil_log2(amax * (1.0 / 6.0))
            s = torch.ldexp(torch.ones_like(amax), e)
            sb = (e + 127).clamp(0, 254).to(torch.uint8).squeeze(-1)
        y = (xb / s).clamp(-6.0, 6.0)
        nib = _e2m1_code(y.abs()) | ((y < 0).to(torch.uint8) << 3)
        nib = nib.flatten(-2)
        data = nib[..., 0::2] | (nib[..., 1::2] << 4)
        return torch.cat([data, sb], dim=-1)

    def reference(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        """Brute force: the nearest e2m1 magnitude (ties to the even code) found by distance to
        the code table, and every power-of-two scale from an exact float64 `ceil(log2(.))` instead of
        the op's `frexp`."""
        n, blk, fp4, sk = _KINDS[kind]
        assert x.shape[-1] == n
        xb = x.reshape(-1, n // blk, blk).float()
        amax = xb.abs().amax(dim=-1, keepdim=True)
        if not fp4:
            e = _ceil_log2_f64(amax.clamp(min=1e-4) / 448.0)
            s = torch.exp2(e)
            q = (xb / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8).flatten(-2)
            sb = (e + 127).clamp(0, 254).to(torch.uint8).squeeze(-1)
            out = torch.cat([q, sb], dim=-1)
            return out.reshape(*x.shape[:-1], row_bytes(kind))
        if sk == "e4m3":
            s8 = (amax.clamp(min=6 * 2.0**-9) / 6.0).to(torch.float8_e4m3fn)
            s = s8.float()
            sb = s8.view(torch.uint8).squeeze(-1)
        else:
            prod = amax.clamp(min=6 * 2.0**-126) * (1.0 / 6.0)
            e = _ceil_log2_f64(prod)
            s = torch.exp2(e)
            sb = (e + 127).clamp(0, 254).to(torch.uint8).squeeze(-1)
        y = (xb / s).clamp(-6.0, 6.0)
        lut = torch.tensor(_E2M1, dtype=torch.float32, device=x.device)
        code = torch.arange(8, device=x.device)
        d = (y.abs().unsqueeze(-1) - lut).abs()
        cand = d == d.amin(dim=-1, keepdim=True)
        even = torch.where(cand & (code % 2 == 0), code, torch.full_like(code, 99)).amin(-1)
        anyc = torch.where(cand, code, torch.full_like(code, 99)).amin(-1)
        c = torch.where(even < 99, even, anyc).to(torch.uint8)
        nib = (c | ((y < 0).to(torch.uint8) << 3)).flatten(-2)
        data = nib[..., 0::2] | (nib[..., 1::2] << 4)
        out = torch.cat([data, sb], dim=-1)
        return out.reshape(*x.shape[:-1], row_bytes(kind))

    def is_valid(self, x: torch.Tensor, kind: str) -> None:
        assert kind in _KINDS, f"kind must be one of {sorted(_KINDS)}"
        assert x.shape[-1] == _KINDS[kind][0], (
            f"a {kind} row is {_KINDS[kind][0]} elements wide, got {x.shape[-1]}"
        )
        # fp16 would be accepted (cast to fp32) but is not certified; float64 loses nothing.
        assert x.dtype in (torch.float32, torch.bfloat16, torch.float64), (
            "pack_rows takes float inputs; an integer tensor is silently reinterpreted as values"
        )


class Csa2UnpackRows(OpWrapper):
    """Inverse of `csa2_pack_rows`: uint8 `[..., row_bytes]` -> `[..., n]` in `out_dtype`."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _kind_cells(1, 8192)
    note = """
    Plain torch. The e2m1 magnitude is computed arithmetically (no lookup table), which is
    what keeps it free of host-to-device copies under CUDA-graph capture.
    """

    def __call__(
        self, buf: torch.Tensor, kind: str, out_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        n, blk, fp4, sk = _KINDS[kind]
        nd = n // 2 if fp4 else n
        data, sb = buf[..., :nd], buf[..., nd:]
        if sk == "e4m3":
            s = sb.contiguous().view(torch.float8_e4m3fn).float()
        else:
            s = torch.ldexp(torch.ones_like(sb, dtype=torch.float32), sb.to(torch.int32) - 127)
        if fp4:
            lo, hi = data & 0xF, data >> 4
            nib = torch.stack([lo, hi], dim=-1).flatten(-2)
            c = (nib & 7).to(torch.int32)
            sub = c.float() * 0.5  # codes 0, 1: 0 and 0.5
            norm = torch.ldexp(1.0 + (c & 1).float() * 0.5, (c >> 1) - 1)  # codes 2..7
            v = torch.where(c < 2, sub, norm) * (1.0 - 2.0 * (nib >> 3).float())
        else:
            v = data.contiguous().view(torch.float8_e4m3fn).float()
        v = v.unflatten(-1, (n // blk, blk)) * s.unsqueeze(-1)
        return v.flatten(-2).to(out_dtype)

    def reference(
        self, buf: torch.Tensor, kind: str, out_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Table lookup per nibble / fp8 byte, scales by `2.0 ** (byte - 127)` in float64."""
        n, blk, fp4, sk = _KINDS[kind]
        nd = n // 2 if fp4 else n
        lead = buf.shape[:-1]
        flat = buf.reshape(-1, buf.shape[-1])
        data, sb = flat[:, :nd], flat[:, nd:]
        if sk == "e4m3":
            scales = sb.contiguous().view(torch.float8_e4m3fn).double()
        else:
            scales = torch.exp2(sb.double() - 127.0)
        if fp4:
            lut = torch.tensor(
                _E2M1 + tuple(-v for v in _E2M1), dtype=torch.float64, device=buf.device
            )
            nib = torch.stack([data & 0xF, data >> 4], dim=-1).flatten(-2).long()
            vals = lut[nib]
        else:
            vals = data.contiguous().view(torch.float8_e4m3fn).double()
        out = (vals.view(-1, n // blk, blk) * scales.unsqueeze(-1)).flatten(-2)
        return out.to(out_dtype).reshape(*lead, n)

    def is_valid(
        self, buf: torch.Tensor, kind: str, out_dtype: torch.dtype = torch.bfloat16
    ) -> None:
        assert kind in _KINDS, f"kind must be one of {sorted(_KINDS)}"
        assert buf.dtype == torch.uint8, "unpack_rows reads uint8 byte rows"
        assert buf.shape[-1] == row_bytes(kind), (
            f"a {kind} byte row is {row_bytes(kind)} wide, got {buf.shape[-1]}"
        )


csa2_pack_rows = Csa2PackRows()
csa2_unpack_rows = Csa2UnpackRows()
