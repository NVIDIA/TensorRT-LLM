# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target lm_head shard GEMV of a verify step of at most 8 rows over a lossless 12-bit coding of its BF16 weight,
and that coding (``encode_chunked`` / ``decode_reference`` / ``round_trip_ok``).

The coding: the BF16 exponents of a trained weight sit almost all in a window of 15 consecutive values
``[base, base + 14]``. Each weight is one byte ``sign << 7 | mantissa`` plus a 4-bit code ``exponent - base``; code 15
marks an exception, which decodes to +0 (it adds nothing to the MMA sums); its exact product ``x * w`` is added from a
per-tile list. The bytes are laid out in ``mma.m16n8k16`` A-fragment order:
per 16-row tile and unit of 4 k16 steps, each lane reads the sign/mantissa bytes of steps 0-1, of steps 2-3, then 4
code words (``[tiles][K / 64][3][32 lanes][16 bytes]``). A coded weight is used only after ``round_trip_ok`` proved
it decodes to the BF16 weight bit for bit.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

__all__ = ["m3_target_head"]

#: Verify rows per call (requests x (draft length + 1)).
MAX_ROWS = 8
#: Warps per CTA (the split-tile scratch holds 2 x 4 x 32 fp32 per warp).
CTA_WARPS = 16
#: The target lm_head's TP4 shard (200064-id vocabulary) and the hidden size.
TARGET_SHARD = 50016
HIDDEN = 6144
#: Codes per window: 0..14 decode, 15 marks an exception.
WINDOW = 15
#: The largest exponent base whose window top plus one (base + 15) is still a finite BF16 exponent.
MAX_BASE = 254 - WINDOW
#: Rows per block of encode_chunked / round_trip_ok (bounded temporaries).
ROWS_PER_CHUNK = 4096


def _pad_rows(weight: torch.Tensor) -> torch.Tensor:
    n = weight.shape[0]
    tiles = (n + 15) // 16
    if tiles * 16 == n:
        return weight
    return torch.cat([weight, weight.new_zeros(tiles * 16 - n, weight.shape[1])])


def _fragment_view(w: torch.Tensor) -> torch.Tensor:
    """[tiles * 16, K] -> [tiles, K / 16, 32 lanes, 8 values] in mma A-fragment order (a permutation of the values)."""
    n, k = w.shape
    # rows: (tile, rh, g); cols: (ks, ch, j, e) -> (tile, ks, g, j, ch, rh, e)
    return (
        w.view(n // 16, 2, 8, k // 16, 2, 4, 2)
        .permute(0, 3, 2, 5, 4, 1, 6)
        .reshape(n // 16, k // 16, 32, 8)
    )


def _exponents(weight: torch.Tensor) -> torch.Tensor:
    return (weight.contiguous().view(torch.int16).to(torch.int32) >> 7) & 0xFF


def _window_base(hist: torch.Tensor) -> int:
    """The base of the 15-exponent window holding the most weights (the lowest such base)."""
    window = torch.stack([hist[b : b + WINDOW].sum() for b in range(0, MAX_BASE + 1)])
    return int(window.argmax())


def encode(weight: torch.Tensor, base: Optional[int] = None):
    """Lossless 12-bit coding of a BF16 [N, K] weight (see the module docstring); ``base`` defaults to this weight's
    own window.

    Returns (packed uint8 [tiles * K / 64 * 1536], base, exc_off int32 [tiles + 1], exc_pos int32 [E] (row in tile
    << 16 | k), exc_w fp32 [E, 2] ({w, 0})).
    """
    w = _pad_rows(weight)
    n, k = w.shape
    tiles, ks = n // 16, k // 16
    assert ks % 4 == 0, "K must be a multiple of 64"
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    expo = (bits >> 7) & 0xFF
    if base is None:
        base = _window_base(torch.bincount(expo.flatten(), minlength=256))
    code = expo - base
    exc = (code < 0) | (code >= WINDOW)
    code = torch.where(exc, torch.full_like(code, WINDOW), code)
    sm = ((bits >> 8) & 0x80) | (bits & 0x7F)
    sm = torch.where(exc, torch.zeros_like(sm), sm)
    # Exceptions: (tile, row in tile, k) in tile-major order; code 15 decodes to +0.
    rows, cols = exc.nonzero(as_tuple=True)
    order = torch.argsort((rows // 16).long() * (16 * k) + (rows % 16).long() * k + cols.long())
    rows, cols = rows[order], cols[order]
    exc_w = torch.stack(
        [
            w[rows, cols].float(),
            torch.zeros((rows.numel(),), device=w.device),
        ],
        dim=1,
    )
    exc_pos = (((rows % 16) << 16) | cols).to(torch.int32)
    exc_off = torch.zeros(tiles + 1, dtype=torch.int32, device=w.device)
    exc_off[1:] = torch.cumsum(torch.bincount((rows // 16).long(), minlength=tiles), 0).to(
        torch.int32
    )
    # Fragment order, then per (tile, group of 4 k16 steps, lane): 32 sign/mantissa bytes and 4 code words.
    smf = _fragment_view(sm.to(torch.uint8).view(n, k)).view(tiles, ks // 4, 4, 32, 8)
    cf = (
        _fragment_view(code.to(torch.uint8).view(n, k))
        .view(tiles, ks // 4, 4, 32, 8)
        .to(torch.int32)
    )
    # Code word of a k16 step: value 2p -> bits 4p..4p+3, value 2p+1 -> bits 16+4p..
    cw = torch.zeros(tiles, ks // 4, 4, 32, dtype=torch.int64, device=w.device)
    for p in range(4):
        cw |= (cf[..., 2 * p].long() << (4 * p)) | (cf[..., 2 * p + 1].long() << (16 + 4 * p))
    cw = cw.to(torch.int32)  # wraps for bit 31; the kernel reads it as uint32
    # Word 0 of a lane = the bytes of steps 0 and 1; word 1 = steps 2 and 3; word 2 = the code words of steps 0..3.
    seg01 = (
        smf.permute(0, 1, 3, 2, 4).reshape(tiles, ks // 4, 32, 2, 16).permute(0, 1, 3, 2, 4)
    )  # [.., 2, 32, 16]
    seg2 = cw.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(tiles, ks // 4, 1, 32, 16)
    packed = torch.cat([seg01.contiguous(), seg2], dim=2)  # [tiles, ks / 4, 3, 32, 16]
    return packed.contiguous().view(-1), base, exc_off, exc_pos, exc_w.contiguous()


def encode_chunked(weight: torch.Tensor, rows_per_chunk: int = ROWS_PER_CHUNK):
    """``encode`` over row blocks (bounded temporaries), with the exponent window chosen on the whole weight."""
    n, _ = weight.shape
    assert rows_per_chunk % 16 == 0
    hist = torch.zeros(256, dtype=torch.int64, device=weight.device)
    for r in range(0, n, rows_per_chunk):
        hist += torch.bincount(_exponents(weight[r : r + rows_per_chunk]).flatten(), minlength=256)
    base = _window_base(hist)
    packed, offs, poss, ws = [], [], [], []
    n_exc = 0
    for r in range(0, n, rows_per_chunk):
        p, _, off, pos, ew = encode(weight[r : r + rows_per_chunk], base=base)
        packed.append(p)
        offs.append(off[:-1] + n_exc)
        poss.append(pos)
        ws.append(ew)
        n_exc += pos.numel()
    exc_off = torch.cat(offs + [torch.tensor([n_exc], dtype=torch.int32, device=weight.device)])
    return torch.cat(packed), base, exc_off, torch.cat(poss), torch.cat(ws)


def decode_reference(packed: torch.Tensor, base: int, n: int, k: int) -> torch.Tensor:
    """BF16 [N, K] from the coded stream with the in-kernel rule (exceptions, code 15, decode to +0)."""
    tiles, ks = (n + 15) // 16, k // 16
    p = packed.view(tiles, ks // 4, 3, 32, 16)
    sm = (
        p[:, :, :2]
        .permute(0, 1, 3, 2, 4)
        .reshape(tiles, ks // 4, 32, 4, 8)
        .permute(0, 1, 3, 2, 4)
        .to(torch.int32)
    )
    cw = (
        p[:, :, 2]
        .contiguous()
        .view(torch.int32)
        .view(tiles, ks // 4, 32, 4)
        .permute(0, 1, 3, 2)
        .long()
        & 0xFFFFFFFF
    )
    codes = torch.stack([(cw >> (4 * (i // 2) + 16 * (i % 2))) & 0xF for i in range(8)], dim=-1).to(
        torch.int32
    )
    bits = ((sm & 0x80) << 8) | ((codes + base) << 7) | (sm & 0x7F)
    bits = torch.where(codes == WINDOW, torch.zeros_like(bits), bits)
    frag = bits.reshape(tiles, ks, 32, 8)
    # Inverse of _fragment_view: (tile, ks, g, j, ch, rh, e) -> (tile, rh, g, ks, ch, j, e).
    w = frag.view(tiles, ks, 8, 4, 2, 2, 2).permute(0, 5, 2, 1, 4, 3, 6).reshape(tiles * 16, k)
    return w.to(torch.int16).view(torch.bfloat16)[:n]


def decode(
    packed: torch.Tensor,
    base: int,
    exc_off: torch.Tensor,
    exc_pos: torch.Tensor,
    exc_w: torch.Tensor,
    n: int,
    k: int,
    row0: int = 0,
    row1: Optional[int] = None,
) -> torch.Tensor:
    """BF16 rows [row0, row1) of the coded weight [N, K] with the exceptions applied (what the kernel multiplies by).
    ``row0`` must be a multiple of 16."""
    row1 = n if row1 is None else row1
    assert row0 % 16 == 0
    t0, t1 = row0 // 16, (row1 + 15) // 16
    unit_bytes = (k // 64) * 3 * 32 * 16
    dec = decode_reference(packed[t0 * unit_bytes : t1 * unit_bytes], base, row1 - row0, k).clone()
    counts = (exc_off[1:] - exc_off[:-1]).long()
    tile_of = torch.repeat_interleave(torch.arange(counts.numel(), device=packed.device), counts)
    rows = tile_of * 16 + (exc_pos >> 16).long()
    cols = (exc_pos & 0xFFFF).long()
    sel = (rows >= row0) & (rows < row1)
    dec[rows[sel] - row0, cols[sel]] = exc_w.view(-1, 2)[sel, 0].to(torch.bfloat16)
    return dec


def round_trip_ok(
    packed: torch.Tensor,
    base: int,
    exc_off: torch.Tensor,
    exc_pos: torch.Tensor,
    exc_w: torch.Tensor,
    weight: torch.Tensor,
    rows_per_chunk: int = ROWS_PER_CHUNK,
) -> bool:
    """Whether the coded weight (code values + exception list) decodes to ``weight`` bit for bit, with its exponent
    window below the BF16 infinity exponent."""
    n, k = weight.shape
    if not 0 <= base <= MAX_BASE or k % 64:
        return False
    if (
        packed.numel() != ((n + 15) // 16) * 16 * k * 3 // 2
        or exc_off.numel() != (n + 15) // 16 + 1
    ):
        return False
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(n, r0 + rows_per_chunk)
        dec = decode(packed, base, exc_off, exc_pos, exc_w, n, k, r0, r1)
        if not torch.equal(dec.view(torch.int16), weight[r0:r1].contiguous().view(torch.int16)):
            return False
    return True


class M3TargetHead(OpWrapper):
    """``out[:M, :n]`` (BF16) = x [M, K] (BF16, M <= 8) times the coded BF16 shard W [n, K] transposed: each logit
    the fp32 sum of the exact BF16 products, rounded once to BF16 (the in-tree BF16 head GEMM's arithmetic, in
    another summation order).

    ``w`` / ``exp_base`` / ``exc_off`` / ``exc_pos`` / ``exc_w`` are ``encode_chunked``'s outputs for W; ``part``
    (fp32 [grid * 16 * 256]) and ``counter`` (int32 [ceil(n / 16)], zero between calls, re-armed by the kernel) are
    the split-tile scratch; ``grid`` is the number of 16-warp CTAs (one per SM).
    """

    ARCHS = frozenset({Arch.SM_103})

    # The verify of a fused-verify step at most 8 rows wide: 1 + 3 drafts for one or two requests (4 / 8 rows), and
    # every other row count the kernel's one n8 MMA tile serves; one cell with an exception in many tiles, one with
    # a shard whose rows are not a multiple of 16.
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{m} verify row(s) of the TP4 target head shard",
                spec=dict(m=m, n=TARGET_SHARD, k=HIDDEN),
            )
            for m in range(1, MAX_ROWS + 1)
        ),
        Cell(
            why="4 rows with outliers and exact zeros in many tiles (the exception lists)",
            spec=dict(m=4, n=TARGET_SHARD, k=HIDDEN, exceptions=True),
        ),
        Cell(
            why="2 rows over a shard of 4000 rows (250 tiles), K 1024",
            spec=dict(m=2, n=4000, k=1024, exceptions=True),
        ),
    )

    note = """
    The weight the kernel multiplies by is the checkpoint's BF16 weight bit for bit (round_trip_ok); an exception
    adds nothing to the MMA sums and its exact product x * w is added at its tile's flush. A logit therefore differs
    from the in-tree head's only by the fp32 summation order before its single BF16 rounding (ulp-level);
    integer-valued data, whose sums are exact in any order, is equal bit for bit. Split tiles are summed in warp
    order: the result is deterministic.
    """

    def __call__(
        self,
        w: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        part: torch.Tensor,
        counter: torch.Tensor,
        exc_off: torch.Tensor,
        exc_pos: torch.Tensor,
        exc_w: torch.Tensor,
        exp_base: int,
        n: int,
        grid: int,
        pdl: bool = True,
    ) -> torch.Tensor:
        torch.ops.trtllm.minimax_m3_target_head(
            w, x, out, part, counter, exc_off, exc_pos, exc_w, exp_base, n, grid, pdl
        )
        return out

    def reference(
        self,
        w: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        part: torch.Tensor,
        counter: torch.Tensor,
        exc_off: torch.Tensor,
        exc_pos: torch.Tensor,
        exc_w: torch.Tensor,
        exp_base: int,
        n: int,
        grid: int,
        pdl: bool = True,
    ) -> torch.Tensor:
        """The BF16 logits [M, n] of x against the decoded weight: the fp32 product, rounded once to BF16.

        ``out``, ``part``, ``counter``, ``grid`` and ``pdl`` do not change the result."""
        weight = decode(w, exp_base, exc_off, exc_pos, exc_w, n, x.shape[1])
        with true_fp32_matmul():
            logits = x.float() @ weight.float().t()
        return logits.to(torch.bfloat16)

    def is_valid(
        self,
        w: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        part: torch.Tensor,
        counter: torch.Tensor,
        exc_off: torch.Tensor,
        exc_pos: torch.Tensor,
        exc_w: torch.Tensor,
        exp_base: int,
        n: int,
        grid: int,
        pdl: bool = True,
    ) -> None:
        # A split tile's last finisher is the CTA whose arrival makes the count reach its owners: a stale count
        # stores a partial sum.
        assert not bool(counter[: (n + 15) // 16].any()), "counter must be zero between calls"
        # The exceptions are read per tile in list order with no bound against the coded stream.
        assert int(exc_off[-1].item()) == exc_pos.numel(), "exc_off must end at the exception count"
        assert bool((exc_off[1:] >= exc_off[:-1]).all()), "exc_off must be non-decreasing"
        assert out.data_ptr() != x.data_ptr(), "out must not alias x"


m3_target_head = M3TargetHead()
