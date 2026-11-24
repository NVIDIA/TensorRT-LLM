# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The head-parallel DFlash2 drafter window: this rank's heads over every key of the window, from its
replicated ring (:mod:`.rubin_ull_dflash2_headtp_ring`), no exchange.

Block row ``r = g * block + j`` (query at ``n_g + j``) attends positions ``[max(0, n_g + j + 1 - W),
n_g + j]`` read from ring rows ``slot_g * R + position mod R``: the history the verify stores wrote
and the block's own latents written before the call. The ``block x heads`` queries of a request
share one pass over its ``W + block - 1`` keys (``kernels/spec/rubin_ull_dflash2_headtp_decode.py``),
split ``BN`` keys a program and merged by log-sum-exp.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.rubin_ull_dflash2_headtp_decode import (
    launch,
    launch_ranked,
    launch_ranked_merge,
    split_shapes,
)

#: Kimi K3.1 DFlash2 drafter: kv_lora_rank 512 + rope 64, window 1024; 96 heads over TP16.
_LATENT = 512
_DIM = 576
#: (BN keys a split, BV value columns a program, BK latent columns a dot, warps); measured on GB300.
TILES = (32, 128, 128, 4)


def headtp_decode_workspace(groups: int, block: int, heads: int, window: int, device, tiles=TILES):
    """The split buffers for up to ``groups`` requests of ``block`` rows, made before capture."""
    o_shape, ml_shape = split_shapes(groups, block, heads, window, tiles, _LATENT)
    return SimpleNamespace(
        groups=groups,
        block=block,
        heads=heads,
        window=window,
        tiles=tuple(tiles),
        o_part=torch.empty(o_shape, dtype=torch.float32, device=device),
        ml=torch.empty(ml_shape, dtype=torch.float32, device=device),
    )


def _spec(starts, block=8, heads=6, slots=None, ring=1088, dtype="fp8") -> dict:
    return dict(
        starts=tuple(starts),
        block=block,
        heads=heads,
        slots=tuple(slots) if slots else tuple(range(len(starts))),
        ring=ring,
        dtype=dtype,
    )


class RubinUllDFlash2HeadTpDecode(OpWrapper):
    """``out[g * block + j, h] = softmax_i(scale * query . key_i) @ key_i[:512]`` over the keys at
    positions ``[max(0, n_g + j + 1 - W), n_g + j]``, key ``p`` the ring row ``slots[g] * R + p mod
    R`` (``R = ring_rows``), an FP8 ring read times ``kv_scale``. ``query`` ``[G * block, heads,
    576]`` BF16 holds the absorbed queries (``[q_nope W_UK | q_pe]``), ``ring`` the layer's rows
    ``[slots * R, 576]``. Returns a new ``[G * block, heads, 512]`` BF16 tensor; ``ws`` is
    :func:`headtp_decode_workspace` of at least ``G`` requests."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="b1 at the served window, 6 heads (history past the ring: wrapped)",
            spec=_spec((190000,)),
        ),
        Cell(why="b1, no history: only the block's own keys", spec=_spec((0,))),
        Cell(
            why="b3 short histories: one key, a page less one, W - 1",
            spec=_spec((1, 63, 1023), slots=(2, 0, 1)),
        ),
        Cell(
            why="b4 around the window and the ring: W, R - 1, R, 3 R + 5",
            spec=_spec((1024, 1087, 1088, 3269)),
        ),
        Cell(why="b2, block 5 (K = 4)", spec=_spec((4097, 1017), block=5)),
        Cell(why="b8 (c8: 64 rows), 6 heads", spec=_spec(tuple(range(5000, 5800, 100)))),
    )

    note = """
    One partial launch (splits x 512 / BV x requests programs) and one combine. FP8 keys read and
    rounded to BF16 in registers (as the helix_mla pack), FP32 scores and softmax, BF16 P and V,
    FP32 accumulation; the split merge in FP32, one BF16 rounding at the end. Not bitwise the
    helix_mla path (other key order and split points): the test bounds both against an FP32
    reference and against each other.
    """

    def __call__(
        self,
        query: torch.Tensor,
        ring: torch.Tensor,
        slots: torch.Tensor,
        starts: torch.Tensor,
        ring_rows: int,
        softmax_scale: float,
        kv_scale: float,
        ws: SimpleNamespace,
    ) -> torch.Tensor:
        out = torch.empty(
            (query.shape[0], query.shape[1], _LATENT), dtype=query.dtype, device=query.device
        )
        launch(
            query,
            ring,
            slots,
            starts,
            out,
            ws.o_part,
            ws.ml,
            softmax_scale,
            kv_scale,
            ring_rows,
            ws.window,
            ws.tiles,
        )
        return out

    def reference(
        self,
        query: torch.Tensor,
        ring: torch.Tensor,
        slots: torch.Tensor,
        starts: torch.Tensor,
        ring_rows: int,
        softmax_scale: float,
        kv_scale: float,
        ws: SimpleNamespace,
    ) -> torch.Tensor:
        """FP32: each row's keys gathered from the ring, read as FP32 times ``kv_scale``."""
        rows, heads = query.shape[0], query.shape[1]
        block, window = rows // slots.shape[0], ws.window
        out = torch.zeros(rows, heads, _LATENT, dtype=torch.float32, device=query.device)
        for r in range(rows):
            g, j = divmod(r, block)
            end = int(starts[g]) + j + 1
            pos = torch.arange(max(0, end - window), end, device=query.device)
            keys = ring[int(slots[g]) * ring_rows + pos % ring_rows].float() * kv_scale
            logits = query[r].float() @ keys.T * softmax_scale
            out[r] = logits.softmax(-1) @ keys[:, :_LATENT]
        return out

    def is_valid(
        self,
        query: torch.Tensor,
        ring: torch.Tensor,
        slots: torch.Tensor,
        starts: torch.Tensor,
        ring_rows: int,
        softmax_scale: float,
        kv_scale: float,
        ws: SimpleNamespace,
    ) -> None:
        assert query.dtype == torch.bfloat16 and query.is_contiguous(), "contiguous BF16 query"
        assert query.shape[2] == _DIM and query.shape[1] == ws.heads
        assert slots.shape[0] == starts.shape[0] <= ws.groups, "at most the workspace's requests"
        assert query.shape[0] == slots.shape[0] * ws.block, "block rows per request"
        assert ring.dtype in (torch.bfloat16, torch.float8_e4m3fn) and ring.dim() == 2
        assert ring.shape[1] == _DIM and ring.is_contiguous() and ring.shape[0] % ring_rows == 0
        assert ring_rows >= ws.window + ws.block - 1, "a window and a block fit the ring"


rubin_ull_dflash2_headtp_decode = RubinUllDFlash2HeadTpDecode()


# ---- the served headtp window: helix_mla's order of rounding ------------------------------------------------------
#: helix_mla's partition: 64-position pages, page g on CP rank g % 16.
RANKED_PAGE, RANKED_CP = 64, 16
#: The ranked partial's (BC latent columns a W_UV chunk, BK latent columns a QK dot, warps, splits of the 512
#: latent columns over CTAs, ptxas register cap a thread or None): the kernel's launch defaults (a GB300
#: microbenchmark) at one split. 16 page groups x 6 heads is 96 CTAs a request; at two splits one
#: request's 192 CTAs exceed GB300's 152 SMs (one CTA an SM at 140 registers x 256 threads).
RANKED_TILES = (128, 64, 8, 1, None)


def headtp_ranked_workspace(
    groups: int, block: int, heads: int, device, v: int = 128, splits: int = 2
):
    """Buffers of :func:`headtp_decode_ranked` for up to ``groups`` requests and up to ``splits`` column ranges
    (FP32 sums; ``part16``: one split's BF16 partials), made before capture."""
    n = RANKED_CP * groups * block * heads
    return SimpleNamespace(
        part=torch.empty((splits * n, v), dtype=torch.float32, device=device),
        part16=torch.empty((n, v), dtype=torch.bfloat16, device=device),
        lse=torch.empty((n,), dtype=torch.float32, device=device),
    )


def headtp_decode_ranked(
    query,
    ring,
    slots,
    starts,
    ring_rows,
    softmax_scale,
    kv_scale,
    window,
    v_b_t,
    rws,
    tiles=RANKED_TILES,
):
    """The window in helix_mla's order of rounding, steps 1 and 2 in one launch: per helix page group ``r``
    (the keys CP rank ``r`` would hold) the normalized BF16 partial times W_UV, as FP32 sums over ``splits``
    ranges of the latent columns, and its log-sum-exp. Returns ``(part [splits, 16, rows, heads, v] FP32, lse
    [16, rows, heads] FP32)`` for :func:`headtp_ranked_lse_merge`, which rounds each group's sum of the splits
    once to BF16; at one split ``part`` is BF16, the partials already rounded (the same bits, half the bytes
    and A's merge). ``tiles``: the partial's (BC, BK, warps, splits, maxnreg)."""
    rows, heads = query.shape[0], query.shape[1]
    v = v_b_t.shape[-1]
    bc, bk, warps, splits, maxnreg = tiles
    n = RANKED_CP * rows * heads
    buf = rws.part16 if splits == 1 else rws.part
    assert buf.dtype == (torch.bfloat16 if splits == 1 else torch.float32) and buf.shape[1] == v, (
        "v-wide partials"
    )
    assert buf.shape[0] >= splits * n, "the workspace holds every split"
    launch_ranked(
        query,
        ring,
        slots,
        starts,
        v_b_t,
        buf,
        rws.lse,
        softmax_scale,
        kv_scale,
        ring_rows,
        window,
        RANKED_PAGE,
        RANKED_CP,
        bc=bc,
        bk=bk,
        warps=warps,
        splits=splits,
        maxnreg=maxnreg,
    )
    part = buf[: splits * n].view(splits, RANKED_CP, rows, heads, v)
    return part, rws.lse[:n].view(RANKED_CP, rows, heads)


def headtp_ranked_lse_merge(part: torch.Tensor, lse: torch.Tensor, block: int) -> torch.Tensor:
    """Step 3: each group's splits added in order and rounded once to BF16, then the FP32 log-sum-exp merge of
    the groups' partials, rounded once; ``[rows, heads * v]``. :func:`headtp_ranked_merge` is its reference
    (bitwise). BF16 partials (one split) take the original merge launch (a program a row and head); the FP32
    sums four programs along V, the merge's time at a quarter of the bytes each."""
    splits, cp, rows, heads, v = part.shape
    out = torch.empty(rows, heads * v, dtype=torch.bfloat16, device=part.device)
    vsplit = 1 if part.dtype == torch.bfloat16 else 4
    launch_ranked_merge(
        part, lse, out, rows // block, block, heads, cp, splits=splits, vsplit=vsplit
    )
    return out


def headtp_ranked_merge(part: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
    """Step 3 in torch: ``part`` either ``[16, rows, heads, v]`` BF16 group partials (W_UV applied per group) or
    ``[splits, 16, rows, heads, v]`` FP32 column-range sums of them (added in split order, rounded once to
    BF16; one split may be BF16 already), merged by log-sum-exp in FP32, rounded once to BF16, as
    ``rubin_dflash2_helix_combine`` merges the ranks' partials; ``[rows, heads * v]``."""
    if part.dim() == 5:
        acc = part[0].clone()
        for s in range(1, part.shape[0]):
            acc += part[s]
        part = acc.to(torch.bfloat16)
    live = torch.isfinite(lse)
    peak = torch.where(live, lse, float("-inf")).amax(dim=0, keepdim=True)
    peak = torch.where(torch.isfinite(peak), peak, 0.0)
    weights = torch.where(live, torch.exp(torch.where(live, lse, 0.0) - peak), 0.0)
    merged = torch.where(live[..., None], weights[..., None] * part.float(), 0.0).sum(dim=0)
    merged = merged / weights.sum(dim=0).clamp_min(torch.finfo(torch.float32).tiny)[..., None]
    return merged.to(part.dtype).reshape(part.shape[1], -1)


__all__ = [
    "RANKED_CP",
    "RANKED_TILES",
    "TILES",
    "headtp_decode_ranked",
    "headtp_decode_workspace",
    "headtp_ranked_lse_merge",
    "headtp_ranked_merge",
    "headtp_ranked_workspace",
    "rubin_ull_dflash2_headtp_decode",
]
