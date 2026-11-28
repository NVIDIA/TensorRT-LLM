# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The QSA cold-page codec of a context server's attention pages (`TRTLLM_QWEN4_EXP_CTX_COLDIDX`):
Triton kernels behind `qsa_cold_pages.QsaColdIndexPages`, the cold-page provider of the
qwen4_exp family's KV cache manager, and the marker stores the forward makes behind every anchor
writer (`qsa_cold_index_mark`).

Under `INDEX_KEY_FP8` an index plane's anchor (the pooled key of a four-token group, slot `4g+3`)
is 128 e4m3 bytes in the first half of its 256-byte slot; the rest of the slot is never read
again. A group's raw keys (slots `4g..4g+2`) are read only until the group is pooled, and a
token's coordinates in the position page only at a group's first token. So a page's cold copy
needs, of the side planes, the anchors, the first-token coordinates and the raw keys of the
groups a later step can still pool: the lowest group without an anchor (the request's partial
group) and the one before it (the next turn of a context server resumes at the `L - 2`
snapshot, which lies in the last completed group when `L % 4` is 0 or 1). Which groups are
pooled is marked in the plane itself:

* `qsa_cold_index_mark`: a marker is 16 bytes (4 int32 words `MARKER`) in bytes 128-143 of a
  group's anchor slot. Per row of an anchor writer's call, after the compress and the e4m3
  rounding:
    - an anchor row (`(L + 1) % 4 == 0`) sets the marker of its slot `L`;
    - a first-token row (`L % 4 == 0`, `L > 0`) whose anchor row `L + 3` is NOT in the same call
      clears the marker of slot `L + 3` (it holds no data of this owner yet: a stale marker of a
      previous owner must not survive). When the anchor row is in the call the slot holds the
      raw key of token `L + 3` until the compress has pooled it, which is why the clear runs
      after the compress. Group 0 takes the same clear from its second row (`L == 1`, anchor row
      `3`): a padded row carries `L = 0` and must not clear a real request's page-0 marker;
    - an anchor row whose next group starts on the same page and whose next row `L + 1` is not
      in the call clears the marker of slot `L + 4`, so the lowest unmarked group of a partial
      page is exactly the first group this owner has not completed.
  Every slot is written by at most one program of a call (or by two with the same zero words).

* `ColdIndexTables.encode` / `decode`: gather / scatter between the hot pools and the cold blob
  `cold_base + dst * COLD_BYTES`. Programs `(page, j)`: `j < NVB` copies verbatim buffer `j`
  (the K/V pages, or every buffer in the fallback concatenation); `j >= NVB` is index plane
  `pl = j - NVB` of the packed `qsa_side_planes` buffer (`NPL` planes of 32 x 256 B, then the
  32 x 12 B position page). Cold layout of one page::

      [0, KV)                verbatim buffers, in order
      ANCH_OFF + pl*1024     plane pl: the 128-byte anchors of groups 0..7 (slot 4g+3, bytes 0..127)
      MASK_OFF + pl          plane pl: bit g = group g carries the marker
      POS_OFF                the 8 position entries of tokens 4g (12 B each; plane-0 program)
      ESC_OFF + pl*2*768     plane pl: slot 0 group u - 1 when u >= 1, slot 1 group u when u < 8
                             (u: the lowest unmarked group), raw slots 4e..4e+2

  Decode writes the verbatim buffers, every anchor (bytes 0..127 of slot 4g+3), the marker
  words (MARKER if bit g, else zero), the escaped slots and the 8 position entries. All other
  bytes of the hot side-plane slot are left as they were: no kernel reads them.

Numerics: copies only. Every byte a kernel reads afterwards comes back bitwise; the dead bytes of
a restored page (raw slots of pooled groups, an anchor slot's second half past the marker, the
coordinates of a group's later tokens) are whatever the page held before.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: The 16-byte marker, as four int32 words (all positive, so they pass as plain int constants).
MARKER = (0x7A3C1D55, 0x2E9B4F61, 0x5D0C8E37, 0x13F2A6C9)

INDEX_SLOT_BYTES = 256  # one token of one index plane: 1 head x 128 bf16
GROUPS_PER_PAGE = 8
PLANE_BYTES = 32 * INDEX_SLOT_BYTES  # 8192
POSITION_PAGE_BYTES = 32 * 12  # 384
ANCHOR_BYTES = 128
ESC_GROUP_BYTES = 3 * INDEX_SLOT_BYTES  # 768


def cold_layout(verbatim_bytes: int, num_planes: int) -> dict:
    """Offsets of one cold page (bytes); without side planes, the verbatim buffers concatenated."""
    if num_planes == 0:
        total = verbatim_bytes
        return dict(KV=verbatim_bytes, ANCH_OFF=0, MASK_OFF=0, POS_OFF=0, ESC_OFF=0, NESC=0,
                    END=total, COLD_BYTES=(total + 15) // 16 * 16)  # fmt: skip
    nesc = 2
    anch = verbatim_bytes
    mask = anch + num_planes * GROUPS_PER_PAGE * ANCHOR_BYTES
    pos = mask + 16
    esc = pos + GROUPS_PER_PAGE * 12  # 96 B: stays 16-byte aligned
    end = esc + num_planes * nesc * ESC_GROUP_BYTES
    assert num_planes <= 16 and anch % 16 == 0 and esc % 16 == 0
    return dict(KV=verbatim_bytes, ANCH_OFF=anch, MASK_OFF=mask, POS_OFF=pos, ESC_OFF=esc, NESC=nesc,
                END=end, COLD_BYTES=(end + 4095) // 4096 * 4096)  # fmt: skip


# ---------------------------------------------------------------------- marker stores ---
@triton.jit
def _cold_index_mark_kernel(
    cache32,
    block_table,
    request_indices,
    logical_positions,
    rows,
    stride_page: tl.constexpr,
    stride_token: tl.constexpr,
    bt_stride: tl.constexpr,
    TPB: tl.constexpr,
    CR: tl.constexpr,
    NEXT_CLEAR: tl.constexpr,
    BLOCK_R: tl.constexpr,
    M0: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    M3: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    ok = r < rows
    L = tl.load(logical_positions + r, mask=ok, other=-1).to(tl.int64)
    req = tl.load(request_indices + r, mask=ok, other=0).to(tl.int64)
    valid = ok & (L >= 0)
    page = tl.load(
        block_table + req * bt_stride + tl.where(valid, L, 0) // TPB, mask=valid, other=-1
    ).to(tl.int64)
    valid = valid & (page >= 0)
    is_anchor = valid & ((L + 1) % CR == 0)
    # L == 0 is skipped: a graph-padded row carries L = 0 (an exact no-op of the compress) and must not
    # clear a real request's page-0 marker. Group 0 of a request's first page is cleared by its
    # second row (L == 1) instead, which no padded row carries: a request of 2 or 3 tokens leaves
    # group 0 unanchored, and a stale marker there would hide its raw keys from the encoder.
    is_first = valid & (((L % CR == 0) & (L > 0)) | (L == 1))
    # the anchor row of the row's group (`ahead` rows on): in this call?
    ahead = (CR - 1) - L % CR
    r3 = r + ahead
    ok3 = is_first & (r3 < rows)
    L3 = tl.load(logical_positions + r3, mask=ok3, other=-1).to(tl.int64)
    q3 = tl.load(request_indices + r3, mask=ok3, other=-1).to(tl.int64)
    anchor_here = ok3 & (L3 == L + ahead) & (q3 == req)
    clear_own = is_first & (~anchor_here)

    slot = L % TPB
    w = tl.arange(0, 4)
    magic = tl.where(w == 0, M0, tl.where(w == 1, M1, tl.where(w == 2, M2, M3))).to(tl.int32)
    zero = tl.zeros([BLOCK_R, 4], dtype=tl.int32)
    row_base = page * stride_page
    # set: slot L of anchor rows
    p_set = cache32 + (row_base + slot * stride_token)[:, None] + 32 + w[None, :]
    tl.store(p_set, tl.broadcast_to(magic[None, :], [BLOCK_R, 4]), mask=is_anchor[:, None])
    # clear own: the anchor slot of a first-token row's (or group 0's second row's) group
    p_own = cache32 + (row_base + (slot + ahead) * stride_token)[:, None] + 32 + w[None, :]
    tl.store(p_own, zero, mask=clear_own[:, None])
    if NEXT_CLEAR:
        # clear next (mode 2): slot L + CR of anchor rows whose next group starts on the same page
        # and whose next row L + 1 is not in this call
        r1 = r + 1
        ok1 = is_anchor & (r1 < rows)
        L1 = tl.load(logical_positions + r1, mask=ok1, other=-1).to(tl.int64)
        q1 = tl.load(request_indices + r1, mask=ok1, other=-1).to(tl.int64)
        next_here = ok1 & (L1 == L + 1) & (q1 == req)
        same_page = ((L + CR) // TPB) == (L // TPB)
        clear_next = is_anchor & (~next_here) & same_page
        p_next = cache32 + (row_base + (slot + CR) * stride_token)[:, None] + 32 + w[None, :]
        tl.store(p_next, zero, mask=clear_next[:, None])


def qsa_cold_index_mark(
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    next_clear: bool = True,
) -> None:
    """The marker stores of one anchor writer's call, on its stream behind the compress and the
    e4m3 rounding: the call's rows as the writer got them (graph-padded rows carry `L = 0`, which
    marks nothing)."""
    rows = logical_positions.shape[0]
    if rows == 0:
        return
    assert tokens_per_block % compress_ratio == 0
    cache32 = index_cache.view(torch.int32)
    block_r = 64
    _cold_index_mark_kernel[(triton.cdiv(rows, block_r),)](
        cache32,
        block_table,
        request_indices,
        logical_positions,
        rows,
        cache32.stride(0),
        cache32.stride(1),
        block_table.stride(0),
        TPB=tokens_per_block,
        CR=compress_ratio,
        NEXT_CLEAR=bool(next_clear),
        BLOCK_R=block_r,
        M0=MARKER[0],
        M1=MARKER[1],
        M2=MARKER[2],
        M3=MARKER[3],
        num_warps=2,
    )


# ------------------------------------------------------------------ cold-page codec ---
@triton.jit
def _plane_marks(plane, M0: tl.constexpr, M1: tl.constexpr, M2: tl.constexpr, M3: tl.constexpr):
    """marked[8] of one hot index plane (int32 pointer to its first byte)."""
    g = tl.arange(0, 8)
    w = tl.arange(0, 4)
    mk = tl.load(plane + (g * 4 + 3)[:, None] * 64 + 32 + w[None, :])
    magic = tl.where(w == 0, M0, tl.where(w == 1, M1, tl.where(w == 2, M2, M3))).to(tl.int32)
    hits = tl.sum((mk == magic[None, :]).to(tl.int32), axis=1)
    return hits == 4


@triton.jit
def _copy_verbatim(src_addr, dst_addr, nbytes, VB_CHUNK: tl.constexpr, WORD: tl.constexpr):
    """`nbytes` from `src_addr` to `dst_addr` (integer addresses) in WORD-byte words: 8 where every
    verbatim buffer's base, slot stride and size are multiples of 8, else 1 (the plain fallback
    of a layout the codec declines, whose sizes need not be)."""
    if WORD == 8:
        sp = src_addr.to(tl.pointer_type(tl.int64))
        dp = dst_addr.to(tl.pointer_type(tl.int64))
    else:
        sp = src_addr.to(tl.pointer_type(tl.uint8))
        dp = dst_addr.to(tl.pointer_type(tl.uint8))
    n = (nbytes // WORD).to(tl.int32)
    for kk in range(0, n, VB_CHUNK):
        offs = kk + tl.arange(0, VB_CHUNK)
        m = offs < n
        tl.store(dp + offs, tl.load(sp + offs, mask=m), mask=m)


@triton.jit
def _cold_index_encode_kernel(
    pairs,
    vb_base,
    vb_stride,
    vb_bytes,
    vb_cold,
    side_base,
    side_stride,
    cold_base,
    counters,
    NVB: tl.constexpr,
    NPL: tl.constexpr,
    COLD_BYTES: tl.constexpr,
    ANCH_OFF: tl.constexpr,
    MASK_OFF: tl.constexpr,
    POS_OFF: tl.constexpr,
    ESC_OFF: tl.constexpr,
    NESC: tl.constexpr,
    VB_CHUNK: tl.constexpr,
    WORD: tl.constexpr,
    M0: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    M3: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    dst = tl.load(pairs + 2 * i).to(tl.int64)
    src = tl.load(pairs + 2 * i + 1).to(tl.int64)
    cold = cold_base + dst * COLD_BYTES
    if j < NVB:
        base = tl.load(vb_base + j)
        stride = tl.load(vb_stride + j)
        coff = tl.load(vb_cold + j)
        _copy_verbatim(base + src * stride, cold + coff, tl.load(vb_bytes + j), VB_CHUNK, WORD)
    else:
        pl = j - NVB
        page_addr = side_base + src * side_stride
        plane = (page_addr + pl * 8192).to(tl.pointer_type(tl.int32))
        g = tl.arange(0, 8)
        marked = _plane_marks(plane, M0, M1, M2, M3)
        u = tl.min(tl.where(marked, 8, g))
        maskv = tl.sum(tl.where(marked, 1 << g, 0))
        w32 = tl.arange(0, 32)
        an = tl.load(plane + (g * 4 + 3)[:, None] * 64 + w32[None, :])
        ca = (cold + ANCH_OFF + pl * 1024).to(tl.pointer_type(tl.int32))
        tl.store(ca + g[:, None] * 32 + w32[None, :], an)
        tl.store((cold + MASK_OFF + pl).to(tl.pointer_type(tl.uint8)), maskv.to(tl.uint8))
        k = tl.arange(0, 256)
        km = k < 192
        ce = (cold + ESC_OFF + pl * (NESC * 768)).to(tl.pointer_type(tl.int32))
        # Group u - 1 (the next turn's L - 2 resume can re-pool it) and group u (partial).
        has0 = u >= 1
        e0 = tl.where(has0, u - 1, 0)
        v0 = tl.load(plane + e0 * 256 + k, mask=km & has0)
        tl.store(ce + k, v0, mask=km & has0)
        has1 = u < 8
        e1 = tl.where(has1, u, 0)
        v1 = tl.load(plane + e1 * 256 + k, mask=km & has1)
        tl.store(ce + 192 + k, v1, mask=km & has1)
        plane0 = page_addr.to(tl.pointer_type(tl.int32))
        if pl == 0:
            # the 8 position entries of tokens 4g: position page after the NPL planes
            c = tl.arange(0, 4)
            pm = (c < 3)[None, :] & (g >= 0)[:, None]
            pv = tl.load(plane0 + NPL * 2048 + (g * 12)[:, None] + c[None, :], mask=pm)
            cp = (cold + POS_OFF).to(tl.pointer_type(tl.int32))
            tl.store(cp + (g * 3)[:, None] + c[None, :], pv, mask=pm)
            tl.atomic_add(counters + 0, (u < 8).to(tl.int64))
        else:
            marked0 = _plane_marks(plane0, M0, M1, M2, M3)
            u0 = tl.min(tl.where(marked0, 8, g))
            tl.atomic_add(counters + 1, (u0 != u).to(tl.int64))


@triton.jit
def _cold_index_decode_kernel(
    pairs,
    vb_base,
    vb_stride,
    vb_bytes,
    vb_cold,
    side_base,
    side_stride,
    cold_base,
    NVB: tl.constexpr,
    NPL: tl.constexpr,
    COLD_BYTES: tl.constexpr,
    ANCH_OFF: tl.constexpr,
    MASK_OFF: tl.constexpr,
    POS_OFF: tl.constexpr,
    ESC_OFF: tl.constexpr,
    NESC: tl.constexpr,
    NO_ESCAPE: tl.constexpr,
    VB_CHUNK: tl.constexpr,
    WORD: tl.constexpr,
    M0: tl.constexpr,
    M1: tl.constexpr,
    M2: tl.constexpr,
    M3: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    dst = tl.load(pairs + 2 * i).to(tl.int64)
    src = tl.load(pairs + 2 * i + 1).to(tl.int64)
    cold = cold_base + src * COLD_BYTES
    if j < NVB:
        base = tl.load(vb_base + j)
        stride = tl.load(vb_stride + j)
        coff = tl.load(vb_cold + j)
        _copy_verbatim(cold + coff, base + dst * stride, tl.load(vb_bytes + j), VB_CHUNK, WORD)
    else:
        pl = j - NVB
        page_addr = side_base + dst * side_stride
        plane = (page_addr + pl * 8192).to(tl.pointer_type(tl.int32))
        g = tl.arange(0, 8)
        maskv = tl.load((cold + MASK_OFF + pl).to(tl.pointer_type(tl.uint8))).to(tl.int32)
        marked = ((maskv >> g) & 1) == 1
        u = tl.min(tl.where(marked, 8, g))
        w32 = tl.arange(0, 32)
        ca = (cold + ANCH_OFF + pl * 1024).to(tl.pointer_type(tl.int32))
        an = tl.load(ca + g[:, None] * 32 + w32[None, :])
        tl.store(plane + (g * 4 + 3)[:, None] * 64 + w32[None, :], an)
        w = tl.arange(0, 4)
        magic = tl.where(w == 0, M0, tl.where(w == 1, M1, tl.where(w == 2, M2, M3))).to(tl.int32)
        mw = tl.where(marked[:, None], magic[None, :], 0)
        tl.store(plane + (g * 4 + 3)[:, None] * 64 + 32 + w[None, :], mw)
        if not NO_ESCAPE:
            k = tl.arange(0, 256)
            km = k < 192
            ce = (cold + ESC_OFF + pl * (NESC * 768)).to(tl.pointer_type(tl.int32))
            has0 = u >= 1
            e0 = tl.where(has0, u - 1, 0)
            v0 = tl.load(ce + k, mask=km & has0)
            tl.store(plane + e0 * 256 + k, v0, mask=km & has0)
            has1 = u < 8
            e1 = tl.where(has1, u, 0)
            v1 = tl.load(ce + 192 + k, mask=km & has1)
            tl.store(plane + e1 * 256 + k, v1, mask=km & has1)
        if pl == 0:
            c = tl.arange(0, 4)
            pm = (c < 3)[None, :] & (g >= 0)[:, None]
            cp = (cold + POS_OFF).to(tl.pointer_type(tl.int32))
            pv = tl.load(cp + (g * 3)[:, None] + c[None, :], mask=pm)
            plane0 = page_addr.to(tl.pointer_type(tl.int32))
            tl.store(plane0 + NPL * 2048 + (g * 12)[:, None] + c[None, :], pv, mask=pm)


class ColdIndexTables:
    """Device tables of one lifecycle's layout (built once at configure).

    `verbatim`: `(raw_base, raw_slot_bytes, raw_bytes)` of each buffer copied whole; `side`:
    `(raw_base, raw_slot_bytes)` of the packed side planes, or None for the plain concatenation
    (`num_planes` is then ignored). `counters[0]`: pages that escaped a partial group;
    `counters[1]`: planes whose lowest unmarked group differs from plane 0's."""

    def __init__(self, verbatim, side, num_planes: int, device):
        self.num_vb = len(verbatim)
        self.num_planes = num_planes if side is not None else 0
        vb_bytes = sum(b for _, _, b in verbatim)
        self.layout = cold_layout(vb_bytes, self.num_planes)
        offs, cur = [], 0
        for _, _, b in verbatim:
            offs.append(cur)
            cur += b
        self.vb_base = torch.tensor([a for a, _, _ in verbatim], dtype=torch.int64, device=device)
        self.vb_stride = torch.tensor([s for _, s, _ in verbatim], dtype=torch.int64, device=device)
        self.vb_bytes = torch.tensor([b for _, _, b in verbatim], dtype=torch.int64, device=device)
        self.vb_cold = torch.tensor(offs, dtype=torch.int64, device=device)
        # 8-byte words where every verbatim copy is 8-byte aligned, else bytes (the plain
        # fallback's buffers need not be): either way every byte of a buffer is copied.
        self.word = (
            8 if all(a % 8 == 0 and s % 8 == 0 and b % 8 == 0 for a, s, b in verbatim) else 1
        )
        self.side_base = int(side[0]) if side is not None else 0
        self.side_stride = int(side[1]) if side is not None else 0
        self.counters = torch.zeros(4, dtype=torch.int64, device=device)

    @property
    def cold_page_bytes(self) -> int:
        return self.layout["COLD_BYTES"]

    def _common(self):
        L = self.layout
        return dict(
            NVB=self.num_vb,
            NPL=self.num_planes,
            COLD_BYTES=L["COLD_BYTES"],
            ANCH_OFF=L["ANCH_OFF"],
            MASK_OFF=L["MASK_OFF"],
            POS_OFF=L["POS_OFF"],
            ESC_OFF=L["ESC_OFF"],
            NESC=max(L["NESC"], 1),
            VB_CHUNK=1024,
            WORD=self.word,
            M0=MARKER[0],
            M1=MARKER[1],
            M2=MARKER[2],
            M3=MARKER[3],
            num_warps=4,
        )

    def encode(self, pairs_dev: torch.Tensor, num_pages: int, cold_base: int) -> None:
        if num_pages == 0:
            return
        _cold_index_encode_kernel[(num_pages, self.num_vb + self.num_planes)](
            pairs_dev, self.vb_base, self.vb_stride, self.vb_bytes, self.vb_cold,
            self.side_base, self.side_stride, cold_base, self.counters, **self._common(),
        )  # fmt: skip

    def decode(
        self, pairs_dev: torch.Tensor, num_pages: int, cold_base: int, no_escape: bool = False
    ) -> None:
        """`no_escape` leaves the escaped raw slots out (a negative control for tests)."""
        if num_pages == 0:
            return
        _cold_index_decode_kernel[(num_pages, self.num_vb + self.num_planes)](
            pairs_dev, self.vb_base, self.vb_stride, self.vb_bytes, self.vb_cold,
            self.side_base, self.side_stride, cold_base, NO_ESCAPE=bool(no_escape), **self._common(),
        )  # fmt: skip
