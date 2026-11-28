# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hash ids of a step from the generation targets' per-request look-back ring, in two launches (decode
sizes): `_engram_hash_kernel` (one program per (layer, order, head) over all tokens of the step: each program does one
int64 modulus per token instead of one thread doing all of them) reads the ring, then `_engram_ring_write_kernel` (one
program) writes the step's ids into it; stream order puts every read before the writes.

The plain-torch form (`modeling._EngramHash.lookback`, kept for steps wider than `MAX_ROWS`) is ~40 single-block int64
launches (index, compare, where, xor, mod); the engine's own decode hash (`kernels/blackwell/engram/decode_hash.py`)
reads a full-length per-request history, which these targets (dep and tep8) do not keep."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# each program holds the whole step's rows: up to 2048 (eight warps past 1024 rows); wider steps keep the plain-torch
# look-back
MAX_ROWS = 2048


@triton.jit
def _floor_mod(x, p):
    r = x % p
    return tl.where(r < 0, r + p, r)


@triton.jit
def _compressed(ids_ptr, token_map_ptr, j, ok, vocab):
    t = tl.load(ids_ptr + j, mask=ok, other=0).to(tl.int64)
    t = tl.minimum(tl.maximum(t, 0), vocab - 1)
    return tl.load(token_map_ptr + t, mask=ok, other=0)


@triton.jit
def _slot(tok_req_ptr, req_slot_ptr, j, ok):
    return tl.load(req_slot_ptr + tl.load(tok_req_ptr + j, mask=ok, other=0), mask=ok, other=-1).to(
        tl.int64
    )


@triton.jit
def _engram_hash_kernel(
    ids_ptr,  # [n] token ids (int32 / int64)
    pos_ptr,  # [n] int32 absolute positions
    tok_req_ptr,  # [n] int64 request row of each token
    req_slot_ptr,  # [R] int32 ring slot of each request row
    token_map_ptr,  # [V] int64 compressed vocabulary
    hist_ptr,  # [S + 1, RING] int32 look-back ring (read only here)
    mults_ptr,  # [L, MAX_N] int64
    primes_ptr,  # [L, MAX_N - 1, H] int64
    out_ptr,  # [n, L, (MAX_N - 1) * H] int64, or [L, n, (MAX_N - 1) * H] with LAYER_MAJOR
    n,
    vocab,
    pad_id,
    RING: tl.constexpr,
    MAX_N: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    LAYER_MAJOR: tl.constexpr,
):
    pid = tl.program_id(0)  # (layer, order - 1, head)
    head = pid % H
    order = (pid // H) % (MAX_N - 1) + 1
    layer = pid // (H * (MAX_N - 1))
    r = tl.arange(0, BLOCK)
    ok = r < n
    comp = _compressed(ids_ptr, token_map_ptr, r, ok, vocab)
    pos = tl.load(pos_ptr + r, mask=ok, other=0).to(tl.int64)
    slot = _slot(tok_req_ptr, req_slot_ptr, r, ok)
    rolling = comp * tl.load(mults_ptr + layer * MAX_N)
    for back in tl.static_range(1, MAX_N):
        if back <= order:
            # the step's own row when it is the same request's previous position, else the ring; pad before 0
            pb = pos - back
            j = tl.maximum(r - back, 0)
            inside = (
                (r >= back)
                & (_slot(tok_req_ptr, req_slot_ptr, j, ok) == slot)
                & (tl.load(pos_ptr + j, mask=ok, other=-1).to(tl.int64) == pb)
            )
            cj = _compressed(ids_ptr, token_map_ptr, j, ok & inside, vocab)
            ring = tl.load(
                hist_ptr + slot * RING + _floor_mod(pb, RING),
                mask=ok & (pb >= 0) & ~inside,
                other=0,
            ).to(tl.int64)
            v = tl.where(pb < 0, pad_id, tl.where(inside, cj, ring))
            rolling = rolling ^ (v * tl.load(mults_ptr + layer * MAX_N + back))
    p = tl.load(primes_ptr + (layer * (MAX_N - 1) + (order - 1)) * H + head)
    COLS: tl.constexpr = (MAX_N - 1) * H
    if LAYER_MAJOR:  # each layer's ids one contiguous [n, cols] block
        row = (layer * n + r) * COLS
    else:
        row = r * (L * COLS) + layer * COLS
    tl.store(out_ptr + row + (order - 1) * H + head, _floor_mod(rolling, p), mask=ok)


@triton.jit
def _engram_ring_write_kernel(
    ids_ptr,
    pos_ptr,
    tok_req_ptr,
    req_slot_ptr,
    token_map_ptr,
    hist_ptr,
    n,
    vocab,
    dummy,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Only each request's last RING tokens of the step are written (the others go to the dummy slot)."""
    r = tl.arange(0, BLOCK)
    ok = r < n
    comp = _compressed(ids_ptr, token_map_ptr, r, ok, vocab)
    pos = tl.load(pos_ptr + r, mask=ok, other=0).to(tl.int64)
    slot = _slot(tok_req_ptr, req_slot_ptr, r, ok)
    later = tl.minimum(r + RING, tl.maximum(n - 1, 0))
    s_l = _slot(tok_req_ptr, req_slot_ptr, later, ok)
    p_l = tl.load(pos_ptr + later, mask=ok, other=-1).to(tl.int64)
    last = (r + RING >= n) | (s_l != slot) | (p_l != pos + RING)
    dst = tl.where(last, slot, dummy)
    tl.store(hist_ptr + dst * RING + _floor_mod(pos, RING), comp.to(tl.int32), mask=ok)


class EngramLookbackHash(OpWrapper):
    """`-> [n, L, (max_n - 1) * H]` int64 bucket-local hash ids; also writes the step's compressed ids into `hist`.
    With `layer_major` the ids are stored layer-major (a `[L, n, cols]` buffer, returned as its `[n, L, cols]`
    transpose): each Engram layer's `ids[:, l]` is contiguous, as the tep8 target's table lookup reads it (no
    per-layer copy); the values are the same either way.

    Token `r` (request row `tok_req[r]`, ring slot `s = req_slot[tok_req[r]]`, position `p`): `c0 = token_map[ids[r]]`
    (ids clamped to the map, as the engine's decode hash does);
    look-back `c_b` (b = 1..max_n-1) is the compressed id at `p - b`: the step's own row `r - b` when that row is the
    same slot at `p - b`, else `hist[s, (p - b) % ring]`, and the pad id when `p - b < 0`. Hash of layer `l`, order `i`,
    head `h`: `(XOR_{b <= i} c_b * mults[l, b]) mod primes[l, i - 1, h]` (int64 wrap-around products, non-negative
    remainder). After every read, `hist[s, p % ring] = c0` for the last `ring` tokens of each request in the step; the
    others write the dummy row `hist.shape[0] - 1`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8 c7000 decode: 40 requests of 1 + 3 draft tokens",
            spec=dict(reqs=40, per=4, start=900),
        ),
        Cell(
            why="dep32 decode: 25 requests of 1 + 5 draft tokens",
            spec=dict(reqs=25, per=6, start=3000),
        ),
        Cell(
            why="dep8 c7000 full step: 320 requests of 1 + 3 draft tokens (1280 rows)",
            spec=dict(reqs=320, per=4, start=5000),
        ),
        Cell(
            why="the widest step a program holds: 512 requests of 4 rows (2048)",
            spec=dict(reqs=512, per=4, start=700),
        ),
        Cell(
            why="tep8 decode: one request, 1 + 5 draft tokens, deep in its history",
            spec=dict(reqs=1, per=6, start=900),
        ),
        Cell(
            why="request start: look-back positions before 0 read the pad id",
            spec=dict(reqs=1, per=6, start=1),
        ),
        Cell(
            why="four requests of 6 rows, two of them in the dummy slot",
            spec=dict(reqs=4, per=6, start=200, dummy=2),
        ),
        Cell(
            why="a 40-row chunk: longer than the ring, only its last 16 rows are written",
            spec=dict(reqs=1, per=40, start=3),
        ),
        Cell(
            why="negative ids (-1, -2) read the compressed map's first entry",
            spec=dict(reqs=4, per=6, start=200, neg=True),
        ),
    )
    note = "Exact (integer arithmetic): the test gates hash ids and the ring bit for bit against `reference`."

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tok_req: torch.Tensor,
        req_slot: torch.Tensor,
        token_map: torch.Tensor,
        hist: torch.Tensor,
        mults: torch.Tensor,
        primes: torch.Tensor,
        pad_id: int,
        layer_major: bool = False,
    ) -> torch.Tensor:
        n = input_ids.shape[0]
        layers, max_n = mults.shape
        heads = primes.shape[-1]
        shape = (layers, n) if layer_major else (n, layers)
        out = torch.empty(*shape, (max_n - 1) * heads, dtype=torch.int64, device=input_ids.device)
        if n:
            block = max(16, triton.next_power_of_2(n))
            _engram_hash_kernel[(layers * (max_n - 1) * heads,)](
                input_ids, positions, tok_req, req_slot, token_map, hist, mults, primes, out,
                n, token_map.shape[0], pad_id,
                RING=hist.shape[1], MAX_N=max_n, L=layers, H=heads, BLOCK=block, LAYER_MAJOR=layer_major,
                num_warps=4 if block <= 1024 else 8,
            )  # fmt: skip
            _engram_ring_write_kernel[(1,)](
                input_ids, positions, tok_req, req_slot, token_map, hist, n, token_map.shape[0], hist.shape[0] - 1,
                RING=hist.shape[1], BLOCK=block, num_warps=4 if block <= 1024 else 8,
            )  # fmt: skip
        return out.transpose(0, 1) if layer_major else out

    def reference(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tok_req: torch.Tensor,
        req_slot: torch.Tensor,
        token_map: torch.Tensor,
        hist: torch.Tensor,
        mults: torch.Tensor,
        primes: torch.Tensor,
        pad_id: int,
        layer_major: bool = False,
    ) -> torch.Tensor:
        """The target's plain-torch look-back (`_EngramHash.lookback`), statement for statement (the values: the
        layout `layer_major` picks is the test's to check)."""
        ring = hist.shape[1]
        max_n = mults.shape[1]
        slot = req_slot[tok_req].long()
        # out-of-range ids read the map's ends, as the kernel and the engine's decode hash clamp them
        comp = token_map[input_ids.long().clamp(0, token_map.shape[0] - 1)]
        pos = positions.long()
        n = comp.shape[0]
        rows = torch.arange(n, device=comp.device)
        cols = [comp]
        for back in range(1, max_n):
            pb = pos - back
            j = (rows - back).clamp_min(0)
            inside = (rows >= back) & (slot[j] == slot) & (pos[j] == pb)
            v = torch.where(inside, comp[j], hist[slot, pb % ring].long())
            cols.append(torch.where(pb < 0, torch.full_like(v, pad_id), v))
        later = (rows + ring).clamp_max(max(n - 1, 0))
        last = (rows + ring >= n) | (slot[later] != slot) | (pos[later] != pos + ring)
        hist[torch.where(last, slot, hist.shape[0] - 1), pos % ring] = comp.to(hist.dtype)
        toks = torch.stack(cols, dim=-1)
        prods = toks.unsqueeze(1) * mults
        rolling, hashes = prods[..., 0], []
        for i in range(1, max_n):
            rolling = torch.bitwise_xor(rolling, prods[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
        return torch.cat(hashes, dim=-1)

    def is_valid(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tok_req: torch.Tensor,
        req_slot: torch.Tensor,
        token_map: torch.Tensor,
        hist: torch.Tensor,
        mults: torch.Tensor,
        primes: torch.Tensor,
        pad_id: int,
        layer_major: bool = False,
    ) -> None:
        n = input_ids.shape[0]
        assert n <= MAX_ROWS, (
            "a program holds the step's rows: wider steps take the plain-torch look-back"
        )
        assert (
            positions.dtype == torch.int32
            and positions.shape == (n,)
            and tok_req.dtype == torch.int64
        )
        assert req_slot.dtype == torch.int32 and token_map.dtype == torch.int64
        assert hist.dtype == torch.int32 and hist.is_contiguous()
        assert (
            mults.dtype == primes.dtype == torch.int64
            and mults.is_contiguous()
            and primes.is_contiguous()
        )
        assert primes.shape[:2] == (mults.shape[0], mults.shape[1] - 1)
        assert all(t.is_contiguous() for t in (input_ids, positions, tok_req, req_slot, token_map))


engram_lookback_hash = EngramLookbackHash()
