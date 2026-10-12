# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hash ids on the GPU, with the per-request token history in a paged pool.

The Triton kernel ``_engram_hash_kernel`` below is this entry's implementation
(no ``torch.ops`` op computes it). One launch per call: it maps every token of
the step through the compressed-vocab table, stores the compressed id into a
position-indexed paged pool, reads the up-to-``N - 1`` predecessors that lie
before the step from that pool, and emits the int64 hash ids of every n-gram
(2 .. N) ending at each token, for every Engram layer.

The arithmetic is the checkpoint's ``inference/engram.py`` ``NgramHashState``:
cumulative DEAD / sequence-start blocking to ``pad_id``, int64 products with
per-(layer, shift) multipliers, a rolling XOR, and one prime-sized bucket range
per (n-gram size, head) addressed by ``offsets``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_BLOCK_T = 64
_NUM_WARPS = 4
_INT_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _compressed_id(ids_ptr, mask_ptr, map_ptr, vocab_size, idx, ok, HAS_MASK: tl.constexpr):
    """Compressed id of token row ``idx``, or -1 (DEAD) for a masked / out-of-range id."""
    tok = tl.load(ids_ptr + idx, mask=ok, other=-1).to(tl.int64)
    alive = ok & (tok >= 0) & (tok < vocab_size)
    if HAS_MASK:
        alive = alive & (tl.load(mask_ptr + idx, mask=ok, other=0) != 0)
    return tl.load(map_ptr + tok, mask=alive, other=-1).to(tl.int64)


@triton.jit
def _engram_hash_kernel(
    ids_ptr,
    mask_ptr,
    req_ptr,
    cu_ptr,
    start_ptr,
    pool_ptr,
    pool_row_stride,
    bt_ptr,
    bt_row_stride,
    max_blocks,
    page_size,
    map_ptr,
    vocab_size,
    mult_ptr,
    primes_ptr,
    offsets_ptr,
    out_ptr,
    num_tokens,
    pad_id,
    HAS_MASK: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    COLS: tl.constexpr = (N - 1) * H
    t = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    live = t < num_tokens
    r = tl.load(req_ptr + t, mask=live, other=0).to(tl.int64)
    first_row = tl.load(cu_ptr + r, mask=live, other=0).to(tl.int64)
    start = tl.load(start_ptr + r, mask=live, other=0).to(tl.int64)
    off = t - first_row  # offset of the token inside its request's run
    pos = start + off  # absolute position in the request's sequence
    bt_row = bt_ptr + r * bt_row_stride
    heads = tl.arange(0, H)

    # Shift 0: the token itself. Its compressed id (-1 = DEAD) is stored at its
    # position so that later calls can read it as history.
    cur = _compressed_id(ids_ptr, mask_ptr, map_ptr, vocab_size, t, live, HAS_MASK)
    blk = pos // page_size
    page = tl.load(bt_row + blk, mask=live & (blk < max_blocks), other=-1).to(tl.int64)
    tl.store(
        pool_ptr + page * pool_row_stride + pos % page_size,
        cur.to(tl.int32),
        mask=live & (page >= 0),
    )

    for layer in tl.static_range(L):
        blocked = cur < 0
        rolling = tl.where(blocked, pad_id, cur) * tl.load(mult_ptr + layer * N)
        for s in tl.static_range(1, N):
            q = pos - s
            in_run = off >= s
            # predecessor inside this call's run: read from input_ids
            from_run = _compressed_id(
                ids_ptr, mask_ptr, map_ptr, vocab_size, t - s, live & in_run, HAS_MASK
            )
            # predecessor before the run: read from the pool (written by an earlier call)
            hist_ok = live & (off < s) & (q >= 0)
            qblk = q // page_size
            qpage = tl.load(bt_row + qblk, mask=hist_ok & (qblk < max_blocks), other=-1).to(
                tl.int64
            )
            from_hist = tl.load(
                pool_ptr + qpage * pool_row_stride + q % page_size,
                mask=hist_ok & (qpage >= 0),
                other=-1,
            ).to(tl.int64)
            src = tl.where(in_run, from_run, from_hist)
            # once a shift is blocked (sequence start or DEAD) every older shift is too
            blocked = blocked | (q < 0) | (src < 0)
            rolling = rolling ^ (tl.where(blocked, pad_id, src) * tl.load(mult_ptr + layer * N + s))
            prime = tl.load(primes_ptr + layer * COLS + (s - 1) * H + heads)
            offset = tl.load(offsets_ptr + layer * COLS + (s - 1) * H + heads)
            val = rolling[:, None] % prime[None, :] + offset[None, :]
            tl.store(
                out_ptr + t[:, None] * (L * COLS) + layer * COLS + (s - 1) * H + heads[None, :],
                val,
                mask=live[:, None],
            )


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"engram_hash: {msg}")


def _check_1d(name: str, x: torch.Tensor, n: Optional[int], dtypes) -> None:
    _check(
        x.dim() == 1 and x.stride(0) == 1,
        f"{name} must be a 1-D tensor with unit stride, got shape {tuple(x.shape)} stride {x.stride()}",
    )
    _check(x.dtype in dtypes, f"{name} dtype must be one of {dtypes}, got {x.dtype}")
    if n is not None:
        _check(x.shape[0] == n, f"{name} must have {n} elements, got {x.shape[0]}")


def _launch_engram_hash(
    input_ids: torch.Tensor,
    req_idx_per_token: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    start_pos: torch.Tensor,
    token_pool: torch.Tensor,
    block_table: torch.Tensor,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    token_mask: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Validate (``ValueError``, prefixed ``engram_hash:``), allocate ``out`` when absent, launch once."""
    device = input_ids.device
    _check(input_ids.is_cuda, "input_ids must be a CUDA tensor")
    _check_1d("input_ids", input_ids, None, _INT_DTYPES)
    T = input_ids.shape[0]
    _check_1d("req_idx_per_token", req_idx_per_token, T, _INT_DTYPES)
    _check_1d("start_pos", start_pos, None, _INT_DTYPES)
    B = start_pos.shape[0]
    _check_1d("cu_seq_lens", cu_seq_lens, B + 1, _INT_DTYPES)
    _check_1d("token_map", token_map, None, _INT_DTYPES)
    _check(
        token_pool.dim() == 2 and token_pool.stride(1) == 1,
        "token_pool must be a [pages, page_size] tensor with unit column stride",
    )
    _check(
        token_pool.dtype == torch.int32,
        f"token_pool dtype must be torch.int32, got {token_pool.dtype}",
    )
    _check(
        block_table.dim() == 2 and block_table.stride(1) == 1,
        "block_table must be a [>=B, max_blocks] tensor with unit column stride",
    )
    _check(
        block_table.shape[0] >= B,
        f"block_table must have at least B={B} rows, got {block_table.shape[0]}",
    )
    _check(
        block_table.dtype in _INT_DTYPES,
        f"block_table dtype must be int32 or int64, got {block_table.dtype}",
    )
    _check(
        multipliers.dim() == 2 and multipliers.is_contiguous() and multipliers.dtype == torch.int64,
        "multipliers must be a contiguous int64 [L, N] tensor",
    )
    L, N = multipliers.shape
    _check(N >= 2, f"multipliers must have N >= 2 columns (max n-gram size), got {N}")
    _check(
        primes.dim() == 3 and primes.is_contiguous() and primes.dtype == torch.int64,
        "primes must be a contiguous int64 [L, N - 1, H] tensor",
    )
    _check(
        primes.shape[0] == L and primes.shape[1] == N - 1,
        f"primes must be [L={L}, N-1={N - 1}, H], got {tuple(primes.shape)}",
    )
    H = primes.shape[2]
    _check(H >= 1 and (H & (H - 1)) == 0, f"the head count H must be a power of two, got {H}")
    _check(
        offsets.is_contiguous()
        and offsets.dtype == torch.int64
        and tuple(offsets.shape) == (L, (N - 1) * H),
        f"offsets must be a contiguous int64 [{L}, {(N - 1) * H}] tensor",
    )
    if token_mask is not None:
        _check_1d("token_mask", token_mask, T, (torch.bool,))
    if out is None:
        out = torch.empty((T, L, (N - 1) * H), dtype=torch.int64, device=device)
    else:
        _check(
            out.is_contiguous()
            and out.dtype == torch.int64
            and tuple(out.shape) == (T, L, (N - 1) * H),
            f"out must be a contiguous int64 [{T}, {L}, {(N - 1) * H}] tensor",
        )
    for name, x in (
        ("req_idx_per_token", req_idx_per_token),
        ("cu_seq_lens", cu_seq_lens),
        ("start_pos", start_pos),
        ("token_pool", token_pool),
        ("block_table", block_table),
        ("token_map", token_map),
        ("multipliers", multipliers),
        ("primes", primes),
        ("offsets", offsets),
        ("out", out),
    ):
        _check(x.device == device, f"{name} is on {x.device}, input_ids on {device}")
    if token_mask is not None:
        _check(
            token_mask.device == device,
            f"token_mask is on {token_mask.device}, input_ids on {device}",
        )
    if T == 0:
        return out
    _engram_hash_kernel[(triton.cdiv(T, _BLOCK_T),)](
        input_ids,
        token_mask if token_mask is not None else input_ids,
        req_idx_per_token,
        cu_seq_lens,
        start_pos,
        token_pool,
        token_pool.stride(0),
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        token_pool.shape[1],
        token_map,
        token_map.shape[0],
        multipliers,
        primes,
        offsets,
        out,
        T,
        pad_id,
        HAS_MASK=token_mask is not None,
        L=L,
        N=N,
        H=H,
        BLOCK_T=_BLOCK_T,
        num_warps=_NUM_WARPS,
    )
    return out


#: DeepSeek-V4.1's (layers, max n-gram size, heads): the shapes of the constant tables.
_V41_LNH = dict(L=2, N=4, H=8)

#: Prompt lengths driven as one-shot prefill runs in one call.
_PREFILL_LENS = ((1,), (2,), (3,), (4,), (5, 17, 64, 129, 1000), (8192,))


class EngramHash(OpWrapper):
    """Engram n-gram hash ids of one step's tokens, with each request's history in a paged pool.

    One launch. A call carries ``B`` runs, one per request, concatenated in batch
    order: token row ``t`` belongs to run ``b = req_idx_per_token[t]`` at absolute
    position ``p = start_pos[b] + t - cu_seq_lens[b]``. Inside the call: the
    token-map lookup (DEAD = -1 where ``token_mask`` is False or the id is outside
    ``[0, token_map.numel())``), the store of every token's compressed id at
    ``token_pool[block_table[b, p // P], p % P]`` (``P = token_pool.shape[1]``;
    skipped where the page entry is negative), the read of each shift-``s``
    predecessor from ``input_ids`` inside the run or from the pool before it,
    cumulative blocking to ``pad_id`` at the sequence start and at DEAD tokens,
    and the hashes for every Engram layer::

        r_0 = tok_0 * multipliers[l, 0];   r_s = r_{s-1} XOR tok_s * multipliers[l, s]
        out[t, l, (n - 2) * H + h] = r_{n-1} % primes[l, n - 2, h] + offsets[l, (n - 2) * H + h]

    all in int64. The caller owns the constant tables, the pool and its block
    table, ``token_mask``, and everything after the ids (table gather, the
    reduction across row shards, the projection and the gate). Returns the int64
    ``[T, L, (N - 1) * H]`` hash ids (``out`` when given); ``token_pool`` is
    updated in place. ``T = 0`` returns without a launch.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"prefill {'/'.join(map(str, lens))} P={p}: one-shot runs from 0",
                spec=dict(
                    kind="prefill", lens=lens, page_size=p, index_dtype=torch.int32, **_V41_LNH
                ),
            )
            for lens in _PREFILL_LENS
            for p in (128, 256)
        )
        + tuple(
            Cell(
                why=f"chunked+decode P={p} {str(dt).removeprefix('torch.')}: six requests",
                spec=dict(kind="chunked", page_size=p, index_dtype=dt, **_V41_LNH),
            )
            for p, dt in (
                (128, torch.int32),
                (256, torch.int32),
                (3, torch.int32),
                (128, torch.int64),
            )
        )
        + tuple(
            Cell(
                why=f"verify D={d}: runs of {d + 1}, random acceptance, rewind",
                spec=dict(
                    kind="verify", drafts=d, page_size=128, index_dtype=torch.int32, **_V41_LNH
                ),
            )
            for d in (1, 4, 5)
        )
        + (
            Cell(
                why="DEAD via token_mask spans and out-of-range ids",
                spec=dict(kind="dead", page_size=128, index_dtype=torch.int32, **_V41_LNH),
            ),
            Cell(
                why="windowed table: unmapped / shared scratch pages",
                spec=dict(kind="windowed", page_size=128, index_dtype=torch.int32, **_V41_LNH),
            ),
            Cell(
                why="T=0: empty call, no launch, pool untouched",
                spec=dict(kind="empty", page_size=128, index_dtype=torch.int32, **_V41_LNH),
            ),
        )
    )

    note = """
    Metadata consumed: none; every input is an argument. The DeepSeek-V4
    attention metadata carries req_idx_per_token, cu_seq_lens_cuda and
    cached_token_lens_cuda (as start_pos) in this convention; the pool and its
    block table are the cache manager's ENGRAM_TOKEN role
    (metadata.engram_token_block_table, tokens_per_block = P).

    Constants, built once at load from the checkpoint recipe: token_map is
    engram.py's build_compressed_token_map over the tokenizer (129280 ids -> 99092
    classes); pad_id is the compressed id token_map[2] (equal to 2 for this
    tokenizer, so passing the raw id is invisible on real data and wrong on any
    other map); multipliers[l] are odd, numpy PCG64 seeded 10007 * layer_id with
    the layer ids (1, 14), not their index; primes are 48 distinct, in layer ->
    n-gram -> head order, each the next unused prime above 15999999; offsets[l]
    is the exclusive cumulative sum of the layer's 24 primes. (L, N, H) =
    (2, 4, 8). int64 end to end: the largest product, 99091 * 82619226485591,
    lies between 2^62 and 2^63.

    History: the pool holds one int32 compressed id per position (-1 = DEAD).
    The op writes every position its runs cover and reads only
    [start_pos - N + 1, start_pos) of each run, so chunked prefill, decode and
    speculative verify are the same launch: a verify run writes all its
    positions, the next run starts after the accepted ones, and the rejected
    tail is overwritten before it is read (no commit call). The op cannot create
    history it never saw: a run past position 0 whose N - 1 predecessors never
    went through the op in this pool (a prefix-cache hit, a disaggregated
    hand-off, an encoder bounded replay) reads whatever those slots hold and is
    wrong without an error. The caller stores them first with a call whose runs
    are only those positions; empty runs are allowed and their start_pos and
    block-table row are not read.

    Preconditions the op does not check: req_idx_per_token, cu_seq_lens and
    start_pos agree (each run's rows contiguous, positions consecutive); row b
    of block_table maps every page covering [start_pos[b] - N + 1, start_pos[b])
    and the positions later calls will read (other pages may be unmapped or a
    shared scratch page); no run's read pages are written by the same call; no
    request appears twice in one call. It raises ValueError, prefixed
    "engram_hash:", on a wrong dtype, device, rank, stride or length of any
    tensor and on H not a power of two, before writing anything.

    Launch: fixed, 64 tokens per program, 4 warps, grid ceil(T / 64), no host
    synchronization (CUDA-graph capturable). B is not a launch argument. Triton
    specializes num_tokens (1, divisible by 16, other), the block-table width
    and row stride, the mask and the index dtypes.

    Measured on sm_103 beyond the cells, by the entry's test: every prefix of a
    mixed 16384-token step; the engine envelope (B <= 256 runs, T <= 16384 rows,
    positions below max_seq_len + 14 with DSpark for max_seq_len up to
    1048576, at the engine's block-table widths 132 .. 8196), unmasked and with
    the multimodal target's keep mask on every context run; CUDA-graph capture
    of decode (T = G) and verify (T = 6 G) steps at the 46 graph batches at
    every max_seq_len; ENGRAM_TOKEN pools up to [57999360, 128] (element
    offsets past 2^32); block tables 128 .. 8196 columns, row stride = width or
    width + 5; the encoder-replay history call with empty runs; bit identity
    with the checkpoint's NgramHashState and the golden hash ids.

    Not certified: page sizes other than 128, 256 and 3; T > 16384, B > 256 or
    positions at or past 1048590; table widths or capture batches other than
    those run; pools larger than [57999360, 128]; other (L, N, H); a masked call
    under CUDA-graph capture; a False mask entry on a generation row; empty runs
    in a masked, captured or int64 call; a block table narrower than a run's
    last logical page + 1.
    """

    def raw_call(
        self,
        input_ids: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        start_pos: torch.Tensor,
        token_pool: torch.Tensor,
        block_table: torch.Tensor,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
        token_mask: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return _launch_engram_hash(
            input_ids,
            req_idx_per_token,
            cu_seq_lens,
            start_pos,
            token_pool,
            block_table,
            token_map,
            multipliers,
            primes,
            offsets,
            pad_id,
            token_mask,
            out,
        )

    def reference(
        self,
        input_ids: torch.Tensor,
        req_idx_per_token: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        start_pos: torch.Tensor,
        token_pool: torch.Tensor,
        block_table: torch.Tensor,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
        primes: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
        token_mask: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The hash ids, from the checkpoint's ``NgramHashState.forward`` in torch int64 ops.

        A documented exception to a pure-argument reference, like the paged-cache
        entries: a row's predecessors before its run are what the pool holds, and
        the arguments only say where to look. ``token_pool`` is read at
        ``block_table[b, q // P], q % P`` for those positions (an unmapped page
        or a column past the table reads DEAD) and must be passed as it was
        before the call -- the op writes it. Nothing is written here; ``out`` is
        ignored. The pool writes themselves are certified by the test's
        cross-call model, which carries each request's history independently of
        the pool.
        """
        ids = input_ids.long()
        t_len = ids.numel()
        n_max = multipliers.shape[1]
        page_size = token_pool.shape[1]
        max_blocks = block_table.shape[1]
        vocab = token_map.numel()
        rows = torch.arange(t_len, device=ids.device)
        req = req_idx_per_token.long()
        off = rows - cu_seq_lens.long()[req]
        pos = start_pos.long()[req] + off
        table = block_table.long()

        comp = torch.where(
            (ids >= 0) & (ids < vocab), token_map.long()[ids.clamp(0, vocab - 1)], -1
        )
        if token_mask is not None:
            comp = torch.where(token_mask, comp, -1)
        blocked = comp < 0
        toks = [torch.where(blocked, pad_id, comp)]
        for s in range(1, n_max):
            q = pos - s
            in_run = off >= s
            from_run = comp[(rows - s).clamp_min(0)]
            qc = q.clamp_min(0)
            blk = torch.div(qc, page_size, rounding_mode="floor")
            readable = ~in_run & (q >= 0) & (blk < max_blocks)
            page = torch.where(readable, table[req, blk.clamp_max(max_blocks - 1)], -1)
            stored = token_pool[page.clamp_min(0), qc % page_size].long()
            from_hist = torch.where(page >= 0, stored, -1)
            src = torch.where(in_run, from_run, from_hist)
            blocked = blocked | (q < 0) | (src < 0)
            toks.append(torch.where(blocked, pad_id, src))
        tok = torch.stack(toks, dim=-1)  # [T, N]
        products = tok.unsqueeze(1) * multipliers.long()  # [T, L, N]
        rolling, hashes = products[..., 0], []
        for s in range(1, n_max):
            rolling = torch.bitwise_xor(rolling, products[..., s])
            hashes.append(rolling.unsqueeze(-1) % primes.long()[:, s - 1])
        return torch.cat(hashes, dim=-1) + offsets.long()

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bit for bit: integer ids, no rounding anywhere on either side.

        Any difference is a wrong token, a wrong predecessor, a wrong block or a
        wrong product. The wrong variants the test runs through the same check --
        ``pad_id`` 2 instead of ``token_map[2]``, int32 or float64 products,
        non-cumulative blocking, multipliers seeded by layer index, history not
        carried, the page of ``start_pos - 1`` unmapped -- each change rows.
        """
        assert out.dtype == ref.dtype == torch.int64, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        rows = (out != ref).flatten(1).any(1)
        bad = int(rows.sum())
        assert bad == 0, f"{bad} of {out.shape[0]} rows differ"


engram_hash = EngramHash()
