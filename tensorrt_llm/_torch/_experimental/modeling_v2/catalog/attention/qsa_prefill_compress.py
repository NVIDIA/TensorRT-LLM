# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA block compression of the paged index-key side cache: completed groups to anchors.

Shared by the qwen4_exp tp1 target
(`qwen3_8_flash_next_nvfp4__sm_103__tp1`) and the unified tp4/tep4/dep4 target
(`hf_fc694b5_orig__sm_103__unified`);
promoted from their copies.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# Tier-2 sourcing: a vendored Triton callable reached as ordinary Python. QSA's
# kernel plane has no torch.ops surface.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

from .._op import Arch, Cell, OpWrapper

#: The mRoPE sections the checkpoint declares; the target passes `None` today
#: (`_QSA_MROPE_SECTION`) and these are certified, on the NVFP4 trunk-plane cell,
#: for when it flips.
MROPE_SECTION = (11, 11, 10)

#: This checkpoint's index geometry and the engine's QSA metadata: index head
#: 128, rotary width 64 (32 cos/sin pairs), compress ratio 4, rms eps 1e-6,
#: theta 1e7; `slots` x `width` is `qsa_block_table`. `pages` is the pool.
_CKPT = dict(
    head_dim=128,
    tokens_per_block=32,
    slots=32,
    width=260,
    compress_ratio=4,
    eps=1e-6,
    mrope_section=None,
    logical_dtype=torch.int64,
    request_dtype=torch.int32,
    layout="plain",
)

#: 32 requests of one MTP7 verify block each (8 rows, two completed groups), at starts that put
#: some blocks across a page edge and some inside one.
_MTP7_32 = tuple((29 + 61 * i, 8) for i in range(32))


#: The unified target's index pool at bf16 KV, as the stock `Qwen4ExpHybridCacheManager` lays it out on
#: every member and rank: KV cache manager V2 coalesces the 12 sparse layers' same-size index
#: pages into one pool, so a layer's `get_index_k_buffer` view is `[pages, 32, 1, 128]` with page
#: stride 12 x 4096 elements, starting `ordinal x 4096` into each slot. The position plane is its
#: own contiguous `[pages, 32, 3]` int32 pool.
INDEX_ROLES = 12

#: A rank's attention pool, in pages: the stock engine's `maxNumBlocks` at core config (bf16 KV,
#: fraction 0.6, host-resident n-gram table), measured per member. The manager's index view holds
#: slightly fewer pages than its primary block count, so these bound it from above.
TEP4_PAGES = 258983
DEP4_PAGES = 120530

#: `qsa_block_table` is `[max_num_sequences, max_blocks_per_seq]`: 128 seats (`max_batch_size`, per
#: rank under attention DP), 292 pages a sequence at max_seq_len 9216 and 516 at the MTP member's 16384.
SEATS = 128
CORE_WIDTH = 292
MTP_WIDTH = 516


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


def _mrope(segments, pages, **overrides) -> dict:
    """Prefill from 0 under mRoPE (11, 11, 10)."""
    return _spec(segments, pages, mrope_section=MROPE_SECTION, **overrides)


def _pool(segments, pages, width, ordinal, **overrides) -> dict:
    """The unified target's call over a rank's coalesced index pool: layer `ordinal`'s plane, 128 seats,
    and request 0's last page the pool's last (`top_page`)."""
    return _spec(
        segments,
        pages,
        slots=SEATS,
        width=width,
        layout="coalesced",
        roles=INDEX_ROLES,
        ordinal=ordinal,
        top_page=True,
        **overrides,
    )


def _decode(requests: int, last: int) -> tuple:
    """`requests` decode rows spread over positions 3..`last`; every other one an anchor
    (`4k + 3`), the others at `4k + 0..2`, the last request an anchor at `last`."""
    span = (last - 3) // 4
    segments = []
    for r in range(requests):
        k = r * span // (requests - 1)
        offset = 3 if r % 2 == 0 or r == requests - 1 else r % 3
        segments.append((4 * k + offset, 1))
    return tuple(segments)


#: 128 MTP7 verify blocks (8 rows, two anchors each) at starts spread to the MTP member's last
#: position, 16383: some blocks inside one page, some across a page edge.
_MTP7_128 = tuple((r * (16384 - 8) // 127, 8) for r in range(128))


class QsaPrefillCompress(OpWrapper):
    """Replace each completed group's anchor slot with its pooled, normed, rotated key.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_prefill_compress`: one Triton launch, grid `(rows,)`. A row
    with `(L + 1) % cr == 0` (L = its logical position) is the anchor of the
    group `L - cr + 1 .. L`; every other row does nothing. Each member is
    resolved through `block_table` on its own, so groups may straddle pages::

        pooled = bf16(sum_f32(raw keys of the group) / cr)
        y      = f32(pooled) * rsqrt(mean(f32(pooled)^2) + eps) * (1 + k_norm_weight)
        y      = cos_sin.dtype(y)
        (t, h, w) = position_cache at the group's FIRST member
        pair q takes h if q % 3 == 1 and q < 3 * s1, w if q % 3 == 2 and
          q < 3 * s2, else t   (t for every pair when mrope_section is None)
        NeoX-rotate y[:D/2] by cos_sin[pos[q], q], cos_sin[pos[q], R + q];
          y[D/2:] passes through (R = cos_sin.shape[1] // 2 = D / 4)
        index_cache[anchor] = bf16(y)

    Returns `index_cache`, written in place at anchor slots only; everything
    else, including the non-anchor members' raw keys, is untouched.

    Not idempotent: the anchor slot is the group's last input and its only
    output, so a second call over the same rows re-pools a compressed anchor
    (measured: all 48 of 48 anchors moved, up to 2.94).

    Fusion boundary: pool, Gemma norm, position gather, rotation, store. The
    raw-key and position stores (`qsa_paged_index_store`, which must run
    first), the table and metadata, and the main K/V pool are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="prefill 2 x 128 from 0, mrope_section None: the target's call",
            spec=_spec(((0, 128),) * 2, pages=8),
        ),
        Cell(
            why="decode MTP7: 32 requests x 8 rows, two anchors each, None",
            spec=_spec(_MTP7_32, pages=1100, slots=40, rounding_flips=1),
        ),
        # Under an NVFP4 KV pool the position page shares each physical page with the index
        # pages, so both caches are strided views of one allocation: index page stride
        # 12 x 4096 + 192 = 49344 on the trunk, position strides (24672, 3, 1).
        Cell(
            why="NVFP4 trunk planes: 12 index roles and the position page share each page",
            spec=_mrope(((0, 64),) * 2, pages=4, layout="shared_page", roles=12),
        ),
        # The unified target's calls over a rank's measured pool, every page reachable: the last
        # page's plane sits past 2^31 elements of the pool's base.
        Cell(
            why="tep4 decode: 128 requests x 1 row, the 258983-page coalesced pool, ordinal 11",
            spec=_pool(_decode(128, 9215), TEP4_PAGES, CORE_WIDTH, 11),
        ),
        Cell(
            why="tep4 MTP7 verify: 128 requests x 8 rows = 1024, positions to 16383",
            spec=_pool(_MTP7_128, TEP4_PAGES, MTP_WIDTH, 5),
        ),
        Cell(
            why="tep4 prefill at max_num_tokens: 8192 rows (6144 + 2047 + 1, all from 0)",
            spec=_pool(((0, 6144), (0, 2047), (0, 1)), TEP4_PAGES, CORE_WIDTH, 0, rounding_flips=5),
        ),
        Cell(
            why="dep4 decode: 32 requests x 1 row over a rank's 120530-page pool, 128 seats",
            spec=_pool(_decode(32, 9215), DEP4_PAGES, CORE_WIDTH, 11),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_logical_positions` (int64) and `qsa_req_idx_per_token` (int32)
    sliced to the step's rows, `qsa_block_table` unsliced, the layer's
    `get_index_k_buffer(i)` and the shared `get_qsa_position_buffer()`, the
    indexer's `k_layernorm` weight (bf16 [128]) and eps, a float32
    `[max_positions, 64]` cos|sin table built at load, and `mrope_section`
    None. It runs on every step, decode and prefill rows alike.

    Preconditions:
      - every member's raw key and the first member's coordinates are in the
        caches before the call; this call stores neither.
      - every logical position is non-negative. `L = -1` satisfies the anchor
        test and addresses positions -4..-1: measured, an illegal memory
        access that poisons the process. A value, so not guarded; pad with 0
        (measured an exact no-op).
      - `logical_positions`, `request_indices`, `k_norm_weight` and the last
        dim of `cos_sin` are read with an assumed unit stride while the
        callable checks only shape: each was measured accepted and wrong
        (max abs 4.72, 6.20, 0.80, 7.16). `is_valid` refuses them.
    The callable rejects loudly (not pinned by a test): metadata shape/dtype mismatches, a non-int32
    or non-2-D table, more than one index head, a position plane with fewer
    than 3 axes or not int32, a non-bf16 cache, page sizes that disagree with
    `tokens_per_block`, a malformed cos|sin table, `4 R != D`, a non-2^k `D`
    or `compress_ratio`, a norm weight of the wrong length, CPU operands, and
    `mrope_section` that is not 3 entries summing to `R`. `rows == 0` returns
    before launching.

    `cos_sin`'s dtype is the rotation's precision: bf16 is accepted and was
    measured 1.4% of elements off by up to 1.56e-2 -- not certified. Text
    rows reduce mRoPE to plain RoPE bit for bit (measured, not pinned by a
    test). Plain `@triton.jit`, no autotune, JIT-compiled at first call; the
    cells drove that JIT path on sm_103 (GB300). Every stride is a
    `tl.constexpr`, so each pool layout (contiguous, the NVFP4 shared page,
    the unified target's coalesced bf16 pool) is its own compiled specialization, and
    each is driven by a cell. CUDA-graph capturable at a fixed row count
    (captured at 256 rows, 32 MTP7 verify blocks, and at 128 decode rows over
    the unified target's tep4 pool).

    Under an NVFP4 KV pool both caches are strided views of one per-slot
    allocation (each physical page holds the index roles, then the position
    page); the trunk-plane cell checks the other index roles and the position
    page untouched.

    The unified target's pools (`coalesced` cells), on every member and rank at bf16
    KV: KV cache manager V2 coalesces the 12 sparse layers' index pages into
    one pool, so `get_index_k_buffer(i)` is `[pages, 32, 1, 128]` with page
    stride 49152 elements, starting 4096 x (the layer's ordinal) into each slot;
    the position plane is a contiguous `[pages, 32, 3]` (read off the stock
    manager at tep4 and dep4 rank 0, max_seq_len 9216 and 16384). The cells hold
    a rank's measured pool -- tep4 258983 pages, dep4 120530: the stock
    engine's `maxNumBlocks` at core config, host-resident n-gram table, an
    upper bound on the index view, which the same manager builds slightly
    smaller (1706 index pages for 1842 primary blocks) -- with request 0's last
    page the pool's last, written at 258982 x 49152 elements from the base,
    past 2^31; the table is the engine's `[128, 292]` (core) or `[128, 516]`
    (MTP, max_seq_len 16384), 128 seats at 128 and at 32 requests. The other
    11 layers' planes are checked untouched.

    The kernel's `tl.rsqrt` is the hardware approximation, an fp32 ulp from the
    reference's: every cell is bitwise the reference but two. In the 32-block
    MTP7 one, 1 of 8192 anchor elements rounds one bf16 step away. In the 8192-row
    tep4 prefill, 5 of 262016 do, every one an exact bf16 tie in the reference's
    fp32 value (low 16 bits 0x8000). Measured against the kernel on that cell,
    the nearest wrong variants are far outside that budget: sequential bf16
    pooling 94580 elements (up to 0.03125), coordinates from the group's last
    member 61418, the norm weight without its `+1` 261994.

    Hard limits (read from the kernel; the page offset also driven): positions,
    requests and page ids are widened to int64 before any address arithmetic,
    so no int32 boundary inside the call -- the coalesced cells address pages
    past 2^31 elements of the pool's base. Certified extent: positions to 16383
    through a [128, 516] table over 258983 pages; 8192 rows in one launch.

    Each rank's call is local to its device (the indexer is replicated, no
    collective), so a one-device cell stands for any rank's call at its pool.

    Not certified: `compress_ratio` other than 4, `D` other than 128, a bf16
    or fp16 `cos_sin`, positions past 16383 or the table's rows (the tune's
    max_seq_len 262144 and ~220k-token prompts), mRoPE outside the NVFP4
    trunk-plane cell, the NVFP4 draft planes (one index role a page),
    `tokens_per_block` other than 32 (no group straddles a page), eps other
    than 1e-6, int32 positions or int64 request indices, a strided view of the
    index cache, position plane or table other than the NVFP4 pool's shared
    page and the unified target's coalesced bf16 pool, the kv_fp8 / kv_nvfp4 dep4
    pools at their measured page count (their packed `QSA_SIDE_PLANES` layout
    is the shared-page cell's, at 4 pages), the MTP member's own pool count and
    its separate draft cache (the 1024-row cell runs on the core tep4 pool), a
    tp4 pool (never booted; the indexer is replicated, so tep4's stands in), two
    rows naming one anchor in a launch, and `rows == 0`.
    """

    def __call__(
        self,
        *,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        block_table: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        tokens_per_block: int,
        compress_ratio: int,
        mrope_section: Optional[Tuple[int, int, int]],
    ) -> torch.Tensor:
        _qsa_kernels.triton_qsa_prefill_compress(
            logical_positions=logical_positions,
            request_indices=request_indices,
            block_table=block_table,
            index_cache=index_cache,
            position_cache=position_cache,
            k_norm_weight=k_norm_weight,
            cos_sin=cos_sin,
            eps=eps,
            tokens_per_block=tokens_per_block,
            compress_ratio=compress_ratio,
            mrope_section=mrope_section,
        )
        return index_cache

    def reference(
        self,
        *,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        block_table: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        tokens_per_block: int,
        compress_ratio: int,
        mrope_section: Optional[Tuple[int, int, int]],
    ) -> torch.Tensor:
        """The index cache after the call, as a fresh tensor, from the pre-call state."""
        out = index_cache.clone()
        logical = logical_positions.long()
        anchored = (logical + 1) % compress_ratio == 0
        anchor = logical[anchored]
        if anchor.numel() == 0:
            return out
        device = index_cache.device
        request = request_indices.long()[anchored]
        members = (
            anchor[:, None] - (compress_ratio - 1) + torch.arange(compress_ratio, device=device)
        )
        pages = block_table.long()[request[:, None], members // tokens_per_block]
        slots = members % tokens_per_block

        head_dim = index_cache.shape[3]
        raw = index_cache[pages, slots, 0].float()  # [anchors, cr, D]
        pooled = (raw.sum(dim=1) / compress_ratio).to(torch.bfloat16).float()
        scale = torch.rsqrt(pooled.mul(pooled).sum(dim=-1, keepdim=True) / head_dim + eps)
        normed = (pooled * scale * (k_norm_weight.float() + 1.0)).to(cos_sin.dtype)

        t, h, w = position_cache[pages[:, 0], slots[:, 0], :3].long().unbind(-1)
        pairs_n = cos_sin.shape[1] // 2
        pairs = torch.arange(pairs_n, device=device)
        if mrope_section is None:
            pos = t[:, None].expand(-1, pairs_n)
        else:
            pos = torch.where(
                (pairs % 3 == 1) & (pairs < 3 * mrope_section[1]),
                h[:, None],
                torch.where(
                    (pairs % 3 == 2) & (pairs < 3 * mrope_section[2]), w[:, None], t[:, None]
                ),
            )
        cos, sin = cos_sin[pos, pairs], cos_sin[pos, pairs_n + pairs]
        first, second = normed[:, :pairs_n], normed[:, pairs_n : 2 * pairs_n]
        rotated = torch.cat(
            [first * cos - second * sin, second * cos + first * sin, normed[:, 2 * pairs_n :]],
            dim=-1,
        )
        out[pages[:, -1], slots[:, -1], 0] = rotated.to(torch.bfloat16)
        return out

    def is_valid(
        self,
        *,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        block_table: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        tokens_per_block: int,
        compress_ratio: int,
        mrope_section: Optional[Tuple[int, int, int]],
    ) -> None:
        # Read by the kernel with an assumed unit stride; the callable checks
        # only their shapes. Each measured accepted and silently wrong.
        assert logical_positions.is_contiguous(), "logical_positions must be contiguous"
        assert request_indices.is_contiguous(), "request_indices must be contiguous"
        assert k_norm_weight.is_contiguous(), "k_norm_weight must be contiguous"
        # `ndim != 2` is the callable's own loud error; leave it to the callable.
        assert cos_sin.ndim != 2 or cos_sin.stride(1) == 1, (
            "cos_sin must have unit stride along its last dimension"
        )


qsa_prefill_compress = QsaPrefillCompress()
