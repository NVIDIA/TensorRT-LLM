# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse path's paged K/V append: this step's rows into the HND main pool."""

from __future__ import annotations

from typing import Tuple

import torch

# Tier-2 sourcing: a vendored Triton callable reached as ordinary Python. QSA's
# kernel plane has no torch.ops surface.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

from .._op import Arch, Cell, OpWrapper

#: The qwen4_exp checkpoint's main K/V geometry and the engine's QSA metadata.
#: `heads_q` places k and v as head slices of the packed `[T, 24 + 2 + 2, 256]`
#: QKV buffer; `slots` x `width` is `qsa_block_table` (max_num_sequences x
#: max_blocks_per_seq). `pages` is the pool, independent of the batch.
_CKPT = dict(
    hkv=2,
    head_dim=256,
    heads_q=24,
    tokens_per_block=32,
    slots=32,
    width=260,
    row_dtype=torch.bfloat16,
    cache_dtype=torch.bfloat16,
    request_dtype=torch.int32,
    position_dtype=torch.int64,
    hole=None,
    # Heads of further columns after v in each row: 27 is the target's fused input
    # projection row [q | k | v | gate | index | pad] (55 heads of 256).
    row_pad_heads=0,
    # How the rows are laid out: "packed" = k and v are head slices of one
    # `[T, heads_q + 2 * hkv + row_pad_heads, head_dim]` row; "padded" = k and v are
    # each a contiguous `[T, hkv, head_dim]` tensor.
    row_layout="packed",
    # Real head width: dims past it are zero (the 192 -> 256 padding).
    real_dim=256,
    # Row values: plain randn, or randn times a power of two in 2^-6..2^3 per
    # (row, head), plus this fraction of real dims set to +-(440..640).
    scaled_values=False,
    outlier_fraction=0.0,
    # Pool slot layout: `slot_pages` role pages per slot; the call's K and V are
    # pages 2 * layer and 2 * layer + 1 of each slot.
    slot_pages=2,
    layer=0,
    # Table columns before the ones this step writes: "unique" pages drawn from
    # the pool's permutation, or "shared" random pool pages (a reused prefix).
    prefix="unique",
    # Put the pool's last slot among the pages this step writes.
    include_last_page=False,
    large=False,
)

_F8 = torch.float8_e4m3fn

_DECODE = ((0, 1), (31, 1), (32, 1), (33, 1), (100, 1), (255, 1), (7, 1), (64, 1))

#: 32 requests of one MTP7 verify block each (8 rows), at starts that put some blocks across a
#: page edge and some inside one.
_MTP7_32 = tuple((29 + 61 * i, 8) for i in range(32))

#: Step-5-Preview's sparse (full-attention) layers, per rank: 4 KV heads under
#: attention DP, 1 under attention TP4, 16 query heads per KV head, heads
#: zero-padded from 192 to 256. One slot of the full-attention pool holds the 46
#: K/V pages of the 23 sparse layers (K then V per layer, the Step-5 cache
#: manager's layout); the cells write the last layer's pair (pages 44 and 45).
#: The table is `max_batch_size` 128 rows x `max_blocks_per_seq` 292 (ceil(9217 /
#: 32) rounded up to a multiple of 4); positions reach `max_seq_len - 1` = 9215.
_STEP5 = dict(
    _CKPT,
    head_dim=256,
    real_dim=192,
    slots=128,
    width=292,
    slot_pages=46,
    layer=22,
    scaled_values=True,
    prefix="shared",
    include_last_page=True,
)

#: Pool slots chosen so the last slot's element offset passes 2^31 (1535 x 46 x
#: 4 x 32 x 256 and 6143 x 46 x 32 x 256 elements, about 2.31e9) while the pool
#: stays at 4.3 GiB (bf16) or 2.2 GiB (fp8).
_DP_POOL, _TP_POOL = 1536, 6144

#: Full-attention slots the Step-5 cache manager allocates per rank at
#: kv_cache_fraction 0.7 after the checkpoint's weights over 4 ranks (its sizing
#: test, on GB300): 22449 (dp4 bf16), 37821 (dp4 fp8), 60225 (tp4 bf16) and 89885
#: (tp4 fp8). Only the tp4 fp8 pool (31.5 GiB) fits the test budget.
_TP4_FP8_MEASURED_POOL = 89885

#: 128 decode requests: page-edge positions, the last position of `max_seq_len`,
#: and positions spread over the rest of it.
_STEP5_DECODE_128 = ((0, 1), (31, 1), (32, 1), (33, 1), (9215, 1)) + tuple(
    (1 + (9213 * i) // 122, 1) for i in range(123)
)


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


def _step5(segments, pages, **overrides) -> dict:
    return {**_STEP5, "segments": segments, "pages": pages, **overrides}


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The same storage as integers: fp8 has no comparison ops, and -0.0 != 0.0."""
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.view(as_int)


class QsaPagedKvStore(OpWrapper):
    """Scatter every row's K and V into its request's page of the HND pool.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_paged_kv_store`: one Triton launch over `(rows, Hkv)`. For row t::

        page = block_table[request_indices[t], logical_positions[t] // tokens_per_block]
        slot = logical_positions[t] % tokens_per_block
        k_cache[page, :, slot, :] = k[t]          v_cache[page, :, slot, :] = v[t]

    `logical_positions` is absolute, so prefill, decode and MTP rows are the
    same call. The store is a move, converted once to the pool's dtype
    (bitwise the device's `Tensor.to`; an fp8-e4m3 pool saturates at 448 and
    takes no scale, so a reader treats it as kv scale 1.0). A row that resolves
    to no slot -- negative request, request past the table, negative position,
    page column past the table, or a table entry that is negative (`-1`,
    unallocated) or past the pool -- is skipped without error. Returns
    `(k_cache, v_cache)`, both written in place.

    Fusion boundary: the scatter only. The K/V projection, norm and RoPE of
    K, the head padding, the QSA index plane, selection and attention are the
    caller's; the index plane is never touched.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # The decode and prefill targets read k/v off the fused projection row; the MTP
        # targets split them off a packed QKV row.
        Cell(
            why="decode: 8 rows off the fused projection row (row stride 14080), "
            "at aligned, mid and page-crossing positions, plus one row on an "
            "unallocated (-1) table entry that is skipped",
            spec=_spec(_DECODE, pages=64, row_pad_heads=27, hole="unallocated_page"),
        ),
        Cell(
            why="prefill 1024 rows off the fused projection row (row stride 14080)",
            spec=_spec(((0, 256),) * 4, pages=160, row_pad_heads=27),
        ),
        # MTP7 verify blocks: 8 rows per request, so one block spans up to two pages.
        # The non-default pool the forward reaches: fp8-e4m3.
        Cell(
            why="fp8 decode MTP7: 32 requests x 8 rows (256), blocks straddling pages",
            spec=_spec(_MTP7_32, pages=1200, slots=40, cache_dtype=_F8),
        ),
        # Step-5-Preview sparse layers: 4 KV heads per rank under attention DP, 1
        # under attention TP4, D 256 (192 real), 46-page slots, 292-column table.
        Cell(
            why="Step-5 dp4 decode 1 row at position 9215 (column 287), bf16 pool, "
            "contiguous padded k/v, the pool's last slot",
            spec=_step5(((9215, 1),), pages=_DP_POOL, hkv=4, heads_q=64, row_layout="padded"),
        ),
        Cell(
            why="Step-5 dp4 decode 128 rows (page edges, 9215, spread), bf16 pool, "
            "k/v head slices of a packed 72-head row",
            spec=_step5(_STEP5_DECODE_128, pages=_DP_POOL, hkv=4, heads_q=64),
        ),
        Cell(
            why="Step-5 dp4 prefill 8192 rows of one request from 0, bf16 pool, "
            "contiguous padded k/v",
            spec=_step5(((0, 8192),), pages=_DP_POOL, hkv=4, heads_q=64, row_layout="padded"),
        ),
        Cell(
            why="Step-5 tp4 decode 128 rows, bf16 pool, 1 KV head, k/v head slices "
            "of a packed 18-head row",
            spec=_step5(_STEP5_DECODE_128, pages=_TP_POOL, hkv=1, heads_q=16),
        ),
        Cell(
            why="Step-5 dp4 fp8 decode 128 rows with 1% of values past 448 (saturation), "
            "contiguous padded k/v",
            spec=_step5(
                _STEP5_DECODE_128,
                pages=_DP_POOL,
                hkv=4,
                heads_q=64,
                row_layout="padded",
                cache_dtype=_F8,
                outlier_fraction=0.01,
            ),
        ),
        Cell(
            why="Step-5 tp4 fp8 prefill 8192 rows over 3 requests, a chunk ending at "
            "9215, values past 448, packed rows",
            spec=_step5(
                ((4103, 5113), (0, 3000), (17, 79)),
                pages=_TP_POOL,
                hkv=1,
                heads_q=16,
                cache_dtype=_F8,
                outlier_fraction=0.01,
            ),
        ),
        Cell(
            why="Step-5 tp4 fp8 decode 128 rows into the measured 89885-slot pool "
            "(31.5 GiB, offsets past 2^34 elements)",
            spec=_step5(
                _STEP5_DECODE_128,
                pages=_TP4_FP8_MEASURED_POOL,
                hkv=1,
                heads_q=16,
                cache_dtype=_F8,
                outlier_fraction=0.01,
                large=True,
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_req_idx_per_token` (int32, the row's batch index) and
    `qsa_logical_positions` (int64) sliced to the step's rows,
    `qsa_block_table` unsliced (raw pool page ids, `-1` unallocated), and the
    layer pool's K and V role views of `get_buffers(i, kv_layout="HND")`. In
    the Step-5 target the views are `Step5CacheManager.get_kv_slot_buffers(
    layer, kv_layout="HND")[:, 0]` and `[:, 1]`, indexed by slot ids (the
    table is `kv_cache_block_offsets` divided by the scale of
    `get_block_offset_slot_scale`): a page stride of 46 pages, K and V one
    page apart inside the slot.

    Preconditions:
      - destination slots are unique across the batch: two rows naming one
        (page, slot) both store, unordered. The engine's mapping cannot
        collide.
      - `request_indices` and `logical_positions` are contiguous: the kernel
        gets no stride for them. A strided view is read through its base
        pointer and rows land at other rows' slots, silently (`is_valid`).
    Every other operand is read through its strides; the cells drive k/v as
    head slices of the packed QKV row and of the fused projection row and as
    contiguous `[T, Hkv, 256]` tensors, the pool's K and V role views at
    2-page and 46-page slot strides, and a contiguous table. The callable
    rejects loudly: mismatched or non-3-D/4-D K/V shapes, head geometry,
    `k_cache.shape[2] != tokens_per_block`, row dtypes outside bf16/fp16,
    pool dtypes outside bf16/fp16/fp8-e4m3, a non-int32 or non-2-D table,
    non-integer or wrong-length metadata, CPU operands, an empty pool or
    table (not pinned by a test). `rows == 0` returns before launching.

    Same bytes as the attention op's append (thop `attention` with
    `update_kv_cache=True`, pinned by `test_matches_the_attention_ops_append`):
    on every slot a row names, this op and thop's context and generation
    appends write identical bytes into a bf16 pool and into an fp8 pool
    (quant_mode 128, kv scale 1.0), at Step-5's 64/4 and 16/1 heads x 256,
    through the 46-page slot layout, values past 448 included (both saturate
    to +-448; so does the device `Tensor.to` the reference uses, which also
    maps +-inf to +-448). Measured difference outside the named slots: thop's
    context append into a bf16 pool also writes zeros to the V plane of the
    slots after a context request's last token in its last page (not into an
    fp8 pool, not in generation); this op leaves those slots as they were.
    QSA's readers address only selected positions below the sequence length;
    a dense FMHA over this pool would read V there.

    Strides and the pool/table extents are Triton constexprs: each distinct
    layout or pool size compiles its own kernel. CUDA-graph capturable
    (captured at 256 rows, 32 MTP7 verify blocks, and at Step-5's 128 decode
    rows); a replay follows positions, request indices, table and rows
    rewritten in place.

    Hard limits: both metadata values are cast to int32 inside the kernel
    (`.to(tl.int32)`); an int64 position at `2^32 + 5` was probed once (not
    pinned by a test) and lands on slot 5 of page column 0 -- written, not
    skipped. The engine's positions stay below 2^14. The page id is widened
    to int64 before it multiplies the page stride: the measured tp4 fp8 pool
    (89885 slots x 46 pages x 8192 elements, offsets to 3.39e10 elements) is
    a cell, and every Step-5 cell writes its pool's last slot past 2^31
    elements. The row offset `row * k.stride(0)` is a program id times a
    constexpr stride: computed bound rows x row stride < 2^31 (a row stride
    of 262143 elements at 8192 rows; Step-5's packed rows are 72 x 256 =
    18432 wide), not probed.

    Not certified: duplicate destinations, multiple devices, a scaled fp8
    pool (there is no scale argument), the packed-QKV-row (MTP) layout over a
    bf16 pool at Hkv 2, fp16 rows, head geometry other than D 256 with Hkv
    2, 4 or 1, tokens_per_block other than 32, int32 positions or int64
    request indices, head-major rows, sliced or transposed tables,
    `rows == 0`, of the rows that resolve to no slot any but an unallocated
    (`-1`) table entry (the reference skips each), and Step-5's measured
    pools at dp4 bf16 (22449 slots, 66 GiB), dp4 fp8 (37821, 55 GiB) and tp4
    bf16 (60225, 42 GiB), past the test budget (there the pool extent is a
    constexpr bound only, and their offsets stay below the tp4 fp8 cell's).
    """

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _qsa_kernels.triton_qsa_paged_kv_store(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            request_indices=request_indices,
            logical_positions=logical_positions,
            block_table=block_table,
            tokens_per_block=tokens_per_block,
        )
        return k_cache, v_cache

    def reference(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both caches after the scatter, as fresh tensors; the inputs are not touched."""
        expected_k, expected_v = k_cache.clone(), v_cache.clone()
        requests = request_indices.long()
        positions = logical_positions.long()
        columns = positions.div(tokens_per_block, rounding_mode="floor")
        table_rows, table_width = block_table.shape
        addressable = (
            (requests >= 0) & (requests < table_rows) & (positions >= 0) & (columns < table_width)
        )
        pages = block_table[
            requests.clamp(0, table_rows - 1), columns.clamp(0, table_width - 1)
        ].long()
        addressable &= (pages >= 0) & (pages < k_cache.shape[0])
        page, slot = pages[addressable], (positions % tokens_per_block)[addressable]
        expected_k[page, :, slot, :] = k[addressable].to(expected_k.dtype)
        expected_v[page, :, slot, :] = v[addressable].to(expected_v.dtype)
        return expected_k, expected_v

    def is_valid(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> None:
        # The kernel reads these two as flat unit-stride buffers. Measured: a
        # column of a 2-wide buffer is read through its base pointer and the
        # rows land at other rows' slots, with nothing raised.
        assert request_indices.is_contiguous(), "request_indices must be a contiguous 1-D tensor"
        assert logical_positions.is_contiguous(), (
            "logical_positions must be a contiguous 1-D tensor"
        )


qsa_paged_kv_store = QsaPagedKvStore()
