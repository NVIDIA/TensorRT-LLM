# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MQA of one query row per sequence over an fp8_ds_mla pool, 8 or 16 query heads
unpadded: a CuTe DSL kernel that computes FlashMLA's sm100f sparse fp8 decode."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul
from . import _ds_mla_sparse_decode_kernel as _kernel

ROW_BYTES = _kernel.ROW_BYTES
D_NOPE = _kernel.D_NOPE
D_QK = _kernel.D_QK
TOPK = _kernel.TOPK
TILE = 128
#: The real head counts the kernel runs.
HEADS = _kernel.SUPPORTED_HEADS
#: The split-K counts (CTAs per row) a batch can select.
SPLITS = _kernel.SUPPORTED_SPLITS

#: The tp4 / tep4 (16 heads) and tp8 / tep8 (8 heads) targets at their 1M config (max_seq_len
#: 1048576, chunked prefill, 16 sequences): top-2048 selections over keys up to 1048576 in the
#: fp8_ds_mla pool's 32-token pages of 63 KV-owner rows per block.
_1M = dict(topk=2048, kv_len=1048576, page_size=32, owners=63)
#: The 1M pools by block count: the delivered 1M tp4 boot's 74183 (above the 72757 a 1M tep4
#: rank's boot measured), the far end 107917 (the far cells of the 1M attention entries), and
#: the 147456-block bound (a 1M tep8 rank held 114326). Every one puts slots past 2**31 bytes.
_BOOT_BLOCKS, _FAR_BLOCKS, _BOUND_BLOCKS = 74183, 107917, 147456


@dataclass(frozen=True)
class CompiledDsMlaSparseDecode:
    """Every split variant a batch can select, for one head count on one device, and the SM
    count the selection reads."""

    device_index: int
    num_heads: int
    num_sms: int
    fns: Mapping[int, Callable]


def build(device_id: int, *, num_heads: int) -> CompiledDsMlaSparseDecode:
    """Compile the four split variants for `num_heads` (8 or 16) real heads: about a second
    each, cached per process by the kernel module; allocates nothing."""
    with torch.cuda.device(device_id):
        fns = {ns: _kernel.compile_variant(num_heads, ns, device_id) for ns in SPLITS}
        num_sms = torch.cuda.get_device_properties(device_id).multi_processor_count
    return CompiledDsMlaSparseDecode(
        device_index=device_id, num_heads=num_heads, num_sms=num_sms, fns=fns
    )


def splits_for(rows: int, compiled: CompiledDsMlaSparseDecode) -> int:
    """The split-K count a call with `rows` rows runs."""
    return _kernel.pick_splits(rows, compiled.num_sms)


class CuteDslDsMlaSparseDecode(OpWrapper):
    """For every query row r and head h < `out.shape[1]`::

        K[j]   = [e4m3(row_j[:512]) * tile_scale_j (per 128) | bf16(row_j[528:656])]
        p      = softmax_j(softmax_scale * q[r, h] . K[j]),  j in indices[r] (-1 skipped)
        out[r, h] = sum_j p[j] * K[j][:512]

    `kv_cache`: uint8 `[pages, page_size, 1, 656]`, contiguous; a slot is
    `page * page_size + offset`. `indices`: int32 `[rows, 2048]` slots, -1 padded,
    contiguous. `q`: bf16 `[rows, h_q, 576]`, contiguous, the real heads first: h_q is the
    caller's scratch width (the heads themselves, or a padded scratch another kernel shares)
    and `out` (bf16 `[rows, heads, 512]`, contiguous) names how many heads are real, 8 or 16.
    `compiled`: `build(device, num_heads=heads)`. Writes `out` and returns it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `rows` sequences of one query row, `heads` real heads read from a `h_q`-head
    # query, `topk` slots over `kv_len`-token sequences in a pool of `pages` pages of
    # `page_size` x `owners` rows; `layers` calls; `captured` replays from a CUDA graph;
    # `holes` scatters -1 through short rows; `nope` zeroes the rope halves.
    # The "1M" cells build a pool of `blocks` blocks on the device, every byte 0xFF (e4m3,
    # fp32 and bf16 NaN) but one owner row of the pages 16 decode sequences hold (keys
    # `kv_len`, `kv_len` - 31, ... down to 1); a row r is a decode row of sequence r % 16;
    # `far` puts those pages at the pool's top and the first sequence's last key on its top
    # block; `graph_batches` captures each batch and replays it on new queries and slots;
    # `q_offset` makes q and out tail views from that row of a 32768-row step scratch.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"tep4/tp4 decode batch {b}: 16 heads, graph replay, 3 layers",
                spec=dict(
                    rows=b,
                    heads=16,
                    h_q=16,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * b // 16 + 64,
                    layers=3,
                    captured=True,
                ),
            )
            for b in (1, 2, 4, 8, 16)
        )
        + tuple(
            Cell(
                why=f"tp4 decode batch {b}: 16 heads, graph replay ({s} CTAs per row)",
                spec=dict(
                    rows=b,
                    heads=16,
                    h_q=16,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * b // 16 + 64,
                    layers=2,
                    captured=True,
                ),
            )
            for b, s in ((32, 4), (64, 2), (128, 1), (256, 1))
        )
        + tuple(
            Cell(
                why=f"tep8/tp8 decode batch {b}: 8 heads, graph replay, 3 layers",
                spec=dict(
                    rows=b,
                    heads=8,
                    h_q=8,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * b // 16 + 64,
                    layers=3,
                    captured=True,
                ),
            )
            for b in (1, 2, 4, 8, 16, 64, 256)
        )
        + (
            Cell(
                why="mixed step: generation rows behind context rows read 16 of the "
                "prefill scratch's 64 heads",
                spec=dict(
                    rows=24,
                    heads=16,
                    h_q=64,
                    topk=2048,
                    kv_len=9000,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                ),
            ),
            Cell(
                why="mixed step at world 8: 8 of the prefill scratch's 64 heads",
                spec=dict(
                    rows=5,
                    heads=8,
                    h_q=64,
                    topk=2048,
                    kv_len=9000,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                ),
            ),
            Cell(
                why="short sequences: -1 tails and scattered holes",
                spec=dict(
                    rows=16,
                    heads=16,
                    h_q=16,
                    topk=2048,
                    kv_len=700,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                    holes=True,
                ),
            ),
            Cell(
                why="short sequences at 8 heads: -1 tails and scattered holes",
                spec=dict(
                    rows=40,
                    heads=8,
                    h_q=8,
                    topk=2048,
                    kv_len=700,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                    holes=True,
                ),
            ),
            Cell(
                why="NoPE layer: zero query and key rope halves",
                spec=dict(
                    rows=16,
                    heads=16,
                    h_q=16,
                    topk=2048,
                    kv_len=9000,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                    nope=True,
                ),
            ),
        )
        # The tep4/tp4 and tep8/tp8 cells at 64-token pages: the same rows, heads, top-k,
        # lengths and graph replays, half the pages for the same token capacity.
        + tuple(
            Cell(
                why=f"64-token pages: decode batch {b}: {h} heads, graph replay, 3 layers",
                spec=dict(
                    rows=b,
                    heads=h,
                    h_q=h,
                    topk=2048,
                    kv_len=32768,
                    page_size=64,
                    owners=63,
                    pages=550 * b // 16 + 32,
                    layers=3,
                    captured=True,
                ),
            )
            for h in (16, 8)
            for b in (1, 4, 16, 64)
        )
        + (
            Cell(
                why="64-token pages: short sequences: -1 tails and scattered holes",
                spec=dict(
                    rows=16,
                    heads=16,
                    h_q=16,
                    topk=2048,
                    kv_len=700,
                    page_size=64,
                    owners=63,
                    pages=256,
                    layers=1,
                    holes=True,
                ),
            ),
            Cell(
                why="64-token pages: NoPE layer: zero query and key rope halves",
                spec=dict(
                    rows=16,
                    heads=16,
                    h_q=16,
                    topk=2048,
                    kv_len=9000,
                    page_size=64,
                    owners=63,
                    pages=256,
                    layers=1,
                    nope=True,
                ),
            ),
        )
    )
    # The 1M config's call sites: DecodeTarget's generation rows (captured at every batch the
    # LLM API's default list reaches, 1..16) at 16 and 8 heads on the boot, far and bound pools,
    # the larger batches every split variant runs, PrefillTarget._attn_gen's generation rows
    # behind a chunk (16 / 8 of the prefill scratch's 64 heads, tail views of the step's
    # scratch), a NoPE layer's shared row, and the boot captures' short keys.
    CELLS += (
        Cell(
            why="1M tp4/tep4 decode graphs at every batch 1..16, boot pool (large): 16 heads, "
            "74183 blocks (98.1 GB), rows of 1048576 keys down to 1, the pool's last slot read, "
            "3 layers",
            spec=dict(
                _1M,
                rows=16,
                heads=16,
                h_q=16,
                blocks=_BOOT_BLOCKS,
                layers=3,
                captured=True,
                graph_batches=tuple(range(1, 17)),
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M tp8/tep8 decode graphs at every batch 1..16, bound pool (large): 8 heads, "
            "147456 blocks (195 GB), the top slot read, 3 layers",
            spec=dict(
                _1M,
                rows=16,
                heads=8,
                h_q=8,
                blocks=_BOUND_BLOCKS,
                layers=3,
                captured=True,
                graph_batches=tuple(range(1, 17)),
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M tp4/tep4 decode graph 16, far pool (large): 107917 blocks (142.7 GB), the "
            "top slot read at byte 1.427e11, 3 layers",
            spec=dict(
                _1M,
                rows=16,
                heads=16,
                h_q=16,
                blocks=_FAR_BLOCKS,
                layers=3,
                captured=True,
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M decode batches 32, 64, 128 and 256 at 16 heads (every split variant), keys to "
            "1048576 on the boot pool (large), captured",
            spec=dict(
                _1M,
                rows=256,
                heads=16,
                h_q=16,
                blocks=_BOOT_BLOCKS,
                layers=2,
                captured=True,
                graph_batches=(32, 64, 128, 256),
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M decode batches 32, 64 and 256 at 8 heads, keys to 1048576 on the boot pool "
            "(large), captured",
            spec=dict(
                _1M,
                rows=256,
                heads=8,
                h_q=8,
                blocks=_BOOT_BLOCKS,
                layers=2,
                captured=True,
                graph_batches=(32, 64, 256),
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M tp4/tep4 mixed step's 13 generation rows (large): 16 of the prefill scratch's "
            "64 heads, q and out tail views from row 32755 of a 32768-row scratch, boot pool",
            spec=dict(
                _1M,
                rows=13,
                heads=16,
                h_q=64,
                blocks=_BOOT_BLOCKS,
                layers=3,
                q_offset=32755,
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M tp8/tep8 mixed step's 16 generation rows (large): 8 of the prefill scratch's "
            "64 heads, q and out tail views from row 32752, boot pool",
            spec=dict(
                _1M,
                rows=16,
                heads=8,
                h_q=64,
                blocks=_BOOT_BLOCKS,
                layers=3,
                q_offset=32752,
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M NoPE decode graph 16 (large): owner row 35 as its nine sharers read it, zero "
            "query and key rope halves, boot pool",
            spec=dict(
                _1M,
                rows=16,
                heads=16,
                h_q=16,
                blocks=_BOOT_BLOCKS,
                layers=9,
                captured=True,
                nope=True,
                far=True,
                large=True,
            ),
        ),
        Cell(
            why="1M pre-merge decode graphs at every batch 1..16 to 2049 keys (2048 of 2049 "
            "valid, -1 tails below 2048), 16 heads, a 512-block pool",
            spec=dict(
                _1M,
                rows=16,
                heads=16,
                h_q=16,
                kv_len=2049,
                blocks=512,
                layers=3,
                captured=True,
                graph_batches=tuple(range(1, 17)),
            ),
        ),
    )

    CELLS += tuple(
        Cell(
            why=f"DSpark 1M mixed step (large): 13 four-token verification blocks, {h} heads "
            "in the prefill scratch's 64 heads, tail view after 32716 context rows",
            spec=dict(
                _1M,
                rows=52,
                heads=h,
                h_q=64,
                blocks=_BOOT_BLOCKS,
                layers=3,
                q_offset=32716,
                far=True,
                large=True,
            ),
        )
        for h in (8, 16)
    )

    note = """
    Call sites: the mistral_large4 NVFP4 tp4 and tep4 targets (also run at world 8 as tp8
    and tep8), the generation rows of every layer, decode and mixed steps. Kernel (sm_103):
    `_ds_mla_sparse_decode_kernel`, one launch, split-K partials merged in the cluster.
    It replaces flash_mla_sparse_decode's head64 kernel on these targets, which needs the
    query padded to 64 heads, a scheduler kernel and a combine kernel.

    Numerics: FlashMLA's (15f13e5, built --use_fast_math), operation for operation. Stage,
    FlashMLA (csrc/sm100/decode/head64/kernel.cuh unless named) -> this kernel:
      scale        sparse_decode.h:387, common.h:12: fp32 sm_scale * 1.44269504f
                   -> scale_log2_of, the same fp32 product on the host.
      K/V dequant  :708 cvt.rz.ue8m0x2.f32 of the fp32 scales, :788 cvt.rn.bf16x2.ue8m0x2,
                   :802 helpers.h fp8x2_to_bf16x2_with_scale (bf16(e4m3) * bf16 scale)
                   -> the same two PTX conversions, cvt e4m3x2 -> bf16x2, mul.rn.bf16x2.
                   Rope bf16 as stored. Invalid slots: :703/:710 TMA coordinate -1 and
                   scale bytes 0 -> TMA zero-fills the row (codes and scales 0); the
                   token is masked below either way, so K and V of it never reach a sum.
      Q            bf16 as given (UTCCP into TMEM, :465) -> TMA into SMEM, bf16.
      QK           :542 rope (clear) then :552 NoPE, a dual GEMM: two fp32 accumulators,
                   half 0 = rope[0:32], NoPE chunks 0,2,4,6; half 1 = rope[32:64], chunks
                   1,3,5,7; K = 16 per MMA in that order; :192 float2_add (add.rn.f32)
                   of the halves -> two TMEM accumulators fed in the same order, add.rn.
      mask         :215 p = -inf for an invalid token -> the same select.
      row max      :222 max of the raw scores, :224 * sm_scale_div_log2 (mul.ftz), :229
                   max with the other 32 tokens -> max over the tile's 64 tokens, then
                   mul.ftz (max commutes with the monotone product).
      real max     :230 real_mi = max(real_mi, cur) -> the same.
      rescale vote :231 __any_sync(cur - mi > 6.0f) over the warp's 32 heads (16 real +
                   16 zero-query heads at 16 real; the zero heads never vote after the
                   first valid tile and vote with the real ones on it) -> OR over the
                   real heads of sub.ftz(cur, mi) > 6.
      new max      :240-244 1.0f / max(cur, mi), exp2f(mi - new_max) -> ex2.approx.ftz of
                   sub.ftz, exactly 1.0 when the max stays.
      exponent     :254 float2_fma (fma.rn.f32) of p, scale, -new_max; :255 exp2f
                   (ex2.approx.ftz) -> the same two instructions.
      P            :258 __float22bfloat162_rn -> cvt.rn.bf16x2.f32.
      denominator  :257 cur_sum += d (add.rn.f32, even and odd tokens of each 32-token
                   half in order), :260 li = fma(li, scale_for_old, x + y) (fma.ftz,
                   add.ftz), :311 li(tokens 0-31) + li(32-63) -> warps 4-5 sum the fp32 P
                   the softmax hands over in that order with those instructions.
      O rescale    :269-283 if block != first and voted: float2_mul (mul.rn.f32) in TMEM
                   -> the same condition and mul.rn.f32, before the tile's PV.
      PV           :580 bf16 P x bf16 V, fp32 accumulation in TMEM, K = 16 tokens per MMA
                   in token order, cleared on the first block -> the same, swapped
                   (O^T = V^T P^T).
      no valid key :301-306 li = 0, mi = -inf -> the same.
      no split     :346 li == 0 ? 0 : __fdividef(1, li + exp2f(-inf - mi)) (= li),
                   :358 float2_mul, :359 __float22bfloat162_rn -> div.approx.ftz,
                   mul.rn.f32, cvt.rn.bf16x2.f32.
      split        :387 __fdividef(1, li), :400 float2_mul into fp32 o_accum, :321
                   lse = log2f(li) + mi -> div.approx.ftz, mul.rn.f32 into the fp32
                   partial in shared memory, lg2.approx.ftz + add.ftz.
      combine      combine.cu:84-88 max lse (0 when -inf), :93-96 sum of exp2f(lse - max)
                   over a xor butterfly, :98 global = sum == 0 ? +inf : log2f(sum) + max,
                   :117 w = exp2f(lse - global), :136 result += w * o (fma.ftz from 0, in
                   split order), :154 bf16 -> the same operations and order, through the
                   cluster's shared memory.
    What is not FlashMLA's: the split partition. FlashMLA's scheduler gives every
    partition ceil(37 B / 152) + 5 block-units of load (32 blocks + 5 overhead per row), so
    a row's split count moves with the batch (32 one-block splits at 1-4 rows, 8 at 16, 4
    at 32, 2 at 64, 1 at 128); this kernel splits a row in 8, 4, 2 or 1 equal parts by
    batch (`_kernel.pick_splits`). Where the two partitions coincide the output is
    bit-identical to FlashMLA's (the GPU test checks it at 16/32/64/128 rows, 16 and 8
    heads, with holes and NoPE). Elsewhere the partition sets which tile maxima the online
    softmax rescales against and which partials the combine merges -- a reordering of the
    same fp32 sums, with P rounded to bf16 against a different running max.

    Addressing: a pool row is fetched whole by a TMA `tile::gather4` of the pool viewed
    `[rows, 82]` int64 (row index int32, byte offset row * 656 in the TMA unit's 64-bit
    address); the index ring, the query and the output are addressed per row from int32
    offsets under 2**31. At the 1M config's pools (74183 to 147456 blocks of 63 x 32 rows,
    up to 2.97e8 rows and 1.95e11 bytes) the gathers read past 2**31 and 2**32 bytes; the
    "1M" cells read the pool's top slot. Keys up to 1048576 change nothing else: the kernel
    sees 2048 slots a row whatever the sequence. The 1M cells are bitwise FlashMLA's output
    where the partitions coincide (16 / 32 / 64 / 128 rows, 16 and 8 heads, on the boot
    pool), as the 32K ones are.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor,
        *,
        compiled: CompiledDsMlaSparseDecode,
    ) -> torch.Tensor:
        fn = compiled.fns[_kernel.pick_splits(q.shape[0], compiled.num_sms)]
        return _kernel.launch(fn, q, kv_cache, indices, out, softmax_scale)

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """fp32, exact dequantization, softmax over the valid slots; a new tensor shaped
        like `out`, rounded to bf16 once."""
        h_out = out.shape[1]
        flat = kv_cache.view(torch.uint8).reshape(-1, ROW_BYTES)
        res = torch.zeros(out.shape, dtype=torch.float32, device=q.device)
        with true_fp32_matmul():
            for r in range(q.shape[0]):
                idx = indices[r]
                idx = idx[idx >= 0].long()
                if idx.numel() == 0:
                    continue
                rows = flat[idx]
                codes = rows[:, :D_NOPE].contiguous().view(torch.float8_e4m3fn).float()
                scale = rows[:, D_NOPE : D_NOPE + 16].contiguous().view(torch.float32)
                nope = (codes.view(-1, D_NOPE // TILE, TILE) * scale.unsqueeze(-1)).view(-1, D_NOPE)
                rope = rows[:, D_NOPE + 16 :].contiguous().view(torch.bfloat16).float()
                k = torch.cat([nope, rope], 1)
                s = (q[r, :h_out].float() @ k.T) * softmax_scale
                res[r] = torch.softmax(s, -1) @ nope
        return res.to(out.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        softmax_scale: float,
        out: torch.Tensor,
        *,
        compiled: CompiledDsMlaSparseDecode,
    ) -> None:
        heads = out.shape[1]
        if heads not in HEADS or heads != compiled.num_heads:
            raise ValueError(f"{heads} heads: the kernel runs {HEADS}, compiled for "
                             f"{compiled.num_heads}")  # fmt: skip
        if q.dim() != 3 or q.shape[1] < heads or q.shape[2] != D_QK:
            raise ValueError(f"q must be [rows, h_q >= {heads}, {D_QK}]")
        if out.shape[0] != q.shape[0] or out.shape[2] != D_NOPE:
            raise ValueError(f"out must be [rows, heads, {D_NOPE}]")
        if indices.shape != (q.shape[0], TOPK) or indices.dtype != torch.int32:
            raise ValueError(f"indices must be int32 [rows, {TOPK}] slots")
        if kv_cache.shape[-1] != ROW_BYTES or kv_cache.dim() != 4 or kv_cache.shape[2] != 1:
            raise ValueError("kv_cache must be [pages, page_size, 1, 656]")
        if q.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError("q and out must be bf16")
        for name, t in (("q", q), ("kv_cache", kv_cache), ("indices", indices), ("out", out)):
            if not t.is_contiguous() or t.data_ptr() % 16:
                raise ValueError(f"{name} must be contiguous and 16-byte aligned")
            if t.device.index != compiled.device_index:
                raise ValueError(f"{name} is not on the compiled device")


cute_dsl_ds_mla_sparse_decode = CuteDslDsMlaSparseDecode()
