# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The packed spec-verify (q > 1) tokenspeed core with a polled q intake (Rubin DCP16).

``mla.core_ts``'s polled kernel (tokenspeed's generic FP8 MLA decode kernel whose
softmax and correction warps poll the CTA's q rows of the DCP mailbox after the PDL
wait, before the Q TMA; the K and V loads do not wait for q) in ``split_partials_q``'s
packed configuration: query row t, global head g of a request is packed row t * H + g,
and an M128 tile is ``pack_q``'s rows q_tile * 128 ... MlaQuery then ends at its
publish (``gather=False``): the core starts on its K tiles while the q rows land.

The q mailbox is a Lamport buffer of FP8 rows (``mla.dcp.DcpBox``): armed with Q_SENT
at allocation, written once per layer by every rank's MlaQuery (multicast stores after
its PDL wait; FP8 bytes are never 0xFF), polled here, and re-armed by the layer's
DcpValue after its PDL wait (this kernel has completed). Layers alternate parity, and
a parity's next writer, the MlaQuery two MLA layers on, publishes only after its own
PDL wait: the re-arm precedes it by transitive grid completion on every rank (each
rank's pushes of the layer in between follow its own re-arm)."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from ..common.policy import batch_table
from ..common.primitives import published
from .dcp import Q_SENT, _without_reducer, long_split, poll_b32x8, split_workspace_ts
from .qvalid import copy_valid, expect_valid, init_valid, wait_valid

# Rubin TP16 DCP16 spec-verify token-row batches (q > 1, the VERIFY_FOLD core) whose
# core polls its q tiles instead of MlaQuery gathering the whole mailbox.
POLLED_VERIFY = batch_table("16:1,24:1,32:1,40:1,48:1,56:1,64:1")
# The wide verify shapes (mla.prepared.choose above 64 rows, the drafter's rows above
# its box): the polled packed core runs tokenspeed's persistent scheduler (bitwise the
# same rows / partials) when its CTA pairs exceed this many waves of the SMs. 1 GPU,
# local 2000, us, today's grid -> persistent: q8 B24 (1.36 waves) 30.28 -> 28.64, B32
# 32.87 -> 30.20, B64 62.55 -> 54.73; q5 B32 (1.21) 29.63 -> 28.58; at <= 0.91 waves
# +0.1..+0.5 us (q8 B16 17.95 -> 18.08).
WIDE_PERSISTENT_WAVES = 1.0


def wide_persistent(reqs, tiles, split, sms):
    """Whether a wide shape's polled core runs persistent: its CTA pairs (requests x
    M128 q tiles x splits) above WIDE_PERSISTENT_WAVES waves of ``sms``."""
    return reqs * tiles * split * 2 / sms > WIDE_PERSISTENT_WAVES


# POLLED_VERIFY batches whose core reads q with validity-reporting copies (mla.qvalid):
# one thread's bulk copies per round instead of every softmax / correction thread
# spinning on the rows while they land, which slows the 16 ranks' multicast landing.
# Their MlaQuery publishes under qvalid's contract (``MlaQuery(valid=True)``). Value =
# mode: 1, bulk copies into the Q operand's shared memory, then the Q TMA; 2, the Q
# TMA itself reports validity (``polled_core``'s ``ts_valid_tma``, grid-constant maps),
# so q is loaded once.
VALID_VERIFY = batch_table("16:2,24:2,32:2,40:2,48:2,56:2,64:2")


def polled_q_core(batch, valid=0, persistent=False, push=None, long=None, wlist=False):
    """``polled_core`` for packed verify rows: each CTA of the 2-CTA pair polls its
    64-row half of the pair's packed M128 q tile (``load_tma_q``'s ``partition_A`` of
    tile blk_coord[1]) and re-arms no x; ``valid``: the VALID_VERIFY mode (1: validity-
    reporting copies, 2: the validity-reporting Q TMA); ``long``: mla.dcp.LONG_SPLIT's
    length-keyed partition; ``wlist``: the work list (``polled_core``)."""
    from tokenspeed_mla.mla_helpers import create_mla_static_tile_scheduler

    from . import core_ts  # (imports tokenspeed's kernel module: on first use)

    base = core_ts.polled_core(
        batch, valid=valid, persistent=persistent, push=push, long=long, wlist=wlist
    )

    class TsCoreQ(base):
        ts_poll = core_ts.POLL_Q  # False only in the tests' positive control
        ts_scheduler = staticmethod(create_mla_static_tile_scheduler)
        ts_invalidate = staticmethod(core_ts.invalidate_barrier)

        @cute.jit
        def ts_q_first(self, blk_coord):
            """``valid`` 2: ``pack_q``'s first row of this CTA's half of tile
            blk_coord[1] (rows past q x H: zero-filled by the TMA, reported clean)."""
            half_rows: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            return blk_coord[1] * self.mma_qk_tiler[0] + (blk_coord[0] % 2) * half_rows

        @cute.jit
        def ts_valid_q(self, table, blk_coord, q_latent, q_rope, spare):
            """``valid``: ``polled_core.ts_valid_q`` (one thread's rounds of validity-
            reporting bulk copies of this CTA's q rows into the Q operand's shared
            memory, re-issued while a copied 16 B chunk is armed; the Q TMA follows)
            on ``pack_q``'s row map: request blk_coord[2], rows blk_coord[1] * 128 +
            (blk_coord[0] % 2) * 64 .., contiguous in the mailbox."""
            rows_total: cutlass.Constexpr = self.total_q_rows
            half_rows: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            latent: cutlass.Constexpr = half_rows * 512  # the CTA's Q latent bytes
            first = blk_coord[1] * self.mma_qk_tiler[0] + (blk_coord[0] % 2) * half_rows
            rows = min(half_rows, rows_total - first)
            if rows > 0:
                q = self.ts_address(table, 0)
                base = q + Int64(blk_coord[2] * rows_total + first) * 576
                nbytes = rows * 576
                head = min(nbytes, latent)
                barrier = (Int32(spare.toint()) + 7) & Int32(-8)
                init_valid(barrier)
                cute.arch.mbarrier_init_fence()
                phase = Int32(0)
                armed = Int32(1)
                while armed != 0:
                    expect_valid(barrier, nbytes)
                    copy_valid(q_latent, base, head, barrier)
                    if nbytes > head:
                        copy_valid(q_rope, base + Int64(head), nbytes - head, barrier)
                    armed = wait_valid(barrier, phase)
                    phase = phase ^ 1
                self.ts_invalidate(barrier)

        @cute.jit
        def ts_poll_q(self, table, split_kv, cache_seqs, tile_sched_params, poller):
            """Poll this CTA's packed q rows (request blk_coord[2], rows
            blk_coord[1] * 128 + (blk_coord[0] % 2) * 64 ..) until no 32 B chunk
            holds the sentinel, then arrive on ts_q_bar (``polled_core.ts_poll_q``
            with ``pack_q``'s row map)."""
            rows_total: cutlass.Constexpr = self.total_q_rows
            half_rows: cutlass.Constexpr = self.mma_qk_tiler[0] // 2
            per: cutlass.Constexpr = -(-half_rows * 18 // self.ts_pollers)
            tile_sched = self.ts_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            if work_tile.is_valid_tile:
                blk_coord = self.ts_item(work_tile.tile_idx, tile_sched, table)
                _, count, _ = self.get_k_tile_count(
                    split_kv, cache_seqs, table, blk_coord
                )
                if count > 0:  # (the load-Q warp waits under the same condition)
                    first = blk_coord[1] * self.mma_qk_tiler[0]
                    first += (blk_coord[0] % 2) * half_rows
                    rows = min(half_rows, rows_total - first)
                    # (a partial last tile: TMA zero-fills the rows past the total)
                    if rows > 0 and cutlass.const_expr(self.ts_poll):
                        last = rows * 18 - 1  # 32 B chunks of 576 B rows
                        q = self.ts_address(table, 0)
                        base = q + Int64(blk_coord[2] * rows_total + first) * 576
                        for i0 in cutlass.range_constexpr(0, per, 8):
                            n: cutlass.Constexpr = min(8, per - i0)
                            ptrs = [
                                base
                                + Int64(min(poller + (i0 + i) * self.ts_pollers, last))
                                * 32
                                for i in range(n)
                            ]
                            loads = [poll_b32x8(p) for p in ptrs]
                            for i in cutlass.range_constexpr(n):
                                words = loads[i]
                                while not published(words, sentinel=Q_SENT):
                                    words = poll_b32x8(ptrs[i])
                    self.ts_q_bar.arrive()

        @cute.jit
        def ts_arm_x(self, table, lane):
            """MlaQuery reads x after its own PDL wait at these batches: nothing to
            re-arm."""

    return TsCoreQ


def verify_table(query_box, reqs, device):
    """The core's address table (``mla.dcp.fork_table``'s layout, one word per request
    at least): this parity's q mailbox address in words 0-1; the x words are unused."""
    address = query_box.data_ptr()
    words = [address & 0xFFFFFFFF, address >> 32] + [0] * (max(8, reqs) - 2)
    table = torch.tensor(words, dtype=torch.int64).to(torch.int32)  # wraps
    return table.to(device)


def push_table(query_box, reqs, box, parity, device, flags):
    """``verify_table`` (words 0-1: this parity's q mailbox address) plus, per owner
    rank o (words 2 + 4 o ..): the address of this rank's source block of o's partial
    box at ``parity`` ([2, DCP, CAP, H, 512] BF16: row b, head h at (b H + h) x 1 KB
    past it) and of o's LSE box at ``parity`` (the spread layout [2, CAP, H, DCP]:
    row b, head h at ((b H + h) DCP + source) x 4 B past it), then (words 2 + 4 DCP ..)
    ``flags``: the core push's per-block ready flags (zeroed int32, one per persistent
    work block; ``core_ts.polled_core`` ``push``)."""
    D, H, cap, rank = box.size, box.heads, box.cap, box.rank
    partial = ((parity * D + rank) * cap * H * 512) * 2
    lse = parity * cap * H * D * 4
    words = [query_box.data_ptr()]
    for owner in range(D):
        words += [box.peer_ptrs[owner] + partial, box.lse_peer_ptrs[owner] + lse]
    words.append(flags.data_ptr())
    split = []
    for address in words:
        split += [address & 0xFFFFFFFF, address >> 32]
    split += [0] * (max(8, reqs) - len(split))
    table = torch.tensor(split, dtype=torch.int64).to(torch.int32)  # wraps
    return table.to(device)


def compile_polled_q(
    heads, page, split, reqs, q, cp, device, causal=True, valid=0, persistent=False,
    push=None, wlist=False,
):  # fmt: skip
    """``compile_split_partials_q`` with ``polled_q_core``'s polled q intake (the
    split argument slot carries ``verify_table``; ``valid``: the VALID_VERIFY mode;
    ``persistent``: tokenspeed's persistent scheduler, valid 2 only). One compile per
    argument set, however the caller spells them. Split 1 compiles without a
    workspace: the kernel writes the BF16 rows and LSEs straight into ``out`` / ``lse``
    (a workspace would take FP32 partials of the one split, for a reducer). ``push``
    (split 1, persistent): the core push's ``early`` (``polled_core``; the slot carries
    ``push_table``); None: no core push. ``wlist`` (persistent, no push): the work
    list after the table's address words (``polled_core``)."""
    mode = int(valid)
    key = heads, page, split, reqs, q, cp, device, causal, mode
    if push is not None:
        assert split == 1 and persistent, "the core push: split 1, persistent only"
        where = cp[0], cp[1], heads // cp[0], q, int(push)
        return _compile_polled_q(*key, persistent, where)
    if wlist:
        assert persistent, "the work list: persistent only"
        return _compile_polled_q(*key, True, None, True)
    if persistent:  # (a separate key: the non-persistent compiles stay as they were)
        return _compile_polled_q(*key, True)
    return _compile_polled_q(*key)


@cache
def _compile_polled_q(
    heads, page, split, reqs, q, cp, device, causal, valid, persistent=False, push=None,
    wlist=False,
):  # fmt: skip
    from tokenspeed_mla import mla_decode as md

    base = md.BlackwellMultiHeadLatentAttentionForwardFP8
    sms = md.get_num_sm(torch.device("cuda", device))
    long = long_split(reqs * q, heads, split, sms)
    md.BlackwellMultiHeadLatentAttentionForwardFP8 = _without_reducer(
        polled_q_core(reqs * q, valid, persistent, push, long, wlist)
    )
    d_tiles = md._get_reducer_d_tiles(reqs, q, heads, sms, split)
    try:
        with torch.accelerator.device_index(device):
            return md._get_compiled_mla_kernel.__wrapped__(
                torch_dtype=torch.float8_e4m3fn,
                page_size=page,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                is_persistent=persistent,
                is_var_seq=True,
                is_var_split_kv=True,  # the table's (unused split) argument
                compute_capability=torch.cuda.get_device_capability(device),
                has_local_visible_lens=False,
                skip_correction_threshold=8.0,
                is_workspace_size_zero=split == 1,
                fold_sq_factor=1,
                causal_mask=causal,
                num_heads=heads,
                seq_len_q=q,
                window_left=-1,
                cp_world=cp[0],
                cp_rank=cp[1],
                cp_interleave_size=cp[2],
                use_pdl=True,
                return_lse=True,
                reducer_d_tiles=d_tiles,
                reducer_max_splits=md._get_reducer_max_splits(split),
                pack_q=True,
                partial_fp16=False,
            )
    finally:
        md.BlackwellMultiHeadLatentAttentionForwardFP8 = base


def polled_partials_q(
    query, cache, block_table, lengths, totals, scales, out, lse, split, cp, table,
    causal=True, valid=0, persistent=False, push=None, wlist=False,
):  # fmt: skip
    """``split_partials_q`` with the polled q intake: ``query`` is the (ungathered)
    q mailbox view [requests, q, H, 576]; ``table``: ``verify_table`` of it;
    ``valid``: the VALID_VERIFY mode (validity-reporting copies / Q TMA);
    ``persistent``: tokenspeed's persistent scheduler (valid 2). Split 1 writes the
    BF16 rows and LSEs into ``out`` / ``lse``; more splits leave FP32 partials and
    split LSEs in the workspace for DcpValue's fold."""
    import tvm_ffi
    from tokenspeed_mla import mla_decode as md

    reqs, q, heads, _ = query.shape
    device, page = query.device, cache.shape[1]
    key = heads, page, split, reqs, q, cp, device.index
    kernel = compile_polled_q(
        *key, causal, valid=valid, persistent=persistent, push=push, wlist=wlist
    )
    rows = md.compute_q_tile_layout(heads, q, 128)[1] * 128
    # (split 1: no workspace, the rows and LSEs go straight to ``out`` / ``lse``)
    buf = split_workspace_ts(device, rows, reqs, split)[0] if split > 1 else None
    cache = cache.view(torch.float8_e4m3fn)
    softmax, output_scale = scales
    args = [query[..., :512], query[..., 512:], cache[..., :512], cache[..., 512:]]
    args += [block_table, out, lse, buf, Int32(split), lengths, totals, table[:reqs]]
    args += [Float32(softmax), Float32(output_scale)]
    with tvm_ffi.use_torch_stream():
        kernel(*args)
