# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative rollback of every attention layer's QSA index-K plane in one launch each way:
gather the verify rows' slots of all planes before the trunk writes them, and after the
draft loop write the gathered values back into the slots of the rows the sampler rejected.

Tier 5 (Triton written for the qwen4_exp tp1 target, `qwen3_8_flash_next_nvfp4__sm_103__tp1`,
and promoted from its copy and the unified 4-GPU target's). The
planes are separate buffers of the cache manager (one per attention layer, each
`[pages, tokens_per_block, heads, dim]` with a free page stride and a contiguous tail); the
kernels reach them through a table of their base addresses and page strides, one program per
(plane, row). The composition this replaces ran one gather per layer during the trunk and a
gather, a `torch.where` and an indexed write per layer after the draft loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper

#: This checkpoint's index plane row: one index KV head of 128 bf16.
ROW = 128

#: The index planes of core's bf16 K/V pool on every member of the 4-GPU target, as the stock
#: `Qwen4ExpHybridCacheManager` (KV cache manager V2) allocates them: the 12 attention layers'
#: `[32, 1, 128]` INDEX_KEY buffers coalesce into one pool, layer k's buffer at element 4096 k of
#: a 49152-element page, so every plane is a `[pages, 32, 1, 128]` view at page stride 49152.
#: 257109 pages is a tep4 rank's attention pool at core's configuration, which the index pool
#: shares: the stock engine's `maxNumBlocks` 258983 less the 1874 recurrent-state slots that
#: total also counts. The largest measured per member.
_TEP4_POOL = dict(planes=12, layout="coalesced", pool_pages=257109)


def _verify(requests: int, p: int) -> dict:
    """A verify step's rows: `requests` generation requests of `p` rows (`max_draft_len + 1`)."""
    return dict(_TEP4_POOL, rows=requests * p, requests=requests, p=p, rejected="bool")


#: The cells both halves are certified at: the tp1 target's own, and the unified target's MTP verify
#: rows at `max_batch_size` 128 over a tep4 rank's pool.
_CELLS: tuple[Cell, ...] = (
    Cell(why="12 planes, 4 verify rows (MTP3, one request)", spec=dict(planes=12, rows=4)),
    Cell(why="12 planes, 512 verify rows (MTP3, 128 requests)", spec=dict(planes=12, rows=512)),
    Cell(
        why="tep4 MTP1 verify: 128 requests x 2 rows (256) over the 257109-page coalesced pool",
        spec=_verify(128, 2),
    ),
    Cell(
        why="tep4 MTP7 verify: 128 requests x 8 rows (1024, the envelope's largest), same pool",
        spec=_verify(128, 8),
    ),
)


@triton.jit
def _gather_kernel(ptrs, page_strides, pages, within, out, rows, ROW: tl.constexpr):
    plane = tl.program_id(0)
    r = tl.program_id(1)
    base = tl.load(ptrs + plane).to(tl.pointer_type(tl.bfloat16))
    stride = tl.load(page_strides + plane)
    page = tl.load(pages + r)
    tok = tl.load(within + r)
    offs = tl.arange(0, ROW)
    vals = tl.load(base + page * stride + tok * ROW + offs)
    tl.store(out + (plane.to(tl.int64) * rows + r) * ROW + offs, vals)


@triton.jit
def _restore_kernel(ptrs, page_strides, pages, within, rejected, old, rows, ROW: tl.constexpr):
    plane = tl.program_id(0)
    r = tl.program_id(1)
    if tl.load(rejected + r) != 0:
        base = tl.load(ptrs + plane).to(tl.pointer_type(tl.bfloat16))
        stride = tl.load(page_strides + plane)
        page = tl.load(pages + r)
        tok = tl.load(within + r)
        offs = tl.arange(0, ROW)
        vals = tl.load(old + (plane.to(tl.int64) * rows + r) * ROW + offs)
        tl.store(base + page * stride + tok * ROW + offs, vals)


def plane_table(planes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """`(base addresses, page strides)` of the planes, int64 on their device (built once per
    cache-manager bind: the planes do not move)."""
    dev = planes[0].device
    for t in planes:
        assert t.dtype == torch.bfloat16 and t.dim() == 4, "index planes [pages, tpb, H, D] bf16"
        assert t.shape[2] * t.shape[3] == ROW, "one 128-wide row a token"
        assert t.stride(1) == ROW and t.stride(2) == t.shape[3] and t.stride(3) == 1, (
            "contiguous [tpb, H, D] tail"
        )
    ptrs = torch.tensor([t.data_ptr() for t in planes], dtype=torch.int64, device=dev)
    strides = torch.tensor([t.stride(0) for t in planes], dtype=torch.int64, device=dev)
    return ptrs, strides


class QsaRollbackGather(OpWrapper):
    """`torch.stack([plane[pages, within] for plane in planes])` as `[L, R, 128]` bf16, with
    the planes given as `plane_table(planes)`: every row's slot of every plane, in plane
    order. Nothing is written but the output.

    Fusion boundary: the gather of all planes; the slots (`pages`, `within`) are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _CELLS
    note = """
    Metadata consumed: none. Call site: `SpeculativeRollback.begin`, before the verify step's
    trunk writes any plane; `pages` / `within` are the generation rows' QSA page and offset
    (`block_table[req, pos // 32]`, `pos % 32`), int64 `[R]`, R = requests x (max_draft_len + 1).
    One program per (plane, row), one 256-byte row each. Preconditions: the planes as
    `plane_table` accepts them (bf16 NHD views, any page stride); every (page, within) inside
    the pool.

    Triton, JIT-compiled at first call for the running device (sm_103 here).

    Addressing is 64-bit: `pages` and the page strides are int64, so a slot past 2^31 elements
    from a plane's base is reached. The tep4 cells gather from (and restore into) the pool's
    last page, 257108 x 49152 = 1.26e10 elements past every plane's base, and its first.

    Layouts certified: separate contiguous planes and the packed side-plane page of an FP8 /
    NVFP4 pool (page stride 12 x 4096 + 192) at the tp1 cells' 4 and 512 rows over 600 pages;
    core's coalesced bf16 pool (page stride 49152) at 256 and 1024 rows over 257109 pages.
    `test_cuda_graph_replay` captures the 1024-row call and replays it at new slots.

    Hard limits (not probed): rows is the launch grid's y extent, at most 65535; out-of-pool or
    duplicated slots are the caller's. Not certified: a pool page count measured under the MTP
    unit's own engine (`max_seq_len` 16384; the cells use core's tep4 count), and the
    side-plane layout at 256 / 1024 rows.
    """

    def __call__(self, ptrs, page_strides, pages, within, num_planes: int) -> torch.Tensor:
        rows = pages.shape[0]
        out = torch.empty([num_planes, rows, ROW], dtype=torch.bfloat16, device=pages.device)
        if rows:
            _gather_kernel[(num_planes, rows)](
                ptrs, page_strides, pages, within, out, rows, ROW=ROW, num_warps=1
            )
        return out

    def reference(self, planes, pages, within, num_planes: int) -> torch.Tensor:
        """Every row's slot of the first `num_planes` planes, by torch's advanced indexing.

        Takes the planes themselves where `__call__` takes `plane_table(planes)`: torch
        cannot dereference the raw base addresses the kernel is handed, so the reference reads
        the tensors the table was built from. A copy, exact.
        """
        return torch.stack([t[pages, within].reshape(-1, ROW) for t in planes[:num_planes]])

    def is_valid(self, ptrs, page_strides, pages, within, num_planes: int) -> None:
        assert ptrs.dtype == torch.int64 and ptrs.numel() == num_planes, (
            "ptrs must be plane_table's int64 table of exactly num_planes addresses; a "
            "num_planes past it reads addresses beyond the table"
        )
        assert page_strides.dtype == torch.int64 and page_strides.numel() == num_planes, (
            "page_strides must be plane_table's int64 table of num_planes strides"
        )
        assert pages.dtype == torch.int64 and within.dtype == torch.int64, (
            "pages and within must be int64"
        )
        assert pages.shape == within.shape and pages.dim() == 1, (
            "pages and within must be one [R] pair: rows are counted off pages, so a shorter "
            "within is read past its end"
        )


class QsaRollbackRestore(OpWrapper):
    """For every row `r` with `rejected[r]` and every plane `l`: `plane_l[pages[r], within[r]]
    = old[l, r]` -- the inverse of `qsa_rollback_gather` over the rejected rows. Rows not
    rejected are not touched.

    Fusion boundary: the restore of all planes; `rejected` is the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _CELLS
    note = """
    Metadata consumed: none. Call site: `SpeculativeRollback.commit_speculative_states`,
    after the draft loop: `old` is the begin step's `qsa_rollback_gather` output, `rejected`
    the bool `[R]` mask `ordinal >= accepted` (int32 / int8 / bool, nonzero = restore; the tp1
    cells drive int32, the tep4 cells the call site's bool).

    Returns nothing: it writes the planes, which it holds only as `plane_table` addresses.
    Triton, JIT-compiled at first call for the running device (sm_103 here).

    Every cell checks that the rejected rows' slots hold `old`, the others the step's values,
    and that nothing else in the pool was written. Addressing, layouts and the CUDA-graph
    replay as `qsa_rollback_gather`'s note states; the same limits.

    Silent, refused by `is_valid`: a non-contiguous `old` (read as dense `[L, R, 128]`;
    measured wrong in `test_guard_refuses_a_strided_old`).
    """

    def __call__(self, ptrs, page_strides, pages, within, rejected, old) -> None:
        num_planes, rows, _ = old.shape
        if rows:
            _restore_kernel[(num_planes, rows)](
                ptrs, page_strides, pages, within, rejected, old, rows, ROW=ROW, num_warps=1
            )

    def reference(self, planes, pages, within, rejected, old) -> torch.Tensor:
        """What each row's slot of each plane holds after the call, `[L, R, 128]`: `old` where
        `rejected`, else the slot's value now. Read before the call, as the planes stand.

        Takes the planes where `__call__` takes `plane_table(planes)`, for the reason
        `QsaRollbackGather.reference` states. Every location outside these slots is unchanged,
        which the test checks over the whole pool.
        """
        now = torch.stack([t[pages, within].reshape(-1, ROW) for t in planes[: old.shape[0]]])
        return torch.where(rejected.bool()[None, :, None], old, now)

    def is_valid(self, ptrs, page_strides, pages, within, rejected, old) -> None:
        assert old.dtype == torch.bfloat16 and old.dim() == 3 and old.shape[2] == ROW, (
            "old must be qsa_rollback_gather's [L, R, 128] bf16 output"
        )
        assert old.is_contiguous(), (
            "old must be contiguous; the kernel reads it as dense [L, R, 128] and a strided "
            "view restores the wrong values"
        )
        assert ptrs.numel() == old.shape[0] and page_strides.numel() == old.shape[0], (
            "plane_table must hold one address and one stride per plane of old"
        )
        assert pages.shape == (old.shape[1],) and within.shape == (old.shape[1],), (
            "pages and within must hold one slot per row of old"
        )
        assert rejected.shape == (old.shape[1],), "rejected must hold one flag per row of old"


qsa_rollback_gather = QsaRollbackGather()
qsa_rollback_restore = QsaRollbackRestore()
