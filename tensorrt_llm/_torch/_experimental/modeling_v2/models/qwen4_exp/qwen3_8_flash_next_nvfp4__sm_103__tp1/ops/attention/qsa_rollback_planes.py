# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The speculative rollback of every attention layer's QSA index-K plane in one launch each way:
gather the verify rows' slots of all planes before the trunk writes them, and after the
draft loop write the gathered values back into the slots of the rows the sampler rejected.

Tier 5 (Triton written for this target). The planes are separate buffers of the cache
manager (one per attention layer, each `[pages, tokens_per_block, heads, dim]` with a free
page stride and a contiguous tail); the kernels reach them through a table of their base
addresses and page strides, one program per (plane, row). The composition this replaces ran
one gather per layer during the trunk and a gather, a `torch.where` and an indexed write per
layer after the draft loop.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's index plane row: one index KV head of 128 bf16.
ROW = 128


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
    CELLS: tuple[Cell, ...] = (
        Cell(why="12 planes, 4 verify rows (MTP3, one request)", spec=dict(planes=12, rows=4)),
        Cell(why="12 planes, 512 verify rows (MTP3, 128 requests)", spec=dict(planes=12, rows=512)),
    )
    note = """
    Metadata consumed: none. Call site: `SpeculativeRollback.begin`, before the verify step's
    trunk writes any plane. One program per (plane, row), one 256-byte row each.
    Preconditions: `pages` / `within` int64 `[R]`; the planes as `plane_table` accepts them.
    """

    def __call__(self, ptrs, page_strides, pages, within, num_planes: int) -> torch.Tensor:
        rows = pages.shape[0]
        out = torch.empty([num_planes, rows, ROW], dtype=torch.bfloat16, device=pages.device)
        if rows:
            _gather_kernel[(num_planes, rows)](
                ptrs, page_strides, pages, within, out, rows, ROW=ROW, num_warps=1
            )
        return out

    def reference(self, ptrs, page_strides, pages, within, num_planes: int) -> torch.Tensor:
        raise NotImplementedError("the cell harness builds planes and compares against torch")

    def is_valid(self, ptrs, page_strides, pages, within, num_planes: int) -> None:
        assert ptrs.dtype == torch.int64 and ptrs.numel() == num_planes
        assert page_strides.dtype == torch.int64 and page_strides.numel() == num_planes
        assert pages.dtype == torch.int64 and within.dtype == torch.int64
        assert pages.shape == within.shape and pages.dim() == 1


class QsaRollbackRestore(OpWrapper):
    """For every row `r` with `rejected[r]` and every plane `l`: `plane_l[pages[r], within[r]]
    = old[l, r]` -- the inverse of `qsa_rollback_gather` over the rejected rows. Rows not
    rejected are not touched.

    Fusion boundary: the restore of all planes; `rejected` is the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="12 planes, 4 verify rows, two rejected", spec=dict(planes=12, rows=4)),
        Cell(why="12 planes, 512 verify rows", spec=dict(planes=12, rows=512)),
    )
    note = """
    Metadata consumed: none. Call site: `SpeculativeRollback.commit_speculative_states`,
    after the draft loop. `rejected` int32/int8/bool `[R]` (nonzero = restore).
    """

    def __call__(self, ptrs, page_strides, pages, within, rejected, old) -> None:
        num_planes, rows, _ = old.shape
        if rows:
            _restore_kernel[(num_planes, rows)](
                ptrs, page_strides, pages, within, rejected, old, rows, ROW=ROW, num_warps=1
            )

    def reference(self, ptrs, page_strides, pages, within, rejected, old) -> None:
        raise NotImplementedError("the cell harness builds planes and compares against torch")

    def is_valid(self, ptrs, page_strides, pages, within, rejected, old) -> None:
        assert old.dtype == torch.bfloat16 and old.dim() == 3 and old.shape[2] == ROW
        assert old.is_contiguous()
        assert ptrs.numel() == old.shape[0] and page_strides.numel() == old.shape[0]
        assert pages.shape == (old.shape[1],) and within.shape == (old.shape[1],)
        assert rejected.shape == (old.shape[1],)


qsa_rollback_gather = QsaRollbackGather()
qsa_rollback_restore = QsaRollbackRestore()
