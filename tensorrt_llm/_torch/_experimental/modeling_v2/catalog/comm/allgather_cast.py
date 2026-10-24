# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vocab-dim all-gather of bf16 logit shards into fp32 full-vocabulary rows, in one launch.

``out = torch.cat(all_gather(shard), -1).float()``: a column-parallel LM head's verify logits as the sampler takes
them, without the NCCL all-gather, the concat copy and the cast. Every rank pushes its shard through a multicast
NVLink workspace and raises per-CTA ready flags (the epoch of the call), so no bit pattern is reserved as a
sentinel and the result is the stock path's bit for bit, -0.0 and NaN payloads included.

``allgather_f32`` (MC-ALLGATHER): the same gather for fp32 shards, copied as is, ``out = torch.cat(all_gather(shard),
-1)``: the drafter's sharded draft logits as the rejection sampler takes them (fp32 [B, K, S] -> [B, K, world * S]),
without the NCCL all-gather and the concat copy. It shares the workspace (a slot holds MAX_ROWS / 2 fp32 rows).

Compiled on the first call per shape, which must run outside CUDA-graph capture (``precompile``). The workspace is
collective on its first use: every rank of the TP group calls ``workspace_args`` at the same point.
"""

from __future__ import annotations

import importlib
import os
import threading
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from .._op import Arch, Cell, OpWrapper

MAX_ROWS = 64  # capacity: fixed: the workspace's rows per call (K3 verify rows at batch 16)
MAX_GRID = 152  # capacity: fixed: the workspace's flag columns per rank
# capacity: fixed: fp32 rows per call (an fp32 row is twice a bf16 row's bytes)
MAX_ROWS_FP32 = MAX_ROWS // 2
# Launched without programmatic dependent launch: a plain launch waits for all prior stream work, so the previous
# call has completed whatever kernels run between two calls (the kernel's ordering condition holds by construction).
# It still releases its own dependents early. ``pdl=1`` on a call launches it with PDL (measurement only).
LAUNCH_PDL = False

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}
_workspaces: Dict[object, dict] = {}
_modules: Dict[str, object] = {}


def _kernel_module():
    """The CuTe DSL kernel beside this file, imported on first use (it pulls in cutlass)."""
    mod = _modules.get("kernel")
    if mod is None:
        mod = _modules["kernel"] = importlib.import_module(".allgather_cast_kernel", __package__)
    return mod


# CTAs by world size and rows: the fastest bitwise grid measured at each K3 verify row count on GB300 (graph-us at
# 96 / 128 / 152 CTAs). TP8 over two trays (RESULT_M31_N3A1K_MICROBENCH + its addendum-4 sweep): 4 rows 12.54 / 12.83 /
# 12.88, 8 rows 18.79 / 16.55 / 16.56, 16 rows 25.27 / 23.25 / 24.10, the other row counts fastest at 152. TP4 on one
# tray (RESULT_M31_N3A1K4_MICROBENCH): 4 rows 12.24 / 11.90 / 11.96, 8 rows 15.22 / 14.29 / 14.61, 16 rows 22.02 /
# 22.43 / 20.47, 32 rows 36.66 / 35.25 / 34.48 (96 is kept at 4 rows: within 0.5 us). Row counts in between take the
# next measured one's grid, rows past the table MAX_GRID; other world sizes take the TP8 table. A function of the
# world size, the rows and the device only, so every rank picks the same grid.
GRID_BY_ROWS = ((4, 96), (8, 128), (12, 152), (16, 128), (32, 152))
GRID_BY_WORLD_ROWS = {4: ((4, 96), (8, 128), (32, 152)), 8: GRID_BY_ROWS}


def pick_grid(rows: int, device=None, world: int = 8) -> int:
    """CTAs for a call of ``rows`` rows over ``world`` ranks (capped by the SM count and MAX_GRID)."""
    sms = torch.cuda.get_device_properties(
        device if device is not None else torch.cuda.current_device()
    ).multi_processor_count
    table = GRID_BY_WORLD_ROWS.get(world, GRID_BY_ROWS)
    grid = next((g for bound, g in table if rows <= bound), MAX_GRID)
    return min(MAX_GRID, sms, grid)


def workspace_args(mapping, shard_cols: int, group=None) -> dict:
    """This TP group's workspace as the call's ``ws_*``, ``rank`` and ``world`` arguments.

    Three rotating buffers of [world][MAX_ROWS, shard_cols] bf16 and the [world][MAX_GRID] ready flags behind one
    multicast mapping, plus the rank-local epoch words; built once per (mapping, shard_cols, group) and reused.
    Collective on first use, outside CUDA-graph capture: every rank of the TP group calls it at the same point.
    ``group`` overrides the communicator (a torch.distributed group when MPI is disabled).
    """
    key = (mapping, shard_cols, id(group) if group is not None else None)
    ws = _workspaces.get(key)
    if ws is not None:
        return ws["args"]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "allgather_cast: the workspace must be allocated outside CUDA-graph capture"
        )
    from tensorrt_llm._torch.distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )

    kern = _kernel_module()
    world = mapping.tp_size
    comm = group if group is not None else _get_mnnvl_workspace_comm(mapping)
    use_fabric_handle = (
        os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    )
    error: Optional[Exception] = None
    try:
        words = kern.workspace_words(MAX_ROWS, shard_cols, world, MAX_GRID)
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, use_fabric_handle)
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        with torch.inference_mode():
            uc.zero_()
            flags = torch.zeros(kern.FLAG_WORDS, dtype=torch.int32, device=uc.device)
        torch.cuda.synchronize()
        args = dict(ws_uc=uc, ws_mc=mc, ws_flags=flags, rank=mapping.tp_rank, world=world)
        ws = dict(handle=handle, comm=comm, args=args)
    except Exception as exc:  # noqa: BLE001 -- reported to every rank below, then re-raised
        error = exc
    # Also the barrier that keeps any rank from raising a peer's flag before the peer has zeroed it.
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError("allgather_cast workspace failed on at least one rank") from error
    _workspaces[key] = ws
    return ws["args"]


def _arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align).mark_layout_dynamic(leading_dim=0)


def supports(shard: torch.Tensor, world: int) -> bool:
    """Whether the kernel takes this shard: bf16 [rows <= MAX_ROWS, S] contiguous, S a multiple of 8, on CUDA."""
    return (
        shard.is_cuda
        and shard.dim() == 2
        and shard.dtype == torch.bfloat16
        and shard.is_contiguous()
        and 0 < shard.shape[0] <= MAX_ROWS
        and shard.shape[1] % 8 == 0
        and shard.data_ptr() % 16 == 0
        and 1 < world <= 8
    )


def supports_f32(shard: torch.Tensor, world: int) -> bool:
    """Whether ``allgather_f32`` takes this shard: fp32 [rows <= MAX_ROWS_FP32, S] contiguous, S a multiple of 4."""
    return (
        shard.is_cuda
        and shard.dim() == 2
        and shard.dtype == torch.float32
        and shard.is_contiguous()
        and 0 < shard.shape[0] <= MAX_ROWS_FP32
        and shard.shape[1] % 4 == 0
        and shard.data_ptr() % 16 == 0
        and 1 < world <= 8
    )


def _consts(rows: int, cols: int, world: int, grid: int, fp32: bool = False) -> tuple:
    kern = _kernel_module()
    return (
        grid,
        rows,
        cols,
        world,
        kern.slot_words(MAX_ROWS, cols),
        kern.flag_base(MAX_ROWS, cols, world),
        MAX_GRID,
        bool(fp32),
    )


def _default_grid(rows: int, device, world: int, fp32: bool) -> int:
    """An fp32 row moves a bf16 row's bytes twice: it takes the grid of twice the rows."""
    return pick_grid(2 * rows if fp32 else rows, device, world)


def _kernel_for(
    shard,
    out,
    ws_uc,
    ws_mc,
    ws_flags,
    rank: int,
    world: int,
    grid: int,
    pdl: int,
    fp32: bool = False,
):
    """The compiled launcher of this call's shape (compiled here on first use, outside capture), its tensor
    arguments and the stream."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute

    kern = _kernel_module()
    if not (supports_f32(shard, world) if fp32 else supports(shard, world)):
        raise ValueError(f"allgather_cast: unsupported shard {tuple(shard.shape)} {shard.dtype}")
    rows, cols = shard.shape
    if grid <= 0:
        grid = _default_grid(rows, shard.device, world, fp32)
    if not 0 < grid <= MAX_GRID or ws_uc.numel() < kern.workspace_words(
        MAX_ROWS, cols, world, MAX_GRID
    ):
        raise ValueError("allgather_cast: the call does not fit the workspace")
    args = (
        _arg(shard.view(-1).view(torch.int32)),
        _arg(out.view(-1).view(torch.int32)),
        _arg(ws_uc.view(-1)),
        _arg(ws_mc.view(-1)),
        _arg(ws_flags.view(-1), align=4),
    )
    use_pdl = LAUNCH_PDL if pdl < 0 else bool(pdl)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(shard.device).cuda_stream)
    consts = _consts(rows, cols, world, grid, fp32)
    key = consts + (use_pdl,)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::allgather_cast must run once per shape outside CUDA-graph capture first"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kern.allgather_cast, *args, int(rank), *consts, use_pdl, stream
                )
    return fn, args, stream


@torch.library.custom_op("trtllm::allgather_cast", mutates_args=())
def _allgather_cast(
    shard: torch.Tensor,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    world: int,
    grid: int,
    pdl: int,
) -> torch.Tensor:
    out = torch.empty(
        shard.shape[0], world * shard.shape[1], dtype=torch.float32, device=shard.device
    )
    fn, args, stream = _kernel_for(shard, out, ws_uc, ws_mc, ws_flags, rank, world, grid, pdl)
    fn(*args, int(rank), stream)
    return out


@_allgather_cast.register_fake
def _(shard, ws_uc, ws_mc, ws_flags, rank, world, grid, pdl):
    return shard.new_empty((shard.shape[0], world * shard.shape[1]), dtype=torch.float32)


@torch.library.custom_op("trtllm::allgather_f32", mutates_args=())
def _allgather_f32(
    shard: torch.Tensor,
    ws_uc: torch.Tensor,
    ws_mc: torch.Tensor,
    ws_flags: torch.Tensor,
    rank: int,
    world: int,
    grid: int,
    pdl: int,
) -> torch.Tensor:
    out = torch.empty(
        shard.shape[0], world * shard.shape[1], dtype=torch.float32, device=shard.device
    )
    fn, args, stream = _kernel_for(
        shard, out, ws_uc, ws_mc, ws_flags, rank, world, grid, pdl, fp32=True
    )
    fn(*args, int(rank), stream)
    return out


@_allgather_f32.register_fake
def _(shard, ws_uc, ws_mc, ws_flags, rank, world, grid, pdl):
    return shard.new_empty((shard.shape[0], world * shard.shape[1]), dtype=torch.float32)


def gather_last_dim_f32(logits: torch.Tensor, ws_uc, ws_mc, ws_flags, rank, world, grid=0, pdl=-1):
    """``torch.cat(all_gather(logits), -1)`` for an fp32 [..., S] shard (the leading dims flattened into rows), or
    None when ``allgather_f32`` does not take it (the caller then runs the stock all-gather; shapes decide, so every
    rank declines alike)."""
    lead = tuple(logits.shape[:-1])
    flat = logits.reshape(-1, logits.shape[-1])
    if not supports_f32(flat, world):
        return None
    out = torch.ops.trtllm.allgather_f32(
        flat, ws_uc, ws_mc, ws_flags, int(rank), int(world), int(grid), int(pdl)
    )
    return out.view(*lead, world * logits.shape[-1])


def is_compiled(
    rows: int, cols: int, world: int, grid: int = 0, device=None, fp32: bool = False
) -> bool:
    """Whether the (rows, cols, world, grid) shape is compiled (for the caller's capture-time decision)."""
    grid = grid if grid > 0 else _default_grid(rows, device, world, fp32)
    return _consts(rows, cols, world, grid, fp32) + (LAUNCH_PDL,) in _compiled


def precompile(
    mapping,
    cols: int,
    rows: Tuple[int, ...],
    device,
    group=None,
    agree: Optional[Callable[[bool], bool]] = None,
) -> bool:
    """Compile every row count in ``rows`` outside capture, then run each shape once (collective: every rank).

    The compiles launch nothing. ``agree`` (every rank's verdict, e.g. a TP all-reduce of success) is consulted
    between the compiles and the first launch, so a compile that fails on one rank turns the item off on every rank
    instead of leaving its peers polling for flags it never raises. Returns whether the shapes are ready."""
    ws = workspace_args(mapping, cols, group)
    ok = True
    try:
        for r in rows:
            shard = torch.zeros(r, cols, dtype=torch.bfloat16, device=device)
            out = torch.empty(r, ws["world"] * cols, dtype=torch.float32, device=device)
            _kernel_for(
                shard, out, ws["ws_uc"], ws["ws_mc"], ws["ws_flags"], ws["rank"], ws["world"], 0, -1
            )
    except Exception as exc:  # noqa: BLE001 -- reported to every rank through ``agree``
        from tensorrt_llm.logger import logger

        logger.warning(f"allgather_cast: compile failed on rank {ws['rank']}: {exc}")
        ok = False
    if agree is not None:
        ok = agree(ok)
    if not ok:
        return False
    for r in rows:
        allgather_cast(torch.zeros(r, cols, dtype=torch.bfloat16, device=device), **ws)
    torch.cuda.synchronize()
    return True


def precompile_f32(
    mapping,
    cols: int,
    rows: Tuple[int, ...],
    device,
    group=None,
    agree: Optional[Callable[[bool], bool]] = None,
) -> bool:
    """``precompile`` for ``allgather_f32`` (fp32 shards of ``cols`` columns, each row count in ``rows``)."""
    ws = workspace_args(mapping, cols, group)
    ok = True
    try:
        for r in rows:
            shard = torch.zeros(r, cols, dtype=torch.float32, device=device)
            out = torch.empty(r, ws["world"] * cols, dtype=torch.float32, device=device)
            _kernel_for(
                shard, out, ws["ws_uc"], ws["ws_mc"], ws["ws_flags"], ws["rank"], ws["world"], 0, -1,
                fp32=True,
            )  # fmt: skip
    except Exception as exc:  # noqa: BLE001 -- reported to every rank through ``agree``
        from tensorrt_llm.logger import logger

        logger.warning(f"allgather_f32: compile failed on rank {ws['rank']}: {exc}")
        ok = False
    if agree is not None:
        ok = agree(ok)
    if not ok:
        return False
    for r in rows:
        allgather_f32(torch.zeros(r, cols, dtype=torch.float32, device=device), **ws)
    torch.cuda.synchronize()
    return True


class AllgatherCast(OpWrapper):
    """fp32 [rows, world * S]: every rank's bf16 [rows, S] shard, in rank order along the last dim, widened to fp32.

    ``shard`` bf16 [rows, S] contiguous (rows <= MAX_ROWS, S a multiple of 8): this rank's vocabulary columns.
    ``ws_*``, ``rank``, ``world``: ``workspace_args``'s. ``grid``: CTAs (0 picks).
    ``pdl``: -1 launches as LAUNCH_PDL (off); 0 / 1 force it (measurement only).

    Every rank returns the same tensor. The call is a collective: every rank of the group calls it with the same
    rows, S and grid, in the same order.
    """

    ARCHS = frozenset({Arch.SM_103})

    # MiniMax-M3.1's target LM head at TP8 over two GB300 trays: vocabulary 200064 (shard 25008), K3 verify rows
    # B * (K + 1): K3 at batches 1..16, K7 at batches 1..8.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="M3.1 TP8, K3 verify at batch 1: 4 rows of the 25008-column shard",
            spec=dict(world=8, rows=4, cols=25008),
        ),
        Cell(
            why="M3.1 TP8, K7 verify at batch 1 (the headline c1): 8 rows",
            spec=dict(world=8, rows=8, cols=25008),
        ),
        Cell(
            why="M3.1 TP8, K7 verify at batch 2: 16 rows",
            spec=dict(world=8, rows=16, cols=25008),
        ),
        Cell(
            why="M3.1 TP8, K3 verify at batch 8: 32 rows",
            spec=dict(world=8, rows=32, cols=25008),
        ),
        Cell(
            why="M3.1 TP8, K3 verify at batch 16: 64 rows, the workspace's capacity",
            spec=dict(world=8, rows=64, cols=25008),
        ),
        Cell(
            why="a row count between the K3 batch sizes (odd vectors per CTA)",
            spec=dict(world=8, rows=7, cols=25008),
        ),
        Cell(
            why="one tray (TP4): the 50016-column shard",
            spec=dict(world=4, rows=4, cols=50016),
        ),
        Cell(
            why="one tray (TP4), K3 verify at batch 12: 48 rows",
            spec=dict(world=4, rows=48, cols=50016),
        ),
    )

    note = """
    Rows past MAX_ROWS (64) and shards that are not whole 16-byte vectors are declined (``supports``); the caller
    takes the stock all-gather. Every rank must make the same decision, and does: it depends only on shapes.
    """

    def __call__(
        self,
        shard: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        world: int,
        grid: int = 0,
        pdl: int = -1,
    ) -> torch.Tensor:
        return torch.ops.trtllm.allgather_cast(
            shard, ws_uc, ws_mc, ws_flags, int(rank), int(world), int(grid), int(pdl)
        )

    def reference(self, shards_in_rank_order: Sequence[torch.Tensor]) -> torch.Tensor:
        """The stock path: the column-parallel head's all-gather, the vocab-dim concat and ``.float()``.

        The documented collective exception to mirroring ``__call__``: it takes every rank's shard in rank order.
        """
        return torch.cat(list(shards_in_rank_order), dim=-1).float()

    def is_valid(
        self,
        shard: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        world: int,
        grid: int = 0,
        pdl: int = -1,
    ) -> None:
        assert supports(shard, world), (tuple(shard.shape), shard.dtype, shard.is_contiguous())
        assert 0 <= rank < world, (rank, world)


allgather_cast = AllgatherCast()


class AllgatherF32(OpWrapper):
    """fp32 [rows, world * S]: every rank's fp32 [rows, S] shard, in rank order along the last dim, copied as is.

    ``shard`` fp32 [rows, S] contiguous (rows <= MAX_ROWS_FP32, S a multiple of 4): this rank's vocabulary columns.
    ``ws_*``, ``rank``, ``world``: ``workspace_args``'s (the allgather_cast workspace of the same S). ``grid``: CTAs
    (0 picks). ``pdl``: as ``allgather_cast``'s. A collective, as ``allgather_cast``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # MiniMax-M3.1's DSpark drafter at TP8 over two GB300 trays: the rejection sampler's draft logits, B * K rows of
    # the 25008-column vocabulary shard.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="M3.1 TP8, K7 draft logits at batch 1 (the headline c1): 7 rows",
            spec=dict(world=8, rows=7, cols=25008),
        ),
        Cell(
            why="M3.1 TP8, K7 draft logits at batch 2: 14 rows",
            spec=dict(world=8, rows=14, cols=25008),
        ),
        Cell(
            why="M3.1 TP8, K3 draft logits at batch 8: 24 rows",
            spec=dict(world=8, rows=24, cols=25008),
        ),
        Cell(
            why="the fp32 capacity: 32 rows",
            spec=dict(world=8, rows=32, cols=25008),
        ),
        Cell(
            why="one tray (TP4): the 50016-column shard, K7 at batch 1",
            spec=dict(world=4, rows=7, cols=50016),
        ),
    )

    note = """
    Rows past MAX_ROWS_FP32 (32) and shards that are not whole 16-byte vectors are declined (``supports_f32``); the
    caller takes the stock all-gather. Every rank declines alike: it depends only on shapes.
    """

    def __call__(
        self,
        shard: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        world: int,
        grid: int = 0,
        pdl: int = -1,
    ) -> torch.Tensor:
        return torch.ops.trtllm.allgather_f32(
            shard, ws_uc, ws_mc, ws_flags, int(rank), int(world), int(grid), int(pdl)
        )

    def reference(self, shards_in_rank_order: Sequence[torch.Tensor]) -> torch.Tensor:
        """The stock path: all_gather + the vocab-dim concat (the documented collective exception)."""
        return torch.cat(list(shards_in_rank_order), dim=-1)

    def is_valid(
        self,
        shard: torch.Tensor,
        ws_uc: torch.Tensor,
        ws_mc: torch.Tensor,
        ws_flags: torch.Tensor,
        rank: int,
        world: int,
        grid: int = 0,
        pdl: int = -1,
    ) -> None:
        assert supports_f32(shard, world), (tuple(shard.shape), shard.dtype, shard.is_contiguous())
        assert 0 <= rank < world, (rank, world)


allgather_f32 = AllgatherF32()
