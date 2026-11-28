# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 sequence-parallel prefill collectives on NVLink peer mappings: drop-ins for `seq_parallel_rows`.

`sp_gather_rows_mnnvl` / `sp_reduce_scatter_rows_mnnvl` take and return what `sp_gather_rows` /
`sp_reduce_scatter_rows` do (lists of tensors, rank ``r`` owning rows ``[r * rows, (r + 1) * rows)``) and run
the target's CuTe DSL kernels (`kimi_k31_pro_mxfp4__sm_103__tep/kernels/comm/sp_collectives`) on a symmetric
workspace of their own, separate from the decode all-reduce's: a `McastGPUBuffer` per TP group, allocated on the first
call (or by `reserve_sp_workspace` at model build) for the largest step and grown only if a larger step comes. Tensors
the kernels do not serve (dim 0 not a multiple of the group, rows not a multiple of 16 bytes, a reduce-scatter of a
non-BF16 tensor) take the NCCL ops.

Memory: the workspace is 64 KiB of flags plus one receive region of ``max_tokens x max_row_bytes`` per rank (both
kernels share it), 235 MB for a 16384-token step of 7168-wide BF16 rows; it is allocated once, not per call.
GB300, 8 ranks over two trays, us per call (engine NCCL -> these kernels): 16384 x 7168 BF16 all-gather 361 -> 324,
reduce-scatter 360 -> 328; the MoE's grouped gather (hidden, latent, FP32 router logits) 613 -> 594 and grouped
reduce-scatter (latent, shared) 528 -> 501. The kernels reach about 90% of the rate a bare 16-byte peer-store copy
gets on the same link (300 us for the 16384 x 7168 block).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from tensorrt_llm._torch.distributed import ops as _dist_ops
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

from ...kernels.comm import sp_collectives as _k
from .seq_parallel_rows import sp_gather_rows, sp_reduce_scatter_rows

#: Widest row the workspace is sized for by default: the 7168-wide BF16 hidden state.
DEFAULT_MAX_ROW_BYTES = 7168 * 2
#: Default step size the workspace is first sized for (a 16384-token TEP8 context step).
DEFAULT_MAX_TOKENS = 16384

#: Largest TP group the kernels serve; larger groups take the NCCL ops.
MAX_RANKS = _k.MAX_WORLD

#: Launch configuration: rows per synchronization chunk, threads per CTA, 16-byte vectors in flight per thread,
#: and the consumer CTAs of each kernel (the rest of the SMs push).
TUNING = dict(chunk_rows=16, threads=512, unroll=8, rs_consumers=24, ag_consumers=16, rs_slices=8)


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class SpWorkspace:
    """One TP group's symmetric workspace: flags and the receive region (``[W, rows, row_bytes]``) both kernels
    share, sized for ``max_tokens`` x ``max_row_bytes``."""

    def __init__(self, mapping: Mapping, max_tokens: int, max_row_bytes: int):
        world = mapping.tp_size
        if world > _k.MAX_WORLD:
            raise ValueError(
                f"sequence-parallel kernels serve up to {_k.MAX_WORLD} ranks, not {world}"
            )
        rows = -(-max_tokens // world)
        self.mapping = mapping
        self.world = world
        self.rank = mapping.tp_rank
        self.max_rows = rows
        self.max_row_bytes = max_row_bytes
        self.total_bytes = _k.FLAGS_BYTES + _round_up(rows * world * max_row_bytes, 1 << 16)
        comm = _dist_ops._get_mnnvl_workspace_comm(mapping)
        self.handle = _dist_ops._make_mnnvl_mcast_buffer(
            comm, self.total_bytes, mapping, mapping.is_multi_node()
        )
        device = torch.device("cuda", _dist_ops._mnnvl_device_index(mapping))
        local = self.handle.get_uc_buffer(self.rank, (self.total_bytes,), torch.uint8, 0)
        self.base = local.data_ptr()
        peers = [
            self.handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(world)
        ]
        self.peers = torch.tensor(peers, dtype=torch.int64, device=device)
        local[: _k.FLAGS_BYTES].zero_()
        torch.cuda.synchronize()
        # Every rank's flags are zero before any rank pushes or polls.
        if not _dist_ops._mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("sequence-parallel workspace setup failed on a rank")
        self._comm = comm
        # Calls of either kernel, in order (the same on every rank).
        self.epoch = 0
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    def fits(self, rows: int, row_bytes: int) -> bool:
        return (
            rows <= self.max_rows
            and row_bytes <= self.max_row_bytes
            and -(-rows // TUNING["chunk_rows"]) <= _k.MAX_CHUNKS
        )


#: Workspaces per TP group (the mapping's TP rank list).
_WORKSPACES: Dict[tuple, SpWorkspace] = {}


def _key(mapping: Mapping) -> tuple:
    return (tuple(mapping.tp_group), mapping.tp_rank)


def reserve_sp_workspace(
    mapping: Mapping, max_tokens: int, max_row_bytes: int = DEFAULT_MAX_ROW_BYTES
) -> SpWorkspace:
    """Allocate (or grow to) the TP group's workspace for steps of up to ``max_tokens`` rows of up to
    ``max_row_bytes`` bytes. Collective: every rank of the group calls it at the same point."""
    ws = _WORKSPACES.get(_key(mapping))
    rows = -(-max_tokens // mapping.tp_size)
    if ws is None or not ws.fits(rows, max_row_bytes):
        tokens = max(max_tokens, ws.max_rows * mapping.tp_size if ws else 0)
        width = max(max_row_bytes, ws.max_row_bytes if ws else 0)
        _WORKSPACES.pop(_key(mapping), None)
        ws = SpWorkspace(mapping, tokens, width)
        _WORKSPACES[_key(mapping)] = ws
    return ws


def _workspace(mapping: Mapping, rows: int, row_bytes: int) -> SpWorkspace:
    ws = _WORKSPACES.get(_key(mapping))
    if ws is not None and ws.fits(rows, row_bytes):
        return ws
    return reserve_sp_workspace(
        mapping,
        max(rows * mapping.tp_size, DEFAULT_MAX_TOKENS),
        max(row_bytes, DEFAULT_MAX_ROW_BYTES),
    )


def _stream():
    from cuda.bindings import driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _launch(
    compile_fn, consumers_key: str, ws: SpWorkspace, x: torch.Tensor, out: torch.Tensor, rows: int
) -> torch.Tensor:
    t = TUNING
    consumers = min(t[consumers_key], ws.num_sms // 2)
    kernel = compile_fn(
        ws.world,
        t["chunk_rows"],
        ws.num_sms - consumers,
        consumers,
        t["threads"],
        t["unroll"],
        t["rs_slices"],
    )
    ws.epoch = (ws.epoch + 1) & 0xFFFFFFFF
    kernel(
        _k.Int64(x.data_ptr()),
        _k.Int64(out.data_ptr()),
        _k.Int64(ws.peers.data_ptr()),
        _k.Int64(ws.base),
        _k.Int32(rows),
        _k.Int32(_row_bytes(x)),
        _k.Int32(ws.rank),
        _k.Int32(ws.epoch if ws.epoch < 2**31 else ws.epoch - 2**32),
        _stream(),
    )
    return out


def _gather_one(ws: SpWorkspace, x: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0]
    out = torch.empty((rows * ws.world,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    return _launch(_k.compiled_all_gather, "ag_consumers", ws, x, out, rows)


def _reduce_scatter_one(ws: SpWorkspace, x: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0] // ws.world
    out = torch.empty((rows,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
    return _launch(_k.compiled_reduce_scatter, "rs_consumers", ws, x, out, rows)


def _row_bytes(x: torch.Tensor) -> int:
    return (x.numel() // x.shape[0]) * x.element_size() if x.dim() and x.shape[0] else 0


def _aligned_rows(x: torch.Tensor, mapping: Mapping) -> bool:
    rb = _row_bytes(x)
    return (
        2 <= mapping.tp_size <= MAX_RANKS
        and x.is_cuda
        and x.is_contiguous()
        and x.dim() >= 1
        and x.shape[0] > 0
        and rb > 0
        and rb % 16 == 0
        and x.data_ptr() % 16 == 0
    )


def gather_serves(x: torch.Tensor, mapping: Mapping) -> bool:
    return _aligned_rows(x, mapping) and x.shape[0] <= TUNING["chunk_rows"] * _k.MAX_CHUNKS


def reduce_scatter_serves(x: torch.Tensor, mapping: Mapping) -> bool:
    return (
        _aligned_rows(x, mapping)
        and x.shape[0] <= mapping.tp_size * TUNING["chunk_rows"] * _k.MAX_CHUNKS
        and x.dtype == torch.bfloat16
        and x.shape[0] % mapping.tp_size == 0
    )


class SpGatherRowsMnnvl(OpWrapper):
    """`sp_gather_rows` (all-gather along dim 0 in rank order, a list in one call) on NVLink peer mappings.

    Each tensor the kernel serves (contiguous, rows a multiple of 16 bytes; any dtype) is one push all-gather
    launch; the others go to `sp_gather_rows` (the engine NCCL all-gather) in one grouped call. Returns new tensors,
    identical on every rank; the inputs are not written.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K3.1 SP attention input: normed hidden rows of a 16k step (2048 rows per rank at TEP8)",
            spec=dict(world=4, shapes=((2048, 7168),), dtypes=(torch.bfloat16,)),
        ),
        Cell(
            why="K3.1 SP MoE input: hidden + latent + fp32 router logits (one grouped call)",
            spec=dict(
                world=4,
                shapes=((1024, 7168), (1024, 3584), (1024, 896)),
                dtypes=(torch.bfloat16, torch.bfloat16, torch.float32),
            ),
        ),
        Cell(
            why="K3.1 SP final norm rows of an uneven step (2049 rows per rank: a partial last chunk)",
            spec=dict(world=4, shapes=((2049, 7168),), dtypes=(torch.bfloat16,)),
        ),
    )
    note = """
    A copy: bitwise against the rank-order concatenation (the NCCL all-gather's result). Workspace: the TP
    group's sequence-parallel workspace (see the module docstring); calls must be made in the same order on every
    rank with the same shapes, eagerly (the sequence-parallel path never runs under CUDA-graph capture: the kernels
    take a host-side call counter).
    """

    def __call__(self, inputs: List[torch.Tensor], mapping: Mapping) -> List[torch.Tensor]:
        if mapping.tp_size == 1:
            return list(inputs)
        outs: List[Optional[torch.Tensor]] = [None] * len(inputs)
        rest = [i for i, x in enumerate(inputs) if not gather_serves(x, mapping)]
        for i, x in enumerate(inputs):
            if i not in rest:
                ws = _workspace(mapping, x.shape[0], _row_bytes(x))
                outs[i] = _gather_one(ws, x)
        if rest:
            for i, y in zip(rest, sp_gather_rows([inputs[i] for i in rest], mapping)):
                outs[i] = y
        return outs

    def reference(
        self, inputs_in_rank_order: Sequence[Sequence[torch.Tensor]], mapping: Mapping = None
    ) -> List[torch.Tensor]:
        """Every rank's list, in rank order (the collective exception); ``mapping`` is ignored."""
        return sp_gather_rows.reference(inputs_in_rank_order, mapping)


class SpReduceScatterRowsMnnvl(OpWrapper):
    """`sp_reduce_scatter_rows` (sum the ranks' ``[rows * tp, ...]`` partials, keep this rank's ``[rows, ...]``)
    on NVLink peer mappings: one push reduce-scatter launch per BF16 tensor (others: the NCCL op, grouped).

    Returns new tensors; the inputs are not written.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K3.1 SP attention o_proj partials of a 16k step (8192 rows at TP4 here, 16384 at TEP8)",
            spec=dict(world=4, shapes=((8192, 7168),), dtypes=(torch.bfloat16,)),
        ),
        Cell(
            why="K3.1 SP MoE routed latent + shared expert partials (one grouped call)",
            spec=dict(
                world=4,
                shapes=((4096, 3584), (4096, 7168)),
                dtypes=(torch.bfloat16, torch.bfloat16),
            ),
        ),
        Cell(
            why="K3.1 SP uneven step: 2049 rows per rank (a partial last chunk)",
            spec=dict(world=4, shapes=((2049 * 4, 7168),), dtypes=(torch.bfloat16,)),
        ),
    )
    note = """
    Reduction order: every element is the FP32 sum of the W partials in rank order 0, 1, ..., W - 1 (each BF16
    partial widened exactly, one FP32 add per rank, round to nearest), rounded once to BF16 -- the reference
    computes exactly that and the kernel matches it bit for bit. The NCCL op it replaces sums in its ring order
    with BF16 roundings between hops, so the result is not bitwise to NCCL (it is closer to the exact sum); the
    sequence-parallel path is accuracy-gated. Workspace and call-order rules as `SpGatherRowsMnnvl`.
    """

    def __call__(self, inputs: List[torch.Tensor], mapping: Mapping) -> List[torch.Tensor]:
        if mapping.tp_size == 1:
            return list(inputs)
        outs: List[Optional[torch.Tensor]] = [None] * len(inputs)
        rest = [i for i, x in enumerate(inputs) if not reduce_scatter_serves(x, mapping)]
        for i, x in enumerate(inputs):
            if i not in rest:
                ws = _workspace(mapping, x.shape[0] // mapping.tp_size, _row_bytes(x))
                outs[i] = _reduce_scatter_one(ws, x)
        if rest:
            for i, y in zip(rest, sp_reduce_scatter_rows([inputs[i] for i in rest], mapping)):
                outs[i] = y
        return outs

    def reference(
        self, inputs_in_rank_order: Sequence[Sequence[torch.Tensor]], mapping: Mapping
    ) -> List[torch.Tensor]:
        """Every rank's list, in rank order (the collective exception): this ``mapping.tp_rank``'s slice of the
        FP32 rank-order sum, rounded once to the input dtype."""
        num = len(inputs_in_rank_order[0])
        out = []
        for i in range(num):
            total = torch.zeros_like(inputs_in_rank_order[0][i], dtype=torch.float32)
            for r in inputs_in_rank_order:
                total.add_(r[i].float())
            rows = total.shape[0] // mapping.tp_size
            out.append(
                total[mapping.tp_rank * rows : (mapping.tp_rank + 1) * rows].to(
                    inputs_in_rank_order[0][i].dtype
                )
            )
        return out


sp_gather_rows_mnnvl = SpGatherRowsMnnvl()
sp_reduce_scatter_rows_mnnvl = SpReduceScatterRowsMnnvl()
