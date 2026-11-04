# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 and TP8 all-reduce fused with the residual add and RMSNorm that consume it."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm._lamport import (
    TWOSHOT_ABOVE,
    LamportWorkspace,
    check_rows,
    tail,
)

# A target builds the workspace this entry is handed through the entry itself,
# so `_lamport` stays private to the catalog.
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm._lamport import (
    build_workspace as build_workspace,
)

# Mistral Large 4's configuration: TP4, hidden 6144, rms eps 1e-5; the workspace holds
# max_num_tokens, 9216 rows.
_ML4 = dict(world_size=4, hidden=6144, eps=1e-5, max_rows=9216)
# Mistral Large 4 NVFP4's: the same TP width, hidden and eps; its max_num_tokens, and so its
# workspace, is 32768 rows (max_seq_len 32768, no chunked prefill).
_ML4_32 = dict(world_size=4, hidden=6144, eps=1e-5, max_rows=32768)
# Mistral Large 4 NVFP4 at tep8 and tp8: the same hidden, eps and 32768-row workspace over 8 ranks
# on two GB300 nodes of 4 (one NVL72 segment, MNNVL), started as MPI ranks as the engine
# starts them. Both targets make the same two calls per layer at the same rows; the second
# sums tep8's expert-window partials, tp8's expert width-slice partials.
_ML4_32_TEP8 = dict(world_size=8, nodes=2, hidden=6144, eps=1e-5, max_rows=32768)


class CuteDslAllreduceAddRmsnorm(OpWrapper):
    """`residual += sum over ranks of partial`; `out = rmsnorm(residual) * weight`.

    `partial`, `residual` and `out` are contiguous bf16 `[rows, hidden]`,
    `weight` bf16 `[hidden]`, `rows` up to `workspace.max_rows`. `residual` is
    updated in place, `out` is written and returned. The sum is fp32 in rank
    order on every rank; the residual add is fp32 and rounds once to bf16, and
    the norm reads that bf16 residual. `eps` is the workspace's, fixed at build.

    Up to `TWOSHOT_ABOVE` rows the kernel is one-shot: every rank pulls every
    peer's whole partial. Above it, two-shot: each rank reduces 1/world of the
    rows (a quarter at TP4, an eighth at TP8) and the reduced rows are
    gathered back. The two-shot gather moves the reduced rows as bf16, so the
    residual add there rounds twice (sum, then sum + residual). The choice is
    a function of the row count alone and is made here, identically on every
    rank.

    The kernels are `catalog/comm/_tp4_allreduce_fused_rms_kernel.py`
    (CuTe DSL), compiled by `build_workspace` (`catalog/comm/_lamport.py`);
    `fused` is the one-shot launcher, `fused_twoshot` the two-shot one.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Mistral Large 4 FP8 and NVFP4: both closing collectives of each of their 72 layers
    # (o_proj into the post-attention norm; the MoE or dense MLP into the next layer's input
    # norm or the final norm), on captured decode graphs (batches 1/2/4/8/16, padded) and
    # eager prefill steps of one workspace. NVFP4 runs without chunked prefill, so one step
    # carries up to max_num_tokens, 32768 rows (one prompt, or up to 16 packed). NVFP4 tep8
    # and tp8 make the same two calls per layer over 8 ranks on two nodes (the "NVFP4 tep8"
    # cells are both targets'); their engines capture decode graphs at 1/2/4/8/16 padded, and
    # the cells replay a step at every batch 1..16.
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"DSpark graph {geometry['world_size']} ranks, {rows} verification/draft rows",
                spec=dict(geometry, rows=rows, graph=True),
            )
            for geometry in (_ML4_32, _ML4_32_TEP8)
            for rows in range(17, 65)
        ),
        Cell(
            why="Mistral Large 4 decode batch 1: a captured 144-launch step, replayed",
            spec=dict(_ML4, rows=1, graph=True),
        ),
        Cell(
            why="Mistral Large 4 decode batch 2: a captured 144-launch step, replayed",
            spec=dict(_ML4, rows=2, graph=True),
        ),
        Cell(
            why="Mistral Large 4 decode batch 4: a captured 144-launch step, replayed",
            spec=dict(_ML4, rows=4, graph=True),
        ),
        Cell(
            why="Mistral Large 4 decode batch 8: a captured 144-launch step, replayed",
            spec=dict(_ML4, rows=8, graph=True),
        ),
        Cell(
            why="Mistral Large 4 decode batch 16, max_batch_size: a captured step, replayed",
            spec=dict(_ML4, rows=16, graph=True),
        ),
        Cell(
            why="Mistral Large 4 prefill at the last one-shot row count, 3072",
            spec=dict(_ML4, rows=3072),
        ),
        Cell(
            why="Mistral Large 4 prefill at the first two-shot row count, 3073",
            spec=dict(_ML4, rows=3073),
        ),
        Cell(
            why="Mistral Large 4 prefill at 9215 rows: rank 3's last row block holds 3 rows",
            spec=dict(_ML4, rows=9215),
        ),
        Cell(
            why="Mistral Large 4 max_num_tokens, 9216 rows: the whole 9216-row workspace",
            spec=dict(_ML4, rows=9216),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 decode batch 1, 32768-row workspace: a captured step, replayed",
            spec=dict(_ML4_32, rows=1, graph=True),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 decode batch 2, 32768-row workspace: a captured step, replayed",
            spec=dict(_ML4_32, rows=2, graph=True),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 decode batch 4, 32768-row workspace: a captured step, replayed",
            spec=dict(_ML4_32, rows=4, graph=True),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 decode batch 8, 32768-row workspace: a captured step, replayed",
            spec=dict(_ML4_32, rows=8, graph=True),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 decode batch 16, max_batch_size, 32768-row workspace: replayed",
            spec=dict(_ML4_32, rows=16, graph=True),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 prefill at the last one-shot row count, 3072",
            spec=dict(_ML4_32, rows=3072),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 prefill at the first two-shot row count, 3073",
            spec=dict(_ML4_32, rows=3073),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 prefill at 4096 rows: 256 row groups, one per two-shot CTA",
            spec=dict(_ML4_32, rows=4096),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 prefill at 4097 rows: 257 row groups, the two-shot CTAs loop",
            spec=dict(_ML4_32, rows=4097),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 prefill at 32767 rows: rank 3's last row block holds 3 rows",
            spec=dict(_ML4_32, rows=32767),
        ),
        Cell(
            why="Mistral Large 4 NVFP4 max_num_tokens, 32768 rows: the whole 32768-row workspace",
            spec=dict(_ML4_32, rows=32768),
        ),
        *(
            Cell(
                why=f"NVFP4 tep8 decode batch {rows}, 8 ranks on 2 nodes: a captured step, replayed",
                spec=dict(_ML4_32_TEP8, rows=rows, graph=True),
            )
            for rows in range(1, 17)
        ),
        Cell(
            why="NVFP4 tep8 prefill at the last one-shot row count, 3072: 7 peers' partials, 4 off-node",
            spec=dict(_ML4_32_TEP8, rows=3072),
        ),
        Cell(
            why="NVFP4 tep8 prefill at the first two-shot row count, 3073: a one-row last group",
            spec=dict(_ML4_32_TEP8, rows=3073),
        ),
        Cell(
            why="NVFP4 tep8 prefill at 4096 rows: 128 row groups of 32, one per two-shot CTA",
            spec=dict(_ML4_32_TEP8, rows=4096),
        ),
        Cell(
            why="NVFP4 tep8 prefill at 4097 rows: 129 row groups, the last holding one row",
            spec=dict(_ML4_32_TEP8, rows=4097),
        ),
        Cell(
            why="NVFP4 tep8 prefill at 8192 rows: 256 row groups, one per two-shot CTA (the cap)",
            spec=dict(_ML4_32_TEP8, rows=8192),
        ),
        Cell(
            why="NVFP4 tep8 prefill at 8193 rows: 257 row groups, the two-shot CTAs loop",
            spec=dict(_ML4_32_TEP8, rows=8193),
        ),
        Cell(
            why="NVFP4 tep8 prefill at 32767 rows: rank 7's last row block holds 3 rows",
            spec=dict(_ML4_32_TEP8, rows=32767),
        ),
        Cell(
            why="NVFP4 tep8 max_num_tokens, 32768 rows: the whole 32768-row workspace over 8 ranks",
            spec=dict(_ML4_32_TEP8, rows=32768),
        ),
    )

    note = """
    DSpark adds captured steps at every row count 17..64 for the 4-rank and
    8-rank 32768-row workspaces. Together with the existing 1..16 cells,
    this covers the target's four verification rows and draft's three rows
    per request through batch 16. Eager prefill bounds stay at 32768 rows.

    A target builds one workspace with `max_rows` = its token budget (Mistral
    Large 4 FP8: max_num_tokens, 9216; Mistral Large 4 NVFP4 at world
    4 and 8: max_num_tokens, 32768) through `build_workspace(hidden, eps,
    max_rows, rank, world)`, imported from this module, and calls this entry
    at every row count up to it; there is no stock fallback.

    World size is a cell axis, and so is the topology under it. The
    `world_size` 4 cells ran on 4 GB300 of one node, as spawned processes
    whose group came up from an exported rendezvous address. The `world_size`
    8 cells ran on 8 MPI ranks over two GB300 nodes of 4 (`nodes=2`, one
    NVL72 segment: every peer on NVLink, 4 of the 7 on the other node), whose
    group `ensure_process_group` brought up from rank 0's address broadcast
    over `MPI.COMM_WORLD`. Neither set says anything about the other world.

    Metadata consumed: none. The row count is the input's. The slot is read
    from the workspace's device counter and advanced by the kernel itself, so
    a captured graph replays onto whichever slot is current: the decode cells
    capture one Mistral Large 4 step (144 launches, one residual carried through
    them in place) and replay it between eager prefill calls on one workspace.

    Preconditions, measured on the cells' path (sm_103):
    - `build_workspace` is collective (every rank, the same arguments), runs
      outside any graph capture and compiles four CuTe DSL kernels (0.5, 0.7,
      0.6 and 1.7 s with the DSL's caches off). The peer buffers are
      3 x world x max_rows x hidden bf16 plus a quarter of that in int32
      flags, in symmetric memory: 1.58 GiB per rank at 9216 rows, 5.63 GiB at
      32768, 11.25 GiB at 32768 and world 8. A first build in a process
      measured 2.14 GiB of device memory taken at 9216 rows; a first build
      at 32768 rows 6.18 GiB, in 4.7 s; at world 8, with the group already
      up, 11.80 GiB in 6.6 s.
    - Under MPI, every rank must bring the group up the same way:
      `ensure_process_group` takes the rendezvous address from the
      environment when `MASTER_ADDR`, `MASTER_PORT`, `RANK` and `WORLD_SIZE`
      are all set, and broadcasts rank 0's otherwise. The two-node session
      the world-8 cells ran on sets all four in each node's environment (on
      the first node: `MASTER_ADDR` that node, `MASTER_PORT` 64710, `RANK`
      0, `WORLD_SIZE` 2). Measured: with them removed on the first node
      only, its ranks took the broadcast and the other node's dialled the
      environment's address until their 600 s timeout; rank 0 failed after
      601 s with "4/8 clients joined". A launcher that forwards one node's
      environment to every rank keeps the ranks in one case; the world-8
      cells removed the four on every rank and ran the broadcast.
    - The one-shot publish is a multicast store when the build finds a
      multicast address on every rank's buffer (agreed across the group),
      and per-peer stores otherwise. Every cell ran with multicast on, which
      the GB300 nodes they ran on provide, at world 8 across the two nodes
      too; the per-peer path is in no cell.
    - Every rank calls with the same row count. With three ranks at 16 rows
      and one at 8, the three had not completed after 20 s (the fourth had);
      at 4096 rows against 2048, none had. A disagreeing rank hangs the group
      and nothing raises.
    - `partial`, `residual` and `out` contiguous bf16 `[rows, hidden]`,
      `weight` bf16 `[hidden]`, all 16 B aligned. A row window of a larger
      contiguous buffer is accepted and answered correctly (one-shot at 16
      rows: bit-exact; two-shot at 4096 rows: inside the two-shot band). The
      launcher refuses the rest before anything launches, leaving the slot
      counter as it was (probed at 16 rows), each with a `ValueError: ...
      when calling: <the launcher's signature>`: a strided view (`Mismatched
      input_tensor.strides[0] on argument #0`; `norm_output_tensor` on #3),
      another dtype (`Mismatched Tensor on argument #0`; #2 for an fp32
      weight), another hidden (`Mismatched input_tensor.shape[1] on argument
      #0`), operands of different row counts (`Mismatched
      residual_tensor.shape[0] on argument #1`), a pointer off 16 B
      (`Misaligned Tensor data on argument #0`).
    - The kernels launch with PDL and wait on their predecessor before any
      load; the one-shot kernel releases its dependents at entry (read in
      the kernel source, not measured). So a PDL-launched successor must
      wait on it before reading `out` or `residual`.

    Two-shot schedule: `min(ceil(rows / (4 x world)), 256)` producer CTAs,
    each owning (4 x world)-row groups (4 rows per rank, block g x world + r
    rank r's), plus 1024 consumer CTAs. At world 4, 16-row groups: up to
    4096 rows each producer takes one group, from 4097 they loop. At world
    8, 32-row groups: up to 8192 rows one group each, from 8193 they loop.
    The block owner sums every rank's rows in fp32 in rank order 0..world-1;
    at world 8 the two-shot residual was bit-exact to that arithmetic
    (bf16(bf16(sum) + residual)) at every two-shot cell, 3073 to 32768 rows,
    and the one-shot residual (3072 rows, every replayed decode step) to the
    reference's.

    Cost after a large call: each call clears the slot the next call will
    use, as many vectors as that slot last held, so the second call after a
    large call clears that call's rows' worth, and a small call there costs
    more than it does otherwise.

    Hard limits (sm_103; the kernels' index arithmetic read, and probed where
    the 32768-row workspace reaches it)
      peer-buffer offset  (slot x world + region) x max_rows x hidden + row x
                          hidden + col, Int64 in every kernel. The buffer
                          passes 2^31 elements from max_rows 29,128 at world
                          4 and 14,564 at world 8; at 32768 it holds
                          2,415,919,104 (world 4) and 4,831,838,208 (world 8,
                          past 2^32). Probed at 32768, world 4: rank 3's
                          one-shot publish on slot 2 (from element
                          2,214,592,512, past 2^32 bytes too) lands in place
                          and is cleared, and a two-shot launch clears slot 2
                          to element 2,415,919,103: passed; two-shot moves
                          data through regions 0 and 1 only, below
                          2,013,265,920. World 8: rank 7's one-shot publish on
                          slot 2 (from element 4,630,511,616, past 2^32
                          elements) lands in place and is cleared, a 32768-row
                          two-shot call on slot 2 (regions 0 and 1 from
                          element 3,221,225,472, past 2^31) passes its gate,
                          and a two-shot launch clears slot 2 to element
                          4,831,838,207: passed.
      workspace numel     max_rows x hidden, the compiled Int32 extent:
                          passes 2^31 at 349,526 rows at 6144, not probed
      buffer mode-0       world x max_rows x hidden: passes 2^31 at 87,382
      stride              rows at world 4, 43,691 at world 8 (hidden 6144),
                          computed, not probed
      rows x hidden       input offsets and the vector counts in Int32: pass
                          2^31 at 349,526 rows, not probed (rows <= max_rows)
      rows > max_rows     `is_valid` refuses: the kernel would write past every
                          peer's buffer (not driven unguarded)
    """

    def __call__(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        workspace: LamportWorkspace,
    ) -> torch.Tensor:
        compiled = workspace.fused_twoshot if partial.shape[0] > TWOSHOT_ABOVE else workspace.fused
        compiled(partial, residual, weight, out, residual, *tail(workspace, fused=True))
        return out

    def reference(
        self,
        partials: Sequence[torch.Tensor],
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        workspace: LamportWorkspace,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(norm, new_residual)`, both bf16; `residual` is not modified.

        `partials` is the group's partials in rank order (the collective
        exception to mirroring `__call__`). The sum is fp32 in rank order.
        `new_residual` is `bf16(sum + residual)`, rounded once. The norm is
        computed in fp64 from that bf16 residual with `workspace.eps` and
        rounded once. `out` is ignored.
        """
        acc = torch.zeros(residual.shape, dtype=torch.float32, device=residual.device)
        for p in partials:
            acc += p.float()
        new_residual = (acc + residual.float()).to(torch.bfloat16)
        x = new_residual.double()
        inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + workspace.eps)
        norm = (x * inv_rms * weight.double()).to(torch.bfloat16)
        return norm, new_residual

    def is_valid(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        workspace: LamportWorkspace,
    ) -> None:
        check_rows(int(partial.shape[0]), workspace)


cute_dsl_allreduce_add_rmsnorm = CuteDslAllreduceAddRmsnorm()
