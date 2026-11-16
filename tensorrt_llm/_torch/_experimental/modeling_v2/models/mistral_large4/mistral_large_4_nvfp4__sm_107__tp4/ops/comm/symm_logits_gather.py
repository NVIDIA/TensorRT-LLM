# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vocabulary gather of the lm_head's column shards, widened to fp32 and soft-capped: every
rank stores its bf16 shard into every rank's symmetric buffer over NVLink, a device barrier
over the buffer's signal pads, then every rank widens and soft-caps its own buffer locally.

It replaces the NCCL all-gather, the shard reorder, the fp32 upcast and the three soft-cap ops
of the decode logits, target and draft alike. The output is bit-identical: the gather and the
upcast are copies, and the soft-cap is `cap * tanh(x * fl(1 / cap))` with the same libdevice
`tanhf` the torch op calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..dspark import _kernels

#: Vocabulary columns per CTA of the scatter and of the soft-capped read.
_BLOCK_V = 4096


@dataclass(frozen=True)
class LogitsGatherWorkspace:
    """One `[slots, world, rows, shard]` bf16 symmetric buffer per rank, block `p` of a slot
    holding rank `p`'s shard, and the peers' slot bases.

    Slot 0 serves the target's logits and slot 1 the draft's. A slot is rewritten one step
    later, past the step's all-reduces, which every rank reaches only after its read of the
    slot: those collectives order a rank's next write behind every peer's last read.
    """

    rank: int
    world: int
    rows: int
    shard: int
    buffer: torch.Tensor
    handle: object
    #: `[slots, world]` int64: the device address of every rank's slot, mapped into this rank.
    peer_slots: torch.Tensor


def build_workspace(rows: int, shard: int, rank: int, world: int, slots: int = 2):
    """Allocate and rendezvous the symmetric buffer. Collective over the default process group,
    which the target's Lamport workspace has brought up; call it on every rank with the same
    arguments, outside any CUDA-graph capture."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    dev = torch.device("cuda", torch.cuda.current_device())
    buffer = symm_mem.empty((slots, world, rows, shard), dtype=torch.bfloat16, device=dev)
    group = dist.group.WORLD
    handle = symm_mem.rendezvous(buffer, group=getattr(group, "group_name", group))
    peers = [handle.get_buffer(peer, buffer.shape, buffer.dtype) for peer in range(world)]
    peer_slots = torch.tensor(
        [[peers[peer][slot].data_ptr() for peer in range(world)] for slot in range(slots)],
        dtype=torch.int64,
        device=dev,
    )
    return LogitsGatherWorkspace(rank, world, rows, shard, buffer, handle, peer_slots)


class SymmLogitsGatherSoftcap(OpWrapper):
    """`cap * tanh(allgather(local, dim=-1).float() / cap)` over the workspace's ranks.

    `local` is this rank's `[rows, shard]` bf16 lm_head output, `rows <= workspace.rows`;
    returns `[rows, world * shard]` fp32. Every rank calls it with the same rows and slot.
    """

    ARCHS = frozenset({Arch.SM_107})
    note = """
    Collective. A call with more rows than the workspace holds must take the NCCL path on every
    rank; the caller decides from the step's row count, which every rank shares.
    """
    CELLS = tuple(
        Cell(
            why=f"tp{world} {what}: {rows} rows",
            spec=dict(world=world, rows=rows, shard=204800 // world, cap=30.0),
        )
        for world in (4, 8)
        for what, rows in (("C1 target", 4), ("C1 draft", 3), ("C16 target", 64))
    )

    def __call__(
        self, local: torch.Tensor, workspace: LogitsGatherWorkspace, slot: int, cap: float
    ) -> torch.Tensor:
        rows, shard = local.shape
        chunks = triton.cdiv(shard, _BLOCK_V)
        _kernels.scatter_shard_kernel[(rows, chunks)](
            local, local.stride(0), workspace.peer_slots[slot], workspace.rank, workspace.rows,
            shard, WORLD=workspace.world, BLOCK_V=_BLOCK_V, num_warps=4,
        )  # fmt: skip
        workspace.handle.barrier(channel=0)
        out = local.new_empty((rows, workspace.world * shard), dtype=torch.float32)
        inv_cap = float(np.float32(1.0) / np.float32(cap))
        _kernels.softcap_gather_kernel[(rows, workspace.world, chunks)](
            workspace.buffer[slot], workspace.rows, shard, out, out.stride(0), inv_cap,
            float(cap), BLOCK_V=_BLOCK_V, num_warps=4, enable_fp_fusion=False,
        )  # fmt: skip
        return out

    def reference(
        self, local: torch.Tensor, workspace: LogitsGatherWorkspace, slot: int, cap: float
    ) -> torch.Tensor:
        import torch.distributed as dist

        shards = [torch.empty_like(local) for _ in range(workspace.world)]
        dist.all_gather(shards, local.contiguous())
        return torch.tanh(torch.cat(shards, dim=-1).float() / cap) * cap


symm_logits_gather_softcap = SymmLogitsGatherSoftcap()
