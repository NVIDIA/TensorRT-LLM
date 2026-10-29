# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind the engine's in-node IPC workspace and retain its actual allocations.

The pointer table contains addresses, not payload. flags/layout are real mutable Tensors;
peer payload remains opaque cudaMalloc memory. This contract supports eager execution and
ordinary CUDA graphs, and deliberately does not support functionalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from tensorrt_llm._ipc_utils import IpcMemory
from tensorrt_llm._torch.distributed import ops as distributed_ops
from tensorrt_llm._torch.distributed.allreduce_helper import force_all_reduce_deterministic
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.mapping import Mapping


@dataclass(frozen=True)
class IpcWorkspace:
    table: torch.Tensor
    flags: torch.Tensor
    layout: torch.Tensor
    owners: tuple[Any, ...]
    rank: int
    nranks: int
    buffer_bytes: int
    hidden: int
    plain_rows: frozenset[int]
    finalize_rows: frozenset[int]

    def args(self):
        return self.table, self.flags, self.layout, self.rank, self.nranks, self.buffer_bytes


def _bind_local(
    mapping: Mapping, workspaces: Sequence[torch.Tensor | None], max_rows: int, hidden: int
) -> IpcWorkspace:
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise ValueError("IPC binding must happen eagerly before capture")
    if (
        mapping.tp_size != 4
        or mapping.moe_ep_size != 1
        or mapping.moe_tp_size != 4
        or mapping.enable_attention_dp
        or mapping.pp_size != 1
        or mapping.cp_size != 1
        or mapping.is_multi_node()
        or hidden != 6144
        or not 1 <= max_rows <= 24
    ):
        raise ValueError("IPC requires in-node TP4/EP1 BF16 hidden6144, at most24 rows")
    if force_all_reduce_deterministic():
        raise ValueError("deterministic all-reduce policy uses the engine implementation")
    if not workspaces or workspaces[0] is None:
        raise ValueError("no existing all-reduce workspace")
    table = workspaces[0]
    if any(other is not table for other in workspaces):
        raise ValueError("participating layers do not share the same IPC workspace")
    # The public getter intentionally returns only the table. Read its ownership record once;
    # do not call it here, since a cache miss would allocate/open an IPC group collectively.
    cache = getattr(distributed_ops._thread_local, f"allreduce_workspaces_{mapping.pp_rank}", {})
    entry = cache.get(mapping)
    if entry is None or len(entry) != 2 or entry[1] is not table:
        raise ValueError("existing IPC ownership record is unavailable on this construction thread")
    owners = tuple(entry[0])
    if len(owners) != 5 or not all(isinstance(x, IpcMemory) for x in owners[:3]):
        raise ValueError("unexpected IPC owner layout")
    if any(
        x.mapping != mapping
        or not x.open_ipc
        or not x.local_ptr
        or len(x.peer_ptrs) != 4
        or x.peer_ptrs[mapping.tp_rank] != x.local_ptr
        or any(not ptr for ptr in x.peer_ptrs)
        for x in owners[:3]
    ):
        raise ValueError("IPC peer allocations are not open")
    flags, layout = owners[3:]
    for name, tensor, dtype, count in (
        ("table", table, torch.int64, 14),
        ("flags", flags, torch.int32, 3),
        ("layout", layout, torch.int64, 2),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or not tensor.is_cuda
            or tensor.device != table.device
            or tensor.dtype != dtype
            or tensor.ndim != 1
            or tensor.numel() != count
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"unexpected IPC {name} tensor")
    expected = [ptr for owner in owners[:3] for ptr in owner.serialize()]
    expected += [flags.data_ptr(), layout.data_ptr()]
    # One initialization-time snapshot: never extract device state in forward/capture.
    if table.cpu().tolist() != expected:
        raise ValueError("IPC pointer table does not identify its retained owners")
    buffer_bytes = int(layout.cpu().tolist()[1])
    if buffer_bytes <= 0 or buffer_bytes % 16 or max_rows * hidden * 2 * 4 > buffer_bytes:
        raise ValueError("IPC Lamport buffer is too small for the planned shapes")
    with torch.cuda.device(table.device):
        plain = frozenset(
            m
            for m in range(1, max_rows + 1)
            if torch.ops.trtllm.glm_intra_node_reduce_supported(m, hidden, 4, False)
        )
        finalize = frozenset(
            m
            for m in range(1, max_rows + 1)
            if torch.ops.trtllm.glm_intra_node_reduce_supported(m, hidden, 4, True)
        )
    if not plain or not finalize:
        raise ValueError("owned IPC kernels have no supported launch on this device")
    return IpcWorkspace(
        table, flags, layout, owners, mapping.tp_rank, 4, buffer_bytes, hidden, plain, finalize
    )


def bind_ipc_workspace(
    mapping: Mapping,
    workspaces: Sequence[torch.Tensor | None],
    *,
    max_rows: int = 24,
    hidden: int = 6144,
) -> tuple[IpcWorkspace | None, str]:
    """Called on every rank in the same order, once after ordinary layer construction.

    A rank-local failure becomes a uniform unsupported result. There is no allocation,
    cache mutation, flag reset, or repeated lookup in model forward.
    """
    binding = None
    reason = ""
    try:
        binding = _bind_local(mapping, workspaces, max_rows, hidden)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    status = (
        reason,
        sorted(binding.plain_rows) if binding else [],
        sorted(binding.finalize_rows) if binding else [],
    )
    results = Distributed.get(mapping).tp_allgather(status, small_payload=True)
    failures = [f"rank {rank}: {item[0]}" for rank, item in enumerate(results) if item[0]]
    if failures:
        return None, "; ".join(failures)
    if any(item[1:] != results[0][1:] for item in results):
        return None, "IPC launch capabilities differ between ranks"
    return binding, ""


def validate_ipc_operands(residual, gamma, table, flags, layout, rank, nranks, buffer_bytes):
    """Executable entry contract, called by tests through validating(), never by forward."""
    assert residual.ndim == 2 and 1 <= residual.shape[0] <= 24 and residual.shape[1] == 6144
    assert residual.dtype == gamma.dtype == torch.bfloat16 and gamma.shape == (6144,)
    assert nranks == 4 and 0 <= rank < nranks
    assert table.shape == (14,) and table.dtype == torch.int64
    assert flags.shape == (3,) and flags.dtype == torch.int32
    assert layout.shape == (2,) and layout.dtype == torch.int64
    for value in (residual, gamma, table, flags, layout):
        assert value.is_cuda and value.device == residual.device and value.is_contiguous()
    assert buffer_bytes >= residual.numel() * 2 * nranks and buffer_bytes % 16 == 0
    assert table.cpu().tolist()[-2:] == [flags.data_ptr(), layout.data_ptr()]
    assert layout[1].item() == buffer_bytes


def residual_norm_reference(total, residual, gamma, epsilon):
    summed = total.to(torch.bfloat16) + residual
    normed = summed.float() * torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + epsilon)
    return (normed * gamma.float()).to(torch.bfloat16), summed
