# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 Rubin routed-latent all-reduce + RMSNorm (MNNVL one-shot) with an early PDL trigger.

Bit-identical to the C++ one-shot MNNVL all-reduce with the RMSNorm fusion, but it releases its
programmatic dependents right after its own ``griddepcontrol.wait``, so latent_up streams its
weight while the all-reduce waits for the other ranks: one 512-thread CTA per row. The kernel is
``kernels/comm/rubin_latent_ar_norm.py`` (``run_k3_latent_ar_norm_row``); ``__call__`` calls its
launcher directly. :func:`rubin_latent_ar_workspace` builds the entry's own MNNVL Lamport
workspace (re-exported by ``ops/comm/rubin_comm.py``).
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import (
    MNNVLAllReduce,
    get_or_scale_allreduce_mnnvl_workspace,
)
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.mapping import Mapping

from ._rubin_comm_common import lamport_allreduce_reference
from .rubin_push_allreduce import use_fabric_handle

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.comm.rubin_latent_ar_norm import run_k3_latent_ar_norm_row as _launch_row

#: The kernel's row width (the routed latent) and the ranks summed per partial sum. The kernel
#: rejects other group sizes itself.
LATENT = 3584
RANK_CHUNK = 8
#: Kimi K3.1's routed_expert_norm epsilon, and the group the op tests drive (one VR200 tray).
EPS = 1e-5
WORLD = 4


def latent_ar_norm_reference(
    inputs_in_rank_order: Sequence[torch.Tensor],
    gamma: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """The bf16 sum ``s`` of every rank's partial (``-0`` sent as ``+0``, fp32, eight ranks per
    partial sum, rounded once), then ``bf16(s * rsqrt(mean(s^2) + eps) * gamma)`` in fp32, rounded
    once (the kernel squares in bf16; the band of its test covers that)."""
    s = lamport_allreduce_reference(inputs_in_rank_order, RANK_CHUNK)
    x = s.float()
    y = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * gamma.float()
    return y.to(torch.bfloat16)


def _check(
    x: torch.Tensor,
    gamma: torch.Tensor,
    out: torch.Tensor,
    flags: torch.Tensor,
    rank: int,
    world_size: int,
) -> None:
    """What the launchers take without checking: they read and write raw pointers."""
    for name, t in (("x", x), ("out", out)):
        assert t.dim() == 2 and t.shape[1] == LATENT, f"{name} must be [M, {LATENT}]"
        assert t.dtype == torch.bfloat16 and t.is_contiguous(), f"{name}: bf16 contiguous"
        assert t.data_ptr() % 16 == 0, f"{name}: 16-byte aligned"
    assert out.shape == x.shape, "out must have x's shape"
    assert gamma.dtype == torch.bfloat16 and tuple(gamma.shape) == (LATENT,), "gamma: bf16 [3584]"
    assert gamma.is_contiguous() and gamma.data_ptr() % 16 == 0, "gamma: contiguous, aligned"
    for t in (gamma, out, flags):
        assert t.device == x.device, "every tensor on x's device"
    assert 0 <= rank < world_size, f"rank {rank} outside a group of {world_size}"


class RubinLatentArNormRow(OpWrapper):
    """``out`` [M, 3584] bf16 = RMSNorm(``gamma``, ``eps``) of the sum of every rank's ``x``, one
    512-thread CTA per row (no row limit in the kernel). Returns ``out``."""

    ARCHS: frozenset[Arch] = frozenset()
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"{rows} rows ({site}), {WORLD} ranks", spec=dict(world=WORLD, rows=rows))
        for rows, site in (
            (1, "one decode row"),
            (8, "TEP8 b1 k7"),
            (96, "helix8 b16 k5"),
            (128, "the largest row count the targets send here"),
        )
    )
    note = """
    Every rank of the group launches it with its ``rank`` on the same MNNVL Lamport workspace
    (``uc_addr`` local unicast base, ``mc_addr`` multicast base, ``flags`` the shared 9-word
    rotation state); the C++ one- and two-shot kernels can use the workspace before or after it.
    One Lamport buffer must hold M x world x 3584 x 2 bytes (the launchers do not check). The
    norm sums bf16 squares in element order and takes rsqrt.approx, as the C++ kernel does. The
    kernel releases its dependents right after its grid wait, before the reduction. The cells
    drive 4 ranks (one VR200 tray); the targets' 8-rank groups are driven by target-level runs.
    """

    def __call__(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        uc_addr: int,
        mc_addr: int,
        flags: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        _launch_row(
            x,
            gamma,
            eps,
            out,
            uc_addr=uc_addr,
            mc_addr=mc_addr,
            flags=flags,
            rank=rank,
            world_size=world_size,
        )
        return out

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        gamma: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """What every rank gets back (the collective exception: every rank's ``x`` in rank
        order)."""
        return latent_ar_norm_reference(inputs_in_rank_order, gamma, eps)

    def is_valid(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float,
        out: torch.Tensor,
        uc_addr: int,
        mc_addr: int,
        flags: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        _check(x, gamma, out, flags, rank, world_size)


rubin_latent_ar_norm_row = RubinLatentArNormRow()


#: The 9-word state of a Lamport workspace: current index, dirty index, bytes per buffer, dirty
#: stages, bytes to clear (x4), arrivals.
_FLAGS = (0, 2, None, 0, 0, 0, 0, 0, 0)


def check_group(mapping: Mapping, group_size: int) -> None:
    """Fail unless ``mapping``'s TP group is the layout's group of ``group_size`` ranks."""
    if mapping.tp_size != group_size:
        raise ValueError(
            f"the TP group of this mapping has {mapping.tp_size} ranks, the layout reduces over "
            f"{group_size}: pass the mapping whose TP group is the layout's group"
        )


def rubin_latent_ar_workspace(
    mapping: Mapping,
    group_size: int,
    hidden_size: int,
    max_tokens: int,
    oneshot_max_bytes: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """The routed latent all-reduce's own MNNVL Lamport workspace, for up to ``max_tokens`` rows.

    Three Lamport buffers and the 9-word state, laid out as the shared MNNVL workspace
    (``get_or_scale_allreduce_mnnvl_workspace``), each buffer large enough for the two-shot sizes
    of up to ``max_tokens`` rows and for every rank's rows in one buffer up to
    ``oneshot_max_bytes`` (rows x ``hidden_size`` x ranks x 2 bytes). The buffers start
    Lamport-clean (-0.0). Collective over the TP group.

    Returns a dict: ``handle`` (keeps the buffers alive), ``buffer`` (the local unicast view
    ``[3, n]``), ``mc`` (the multicast view), ``flags``, ``max_tokens``, ``buffer_bytes`` (bytes
    of one buffer), ``rank``, ``world_size``.
    """
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    check_group(mapping, group_size)
    # The shared workspace's communicator is split by the same group.
    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, dtype)["comm"]
    need = max(
        MNNVLAllReduce.get_required_workspace_size(m, hidden_size, mapping.tp_size, dtype)
        for m in range(1, max_tokens + 1)
    )
    if oneshot_max_bytes:
        row_bytes = hidden_size * mapping.tp_size * dtype.itemsize
        need = max(need, min(max_tokens, oneshot_max_bytes // row_bytes) * row_bytes)
    buffer_bytes = -(-need // (1 << 20)) << 20
    handle = McastGPUBuffer(
        3 * buffer_bytes,
        mapping.tp_size,
        mapping.tp_rank,
        mapping.local_rank,
        use_fabric_handle(mapping),
        comm.py2f(),
    )
    buffer = handle.get_uc_buffer(mapping.tp_rank, (3 * buffer_bytes // 4,), torch.float32, 0)
    buffer.fill_(-0.0)
    torch.cuda.synchronize()
    comm.Barrier()
    flags = torch.tensor(
        [buffer_bytes if v is None else v for v in _FLAGS],
        dtype=torch.uint32,
        device=buffer.device,
    )
    return {
        "handle": handle,
        "buffer": buffer.view(dtype).view(3, -1),
        "mc": handle.get_mc_buffer((3 * buffer_bytes // 4,), torch.float32, 0),
        "flags": flags,
        "max_tokens": max_tokens,
        "buffer_bytes": buffer_bytes,
        "rank": mapping.tp_rank,
        "world_size": mapping.tp_size,
    }
