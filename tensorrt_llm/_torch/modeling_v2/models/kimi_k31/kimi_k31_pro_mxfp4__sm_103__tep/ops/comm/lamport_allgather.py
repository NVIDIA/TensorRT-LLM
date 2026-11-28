# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 FP32 all-gather along the last dim in one multicast kernel over the MNNVL workspace."""

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_allgather as _ag
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
    allgather_module,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.allreduce_module import mnnvl_group
from tensorrt_llm.mapping import Mapping

from .oproj_allreduce_push import lamport_workspace

#: Largest message (rows x world x width FP32 words, bytes) the kernel serves by default; the gathered rows
#: go through one Lamport buffer, sized at least this large by the engine's one-shot threshold.
MAX_BYTES = 1 << 20
#: Largest per-call ``max_bytes`` a caller may raise the cap to: half the engine's smallest MNNVL Lamport buffer
#: (1024 x 8192 BF16 = 16 MiB), so a served call never grows the workspace (safe under CUDA-graph capture).
RAISED_MAX_BYTES = 8 << 20
#: The DFlash2 drafter's (id, value) pairs of its top-32 candidates: 64 FP32 words per row and rank.
_PAIR_WIDTH = 64
#: Kimi K3.1 hidden size: the drafter fc's output width, whose per-rank column slices the column-parallel fc gathers.
_HIDDEN = 7168


def lamport_allgather_serves(
    input: torch.Tensor,
    mapping: Mapping,
    all_reduce: Optional[AllReduce],
    max_bytes: int = MAX_BYTES,
) -> bool:
    """Whether the multicast kernel serves this call (else the op runs the engine all-gather); ``max_bytes`` caps the
    gathered message (at most ``RAISED_MAX_BYTES``)."""
    group = mnnvl_group(all_reduce)
    if group is None:
        return False
    world = group[0].tp_size
    width = input.shape[-1] if input.dim() else 0
    return (
        world > 1
        and world == mapping.tp_size
        and input.is_cuda
        and input.dtype == torch.float32
        and input.is_contiguous()
        and width % 4 == 0
        and input.numel() > 0
        and input.numel() * world * 4 <= min(max_bytes, RAISED_MAX_BYTES)
        and torch.cuda.get_device_capability(input.device) in ((10, 0), (10, 3))
    )


class LamportAllgather(OpWrapper):
    """Concatenate every TP rank's FP32 `input` along its last dim, in rank order.

    `mapping` names the TP group (as `allgather_module`'s); `all_reduce` is an engine `AllReduce` module
    built for the same group, whose MNNVL Lamport workspace the kernel uses. Where the group runs MNNVL,
    `input` is contiguous FP32 with a last dim that is a multiple of 4 and the message is at most
    `MAX_BYTES`, one kernel multicasts and polls the rows; otherwise `allgather_module` (the engine NCCL
    all-gather). Returns a new tensor `[..., world * width]`, identical on every rank.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"K3.1 DFlash2 candidate pairs, {rows} draft rows: {why}",
                spec=dict(world=4, shape=shape, special=special),
            )
            for rows, shape, special, why in (
                (1, (1, 1, _PAIR_WIDTH), False, "one request, one row"),
                (8, (1, 8, _PAIR_WIDTH), False, "ll B1 block of 8"),
                (24, (4, 6, _PAIR_WIDTH), False, "mid B4 blocks of 6"),
                (72, (12, 6, _PAIR_WIDTH), False, "M2 B12 blocks of 6"),
                (8, (1, 8, _PAIR_WIDTH), True, "-0.0 / inf / nan values travel"),
                (2048, (256, 8, _PAIR_WIDTH), False, "past MAX_BYTES at 4 ranks: engine fallback"),
            )
        )
        + tuple(
            Cell(
                why=f"K3.1 DFlash2 drafter fc output columns, {rows} rows: {why}",
                spec=dict(world=4, shape=(rows, _HIDDEN // 4 // 2), special=False),
            )
            for rows, why in (
                (8, "ll B1 block of 8, TP4 column slice (BF16 pairs as 32-bit words)"),
                (40, "B5 blocks of 8"),
            )
        )
        + tuple(
            Cell(
                why=f"K3.1 DFlash2 drafter fc output columns at the raised cap, {rows} rows: {why}",
                spec=dict(
                    world=4,
                    shape=(rows, _HIDDEN // 4 // 2),
                    special=False,
                    max_bytes=RAISED_MAX_BYTES,
                ),
            )
            for rows, why in (
                (155, "M2-size verify rows, past MAX_BYTES and served under RAISED_MAX_BYTES"),
                (600, "past RAISED_MAX_BYTES at 4 ranks: engine fallback"),
            )
        )
    )
    note = """
    The column-parallel DFlash2 drafter fc gathers its ranks' BF16 output columns through this entry as 32-bit words
    (each word two BF16 values; the cells hold random words, which is what such a view carries), with `max_bytes`
    raised to `RAISED_MAX_BYTES`; every other caller keeps the default `MAX_BYTES`. A raised cap stays within half
    the engine's smallest Lamport buffer, so it never grows the workspace.

    Not bitwise in one case: `-0.0` is the Lamport sentinel's bit pattern, so
    it is gathered as `+0.0` on every rank (the two compare equal; every other
    word, NaN payloads included, is moved unchanged). The DFlash2 candidate
    merge compares the values and reads the ids (non-negative), so its result
    does not change. The workspace is the group's engine MNNVL Lamport
    workspace, shared with every MNNVL all-reduce of the group: calls alternate
    on its rotation state, must be in the same order on every rank and ordered
    by stream. The call grows the workspace eagerly when needed; grow it before
    CUDA-graph capture (a warm-up call at the largest row count).
    """

    def __call__(
        self,
        input: torch.Tensor,
        mapping: Mapping,
        all_reduce: Optional[AllReduce],
        max_bytes: int = MAX_BYTES,
    ) -> torch.Tensor:
        if not lamport_allgather_serves(input, mapping, all_reduce, max_bytes):
            return allgather_module(input, mapping, -1, None)
        width = input.shape[-1]
        rows = input.numel() // width
        ws = lamport_workspace(all_reduce, rows, 2 * width, one_shot=True)
        out = torch.empty(
            (*input.shape[:-1], ws.world * width), dtype=torch.float32, device=input.device
        )
        _ag.run_lamport_allgather(
            ws, input.view(rows, width), out.view(rows, ws.world * width), get_env_enable_pdl()
        )
        return out

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        mapping: Mapping,
        all_reduce: Optional[AllReduce],
        max_bytes: int = MAX_BYTES,
    ) -> torch.Tensor:
        """What every rank should get back.

        The collective exception to mirroring `__call__`: takes every rank's `input` in rank order
        (`mapping`, `all_reduce` and `max_bytes` are ignored: both transports return the same words). The
        concatenation along the last dim, with `-0.0` written as `+0.0` (see the note).
        """
        out = torch.cat(list(inputs_in_rank_order), dim=-1)
        return torch.where((out == 0) & torch.signbit(out), torch.zeros_like(out), out)

    def is_valid(
        self,
        input: torch.Tensor,
        mapping: Mapping,
        all_reduce: Optional[AllReduce],
        max_bytes: int = MAX_BYTES,
    ) -> None:
        assert max_bytes <= RAISED_MAX_BYTES, (
            "max_bytes above RAISED_MAX_BYTES could grow the shared workspace (never under graph capture)"
        )
        group = mnnvl_group(all_reduce)
        if group is not None:
            assert group[0].tp_size == mapping.tp_size, (
                "all_reduce must be built for the gathered TP group"
            )


lamport_allgather = LamportAllgather()
