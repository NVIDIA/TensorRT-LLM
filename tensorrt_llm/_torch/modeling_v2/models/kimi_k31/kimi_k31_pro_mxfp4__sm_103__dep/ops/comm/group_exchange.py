# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 all-gather / all-to-all of rows within a group of ranks, one persistent kernel per call over the
group's MNNVL Lamport workspace (``kimi_k31_pro_mxfp4__sm_103__dep/kernels/comm/lamport_group_exchange``)."""

from typing import Sequence

import torch

from tensorrt_llm._torch.distributed import ops as _dist_ops
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm.mapping import Mapping

from ...kernels.comm import lamport_group_exchange as _gx

#: Bytes of one Lamport buffer of a group workspace (the workspace holds three); a call that needs more runs in
#: row chunks. 16 MiB holds the other rank's rows of a 341-row [q | k | v | g] column-half exchange in one call.
GROUP_BUFFER_BYTES = 16 << 20

#: Lamport views of group workspaces, keyed by the workspace's local buffer address, its size and the rank.
_VIEWS: dict = {}


def group_workspace(
    group: Mapping, buffer_bytes: int = GROUP_BUFFER_BYTES
) -> "_lam.LamportWorkspace":
    """The Lamport view of the MNNVL workspace of ``group``'s TP group (``tp_size`` ranks), at least
    ``buffer_bytes`` per buffer.

    The first call for a group is collective over every rank of ``group``'s world (it splits the session
    communicator and maps the multicast buffer); make it at model build, in the same order on every rank, and
    never under CUDA-graph capture. Later calls return the cached view.
    """
    workspace = _dist_ops.get_or_scale_allreduce_mnnvl_workspace(
        group, torch.bfloat16, buffer_size_bytes=buffer_bytes
    )
    key = (workspace["uc_buffer"].data_ptr(), int(workspace["buffer_size_bytes"]), group.tp_rank)
    view = _VIEWS.get(key)
    if view is None:
        view = _lam.LamportWorkspace.from_mnnvl(workspace, group.tp_size, group.tp_rank)
        _VIEWS[key] = view
    return view


def _words(x: torch.Tensor) -> torch.Tensor:
    """``x`` [rows, ...] as FP32 words [rows, row bytes / 4]."""
    return x.reshape(x.shape[0], -1).view(torch.float32)


class LamportGroupExchange(OpWrapper):
    """All-gather (``bcast``) or all-to-all of rows within the group of ``ws`` (``ws.world`` ranks).

    All-gather: every rank passes ``x [m, ...]``; every rank gets ``[m, world * row]`` whose row ``t`` is the
    ranks' rows ``t`` concatenated in rank order. All-to-all: every rank passes ``x [m * world, ...]`` whose row
    ``t * world + d`` goes to rank ``d``; rank ``d`` gets ``[m, world * row]`` whose row ``t`` is the ranks' rows
    ``t * world + d`` concatenated in rank order. ``sections`` > 1 splits a row into that many equal sections and
    concatenates per section instead (a column-parallel ``[q | k | v | g]`` comes back as ``[q | k | v | g]``).
    The result has ``x``'s dtype; ``x`` is moved as raw 32-bit words, so any dtype whose section is a multiple of
    16 bytes travels.

    A call needs the other ranks' rows in one Lamport buffer; a larger call runs in row chunks, one kernel each.
    A one-rank group returns ``x`` itself.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=why,
            spec=dict(
                world=world, rows=rows, row=row, sections=sections, bcast=bcast, special=special
            ),
        )
        for why, world, rows, row, sections, bcast, special in (
            (
                "DEP16 KDA pair: all-gather of 100 rows x 7168 BF16 (c512 rows per rank)",
                2,
                100,
                7168,
                1,
                True,
                False,
            ),
            (
                "DEP16 KDA pair: [q | k | v | g] column halves of 200 rows back to their owners",
                2,
                100,
                24576,
                4,
                False,
                False,
            ),
            ("DEP16 KDA pair: one width-5 verify request, all-gather", 2, 5, 7168, 1, True, False),
            (
                "DEP16 KDA pair: one width-5 verify request, all-to-all",
                2,
                5,
                24576,
                4,
                False,
                False,
            ),
            (
                "DEP16 KDA pair: a 600-row prefill chunk past one buffer, all-to-all in 2 chunks",
                2,
                600,
                24576,
                4,
                False,
                False,
            ),
            ("tray of 4: all-gather of 100 rows x 7168 BF16", 4, 100, 7168, 1, True, False),
            ("tray of 4: all-to-all, 4 sections", 4, 100, 12288, 4, False, False),
            ("-0.0 / inf / nan words travel (pair all-to-all)", 2, 8, 512, 1, False, True),
        )
    )
    note = """
    Not bitwise in one case: `-0.0` is the Lamport sentinel's bit pattern, so a
    32-bit word equal to it arrives from another rank as `+0.0` (a BF16 pair
    whose high half is -0.0 and low half +0.0 is that word). Every other word,
    NaN payloads included, is moved unchanged; a rank's own rows are copied.
    Calls on a workspace alternate on its rotation state: they must be made in
    the same order on every rank of the group and ordered by stream. The
    workspace is sized when it is created (`group_workspace`); the call never
    grows it, so it captures into and replays from CUDA graphs.
    """

    def __call__(
        self, x: torch.Tensor, ws: "_lam.LamportWorkspace", sections: int, bcast: bool
    ) -> torch.Tensor:
        world = ws.world
        if world == 1:
            return x
        rows = x.shape[0]
        m = rows if bcast else rows // world
        src = _words(x)
        width = src.shape[1]
        out = torch.empty((m, world * width), dtype=torch.float32, device=x.device)
        chunk = max(1, ws.bytes_per_buffer // _gx.required_bytes(1, width, world))
        pdl = get_env_enable_pdl()
        step = 1 if bcast else world
        for t0 in range(0, m, chunk):
            t1 = min(m, t0 + chunk)
            _gx.run_group_exchange(
                ws, src[t0 * step : t1 * step], out[t0:t1], sections, bcast, use_pdl=pdl
            )
        return out.view(x.dtype).view(m, -1)

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        ws: "_lam.LamportWorkspace",
        sections: int,
        bcast: bool,
    ) -> torch.Tensor:
        """What rank ``ws.rank`` should get back.

        The collective exception to mirroring `__call__`: takes every rank's `x` in rank order. The other ranks'
        words equal to `-0.0` arrive as `+0.0` (see the note).
        """
        world, rank = len(inputs_in_rank_order), ws.rank
        parts = []
        for s, x in enumerate(inputs_in_rank_order):
            w = _words(x).view(torch.int32)
            if not bcast:
                w = w.view(w.shape[0] // world, world, -1)[:, rank]
            if s != rank:
                w = torch.where(w == torch.iinfo(torch.int32).min, torch.zeros_like(w), w)
            parts.append(w.reshape(w.shape[0], sections, -1))
        out = torch.stack(parts, dim=2).reshape(parts[0].shape[0], -1)  # [m, sections, world, sw]
        x0 = inputs_in_rank_order[0]
        return out.view(torch.float32).view(x0.dtype).view(out.shape[0], -1)

    def is_valid(
        self, x: torch.Tensor, ws: "_lam.LamportWorkspace", sections: int, bcast: bool
    ) -> None:
        assert x.is_cuda and x.is_contiguous(), "x must be a contiguous CUDA tensor"
        row_bytes = x[0].numel() * x.element_size() if x.shape[0] else 0
        assert row_bytes % (16 * sections) == 0, (
            "every section of a row must be a multiple of 16 bytes"
        )
        assert bcast or x.shape[0] % ws.world == 0, "an all-to-all takes world rows per output row"


lamport_group_exchange = LamportGroupExchange()
