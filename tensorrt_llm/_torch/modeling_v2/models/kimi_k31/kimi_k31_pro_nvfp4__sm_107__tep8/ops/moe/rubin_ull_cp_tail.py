# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 column-parallel MoE tail on a BF16 latent_up (SM107, up to 32 rows): this rank's
128-blocks of latent_up's output, pushed into the all-gather rows of the tail's call.

The tail of a generation step then reads 1 / W of latent_up per rank: every rank computes its
columns ``[c0, c0 + n)`` over the full normalized latent and multicasts them into the call's
all-gather rows (one-shot); the shared experts' BF16 partial goes into the same call's slots
(``rubin_lamport_push``); the next layer's pre-attention mix finishes the call
(``finish_ag``): ``bf16(latent_up + bf16(all-reduced shared))``, the replicated tail's
``bf16(routed + shared)``. :func:`rubin_ull_tail_workspace` builds the call's workspace and
:func:`rubin_ull_cp_tail_share` this rank's columns and tactics.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._rubin_ull_common import (
    HIDDEN,
    LATENT,
    MAX_M,
    block_slice,
    push_payload,
    push_tactic,
    reduce_chunk,
    required_bytes,
)

if IS_CUTLASS_DSL_AVAILABLE:
    from ...kernels.moe.rubin_ull_push_gemm import AG, run_ull_push, tactic_of


def _cell(why: str, rows: int, world: int) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, world=world))


class RubinUllCpTail(OpWrapper):
    """Push ``latent [m, 3584] @ w_rows[n, 3584]^T`` (BF16) into features ``[c0, c0 + n)`` of the
    all-gather rows of the next call on ``ws`` (a ``LamportWorkspace``, one-shot). Returns None.

    ``w_rows`` are latent_up's rows ``[c0, c0 + n)`` (a view; whole 128-row blocks, the rank's
    :func:`block_slice` of the hidden size); ``tactic`` a ``(mma_m, mma_n, split_k, ab_stages)``
    tuple (``_rubin_ull_common.push_tactic``). Collective with the other ranks' pushes into the
    same call; the call's slots carry the shared partial and its consumer finishes it.
    """

    ARCHS: frozenset[Arch] = frozenset()

    CELLS: Tuple[Cell, ...] = (
        _cell("8 rows (a TEP8 b1 k7 verify step) over TP4: 1792 columns per rank", 8, 4),
        _cell("one row", 1, 4),
        _cell("16 rows: the next row tile", 16, 4),
        _cell("32 rows: the largest call", 32, 4),
        _cell("8 rows over TP2", 8, 2),
    )

    note = """
    One launch: the split-K GEMM's owner CTA stages each output tile token-major and multicasts
    it into the all-gather rows of every rank. The product is the split-K FP32 sum rounded once,
    a reordered accumulation of the reference's (the test states the band). The first call of a
    (tactic, world, columns) compiles and must be eager. The cells are one VR200 tray's worlds;
    TP8, the TEP8 group, spans two trays (896 columns per rank).
    """

    def __call__(
        self,
        latent: torch.Tensor,
        w_rows: torch.Tensor,
        ws,
        tactic: Tuple[int, int, int, int],
        *,
        c0: int,
    ) -> None:
        m, n = latent.shape[0], w_rows.shape[0]
        run_ull_push(
            latent,
            w_rows,
            torch.empty((m, n), dtype=torch.bfloat16, device=latent.device),
            peer_ptrs=ws.peer_ptrs.data_ptr(),
            mc_base=ws.mc_base,
            flags=ws.flags.data_ptr(),
            rank=ws.rank,
            world=ws.world,
            tactic=tactic_of(tactic),
            target=AG,
            row=HIDDEN,
            col0=c0,
        )

    def reference(
        self,
        latent: torch.Tensor,
        w_rows: torch.Tensor,
        ws,
        tactic: Tuple[int, int, int, int],
        *,
        c0: int,
    ) -> torch.Tensor:
        """The ``[m, n]`` BF16 values this rank stores at features ``[c0, c0 + n)`` of the
        all-gather rows: the product summed in FP32 and rounded once, ``-0`` as ``+0``. ``ws`` and
        ``tactic`` do not change them."""
        return push_payload(latent, w_rows)

    def is_valid(
        self,
        latent: torch.Tensor,
        w_rows: torch.Tensor,
        ws,
        tactic: Tuple[int, int, int, int],
        *,
        c0: int,
    ) -> None:
        assert latent.dtype == w_rows.dtype == torch.bfloat16, "BF16 latent and weight rows"
        assert latent.dim() == 2 and latent.shape[1] == LATENT, "latent [m, 3584]"
        assert 1 <= latent.shape[0] <= MAX_M, f"1 to {MAX_M} rows"
        assert w_rows.dim() == 2 and w_rows.shape[1] == LATENT, "w_rows [n, 3584]"
        # The operands are read as K-major tiles through their base addresses.
        assert latent.is_contiguous() and w_rows.is_contiguous(), "contiguous operands"
        # Another rank's columns land on top of this rank's without an error.
        assert (c0, w_rows.shape[0]) == block_slice(HIDDEN, ws.world, ws.rank), (
            "the rank's own 128-blocks of the hidden size"
        )
        need = required_bytes(latent.shape[0], HIDDEN, ws.world)
        assert need <= ws.bytes_per_buffer, "a workspace with room for the all-gather rows"


rubin_ull_cp_tail = RubinUllCpTail()


def rubin_ull_tail_workspace(mapping, group_size: int, max_tokens: int = MAX_M) -> dict:
    """The MoE tail's own MNNVL Lamport workspace (a runtime builder, not an entry): three
    buffers of ``max_tokens x (W + 1)`` BF16 rows of the hidden size (the one-shot slots plus the
    all-gather rows) and the 9-word state. Collective over ``mapping``'s TP group: every rank calls
    it at the same point of ``post_load_weights``, outside CUDA graph capture.

    Returns ``{"ws": LamportWorkspace, "handle": the multicast buffer, "chunk"}``; keep the
    handle alive. ``chunk`` is the mix finish's ranks per partial sum: over at most 8 ranks both
    C++ MNNVL all-reduce kernels sum the slots in one chunk in rank order, so the finish
    reproduces the shared experts' all-reduce whichever kernel it ran.
    """
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    from ....kernels.comm.rubin_lamport_ar import LamportWorkspace
    from ....ops.comm.rubin_push_allreduce import use_fabric_handle

    if mapping.tp_size != group_size:
        raise ValueError(f"the TP group has {mapping.tp_size} ranks, the layout {group_size}")
    if group_size > 8:
        raise ValueError(f"{group_size} ranks: the finish sums one chunk of at most 8 slots")
    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)["comm"]
    need = required_bytes(max_tokens, HIDDEN, mapping.tp_size)
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
        [0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=buffer.device
    )
    peers = [
        handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(mapping.tp_size)
    ]
    ws = LamportWorkspace(
        world=mapping.tp_size,
        rank=mapping.tp_rank,
        uc_local=buffer.data_ptr(),
        mc_base=handle.get_mc_buffer((1,), torch.uint8, 0).data_ptr(),
        peer_ptrs=torch.tensor(peers, dtype=torch.int64, device=buffer.device),
        flags=flags,
        bytes_per_buffer=buffer_bytes,
    )
    return {
        "ws": ws,
        "handle": handle,
        "chunk": reduce_chunk(group_size),
    }


def rubin_ull_cp_tail_share(world: int, rank: int, max_tokens: int = MAX_M) -> tuple:
    """``(c0, n, tactics)`` of ``rank`` (a runtime builder, not an entry): its 128-blocks
    ``[c0, c0 + n)`` of latent_up's rows (the hidden size) and the push GEMM's tactic for each
    row count, ``tactics[m]`` for ``m`` in ``[1, max_tokens]`` (``tactics[0]`` is None)."""
    if not 1 <= max_tokens <= MAX_M:
        raise ValueError(f"{max_tokens} rows: the tail serves 1 to {MAX_M}")
    c0, n = block_slice(HIDDEN, world, rank)
    return c0, n, (None, *(push_tactic(m, n, LATENT) for m in range(1, max_tokens + 1)))
