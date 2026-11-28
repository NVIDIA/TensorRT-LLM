# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 vocab-parallel logits gather: BF16 vocab shards to FP32 full-vocab rows in one multicast kernel."""

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

#: Largest gathered message (rows x world x shard BF16 values, bytes) one kernel call carries: half the engine's
#: smallest MNNVL Lamport buffer (1024 x 8192 BF16), so a call never grows the workspace. A gather of more rows
#: runs as row chunks of at most this size (25 rows of K3.1's vocabulary), each its own Lamport round.
MAX_BYTES = 8 << 20
#: Kimi K3.1 vocabulary (the LM head's out_features).
_VOCAB = 163840


def rows_per_call(shard: int, world: int) -> int:
    """Most gathered rows of a ``world`` x ``shard`` BF16 row one kernel call carries (``MAX_BYTES``)."""
    return MAX_BYTES // (world * shard * 2)


def row_chunks(rows: int, per_call: int) -> list[tuple[int, int]]:
    """``[begin, end)`` bounds of the fewest even row chunks of at most ``per_call`` rows."""
    count = -(-rows // per_call)
    base, extra = divmod(rows, count)
    bounds, begin = [], 0
    for i in range(count):
        end = begin + base + (1 if i < extra else 0)
        bounds.append((begin, end))
        begin = end
    return bounds


def lamport_vocab_gather_serves(
    local: torch.Tensor, mapping: Mapping, all_reduce: Optional[AllReduce]
) -> bool:
    """Whether the multicast kernel serves this call (else the op runs the engine all-gather + ``.float()``)."""
    group = mnnvl_group(all_reduce)
    if group is None:
        return False
    world = group[0].tp_size
    return (
        world > 1
        and world == mapping.tp_size
        and local.is_cuda
        and local.dim() == 2
        and local.dtype == torch.bfloat16
        and local.is_contiguous()
        and local.shape[1] % 8 == 0
        and local.numel() > 0
        and rows_per_call(local.shape[1], world) > 0
        and torch.cuda.get_device_capability(local.device) in ((10, 0), (10, 3))
    )


class LamportVocabGather(OpWrapper):
    """``allgather(local, dim=-1).float()``: every TP rank's BF16 vocab shard ``local [rows, shard]``,
    concatenated in rank order and widened to FP32 ``[rows, world * shard]``.

    `mapping` names the TP group of the vocab-parallel LM head; `all_reduce` is an engine `AllReduce` module
    built for the same group, whose MNNVL Lamport workspace the kernel uses. Where the group runs MNNVL,
    `local` is contiguous 2-D BF16 with a shard width that is a multiple of 8, one kernel multicasts this rank's
    shard into every rank's Lamport buffer, polls the others' and writes the FP32 rows; more rows than one call
    carries (`MAX_BYTES`) run as even row chunks, one kernel call each, into the same output. Otherwise the engine
    NCCL all-gather (`allgather_module`) and `.float()`. Returns a new FP32 tensor, identical on every rank.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"K3.1 vocab-parallel logits, TP4, {rows} rows: {why}",
            spec=dict(world=4, rows=rows, shard=_VOCAB // 4, special=special),
        )
        for rows, special, why in (
            (1, False, "one row"),
            (5, False, "B1 target rows of a DFlash2 width-5 block"),
            (7, False, "B1 drafter rows of a width-8 block (7 draft positions)"),
            (8, False, "B1 target rows of a width-8 block (the ll bar)"),
            (8, True, "-0.0 / inf / nan values travel"),
            (25, False, "the most rows one call carries"),
            (26, False, "two calls of 13 rows"),
            (35, False, "TEP16 B7 width-5 target rows (two calls)"),
            (48, False, "B6 width-8 target rows (two calls)"),
            (60, True, "TEP16 B12 width-5 target rows (three calls), specials travel"),
            (128, False, "B16 width-8 target rows (six calls)"),
        )
    )
    note = """
    The vocab-parallel LM head's gather of its BF16 logits shards, as the K3.1 DFlash2 worker runs it twice
    per decode step: the target logits before sampling and the drafter's block logits before candidate
    selection (`allgather(local, dim=-1).float()` in the engine).

    Bitwise to the engine gather + `.float()` except one case: a 32-bit word whose bits are the Lamport
    sentinel 0x80000000 (an even column of exactly `+0.0` followed by an odd column of exactly `-0.0`)
    travels as 0, so that `-0.0` arrives as `+0.0` on every rank (the two compare equal). The reference
    models it. Every other value, NaN payloads included, is widened unchanged. A row is copied, never reduced,
    so a chunked gather writes the same bits as one call over all rows would.

    The workspace is the group's engine MNNVL Lamport workspace, shared with every MNNVL all-reduce of the
    group: calls alternate on its rotation state, must be in the same order on every rank and ordered by
    stream. Every call (one per row chunk) fits half the engine's smallest buffer, so it never grows the
    workspace (safe under CUDA-graph capture).

    Not driven here: groups wider than one GB300 tray (the attention TP16 / TP32 bar layouts are timed by
    the multi-tray microbenchmark), the MPI-disabled `_pg` path of the fallback.
    """

    def __call__(
        self, local: torch.Tensor, mapping: Mapping, all_reduce: Optional[AllReduce]
    ) -> torch.Tensor:
        if not lamport_vocab_gather_serves(local, mapping, all_reduce):
            return allgather_module(local, mapping, -1, None).float()
        rows, shard = local.shape
        world = mnnvl_group(all_reduce)[0].tp_size
        out = torch.empty((rows, world * shard), dtype=torch.float32, device=local.device)
        use_pdl = get_env_enable_pdl()
        for begin, end in row_chunks(rows, rows_per_call(shard, world)):
            ws = lamport_workspace(all_reduce, end - begin, shard, one_shot=True)
            _ag.run_lamport_vocab_gather(ws, local[begin:end], out[begin:end], use_pdl)
        return out

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        mapping: Mapping,
        all_reduce: Optional[AllReduce],
    ) -> torch.Tensor:
        """What every rank should get back.

        The collective exception to mirroring `__call__`: takes every rank's `local` in rank order
        (`mapping` and `all_reduce` are ignored). The concatenation along the last dim widened to FP32,
        with the sentinel word's `-0.0` written as `+0.0` (see the note).
        """
        cat = torch.cat([x.contiguous() for x in inputs_in_rank_order], dim=-1)
        words = cat.view(torch.int32)
        sentinel = torch.tensor(-(2**31), dtype=torch.int32, device=cat.device)
        words = torch.where(words == sentinel, torch.zeros_like(words), words)
        return words.view(torch.bfloat16).float()

    def is_valid(
        self, local: torch.Tensor, mapping: Mapping, all_reduce: Optional[AllReduce]
    ) -> None:
        assert local.dim() == 2, "local must be [rows, shard]"
        group = mnnvl_group(all_reduce)
        if group is not None:
            assert group[0].tp_size == mapping.tp_size, (
                "all_reduce must be built for the gathered TP group"
            )


lamport_vocab_gather = LamportVocabGather()
