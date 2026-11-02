# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot all-gather over a caller-owned :class:`MnnvlWorkspace` of fp32 rows whose leading columns travel, and are
gathered, as bf16 (Kimi K3's sharded MoE head: the latent columns in bf16, the router logits in fp32)."""

from typing import Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .mnnvl_workspace import MnnvlWorkspace

native_ops = load_family_ops(__name__)

__all__ = ["MnnvlWorkspace", "mnnvl_allgather_split", "required_buffer_bytes"]


def required_buffer_bytes(
    num_tokens: int, bf16_columns: int, fp32_columns: int, world_size: int
) -> int:
    """Bytes of one Lamport buffer a call occupies: every rank's rows, the bf16 columns at 2 bytes and the fp32 columns
    at 4."""
    return num_tokens * world_size * (bf16_columns * 2 + fp32_columns * 4)


class MnnvlAllgatherSplit(OpWrapper):
    """Gather ``input`` (fp32 ``[num_tokens, columns]``, this rank's slice) from every rank of ``workspace``'s TP group.
    Returns ``(bf16_out, fp32_out)``: ``bf16_out[t, r * bf16_columns + j] = bf16(input_r[t, j])`` and ``fp32_out[t, r
    * fp32_columns + j] = input_r[t, bf16_columns + j]``, ``fp32_columns = columns - bf16_columns``, in rank order.
    Takes one turn of the workspace's Lamport rotation, as an all-reduce on it does: every rank of the group makes the
    same MNNVL calls on it in the same order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{label}: bf16 rounding and fp32 words at every decode and wide token count",
            spec=dict(
                bf16_columns=b,
                fp32_columns=f,
                tokens=(1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64),
                dtype=torch.float32,
            ),
        )
        for label, b, f in (
            ("K3 head shard at TP2", 1792, 448),
            ("K3 head shard at TP4", 896, 224),
            ("K3 head shard at TP8", 448, 112),
            ("K3 head shard at TP16", 224, 56),
            ("smallest vector-aligned split", 8, 4),
        )
    )

    note = """
    The leading columns are rounded to bf16, ties to even; the remaining
    fp32 words travel unchanged. Both outputs replace -0.0 by +0.0 because
    fp32 -0.0 is the Lamport empty word. Inputs are read-only and outputs
    are fresh, contiguous tensors on the input device. The cells exercise
    normal bf16-column values over 2^-16..2^16, exact midpoints, negative
    zero in both parts, and fp32 infinities, NaN, signed denormals and the
    largest float. Outputs are checked bitwise against torch and across ranks.

    K3 gathers latent columns 3584/W and router logits 896/W after its
    row-sharded GEMV, for wide decode (9..64 tokens) and small decode
    without the fused MoE front. GEMV, routing and quantization stay outside;
    k3_route_quant_ag fuses this exchange with k3_route_quant.

    The caller owns MnnvlWorkspace; mnnvl_allreduce_attn_res.note states its
    construction, ownership and failure contract. Create it eagerly in
    post_load_weights, collectively over the TP group, before capture.
    A returned allocation failure is agreed across ranks; a failure inside
    handle exchange can strand peers. Creation arms all words and flags
    before any rank returns. Capture refusals on one rank or all ranks
    preserve existing workspaces, and the next eager create is exercised.

    Three fixed-size Lamport buffers share one multicast mapping.
    required_buffer_bytes is T*W*(2*B+4*F), 688128 bytes for K3 at T=64.
    The wrapper rejects an oversized call with ValueError before mutation;
    the raw op also checks its footprint. No workspace grows. The test
    sizes its buffer for the largest shared call, 1.75 MiB at W=4.
    All MNNVL entries share one rotation when given the same workspace:
    this gather and both all-reduce paths each take one turn. The flags
    are uint32[9]: current, dirty, capacity, dirty stage count, four dirty
    byte counts, arrival count. After this call they record next=current+1
    mod 3, dirty=current, unchanged capacity, one stage, (T*W*(2*B+4*F),
    0,0,0), and zero arrivals. The next call waits for the preceding grid,
    uses an armed buffer, and clears the previous dirty stages by their
    recorded sizes and layouts, including both stages after a two-shot sum.

    Every rank must issue the same number and order of calls, with the same
    op, T, B and F at each position, across eager calls and graph replays.
    Across different workspaces the stream's collective order must also
    agree: independent rotations are not independent orders. Timing need
    not agree; queued calls wait on their peers' matching positions.
    Swapping two same-shaped calls on one rank silently swaps that rank's
    gathered columns; the next call is correct after the pair realigns.
    Different layouts at the same position can hang; extra/missing calls
    were not exercised. These distributed invariants cannot be checked
    by a rank-local is_valid. Dtype, layout, alignment, split widths, empty
    input, raw world-size mismatch and capacity already fail loudly.
    The wrapper passes workspace.world_size; the raw op's shape argument
    also permits a fake implementation for tracing, not certified here.

    Tests retain 16 dipping/regrowing steps of six layers, two independent
    workspaces with 20 irregularly interleaved calls, mixed K3-order calls
    at T=8,2,16,64,1,7,32,8,3,16, and three rounds of 52 mixed calls queued
    without host synchronization. Randomly late ranks expose stale words.
    Clearing only the current call's rows previously passed single-call
    tests but failed dip/regrow sequences. A five-call captured mix is
    replayed eight times with fresh inputs and eager calls between replays;
    every result and rotation is checked, including negative controls.

    No metadata beyond workspace, process cache or compilation is used.
    The workspace handle keeps the multicast-pointer registry entry alive.
    TRTLLM_ENABLE_PDL is read once per process, default on from SM90.
    The kernel releases dependents after its grid wait, before outputs
    complete; consumers must wait for the grid. PDL does not change results.
    The matrix accepts mpirun or srun and W=2,4,8,16; certified runs used
    W=4 on one GB200 tray with POSIX handles and W=16 with fabric handles,
    PDL on. The gather accepts any W; shared all-reduces accept
    W=2,4,8,16,32,64. Zero-width parts, bf16-column denormals/nonfinites,
    accepted calls above 64 tokens and the fake implementation are untested.
    """

    def __call__(
        self,
        input: torch.Tensor,
        bf16_columns: int,
        workspace: MnnvlWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens, columns = input.shape
        need = required_buffer_bytes(
            num_tokens, bf16_columns, columns - bf16_columns, workspace.world_size
        )
        if need > workspace.buffer_bytes:
            raise ValueError(
                f"mnnvl_allgather_split: the call needs {need} bytes per Lamport buffer, the workspace has "
                f"{workspace.buffer_bytes}"
            )
        bf16_out, fp32_out = native_ops.mnnvl_allgather_split(
            input,
            bf16_columns,
            workspace.world_size,
            workspace.comm_buffer(torch.bfloat16),
            workspace.buffer_flags,
        )
        return bf16_out, fp32_out

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        bf16_columns: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every rank's leading columns rounded to bf16 by torch (round to nearest even) and its other columns as
        they are, in rank order, -0.0 made +0.0."""
        bf16_out = torch.cat([x[:, :bf16_columns] for x in inputs_in_rank_order], dim=1).bfloat16()
        fp32_out = torch.cat([x[:, bf16_columns:] for x in inputs_in_rank_order], dim=1)
        bf16_words = bf16_out.view(torch.int16)
        fp32_words = fp32_out.view(torch.int32)
        return (
            torch.where(bf16_words == torch.iinfo(torch.int16).min, 0, bf16_words).view(
                torch.bfloat16
            ),
            torch.where(fp32_words == torch.iinfo(torch.int32).min, 0, fp32_words).view(
                torch.float32
            ),
        )


mnnvl_allgather_split = MnnvlAllgatherSplit()
