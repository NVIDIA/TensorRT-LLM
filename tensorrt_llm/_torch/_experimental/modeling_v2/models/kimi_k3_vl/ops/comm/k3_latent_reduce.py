# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's latent all-reduce at decode size as the consumer of pushed partials: the sum over the TP group of the
routed partials every rank's producer stored into a caller-owned :class:`K3LatentExchange`, in the MNNVL one-shot's
order."""

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Importing the op module registers trtllm::k3_latent_reduce; the state type is the one its buffers come from.
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_fused_moe.latent_op import (
    K3LatentExchange,
)

__all__ = ["K3LatentExchange", "k3_latent_reduce"]


class K3LatentReduce(OpWrapper):
    """Return the latent rows ``[num_tokens, 3584]`` bf16: the sum over ``exchange``'s TP group of the partial rows
    every rank pushed into it since the previous reduce. Empties the words it read and advances ``exchange`` by one
    call: every rank pushes, then reduces, the same token count on it in the same order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{tokens} decode tokens, {kind} partials: {reason}",
            spec=dict(tokens=tokens, kind=kind, hidden=3584, dtype=torch.bfloat16, worlds=(4, 16)),
        )
        for tokens in range(1, 9)
        for kind, reason in (
            ("exact", "exact sums and re-arming both buffer halves"),
            ("cancel", "rank-order-sensitive cancellation against the MNNVL one-shot"),
            ("randn", "normally distributed partials against the MNNVL one-shot"),
        )
    )

    note = """
    The reference takes the producers' partials in TP-rank order, not an
    exchange. It sums fp32 chunks of eight ranks from +0, adds the chunks
    from +0 in order and rounds once to bf16, matching reduceOneshotLamport.
    Outputs are fresh contiguous bf16 [M, 3584] on exchange.uc.device;
    the current stream on that device is used. Computing and pushing the
    routed partial, shared experts and the latent up projection are outside.
    More than eight tokens need another all-reduce.

    K3LatentExchange is caller-owned, created by the target during
    post_load_weights with create(mapping, fabric_handle=None). It owns
    the multicast handle, TP-group communicator, rank and world_size;
    uc and mc view int32 [2, 8, W, 1792] (bf16 pairs, even column low).
    0x80000000 is empty. The views occupy 448 KiB at W=4, 896 KiB at W=8,
    1.75 MiB at W=16, before multicast-granularity rounding. flags is int32
    [4]: call count at 0, CTA arrivals at 2, unused zero words at 1 and 3.
    All words start empty and flags zero; after each call flags is
    [count, 0, 0, 0]. The exchange is valid while its owning handle lives.

    Creation is eager and collective over the TP group only. MPI uses
    MPI_Comm_create_group, not a session-wide split. All group ranks agree
    on capture/free-memory feasibility before allocating, and on returned
    allocation failures; any refusal raises RuntimeError on every rank
    and frees the newly made MPI communicator. A rank stuck inside the
    handle exchange can strand its peers. Creation refuses graph capture
    and returns after every rank has armed the buffer. fabric_handle
    defaults to mapping.is_multi_node(): fabric handles across nodes,
    POSIX descriptors otherwise; no environment setting controls it.
    TP sizes other than 4, 8, 16 raise ValueError before collective work.

    One producer push of M rows must be followed by exactly one reduce of
    that M before the next push, in identical call order and token counts
    on every rank, across layers, eager calls and graph replays. Producers
    use push_args() (uc, mc, flags, rank), read flags[0] after their
    grid-dependency wait, and store bf16 pairs through mc into their rank's
    slot in half flags[0]&1. They must turn -0.0 into +0.0: the sentinel
    otherwise looks unwritten forever. Pushes do not wait for peers.
    These inter-rank and producer invariants cannot be checked by is_valid
    from this consumer's arguments.

    The reduce reads its half and counts CTAs before its grid-dependency
    wait, polls rows 0..M-1 afterwards, empties every word it read after
    storing output, and CTA 0 resets arrivals and advances the count after
    all M*CTAs arrivals. Rows M..7 are untouched. Only count parity matters,
    including int32 wrap. Kernels between a reduce and the next push must
    wait for predecessors, or launch without PDL, so the next push cannot
    read the old count. Hosts may enqueue arbitrarily far ahead; GPUs run
    at most one call ahead, so a half is empty before its reuse two calls
    later. One exchange serves all MoE layers; two exchanges have separate
    counts but must keep the same relative ordering on every rank, as must
    other polling collectives, or they deadlock. It is not MnnvlWorkspace,
    K3SandwichWorkspace or K3SandwichLatentExchange.

    Swapping same-shaped pushes silently pairs different calls. Reducing
    fewer rows than were pushed leaves stale rows that a later larger
    reduce can consume; both are negative controls. Missing pushes, the
    wrong half or divergent call counts can wait forever and are not run.
    Unsupported M=0,9 is rejected before touching state. Cold compilation
    under capture raises before launch, leaving the exchange untouched.
    The result-neutral process cache is keyed by (W, CTAs per token, PDL);
    M is runtime. TRTLLM_ENABLE_PDL is read each call (default 1), affecting
    scheduling, not results; each key needs an eager call before capture.
    The wrapper fixes ctas_per_token=0: four CTAs through W=8, fourteen at
    W=16. Explicit 4/14/28 choices belong to the kernel test, not this entry.

    The matrix emulates pushes with copies, including negative zero; actual
    push-only routed-expert producers and their device-side half selection
    and overlap under PDL are not certified here. Captured copies fix their
    half, so captured steps and eager calls between replays are even.
    Tests retain 11 dip/regrow steps of 12 layers, skewed ranks and whole-step
    run-ahead, 20 irregularly interleaved calls on two exchanges, int32 wrap,
    and alternating M=8/3 graphs replayed four times each with fresh partials.
    Every result is bitwise checked against the reference and across ranks;
    cancellation changes at least a quarter of elements under another order.
    CI certifies four GB200 ranks, POSIX handles; a recorded 16-rank,
    four-tray fabric-handle run also passed. W=8 is not measured.
    The op declares lat_uc and lat_flags mutable; producer mc writes
    belong to the producers' schemas.
    """

    def __call__(self, num_tokens: int, exchange: K3LatentExchange) -> torch.Tensor:
        return torch.ops.trtllm.k3_latent_reduce(exchange.uc, exchange.flags, num_tokens)

    def reference(self, inputs_in_rank_order: Sequence[torch.Tensor]) -> torch.Tensor:
        total = torch.zeros_like(inputs_in_rank_order[0], dtype=torch.float32)
        for base in range(0, len(inputs_in_rank_order), 8):
            acc = torch.zeros_like(total)
            for part in inputs_in_rank_order[base : base + 8]:
                acc = acc + part.float()
            total = total + acc
        return total.bfloat16()


k3_latent_reduce = K3LatentReduce()
