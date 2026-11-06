# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's post-attention sandwich: the row-parallel attention output projection, its TP all-reduce and the
residual update (attention-residual selection + RMSNorm) in one kernel, over a caller-owned
:class:`K3SandwichWorkspace`."""

from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

# Importing the op module registers trtllm::k3_sandwich_*; the state type is the one the ops' buffers come from.
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_sandwich.op import (
    K3SandwichWorkspace,
)

__all__ = ["K3SandwichWorkspace", "k3_sandwich_oproj"]


_WORKSPACE_NOTE = """
    K3SandwichWorkspace is caller-owned, one per TP group, created eagerly
    by the target in post_load_weights and shared by all three sandwich
    entries, including both drafter forms. uc and mc are unicast/multicast
    views of two halves [8, W, 7168] bf16, stored as int32 pairs with
    0x80000000 empty; pushes replace -0.0 by +0.0. The views occupy
    2*8*W*7168*2 bytes (0.92 MB at W=4, 3.67 MB at W=16), independent of M.
    flags is int32 [64]: 56 CTA counters and eight unused zero words.
    The object also owns the multicast handle, TP communicator, rank and
    world_size; it must live through every call and replay.

    create(mapping, fabric_handle=None) is collective over the TP group
    only (MPI_Comm_create_group); every group rank must call together.
    Capture and free-memory feasibility are agreed before allocating;
    a refusal or returned allocation failure raises RuntimeError on every
    rank and frees the newly created MPI communicator. A rank failing
    inside handle exchange may leave its peers waiting. Creation refuses
    capture, empties every word and zeros counters before the final
    agreement lets any rank return. fabric_handle defaults to
    mapping.is_multi_node(): fabric handles across nodes, POSIX descriptors
    otherwise; no environment variable is read for creation.

    All ranks must issue the same sandwich calls, forms and M in the same
    order, across layers, decode steps, eager calls and graph replays.
    Replicated residual streams, snapshot counts, prefix presence and norm
    weights must agree for identical results. Those distributed invariants
    cannot be checked by is_valid from one rank's arguments. Independent
    workspaces have independent counters, but their relative order on one
    stream must agree across ranks, including other polling collectives;
    reversed orders can deadlock. This is not an MnnvlWorkspace and calls
    do not advance MNNVL counters.

    Each CTA reads its count after griddepcontrol.wait, selects its parity
    half, pushes and increments its count once. Every reader empties words
    immediately after reading; the next reuse is two calls later, after
    the peer has finished reading, independent of token-count changes.
    Every intervening PDL kernel must wait for its predecessor. Dependents
    are released after all CTAs push but before outputs are written, so a
    PDL consumer must perform its own grid-dependency wait before reading.
    Same-shaped swapped calls silently pair the wrong rank inputs; an
    extra/missing call can leave later calls unmatched. No shared registry
    is kept; ws_uc, ws_mc and ws_flags are declared mutable.

    Matrices retain armed-size/counter checks, all-rank and one-rank capture
    refusals before allocation, 20 irregular calls on two workspaces,
    dip/regrow sequences at M=8,8,8,2,7,8,1,1,8,3,8 with a rank 5 ms late,
    graph replays with fresh inputs and eager calls between them, and the
    swapped-call negative control followed by recovery. The shared matrix
    interleaves three target layers (oproj then tail, one bank-row output)
    and two drafter layers (plain then SwiGLU) at (target M, drafter M)
    (8,3), (2,8), (7,7), then captures (8,4) and replays six times with
    eager calls of all three ops between replays. Size-varying sequences
    catch the stale-row re-arm failure once found in k3_spec_accept.
    CI uses four GB200 ranks, one GPU each, POSIX handles and TP16 per-rank
    shapes. A recorded four-tray fabric-handle run at W=16 also passed;
    W<=8 exercises only one rank-summation chunk. Kernel-test bitwise
    comparisons at W=16 are separate evidence, not this entry's receipt.
    """


def _reduce_reference(
    partials: Sequence[torch.Tensor], carry: Optional[torch.Tensor]
) -> torch.Tensor:
    total = torch.zeros_like(partials[0], dtype=torch.float32)
    for base in range(0, len(partials), 8):
        acc = torch.zeros_like(total)
        for part in partials[base : base + 8]:
            acc = acc + part.float()
        total = total + acc
    updated = total.bfloat16()
    if carry is not None:
        updated = (carry.float() + updated.float()).bfloat16()
    return updated


def _residual_reference(
    updated: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    v = torch.cat([block_residual, updated.unsqueeze(0)], dim=0).float()
    rs = (v.square().mean(dim=-1) + rms_eps).rsqrt()
    logits = (v * rs[..., None] * (rms_weight.float() * res_weight.float())).sum(dim=-1)
    probs = torch.softmax(logits, dim=0)
    mixed = (probs[..., None] * v).sum(dim=0).bfloat16()
    out = (
        mixed.float() * (mixed.float().square().mean(dim=-1, keepdim=True) + output_rms_eps).rsqrt()
    )
    return (out * output_rms_weight.float()).bfloat16(), mixed


def _valid_residual_inputs(
    rows: int,
    prefix: Optional[torch.Tensor],
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
) -> None:
    if prefix is not None:
        assert prefix.dtype == torch.bfloat16 and prefix.shape == (rows, 7168)
    if block_residual.shape[0]:
        assert block_residual.dtype == torch.bfloat16
        assert block_residual.ndim == 3 and block_residual.shape[1:] == (rows, 7168)
    for weight in (res_weight, rms_weight, output_rms_weight):
        assert weight.dtype == torch.bfloat16 and weight.shape == (7168,)


class K3SandwichOproj(OpWrapper):
    """Return ``(normed, updated)``: ``updated = prefix + allreduce(core @ o_weight^T)`` over ``workspace``'s TP group
    (the sum alone without ``prefix``), ``normed`` = RMSNorm(attn_res(block_residual..., updated)). Advances
    ``workspace`` by one call: every rank of the group makes the same calls on it in the same order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{tokens} decode tokens, {snapshots} snapshots, prefix={prefix}: attention projection",
            spec=dict(
                tokens=tokens,
                snapshots=snapshots,
                prefix=prefix,
                weights=tokens % 3,
                hidden=7168,
                k=768,
                dtype=torch.bfloat16,
                eps=1e-6,
                worlds=(4, 16),
            ),
        )
        for tokens in range(1, 9)
        for snapshots in (0, 1, 4, 8)
        for prefix in (True, False)
    )

    note = (
        _WORKSPACE_NOTE
        + """
    reference takes core and o_weight sequences in TP-rank order and the
    replicated residual inputs. Projection accumulates in true fp32, each
    rank's partial rounds once to bf16, the chunked rank sum rounds to
    bf16, then prefix is added and rounded. Attention-residual selection
    scores RMS-normalized [block_residual..., updated] against
    rms_weight*res_weight, softmaxes and mixes, then applies output RMSNorm.
    The kernel follows oneshotAllreduceAttnResKernel's statistics,
    summation orders and roundings; its bitwise equality to a separate
    o_proj and MNNVL attention-residual all-reduce is kernel-test evidence.
    Both outputs are fresh contiguous bf16 [M,7168]; inputs are read-only.
    Attention, snapshot ownership, prefix chaining and the consuming MoE
    remain outside; k3_sandwich_tail is the next pre-attention step.

    The op loudly rejects unsupported core/o_weight shapes, dtype/layout,
    M outside 1..8 or more than eight snapshots. is_valid checks the
    residual tensors and weights the op does not check; reshape may copy
    noncontiguous residual views. The result-neutral process compile cache
    is keyed by (W, publish, input source, PDL), W from workspace size.
    Every key needs an eager call before capture or raises RuntimeError.
    TRTLLM_ENABLE_PDL is read each call, default on, and changes scheduling
    rather than results.

    x_slab=None, slab_buf=0, src_slab=None and src_buf=0 stay inert:
    publishing normed or polling core uses three sentinel-armed buffers
    rotated by the caller's ordinal, not workspace state. Slab ownership
    and rotation would need their own state contract. A polled source
    writes outputs before its final dependency wait, so predecessors must
    no longer read their storage (synchronize or disable PDL); that mode
    is not exposed or certified here. The underlying schema also declares
    x_slab mutable.

    Single calls cover every M and snapshots 0,1,4,8 with/without prefix;
    sequences cover snapshots 0..8, twelve chained layers and eight graph
    replays at M=8. Exact power-of-two projection inputs make updated
    bitwise comparable; normed uses the existing 2e-2 maximum-magnitude
    fp32-reference band. Every output is bitwise equal across ranks.
    """
    )

    def __call__(
        self,
        core: torch.Tensor,
        o_weight: torch.Tensor,
        prefix: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
        workspace: K3SandwichWorkspace,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed, updated = torch.ops.trtllm.k3_sandwich_oproj(
            core,
            o_weight,
            prefix,
            block_residual,
            res_weight,
            rms_weight,
            output_rms_weight,
            rms_eps,
            output_rms_eps,
            workspace.uc,
            workspace.mc,
            workspace.flags,
            workspace.rank,
        )
        return normed, updated

    def reference(
        self,
        cores_in_rank_order: Sequence[torch.Tensor],
        weights_in_rank_order: Sequence[torch.Tensor],
        prefix: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with true_fp32_matmul():
            partials = [
                (core.float() @ weight.float().t()).bfloat16()
                for core, weight in zip(cores_in_rank_order, weights_in_rank_order)
            ]
        updated = _reduce_reference(partials, prefix)
        normed, _ = _residual_reference(
            updated,
            block_residual,
            res_weight,
            rms_weight,
            output_rms_weight,
            rms_eps,
            output_rms_eps,
        )
        return normed, updated

    def is_valid(
        self,
        core: torch.Tensor,
        o_weight: torch.Tensor,
        prefix: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
        workspace: K3SandwichWorkspace,
    ) -> None:
        _valid_residual_inputs(
            core.shape[0], prefix, block_residual, res_weight, rms_weight, output_rms_weight
        )


k3_sandwich_oproj = K3SandwichOproj()
