# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts of one or two decode tokens with every expert on the rank (moe TP16 x EP1): one
weight-stream CuTe DSL kernel over a caller-owned :class:`K3MoeM1State`, returning this rank's routed partial or
pushing it into a latent exchange for ``trtllm::k3_latent_reduce``."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.moe.k3_moe import k3_moe

from ...kernels.k3_moe_engines.op import K3MoeM1Layer, K3MoeM1State

__all__ = ["K3MoeM1Layer", "K3MoeM1State", "k3_moe_m1", "k3_moe_m1_push"]


class K3MoeM1(OpWrapper):
    """This rank's routed partial ``[M, 3584]`` bf16 of ``layer``'s experts for the M tokens of its state. Advances
    the state's workspace by one call: the calls on one state run in one stream order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16 x EP1 single-token weight stream with 192 real columns padded to 256",
            spec=dict(m=1, num_local=896, i_tp=192, i_pad=256, samples=8),
        ),
        Cell(
            why="two tokens share the M1 workspace without changing either token's routed sum",
            spec=dict(m=2, num_local=896, i_tp=192, i_pad=256, samples=8),
        ),
    )

    note = """
    This is the M1 schedule of k3_moe's routed sum: FC1, SiTU caps 4/25,
    MXFP8 intermediate with round-up E8M0 scales per 32 columns, FC2 and
    the routing-weighted combine. Routing, latent quantization, rank sum
    and shared expert remain outside. Contiguous inputs are e4m3 [M,3584],
    uint8 M*112 scale bytes (any shape), int32 ids and bf16 weights [M,16].
    Offset 0 and all 896 experts are certified. TRTLLM-Gen uint8 buffers
    [E,2*i_pad,1792], [E,2*i_pad,112], [E,3584,i_pad/2],
    [E,3584,i_pad/32] are read in place; loader padding is zero and only
    i_tp real columns are streamed. reference uses the prepared-operand
    exception of mxe4m3_mxe2m1_block_scale_moe_runner: logical checkpoint
    slices plus the call's routing and MXFP8 latent, not shuffled buffers.
    The combine uses ascending local ids in up to five slices, with
    separately rounded products and sums. It is deterministic; FC1's two
    partials can differ from k3_moe by one bf16 ulp of the row maximum in
    tests. No local experts gives zero. out, when supplied, is returned.

    The caller creates one K3MoeM1State per device and token count in
    post_load_weights and each layer retains its weights without copying
    or allocating. create(device,i_tp,i_pad,num_local,num_tokens,
    push=((tp_size,1),)) allocates zeroed scratch and compiles the plain
    and listed push builds without launching. It is eager, noncollective,
    and refuses capture; unlisted configurations also need an eager first
    call. CuTe DSL, sm_100 and at least 112 SMs are required. Weight shape,
    padding, contiguity and unsupported token counts fail loudly; i_pad is
    a multiple of 128, 0<i_tp<=i_pad a multiple of 32 and E<=896.

    Layers and both forms share int8 hbuf [16*M,M,208] (3328 bytes at M=1,
    13312 at M=2), int32 counts [4] (two parity slots used), and int32
    epochs [SMs]. All calls on one state, eager and replayed, run in one
    stream order. PDL waits precede producer reads and global writes;
    CTA 0 clears the next parity slot and every CTA advances its epoch.
    Intermediate rows are read only within their call. No reset is needed,
    including across int32 wrap; concurrent unordered streams are outside
    the contract. Two states and the M2 workspace are independent.
    There is no implicit metadata beyond the layer/state and exchange;
    process-wide compiled-code caches are keyed by device/configuration.

    Push writes this rank's slot of half exchange.flags[0]&1 in every
    rank's int32 [2,8,world,1792] K3LatentExchange, -0 stored as +0.
    It reads parity after its dependency wait and writes no exchange flag.
    Exactly one k3_latent_reduce of M tokens must follow before the next
    push, in the same rank order; reduction empties the half and advances
    count. Four-rank checks and a repeated-partial 16-slot exchange match
    MNNVL one-shot bit for bit, including count wrap. Recorded 16-rank
    runs use one distinct partial per slot. Mutated buffers are hbuf,
    counts, epochs, out and the push exchange's multicast storage.
    Tests retain eight payloads per cell, bitwise repeats, twelve steps of
    three layers, four rewritten-input graph replays with eager calls,
    two-state interleaving, epoch wrap and eager-construction refusals.
    Kernel tests additionally cover TP4 x EP4 at M=1 and compare k3_moe.
    """

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        layer: K3MoeM1Layer,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer(x_fp8, x_sf, topk_ids, topk_weights, local_expert_offset, out)

    def reference(
        self,
        raw,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
    ) -> torch.Tensor:
        return k3_moe.reference(raw, x_fp8, x_sf, topk_ids, topk_weights, local_expert_offset)

    def is_valid(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        layer: K3MoeM1Layer,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        device = layer.state.hbuf.device
        tensors = (x_fp8, x_sf, topk_ids, topk_weights, *layer.weights)
        if out is not None:
            tensors += (out,)
        assert x_sf.dtype == torch.uint8, "MXFP8 scales are read as UE8M0 bytes"
        assert device.index == torch.cuda.current_device(), "the state's device must be current"
        assert all(t.device == device for t in tensors), "inputs must be on the state's device"


k3_moe_m1 = K3MoeM1()


def k3_moe_m1_push(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeM1Layer,
    exchange,
) -> None:
    """The same partial stored into this rank's slot of every rank's ``exchange`` (a TP group's
    ``K3LatentExchange``) instead of returned; one ``trtllm::k3_latent_reduce`` of the M tokens on that exchange must
    follow before the next push. Advances the state's workspace by one call."""
    layer.push(
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        local_expert_offset,
        exchange.mc,
        exchange.flags,
        exchange.rank,
    )
