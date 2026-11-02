# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts of two decode tokens with every expert on the rank (moe TP16 x EP1, intermediate <= 256):
one weight-stream CuTe DSL kernel over a caller-owned :class:`K3MoeM2State`, returning this rank's routed partial or
pushing it into a latent exchange for ``trtllm::k3_latent_reduce``."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.moe.k3_moe import k3_moe

from ...kernels.k3_moe_engines.op import K3MoeM2Layer, K3MoeM2State

__all__ = ["K3MoeM2Layer", "K3MoeM2State", "k3_moe_m2", "k3_moe_m2_push"]


class K3MoeM2(OpWrapper):
    """This rank's routed partial ``[2, 3584]`` bf16 of ``layer``'s experts for two tokens. Advances the state's
    workspace by one call: the calls on one state run in one stream order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16 x EP1 two-token grouped schedule with 192 real columns padded to 256",
            spec=dict(m=2, num_local=896, i_tp=192, i_pad=256, samples=8),
        ),
    )

    note = """
    This computes k3_moe_m1's M=2 routed sum with another schedule: distinct
    local experts take slots in ascending id, both tokens' intermediates
    are computed in every slot, and FC2 starts for each group of four slots
    when its FC1 rows are ready. MMA columns are independent. FC1, SiTU
    caps 4/25, round-up per-32-column MXFP8 intermediate, FC2 and weighted
    combine are inside; routing, latent quantization, rank sum and shared
    expert are outside. The ascending-id combine uses up to five slices
    with products/sums rounded separately. Repeated calls are bit-identical;
    FC1 partial sums can differ from k3_moe by one bf16 ulp of the row max
    in tests. reference uses the same prepared-operand exception as
    mxe4m3_mxe2m1_block_scale_moe_runner: logical checkpoint slices and the
    call's routing/MXFP8 latent instead of reconstructing shuffled weights.

    Inputs are contiguous e4m3 [2,3584], uint8 224 scale bytes (any shape),
    int32 ids and bf16 weights [2,16]. Offset 0 and all 896 experts are
    certified. The uint8 TRTLLM-Gen buffers [E,2*i_pad,1792],
    [E,2*i_pad,112], [E,3584,i_pad/2], [E,3584,i_pad/32] are read in
    place with the loader's zero padding 192->256; only i_tp real values
    are streamed. out, when given, is the returned bf16 [2,3584].

    One caller-owned K3MoeM2State per device, shared by every layer and
    both plain/push forms, is created in post_load_weights with
    create(device,i_tp,i_pad,num_local,push=((tp_size,1),)). Layer handles
    retain the weight tensors without allocating. create allocates zeroed
    scratch and compiles the plain and listed push builds without launching.
    It is eager, noncollective and refuses capture; any unlisted build's
    first call must also precede capture. CuTe DSL, sm_100, at least 112
    SMs, E<=896, i_pad a multiple of 128 and <=256, and 0<i_tp<=i_pad a
    multiple of 32 are required. Unsupported weights, padding, token count,
    shape and contiguity are the op's loud checks.

    Scratch is int8 hbuf [32,2,208] (13312 bytes), int32 counts [2,8,32]
    (eight groups, one 128-byte line each, in two parity sets), and int32
    epochs [SMs]. Calls on one state run in one stream order, eager and
    replayed alike. PDL waits precede producer reads/global writes, CTA 0
    clears the next parity set, and every CTA advances its epoch. No
    intermediate survives into another call and no reset is needed, even
    across int32 wrap. Unordered concurrent streams are outside the
    contract. Two states and M1's state are independent. No implicit
    metadata beyond layer/state and exchange is read; process-wide
    compiled-code caches are keyed by device/configuration.

    Push stores both tokens in this rank's slot of half
    exchange.flags[0]&1 of every rank's int32 [2,8,world,1792]
    K3LatentExchange, converting -0 to +0. It reads parity after its wait
    and writes no exchange flag. Exactly one k3_latent_reduce of two
    tokens follows before another push, in the same order on every rank.
    Reduction empties the half and advances count. Four-rank and repeated-
    partial 16-slot checks equal MNNVL one-shot bit for bit across count
    wrap; recorded 16-rank runs use distinct partials. Mutated buffers are
    hbuf, counts, epochs, out and the push exchange's multicast storage.
    Tests retain eight payloads, bitwise repeats, twelve steps of three
    layers, four rewritten-input graph replays with eager calls, two-state
    interleaving, epoch wraps and construction refusals. Kernel tests also
    compare k3_moe with both tokens routed to the same or disjoint experts.
    """

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        layer: K3MoeM2Layer,
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
        layer: K3MoeM2Layer,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        device = layer.state.hbuf.device
        tensors = (x_fp8, x_sf, topk_ids, topk_weights, *layer.weights)
        if out is not None:
            tensors += (out,)
        assert x_sf.dtype == torch.uint8, "MXFP8 scales are read as UE8M0 bytes"
        assert device.index == torch.cuda.current_device(), "the state's device must be current"
        assert all(t.device == device for t in tensors), "inputs must be on the state's device"


k3_moe_m2 = K3MoeM2()


def k3_moe_m2_push(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeM2Layer,
    exchange,
) -> None:
    """The same partials stored into this rank's slot of both tokens' rows of every rank's ``exchange`` (a TP
    group's ``K3LatentExchange``) instead of returned; one ``trtllm::k3_latent_reduce`` of the two tokens on that
    exchange must follow before the next push. Advances the state's workspace by one call."""
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
