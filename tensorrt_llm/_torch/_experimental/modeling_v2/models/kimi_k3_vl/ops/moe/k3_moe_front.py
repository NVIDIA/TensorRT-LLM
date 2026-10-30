# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's MoE front at decode size in one kernel: the sharded MoE head GEMV, its all-gather over a caller-owned
:class:`K3MoeHeadWorkspace`, the top-16 routing, the MXFP8 latent, and the shared experts' gate_up + SiTU."""

from typing import Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

# Importing front_op registers trtllm::k3_moe_front. front_weight packs the front's one weight at load time;
# weight_supported reads metadata only.
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_fused_moe.front_op import (
    front_weight,
    weight_supported,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_fused_moe.op import (
    K3MoeHeadWorkspace,
)

from .k3_route_quant import k3_route_quant

__all__ = ["K3MoeHeadWorkspace", "front_weight", "k3_moe_front", "weight_supported"]


class K3MoeFront(OpWrapper):
    """Return ``(topk_ids, topk_weights, quantized, scales, shared)`` for the MoE input ``x`` (bf16 ``[M <= 8, 7168]``,
    the same on every rank): the top-16 routing of the gathered router logits with ``e_score_correction_bias`` and the
    MXFP8 latent with its UE8M0 scales, as ``trtllm::k3_route_quant`` returns them for the gathered head, and the shared
    experts' activation (bf16 ``[M, shared_cols]``). ``w_front`` from :func:`front_weight`. Advances ``workspace`` by
    one call: every rank of the group makes the same front calls on it in the same order.

    ``publish``: also release the workspace's per-token ready words, which the next ``moe/k3_moe`` call on a head_flags
    state (``head=workspace``) acquires; every publishing call is followed by exactly one such call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="four-rank gathered head and TP16-width shared activation, every decode row count",
            spec=dict(
                world_size=4,
                ms=tuple(range(1, 9)),
                hidden=7168,
                shared_cols=384,
                routed_scaling_factor=2.827,
                gate_cap=4.0,
                linear_cap=25.0,
                publish=False,
            ),
        ),
        Cell(
            why="publishing front releases per-token routing and latent to head-flags MoE",
            spec=dict(
                world_size=4,
                ms=tuple(range(1, 9)),
                hidden=7168,
                shared_cols=384,
                routed_scaling_factor=2.827,
                gate_cap=4.0,
                linear_cap=25.0,
                publish=True,
            ),
        ),
    )

    note = """
    Each rank supplies the same bf16 x [M,7168] and its own packed front
    weight. The head has 3584/W latent rows followed by 896/W router rows;
    front_weight pads to 128-row tiles and interleaves every 16 shared gate
    rows with 16 up rows. At W=4 its shape is [1920,7168], head rows 1120.
    Shared width is 384, the TP16 slice of two 3072-wide shared experts.
    The bias is fp32 [896]. Routing returns int32 ids and bf16 weights
    [M,16]; quantization returns e4m3 [M,3584] and uint8 UE8M0 [M,112];
    shared is bf16 [M,384]. All outputs are new contiguous tensors.
    The reference takes inputs and logical head weights in ascending group-rank order,
    plus the calling rank's logical gate/up rows: it computes the gathered
    head rather than reproducing the front_weight packing. Latent and shared
    GEMV outputs round to bf16 before routing/SiTU; latent -0 becomes +0.
    Head and shared sums accumulate eight split-K partials in cluster-rank
    order from +0. The exact-payload cells use x multiples of 1/8 in
    [-1/4,1/4], latent/shared weights multiples of 1/16 in [-1/8,1/8],
    router weights multiples of 1/8 in [-1/4,1/4]. Tests retain byte-exact
    stock routing/quantization, run-to-run and M-independent rows, and the
    shared activation's 2e-2 relative gate for fast tanh/sigmoid. General
    payload sums can differ from another GEMM: kernel tests bound changed
    routing to a 16th/17th-key gap <=1e-4 and >99.9% equal MXFP8 bytes.
    Routed experts, shared-down/rank reduction, routed reduction and latent-
    up remain outside this front; its producer owns x.

    One caller-owned K3MoeHeadWorkspace per TP group is created collectively
    in post_load_weights, before capture. create(mapping,fabric_handle=None)
    exchanges handles only within that TP group; fabric defaults to
    mapping.is_multi_node(). Its handle owns multicast memory and must stay
    alive. Every group rank must call even for a refusal: capture/free-memory
    eligibility and allocation success are agreed, all ranks raise if any
    cannot allocate, and MPI communicators are freed on refusal. The second
    agreement prevents pushes before peers have emptied buffers. Failure
    inside handle exchange can leave peers waiting. No environment is read
    at construction.

    uc/mc alias 157696 int32 words per rank at W=4,8,16: 43008 words of
    two [8,W,slot] Lamport halves (latent bf16 vectors and unused logit
    vectors), then 114688 router partial words [2,8,W,8,896/W].
    All start empty (0x80000000). flags int32 [4] is buffer index, unused
    zero, ready epoch, reader sign-in count; ready int32 [32] uses the first
    16 words for routing [0,8) and latent [8,16), initially zero. Rank,
    world, communicator and allocation handle complete the workspace.
    Every rank must issue the same sequence, token count and x, eager or
    replayed, in one stream order across workspaces and other collectives.
    Two workspaces have independent rotations/epochs but not independent
    cross-rank orders. Swapping same-shaped calls on one rank silently
    gathers mixed inputs; different M or missing calls can hang.

    Each front pushes into flags[0]'s half, waits for peers, empties what it
    reads, flips flags[0] once after all reader CTAs sign in, and resets
    flags[3]. No external clear or prior-M record is needed. A publishing
    call reads flags[2] before releasing dependents and releases token ready
    words as epoch+1. Exactly one head-flags k3_moe must follow, advancing
    the epoch and unused-token words; after it all 16 words equal the epoch.
    Plain front leaves the epoch/ready words alone. Nobody else may reset
    them: stale epoch+1 words admit data before publication. This holds
    across -1->0 and 2**31-1->-2**31. Workspaces are separate from MNNVL
    and sandwich state. Tests retain dip/regrow steps over three layers with
    rank skew, twenty interleaved-workspace calls, six rewritten-input graph
    replays with eager calls, epoch wraps, capture refusals, unsupported-M
    refusal with untouched state, and the mixed-input negative control.

    TRTLLM_ENABLE_PDL (default on) is read each call, scheduling only.
    The result-neutral compile cache keys device, W, shared width, tiles,
    clusters, K, ring, both caps, publish, PDL and half-tile geometry. First
    calls for each key must compile eagerly; cold capture raises before
    launch. ring=4 is not exposed. Cached device cluster capacity selects
    64-row half-tiles when they fit beside shared tiles and two role
    clusters, otherwise 128-row tiles: GB200's 15 clusters give 128-row
    tiles at W=4,8 and half-tiles at W=16 with shared width 384.
    The four-rank matrix certifies the former; recorded 16-rank matrix
    and kernel runs exercise the latter. Geometry has a separate CPU test.
    Unsupported shapes/worlds or small workspace fail loudly; supported
    W is 4,8,16, K a multiple of 1024, shared width a multiple of 64 with
    tiles fitting the device. mutates_args covers uc, mc, flags and ready.
    Rank agreement and producer/consumer order remain caller obligations,
    not locally observable is_valid checks.
    """

    def __call__(
        self,
        x: torch.Tensor,
        w_front: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        shared_cols: int,
        gate_cap: float,
        linear_cap: float,
        workspace: K3MoeHeadWorkspace,
        publish: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.k3_moe_front(
            x,
            w_front,
            e_score_correction_bias,
            routed_scaling_factor,
            shared_cols,
            gate_cap,
            linear_cap,
            workspace.uc,
            workspace.mc,
            workspace.flags,
            workspace.rank,
            workspace.world_size,
            ag_ready=workspace.ready if publish else None,
        )

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        head_weights: Sequence[torch.Tensor],
        gate_up_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        shared_cols: int,
        gate_cap: float,
        linear_cap: float,
        rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_cols = 3584 // len(per_rank_inputs)
        with true_fp32_matmul():
            heads = [x.float() @ w.float().t() for x, w in zip(per_rank_inputs, head_weights)]
            latent = torch.cat([h[:, :latent_cols] for h in heads], dim=1).bfloat16()
            latent = torch.where(latent == 0, torch.zeros_like(latent), latent)
            logits = torch.cat([h[:, latent_cols:] for h in heads], dim=1)
            ids, weights, quantized, scales = k3_route_quant.reference(
                logits, e_score_correction_bias, latent, routed_scaling_factor
            )
            gate_up = (
                (per_rank_inputs[rank].float() @ gate_up_weight.float().t()).bfloat16().float()
            )
        gate, up = gate_up[:, :shared_cols], gate_up[:, shared_cols:]
        shared = (
            gate_cap
            * torch.tanh(gate / gate_cap)
            * torch.sigmoid(gate)
            * (linear_cap * torch.tanh(up / linear_cap))
        )
        return ids, weights, quantized, scales, shared.bfloat16()


k3_moe_front = K3MoeFront()
