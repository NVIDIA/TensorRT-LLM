# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's routed experts at decode size: ``trtllm::k3_moe``, this rank's routed partial from the persistent CuTe
DSL kernel (FC1 + SiTU + FC2 with the routing-weighted combine over the TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers, read in
place), on caller-owned state: a :class:`K3MoeState` (up to 8 tokens) or :class:`K3MoeWideState` (up to 64) and one
:class:`K3MoeLayer` per MoE layer. Its inputs are the outputs of ``moe/k3_route_quant`` or ``moe/k3_moe_front``. The
push form stores the partial into every rank's ``K3LatentExchange`` for ``comm/k3_latent_reduce`` instead."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_fused_moe.latent_op import (
    K3LatentExchange,
)

# The state types (they launch nothing per call); importing the op module registers trtllm::k3_moe. is_supported reads
# metadata only.
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_fused_moe.op import (
    K3MoeHeadWorkspace,
    K3MoeLayer,
    K3MoeState,
    K3MoeWideState,
    is_supported,
)

__all__ = [
    "K3MoeHeadWorkspace",
    "K3MoeLayer",
    "K3MoeState",
    "K3MoeWideState",
    "is_supported",
    "k3_moe",
    "k3_moe_push",
]


def _deq_w(packed, sf):
    lut = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
        device=packed.device,
    )
    vals = torch.empty(packed.shape[0], packed.shape[1] * 2, device=packed.device)
    vals[:, 0::2] = lut[(packed & 0xF).long()]
    vals[:, 1::2] = lut[(packed >> 4).long()]
    return vals * torch.exp2(sf.float() - 127.0).repeat_interleave(32, dim=1)


def _requant(act):
    """The FC1 epilogue's MXFP8 requantization per 32 columns (round-up scale), dequantized."""
    rows, cols = act.shape
    blocks = act.reshape(rows, cols // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    ex = torch.ceil(torch.log2(amax / 448.0))
    ex = torch.where(amax == 0, torch.full_like(amax, -127.0), ex).clamp(-127.0, 127.0)
    scale = torch.exp2(ex)
    q8 = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q8.float() * scale).reshape(rows, cols)


class K3Moe(OpWrapper):
    """This rank's routed partial bf16 ``[M, 3584]`` for the routing and MXFP8 latent of ``k3_route_quant`` or
    ``k3_moe_front`` (``x_fp8`` float8_e4m3fn ``[M, 3584]``, ``x_sf`` its UE8M0 scales ``[M, 112]``, ``topk_ids``
    int32 and ``topk_weights`` bf16 ``[M, 16]``), over ``layer``'s experts (global ids
    ``[local_expert_offset, local_expert_offset + num_local)``); M <= 8 on a K3MoeState, <= 64 on a K3MoeWideState.

    ``head``: the TP group's K3MoeHeadWorkspace, for and only for a ``head_flags`` state's layers: the call then
    acquires the front's outputs through the workspace's ready words and advances its epoch, so the front call before
    it must have published them (``moe/k3_moe_front`` with ``publish=True``). ``out``: bf16, contiguous, at
    least ``[M, 3584]``; the call writes its first M rows and returns an empty ``[0, 3584]`` tensor. Writes the state's
    slab (left armed) and partial rows, and the layer's counters (left zero).

    Two calls on one layer, in either form, must not run back to back: a call claims its first tile from the layer's
    counters before its grid-dependency wait. Between the two, some kernel must wait for its predecessor before it
    triggers its dependents (``griddepcontrol.wait``, then ``launch_dependents``), or the stream is synchronized."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="M<=8 random routing over the TP4 x EP4 expert window",
            spec=dict(build="m8", case="random", ms=tuple(range(1, 9))),
        ),
        Cell(
            why="M<=8 disjoint local experts fill all 128 groups at M=8",
            spec=dict(build="m8", case="16_local_disjoint", ms=tuple(range(1, 9))),
        ),
        Cell(
            why="M<=8 no local experts must produce zero rows",
            spec=dict(build="m8", case="none_local", ms=tuple(range(1, 9))),
        ),
        Cell(
            why="wide random routing spans the eight-token boundary and M=64",
            spec=dict(build="m64", case="random", ms=(1, 2, 7, 8, 9, 16, 33, 40, 64)),
        ),
        Cell(
            why="wide expert sharing reaches the 324-group capacity at M=64",
            spec=dict(build="m64", case="group_cap", ms=(1, 2, 7, 8, 9, 16, 33, 40, 64)),
        ),
        Cell(
            why="wide no-local routing leaves zero rows at every certified batch size",
            spec=dict(build="m64", case="none_local", ms=(1, 2, 7, 8, 9, 16, 33, 40, 64)),
        ),
        Cell(
            why="caller-owned output writes only active rows on the eight-token build",
            spec=dict(build="m8", out=True, ms=(3, 8)),
        ),
        Cell(
            why="caller-owned output writes only active rows on the wide build",
            spec=dict(build="m64", out=True, ms=(9, 64)),
        ),
        Cell(
            why="four-rank front routing feeds the plain persistent build",
            spec=dict(build="front", head_flags=False, world_size=4, ms=tuple(range(1, 9))),
        ),
        Cell(
            why="four-rank publishing front hands routing and latent to the head-flags build",
            spec=dict(build="front", head_flags=True, world_size=4, ms=tuple(range(1, 9))),
        ),
    )

    note = """
    The cells use 224 local experts, intermediate 768, hidden 3584, top-16,
    and offset 224 in the single-rank test; the front matrix uses offset
    224*(rank%4). The inputs are contiguous e4m3 [M,3584], UE8M0 uint8
    [M,112], int32 ids and bf16 weights [M,16]. The four uint8 weight
    buffers are TRTLLM-Gen W4A8_MXFP4_MXFP8, read in place: [224,1536,1792],
    [224,1536,112], [224,3584,384], [224,3584,24]. Up/gate rows are
    interleaved and shuffled in 32-row blocks; scales are 128x4 interleaved.
    Like mxe4m3_mxe2m1_block_scale_moe_runner, reference takes the logical
    checkpoint slices the prepared operands were built from, rather than
    reconstructing the loader's layout. It consumes the call's routing and
    quantized latent; head and out affect ownership, not the arithmetic.
    SiTU caps are fixed at 4 and 25; the intermediate is MXFP8 with a
    round-up E8M0 scale per 32 columns. Routing, input quantization, rank
    reduction, latent-up, shared experts and residual are outside the op.

    The caller creates one state per device and one layer per MoE layer in
    post_load_weights, after weights are final. Layers keep the tensors,
    not copies: rebinding weights silently leaves old experts; reload in
    place or rebuild layers. Offset is not stored, so the caller must match
    it to the buffers' first global expert. Construction is noncollective
    and eager; every build's first call, including push, compiles eagerly
    before capture. Constructors and cold captured calls raise RuntimeError.
    All layers of a state share scratch and run in one stream order; two
    streams on one state are not supported. Separate states are independent.

    The m8 group capacity is min(num_local,128); the wide capacity is
    num_local+(1024-num_local)//8, 324 here. Scratch c is int8 [G,8,i_tp],
    cs int8 [G,8,i_tp/8], part fp32 [1024,3584] at these cells; counters
    are int32 [32+2G] per layer. c is armed with 0x80, cs with bytes 0..3
    of each 16-byte group 0xff and the rest zero, counters zero. part starts
    zero, then retains the last call's rows: unused token-slot loads in the
    m8 combine are discarded. The last readers restore sentinels and reset
    counters, so smaller calls need no separate clear. State also holds its
    build options, module and compile-cache membership.

    A plain build claims its first counter tile before its PDL wait. Between
    calls on the same layer, a kernel must wait before triggering its own
    dependents, or the stream must synchronize. k3_route_quant and
    k3_moe_front satisfy that order. No preceding kernel may write those
    counters. Other inputs and scratch are touched after the wait.
    A head-flags build instead acquires ready[t] and ready[8+t] at epoch+1,
    streams weights and writes slab/counters before waiting for the front's
    grid at FC2. The front must publish each token's data before its words,
    and not write weights, scratch, counters or epoch. Every publishing
    front is paired with exactly one head-flags call on the same workspace.
    At the last tile claim the call advances flags[2] and the unused tokens'
    ready words, leaving all 16 words equal to the epoch even across int32
    wraps. A missing front hangs; an unconsumed publish or reset epoch can
    admit stale data. Dependents of either build must wait before reading y.

    Push is M<=8 only: it writes this partial to slot (default exchange.rank)
    of half exchange.flags[0]&1 in every rank's int32 [2,8,slots,1792]
    exchange, converting -0 to +0 and writing zero for no local experts.
    It reads the half after its dependency wait and writes no exchange flag.
    Exactly one k3_latent_reduce of the same M must follow before another
    push on every rank in the same order; reduction empties the half and
    advances its count. The 4-rank and repeated-partial 16-slot push checks
    are bit-exact against MNNVL one-shot, including count wrap; recorded
    16-rank runs use distinct partials. Push helpers and state types remain
    public. out writes its first M rows, returns empty [0,3584], and leaves
    trailing rows untouched.

    Tests retain stock-runner gates, zero-local cases, re-arm checks,
    dip/regrow sequences, interleaved states, six graph replays with eager
    calls between them, head epoch wraps and stale-weight negative controls.
    The head-flags/front matrix is certified at four ranks and also has a
    recorded 16-rank run; kernel tests retain exhaustive numerics.
    K3MoeState(use_pdl=None) reads TRTLLM_ENABLE_PDL at construction
    (default on); wide defaults True. The op reads no environment.
    Process-wide module/build caches keyed by device and build options are
    result-neutral. Shape, dtype, contiguity, build/head mismatch and output
    capacity errors remain the op's loud checks; stream and producer
    ownership cannot be inferred from a call's arguments.
    """

    def __call__(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        layer: K3MoeLayer,
        head: Optional[K3MoeHeadWorkspace] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = layer.state
        if (head is not None) != state.head_flags:
            raise ValueError(
                "k3_moe: head is given for, and only for, the layers of a head_flags K3MoeState"
            )
        return torch.ops.trtllm.k3_moe(
            x_fp8,
            x_sf,
            topk_ids,
            topk_weights,
            *layer.weights,
            state.c,
            state.cs,
            state.part,
            layer.counters,
            local_expert_offset,
            state.num_local,
            state.num_ctas,
            state.m_max,
            state.use_pdl,
            None if head is None else head.ready,
            None if head is None else head.flags,
            out,
        )

    def reference(
        self,
        raw,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
    ) -> torch.Tensor:
        rows, hidden = x_fp8.shape
        x = x_fp8.float() * torch.exp2(
            x_sf.reshape(rows, hidden // 32).float() - 127.0
        ).repeat_interleave(32, dim=1)
        out = torch.zeros(rows, hidden, dtype=torch.float32, device=x_fp8.device)
        with true_fp32_matmul():
            for e in range(len(raw["up"])):
                tok, slot = (topk_ids == local_expert_offset + e).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                xe = x[tok]
                up = xe @ _deq_w(raw["up"][e], raw["up_s"][e]).t()
                gate = xe @ _deq_w(raw["gate"][e], raw["gate_s"][e]).t()
                act = (
                    4.0
                    * torch.tanh(gate / 4.0)
                    * torch.sigmoid(gate)
                    * (25.0 * torch.tanh(up / 25.0))
                )
                y = _requant(act) @ _deq_w(raw["down"][e], raw["down_s"][e]).t()
                out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
        return out.bfloat16()

    def is_valid(
        self,
        x_fp8: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        local_expert_offset: int,
        layer: K3MoeLayer,
        head: Optional[K3MoeHeadWorkspace] = None,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        device = layer.state.c.device
        assert device.index == torch.cuda.current_device(), "the state's device must be current"
        tensors = (x_fp8, x_sf, topk_ids, topk_weights, *layer.weights)
        if head is not None:
            tensors += (head.ready, head.flags)
        if out is not None:
            tensors += (out,)
        assert all(t.device == device for t in tensors), "inputs must be on the state's device"


k3_moe = K3Moe()


def k3_moe_push(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_offset: int,
    layer: K3MoeLayer,
    exchange: K3LatentExchange,
    slot: Optional[int] = None,
    head: Optional[K3MoeHeadWorkspace] = None,
) -> None:
    """The push form of :func:`k3_moe` for ``M <= 8`` on a K3MoeState: the same partial, stored into slot ``slot``
    (default ``exchange.rank``) of every rank's ``exchange`` (a TP group's ``K3LatentExchange``) instead of returned.
    One ``comm/k3_latent_reduce`` of the M tokens on that exchange must follow before the next push, on every rank in
    the same order. ``head`` as in :func:`k3_moe`. Writes the state's slab (left armed) and partial rows, the layer's
    counters (left zero), and every rank's exchange. As for :func:`k3_moe`, two calls on one layer must not run back to
    back."""
    state = layer.state
    if (head is not None) != state.head_flags:
        raise ValueError(
            "k3_moe_push: head is given for, and only for, the layers of a head_flags K3MoeState"
        )
    torch.ops.trtllm.k3_moe(
        x_fp8, x_sf, topk_ids, topk_weights, *layer.weights, state.c, state.cs, state.part, layer.counters,
        local_expert_offset, state.num_local, state.num_ctas, state.m_max, state.use_pdl,
        None if head is None else head.ready, None if head is None else head.flags, None,
        exchange.uc, exchange.mc, exchange.flags, exchange.rank if slot is None else slot,
    )  # fmt: skip
