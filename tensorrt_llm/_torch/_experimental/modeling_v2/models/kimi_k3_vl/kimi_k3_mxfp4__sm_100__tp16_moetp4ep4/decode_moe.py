# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode path's latent MoE on the catalog's Kimi K3 MoE entries.

**At most 8 tokens** (`MAX_TOKENS`), a MoE layer runs:

* `moe/k3_moe_front`, one kernel: this rank's slice of the MoE head (its latent-down rows and its router rows) as one
  GEMV, the slices' all-gather over the TP group's `K3MoeHeadWorkspace`, the top-16 routing, the MXFP8 latent, and
  the shared experts' gate_up + SiTU;
* `moe/k3_moe`: this rank's routed partial over its experts (on a `K3MoeState`);
* the latent all-reduce. On a pushing step (`DecodeStep.latent_push`: a pure decode step captured into a CUDA graph
  whose attention layers all run the decode kernels) the routed experts run as `moe/k3_moe`'s push form, which stores
  this rank's partial into every rank's `K3LatentExchange`, and `comm/k3_latent_reduce` sums the partials in the MNNVL
  one-shot's order, so with the same bits, while the experts' grid completes. On every other step it is the routed
  experts' all-reduce (one-shot, as `decode_comm.use_decode_one_shot` sets);
* the tail. The latent norm's weight is folded into the latent up projection at load, so
  `[RMSNorm(latent) slice | shared activation] @ [latent up columns | shared down]` is this rank's share of the MoE
  output. Where the next layer's pre-attention step (or the final norm) reduces it, the layer hands it on unreduced
  as a `PendingTail`, which the consumer runs with its all-reduce and residual update as one `comm/k3_sandwich_tail`
  kernel (`decode_comm.py`). Elsewhere the replicated tail runs: the latent RMS applied to the fp32 output of one
  GEMV with the folded latent up weight, plus the shared experts' down projection and its all-reduce.

**A wide decode step** (9 to 64 tokens, `WIDE_MAX_TOKENS`) keeps the sharded head and the row-parallel tail on
M-general ops: the head GEMV, `comm/mnnvl_allgather_split`, then `moe/k3_route_quant` and `moe/k3_moe` (on a
`K3MoeWideState`) beside the shared gate_up + SiTU, the latent all-reduce, and one GEMV of
`[RMSNorm(latent) slice | padding | shared activation]` with the tail weight. That is this rank's unreduced share,
which the consumer reduces with a plain all-reduce.

The GEMVs run on the decode GEMV sites of `decode_gemv.py` where they take the call, else on the stock GEMM ops.

`K3DecodeMoe` holds what every MoE layer shares: the head workspace and the latent exchange (both collective over
the TP group), the two `k3_moe` builds' scratch, and the TP group's MNNVL workspace (`decode_comm.K3DecodeComm`'s).
`K3DecodeMoeLayer` holds one layer's decode weights and its `k3_moe` counters. The target builds both in
`post_load_weights`, before any CUDA-graph capture, and runs every kernel once there so none compiles under a capture.

Every MoE layer's push and reduce go to the one exchange, in the stream's order: each push is followed by exactly one
reduce of the same token count before the next push, on every rank in the same order (the exchange's call-order
invariant, `comm/k3_latent_reduce`). Every kernel between a reduce and the next push waits for its predecessor (or
launches without programmatic dependent launch), which the decode kernels a pushing step runs do; pushes run only in
CUDA-graph replays, so no other step's kernels sit between two pushes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.k3_latent_reduce import (
    K3LatentExchange,
    k3_latent_reduce,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_allgather_split import (
    mnnvl_allgather_split,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.mnnvl_workspace import (
    MnnvlWorkspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe import (
    K3MoeHeadWorkspace,
    K3MoeLayer,
    K3MoeState,
    K3MoeWideState,
    is_supported,
    k3_moe,
    k3_moe_push,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_moe_front import (
    front_weight,
    k3_moe_front,
    weight_supported,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.k3_route_quant import k3_route_quant
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel

from .decode_comm import K3DecodeComm, PendingTail, wide_all_reduce

# The most tokens of the front and the small k3_moe build (one token tile), and of the wide build.
MAX_TOKENS = 8
WIDE_MAX_TOKENS = 64

# The tail weight's latent columns are zero-padded to whole 128-column k-tiles of the tail kernels
# (comm/k3_sandwich_tail takes the TP16 tail weight [7168, 256 + 384]).
TAIL_K_TILE = 256


@dataclass(eq=False)
class K3DecodeMoe:
    """What every MoE layer's decode path shares on one device: the TP group's ``K3MoeHeadWorkspace`` (the front's
    all-gather), the ``k3_moe`` builds for up to 8 and up to 64 tokens (their scratch; the layers run one at a time on
    one stream), the TP group's ``MnnvlWorkspace`` for a wide step's head all-gather, and the TP group's
    ``K3LatentExchange`` for a pushing step's latent all-reduce (None: every step keeps the routed experts'
    all-reduce). Built by `create`."""

    head: K3MoeHeadWorkspace
    small: K3MoeState
    wide: K3MoeWideState
    mnnvl: MnnvlWorkspace
    exchange: Optional[K3LatentExchange] = None

    @classmethod
    def create(
        cls, mapping, device, i_tp: int, num_local: int, mnnvl: MnnvlWorkspace, push: bool = True
    ) -> "K3DecodeMoe":
        """The state for ``mapping``'s TP group on ``device``, for experts of ``i_tp`` intermediate columns per rank,
        ``num_local`` of them on this rank. Collective (the head workspace and the latent exchange): every rank of the
        group calls it at the same point, eagerly, before any CUDA-graph capture. ``push``: build the latent exchange
        (where it takes the TP size: 4, 8 or 16 ranks); without it every step keeps the routed experts' all-reduce."""
        head = K3MoeHeadWorkspace.create(mapping)
        exchange = None
        if push:
            try:
                exchange = K3LatentExchange.create(mapping)
            # Raised before any collective step, on every rank of the group alike: a TP size the exchange does not take.
            except ValueError:
                exchange = None
        return cls(
            head,
            K3MoeState(device, i_tp, num_local),
            K3MoeWideState(device, i_tp, num_local),
            mnnvl,
            exchange,
        )


def _experts(moe: nn.Module) -> tuple:
    """The routed experts' TRTLLM-Gen W4A8_MXFP4_MXFP8 buffers ``k3_moe`` reads in place."""
    backend = moe.routed_experts.backend
    return (
        backend.w3_w1_weight,
        backend.w3_w1_weight_scale,
        backend.w2_weight,
        backend.w2_weight_scale,
    )


def layout_gaps(moe: nn.Module, tp_size: int, max_snapshots: int) -> list:
    """Why the decode path does not take MoE layer ``moe`` (empty when it takes it), read once the routed experts'
    buffers exist: the checks in order, up to the first that fails. ``max_snapshots``: the model's snapshot bank
    rows."""
    gate, backend = moe.gate, moe.routed_experts.backend
    shared = moe.shared_experts
    gate_up, down = shared.gate_up_proj.weight, shared.down_proj.weight
    if not moe._reduce_routed_output:
        return ["a routed output the model does not reduce"]
    if (moe.num_experts, moe.top_k, moe.moe_hidden_size, moe.hidden_size) != (896, 16, 3584, 7168):
        return [
            f"experts / top-k / latent / hidden {(moe.num_experts, moe.top_k, moe.moe_hidden_size)}"
        ]
    if (gate.num_expert_group, gate.topk_group) != (1, 1):
        return ["grouped routing"]
    if moe.moe_hidden_size % (8 * tp_size) or moe.num_experts % (4 * tp_size):
        return [f"a head slice of TP {tp_size}"]
    latent = (gate.weight, moe.routed_expert_down_proj.weight, moe.routed_expert_up_proj.weight)
    if not isinstance(moe.routed_expert_up_proj, nn.Linear) or any(
        w.dtype != torch.bfloat16 for w in latent
    ):
        return ["latent projections or a router other than bf16"]
    if (
        gate_up.dtype != torch.bfloat16
        or down.dtype != torch.bfloat16
        or shared.gate_up_proj.bias is not None
        or shared.down_proj.bias is not None
        or down.shape[0] != moe.hidden_size
    ):
        return ["a shared expert other than bf16 and unbiased"]
    names = (
        "w3_w1_weight",
        "w3_w1_weight_scale",
        "w2_weight",
        "w2_weight_scale",
        "expert_size_per_partition",
    )
    if (
        not all(hasattr(backend, name) for name in names)
        or not is_supported(*_experts(moe), backend.expert_size_per_partition)[0]
    ):
        return ["routed-expert buffers k3_moe does not read"]
    shared_cols, width = gate_up.shape[0] // 2, moe.moe_hidden_size // tp_size
    if not weight_supported(tp_size, shared_cols, gate_up.shape[1], gate_up.device):
        return ["a MoE front of this TP size and shared width"]
    tail_cols = width + (-width % TAIL_K_TILE) + shared_cols
    probe = gate_up.new_empty
    if not K3DecodeComm.takes_tail(
        probe(1, moe.moe_hidden_size),
        probe(1, shared_cols),
        probe(moe.hidden_size, tail_cols),
        max_snapshots,
    ):
        return ["a row-parallel tail the sandwich tail kernel does not take"]
    return []


def fold_latent_norm(moe: nn.Module) -> None:
    """Fold the latent RMSNorm's weight into the latent up projection's columns; the norm keeps a weight of ones, so
    every step computes the same function and the tails may normalize before slicing."""
    up, norm = moe.routed_expert_up_proj, moe.routed_expert_norm
    with torch.no_grad():
        up.weight.mul_(norm.weight.to(up.weight.dtype)[None, :])
        norm.weight.fill_(1)


@dataclass(eq=False)
class K3DecodeMoeLayer:
    """One MoE layer's decode path: the front weight (this rank's head slice padded to whole tiles, then the shared
    gate_up re-ordered), the head slice (a view of it), the row-parallel tail weight
    ``[latent up columns lo:lo+width | padding | shared down]``, the routing bias, and its ``k3_moe`` handles on the
    shared builds. Built by `create` once the weights are final (and the latent norm folded)."""

    state: K3DecodeMoe
    front_weight: torch.Tensor
    head_weight: torch.Tensor
    tail_weight: torch.Tensor
    tail_pad: Optional[torch.Tensor]
    # The gate's routing bias detached (a view of the parameter): an op given a tensor that requires grad while
    # autograd records, as in post_load_weights, returns outputs that require grad, which k3_moe cannot take.
    bias: torch.Tensor
    lo: int
    width: int
    shared_cols: int
    small: K3MoeLayer
    wide: K3MoeLayer

    @classmethod
    def create(
        cls, moe: nn.Module, state: K3DecodeMoe, tp_rank: int, tp_size: int
    ) -> "K3DecodeMoeLayer":
        """Build MoE layer ``moe``'s decode weights and its handles on ``state``'s builds (``layout_gaps`` empty)."""
        width = moe.moe_hidden_size // tp_size
        experts = moe.num_experts // tp_size
        gate_up = moe.shared_experts.gate_up_proj.weight
        shared_down = moe.shared_experts.down_proj.weight
        up = moe.routed_expert_up_proj.weight
        lo = tp_rank * width
        pad = -width % TAIL_K_TILE
        with torch.no_grad():
            head = torch.cat(
                [
                    moe.routed_expert_down_proj.weight[lo : lo + width],
                    moe.gate.weight[tp_rank * experts : (tp_rank + 1) * experts],
                ]
            )
            front = front_weight(head, gate_up)
            parts = [up[:, lo : lo + width]]
            if pad:
                parts.append(up.new_zeros(up.shape[0], pad))
            parts.append(shared_down)
            tail = torch.cat(parts, dim=1).contiguous()
        weights = _experts(moe)
        return cls(
            state=state,
            front_weight=front,
            head_weight=front[: head.shape[0]],
            tail_weight=tail,
            tail_pad=up.new_zeros(WIDE_MAX_TOKENS, pad) if pad else None,
            bias=moe.gate.e_score_correction_bias.detach(),
            lo=lo,
            width=width,
            shared_cols=gate_up.shape[0] // 2,
            small=state.small.layer(*weights),
            wide=state.wide.layer(*weights),
        )

    def warm_up(self, moe: nn.Module) -> None:
        """One call of every kernel of the decode path on zero inputs (M = 1), so none compiles under a capture: the
        front (collective: every rank makes the same call), both ``k3_moe`` builds, ``k3_route_quant`` and, with the
        latent exchange, the push build and the reduce (one push + reduce pair, collective). Once per model: every
        layer's calls compile the same builds."""
        device = self.front_weight.device
        x = torch.zeros(1, moe.hidden_size, dtype=torch.bfloat16, device=device)
        ids, weights, x_fp8, x_sf, _ = self._front(moe, x)
        offset = moe.routed_experts.backend.slot_start
        k3_moe(x_fp8, x_sf, ids, weights, offset, self.small)
        exchange = self.state.exchange
        if exchange is not None:
            # Two calls on one layer must not run back to back (moe/k3_moe): the push build starts once the call
            # above has ended.
            torch.cuda.synchronize(device)
            k3_moe_push(x_fp8, x_sf, ids, weights, offset, self.small, exchange)
            k3_latent_reduce(1, exchange)
        logits = torch.zeros(1, moe.num_experts, dtype=torch.float32, device=device)
        latent = torch.zeros(1, moe.moe_hidden_size, dtype=torch.bfloat16, device=device)
        ids, weights, x_fp8, x_sf = k3_route_quant(
            logits, self.bias, latent, float(moe.gate.routed_scaling_factor), early_trigger=True
        )
        k3_moe(x_fp8, x_sf, ids, weights, offset, self.wide)
        torch.cuda.synchronize(device)

    def _front(self, moe: nn.Module, x: torch.Tensor):
        return k3_moe_front(
            x.contiguous(),
            self.front_weight,
            self.bias,
            float(moe.gate.routed_scaling_factor),
            self.shared_cols,
            *moe._situ_betas,
            self.state.head,
        )

    def takes(self, hidden_states: torch.Tensor, step, partial_tail: bool) -> bool:
        """Whether this decode path runs the layer on ``step`` (bf16 rows ``hidden_states``): any step of at most
        `MAX_TOKENS` tokens; with ``partial_tail``, also a wide decode step."""
        rows = hidden_states.shape[0]
        if hidden_states.dtype != torch.bfloat16 or step is None:
            return False
        return 0 < rows <= MAX_TOKENS or (partial_tail and step.wide and rows <= WIDE_MAX_TOKENS)

    def forward(
        self,
        moe: nn.Module,
        hidden_states: torch.Tensor,
        gemvs,
        partial_tail: bool,
        push: bool = False,
    ) -> Union[torch.Tensor, PendingTail]:
        """The MoE output of ``hidden_states`` (``takes`` holds): with ``partial_tail``, this rank's unreduced share
        (a ``PendingTail`` at most `MAX_TOKENS` tokens); else the reduced output. ``gemvs``: the decode GEMVs' state,
        or None. ``push`` (the step's ``DecodeStep.latent_push``): at most `MAX_TOKENS` tokens, the latent all-reduce
        is the routed experts' push form plus ``comm/k3_latent_reduce`` on the state's exchange (the same bits as the
        routed experts' all-reduce); a state without the exchange ignores it."""
        if hidden_states.shape[0] > MAX_TOKENS:
            return self._wide(moe, hidden_states, gemvs)
        ids, weights, x_fp8, x_sf, shared_act = self._front(moe, hidden_states)
        offset = moe.routed_experts.backend.slot_start
        exchange = self.state.exchange
        if push and exchange is not None:
            k3_moe_push(x_fp8, x_sf, ids, weights, offset, self.small, exchange)
            latent = k3_latent_reduce(hidden_states.shape[0], exchange)
        else:
            routed = k3_moe(x_fp8, x_sf, ids, weights, offset, self.small)
            latent = moe.routed_experts.all_reduce(routed)
        if partial_tail:
            return PendingTail(
                latent.contiguous(),
                shared_act.contiguous(),
                self.tail_weight,
                self.lo,
                float(moe.routed_expert_norm.variance_epsilon),
            )
        # The replicated tail: the latent RMS on the fp32 accumulator of the folded latent up projection, plus the
        # shared experts' reduced output, rounded to bf16 once.
        shared = moe.shared_experts
        shared_out = shared.down_proj(shared_act, layer_idx=shared.layer_idx)
        up = _gemv(gemvs, "moe_up", latent, moe.routed_expert_up_proj.weight, out_fp32=True)
        scale = torch.rsqrt(
            latent.float().pow(2).mean(-1, keepdim=True) + moe.routed_expert_norm.variance_epsilon
        )
        return (up * scale + shared_out.float()).bfloat16()

    def _wide(self, moe: nn.Module, hidden_states: torch.Tensor, gemvs) -> torch.Tensor:
        """A wide decode step's MoE: this rank's unreduced share of the output, ``[M, hidden]`` bf16."""
        x = hidden_states.contiguous()
        head = _gemv(gemvs, "moe_head", x, self.head_weight, out_fp32=True)
        routed_in, router_logits = mnnvl_allgather_split(head, self.width, self.state.mnnvl)
        shared = moe.shared_experts

        def _routed_partial():
            ids, weights, x_fp8, x_sf = k3_route_quant(
                router_logits, self.bias, routed_in.contiguous(),
                float(moe.gate.routed_scaling_factor), early_trigger=True,
            )  # fmt: skip
            return k3_moe(
                x_fp8, x_sf, ids, weights, moe.routed_experts.backend.slot_start, self.wide
            )

        def _shared_activation():
            return shared._apply_activation(
                _gemv(gemvs, "moe_shared_gate_up", x, shared.gate_up_proj.weight)
            )

        routed, shared_act = maybe_execute_in_parallel(
            _routed_partial,
            _shared_activation,
            moe.moe_main_event,
            moe.moe_shared_event,
            moe.shared_expert_stream,
            disable_on_compile=True,
        )
        # The latent norm's weight is folded into the tail weight: normalize the whole latent row, keep this rank's
        # columns.
        normed = moe.routed_expert_norm(wide_all_reduce(moe.routed_experts.all_reduce, routed))
        parts = [normed[:, self.lo : self.lo + self.width]]
        if self.tail_pad is not None:
            parts.append(self.tail_pad[: x.shape[0]])
        parts.append(shared_act)
        return _gemv(gemvs, "moe_tail", torch.cat(parts, dim=1), self.tail_weight)


def _gemv(
    gemvs, site: str, x: torch.Tensor, weight: torch.Tensor, out_fp32: bool = False
) -> torch.Tensor:
    """``x @ weight.T`` on ``site``'s decode GEMV where it takes the call (``decode_gemv.K3DecodeGemvs.project``),
    else on the stock GEMM: cuBLAS through ``trtllm::dsv3_router_gemm_op`` for an fp32 output, ``F.linear`` else."""
    y = None if gemvs is None else gemvs.project(site, x, weight)
    if y is not None:
        return y
    if out_fp32:
        # The op reads the weight with a leading dimension of K: a strided weight would give wrong values.
        if not weight.is_contiguous():
            raise ValueError(
                f"the fp32 GEMM needs a contiguous weight, got strides {weight.stride()}"
            )
        return torch.ops.trtllm.dsv3_router_gemm_op(
            x.contiguous(), weight.t(), bias=None, out_dtype=torch.float32
        )
    return torch.nn.functional.linear(x, weight)
