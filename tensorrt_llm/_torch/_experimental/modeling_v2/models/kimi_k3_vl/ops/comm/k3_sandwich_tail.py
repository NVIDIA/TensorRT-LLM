# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's pre-attention sandwich: the row-parallel MoE tail (the latent up projection of the normed latent slice and
the shared experts' down projection), its TP all-reduce and the next layer's residual update (attention-residual
selection + RMSNorm) in one kernel, over a caller-owned :class:`K3SandwichWorkspace`."""

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

from .k3_sandwich_oproj import (
    _WORKSPACE_NOTE,
    _reduce_reference,
    _residual_reference,
    _valid_residual_inputs,
)

__all__ = ["K3SandwichWorkspace", "k3_sandwich_tail"]


class K3SandwichTail(OpWrapper):
    """Return ``(normed, updated)``: ``updated = prefix + allreduce([rmsnorm(latent)[:, lo:lo+224] | act] @
    tail_weight^T)`` over ``workspace``'s TP group (the sum alone without ``prefix``), ``normed`` =
    RMSNorm(attn_res(block_residual..., updated)). ``tap``: also store the pre-norm attention-residual mixture there
    (``updated`` with ``tap_updated``). ``updated_out``: store ``updated`` there; it is then the returned ``updated``.
    Advances ``workspace`` by one call: every rank of the group makes the same calls on it in the same order."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{tokens} decode tokens, {snapshots} snapshots, prefix={prefix}: latent slice shift {shift}",
            spec=dict(
                case="single",
                tokens=tokens,
                snapshots=snapshots,
                prefix=prefix,
                weights=tokens % 3,
                shift=shift,
                hidden=7168,
                latent=3584,
                latent_slice=224,
                latent_pad=256,
                act=384,
                dtype=torch.bfloat16,
                eps=1e-6,
                worlds=(4, 16),
            ),
        )
        for shift, (tokens, snapshots, prefix) in enumerate(
            (tokens, snapshots, prefix)
            for tokens in range(1, 9)
            for snapshots in (0, 1, 4, 8)
            for prefix in (True, False)
        )
    ) + tuple(
        Cell(
            why=f"{tokens} decode tokens: strided mixture/updated taps and a bank-row updated_out",
            spec=dict(
                case="options",
                tokens=tokens,
                snapshots=3,
                options=({"tap": "mix"}, {"tap": "updated"}, {"updated_out": True}),
            ),
        )
        for tokens in range(1, 9)
    )

    note = (
        _WORKSPACE_NOTE
        + """
    reference takes the replicated reduced latent and the acts, tail
    weights and lo values in TP-rank order, with the replicated residual
    inputs. It accumulates both projections and the latent RMS in true
    fp32, applies the full latent row's RMS scale to the latent accumulator,
    adds the shared-expert accumulator, then rounds once per partial.
    The rank sum, prefix addition and attention-residual update follow
    k3_sandwich_oproj. A latent norm weight must already be folded into
    tail_weight[:, :224]; columns 224:256 are zero padding and 256:640
    hold the shared-expert projection. The kernel reads 256 latent columns
    from lo, with out-of-row columns zero, so padding must contribute zero.

    latent [M,3584] is replicated; act [M,384] and tail_weight [7168,640]
    are rank-local. Every lo=0,224,...,3360 is covered on every rank by
    rotating slice starts across the single-call cells, including the last
    slice's out-of-row padding. Inputs are read-only. Outputs are contiguous
    bf16 [M,7168], newly allocated unless updated_out supplies updated.
    The kernel itself returns an empty [0,7168] in that case; this wrapper
    returns the caller's updated_out. It must not overlap a tensor read by
    the call. tap stores either the bf16 pre-norm mixture or, when
    tap_updated, updated. It is certified as a column slice of [M,5*7168],
    with unit column stride, row stride divisible by eight and 16-byte
    alignment; updated_out is an aligned contiguous row of a [3,M,7168]
    bank. Optional outputs change placement, not values; other columns
    and bank rows must stay untouched.

    Unsupported latent/act/tail-weight shapes, bf16 layouts, M, snapshot
    counts, latent alignment, tap layout/alignment and updated_out
    layout/alignment fail loudly in the op. is_valid checks the unguarded
    lo, zero padding and residual tensor contracts, and updated_out
    overlap with contiguous inputs the kernel reads without a reshape copy.
    Every rank must use the same latent and replicated residual stream.
    Computing/reducing latent, shared-expert gate_up and activation,
    folding norm weights, snapshot ownership, prefix chaining and the
    consuming attention remain outside.

    The result-neutral process cache is keyed by (W, publish, input
    source, tap kind, PDL); no tap, mixture tap and updated tap are three
    kernels, each requiring an eager first call before capture or raising
    RuntimeError. updated_out is not a key. TRTLLM_ENABLE_PDL is read each
    call, default on, and affects scheduling, not results.
    x_slab/slab_buf and src_slab/src_buf remain inert, as in oproj.
    lat_uc/lat_flags are also not exposed: folding the latent all-reduce
    uses a separate K3SandwichLatentExchange with two [8,W,3584] bf16
    halves, a mod-six count and a latent-scale slab. Each push-only k3_moe
    must then be followed by exactly one tail on that same exchange;
    latent supplies shape only. That state and its ordering are not
    certified here. Polling/folded modes write outputs before the final
    dependency wait; predecessors must stop reading their storage first
    (synchronize or disable PDL). The underlying schema declares ws_uc,
    ws_mc, ws_flags, x_slab, lat_uc, lat_flags, tap and updated_out mutable.

    The old matrix's fp64 projection reference is relocated as true-fp32
    arithmetic; power-of-two operands still make both products exact.
    The kernel's latent rsqrt differs from torch's, so updated retains
    its 8e-3 maximum-magnitude band (about one bf16 ulp); normed and mixture
    taps retain 2e-2. Tapped updated is bitwise equal to returned updated,
    and every output is bitwise equal across ranks. The wrong-pairing
    control must put more than half of updated outside its band and the
    largest error above ten times it. Twelve-layer sequences cover
    snapshots 0..8; the M=8 graph includes both taps and a bank-row output,
    replayed eight times with rewritten inputs and eager calls between.
    """
    )

    def __call__(
        self,
        latent: torch.Tensor,
        act: torch.Tensor,
        tail_weight: torch.Tensor,
        lo: int,
        lat_eps: float,
        prefix: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
        workspace: K3SandwichWorkspace,
        tap: Optional[torch.Tensor] = None,
        tap_updated: bool = False,
        updated_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed, updated = torch.ops.trtllm.k3_sandwich_tail(
            latent,
            act,
            tail_weight,
            lo,
            lat_eps,
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
            tap=tap,
            tap_updated=tap_updated,
            updated_out=updated_out,
        )
        return normed, (updated if updated_out is None else updated_out)

    def reference(
        self,
        latent: torch.Tensor,
        acts_in_rank_order: Sequence[torch.Tensor],
        weights_in_rank_order: Sequence[torch.Tensor],
        los_in_rank_order: Sequence[int],
        lat_eps: float,
        prefix: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
        tap: Optional[torch.Tensor] = None,
        tap_updated: bool = False,
        updated_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lat = latent.float()
        scale = (lat.square().mean(dim=1, keepdim=True) + lat_eps).rsqrt()
        partials = []
        with true_fp32_matmul():
            for act, weight, lo in zip(
                acts_in_rank_order, weights_in_rank_order, los_in_rank_order
            ):
                w = weight.float()
                acc_lat = lat[:, lo : lo + 224] @ w[:, :224].t()
                acc_act = act.float() @ w[:, 256:].t()
                partials.append((acc_lat * scale + acc_act).bfloat16())
        updated = _reduce_reference(partials, prefix)
        normed, mixed = _residual_reference(
            updated,
            block_residual,
            res_weight,
            rms_weight,
            output_rms_weight,
            rms_eps,
            output_rms_eps,
        )
        if tap is not None:
            tap.copy_(updated if tap_updated else mixed)
        if updated_out is not None:
            updated_out.copy_(updated)
            updated = updated_out
        return normed, updated

    def is_valid(
        self,
        latent: torch.Tensor,
        act: torch.Tensor,
        tail_weight: torch.Tensor,
        lo: int,
        lat_eps: float,
        prefix: Optional[torch.Tensor],
        block_residual: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        output_rms_weight: torch.Tensor,
        rms_eps: float,
        output_rms_eps: float,
        workspace: K3SandwichWorkspace,
        tap: Optional[torch.Tensor] = None,
        tap_updated: bool = False,
        updated_out: Optional[torch.Tensor] = None,
    ) -> None:
        assert lo in range(0, 3361, 224), "lo must select one of the sixteen latent slices"
        assert bool((tail_weight[:, 224:256] == 0).all()), "latent padding must be zero"
        _valid_residual_inputs(
            latent.shape[0], prefix, block_residual, res_weight, rms_weight, output_rms_weight
        )
        if updated_out is not None:
            start = updated_out.data_ptr()
            end = start + updated_out.numel() * updated_out.element_size()
            for value in (
                latent,
                act,
                tail_weight,
                prefix,
                block_residual,
                res_weight,
                rms_weight,
                output_rms_weight,
            ):
                if value is not None and value.numel() and value.is_contiguous():
                    read_start = value.data_ptr()
                    read_end = read_start + value.numel() * value.element_size()
                    assert end <= read_start or read_end <= start, (
                        "updated_out must not overlap a tensor the call reads"
                    )


k3_sandwich_tail = K3SandwichTail()
