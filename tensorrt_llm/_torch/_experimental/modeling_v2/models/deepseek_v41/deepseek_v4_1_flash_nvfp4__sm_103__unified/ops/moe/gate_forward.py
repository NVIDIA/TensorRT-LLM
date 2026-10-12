# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sqrt(softplus) MoE routing: biased top-k selection, unbiased renormalized weights.

The op is `torch.ops.trtllm.gate_forward`, registered in
`cpp/tensorrt_llm/thop/moe/routing/moeGateOp.cpp`; this entry always runs its
bias mode (`is_hash=False`).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: (n_routed_experts, num_experts_per_tok): the V4.1 main MoE, the DSpark drafter
#: MoE, and the third pair the op dispatches.
_SURFACES = ((384, 6), (128, 3), (256, 6))
#: Decode-like through prefill-like token counts.
_SWEEP_T = (1, 2, 3, 4, 5, 6, 7, 8, 15, 30, 40, 64, 160, 1024, 4096, 8192, 262144)
#: (logit sigma, bias centre): the checkpoint's per-layer biases sit in a narrow band
#: around a large offset (9.1..57.6 over the 40 layers); the drafter's are exactly 0.
_DISTS = ((1.0, 0.0), (4.0, 12.0), (12.0, 57.0))
_ROUTE_SCALE = 1.5  # routed_scaling_factor


def _warp_sum(w: torch.Tensor) -> torch.Tensor:
    """Sum of the topk weights in the kernel's order: a 32-lane xor butterfly
    (offsets 16, 8, 4, 2, 1) with lane j holding slot j and the other lanes 0.

    For 6 addends this is ((w0 + w4) + w2) + ((w1 + w5) + w3); for 3 it is
    (w0 + w2) + w1.
    """
    lanes = [w[:, j] for j in range(w.shape[1])]
    lanes += [torch.zeros_like(lanes[0])] * (32 - len(lanes))
    for offset in (16, 8, 4, 2, 1):
        lanes = [lanes[j] + lanes[j ^ offset] for j in range(32)]
    return lanes[0]


def _cell(why: str, surface: tuple[int, int], tokens: int, **overrides) -> Cell:
    spec = dict(
        E=surface[0],
        topk=surface[1],
        T=tokens,
        sigma=3.0,
        center=10.0,
        route_scale=_ROUTE_SCALE,
        out=False,
    )
    spec.update(overrides)
    return Cell(why=why, spec=spec)


class GateForward(OpWrapper):
    """Route each token to `topk` experts by `sqrt(softplus(logit)) + bias`.

    For every row of `router_logits` `[T, E]`, in fp32:
    `s = sqrt(L > 20 ? L : log1p(exp(L)))` (torch's softplus threshold); the
    `topk` experts with the largest `s + bias`, by decreasing score, equal
    scores in ascending expert id; their weights from the *unbiased* `s`,
    divided by `sum + 1e-20` (the sum in a warp-butterfly order) and multiplied
    by `route_scale`. Bias mode only.

    Fusion boundary: softplus, sqrt, the bias add, top-k, the gather, the
    renormalization and the scaling are inside the one launch. The caller owns
    the router GEMM (fp32 logits), any `bias` / `bias_vl` choice and everything
    downstream. `out_weights` (fp32) and `out_indices` (int32), when given, are
    written in place; otherwise both are allocated. Returns
    `(topk_weights [T, topk] fp32, topk_ids [T, topk] int32)`, weights first.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            _cell(
                f"{e}/{k} T={t} sigma={sigma:g} bias={center:g}: token sweep",
                (e, k),
                t,
                sigma=sigma,
                center=center,
            )
            for e, k in _SURFACES
            for t in _SWEEP_T
            for sigma, center in _DISTS
        )
        + tuple(
            _cell(f"{e}/{k} route_scale={s:g}: T=512 sigma=3 bias=10", (e, k), 512, route_scale=s)
            for e, k in _SURFACES[:2]
            for s in (1.0, 1.5, 2.5, 0.0, -1.0)
        )
        + tuple(
            _cell(
                f"{e}/{k} caller out buffers: T=8192, written and returned", (e, k), 8192, out=True
            )
            for e, k in _SURFACES[:2]
        )
        + tuple(
            _cell(f"{e}/{k} T=0: two empty [0, topk] outputs", (e, k), 0) for e, k in _SURFACES[:2]
        )
    )

    note = """
    Metadata consumed: none. Stateless: no cache, workspace or tuner; runs on
    the ambient stream; repeated calls and a call on another stream give
    identical bits. CUDA-graph capturable (the allocation of absent outputs is
    a graph-pool allocation).

    Matches the checkpoint's `Gate.forward` (`gate_temp` 1, `norm_topk_prob`
    true) bit for bit on real data. The kernel uses torch's softplus threshold
    (`L > 20` gives `L`, else `log1pf(expf(L))`) and divides by `sum + 1e-20`,
    as the checkpoint does, so from the kernel source: every finite logit gives
    a finite score (no expf overflow; the golden prompts' logits reach 52.63),
    and a row whose selected scores are all exactly 0 (every logit <= ~-104, or
    -inf) gets weights `0 / 1e-20 * route_scale`, i.e. 0. Only a `+inf` logit
    still gives NaN (that expert, `inf / inf`) and 0 for the row's others, and
    a NaN logit is selected first and makes the row's weights NaN. One
    difference remains, at exact ties: equal selection scores come out in
    ascending expert id where `torch.topk` emits another order; with the
    checkpoint's large biases 29 of 161,360 golden rows reorder tied experts
    (same set, weights within 2 ulp).

    This entry passes one bias per call: the image-span
    `torch.where(image_mask, bias_vl, bias)` is two calls plus a per-token
    select of both outputs (bit-exact). The op's optional `image_mask` /
    `vision_bias` arguments (a one-call per-token bias) are not exposed by this
    entry and not certified here. Weights
    are fp32; the trtllm-gen MoE runners take bf16 `topk_weights`, so the
    caller casts. The summation order (the 32-lane butterfly above) is pinned:
    another order matched only 78-83% of the weights.

    Preconditions: `(E, topk)` one of `(384, 6)`, `(128, 3)`, `(256, 6)` (the
    op raises otherwise); fp32 logits and bias, fp32 / int32 outputs, CUDA
    tensors (the op raises on each). A CPU `bias` is not rejected and is read
    from host memory: not certified. Layouts the op does not police are in
    `is_valid`; an empty bias is an illegal memory access that kills the CUDA
    context. T = 0 returns two `[0, topk]` tensors without a launch: the op's
    grid would have zero blocks, which CUDA rejects as an invalid launch.

    Row axis: one ahead-of-time kernel per `(E, topk)`, one warp per row, the
    row count only sizes the grid. At `(384, 6)`, `route_scale` 1.5, every T
    1..8192 (the target's per-call MoE chunk cap) returns the first T rows of
    the 8192-row call bitwise; at `(128, 3)` with a zero bias (the drafter's
    call) every T 1..1280 does. Captured at `(384, 6)` at the 167 per-rank
    sizes 1..6144 (G and 6G, and the attention-DP gathered 2G/12G, 4G/24G,
    8G/48G over the batch sizes 1..32, 40..128 step 8, 192, 256), plus dep8's
    9216 and 12288 rows as two chunk calls (8192 + 1024, 8192 + 4096) in one
    graph; at `(128, 3)` at the 46 sizes 5G, 5..1280.

    Not certified: hash mode; a CPU bias; at `(384, 6)` capture of one call
    above 6144 rows or at a size outside the 167, and other chunk splits; at
    `(128, 3)` the prefix sweep above 1280, capture outside the 46 sizes (other
    than T 40) or with a nonzero bias; several calls in one graph other than
    dep8's two chunks; the prefix sweep and capture list at `(256, 6)`; the
    drafter's real input distribution (certified on proxy inputs).
    """

    def raw_call(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        topk: int,
        route_scale: float,
        out_weights: Optional[torch.Tensor] = None,
        out_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = router_logits.shape[0]
        if out_weights is None:
            out_weights = torch.empty(
                (num_tokens, topk), dtype=torch.float32, device=router_logits.device
            )
        if out_indices is None:
            out_indices = torch.empty(
                (num_tokens, topk), dtype=torch.int32, device=router_logits.device
            )
        if num_tokens == 0:
            # The kernel's grid is one block per four rows, and a zero-block launch is
            # cudaErrorInvalidValue; there is nothing to route, so nothing is launched.
            return out_weights, out_indices
        # The hash mode's two inputs; empty in bias mode (never read).
        no_ids = torch.empty(0, dtype=torch.int32, device=router_logits.device)
        torch.ops.trtllm.gate_forward(
            router_logits,
            bias,
            no_ids,
            no_ids,
            out_weights,
            out_indices,
            topk,
            route_scale,
            False,
        )
        return out_weights, out_indices

    def reference(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        topk: int,
        route_scale: float,
        out_weights: Optional[torch.Tensor] = None,
        out_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The checkpoint's `Gate.forward` in fp32, from native torch ops.

        The kernel's numerics: `sqrt(softplus(x))` with the softplus threshold
        20 (`x > 20` gives `x`, else `log1p(exp(x))`), a *stable* descending
        sort of `scores + bias` (equal selection scores in ascending expert id,
        the kernel's tie-break), the gather from the bare scores, the
        warp-order sum, `/ (sum + 1e-20)` and then `* route_scale`.
        `out_weights` and `out_indices` are ignored and not mutated; the
        expected outputs are new tensors.
        """
        x = router_logits.float()
        scores = F.softplus(x, beta=1.0, threshold=20.0).sqrt()
        choice = scores + bias.float()
        ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk]
        weights = scores.gather(1, ids)
        weights = weights / (_warp_sum(weights).unsqueeze(-1) + 1e-20)
        weights = weights * route_scale
        return weights, ids.to(torch.int32)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        bias: torch.Tensor,
        topk: int,
        route_scale: float,
        out_weights: Optional[torch.Tensor] = None,
        out_indices: Optional[torch.Tensor] = None,
    ) -> None:
        # Pure-metadata guards. Each violation below was observed on sm_103 to be
        # accepted by the op and to return a plausible-looking wrong answer (or to
        # read/write outside the tensor) instead of raising: the kernel addresses
        # every tensor as a dense row-major buffer from data_ptr(), reads E =
        # size(1) bias elements and writes T * topk outputs regardless of shapes.
        assert router_logits.dim() == 2, (
            f"router_logits must be 2-D [T, E], got {tuple(router_logits.shape)}; a 3-D "
            "input is read as size(0) rows of size(1) experts"
        )
        assert router_logits.is_contiguous(), (
            "router_logits must be contiguous; a strided view is read as a dense "
            "[T, E] buffer and silently gives wrong routing"
        )
        num_tokens, num_experts = router_logits.shape
        assert tuple(bias.shape) == (num_experts,), (
            f"bias must be [{num_experts}], got {tuple(bias.shape)}; the kernel reads "
            f"{num_experts} elements from bias.data_ptr() whatever its length"
        )
        assert bias.is_contiguous(), (
            "bias must be contiguous; a strided view is read as a dense [E] buffer "
            "and silently gives wrong routing"
        )
        for name, out in (("out_weights", out_weights), ("out_indices", out_indices)):
            if out is None:
                continue  # raw_call allocates it at [T, topk]
            assert tuple(out.shape) == (num_tokens, topk) and out.is_contiguous(), (
                f"{name} must be contiguous [{num_tokens}, {topk}], got "
                f"{tuple(out.shape)} strides {out.stride()}; the kernel writes a dense "
                "[T, topk] block from data_ptr()"
            )

    def compare(
        self,
        out: tuple[torch.Tensor, torch.Tensor],
        ref: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Bit-exact on both the ids and the weights.

        The kernel evaluates the same fp32 operations in the same order as the
        reference: the softplus threshold 20 and `log1pf(expf(x))` below it
        (bit-identical to torch's softplus), a correctly rounded `sqrtf`, one
        fp32 add of the bias, the warp-order sum, one add of 1e-20, one
        division, one multiply. Measured on sm_103 bit-exact on every element of
        every certified case, so any tolerance would only hide a defect; the
        wrong-reference controls (weights from the biased score, sigmoid scoring,
        a missing `route_scale`) differ on 100% of the weights.
        """
        (weights, ids), (ref_weights, ref_ids) = out, ref
        assert weights.dtype == torch.float32 and ids.dtype == torch.int32, (
            weights.dtype,
            ids.dtype,
        )
        assert weights.shape == ref_weights.shape and ids.shape == ref_ids.shape, (
            weights.shape,
            ids.shape,
        )
        assert torch.equal(ids, ref_ids), (
            f"{int((ids != ref_ids).any(-1).sum())} of {ids.shape[0]} rows route differently"
        )
        assert torch.equal(weights, ref_weights), (
            f"{int((weights != ref_weights).sum())} of {weights.numel()} weights differ, "
            f"max abs {float((weights - ref_weights).abs().max()):.3e}"
        )


gate_forward = GateForward()
