# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed-MoE prologue: top-16 routing + MXFP8 quantization of the latent in one launch."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import mxfp8_quantize

from .route_topk import k31_route_topk

if IS_CUTLASS_DSL_AVAILABLE:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import (
        moe_prologue as _k31_moe_prologue,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import (
        route_tail as _k3_route_tail,
    )

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 3584
MAX_ROWS = 65536
#: Up to this many rows the attention-DP separated routing takes the routing-tail kernel, above it noaux_tc.
ROUTE_TAIL_MAX_ROWS = 2048
#: Up to this many rows the fused C++ routing + quantization op serves (the general target under attention DP); the
#: routing tail starts above it.
FUSED_C_MAX_ROWS = 64
#: Hidden alignment of the routed-expert input quantization (the TRTLLM-Gen W4A8 weights' K alignment).
INPUT_HIDDEN_ALIGNMENT = 512


def _runner_routing(logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The TRTLLM-Gen runner's in-kernel one-group DeepSeek-V3 routing (unit scale), in plain torch: the
    tanh-form sigmoid, the top-16 of ``sigmoid + bias`` (descending, ties to the lower expert), the
    recovered sigmoid ``(sigmoid + bias) - bias``, its butterfly warp sum (lanes 16..31 add zeros, then
    xor 8, 4, 2, 1), the FP32 quotient rounded to BF16."""
    s = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    key = s + bias.float()
    ids = torch.sort(key, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    recovered = torch.gather(key, 1, ids) - bias.float()[ids]
    total = recovered
    lanes = torch.arange(TOP_K, device=logits.device)
    for off in (8, 4, 2, 1):
        total = total + total[:, lanes ^ off]
    return ids.to(torch.int32), (recovered / total).to(torch.bfloat16)


def _rows(rows: int, why: str, scale: float = 1.0) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, scale=scale))


class K31MoePrologue(OpWrapper):
    """``(topk_ids, topk_weights, x_fp8, x_sf)`` of FP32 router logits ``[M, 896]`` and the BF16
    routed-expert input ``x [M, 3584]``: what the routed-expert scheduler's separated routing
    produces before the experts (top-16 by ``sigmoid(logit) + bias``, renormalized unbiased
    sigmoid weights times ``routed_scaling_factor`` in BF16, and ``mxfp8_quantize(x)`` with linear
    UE8M0 scales), from one CuTe DSL launch whose routing and quantization CTAs run side by side.

    Returns ``int32 [M, 16]``, ``bf16 [M, 16]``, ``float8_e4m3fn [M, 3584]``, ``uint8 [M, 112]``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _rows(1, "one row: one routing CTA, one quantization CTA"),
        _rows(8, "TEP32: one speculative verify step (FP64-rounded weights, <= 64 rows)"),
        _rows(64, "last row count of the FP64-rounded weights (the fused C++ op's range)"),
        _rows(65, "first row count of the FP32-then-BF16 weights (routing tail + mxfp8_quantize)"),
        _rows(224, "DEP16 at 32 sequences x 7 rows: 28 routing CTAs + 112 quantization CTAs"),
        _rows(2048, "last row count of the attention-DP routing tail: 256 + 256 CTAs"),
        _rows(4096, "a DEP16 prefill MoE chunk on its source rank (separated routing: noaux_tc)"),
        _rows(16384, "a TEP8 prefill chunk: 2048 routing CTAs + 2048 quantization CTAs"),
        _rows(65536, "largest launch"),
        _rows(300, "a scale other than 1 (the FP64 renormalization path)", scale=2.5),
        *(
            Cell(
                why=f"attention TP, {r} rows: the runner's in-kernel routing weights",
                spec=dict(rows=r, scale=1.0, runner_weights=True),
            )
            for r in (1, 8, 64, 72, 2048, 16384)
        ),
    )

    note = """
    `runner_weights=True` (attention TP, where the experts otherwise route inside the TRTLLM-Gen runner
    from the logits): the same experts in the same order (the runner's top-k key is noaux_tc's:
    `sigmoid + bias` bits above `65535 - expert`) and the runner's weights - the sigmoid recovered
    as `(sigmoid + bias) - bias` in FP32, the butterfly warp sum of the 16, an FP32 round-to-nearest
    division, BF16 - bit for bit (unit routed scale; the op raises otherwise). The separated
    routing paths' weights differ from those in about 1 of 10^5-10^6 weights (FP64 or FP32
    renormalization of the unbiased sigmoid), enough to change greedy outputs.

    Bitwise contract with `reference` (the separated routing) otherwise: up to 64 rows
    `kimi_k3_noaux_tc_mxfp8_quant` (weights rounded once from the FP64 expression), above that
    `k31_route_topk` (to 2048 rows) or `noaux_tc_op` (the BF16 cast of noaux_tc's FP32 weight
    either way) followed by `mxfp8_quantize(x, False, alignment=512)`. The selection code is the routing-tail kernel's
    (ties to the lower expert id, as noaux_tc). K3.1 routes with one expert group; 896 experts, top-16 and the
    3584-wide latent are compile-time constants (logits FP32, bias FP32, x BF16, all dense and
    16-byte aligned; the op raises otherwise). The kernel keeps no state: a captured CUDA graph
    replays it as is. Launched with programmatic dependent launch (waits on the previous grid
    before reading logits / x, releases dependents after its stores).
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        x: torch.Tensor,
        runner_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _k31_moe_prologue.k31_moe_prologue(
            router_logits, e_score_correction_bias, routed_scaling_factor, x, runner_weights
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        x: torch.Tensor,
        runner_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The separated routing, two launches (one up to 64 rows); with ``runner_weights`` the TRTLLM-Gen
        runner's in-kernel routing in plain torch (see the note) and ``mxfp8_quantize``."""
        m = router_logits.shape[0]
        if runner_weights:
            ids, weights = _runner_routing(router_logits, e_score_correction_bias)
            xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=INPUT_HIDDEN_ALIGNMENT)
            return ids, weights, xq, xsf.view(m, -1)
        if m <= FUSED_C_MAX_ROWS:
            ids, weights, xq, xsf = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
                router_logits, e_score_correction_bias, x, routed_scaling_factor
            )
            return ids, weights, xq, xsf.view(m, -1)
        if m <= ROUTE_TAIL_MAX_ROWS:
            ids, weights = _k3_route_tail.k31_route_topk(
                router_logits, e_score_correction_bias, routed_scaling_factor
            )
        else:
            weights, ids = torch.ops.trtllm.noaux_tc_op(
                router_logits, e_score_correction_bias, 1, 1, TOP_K, routed_scaling_factor
            )
            ids, weights = ids.to(torch.int32), weights.to(torch.bfloat16)
        xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=INPUT_HIDDEN_ALIGNMENT)
        return ids, weights, xq, xsf.view(m, -1)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        x: torch.Tensor,
        runner_weights: bool = False,
    ) -> None:
        """Nothing to add: the op raises on every input it would answer wrongly (dtype, width,
        stride, alignment, row count), which the test drives."""
        return None


k31_moe_prologue = K31MoePrologue()


class K31MoeRoute(OpWrapper):
    """The routing half of :class:`K31MoePrologue` alone: ``(topk_ids int32 [M, 16], topk_weights bf16 [M, 16])`` of
    FP32 router logits ``[M, 896]``, bit-identical to the prologue's on the same logits (its routing CTAs, launched
    without the quantization CTAs)."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"{why}, {r} rows", spec=dict(rows=r, scale=1.0, runner_weights=runner_weights))
        for runner_weights, why in (
            (True, "attention TP (the runner's weights): the MoE front's routing"),
            (False, "attention DP"),
        )
        for r in (1, 8, 64, 72, 2048)
    )

    def __call__(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        runner_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _k31_moe_prologue.k31_moe_route(
            router_logits, e_score_correction_bias, routed_scaling_factor, runner_weights
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
        runner_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """:class:`K31MoePrologue`'s reference routing (a zero input stands in for the latent it does not read)."""
        x = router_logits.new_zeros((router_logits.shape[0], HIDDEN), dtype=torch.bfloat16)
        ids, weights, _, _ = k31_moe_prologue.reference(
            router_logits, e_score_correction_bias, routed_scaling_factor, x, runner_weights
        )
        return ids, weights


class K31MoeQuant(OpWrapper):
    """The quantization half of :class:`K31MoePrologue` alone: ``(x_fp8 [M, 3584], x_sf uint8 [M, 112])`` of the
    BF16 latent, bit-identical to the prologue's (its quantization CTAs alone)."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=why, spec=dict(rows=r))
        for r, why in (
            (1, "one row"),
            (72, "TEP16 decode rows"),
            (4096, "a prefill chunk"),
        )
    )

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _k31_moe_prologue.k31_moe_quant(x)

    def reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``mxfp8_quantize`` with linear UE8M0 scales, as the prologue's reference."""
        xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=INPUT_HIDDEN_ALIGNMENT)
        return xq, xsf.view(x.shape[0], -1)


k31_moe_route = K31MoeRoute()
k31_moe_quant = K31MoeQuant()


def _routing_served(
    router_logits, bias, routed_scaling_factor: float, runner_weights: bool
) -> bool:
    """Whether :data:`k31_moe_prologue` routes these logits: FP32 contiguous ``[M, 896]`` with ``M`` in
    1..65536, an FP32 contiguous bias, and unit scale where it reproduces the runner's weights."""
    m = router_logits.shape[0] if router_logits.dim() == 2 else -1
    return (
        router_logits.dim() == 2
        and router_logits.shape[1] == NUM_EXPERTS
        and router_logits.dtype == torch.float32
        and router_logits.is_contiguous()
        and bias.dtype == torch.float32
        and bias.is_contiguous()
        and 1 <= m <= MAX_ROWS
        and not (runner_weights and routed_scaling_factor != 1.0)
    )


def _input_served(x, rows: int) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and x.shape == (rows, HIDDEN)
        and x.is_contiguous()
    )


def route_quant(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    x: torch.Tensor,
    runner_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """``(topk_ids, topk_weights, x_fp8, x_sf)`` of one routed-expert call from :data:`k31_moe_prologue`, or None
    where the op does not serve it (the caller then routes with :func:`separated_routing` and quantizes with
    :func:`quantize_input`).

    ``runner_weights``: attention TP, whose routed experts route inside the TRTLLM-Gen runner by default, so the
    separated routing produces the runner's weights; attention DP routes separately anyway (the fused op / the
    routing tail), and takes those weights. K3.1 routes with one expert group, top-16 of 896."""
    m = router_logits.shape[0] if router_logits.dim() == 2 else -1
    if not _routing_served(
        router_logits, bias, routed_scaling_factor, runner_weights
    ) or not _input_served(x, m):
        return None
    return k31_moe_prologue(router_logits, bias, routed_scaling_factor, x, runner_weights)


def route(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    runner_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The routing half of :func:`route_quant` alone, ``(topk_ids, topk_weights)`` (its ids and weights on these
    logits and an input it serves), or None where it declines the logits."""
    if not _routing_served(router_logits, bias, routed_scaling_factor, runner_weights):
        return None
    return k31_moe_route(router_logits, bias, routed_scaling_factor, runner_weights)


def quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The quantization half of :func:`route_quant` alone, ``(x_fp8, x_sf)`` (those of :func:`route_quant` on the
    same ``x``), or None where it declines ``x`` (dtype, shape, layout)."""
    m = x.shape[0] if isinstance(x, torch.Tensor) and x.dim() == 2 else -1
    if not 1 <= m <= MAX_ROWS or not _input_served(x, m):
        return None
    return k31_moe_quant(x)


def runner_routing_bias(
    bias: torch.Tensor, routed_scaling_factor: float, runner_weights: bool
) -> torch.Tensor | None:
    """The correction bias where :func:`route_quant` routes with the runner's weights at unit scale (what
    ``k31_moe_prologue(logits, bias, 1.0, x, runner_weights=True)`` reproduces, so a caller may fold it into
    another kernel), else None."""
    if not runner_weights or routed_scaling_factor != 1.0:
        return None
    if bias.dtype != torch.float32 or not bias.is_contiguous():
        return None
    return bias


def separated_routing(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    route_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(topk_ids int32, topk_weights)`` of the DeepSeek-V3 one-group routing the routed-expert scheduler runs
    where no fused route + quantize serves the call: with ``route_tail`` (attention DP) the routing-tail kernel at
    65..2048 rows (the same experts in the same order, noaux_tc's weights already rounded to BF16), else
    ``noaux_tc_op`` (FP32 weights). The caller casts the weights to BF16."""
    m = router_logits.shape[0]
    if (
        route_tail
        and ROUTE_TAIL_MAX_ROWS >= m > FUSED_C_MAX_ROWS
        and router_logits.dtype == torch.float32
        and router_logits.dim() == 2
        and router_logits.is_contiguous()
    ):
        return k31_route_topk(router_logits, bias, float(routed_scaling_factor))
    weights, ids = noaux_tc_op(router_logits, bias, 1, 1, TOP_K, routed_scaling_factor)
    return ids.to(torch.int32), weights.to(torch.float32)


def quantize_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_fp8, x_sf)`` of the routed-expert input as the TRTLLM-Gen W4A8 backend quantizes it: MXFP8 with linear
    UE8M0 scales, the hidden width aligned to 512 (K3.1's 3584 needs no padding), scales ``[M, 112]``."""
    x_fp8, x_sf = mxfp8_quantize(x, False, INPUT_HIDDEN_ALIGNMENT)
    return x_fp8, x_sf.view(x_fp8.shape[0], -1)
