# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 BF16 GEMMs on SM107 at a pinned cuBLASLt algorithm (helix8's generation GEMMs)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.gemm.rubin_lt_pin import LtPin, LtUnavailable, lt_gemm, lt_problem

__all__ = ["LtPin", "LtUnavailable", "rubin_lt_gemm"]


def _cell(why: str, rows: int, n: int, k: int, pin: LtPin, today: str, exact: bool = True) -> Cell:
    """``today``: the call the pin replaces, ``mm`` (``cublas_mm``), ``linear`` (``rubin_linear``),
    ``router`` (``dsv3_router_gemm_op``, FP32 logits) or ``uv`` (``bmm_out`` over 96 heads).
    ``exact``: the pin writes today's words (B-class); else another summation order (A-class)."""
    return Cell(why=why, spec=dict(rows=rows, n=n, k=k, pin=pin, today=today, exact=exact))


def _p(tile: int, stages: int, cluster: int, custom: int, splitk: int = 1, reduction: int = 0):
    """An NVJET algorithm (cuBLASLt algorithm 66), no swizzle, inner shape 0."""
    return LtPin(66, tile, stages, cluster, splitk, reduction, 0, custom, 0)


class RubinLtGemm(OpWrapper):
    """``out = a @ b`` in FP32 with one rounding to ``out``'s dtype, through cuBLASLt with the
    algorithm ``pin`` names (the first result of cuBLASLt's heuristic for the problem that matches
    every field the pin sets). ``a`` BF16 ``[M, K]`` or ``[B, M, K]``; ``b`` BF16 ``[K, N]`` or
    ``[B, K, N]``, dense or a transposed view (a stored ``[N, K]`` weight's ``.t()``); ``out`` BF16
    or FP32 ``[M, N]`` or ``[B, M, N]`` with unit column stride, written and returned. One launch
    on the current stream (two with split-K: the GEMM and cuBLAS's reduce).

    Where the pinned algorithm accumulates as the one cuBLASLt's heuristic ranks first does, the
    words are those of ``cublas_mm`` / ``F.linear`` / ``bmm_out`` on the same operands: an
    algorithm's tile, cluster and stages decide which CTA computes an output element and when, its
    K chain and split-K partition how. Which is the case is measured per problem; a caller checks
    it on the device before it relies on it. An algorithm with another K chain or split (an A-class
    pin) sums the same FP32 products in another order: within the reference band, not on the bits.
    Raises ``LtUnavailable`` where the heuristic does not offer the pinned algorithm, and under
    capture for a layout without a plan.
    """

    ARCHS = frozenset({Arch.SM_107})

    # helix8's pinned generation GEMMs (6 rows a request at k5: b2..b4 12-24 rows, b9..b12 54-72,
    # b21 126, b24 144), each with the call it replaces (``today``): a stored weight's ``.t()`` times
    # BF16 rows, or W_UV over 96 heads. The B-class cells write today's words; the A-class cells
    # (``exact`` False) another FP32 summation order of the same products.
    CELLS: tuple[Cell, ...] = (
        _cell("MoE latent_up, 54 rows: 3584 -> 7168", 54, 7168, 3584, _p(15, 35, 11, 1), "mm"),
        _cell("MoE latent_up, 60 rows", 60, 7168, 3584, _p(15, 35, 11, 1), "mm"),
        _cell("MoE latent_up, 144 rows", 144, 7168, 3584, _p(326, 35, 11, 3), "mm"),
        _cell(
            "MoE latent_down, 60 rows: 7168 -> 3584, split-K 4",
            60,
            3584,
            7168,
            _p(18, 35, 6, 0, 4, 4),
            "mm",
        ),
        _cell("MoE latent_down, 144 rows", 144, 3584, 7168, _p(315, 35, 4, 1), "mm"),
        _cell(
            "MoE router, 144 rows: 7168 -> 896 FP32 logits, split-K 8",
            144,
            896,
            7168,
            _p(321, 35, 3, 3, 8, 2),
            "router",
        ),
        _cell(
            "MLA [q_a | kv_a], 18 rows: 7168 -> 2112, split-K 8",
            18,
            2112,
            7168,
            _p(26, 35, 4, 1, 8, 1),
            "mm",
        ),
        _cell("MLA [q_a | kv_a], 24 rows, split-K 8", 24, 2112, 7168, _p(26, 35, 4, 1, 8, 2), "mm"),
        _cell("MLA [q_a | kv_a], 144 rows", 144, 2112, 7168, _p(315, 35, 3, 1), "mm"),
        _cell("MLA W_UV, 54 rows: 96 heads x 512 -> 128", 54, 128, 512, _p(315, 35, 3, 1), "uv"),
        _cell("MLA o_proj, 66 rows: 1536 -> 7168", 66, 7168, 1536, _p(318, 35, 11, 1), "linear"),
        _cell("MLA o_proj, 72 rows", 72, 7168, 1536, _p(318, 35, 11, 1), "linear"),
        _cell("MLA o_proj, 144 rows", 144, 7168, 1536, _p(26, 35, 3, 3), "linear"),
        _cell("DSA indexer wq_b, 144 rows: 1536 -> 4096", 144, 4096, 1536, _p(15, 35, 3, 1), "mm"),
        _cell("dense gate_up, 144 rows: 7168 -> 8448", 144, 8448, 7168, _p(326, 35, 11, 3), "mm"),
        _cell("dense down, 54 rows: 4224 -> 7168", 54, 7168, 4224, _p(15, 35, 11, 1), "mm"),
        _cell("dense down, 60 rows", 60, 7168, 4224, _p(15, 35, 11, 1), "mm"),
        _cell("dense down, 144 rows", 144, 7168, 4224, _p(326, 35, 11, 3), "mm"),
        _cell("drafter q_b, 18 rows: 1536 -> 18432", 18, 18432, 1536, _p(13, 35, 11, 1), "linear"),
        _cell("drafter q_b, 24 rows", 24, 18432, 1536, _p(13, 35, 11, 1), "linear"),
        _cell("drafter q_b, 60 rows", 60, 18432, 1536, _p(18, 35, 11, 1), "linear"),
        _cell("drafter q_b, 126 rows", 126, 18432, 1536, _p(21, 35, 7, 2), "linear"),
        _cell("drafter q_b, 144 rows", 144, 18432, 1536, _p(165, 35, 5, 2), "linear"),
        _cell(
            "drafter o_proj, 66 rows: 1536 -> 7168", 66, 7168, 1536, _p(318, 35, 11, 1), "linear"
        ),
        _cell("drafter o_proj, 72 rows", 72, 7168, 1536, _p(318, 35, 11, 1), "linear"),
        _cell("drafter o_proj, 144 rows", 144, 7168, 1536, _p(26, 35, 3, 3), "linear"),
        _cell(
            "drafter gate_up, 144 rows: 7168 -> 3584", 144, 3584, 7168, _p(315, 35, 3, 3), "linear"
        ),
        _cell("drafter down, 66 rows: 1792 -> 7168", 66, 7168, 1792, _p(318, 35, 11, 1), "linear"),
        _cell("drafter down, 72 rows", 72, 7168, 1792, _p(318, 35, 11, 1), "linear"),
        _cell("drafter down, 144 rows", 144, 7168, 1792, _p(318, 35, 3, 1), "linear"),
        _cell(
            "drafter attn conv, 144 rows: 7168 -> 1792",
            144,
            1792,
            7168,
            _p(315, 35, 3, 3),
            "linear",
        ),
        _cell(
            "drafter MLP conv, 144 rows: 7168 -> 1792", 144, 1792, 7168, _p(315, 35, 3, 1), "linear"
        ),
        # A-class: split-K 1 or another split than today's.
        _cell("A: MoE latent_up, 12 rows", 12, 7168, 3584, _p(312, 35, 3, 1), "mm", False),
        _cell(
            "A: MoE latent_down, 12 rows, split-K 8",
            12,
            3584,
            7168,
            _p(449, 35, 3, 1, 8, 2),
            "mm",
            False,
        ),
        _cell("A: MoE latent_down, 54 rows", 54, 3584, 7168, _p(12, 35, 4, 0), "mm", False),
        _cell("A: MoE latent_down, 60 rows", 60, 3584, 7168, _p(12, 35, 4, 0), "mm", False),
        _cell("A: MoE latent_down, 66 rows", 66, 3584, 7168, _p(15, 35, 6, 3), "mm", False),
        _cell("A: MoE latent_down, 72 rows", 72, 3584, 7168, _p(15, 35, 6, 0), "mm", False),
        _cell(
            "A: MLA [q_a | kv_a], 12 rows, split-K 8",
            12,
            2112,
            7168,
            _p(394, 35, 4, 1, 8, 2),
            "mm",
            False,
        ),
        _cell("A: DSA indexer wq_b, 12 rows", 12, 4096, 1536, _p(314, 35, 4, 1), "mm", False),
        _cell("A: dense gate_up, 18 rows", 18, 8448, 7168, _p(315, 35, 3, 3), "mm", False),
        _cell("A: dense gate_up, 24 rows", 24, 8448, 7168, _p(315, 35, 3, 3), "mm", False),
        _cell(
            "A: drafter gate_up, 12 rows, split-K 8",
            12,
            3584,
            7168,
            _p(449, 35, 3, 1, 8, 2),
            "linear",
            False,
        ),
        _cell("A: drafter gate_up, 54 rows", 54, 3584, 7168, _p(12, 35, 4, 0), "linear", False),
        _cell("A: drafter gate_up, 60 rows", 60, 3584, 7168, _p(12, 35, 4, 0), "linear", False),
        _cell("A: drafter gate_up, 66 rows", 66, 3584, 7168, _p(15, 35, 6, 0), "linear", False),
        _cell("A: drafter gate_up, 72 rows", 72, 3584, 7168, _p(15, 35, 6, 0), "linear", False),
    )

    note = """
    The plan of a layout and pin (descriptors, algorithm, workspace) is built on its first call,
    outside capture, and kept for the process; calls of a plan with a workspace (split-K) must be
    ordered on one stream. The operands are read through their strides: a unit stride in one
    dimension of each.
    """

    def __call__(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, pin: LtPin
    ) -> torch.Tensor:
        lt_gemm(a, b, out, pin)
        return out

    def reference(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, pin: LtPin
    ) -> torch.Tensor:
        """FP32 products of the BF16 operands (TF32 off), rounded once to ``out``'s dtype; torch
        accumulates in its own order, so within the dtype's accumulation band, not on the bits."""
        with true_fp32_matmul():
            ref = torch.matmul(a.float(), b.float())
        return ref.to(out.dtype)

    def is_valid(
        self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, pin: LtPin, **kwargs
    ) -> None:
        """cuBLASLt reads and writes through the strides it is given: an operand without a unit
        stride, a mistyped one or a misshaped product is read or written wrong."""
        assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "a and b must be BF16"
        assert out.dtype in (torch.bfloat16, torch.float32), "out must be BF16 or FP32"
        assert b.device == a.device and out.device == a.device, "one device"
        assert isinstance(pin, LtPin) and pin.algo is not None, "a pin names its algorithm"
        assert all(v is None or isinstance(v, int) for v in pin), "pin fields are ints or None"
        assert a.shape[-2] >= 1, "no rows"
        try:
            lt_problem(a, b, out)
        except ValueError as exc:
            raise AssertionError(str(exc)) from None


rubin_lt_gemm = RubinLtGemm()
