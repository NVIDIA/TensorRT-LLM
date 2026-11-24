# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A low-M BF16 projection that pushes its output, a TP partial, into the Lamport slots of the
all-reduce after it (``kernels/spec/rubin_draft_gemm_push``: FlashInfer's split-K kernel with a
push epilogue). ``rubin_lamport_finish`` completes the call."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

from ..comm._rubin_comm_common import LAMPORT_NOTE, check_rows, sanitize_bf16

if IS_CUTLASS_DSL_AVAILABLE:
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

    from ...kernels.spec.rubin_draft_gemm_push import check_act_tactic, validate_push_tactic
    from ...kernels.spec.rubin_draft_gemm_push import run_gemm as _run_store
    from ...kernels.spec.rubin_draft_gemm_push import run_gemm_push as _launch

if TYPE_CHECKING:
    from ...kernels.comm.rubin_lamport_ar import LamportWorkspace


def _cell(
    why: str, rows: int, n: int, k: int, one_shot: bool, tactic: tuple, act: bool = False
) -> Cell:
    spec = dict(world=4, rows=rows, n=n, k=k, one_shot=one_shot, tactic=tactic)
    if act:
        spec["act"] = True
    return Cell(why=why, spec=spec)


class RubinUllBf16GemmPush(OpWrapper):
    """Push ``x [M, K] @ weight[N, K]^T`` (BF16, FP32 accumulation rounded once to BF16) into the
    slots of the next call on ``ws`` (a ``LamportWorkspace`` of the kernel module, ``N`` its
    hidden width), on the transport ``one_shot`` names, as ``rubin_lamport_push`` pushes a
    tensor. ``out`` (contiguous BF16 ``[M, N]``) sets the launch and is not written. ``tactic``
    is a low-M plan's ``("splitk", mma_m, mma_n, split_k, ab_stages)``. Returns None; the finish
    reads the slots. With ``act``, ``x`` is a gate / up output ``[M, 2K]`` (``M`` up to
    ``ACT_MAX_M``; each token CTA produces its own 8-row activation tile) and the product is of
    ``silu(x[:, :K]) * x[:, K:]`` (FlashInfer's ``silu_and_mul`` arithmetic, computed in the
    activation load): bit for bit ``silu_and_mul`` then this op.
    """

    # The served shapes are the helix16 drafter's TP16 partials and helix8's DSA o_proj (12 heads x
    # 128 per rank, dsa_oproj_push); the kernel is collective, so its cells run in the four-rank
    # test (rubin_multirank/test_rubin_bf16_gemm_push.py).
    CELLS: tuple[Cell, ...] = (
        _cell(
            "DFlash2 drafter o_proj (6 heads), one request's block, two-shot",
            8,
            7168,
            768,
            False,
            ("splitk", 128, 8, 1, 6),
        ),
        _cell(
            "DFlash2 drafter down / 16, one request's block, two-shot",
            8,
            7168,
            896,
            False,
            ("splitk", 128, 8, 1, 6),
        ),
        _cell(
            "drafter o_proj, one row, one-shot",
            1,
            7168,
            768,
            True,
            ("splitk", 128, 8, 1, 6),
        ),
        _cell(
            "DFlash2 drafter down / 16 with silu_and_mul in the activation load (act), two-shot",
            8,
            7168,
            896,
            False,
            ("splitk", 128, 8, 1, 6),
            act=True,
        ),
        _cell(
            "down / 16 with act at the Rubin row's 64-row tiles (_DRAFT_PUSH_TACTIC_RUBIN), two-shot",
            8,
            7168,
            896,
            False,
            ("splitk", 64, 8, 1, 12),
            act=True,
        ),
        _cell(
            "DFlash2 drafter down / 16 with act, two requests' blocks (16 rows: two token CTAs), "
            "two-shot",
            16,
            7168,
            896,
            False,
            ("splitk", 128, 8, 1, 6),
            act=True,
        ),
        _cell(
            "act, 13 rows (the second token CTA's rows past them zero), two-shot",
            13,
            7168,
            896,
            False,
            ("splitk", 128, 8, 1, 6),
            act=True,
        ),
        _cell(
            "act at 64-row tiles, 16 rows (224 CTAs), two-shot",
            16,
            7168,
            896,
            False,
            ("splitk", 64, 8, 1, 12),
            act=True,
        ),
        _cell(
            "act, three rows (rows past them zero), one-shot",
            3,
            7168,
            896,
            True,
            ("splitk", 128, 8, 1, 6),
            act=True,
        ),
        _cell(
            "a K split of 2 (the pushed partial is still one rounding)",
            8,
            7168,
            768,
            False,
            ("splitk", 128, 8, 2, 6),
        ),
        # helix8's DSA o_proj (dsa_oproj_push) at its tactics: 64-row weight tiles up to 32 rows;
        # past them 128-row weight tiles in N tiles of 32 (two, three) or one 64-token tile.
        _cell(
            "helix8 DSA o_proj, b2 k5 (12 rows), two-shot",
            12,
            7168,
            1536,
            False,
            ("splitk", 64, 16, 1, 8),
        ),
        _cell(
            "helix8 DSA o_proj, b4 k5 (24 rows), two-shot",
            24,
            7168,
            1536,
            False,
            ("splitk", 64, 32, 1, 8),
        ),
        _cell(
            "helix8 DSA o_proj, b6 k5 (36 rows): 128-row weight tiles, two N tiles of 32, two-shot",
            36,
            7168,
            1536,
            False,
            ("splitk", 128, 32, 1, 5),
        ),
        _cell(
            "helix8 DSA o_proj, b9 k5 (54 rows): one 64-token tile, two-shot",
            54,
            7168,
            1536,
            False,
            ("splitk", 64, 64, 1, 6),
        ),
        _cell(
            "helix8 DSA o_proj, b12 k5 (72 rows): 128-row weight tiles, three N tiles of 32, two-shot",
            72,
            7168,
            1536,
            False,
            ("splitk", 128, 32, 1, 5),
        ),
    )

    note = (
        LAMPORT_NOTE
        + """
    The pushed values are this kernel's plain output (``run_gemm`` stores the same tile): with
    ``split_k`` 1 the product is bitwise cuBLAS at the drafter's shapes (GB300), so the finish of
    the call reproduces ``plain GEMM -> C++ all-reduce`` bit for bit. The kernel compiles once
    per (tactic, PDL, world, N, transport): make it before CUDA-graph capture.
    """
    )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        ws: LamportWorkspace,
        one_shot: bool,
        tactic: tuple,
        use_pdl: bool,
        act: bool = False,
    ) -> None:
        target = SimpleNamespace(workspace=ws, one_shot=one_shot, pushed=False)
        _launch(x, weight, out, target, SplitKTactic(*tactic[1:]), pdl=use_pdl, act=act)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        ws: LamportWorkspace,
        one_shot: bool,
        tactic: tuple,
        use_pdl: bool,
        act: bool = False,
    ) -> torch.Tensor:
        """The payload of this rank's slots: the FP64 product rounded once to BF16, ``-0`` sent
        as ``+0`` (with ``act``, of the FP32 ``silu(gate) * up`` rounded to BF16). A finish of the
        call is ``lamport_allreduce_reference`` of every rank's payload."""
        if act:
            gate, up = x.float().chunk(2, dim=-1)
            x = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        return sanitize_bf16((x.double() @ weight.double().t()).to(torch.bfloat16))

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        ws: LamportWorkspace,
        one_shot: bool,
        tactic: tuple,
        use_pdl: bool,
        act: bool = False,
    ) -> None:
        for t, name in ((x, "x"), (weight, "weight"), (out, "out")):
            if t.dtype != torch.bfloat16 or t.dim() != 2 or not t.is_contiguous():
                raise ValueError(f"bf16_gemm_push: {name} must be contiguous 2-D BF16")
        m, k = x.shape
        if act:
            if k % 2 or x.data_ptr() % 16:
                raise ValueError(f"bf16_gemm_push: act takes a 16-byte aligned [M, 2K], {x.shape}")
            k //= 2
            check_act_tactic(SplitKTactic(*tactic[1:]), k, m)
        n = weight.shape[0]
        if weight.shape[1] != k or tuple(out.shape) != (m, n):
            raise ValueError(f"bf16_gemm_push: x {tuple(x.shape)}, weight {tuple(weight.shape)}")
        if tactic[0] != "splitk":
            raise ValueError(f"bf16_gemm_push: the push kernel is the split-K one, not {tactic}")
        validate_push_tactic(SplitKTactic(*tactic[1:]), m, n, k)
        check_rows("out", ws, out)


bf16_gemm_push = RubinUllBf16GemmPush()


def bf16_gemm_push_values(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic: tuple
) -> torch.Tensor:
    """``out = x [M, K] @ weight[N, K]^T``: the values :data:`bf16_gemm_push` pushes for these
    operands and ``tactic``, stored by the push kernel's store-only variant (a runtime check, not an
    entry: a target compares them with the GEMM a push replaces). Raises ``ValueError`` for a
    tactic or shape the push kernel cannot run. Returns ``out``."""
    if tactic[0] != "splitk":
        raise ValueError(f"bf16_gemm_push: the push kernel is the split-K one, not {tactic}")
    for t, name in ((x, "x"), (weight, "weight"), (out, "out")):
        if t.dtype != torch.bfloat16 or t.dim() != 2 or not t.is_contiguous():
            raise ValueError(f"bf16_gemm_push: {name} must be contiguous 2-D BF16")
    return _run_store(x, weight, out, SplitKTactic(*tactic[1:]))


__all__ = ["bf16_gemm_push", "bf16_gemm_push_values"]
