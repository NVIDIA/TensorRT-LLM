# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MR8's split-K projection run through its prepared runner
(``kernels/vendor/vllm_mr8/.../cutedsl/gemm/projection.py``'s ``PreparedProjection``): the BF16 split-K GEMM,
or the W8A16 one on a ``common.w8.W8Weight``."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _cell(why: str, rows: int, w8: bool) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, w8=w8))


def projection_weight(projection) -> torch.Tensor:
    """The FP64 [N, K] weight ``projection`` multiplies by: its BF16 matrix, or its W8A16 codes x
    2^(UE8M0 - 127) per 1 x 32 block."""
    weight = projection.weight
    if projection.w8 is None:
        return weight.double()
    codes = weight.codes().view(torch.float8_e4m3fn).double()
    n, k = codes.shape
    scale = torch.exp2(weight.scales().double() - 127.0)
    return (codes.view(n, k // 32, 32) * scale.unsqueeze(-1)).view(n, k)


class PreparedProjectionGemm(OpWrapper):
    """``out = x @ W^T`` with ``W`` the ``projection``'s weight (BF16 [N, K], or the W8A16 buffer
    whose dequantization is exact in BF16), FP32 accumulation over the split K, rounded once to
    ``projection.output_dtype``. ``x`` BF16 [M, K] with ``M`` a prepared batch, ``out`` [M, N].
    Returns ``out``."""

    ARCHS = frozenset({Arch.SM_103})

    # The KDA layers' packed in-projection [q, k, v, g (6 heads x 128), f_a (128), beta (6), pad]
    # = 3216 x 7168 at TP16, with MR8's TP16 policies (``providers.projection_policies``) for q8
    # verify rows; the BF16 weight (the target loads no W8A16 KDA in-projection).
    CELLS: tuple[Cell, ...] = (
        _cell("KDA in-projection BF16, b1 verify (8 rows)", 8, False),
        _cell("KDA in-projection BF16, b2 verify (16 rows)", 16, False),
        _cell("KDA in-projection BF16, b4 verify (32 rows)", 32, False),
    )

    note = """
    The runner is prepared per layer before CUDA-graph capture (``FlatKda._prepare_projection``:
    the tactic per batch, the compiled code, the weight's TMA maps); a batch it was not prepared
    for raises ValueError. The W8A16 kernel converts the codes in shared memory and keeps the BF16
    kernel's MMA, K order and reduction: bitwise the BF16 kernel on the dequantized weight.
    """

    def __call__(self, projection, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return projection.run(x, out)

    def reference(self, projection, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The FP64 product rounded once to the runner's output dtype; ``out`` is not read."""
        return (x.double() @ projection_weight(projection).t()).to(projection.output_dtype)

    def is_valid(self, projection, x: torch.Tensor, out: torch.Tensor) -> None:
        """The runner checks the batch and K; it reads ``x`` and writes ``out`` through matrix views
        of their own strides, so a narrower ``out`` or a wrong dtype is written past or misread."""
        n = projection.weight.shape[0]
        if out.shape[0] != x.shape[0] or out.shape[1] < n or out.stride(1) != 1:
            raise ValueError("prepared_projection: out must hold [M, N] rows with unit stride")
        if x.dtype != torch.bfloat16 or out.dtype != projection.output_dtype:
            raise ValueError("prepared_projection: BF16 x, out in the runner's output dtype")


prepared_projection = PreparedProjectionGemm()
