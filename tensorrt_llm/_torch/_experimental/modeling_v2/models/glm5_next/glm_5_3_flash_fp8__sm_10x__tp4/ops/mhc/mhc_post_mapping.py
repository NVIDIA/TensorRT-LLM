# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mHC post-mapping: a sublayer's output mixed back into the residual streams.

``torch.ops.trtllm.mhc_post_mapping`` (``mhcPostMappingKernel``: one CTA of 256 threads per row, eight
BF16 values per thread and step), the kernel ``mHC.post_mapping`` launches. GLM-5.3-Flash wraps every
sublayer in a hyper-connection of four residual streams. The target needs this op once per forward
step, after the last sublayer, where no next pre-mapping follows; ``mhc_fused_hc`` covers every other
boundary.

For ``M`` rows of ``mult = 4`` streams of ``hidden`` values, with ``x`` the sublayer output, ``post``
its post coefficients and ``comb`` its Sinkhorn-normalized stream combination::

    out[t, j, :] = post[t, j] * x[t, :] + sum_k comb[t, k, j] * residual[t, k, :]

that is ``post * x + comb^T @ residual``: row ``k`` of ``comb`` is a source stream and column ``j`` a
destination stream. The kernel accumulates in FP32 in that order (the ``post`` product, then
``k = 0..3`` as fused multiply-adds) and rounds each output to BF16 once.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The stream count the kernel is compiled for, and GLM-5.3-Flash's hidden size.
MULT = 4
HIDDEN = 4096


class MhcPostMapping(OpWrapper):
    """``post * x + comb^T @ residual`` per row, rounded to BF16: the streams after a sublayer.

    Args:
        x: ``[M, hidden]`` BF16, contiguous: the sublayer's output, after its TP reduction.
        residual: ``[M, 4, hidden]`` BF16, contiguous: the streams the sublayer's pre-mapping read.
        post: ``[M, 4]`` or ``[M, 4, 1]`` FP32, contiguous: the sublayer's post coefficients,
            ``post_mult * sigmoid(...)`` from its pre-mapping (in ``[0, 2]`` for GLM-5.3-Flash).
        comb: ``[M, 4, 4]`` FP32, contiguous: the sublayer's Sinkhorn-normalized combination,
            indexed ``[source stream, destination stream]``.

    Returns:
        ``[M, 4, hidden]`` BF16, a new contiguous tensor of ``residual``'s shape.
    """

    ARCHS = frozenset({Arch.SM_100})

    _GLM = dict(mult=MULT, hidden=HIDDEN)
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode of one request without MTP: one CTA", spec=dict(rows=1, **_GLM)),
        Cell(why="MTP-3 verify of one request (4 rows)", spec=dict(rows=4, **_GLM)),
        Cell(why="MTP-3 verify of two requests (8 rows)", spec=dict(rows=8, **_GLM)),
        Cell(why="MTP-3 verify of 8 requests (32 rows)", spec=dict(rows=32, **_GLM)),
        Cell(
            why="MTP-3 verify of 64 requests (256 rows): more CTAs than B200 has SMs",
            spec=dict(rows=256, **_GLM),
        ),
        Cell(
            why="a quarter of a 13,312-token prompt: one rank's rows under sequence parallelism",
            spec=dict(rows=3328, **_GLM),
        ),
    )

    note = """
    The kernel is launched without programmatic dependent launch and without a zero-row guard: a
    launch of no rows leaves an invalid-configuration error pending for the next CUDA call, so
    `is_valid` refuses it. The output is allocated per call, from the graph's private pool under CUDA
    graph capture. `x` and `post` / `comb` are read as given: the wrapper neither casts nor copies
    them.
    """

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        rows, hidden = x.shape
        out = torch.empty_like(residual, memory_format=torch.contiguous_format)
        torch.ops.trtllm.mhc_post_mapping(residual, x, post, comb, out, rows, hidden)
        return out

    def reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """FP32 in the kernel's order (the ``post`` product, then sources ``k = 0..3``), rounded once.

        Element-wise products and sums only, so no matmul precision setting reaches it.
        """
        rows, hidden = x.shape
        streams = residual.float().reshape(rows, MULT, hidden)
        coeff = comb.float().reshape(rows, MULT, MULT)
        out = post.float().reshape(rows, MULT, 1) * x.float().reshape(rows, 1, hidden)
        for k in range(MULT):
            out += coeff[:, k, :, None] * streams[:, k, None, :]
        return out.to(torch.bfloat16).reshape(residual.shape)

    def is_valid(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> None:
        assert x.dim() == 2, f"x is [M, hidden]; got {tuple(x.shape)}"
        rows, hidden = x.shape
        assert rows >= 1, "a launch of no rows leaves an error pending for the next CUDA call"
        assert hidden % 8 == 0, "each thread moves 8 BF16 values per step; the kernel has no tail"
        assert residual.numel() == rows * MULT * hidden and residual.shape[-1] == hidden, (
            f"residual must hold [{rows}, {MULT}, {hidden}]; got {tuple(residual.shape)}"
        )
        assert post.numel() == rows * MULT and comb.numel() == rows * MULT * MULT, (
            f"post holds [{rows}, {MULT}] and comb [{rows}, {MULT}, {MULT}]"
        )
        for name, t in (("x", x), ("residual", residual), ("post", post), ("comb", comb)):
            assert t.is_contiguous(), f"{name} is read with dense row-major strides"


mhc_post_mapping = MhcPostMapping()
