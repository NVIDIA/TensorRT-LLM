# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bias-free bf16 projection of a decode-sized step, as the engine's CSA2 module projects its bf16 weights
(`attention/backends/sparse/csa2/compressor.py::bf16_projection`): the low-M warp-MMA kernel with bf16 weights
(`low_m_gemm.bf16_lowm_linear`, fp32 accumulation rounded once to bf16) up to 64 rows of a decode-graph step
(`flashinfer_mxfp8_decode_graph_capture`, as the engine gates it), else `F.linear` as the engine's `linear(x)`. Used
by the dep and tep8 targets for the indexer's head weights `[32, 5120]` and the compressor's value | gate
`[1024, 5120]` (ratio 2) or value `[512, 5120]` (ratio 1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

LOWM_MAX_ROWS = 64  # the engine's `compressor.LOWM_MAX_ROWS`


def _lowm_fits(m: int, n: int, k: int) -> bool:
    """Only in a decode-graph step (`flashinfer_mxfp8_decode_graph_capture_active`), as the engine's `bf16_projection`
    (context chunks and eager steps run cuBLAS)."""
    from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture_active
    from tensorrt_llm._torch.modules.low_m_gemm import lowm_fp8_mma_fits

    return (
        0 < m <= LOWM_MAX_ROWS
        and flashinfer_mxfp8_decode_graph_capture_active()
        and lowm_fp8_mma_fits(m, n, k)
    )


class Csa2Bf16LowmProj(OpWrapper):
    """`x [M, K]` bf16 @ `weight [N, K]` bf16 -> `[M, N]` bf16 (fp32 accumulation, one rounding); with `scale`, the
    engine's `(y * scale)` into fp32 in a second launch (the indexer's head weights, bit for bit as the engine's
    `_project_index_weights`). `out_dtype=torch.bfloat16`: that launch writes bf16, as the engine's candidate
    consumers (`_index_weight_dtype`: their sparse logits kernel reads bf16 head weights; the same values as the fp32
    form cast to bf16, since the product is rounded to bf16 first)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="decode: indexer head weights (32 heads), scaled into fp32",
            spec=dict(m=6, n=32, k=5120, scale=True, graph=True),
        ),
        Cell(
            why="decode: ratio-2 compressor value | gate (2 x 512)",
            spec=dict(m=6, n=1024, k=5120, scale=False, graph=True),
        ),
        Cell(
            why="dep32 decode, 24 rows: indexer head weights",
            spec=dict(m=24, n=32, k=5120, scale=True, graph=True),
        ),
        Cell(
            why="dep32 decode, 24 rows: ratio-1 compressor value",
            spec=dict(m=24, n=512, k=5120, scale=False, graph=True),
        ),
        Cell(
            why="64 rows: the low-M kernel's edge",
            spec=dict(m=64, n=1024, k=5120, scale=False, graph=True),
        ),
        Cell(
            why="dep8 c7000 decode, 160 rows: cuBLAS", spec=dict(m=160, n=1024, k=5120, scale=False)
        ),
        Cell(
            why="a five-token context chunk (eager): cuBLAS, as stock's eager bf16_projection",
            spec=dict(m=5, n=32, k=5120, scale=True),
        ),
        Cell(
            why="tep8 decode: a candidate consumer's head weights, scaled into bf16",
            spec=dict(m=6, n=32, k=5120, scale=True, out_dtype=torch.bfloat16, graph=True),
        ),
    )
    note = "The two paths differ only in summation order (fp32 accumulation, bf16 out): gate 1 bf16 ulp."

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: float | None = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        m, k = x.shape
        if _lowm_fits(m, weight.shape[0], k) and x.is_contiguous() and x.data_ptr() % 16 == 0:
            from tensorrt_llm._torch.modules.low_m_gemm import bf16_lowm_linear

            y = bf16_lowm_linear(x, weight, torch.bfloat16)
        else:
            y = F.linear(
                x, weight
            )  # the engine's `linear(x)` (`F.linear`): another cuBLAS call differs in tiles of large M
        if scale is None:
            return y
        return torch.mul(y, scale, out=torch.empty(y.shape, dtype=out_dtype, device=y.device))

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: float | None = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """float64 product of the bf16 operands rounded to bf16 (then scaled in fp32, stored as `out_dtype`)."""
        y = (x.double() @ weight.double().t()).to(torch.bfloat16)
        return y if scale is None else (y.float() * scale).to(out_dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: float | None = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> None:
        assert (
            x.dtype == weight.dtype == torch.bfloat16
            and x.dim() == weight.dim() == 2
            and x.shape[1] == weight.shape[1]
        )
        assert weight.is_contiguous() and weight.data_ptr() % 16 == 0 and x.stride(-1) == 1
        assert out_dtype in (torch.float32, torch.bfloat16), (
            "the scaled form is fp32, or bf16 for a candidate consumer"
        )


csa2_bf16_lowm_proj = Csa2Bf16LowmProj()
