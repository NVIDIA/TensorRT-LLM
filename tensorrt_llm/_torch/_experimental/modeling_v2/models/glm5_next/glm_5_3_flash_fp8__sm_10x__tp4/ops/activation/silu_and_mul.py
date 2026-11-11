# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated SiLU over a fused `[gate | up]` row, with the optional SwiGLU clamp, via the trtllm
Triton silu_and_mul kernel."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: GLM-5.3-Flash's `swiglu_limit`.
GLM_SWIGLU_LIMIT = 10.0


class SiluAndMul(OpWrapper):
    """`silu(gate) * up` over `x = [gate | up]` (`d = x.shape[1] // 2`), new `[M, d]` tensor out.

    The gate is the FIRST half and the up projection the second -- the order the fused
    `gate_up` weight of a dense MLP or shared expert stores (`[gate; up]` rows). The routed
    experts of `fp8_block_scale_moe_runner` store the opposite order (`[up; gate]`), so the two
    halves are not interchangeable between those weights.

    With `swiglu_limit = L > 0`, before the product: `gate = min(gate, L)` (upper side only)
    and `up = clamp(up, -L, L)`. Then `out = gate * sigmoid(alpha * gate) * (up + beta)` in
    fp32, with `alpha = 1`, `beta = 0` unless given, rounded once to `dtype` (default
    `x.dtype`). With `scale` (and an fp8 `dtype`) the result is divided by `scale[0]` and
    saturated to the dtype's finite range before the cast.

    Fusion boundary: the clamp, SiLU, product and optional output quantization; no GEMM.
    """

    ARCHS = frozenset({Arch.SM_100})

    # Only GLM-5.3-Flash calls this entry: its dense MLP and shared expert, with the clamp on.
    # Its own cells are the kernel's tiling edges at those widths; the target's row counts are
    # in the test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one row of the shared expert at TP4: d = 512 is narrower than the kernel's "
            "1024-element block, so every program is masked",
            spec=dict(rows=1, cols=1024, swiglu_limit=GLM_SWIGLU_LIMIT),
        ),
        Cell(
            why="one row of the dense MLP at TP4: d = 3072 spans three 1024-element blocks",
            spec=dict(rows=1, cols=6144, swiglu_limit=GLM_SWIGLU_LIMIT),
        ),
    )

    note = """
    Source: a Triton kernel (`silu_and_mul_kernel` in `_torch/modules/swiglu.py`, launched by
    the `trtllm::silu_and_mul` custom op in `_torch/custom_ops/torch_custom_ops.py`), one
    program per row and 1024-element column block; JIT-compiled per distinct `swiglu_limit`,
    `swiglu_alpha` and `swiglu_beta` (all three are compile-time constants).

    The clamp is compiled in only for `swiglu_limit > 0`: `None`, `0.0` and negative limits
    all mean no clamp. That differs from `fp8_block_scale_moe_runner`, whose clamp is applied
    whenever a limit is given.

    Preconditions the op enforces loudly: `x` 2-D with an even width. Each row is read from
    `x.stride(0)`, so a row-sliced view of a wider buffer is fine; the columns are read as
    dense (see `is_valid`). The output is a new contiguous tensor.

    Certified: bf16 in and out, `swiglu_limit` 10.0 (GLM-5.3-Flash), `alpha`/`beta` unset,
    no `scale`, at GLM's per-rank widths (2 x 512 and 2 x 3072) from 1 to 13,312 rows.

    Not certified: an fp8 output (`scale` + `dtype`), `swiglu_alpha` / `swiglu_beta`, other
    input dtypes, CUDA-graph capture, sm_103.
    """

    def __call__(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        swiglu_limit: Optional[float] = None,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.silu_and_mul(
            x,
            scale=scale,
            dtype=dtype,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )

    def reference(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        swiglu_limit: Optional[float] = None,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
    ) -> torch.Tensor:
        """fp32 throughout, one rounding to the output dtype; no matmul."""
        d = x.shape[-1] // 2
        gate = x[..., :d].float()
        up = x[..., d:].float()
        if swiglu_limit is not None and swiglu_limit > 0.0:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        alpha = 1.0 if swiglu_alpha is None else swiglu_alpha
        beta = 0.0 if swiglu_beta is None else swiglu_beta
        out = gate * torch.sigmoid(alpha * gate) * (up + beta)
        out_dtype = x.dtype if dtype is None else dtype
        if scale is not None:
            bound = torch.finfo(out_dtype).max
            out = (out / scale.reshape(-1)[0].float()).clamp(-bound, bound)
        return out.to(out_dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        swiglu_limit: Optional[float] = None,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
    ) -> None:
        """Two inputs the op takes and answers wrongly, both read from the kernel source.

        A non-2-D `x` or an odd width the op rejects itself.
        """
        # The kernel addresses column j of row i at `x + i * x.stride(0) + j`: the row stride
        # is honoured, the column stride is assumed to be 1.
        assert x.stride(-1) == 1, (
            f"x's last dim must be contiguous (stride 1), got stride {x.stride(-1)}; a strided "
            "view is read as if dense and silently gives the wrong halves"
        )
        # Without `dtype` the output stays in x's dtype, but the kernel still divides by the
        # scale: scaled values land in an unquantized tensor.
        assert scale is None or dtype is not None, (
            "scale without dtype divides the result by scale and stores it unquantized in "
            "x's dtype; pass the fp8 dtype the scale quantizes to"
        )


silu_and_mul = SiluAndMul()
