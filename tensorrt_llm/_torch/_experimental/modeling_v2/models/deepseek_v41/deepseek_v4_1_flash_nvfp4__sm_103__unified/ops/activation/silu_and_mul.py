# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU with an optional clamp (`swiglu_limit`) via the in-tree Triton silu_and_mul kernel.

The op is `torch.ops.trtllm.silu_and_mul`, registered in
`tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` around `silu_and_mul_kernel`
in `tensorrt_llm/_torch/modules/swiglu.py`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: DeepSeek-V4.1's shared-expert clamp.
_LIMIT = 10.0

#: d = 2304 is the shared expert's intermediate width with the expert replicated
#: per rank; d = 576 is the same expert TP-split 4 ways. Rows span decode (1, 8,
#: 32) and prefill chunks (4096, 8192). Neither d is a multiple of the kernel's
#: 1024-wide column block, so the masked tail block runs in every case.
_SHAPES = (
    (1, 2304, "decode"),
    (8, 2304, "decode"),
    (32, 2304, "decode"),
    (4096, 2304, "prefill chunk"),
    (8192, 2304, "prefill chunk"),
    (1, 576, "the 4-way TP-split width"),
    (4096, 576, "the 4-way TP-split width"),
)

#: Row counts the core target produced in its runtime-value dumps (boot, GSM8K
#: and long-context runs), then primes and values either side of 128-row edges.
_RUNTIME_ROWS = (2, 3, 4, 5, 6, 7, 11, 14, 16, 24, 48, 323, 576, 1935, 2048, 3305, 4617, 6523, 8179)
_EDGE_ROWS = (13, 31, 127, 129, 1021, 4093, 8191)


def _cell(why: str, rows: int, d: int = 2304, **overrides) -> Cell:
    spec = dict(
        rows=rows,
        d=d,
        x_dtype=torch.bfloat16,
        dtype=None,
        swiglu_limit=_LIMIT,
        swiglu_alpha=None,
        swiglu_beta=None,
        row_pad=0,
    )
    spec.update(overrides)
    return Cell(why=why, spec=spec)


class SiluAndMul(OpWrapper):
    """`silu(min(g, L)) * clamp(u, -L, L)` over a packed `[gate; up]` last dim, as a new tensor.

    `x` is `[rows, 2 * d]`; `g = x[:, :d]`, `u = x[:, d:]`, loaded and computed
    in fp32. The clamp applies only when `swiglu_limit` is a positive float; the
    gate is clamped from above only, the up projection on both sides.
    `swiglu_alpha` / `swiglu_beta` (default 1.0 / 0.0) generalise the body to
    `g * sigmoid(alpha * g) * (u + beta)`. The output dtype is `dtype` if given,
    else `x.dtype`; with `scale` the result is divided by it and saturated to
    `dtype`'s finite range before the cast.

    Fusion boundary: the clamp, the activation, the gating multiply and the
    optional per-tensor output scale with its saturating cast are inside the one
    Triton launch. The caller owns the gate/up projection that produces `x` and
    whatever consumes the output (for example an MXFP8 quantize before the down
    projection); no block-scale quantization happens here. Returns a newly
    allocated contiguous `[rows, d]` tensor; `x` is not mutated.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            _cell(f"bf16->bf16 limit 10 [{rows}, 2x{d}]: {what}", rows, d)
            for rows, d, what in _SHAPES
        )
        + tuple(
            _cell(
                f"fp32->bf16 limit 10 [{rows}, 2x{d}]: dtype=bf16, rounded at the store",
                rows,
                d,
                x_dtype=torch.float32,
                dtype=torch.bfloat16,
            )
            for rows, d, _ in _SHAPES
        )
        + tuple(
            _cell(
                f"fp32->fp32 limit 10 [{rows}, 2x2304]: fp32 output",
                rows,
                x_dtype=torch.float32,
            )
            for rows in (8, 4096)
        )
        + tuple(
            _cell(f"bf16->bf16 limit 10 [{rows}, 2x2304]: a runtime-dump row count", rows)
            for rows in _RUNTIME_ROWS
        )
        + tuple(
            _cell(f"bf16->bf16 limit 10 [{rows}, 2x2304]: a prime or 128-row edge", rows)
            for rows in _EDGE_ROWS
        )
        + (
            _cell("bf16->bf16 limit 10 [32768, 2x2304]: the decoder bounded-replay top", 32768),
            _cell(
                "alpha=1.0 beta=0.0 passed: bit-identical to None, [64, 2x2304]",
                64,
                swiglu_alpha=1.0,
                swiglu_beta=0.0,
            ),
            _cell("row stride 2d + 256, unit inner stride: [256, 2x2304]", 256, row_pad=256),
            _cell("swiglu_limit=None: no clamp, [64, 2x2304]", 64, swiglu_limit=None),
            _cell("swiglu_limit=0.0: no clamp, [64, 2x2304]", 64, swiglu_limit=0.0),
            _cell("zero rows: [0, 2x2304] gives an empty [0, 2304] output", 0),
        )
    )

    note = """
    Metadata consumed: none. Stateless.

    With `swiglu_limit=10.0` and `swiglu_alpha` / `swiglu_beta` left `None` this
    is DeepSeek-V4.1's expert activation. It is not GPT-OSS's `swigluoai` body
    `g * sigmoid(1.702 g) * (u + 1)`, reachable only with `swiglu_alpha=1.702,
    swiglu_beta=1.0`. The first half of the last dim is the gate: an FC1 weight
    stacked `[up; gate]` computes the wrong function without any error.

    Preconditions: `x` 2-D on CUDA with an even last dim; a unit inner stride
    (see `is_valid`), any row stride >= `2 * d` is read correctly. Only a
    positive `swiglu_limit` clamps: `None`, `0.0` and negative values all mean no
    clamp. For bf16 output from fp32 input pass `dtype=torch.bfloat16`. The op
    raises ValueError on a 3-D `x` and a bare AssertionError on an odd last dim.

    `swiglu_limit`, `swiglu_alpha`, `swiglu_beta` and the dtype pair are Triton
    constexprs: each distinct value compiles its own kernel on first use. The
    row count is not part of the compiled kernel's key (rows are only the size
    of grid axis 0, program `(i, j)` reads and writes row `i`): every prefix
    `x[:r]`, r = 1..32768, at d = 2304, bf16, limit 10 returns the first `r` rows
    of the 32768-row result bitwise, with no new compiled variant. A CUDA-graph
    capture of a call whose constants have never run eagerly was not tested;
    after one eager call the capture works at row counts never run eagerly.

    Row counts the target hands this op reach 32768 (16384 per rank at the
    largest `max_num_tokens`; the decoder bounded-replay pass adds finishing
    requests' 128-row tails, up to 128 x 256). The target calls it only at
    d = 2304; d = 576 is the 4-way TP split it does not use. CUDA graphs are
    certified at d = 2304, bf16, limit 10 at 114 row counts: G, 5G and 6G
    (decode, DSpark drafter block, DSpark target verify) over the captured batch
    sizes 1..32, 40..128 step 8, 192 and 256, one shared pool, largest first.

    Hazards: with the clamp on, a NaN gate gives `silu(10) * u` and a NaN up is
    treated as `+10` where `torch.clamp` propagates NaN (probe, not certified).
    Inputs beyond the limit are what separate the clamp variants: a symmetric
    gate clamp and the unclamped body are indistinguishable from this op on
    inputs within +-9.5.

    Not certified: the `scale` path, fp16 input, `swiglu_alpha` / `swiglu_beta`
    other than 1.0 / 0.0, `swiglu_limit` other than 10.0, d other than 2304 and
    576, more than 32768 rows, d = 576 at row counts other than 1 and 4096, and
    CUDA-graph capture at d = 576 or at a row count outside the 114.
    """

    def raw_call(
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
        """fp32: gate = min(g, L), up = clamp(u, -L, L), silu(gate) * up, rounded once to the output dtype.

        With alpha / beta at None or 1.0 / 0.0 the body is `F.silu(gate) * up`;
        otherwise `gate * sigmoid(alpha * gate) * (up + beta)`. `scale`, when
        given, divides and saturates to the output dtype's finite range before
        the cast (not certified).
        """
        out_dtype = dtype if dtype is not None else x.dtype
        d = x.shape[-1] // 2
        gate = x[:, :d].float()
        up = x[:, d:].float()
        if swiglu_limit is not None and swiglu_limit > 0:
            gate = torch.clamp(gate, max=swiglu_limit)
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        if swiglu_alpha in (None, 1.0) and swiglu_beta in (None, 0.0):
            r = F.silu(gate) * up
        else:
            alpha = 1.0 if swiglu_alpha is None else swiglu_alpha
            beta = 0.0 if swiglu_beta is None else swiglu_beta
            r = gate * torch.sigmoid(alpha * gate) * (up + beta)
        if scale is not None:
            top = torch.finfo(out_dtype).max
            r = torch.clamp(r / scale.float().reshape(-1)[0], -top, top)
        return r.to(out_dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
        swiglu_limit: Optional[float] = None,
        swiglu_alpha: Optional[float] = None,
        swiglu_beta: Optional[float] = None,
    ) -> None:
        # The kernel walks each row with a unit element stride and takes only
        # `x.stride(0)` from the tensor, so an inner stride other than 1 is read
        # as if it were 1: a transposed view returned a bit-exact fraction of
        # 0.00047 (max abs diff 200) and no error on this machine. Row-strided
        # views with a unit inner stride are handled correctly and are not
        # rejected.
        assert x.stride(-1) == 1, (
            f"x must have a unit stride along its last dim; got strides {x.stride()} "
            f"-- the kernel ignores the inner stride and would read the wrong elements"
        )


silu_and_mul = SiluAndMul()
