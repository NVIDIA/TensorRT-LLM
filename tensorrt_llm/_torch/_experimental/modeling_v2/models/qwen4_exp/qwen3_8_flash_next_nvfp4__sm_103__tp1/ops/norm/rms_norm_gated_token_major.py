# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated RMS normalization with the gate read from a token-major 3-D view,
with optional fused static-scale FP8 output quantization."""

from __future__ import annotations

from typing import Optional

import torch

# Registration is lazy: the op is defined beside the mamba/GDN mixer, and
# `import tensorrt_llm._torch.custom_ops` alone does not make
# torch.ops.trtllm.rms_norm_gated_token_major resolve.
import tensorrt_llm._torch.modules.mamba.layernorm_gated  # noqa: F401
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's GDN epilogue: 48 value heads of 128, `output_gate_type: sigmoid`.
_CKPT = dict(heads=48, width=128, dtype=torch.bfloat16, eps=1e-6, gate_activation="sigmoid")
#: The same geometry under the op's default silu gate.
_SILU = dict(_CKPT, gate_activation="silu")


class RmsNormGatedTokenMajor(OpWrapper):
    """`rmsnorm(x) * weight * act(z)`, new tensor out, row `r` of `x` gated by `z[r // heads, r % heads]`.

    `x` is `[T * heads, N]`; `z` is `[T, heads, N]` (heads = `z.shape[1]`),
    its `(heads, N)` block contiguous per token and its token stride free, so a
    column slice of a wider projection is read in place. `act(v)` is
    `v * sigmoid(v)` for `"silu"` and `sigmoid(v)` for `"sigmoid"`. All fp32,
    rounded once. Norm before gate, plain `w` (not `1 + w`).

    With `fp8_scale` (one fp32 element, the downstream linear's static input
    scale) the output is `float8_e4m3fn`: the gated value is rounded to
    `x.dtype`, multiplied by `1 / fp8_scale[0]` and saturated at +-448 --
    byte-identical to the unfused norm -> static-quantize pair.

    Fusion boundary: norm, weight, gate and the optional quantize. The caller
    owns both projections producing `x` and `z`, and every reshape.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: 1 token, checkpoint GDN epilogue, sigmoid", spec=dict(tokens=1, **_CKPT)),
        Cell(why="17 tokens, checkpoint geometry, sigmoid", spec=dict(tokens=17, **_CKPT)),
        Cell(
            why="prefill 2048 tokens, checkpoint geometry, sigmoid", spec=dict(tokens=2048, **_CKPT)
        ),
        Cell(why="silu gate (the op's default), 2 tokens", spec=dict(tokens=2, **_SILU)),
        Cell(why="silu gate, 512 tokens", spec=dict(tokens=512, **_SILU)),
        Cell(
            why="z a column slice of a 16384-wide projection, 1 token",
            spec=dict(tokens=1, z="column_slice", **_SILU),
        ),
        Cell(
            why="z a column slice: token stride 16384, 512 tokens",
            spec=dict(tokens=512, z="column_slice", **_CKPT),
        ),
        Cell(
            why="x row stride 2N: only stride(-1) must be 1",
            spec=dict(tokens=17, x="strided_rows", **_SILU),
        ),
        Cell(
            why="7 rows: not a multiple of the kernel's 4 rows per program",
            spec=dict(_SILU, tokens=7, heads=1),
        ),
        Cell(why="15 rows over 3 heads: row mask", spec=dict(_SILU, tokens=5, heads=3)),
        Cell(why="6 rows, 1 token x 6 heads: row mask", spec=dict(_SILU, tokens=1, heads=6)),
        Cell(
            why="N 192: not a power of two, generic fallback kernel",
            spec=dict(_SILU, tokens=9, heads=4, width=192),
        ),
        Cell(why="N 512: wider than 256, fallback", spec=dict(_SILU, tokens=9, heads=4, width=512)),
        Cell(why="N 1024: fallback", spec=dict(_SILU, tokens=9, heads=4, width=1024)),
        Cell(
            why="z head stride 2N: not per-token contiguous, fallback",
            spec=dict(_SILU, tokens=9, heads=4, z="head_strided"),
        ),
        Cell(why="fp16", spec=dict(_SILU, tokens=17, dtype=torch.float16)),
        Cell(why="fp32", spec=dict(_SILU, tokens=17, dtype=torch.float32)),
        Cell(
            why="eps 1.0 on rows scaled by 0.01: eps dominates the mean square",
            spec=dict(_SILU, tokens=4, heads=8, eps=1.0, x_scale=0.01),
        ),
        Cell(why="fp8 out, scale 1.0", spec=dict(tokens=17, fp8_scale=1.0, **_SILU)),
        Cell(why="fp8 out, scale 0.5", spec=dict(tokens=17, fp8_scale=0.5, **_SILU)),
        Cell(why="fp8 out, scale 0.0234375", spec=dict(tokens=17, fp8_scale=0.0234375, **_SILU)),
    )

    note = """
    Metadata consumed: none. Two Triton kernels: a multi-row kernel (4 rows per
    program) when `x.stride(-1) == 1`, `z.stride(2) == 1`, `z.stride(1) == N`
    and N is a power of two <= 256 -- the checkpoint's N = 128 -- else the
    generic single-row kernel on a packed copy of `z`. PDL on the multi-row
    path only when the grid is smaller than the SM count and
    `TRTLLM_ENABLE_PDL != 0`; results are bitwise identical with it off.

    Loud, so not guarded: 3-D `x` (ValueError), a `z` not covering `x`
    (AssertionError "does not match x shape"; 2-D `z` ValueError), an unknown
    `gate_activation` (ValueError before launch), and on the fallback a
    non-unit `x.stride(-1)` (bare AssertionError). `is_valid` covers what the
    multi-row kernel accepts silently: a weight longer than N (truncated to
    `weight[:N]`) or shorter (read out of bounds), and an `fp8_scale` with
    extra elements (only `[0]` read) or not fp32 (widened silently).

    `weight` and `z` are widened to fp32 and need not match `x.dtype`. The
    output is freshly allocated and contiguous even for a strided `x`; no input
    is mutated.

    Numerics: bf16 bitwise identical to the fp32 reference up to 816 rows; at
    24576 and 196608 rows one-ulp differences on ~1e-5 of elements, inside the
    default tolerance. fp8 output byte-identical. Nothing loosened.

    Hard limits
      N * itemsize  fallback holds a row in 64 KB  boundary 65536 B
      probed at bf16 N = 32768 (returns) and 40960 (RuntimeError)
    """

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
    ) -> torch.Tensor:
        return torch.ops.trtllm.rms_norm_gated_token_major(
            x, z, weight, eps, fp8_scale, gate_activation
        )

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
    ) -> torch.Tensor:
        """fp32 throughout; row `t * heads + h` pairs with `z[t, h]`.

        With `fp8_scale` the gated value is rounded to `x.dtype` first, then
        multiplied by the fp32 reciprocal of the scale, as the kernel does.
        """
        num_rows, width = x.shape
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()
        zf = z.reshape(num_rows, width).float()
        gate = torch.sigmoid(zf)
        if gate_activation != "sigmoid":
            gate = gate * zf
        y = y * gate
        if fp8_scale is None:
            return y.to(x.dtype)
        return (y.to(x.dtype).float() * (1.0 / fp8_scale.float())).to(torch.float8_e4m3fn)

    def is_valid(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        fp8_scale: Optional[torch.Tensor] = None,
        gate_activation: str = "silu",
    ) -> None:
        assert weight.shape == (x.shape[-1],), (
            f"weight must be 1-D of length x.shape[-1]={x.shape[-1]}; got "
            f"{tuple(weight.shape)} -- the multi-row kernel reads weight[0:N] "
            "unmasked, so a longer weight is silently truncated and a shorter "
            "one is read out of bounds"
        )
        if fp8_scale is not None:
            assert fp8_scale.numel() == 1, (
                f"fp8_scale must hold exactly one element; got {fp8_scale.numel()} "
                "-- the multi-row kernel loads fp8_scale[0] and ignores the rest"
            )
            assert fp8_scale.dtype == torch.float32, (
                f"fp8_scale must be float32; got {fp8_scale.dtype} -- the "
                "multi-row kernel widens it and quantizes against the "
                "lower-precision scale"
            )


rms_norm_gated_token_major = RmsNormGatedTokenMajor()
