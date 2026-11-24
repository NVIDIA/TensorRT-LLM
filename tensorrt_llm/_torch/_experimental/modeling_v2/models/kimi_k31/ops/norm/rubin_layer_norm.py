# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LayerNorm over the last dim in FP32, rounded back to the input dtype: the DSA indexer's key norm.

The engine's ``LayerNorm`` module (the engine DSA indexer's key norm) runs its forward through
``torch.compile`` (``maybe_compile``, one fused kernel). This entry compiles the same function
body, so the key norm is the same kernel and the same bits, without the module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    xf = x.to(torch.float32)
    out = F.layer_norm(
        xf, (xf.shape[-1],), weight=weight.to(torch.float32), bias=bias.to(torch.float32), eps=eps
    )
    return out.to(x.dtype)


# The engine LayerNorm's compile settings (dynamic shapes: the row count changes every step).
_compiled_layer_norm = torch.compile(_layer_norm, dynamic=True)


class RubinLayerNorm(OpWrapper):
    """``layer_norm(x.float(), weight.float(), bias.float(), eps).to(x.dtype)`` over the last dim.

    ``x`` ``[M, D]``; ``weight`` and ``bias`` ``[D]`` (FP32 or the input dtype). Returns a new
    ``[M, D]`` tensor in ``x``'s dtype.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="indexer key norm, TEP8 decode: 8 rows x 128",
            spec=dict(m=8, d=128, dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="indexer key norm, DEP16 / helix8 decode: 96 rows x 128",
            spec=dict(m=96, d=128, dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="indexer key norm, a 16k prefill chunk",
            spec=dict(m=16384, d=128, dtype=torch.bfloat16, eps=1e-5),
        ),
    )
    note = """
    The kernel is inductor's code for the function body, compiled on the first call of a dtype
    pair; the call must not be the first one under CUDA graph capture (the targets warm up
    eagerly first, as the engine module does).
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:
        return _compiled_layer_norm(x, weight, bias, eps)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """fp64 mean and biased variance, one rounding to ``x``'s dtype at the end."""
        xd = x.double()
        mean = xd.mean(-1, keepdim=True)
        var = (xd - mean).pow(2).mean(-1, keepdim=True)
        out = (xd - mean) * torch.rsqrt(var + eps) * weight.double() + bias.double()
        return out.to(x.dtype)

    def is_valid(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
    ) -> None:
        # A weight or bias of another length broadcasts only when it is 1 long: every channel
        # then takes channel 0's affine, with no error.
        assert weight.shape == (x.shape[-1],) and bias.shape == (x.shape[-1],), (
            "weight and bias must hold one value per normalized channel"
        )


rubin_layer_norm = RubinLayerNorm()
