# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark RMSNorm + RoPE of row-strided BF16 rows into a caller-owned output, through a
compiled handle (CuTe DSL): the per-layer context-latent path of the MLA drafter."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._dspark_mla_rope import _compile_rmsnorm_rope_strided
from .k3_dspark_rmsnorm_rope import rmsnorm_rope_reference


@dataclass(frozen=True)
class CompiledRmsnormRope:
    """A compiled ``k3_dspark_rmsnorm_rope`` kernel and the configuration it was built for."""

    hidden_dim: int
    rope_dim: int
    num_heads: int
    eps: float
    apply_weight: bool
    apply_rmsnorm: bool
    inverse_rope: bool
    norm_dim: int
    kernel: Any


@functools.cache
def build_rmsnorm_rope_into(
    hidden_dim: int,
    rope_dim: int,
    num_heads: int,
    eps: float,
    apply_weight: bool,
    apply_rmsnorm: bool,
    inverse_rope: bool,
    norm_dim: int,
) -> CompiledRmsnormRope:
    """The handle for this configuration (compiled once per process)."""
    kernel = _compile_rmsnorm_rope_strided(
        hidden_dim, rope_dim, num_heads, eps, apply_weight, apply_rmsnorm, inverse_rope, norm_dim
    )
    return CompiledRmsnormRope(
        hidden_dim,
        rope_dim,
        num_heads,
        eps,
        apply_weight,
        apply_rmsnorm,
        inverse_rope,
        norm_dim,
        kernel,
    )


class K3DsparkRmsnormRopeInto(OpWrapper):
    """``k3_dspark_rmsnorm_rope`` with ``compiled``'s configuration, written into ``out``
    (compact BF16 ``[rows, hidden_dim]``). Returns ``out``.

    ``x`` is BF16 ``[rows, hidden_dim]`` with a unit inner stride and any row stride (read in
    place), ``weight`` BF16 ``[norm_dim]`` (or None without ``apply_weight``), ``freqs`` FP32
    ``[rows / num_heads, rope_dim // 2, 2]``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS = (
        Cell(
            why="tp1 context server: one context chunk of max_num_tokens 16384 rows, each "
            "layer's slice of the [N, 5, 576] latent GEMM output (row stride 2880)",
            spec=dict(rows=16384, layers=5),
        ),
        Cell(
            why="tp1 context server: a one-token chunk",
            spec=dict(rows=1, layers=5),
        ),
        Cell(
            why="dep16 generation: the accepted tokens of 16 requests, 16 x 6 rows",
            spec=dict(rows=96, layers=5),
        ),
    )

    note = """
    The drafter normalizes each layer's slice of its stacked kv_a GEMM output with one handle
    built before the loop, without the support check per call (the caller checks once). Same
    kernel and arithmetic as ``k3_dspark_rmsnorm_rope``.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs: torch.Tensor,
        out: torch.Tensor,
        compiled: CompiledRmsnormRope,
    ) -> torch.Tensor:
        compiled.kernel(x, weight, freqs, out)
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs: torch.Tensor,
        out: torch.Tensor,
        compiled: CompiledRmsnormRope,
    ) -> torch.Tensor:
        """``out`` after the call (new); ``out`` is not written."""
        return rmsnorm_rope_reference(
            x,
            weight,
            freqs,
            compiled.num_heads,
            compiled.rope_dim,
            compiled.eps,
            compiled.apply_weight,
            compiled.apply_rmsnorm,
            compiled.inverse_rope,
            compiled.norm_dim,
        )

    def is_valid(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        freqs: torch.Tensor,
        out: torch.Tensor,
        compiled: CompiledRmsnormRope,
    ) -> None:
        # The handle is compiled for these widths; another width is read with its geometry.
        assert x.shape[-1] == out.shape[-1] == compiled.hidden_dim, "width of another handle"
        assert out.is_contiguous() and out.dtype == x.dtype == torch.bfloat16
        assert freqs.shape[0] * compiled.num_heads >= x.shape[0], "fewer frequency rows"
        if compiled.apply_weight:
            assert weight.numel() == compiled.norm_dim, "weight must span the normalized columns"


k3_dspark_rmsnorm_rope_into = K3DsparkRmsnormRopeInto()
