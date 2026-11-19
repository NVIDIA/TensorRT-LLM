# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer Q / K quantization: concatenate two halves of each 128-wide row and quantize it to
FP8 with one scale per row (CUDA)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: 1 / 448 rounded to fp32, the kernel's constant (E4M3's largest finite value is 448).
_INV_E4M3_MAX = torch.tensor(1.0 / 448.0, dtype=torch.float32).item()
_MIN_AMAX = 1.0e-12


def _spec(tokens, heads, **overrides) -> dict:
    """``heads``: 32 for the indexer Q (rows = tokens x 32), 0 for the index key (rows = tokens)."""
    spec = dict(tokens=tokens, heads=heads, pe_dim=64, data="randn")
    spec.update(overrides)
    return spec


class FusedCatFp8(OpWrapper):
    """Row ``m`` of ``cat([pe, nope], -1)`` (bf16, 128 wide) is quantized to E4M3 with scale
    ``s = max(amax_m, 1e-12) * fp32(1 / 448)``, rounded up to a power of two (UE8M0):
    ``out[m] = e4m3(x[m] * (1 / s))``. Returns ``(out, scale)``: E4M3 [M, 128] and fp32 [M, 1],
    newly allocated, with ``M`` the rows of ``pe`` (all leading dims flattened). ``pe`` and ``nope``
    may be strided views (unit innermost stride).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"indexer Q of {tokens} tokens (32 heads, halves of one row-major tensor)",
                spec=_spec(tokens, 32),
            )
            for tokens in (1, 5, 8, 40, 300, 2048)
        ),
        *(
            Cell(
                why=f"index keys of {tokens} tokens (halves of the normalized wk output)",
                spec=_spec(tokens, 0),
            )
            for tokens in (1, 8, 300, 4096)
        ),
        Cell(
            why="an all-zero row: the 1e-12 amax floor",
            spec=_spec(8, 32, data="zeros_row"),
        ),
        Cell(
            why="large-magnitude rows: scales far from 1",
            spec=_spec(64, 32, data="wide"),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this twice per DSA layer per step (Q on the main stream, K on
    the auxiliary stream): the indexer has no rope, so each 128-wide row is split into two
    64-wide halves of the same tensor (q [T, 32, 128] from wq_b, k [T, 128] from the k LayerNorm)
    and passed as pe / nope views; UE8M0 scales (the indexer's scale_fmt). The UE8M0 scale is a
    power of two, so 1 / s and the quantization are exact up to E4M3's round-to-nearest: the
    output equals the reference bit for bit. One warp per row; deterministic.
    """

    def __call__(self, pe: torch.Tensor, nope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fused_cat_fp8(pe, nope, True)

    def reference(self, pe: torch.Tensor, nope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 evaluation with the kernel's scale rounding."""
        x = torch.cat([pe, nope], dim=-1).reshape(-1, pe.shape[-1] + nope.shape[-1]).float()
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(_MIN_AMAX)
        bits = (amax * _INV_E4M3_MAX).view(torch.int32)
        exponent = bits & 0x7F800000
        exponent = torch.where((bits & 0x007FFFFF) != 0, exponent + 0x00800000, exponent)
        scale = exponent.view(torch.float32)
        out = (x * (1.0 / scale)).to(torch.float8_e4m3fn)
        return out, scale

    def is_valid(self, pe: torch.Tensor, nope: torch.Tensor) -> None:
        # The kernel walks the flattened rows with one row stride, stride(-2): leading dims that
        # do not collapse onto it (a slice of heads, a transposed view) are read at the wrong
        # addresses with no error.
        for name, t in (("pe", pe), ("nope", nope)):
            for d in range(t.dim() - 2):
                assert t.stride(d) == t.stride(d + 1) * t.shape[d + 1], (
                    f"{name}'s leading dims must collapse onto stride(-2); got shape "
                    f"{tuple(t.shape)} strides {t.stride()}"
                )


fused_cat_fp8 = FusedCatFp8()
