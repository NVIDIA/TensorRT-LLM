# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ..kernels import fused_adaln


def _column(mod: torch.Tensor, idx: torch.Tensor, col: int, hidden: int) -> torch.Tensor:
    return mod.reshape(mod.shape[0], 6, hidden)[:, col].index_select(0, idx).float()


def _norm_mod_ref(x, w, mod, idx, eps, scale_col, shift_col):
    xf = x.float()
    norm = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).bfloat16()
    scale = _column(mod, idx, scale_col, x.shape[-1])
    shift = _column(mod, idx, shift_col, x.shape[-1])
    return norm.float() * (1 + scale) + shift


def _validate(x, w, mod, idx, *columns):
    hidden = x.shape[-1]
    # The kernel holds six 1024-element register chunks and rounds normalized values to BF16.
    assert 0 < hidden <= 6 * 1024 and x.dtype == torch.bfloat16
    assert w.shape == (hidden,) and w.is_contiguous()
    assert idx.ndim == 1 and idx.is_contiguous() and idx.dtype in (torch.int32, torch.int64)
    assert (mod.ndim == 2 and mod.shape[1] == 6 * hidden) or (
        mod.ndim == 3
        and mod.shape[1:] == (6, hidden)
        and (mod.stride(-1) != 1 or mod.stride(1) == hidden)
    )
    assert all(0 <= col < 6 for col in columns)
    assert bool(((idx >= 0) & (idx < mod.shape[0])).all())


_CELLS = tuple(
    Cell(why=f"{why}, {batch} packed batch row(s)", spec={"hidden": hidden, "batch": batch})
    for hidden, why in (
        (12, "small partial chunk"),
        (1536, "partial second chunk"),
        (5376, "released H3 width"),
        (6144, "six full chunks"),
    )
    for batch in (1, 2)
)


class H3NormMod(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = _CELLS

    def __call__(self, x, w, mod, idx, eps, scale_col, shift_col):
        return fused_adaln.h3_norm_mod(x, w, mod, idx, eps, scale_col, shift_col)

    def reference(self, x, w, mod, idx, eps, scale_col, shift_col):
        return _norm_mod_ref(x, w, mod, idx, eps, scale_col, shift_col)

    def is_valid(self, x, w, mod, idx, eps, scale_col, shift_col):
        _validate(x, w, mod, idx, scale_col, shift_col)


class H3GateResNormMod(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = _CELLS

    def __call__(self, x, a, w, mod, idx, eps, gate_col, scale_col, shift_col):
        return fused_adaln.h3_gate_res_norm_mod(
            x, a, w, mod, idx, eps, gate_col, scale_col, shift_col
        )

    def reference(self, x, a, w, mod, idx, eps, gate_col, scale_col, shift_col):
        gate = _column(mod, idx, gate_col, x.shape[-1])
        residual = (x.float() + gate * a.float()).bfloat16()
        return residual, _norm_mod_ref(residual, w, mod, idx, eps, scale_col, shift_col)

    def is_valid(self, x, a, w, mod, idx, eps, gate_col, scale_col, shift_col):
        _validate(x, w, mod, idx, gate_col, scale_col, shift_col)
        assert a.shape == x.shape


norm_mod = H3NormMod()
gate_res_norm_mod = H3GateResNormMod()
