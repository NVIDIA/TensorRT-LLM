# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-tiled final gate/residual, validated against the existing compiled expression."""

import cache_env  # noqa: F401 -- configure caches before importing CUDA libraries
import torch
import triton
import triton.language as tl

REAL_CHECKS = []
STATE = {
    "validate": False,
    "enabled": False,
    "tile": 2048,
    "warps": 4,
    "round_mul": False,
    "fma": True,
}


def expression(x, a, mod, idx):
    d = x.shape[-1]
    return x + mod[:, 5 * d : 6 * d].index_select(0, idx) * a


compiled_expression = torch.compile(expression, fullgraph=True, dynamic=False)


@triton.jit
def _residual_kernel(
    X,
    A,
    MOD,
    IDX,
    OUT,
    D: tl.constexpr,
    MS: tl.constexpr,
    TILE: tl.constexpr,
    ROUND: tl.constexpr,
    FMA: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    selected = tl.load(IDX + row).to(tl.int64)
    x = tl.load(X + row * D + col, mask=col < D, other=0).to(tl.float32)
    a = tl.load(A + row * D + col, mask=col < D, other=0).to(tl.float32)
    gate = tl.load(MOD + selected * MS + 5 * D + col, mask=col < D, other=0).to(tl.float32)
    if ROUND:
        mul = (gate * a).to(tl.bfloat16).to(tl.float32)
        value = x + mul
    elif FMA:
        value = tl.fma(gate, a, x)
    else:
        value = x + gate * a
    tl.store(OUT + row * D + col, value.to(tl.bfloat16), mask=col < D)


def launch(x, a, mod, idx):
    assert x.shape[0] == 1 and x.is_contiguous() and a.is_contiguous()
    out = torch.empty_like(x)
    d = x.shape[-1]
    rows = x.numel() // d
    _residual_kernel[(rows, triton.cdiv(d, STATE["tile"]))](
        x,
        a,
        mod,
        idx,
        out,
        d,
        mod.stride(0),
        STATE["tile"],
        STATE["round_mul"],
        STATE["fma"],
        num_warps=STATE["warps"],
        enable_fp_fusion=False,
    )
    return out


@torch.library.custom_op("h3_fp8_round3::residual", mutates_args=())
def residual(
    x: torch.Tensor, a: torch.Tensor, mod: torch.Tensor, idx: torch.Tensor
) -> torch.Tensor:
    if STATE.get("validate") and len(REAL_CHECKS) < 1300:
        ref = compiled_expression(x, a, mod, idx)
        candidate = launch(x, a, mod, idx)
        REAL_CHECKS.append(
            dict(
                index=len(REAL_CHECKS),
                shape=list(x.shape),
                mod_shape=list(mod.shape),
                x_dtype=str(x.dtype),
                mod_dtype=str(mod.dtype),
                byte_equal=bool(torch.equal(ref.view(torch.uint8), candidate.view(torch.uint8))),
            )
        )
        import json
        from pathlib import Path

        if len(REAL_CHECKS) % 50 == 0:
            Path("/profile/results/residual-real-checks.json").write_text(
                json.dumps(REAL_CHECKS, indent=2)
            )
    if STATE["enabled"]:
        return launch(x, a, mod, idx)
    return compiled_expression(x, a, mod, idx)


@residual.register_fake
def _(x, a, mod, idx):
    return torch.empty_like(x)
