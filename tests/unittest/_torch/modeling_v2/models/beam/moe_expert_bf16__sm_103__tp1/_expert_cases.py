# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated inputs, the plain-torch golden and the comparison rules of the expert tests.

The expert is one SwiGLU expert of Beam decoder block 3: hidden 5120, intermediate 2560,
bias-free BF16 weights [out, in], BF16 projections, fp32 SiLU and product, one BF16
rounding before the down projection. Every tensor here is generated on the host from a
seed or written down exactly, so the tests need no files.
"""

from __future__ import annotations

import contextlib
import math
import os

import torch
import torch.nn.functional as F

HIDDEN, INTER = 5120, 2560
SEEDS = (17, 41)
TOKENS = (1, 17, 129)
STAGES = ("gate", "up", "silu", "product", "act", "y")
STAGE_DTYPES = {
    "gate": torch.bfloat16,
    "up": torch.bfloat16,
    "silu": torch.float32,
    "product": torch.float32,
    "act": torch.bfloat16,
    "y": torch.bfloat16,
}
BF16_UNIT = 2.0**-7
FP32_FORMULA = 2.0**-16

#: (atol, rtol) per stage for the generated matrix, elementwise
#: ``|actual - expected| <= atol + rtol * |expected|``. rtol is one bf16 unit at any
#: magnitude; each atol is four times, rounded up, the largest excess over that measured
#: between valid implementations of this expert at these shapes (cuBLAS kernels,
#: fp32 and fp64 GEMMs, split-K orders, SiLU and product formulas).
MATRIX_RULES = {
    "gate": (4e-5, BF16_UNIT),
    "up": (4e-5, BF16_UNIT),
    "silu": (0.02, BF16_UNIT),
    "product": (0.009, BF16_UNIT),
    "act": (0.04, BF16_UNIT),
    "y": (0.008, BF16_UNIT),
}
#: Fixtures whose every projection output is one exact product, with values away from
#: any bf16 rounding boundary: projections, activation and output are exact, and SiLU and
#: product are fp32 formulas of exact inputs, within 2^-16 relative.
EXACT_RULES = {
    "gate": (0.0, 0.0),
    "up": (0.0, 0.0),
    "silu": (0.0, FP32_FORMULA),
    "product": (0.0, FP32_FORMULA),
    "act": (0.0, 0.0),
    "y": (0.0, 0.0),
}

_TF32_OVERRIDES = ("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE")


# ----------------------------------------------------------------------------- numerics


def numerical_state() -> dict:
    m = torch.backends.cuda.matmul
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_fp32_precision": m.fp32_precision,
        "cudnn_fp32_precision": torch.backends.cudnn.fp32_precision,
        "cuda_matmul_allow_tf32": m.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "bf16_reduced_precision_reduction": m.allow_bf16_reduced_precision_reduction,
        "autocast_cuda": torch.is_autocast_enabled("cuda"),
    }


_CONTROLLED = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_fp32_precision": "ieee",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "bf16_reduced_precision_reduction": False,
    "autocast_cuda": False,
}


def assert_numerical_controls() -> None:
    for var in _TF32_OVERRIDES:
        assert var not in os.environ, (
            f"{var} substitutes TF32 for fp32 GEMMs; run with env -u {var}"
        )
    state = numerical_state()
    assert state == _CONTROLLED, f"numerical controls not in force: {state}"


@contextlib.contextmanager
def numerical_controls():
    """IEEE fp32 matmul, bf16 GEMMs without reduced-precision reduction, autocast off.

    The golden's fp32 steps and its bf16 GEMMs follow these switches; the comparison rules
    assume them. They are asserted on entry and exit and restored afterwards.
    """
    m = torch.backends.cuda.matmul
    saved = (
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.allow_tf32,
        m.fp32_precision,
        torch.backends.cudnn.fp32_precision,
        m.allow_bf16_reduced_precision_reduction,
    )
    torch.set_float32_matmul_precision("highest")
    m.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
    m.allow_bf16_reduced_precision_reduction = False
    try:
        with torch.autocast("cuda", enabled=False):
            assert_numerical_controls()
            yield
            assert_numerical_controls()
    finally:
        # The legacy switches first: their setters also move the newer ones restored after.
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cudnn.allow_tf32 = saved[1]
        m.fp32_precision = saved[2]
        torch.backends.cudnn.fp32_precision = saved[3]
        m.allow_bf16_reduced_precision_reduction = saved[4]


def golden(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor):
    """The source expert in plain torch, every step, on the inputs' device."""
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    silu = F.silu(gate.float())
    product = silu * up.float()
    act = product.to(torch.bfloat16)
    return dict(gate=gate, up=up, silu=silu, product=product, act=act, y=F.linear(act, down_w))


def violations(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    """Elementwise rule check, both sides widened exactly to fp64; a nonfinite value fails."""
    a = actual.detach().double()
    e = expected.detach().to(actual.device).double()
    diff = (a - e).abs()
    ok = diff <= atol + rtol * e.abs()  # NaN compares False: a nonfinite element fails
    bad = ~ok
    worst = int(torch.argmax(torch.where(ok, torch.zeros_like(diff), diff.nan_to_num(math.inf))))
    return {
        "failing": int(bad.sum()),
        "nonfinite": int((~torch.isfinite(a)).sum()),
        "max_abs": float(diff.nan_to_num(math.inf).max()) if diff.numel() else 0.0,
        "worst": (worst, float(a.reshape(-1)[worst]), float(e.reshape(-1)[worst]))
        if diff.numel()
        else None,
    }


def assert_stages(actual: dict, expected: dict, rules: dict, what: str) -> None:
    """Every stage: its dtype and shape, and each element within that stage's rule."""
    problems = []
    for stage in STAGES:
        a, e = actual[stage], expected[stage]
        if a.dtype != STAGE_DTYPES[stage]:
            problems.append(f"{stage}: dtype {a.dtype}, expected {STAGE_DTYPES[stage]}")
        if a.shape != e.shape:
            problems.append(f"{stage}: shape {tuple(a.shape)}, expected {tuple(e.shape)}")
            continue
        found = violations(a, e, *rules[stage])
        if found["failing"] or found["nonfinite"]:
            problems.append(f"{stage}: {found} under (atol, rtol) {rules[stage]}")
    assert not problems, f"{what}: " + "; ".join(problems)


# ----------------------------------------------------------------------------- generated cases


def matrix_case(seed: int) -> tuple[dict, torch.Tensor]:
    """Seed -> (weights by checkpoint name, the [129, 5120] input; case T uses its first T rows).

    Draw order gate, up, down, input, standard normal fp32 from a host generator; gate and
    up scaled by 1/sqrt(5120), down by 1/sqrt(2560), each rounded once to bf16.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    def draw(rows: int, cols: int, scale: float) -> torch.Tensor:
        return (torch.randn(rows, cols, generator=g, dtype=torch.float32) * scale).to(
            torch.bfloat16
        )

    weights = {
        "gate_proj.weight": draw(INTER, HIDDEN, 1.0 / math.sqrt(HIDDEN)),
        "up_proj.weight": draw(INTER, HIDDEN, 1.0 / math.sqrt(HIDDEN)),
        "down_proj.weight": draw(HIDDEN, INTER, 1.0 / math.sqrt(INTER)),
    }
    return weights, draw(max(TOKENS), HIDDEN, 1.0)


def as_weights(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> dict:
    return {"gate_proj.weight": gate, "up_proj.weight": up, "down_proj.weight": down}


# ----------------------------------------------------------------------------- exact fixtures


def silu64(t: torch.Tensor) -> torch.Tensor:
    return F.silu(t.double())


def bf16_margin(v: torch.Tensor) -> torch.Tensor:
    """Relative distance of each fp64 value from its nearest bf16 rounding boundary."""
    v = v.double()
    nearest = v.to(torch.bfloat16)
    bits = nearest.view(torch.int16)
    above = (bits + 1).view(torch.bfloat16).double()  # one unit further from zero
    below = (bits - 1).view(torch.bfloat16).double()  # one unit nearer to zero
    n = nearest.double()
    dist = torch.minimum((v - (n + above) / 2).abs(), (v - (n + below) / 2).abs())
    return torch.where(v == 0, torch.full_like(v, math.inf), dist / v.abs())


def bf16_grid(lo_exp: int, hi_exp: int, powers_of_two: bool = True) -> torch.Tensor:
    """Every positive bf16 value in [2^lo_exp, 2^(hi_exp + 1)), as fp64."""
    mantissas = torch.arange(0 if powers_of_two else 1, 128, dtype=torch.float64)
    return torch.cat([(1 + mantissas / 128) * 2.0**e for e in range(lo_exp, hi_exp + 1)])


def identity_block_weights() -> dict:
    """gate[i] = x[i], up[i] = x[2560 + i], y[5119 - 2i] = act[i]: one exact term each."""
    i = torch.arange(INTER)
    gate = torch.zeros(INTER, HIDDEN, dtype=torch.bfloat16)
    up = torch.zeros(INTER, HIDDEN, dtype=torch.bfloat16)
    down = torch.zeros(HIDDEN, INTER, dtype=torch.bfloat16)
    gate[i, i] = 1
    up[i, i + INTER] = 1
    down[HIDDEN - 1 - 2 * i, i] = 1
    return as_weights(gate, up, down)


def identity_block_expected(gate: torch.Tensor, up: torch.Tensor) -> dict:
    """What every stage must be, exactly, for exact bf16 `gate`/`up` through identity blocks."""
    s = silu64(gate)
    act = (s.float().double() * up).float().to(torch.bfloat16)
    y = torch.zeros(gate.shape[0], HIDDEN, dtype=torch.bfloat16)
    y[:, HIDDEN - 1 - 2 * torch.arange(INTER)] = act
    return dict(
        gate=gate.to(torch.bfloat16),
        up=up.to(torch.bfloat16),
        silu=s.float(),
        product=(s.float().double() * up).float(),
        act=act,
        y=y,
    )


def signed_gate_fixture() -> tuple[dict, torch.Tensor, dict]:
    """Negative, zero and positive gates over four decades, power-of-two signed ups.

    Gates whose SiLU lies within 2^-16 of a bf16 rounding boundary are dropped, so every
    stage is exact; with a power-of-two up the product only rescales the SiLU.
    """
    magnitudes = [1000.0, 60.0, 24.0, 10.0, 6.0, 4.5, 3.0, 2.25, 1.75, 1.28125, 1.0, 0.875]
    magnitudes += [0.625, 0.375, 0.25, 0.125, 2.0**-4, 2.0**-6, 2.0**-10, 2.0**-20, 2.0**-60]
    candidates = torch.tensor(
        [-m for m in magnitudes] + [0.0, -0.0] + magnitudes, dtype=torch.float64
    )
    gates = candidates[bf16_margin(silu64(candidates)) > FP32_FORMULA]
    ups = torch.tensor(
        [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 4.0, -0.25, 0.125, -8.0], dtype=torch.float64
    )
    i = torch.arange(INTER)
    gate = torch.stack([gates[(i + 37 * t) % len(gates)] for t in range(4)])
    up = torch.stack([ups[(5 * i + 3 * t) % len(ups)] for t in range(4)])
    x = torch.cat([gate, up], dim=1).to(torch.bfloat16)
    return identity_block_weights(), x, identity_block_expected(gate, up)


def precision_fixture() -> tuple[dict, torch.Tensor, dict, torch.Tensor]:
    """Gate/up pairs where the fp32 SiLU and product, rounded once, differ from bf16 steps.

    Over every bf16 gate with |g| in [2^-8, 8) and every non-power-of-two bf16 up with |u|
    in [0.5, 4), keep the pairs whose correct activation and whose prematurely rounded one
    (bf16 SiLU, bf16 product) are both 2^-16 or more from a bf16 rounding boundary and
    differ, then draw 4 x 2560 of them with a fixed seed. Returns the weights, the input,
    the exact stages and the premature activation every element must differ from.
    """
    g_pos, u_pos = bf16_grid(-8, 2), bf16_grid(-1, 1, powers_of_two=False)
    g = torch.cat([-g_pos.flip(0), g_pos])
    u = torch.cat([-u_pos.flip(0), u_pos])
    s = silu64(g)
    product = s.float().double()[:, None] * u[None, :]
    premature = s.float().to(torch.bfloat16).double()[:, None] * u[None, :]
    robust = (bf16_margin(product) > FP32_FORMULA) & (bf16_margin(s) > FP32_FORMULA)[:, None]
    robust &= bf16_margin(premature) > FP32_FORMULA
    differs = product.float().to(torch.bfloat16) != premature.to(torch.bfloat16)
    rows, cols = torch.nonzero(robust & differs, as_tuple=True)
    pick = torch.randperm(rows.numel(), generator=torch.Generator().manual_seed(20261020))
    pick = pick[: 4 * INTER]
    gate, up = g[rows[pick]].reshape(4, INTER), u[cols[pick]].reshape(4, INTER)
    x = torch.cat([gate, up], dim=1).to(torch.bfloat16)
    premature_act = (s.float().to(torch.bfloat16).double()[rows[pick]] * u[cols[pick]]).to(
        torch.bfloat16
    )
    return (
        identity_block_weights(),
        x,
        identity_block_expected(gate, up),
        premature_act.reshape(4, INTER),
    )


def sparse_mapping_fixture() -> tuple[dict, torch.Tensor, dict]:
    """Asymmetric one-term maps with power-of-two markers:
    gate[i] = m_g(i) x[a(i)], up[i] = m_u(i) x[b(i)], y[d(i)] = m_d(i) act[i].

    a and b together cover all 5120 input features, d is a permutation into the 5120
    outputs, and gate and up read different inputs with different markers, so a transposed
    weight, swapped gate/up roles or a wrong down mapping all move some output.
    """
    i = torch.arange(INTER)
    a, b, d = (3 * i + 7) % HIDDEN, (3 * i + 2567) % HIDDEN, (HIDDEN - 1 - 3 * i) % HIDDEN
    m_g = 2.0 ** ((i % 5) - 2).double()
    m_u = -(2.0 ** ((i % 3) - 1).double())
    m_d = 2.0 ** ((i % 4) - 1).double()
    candidates = torch.cat([-bf16_grid(-3, 0).flip(0), bf16_grid(-3, 0)])
    robust = torch.ones_like(candidates, dtype=torch.bool)
    for marker in (0.25, 0.5, 1.0, 2.0, 4.0):
        robust &= bf16_margin(silu64(candidates * marker)) > FP32_FORMULA
    palette = candidates[robust]
    palette = palette[torch.randperm(palette.numel(), generator=torch.Generator().manual_seed(7))]
    x = torch.zeros(3, HIDDEN, dtype=torch.float64)
    for t in range(3):
        x[t, a] = palette[(7 * i + 13 * t) % 64]
        sign = torch.where((i + 2 * t) % 3 == 0, -1.0, 1.0).double()
        x[t, b] = sign * 2.0 ** (((i + t) % 5) - 2).double()
    gate_w = torch.zeros(INTER, HIDDEN, dtype=torch.bfloat16)
    up_w = torch.zeros(INTER, HIDDEN, dtype=torch.bfloat16)
    down_w = torch.zeros(HIDDEN, INTER, dtype=torch.bfloat16)
    gate_w[i, a] = m_g.to(torch.bfloat16)
    up_w[i, b] = m_u.to(torch.bfloat16)
    down_w[d, i] = m_d.to(torch.bfloat16)
    gate, up = x[:, a] * m_g, x[:, b] * m_u
    expected = identity_block_expected(gate, up)
    y = torch.zeros(3, HIDDEN, dtype=torch.bfloat16)
    y[:, d] = (expected["act"].double() * m_d).to(torch.bfloat16)
    expected["y"] = y
    return as_weights(gate_w, up_w, down_w), x.to(torch.bfloat16), expected
