# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and comparison rules shared by the Beam MoE router's tests.

Everything here is generated on the CPU from fixed seeds, so the bytes a test routes are
the same on every machine and can be checked against routes captured elsewhere.

Matrix cases, per seed: a CPU generator seeded with it draws the FP32 router weight
``randn(256, 5120) / sqrt(5120)``, then the FP32 correction bias ``0.05 * randn(256)``,
then FP32 token rows ``randn(T, 5120)``, scaled to unit RMS and rounded to BF16 -- the
already-normalized input the router receives. Weight and bias are drawn first, so they
are the same for every token count of a seed.

Controlled fixtures fix the logits exactly instead: token row ``r`` is one-hot at column
``r`` and the seed-17 weight's column ``r`` is replaced by the logits that row should
produce, so ``logits[r] = weight[:, r]`` whatever the GEMM's reduction order. Each fixture
is a decision the router has to get right and says which.
"""

from __future__ import annotations

import math

import torch

HIDDEN, EXPERTS, GROUPS, PER_GROUP, TOPK_GROUP, TOP_K = 5120, 256, 8, 32, 4, 8
NORM_EPS = 1e-20
SEEDS = (17, 41)
TOKENS = (1, 17, 129)
CASES = tuple((seed, tokens) for seed in SEEDS for tokens in TOKENS)
CASE_IDS = [f"seed{seed}-T{tokens}" for seed, tokens in CASES]

#: Elementwise |actual - reference| <= atol (rtol 0) per compared tensor, both widened to
#: fp64. Each atol is four times the largest difference measured between valid fp32
#: evaluations of this routing at these shapes and inputs -- GEMM reduction orders and
#: layouts, sigmoid forms, normalization forms -- rounded up to one significant digit.
#: A route is never excused by a tolerance: groups and ids must match exactly, in order.
RULES = {
    "logits": 8e-6,
    "scores": 3e-6,
    "selection_scores": 3e-6,
    "group_scores": 2e-6,
    "weights": 8e-7,
}

#: |sum(weights) - 1| bound for every row whose selected scores are ordinary: eight fp32
#: terms summed, divided and summed again err by at most 15 units of 2^-24 (8.9e-7).
NORM_ATOL = 1e-6


# ----------------------------------------------------------------------------- matrix


def matrix_case(seed: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(weight [256, 5120] fp32, correction bias [256] fp32, hidden [T, 5120] bf16) on CPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(EXPERTS, HIDDEN, generator=g, dtype=torch.float32) * (
        1.0 / math.sqrt(HIDDEN)
    )
    bias = torch.randn(EXPERTS, generator=g, dtype=torch.float32) * 0.05
    x = torch.randn(tokens, HIDDEN, generator=g, dtype=torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True))
    return weight, bias, x.to(torch.bfloat16)


# ----------------------------------------------------------------------------- controlled fixtures


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _row(default: float) -> list:
    return [default] * EXPERTS


def _set_group(row: list, group: int, values) -> None:
    for j, v in enumerate(values):
        row[group * PER_GROUP + j] = v


def _logits_for_scores(rows) -> torch.Tensor:
    """FP32 logits whose sigmoid is each row's intended score."""
    return torch.tensor([[_logit(p) for p in r] for r in rows], dtype=torch.float64).float()


def _bias_fixtures() -> list:
    # Each group has two lead experts; the rest sit at distinct values just above 0.30.
    p = [0.30 + 0.0001 * ((e * 37) % EXPERTS) for e in range(EXPERTS)]
    leads = {
        0: (0.70, 0.69),
        1: (0.68, 0.67),
        2: (0.66, 0.65),
        3: (0.64, 0.63),
        4: (0.62, 0.61),
        5: (0.60, 0.59),
        6: (0.50, 0.49),
        7: (0.48, 0.47),
    }
    for group, (a, b) in leads.items():
        p[group * PER_GROUP], p[group * PER_GROUP + 1] = a, b
    logits = _logits_for_scores([p])
    shifted = torch.zeros(EXPERTS, dtype=torch.float32)
    shifted[6 * PER_GROUP] = 0.25
    shifted[6 * PER_GROUP + 1] = 0.27
    return [
        dict(
            name="bias_zero",
            rows=("lead_experts",),
            logits=logits,
            bias=torch.zeros(EXPERTS),
            intent="no bias: groups [0, 1, 2, 3], experts 0, 1, 32, 33, 64, 65, 96, 97",
        ),
        dict(
            name="bias_shift",
            rows=("lead_experts",),
            logits=logits,
            bias=shifted,
            intent="+0.25/+0.27 on experts 192/193 lifts group 6 to first: groups [6, 0, 1, 2]; "
            "the weights stay the normalized unbiased scores of the selected experts",
        ),
    ]


def _group_boundary() -> dict:
    rows = []
    for second in (0.62, 0.67):
        r = _row(0.20)
        _set_group(r, 0, [0.90, 0.89])
        _set_group(r, 1, [0.88, 0.87])
        _set_group(r, 2, [0.86, 0.85])
        _set_group(r, 3, [0.655, 0.645])  # incumbent: top-2 sum 1.300
        _set_group(r, 4, [0.66, second])  # challenger: 1.28 below, 1.33 above
        _set_group(r, 5, [0.95] + [0.05] * (PER_GROUP - 1))  # best single expert, sum 1.00
        _set_group(r, 6, [0.30, 0.29])
        _set_group(r, 7, [0.28, 0.27])
        rows.append(r)
    return dict(
        name="group_boundary",
        rows=("challenger_below", "challenger_above"),
        logits=_logits_for_scores(rows),
        bias=torch.zeros(EXPERTS),
        intent="only group 4's second expert moves (0.62 -> 0.67): groups [0, 1, 2, 3] -> "
        "[0, 1, 2, 4]; scoring groups by their best expert alone would pick group 5",
    )


def _expert_boundary() -> dict:
    rows = []
    for mover in (0.79, 0.82):
        r = _row(0.30)
        _set_group(r, 0, [0.95, 0.93, mover])  # third in its group: group scores do not move
        _set_group(r, 1, [0.91, 0.89])
        _set_group(r, 2, [0.87, 0.85])
        _set_group(r, 3, [0.83, 0.81])
        _set_group(r, 4, [0.60, 0.58])
        _set_group(r, 5, [0.59, 0.57])
        _set_group(r, 6, [0.56, 0.55])
        _set_group(r, 7, [0.54, 0.53])
        rows.append(r)
    return dict(
        name="expert_boundary",
        rows=("mover_below", "mover_above"),
        logits=_logits_for_scores(rows),
        bias=torch.zeros(EXPERTS),
        intent="expert 2 moves 0.79 -> 0.82 across the 8th/9th boundary and expert 97 (0.81) "
        "drops out; the selected groups do not change",
    )


EXCLUSION_SPIKE = 7 * PER_GROUP + 5


def _exclusion() -> dict:
    logits = [0.0005 * ((e * 37) % EXPERTS) for e in range(EXPERTS)]
    logits[EXCLUSION_SPIKE] = 6.0
    bias = []
    for e in range(EXPERTS):
        group = e // PER_GROUP
        if group <= 3:
            bias.append(-2.0)
        elif group <= 6:
            bias.append(-5.0)
        else:
            bias.append(0.0 if e == EXCLUSION_SPIKE else -10.0)
    return dict(
        name="exclusion",
        rows=("spike_in_excluded_group",),
        logits=torch.tensor([logits], dtype=torch.float32),
        bias=torch.tensor(bias, dtype=torch.float32),
        intent=f"expert {EXCLUSION_SPIKE} has the highest selection score but its group loses; "
        "every retained selection score is negative, so skipping the exclusion or masking with "
        "zero instead of -inf both select an excluded expert",
    )


def _ties() -> dict:
    def top7() -> list:
        r = _row(0.30)
        _set_group(r, 0, [0.95, 0.94])
        _set_group(r, 1, [0.93, 0.92])
        _set_group(r, 2, [0.91, 0.90])
        _set_group(r, 3, [0.89, 0.60])
        for g in range(4, 8):
            _set_group(r, g, [0.50 - 0.01 * g, 0.49 - 0.01 * g])
        return r

    rows, names = [], []
    r = top7()
    r[1 * PER_GROUP + 7] = 0.70
    r[1 * PER_GROUP + 25] = 0.70
    rows.append(r)
    names.append("expert_boundary_tie_same_group_39_57")
    r = top7()
    r[0 * PER_GROUP + 3] = 0.70
    r[3 * PER_GROUP + 30] = 0.70
    rows.append(r)
    names.append("expert_boundary_tie_distant_3_126")
    r = _row(0.30)
    _set_group(r, 0, [0.95, 0.94])
    r[1 * PER_GROUP + 1] = 0.92
    r[1 * PER_GROUP + 30] = 0.92
    _set_group(r, 2, [0.91, 0.90])
    _set_group(r, 3, [0.89, 0.88])
    for g in range(4, 8):
        _set_group(r, g, [0.50 - 0.01 * g, 0.49 - 0.01 * g])
    rows.append(r)
    names.append("expert_order_tie_33_62")
    r = _row(0.30)
    _set_group(r, 0, [0.95, 0.94])
    _set_group(r, 1, [0.93, 0.92])
    _set_group(r, 2, [0.91, 0.90])
    _set_group(r, 3, [0.80, 0.78])
    _set_group(r, 6, [0.80, 0.78])
    _set_group(r, 4, [0.50, 0.49])
    _set_group(r, 5, [0.48, 0.47])
    _set_group(r, 7, [0.46, 0.45])
    rows.append(r)
    names.append("group_boundary_tie_3_6")
    r = _row(0.30)
    _set_group(r, 0, [0.95, 0.94])
    _set_group(r, 1, [0.90, 0.89])
    _set_group(r, 4, [0.90, 0.89])
    _set_group(r, 2, [0.85, 0.84])
    _set_group(r, 3, [0.50, 0.49])
    _set_group(r, 5, [0.48, 0.47])
    _set_group(r, 6, [0.46, 0.45])
    _set_group(r, 7, [0.44, 0.43])
    rows.append(r)
    names.append("group_order_tie_1_4")
    rows.append(_row(0.5))
    names.append("all_equal")
    r = _row(0.30)
    _set_group(r, 0, [0.95, 0.94])
    _set_group(r, 1, [0.93, 0.92])
    _set_group(r, 2, [0.91, 0.90])
    _set_group(r, 3, [0.89])
    for g, js in ((0, (5, 17, 31)), (1, (2, 28)), (2, (9, 10, 20)), (3, (4, 30))):
        for j in js:
            r[g * PER_GROUP + j] = 0.70
    for g in range(4, 8):
        _set_group(r, g, [0.50 - 0.01 * g, 0.49 - 0.01 * g])
    rows.append(r)
    names.append("expert_boundary_tie_10way")
    return dict(
        name="ties",
        rows=tuple(names),
        logits=_logits_for_scores(rows),
        bias=torch.zeros(EXPERTS),
        intent="bitwise-equal scores at the 8th/9th expert boundary, inside the top-8 order, at "
        "the 4th/5th group boundary, inside the top-4 group order, everywhere, and a 10-way "
        "boundary tie",
    )


#: The order `torch.topk(..., sorted=True)` returns for each tie row -- the source router's
#: order, measured on sm_103. A tie at a selection boundary keeps the lower index; tied
#: values inside the selection are not in index order.
TIE_ROUTES = {
    "expert_boundary_tie_same_group_39_57": ([0, 1, 2, 3], [0, 1, 32, 33, 64, 65, 96, 39]),
    "expert_boundary_tie_distant_3_126": ([0, 1, 2, 3], [0, 1, 32, 33, 64, 65, 96, 3]),
    "expert_order_tie_33_62": ([0, 1, 2, 3], [0, 1, 33, 62, 64, 65, 96, 97]),
    "group_boundary_tie_3_6": ([0, 1, 2, 3], [0, 1, 32, 33, 64, 65, 96, 97]),
    "group_order_tie_1_4": ([0, 4, 1, 2], [0, 1, 128, 32, 33, 129, 64, 65]),
    "all_equal": ([1, 0, 2, 3], [7, 6, 4, 5, 1, 0, 2, 3]),
    "expert_boundary_tie_10way": ([0, 1, 2, 3], [0, 1, 32, 33, 64, 65, 96, 5]),
}

EPSILON_IDS = [0, 1, 32, 33, 64, 65, 96, 97]


def _epsilon() -> dict:
    def bias_of(e: int) -> float:
        group, j = divmod(e, PER_GROUP)
        return 0.9 - 0.1 * group - 0.001 * j if j < 2 else -0.5 - 0.01 * group - 0.0001 * j

    zero_row = [-120.0] * EXPERTS
    near_row = [-60.0] * EXPERTS
    for k, e in enumerate(EPSILON_IDS):
        near_row[e] = -47.5 - 0.15 * k
    return dict(
        name="epsilon",
        rows=("zero_sum", "near_epsilon_sum"),
        logits=torch.tensor([zero_row, near_row], dtype=torch.float32),
        bias=torch.tensor([bias_of(e) for e in range(EXPERTS)], dtype=torch.float32),
        intent="the bias fixes the selection to experts 0, 1, 32, 33, 64, 65, 96, 97; row 0's "
        "scores underflow to 0 (weights 0, not NaN), row 1's selected scores sum to ~1e-20, so "
        "its weights sum visibly below 1 with exactly the 1e-20 denominator term",
    )


def fixtures() -> dict:
    """Every controlled fixture by name, in a fixed order."""
    specs = [*_bias_fixtures(), _group_boundary(), _expert_boundary(), _exclusion(), _ties()]
    specs.append(_epsilon())
    return {spec["name"]: spec for spec in specs}


def fixture_tensors(spec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(weight, correction bias, hidden) on CPU that make the router see `spec`'s logits."""
    logits = spec["logits"].float()
    rows = logits.shape[0]
    weight = matrix_case(17, 1)[0].clone()
    weight[:, :rows] = logits.t()
    hidden = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    hidden[torch.arange(rows), torch.arange(rows)] = 1.0
    return weight.contiguous(), spec["bias"].float().contiguous(), hidden


# ----------------------------------------------------------------------------- comparison


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float) -> dict:
    """Score one fp32 tensor against `atol` (rtol 0); non-finite values must match exactly."""
    result = {"tensor": name, "atol": atol, "rtol": 0.0, "shape": list(expected.shape)}
    problems = []
    if tuple(actual.shape) != tuple(expected.shape):
        problems.append(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.dtype != torch.float32 or expected.dtype != torch.float32:
        problems.append(f"dtypes {actual.dtype}/{expected.dtype}, need torch.float32")
    if not problems:
        a, e = actual.detach().double(), expected.detach().double()
        finite_a, finite_e = torch.isfinite(a), torch.isfinite(e)
        nonfinite_mismatch = int(((finite_a != finite_e) | (~finite_e & (a != e))).sum())
        both = finite_a & finite_e
        diff = (a[both] - e[both]).abs()
        violations = int((diff > atol).sum())
        result.update(
            max_abs=float(diff.max()) if diff.numel() else 0.0,
            unequal=int((diff > 0).sum()),
            violations=violations,
            nonfinite_mismatch=nonfinite_mismatch,
            numel=int(e.numel()),
        )
        if violations:
            problems.append(f"{violations} elements exceed atol {atol:g} (max {diff.max():.3e})")
        if nonfinite_mismatch:
            problems.append(f"{nonfinite_mismatch} non-finite positions differ")
    result["problems"] = problems
    result["passed"] = not problems
    return result


def compare_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Ordered integer ids: equal element for element, never sorted first."""
    same = tuple(actual.shape) == tuple(expected.shape) and actual.dtype == expected.dtype
    rows = (
        (actual == expected).all(dim=-1)
        if same
        else torch.zeros(expected.shape[0], dtype=torch.bool, device=expected.device)
    )
    mismatched = [int(i) for i in torch.nonzero(~rows).flatten().cpu()]
    return {
        "tensor": name,
        "rule": "exact ordered equality",
        "shape": list(expected.shape),
        "dtype": str(actual.dtype),
        "mismatched_rows": mismatched,
        "passed": same and not mismatched,
        "problems": [] if same and not mismatched else [f"rows {mismatched[:8]} differ"],
    }


def compare_routes(actual: dict, expected: dict, label: str) -> list:
    """Every step of a route against another route of the same input."""
    results = [
        compare_exact("groups", actual["groups"], expected["groups"]),
        compare_exact("ids", actual["ids"], expected["ids"]),
    ]
    for name, atol in RULES.items():
        if name == "weights":
            continue
        results.append(compare(name, actual[name], expected[name], atol))
    results.append(
        compare(
            "masked_selection_scores",
            actual["masked_selection_scores"],
            expected["masked_selection_scores"],
            RULES["selection_scores"],
        )
    )
    # Weights belong to the experts they weigh: compared on rows routed identically.
    same_ids = (actual["ids"] == expected["ids"]).all(dim=-1)
    results.append(
        compare(
            "weights", actual["weights"][same_ids], expected["weights"][same_ids], RULES["weights"]
        )
    )
    for r in results:
        r["against"] = label
    return results


def assert_all_pass(results: list) -> None:
    failed = [
        f"{r['against']}:{r['tensor']}: {'; '.join(r['problems'])}"
        for r in results
        if not r["passed"]
    ]
    assert not failed, "\n".join(failed)
