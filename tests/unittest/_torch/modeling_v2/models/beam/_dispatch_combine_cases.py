# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated inputs, controlled experts and the plain-torch golden of the dispatch/combine tests.

Block 3's MoE combination: hidden 5120, 256 experts, eight per token, routed scaling factor
2.5. The experts here are controlled functions rather than matrices: expert ``e`` maps a
BF16 row ``x`` to ``bf16(fp32(x) * A[e] + B[e])`` with fixed FP32 tables ``A`` and ``B``
``[256, 5120]``, so every output depends on the expert's identity and on the gathered
token's values, and a row sent to the wrong expert or taken from the wrong token changes
it. Every tensor is generated on the host from a seed, so the tests need no files.
"""

from __future__ import annotations

import torch

HIDDEN, EXPERTS, TOP_K, SCALE = 5120, 256, 8, 2.5
SEEDS = (17, 41)
TOKENS = (1, 17, 129)
BF16, FP32, I64 = torch.bfloat16, torch.float32, torch.int64
PROVIDER_SEED = 20261019
PERM_SEED = 20261020
FIXTURE_SEED = 20261021


class AffineExperts:
    """Expert ``e``: ``rows -> bf16(fp32(rows) * A[e] + B[e])``, row-wise; every call logged.

    ``calls`` holds ``(expert_id, rows, output)`` per call, in call order. An empty batch is
    refused, so an expert called for no token fails the test that made the call.
    """

    def __init__(self, a: torch.Tensor, b: torch.Tensor):
        self.a, self.b = a, b
        self.calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def __call__(self, expert_id: int, rows: torch.Tensor) -> torch.Tensor:
        assert type(expert_id) is int, f"expert id must be a Python int, got {type(expert_id)}"
        assert rows.shape[0] > 0, f"expert {expert_id} called with an empty batch"
        out = (rows.float() * self.a[expert_id] + self.b[expert_id]).to(BF16)
        self.calls.append((expert_id, rows, out))
        return out

    def fresh(self) -> "AffineExperts":
        """The same experts with an empty call log."""
        return AffineExperts(self.a, self.b)


def affine_experts(device: str = "cuda", zero: bool = False) -> AffineExperts:
    """Signed coefficients |A| in [0.5, 1.5) and offsets B ~ 0.25 N(0, 1); all zero if asked."""
    if zero:
        return AffineExperts(*(torch.zeros(EXPERTS, HIDDEN, dtype=FP32, device=device),) * 2)
    g = torch.Generator(device="cpu").manual_seed(PROVIDER_SEED)
    magnitude = 0.5 + torch.rand(EXPERTS, HIDDEN, generator=g, dtype=FP32)
    negative = torch.rand(EXPERTS, HIDDEN, generator=g, dtype=FP32) < 0.5
    a = torch.where(negative, -magnitude, magnitude)
    b = 0.25 * torch.randn(EXPERTS, HIDDEN, generator=g, dtype=FP32)
    return AffineExperts(a.to(device), b.to(device))


def matrix_case(seed: int, tokens: int):
    """``(hidden, ids, weights)`` on the host: 129 generated tokens, the first ``tokens`` used.

    Hidden states are unit-variance BF16. Each row's ids are eight distinct experts in
    random (unsorted) order and its weights are normalized as the router normalizes them.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(max(TOKENS), HIDDEN, generator=g, dtype=FP32).to(BF16)
    ids = torch.stack([torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(max(TOKENS))])
    scores = 0.05 + torch.rand(max(TOKENS), TOP_K, generator=g, dtype=FP32)
    w = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    return x[:tokens].clone(), ids[:tokens].to(I64).clone(), w[:tokens].clone()


def permutation(seed: int, tokens: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(PERM_SEED + 1000 * seed + tokens)
    return torch.randperm(tokens, generator=g)


SHARED_IDS = [
    [255, 3, 0, 128, 254, 1, 77, 200],
    [0, 255, 9, 128, 31, 254, 2, 100],
    [7, 0, 255, 1, 64, 129, 250, 33],
    [254, 128, 0, 255, 5, 6, 11, 99],
    [1, 2, 3, 4, 255, 0, 253, 252],
    [128, 129, 130, 0, 131, 132, 255, 133],
]
WITNESS_IDS = [
    [12, 200, 57, 255, 0, 91, 143, 33],
    [200, 12, 255, 64, 18, 91, 222, 5],
    [0, 57, 143, 129, 200, 77, 255, 111],
    [91, 33, 0, 12, 250, 129, 64, 7],
]
WITNESS_BASE_WEIGHTS = [0.34, 0.21, 0.14, 0.11, 0.08, 0.06, 0.04, 0.02]
ZERO_IDS = [
    [0, 17, 255, 42, 128, 3, 200, 91],
    [255, 128, 7, 0, 64, 33, 250, 1],
    [42, 0, 91, 254, 17, 255, 5, 129],
]
#: Tokens 0, 1 and 2 give experts 255, 33 and 129 a weight of exactly zero.
ZERO_ROUTE_SLOT = [2, 5, 7]


def fixtures() -> dict:
    """The focused fixtures on the host, by name: ``(hidden, ids, weights, zero_experts)``.

    shared_extremes       six distinct tokens all sharing experts 0 and 255 (and 128 / 254 /
                          1 several times), unsorted slots, unequal weights that do not sum
                          to one, so a re-normalization would show
    cancellation_witness  signed routes of unequal size (weights 0.02 to 0.34, none a BF16
                          value), so each rounding and the accumulation order are visible
    zero_weight_route     one route per token with weight exactly 0
    all_zero_weights      every weight 0
    all_zero_outputs      ordinary weights; the zero experts return exact zeros
    """
    g = torch.Generator(device="cpu").manual_seed(FIXTURE_SEED)
    fx = {}
    x = torch.randn(6, HIDDEN, generator=g, dtype=FP32).to(BF16)
    w = 0.05 + 0.35 * torch.rand(6, TOP_K, generator=g, dtype=FP32)
    fx["shared_extremes"] = (x, torch.tensor(SHARED_IDS, dtype=I64), w, False)
    x = torch.randn(4, HIDDEN, generator=g, dtype=FP32).to(BF16)
    base = torch.tensor(WITNESS_BASE_WEIGHTS, dtype=FP32)
    rows = []
    for _ in range(4):
        perm = torch.randperm(TOP_K, generator=g)
        jitter = 1.0 + 0.01 * torch.randn(TOP_K, generator=g, dtype=FP32)
        rows.append(base[perm] * jitter)
    raw = torch.stack(rows)
    w = raw / (raw.sum(dim=-1, keepdim=True) + 1e-20)
    fx["cancellation_witness"] = (x, torch.tensor(WITNESS_IDS, dtype=I64), w, False)
    x = torch.randn(3, HIDDEN, generator=g, dtype=FP32).to(BF16)
    raw = 0.05 + torch.rand(3, TOP_K, generator=g, dtype=FP32)
    for t, j in enumerate(ZERO_ROUTE_SLOT):
        raw[t, j] = 0.0
    w = raw / (raw.sum(dim=-1, keepdim=True) + 1e-20)
    for t, j in enumerate(ZERO_ROUTE_SLOT):
        w[t, j] = 0.0
    fx["zero_weight_route"] = (x, torch.tensor(ZERO_IDS, dtype=I64), w, False)
    x = torch.randn(3, HIDDEN, generator=g, dtype=FP32).to(BF16)
    fx["all_zero_weights"] = (x, torch.tensor(ZERO_IDS, dtype=I64), torch.zeros(3, TOP_K), False)
    x = torch.randn(3, HIDDEN, generator=g, dtype=FP32).to(BF16)
    raw = 0.05 + torch.rand(3, TOP_K, generator=g, dtype=FP32)
    w = raw / (raw.sum(dim=-1, keepdim=True) + 1e-20)
    fx["all_zero_outputs"] = (x, torch.tensor(ZERO_IDS, dtype=I64), w, True)
    return fx


#: Wrong answers the golden can produce on request, one step changed each.
FAULTS = (
    "wrong_scatter_mapping",  # contributions added to token (t + 1) mod T
    "no_route_cast",  # weighted route kept fp32
    "bf16_product",  # weight rounded to bf16, product in bf16
    "cast_after_accumulate",  # fp32 routes summed, the sum rounded to bf16 before scaling
    "bf16_accumulator",  # accumulator held in bf16
    "overwrite",  # a contribution replaces the accumulator row instead of adding
    "weights_slot_rotated",  # each route takes the next slot's weight
    "renormalized_weights",  # weights re-normalized per token
    "scale_omitted",
    "scale_duplicated",
)


def golden(x, ids, w, experts, fault: str | None = None, skip_slots=None) -> dict:
    """The source combination in plain torch, every step, on the inputs' device.

    Written independently of the component: every expert id 0..255 is enumerated, each
    expert's (token, slot) pairs come from the routing table by ``nonzero``, and each
    expert's rounded routes are added to their tokens' rows of an fp32 accumulator
    (``acc[t] += route``; a token appears at most once per expert). Returns the stages
    ``product`` / ``route`` (assignment rows in call order), ``acc`` (before scaling),
    ``scaled`` and ``output``, plus ``calls`` = ``[(expert, tokens)]``. ``skip_slots[t]``
    drops slot ``skip_slots[t]`` of token ``t`` from the table.
    """
    tokens = x.shape[0]
    wk = w / (w.sum(dim=-1, keepdim=True) + 1e-20) if fault == "renormalized_weights" else w
    acc = torch.zeros(
        tokens, HIDDEN, dtype=BF16 if fault == "bf16_accumulator" else FP32, device=x.device
    )
    stages = {"product": [], "route": [], "calls": []}
    for e in range(EXPERTS):
        token_slot = (ids == e).nonzero()
        if skip_slots is not None:
            keep = [slot != skip_slots[token] for token, slot in token_slot.tolist()]
            token_slot = token_slot[torch.tensor(keep, dtype=torch.bool, device=ids.device)]
        if token_slot.shape[0] == 0:
            continue
        t, j = token_slot[:, 0], token_slot[:, 1]
        out = experts(e, x[t])
        jw = (j + 1) % TOP_K if fault == "weights_slot_rotated" else j
        weight = wk[t, jw]
        if fault == "bf16_product":
            product = out * weight.to(BF16).unsqueeze(1)
            route = product
        else:
            product = out.float() * weight.unsqueeze(1)
            route = (
                product if fault in ("no_route_cast", "cast_after_accumulate") else product.to(BF16)
            )
        dst = (t + 1) % tokens if fault == "wrong_scatter_mapping" else t
        if fault == "overwrite":
            acc[dst] = route.float()
        elif fault == "bf16_accumulator":
            acc[dst] = (acc[dst].float() + route.float()).to(BF16)
        else:
            acc[dst] += route.float()
        stages["product"].append(product)
        stages["route"].append(route)
        stages["calls"].append((e, t))
    acc = acc.float()
    if fault == "cast_after_accumulate":
        acc = acc.to(BF16).float()
    scaled = {"scale_omitted": acc, "scale_duplicated": acc * SCALE * SCALE}.get(fault, acc * SCALE)
    stages.update(acc=acc, scaled=scaled, output=scaled.to(BF16))
    stages["product"] = torch.cat(stages["product"])
    stages["route"] = torch.cat(stages["route"])
    return stages


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Same dtype, shape and bits (so -0.0 differs from +0.0)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {BF16: torch.int16, FP32: torch.int32}.get(a.dtype)
    if view is None:
        return bool(torch.equal(a, b.to(a.device)))
    return bool(torch.equal(a.contiguous().view(view), b.to(a.device).contiguous().view(view)))
