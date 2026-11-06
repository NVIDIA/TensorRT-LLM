# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain-Torch reference for the MoE feed-forward of Beam decoder block 3.

An independent restatement of the source MoE for generated-input comparison:

* Routing, every step fp32 with TF32 off: logits ``x.float() @ gate.weight^T``, sigmoid
  scores, and selection scores = scores + ``gate.e_score_correction_bias``, which only
  choose. Each of the 8 groups of 32 experts is scored by the sum of its two largest
  selection scores; the 4 best groups are kept, every expert outside them is excluded,
  and the 8 best remaining selection scores are chosen in descending order. The mixing
  weights are the chosen experts' unbiased scores over their sum plus 1e-20.
* Each selected expert, in ascending id order, runs its bias-free bf16 SwiGLU on its
  tokens' rows in ascending token order: bf16 gate and up projections, fp32 SiLU and
  product, one bf16 rounding, bf16 down projection.
* Each expert output times its fp32 weight is rounded to bf16, the routes are summed per
  token in fp32, and the sum is scaled by the routed scaling factor 2.5 once and rounded
  to bf16.

Only torch is imported. Group exclusion is written as group membership; the two
selections stay ``topk(..., sorted=True)``, the source's own calls, because the order of
tied scores is whatever that call returns.

State and inputs are generated on the GPU. Every tensor draws from a generator seeded
with ``crc32(f"{seed}|{name}")``, so its values do not depend on which other tensors were
generated or in which order. A device generator's draws depend on the device and the
torch build: the bytes are reproducible there, not across them.
"""

from __future__ import annotations

import contextlib
import math
import zlib
from collections import OrderedDict
from typing import Callable, Iterator

import torch
import torch.nn.functional as F

HIDDEN = 5120
INTERMEDIATE = 2560
NUM_EXPERTS = 256
N_GROUP = 8
PER_GROUP = NUM_EXPERTS // N_GROUP
TOPK_GROUP = 4
TOP_K = 8
NORM_EPS = 1e-20
ROUTED_SCALING_FACTOR = 2.5
MAX_TOKENS = 129
SEEDS = (17, 41)
LENGTHS = (1, 17, 129)

#: Each expert's three tensors by name relative to ``experts.<e>.``, [out, in].
EXPERT_SHAPES = OrderedDict(
    [
        ("gate_proj.weight", (INTERMEDIATE, HIDDEN)),
        ("up_proj.weight", (INTERMEDIATE, HIDDEN)),
        ("down_proj.weight", (HIDDEN, INTERMEDIATE)),
    ]
)

STATE_RECIPE = {
    "seeding": "one torch.Generator on the device per call, re-seeded for every tensor with "
    "zlib.crc32(f'{seed}|{name}')",
    "gate.weight": "fp32 N(0, 1) / sqrt(5120)",
    "gate.e_score_correction_bias": "fp32 0.05 * N(0, 1)",
    "experts.<e>.gate_proj.weight, experts.<e>.up_proj.weight": "N(0, 1) / sqrt(5120), drawn "
    "in fp32 and rounded once to bf16",
    "experts.<e>.down_proj.weight": "N(0, 1) / sqrt(2560), drawn in fp32 and rounded once to bf16",
    "input": "N(0, 1) of shape [129, 5120] under the name 'input', drawn in fp32 and rounded "
    "once to bf16; a T-token case uses the first T rows",
}

#: ``record(name, value)``: receives the named intermediates of a call.
Record = Callable[[str, object], None]


# ----------------------------------------------------------------------------- numerics


@contextlib.contextmanager
def reference_math() -> Iterator[None]:
    """IEEE fp32 GEMMs (TF32 off), bf16 GEMMs without reduced-precision reductions and
    autocast off for the duration; the caller's settings are restored afterwards."""
    matmul = torch.backends.cuda.matmul
    saved_tf32 = matmul.allow_tf32
    saved_precision = getattr(matmul, "fp32_precision", None)
    saved_bf16 = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_tf32 = False
    if saved_precision is not None:
        # Both switches: with only the legacy one moved, fp32_precision stays "tf32".
        matmul.fp32_precision = "ieee"
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        # The legacy switch first: its setter also moves the newer one restored after it.
        matmul.allow_tf32 = saved_tf32
        if saved_precision is not None:
            matmul.fp32_precision = saved_precision
        matmul.allow_bf16_reduced_precision_reduction = saved_bf16


# ----------------------------------------------------------------------------- state and inputs


def tensor_seed(seed: int, name: str) -> int:
    """The generator seed of the tensor called ``name`` in case ``seed``."""
    return zlib.crc32(f"{seed}|{name}".encode())


def param_shapes() -> "OrderedDict[str, tuple[tuple[int, ...], torch.dtype]]":
    """The MoE's 770 tensors by checkpoint name relative to ``mlp.``, in checkpoint order:
    the router's fp32 weight and correction bias, then each expert's three bf16 matrices."""
    shapes = OrderedDict(
        [
            ("gate.weight", ((NUM_EXPERTS, HIDDEN), torch.float32)),
            ("gate.e_score_correction_bias", ((NUM_EXPERTS,), torch.float32)),
        ]
    )
    for expert_id in range(NUM_EXPERTS):
        for name, shape in EXPERT_SHAPES.items():
            shapes[f"experts.{expert_id}.{name}"] = (shape, torch.bfloat16)
    return shapes


def _draw(
    gen: torch.Generator, seed: int, name: str, shape: tuple[int, ...], device
) -> torch.Tensor:
    """Standard normal fp32 values of ``shape`` from ``gen`` re-seeded for ``name``."""
    gen.manual_seed(tensor_seed(seed, name))
    return torch.randn(shape, generator=gen, dtype=torch.float32, device=device)


def generate_state(
    seed: int, device: str | torch.device = "cuda"
) -> "OrderedDict[str, torch.Tensor]":
    """The seed's 770 tensors on ``device``, in checkpoint order (see `STATE_RECIPE`).

    Every matrix is scaled by 1 / sqrt(its input width), the correction bias by 0.05; the
    expert matrices are rounded once to bf16, the router stays fp32.
    """
    gen = torch.Generator(device=device)
    state: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, (shape, dtype) in param_shapes().items():
        scale = 0.05 if len(shape) == 1 else 1.0 / math.sqrt(shape[1])
        state[name] = (_draw(gen, seed, name, shape, device) * scale).to(dtype).contiguous()
    return state


def generate_input(
    seed: int, tokens: int = MAX_TOKENS, device: str | torch.device = "cuda"
) -> torch.Tensor:
    """``[tokens, 5120]`` bf16: the first ``tokens`` rows of the seed's 129-row input."""
    if not 1 <= tokens <= MAX_TOKENS:
        raise ValueError(f"tokens must be in 1..{MAX_TOKENS}, got {tokens}")
    gen = torch.Generator(device=device)
    rows = _draw(gen, seed, "input", (MAX_TOKENS, HIDDEN), device).to(torch.bfloat16)
    return rows[:tokens]


def state_problems(state: "dict[str, torch.Tensor]", seed: int) -> list[str]:
    """Exact accounting plus the properties the comparisons rely on: finite values, a bias
    that varies across experts, and no two expert matrices alike."""
    problems = []
    shapes = param_shapes()
    if list(state) != list(shapes):
        problems.append("names or order differ from param_shapes()")
    if len({tensor_seed(seed, name) for name in shapes}) != len(shapes):
        problems.append("two tensors share a generator seed")
    for name, (shape, dtype) in shapes.items():
        tensor = state.get(name)
        if tensor is None:
            continue
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            found = f"{tuple(tensor.shape)} {tensor.dtype}"
            problems.append(f"{name}: {found}, expected {shape} {dtype}")
        elif not bool(torch.isfinite(tensor).all()):
            problems.append(f"{name}: nonfinite")
    if problems:
        return problems
    bias = state["gate.e_score_correction_bias"]
    if bool((bias == bias[0]).all()):
        problems.append("the correction bias does not vary across experts")
    for width, names in (
        (HIDDEN, ("gate_proj.weight", "up_proj.weight")),
        (INTERMEDIATE, ("down_proj.weight",)),
    ):
        first_rows = torch.stack(
            [state[f"experts.{e}.{n}"][0].float() for e in range(NUM_EXPERTS) for n in names]
        )
        if torch.unique(first_rows, dim=0).shape[0] != first_rows.shape[0]:
            problems.append(f"two expert matrices of input width {width} share their first row")
    return problems


# ----------------------------------------------------------------------------- the MoE


def route(
    x: torch.Tensor, weight: torch.Tensor, correction_bias: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The source routing of ``x [T, 5120]`` bf16, every step fp32.

    Returns ``logits``, ``scores`` and ``selection`` ``[T, 256]``, ``group_scores``
    ``[T, 8]``, ``groups`` ``[T, 4]`` int64 (descending), ``masked`` ``[T, 256]`` (-inf
    outside the kept groups), ``ids`` ``[T, 8]`` int64 (descending selection score) and
    ``weights`` ``[T, 8]`` fp32.
    """
    with reference_math():
        logits = F.linear(x.float(), weight.float())
        scores = torch.sigmoid(logits)
        selection = scores + correction_bias.float()
        tokens = selection.shape[0]
        group_scores = selection.view(tokens, N_GROUP, PER_GROUP).topk(2, dim=-1).values.sum(dim=-1)
        groups = group_scores.topk(TOPK_GROUP, dim=-1, sorted=True).indices
        expert_group = torch.arange(NUM_EXPERTS, device=x.device) // PER_GROUP
        kept = (expert_group.view(1, NUM_EXPERTS, 1) == groups.unsqueeze(1)).any(dim=-1)
        masked = torch.where(kept, selection, torch.full_like(selection, float("-inf")))
        ids = masked.topk(TOP_K, dim=-1, sorted=True).indices
        chosen = torch.gather(scores, -1, ids)
        weights = chosen / (chosen.sum(dim=-1, keepdim=True) + NORM_EPS)
    return dict(
        logits=logits,
        scores=scores,
        selection=selection,
        group_scores=group_scores,
        groups=groups,
        masked=masked,
        ids=ids,
        weights=weights,
    )


def expert(
    rows: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One expert on ``rows [n, 5120]`` bf16, every step: ``gate``, ``up`` (bf16),
    ``silu``, ``product`` (fp32), ``act`` (bf16, the one rounding) and ``y`` ``[n, 5120]``
    bf16."""
    with reference_math():
        gate = F.linear(rows, gate_weight)
        up = F.linear(rows, up_weight)
        silu = F.silu(gate.float())
        product = silu * up.float()
        act = product.to(torch.bfloat16)
        y = F.linear(act, down_weight)
    return dict(gate=gate, up=up, silu=silu, product=product, act=act, y=y)


def moe(
    x: torch.Tensor, state: "dict[str, torch.Tensor]", record: Record | None = None
) -> torch.Tensor:
    """The source MoE of ``x [T, 5120]`` bf16; returns ``[T, 5120]`` bf16.

    ``state`` holds the 770 tensors by name relative to ``mlp.``. ``record``, when given,
    receives ``routing`` (the dict of `route`), ``expert_calls`` (``[(expert_id, tokens)]``
    in call order), ``accumulator`` (the fp32 sum before scaling) and ``output``.
    """
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == HIDDEN, (
        x.dtype,
        tuple(x.shape),
    )
    routing = route(x, state["gate.weight"], state["gate.e_score_correction_bias"])
    ids, weights = routing["ids"], routing["weights"]
    acc = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    calls = []
    for expert_id in sorted(set(ids.reshape(-1).tolist())):
        # Row-major nonzero: the expert's tokens in ascending order, each once.
        tokens, slots = (ids == expert_id).nonzero(as_tuple=True)
        matrices = [state[f"experts.{expert_id}.{name}"] for name in EXPERT_SHAPES]
        y = expert(x[tokens], *matrices)["y"]
        routed = (y.float() * weights[tokens, slots].unsqueeze(-1)).to(torch.bfloat16)
        acc[tokens] += routed.float()
        calls.append((expert_id, tokens))
    output = (acc * ROUTED_SCALING_FACTOR).to(torch.bfloat16)
    if record is not None:
        record("routing", routing)
        record("expert_calls", calls)
        record("accumulator", acc)
        record("output", output)
    return output
