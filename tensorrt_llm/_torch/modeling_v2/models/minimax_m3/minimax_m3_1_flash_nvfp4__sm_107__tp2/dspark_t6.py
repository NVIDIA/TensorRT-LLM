# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter GEMM routes report of the DSpark worker's prebuild: which kernel each drafter layer projection runs at
the decode row counts ("MXDRAFT" where the MXFP8 GEMV hook executes the row, else "cuBLAS", the Linear's own GEMM).
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple

import torch

from .drafter_mxfp8 import executed_rows

#: The drafter layer projections, by name, in the order every report lists them.
PROJECTIONS = ("qkv", "o_proj", "gate_up", "down")


def named_projections(draft_model) -> Iterator[Tuple[int, str, torch.nn.Module]]:
    """(layer index, projection name, Linear) of every drafter layer projection (the ones MXDRAFT routes)."""
    for i, layer in enumerate(getattr(getattr(draft_model, "model", None), "layers", []) or []):
        attn, mlp = getattr(layer, "self_attn", None), getattr(layer, "mlp", None)
        for name, lin in zip(
            PROJECTIONS,
            (
                getattr(attn, "qkv_proj", None),
                getattr(attn, "o_proj", None),
                getattr(mlp, "gate_up_proj", None),
                getattr(mlp, "down_proj", None),
            ),
        ):
            if lin is not None:
                yield i, name, lin


def route_of(lin, m: int) -> str:
    """The kernel a decode call of ``m`` rows of ``lin`` runs: "MXDRAFT" (its hook executes the row) or "cuBLAS" (the
    Linear's own GEMM)."""
    if m in executed_rows(lin):
        return "MXDRAFT"
    return "cuBLAS"


def drafter_gemm_routes(draft_model, rows: Iterable[int]) -> Dict[str, Dict[int, str]]:
    """{projection: {rows: kernel}} over the drafter layers ("mixed" where the layers disagree)."""
    rows = sorted({int(m) for m in rows if int(m) > 0})
    routes: Dict[str, Dict[int, str]] = {}
    for _, name, lin in named_projections(draft_model):
        per = routes.setdefault(name, {})
        for m in rows:
            got = route_of(lin, m)
            per[m] = got if per.get(m, got) == got else "mixed"
    return routes


def routes_line(draft_model, rows: Iterable[int], rank) -> str:
    """The per-rank engagement line of the drafter GEMM routing, naming each projection and row count."""
    routes = drafter_gemm_routes(draft_model, rows)
    layers = len({i for i, _, _ in named_projections(draft_model)})
    if not routes:
        return f"M3.1 DSpark DRAFTGEMM: no drafter layer projections on rank {rank}"
    body = "; ".join(
        f"{name} " + " ".join(f"m{m} {k}" for m, k in sorted(routes[name].items()))
        for name in PROJECTIONS
        if name in routes
    )
    return f"M3.1 DSpark DRAFTGEMM: drafter GEMM routes on rank {rank}: {body} ({layers} layers)"
