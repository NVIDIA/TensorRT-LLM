# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter GEMM routes of the DSpark worker's prebuild: which kernel each drafter layer projection runs at the
decode row counts ("LowM" where the low-M table routes the row, "MXDRAFT" where the MXFP8 GEMV hook executes it, else
"cuBLAS", the Linear's own GEMM).

DRAFT_LOWM_GEMM (opt-in, ``TRTLLM_M31_ENABLE_DRAFT_LOWM_GEMM=1``; "LowM" in the routes report): drafter projections
whose (N, K, M) ``TABLE`` holds, at decode row counts the MXFP8 GEMV does not execute, run FlashInfer's BF16 low-M
split-K / direct kernel at a fixed tactic instead of cuBLAS (not bitwise to cuBLAS: a different reduction order).
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Mapping, Optional, Set, Tuple

import torch

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.logger import logger

from .drafter_mxfp8 import executed_rows
from .kernels import m3_report

#: (N, K, M) of a TP2 drafter shard at a decode row count -> (kind, tactic). Kind "s" is FlashInfer's split-K kernel
#: with tactic (mma_m, mma_n, split_k, ab_stages): per shape, the fastest deterministic tactic whose error against an
#: FP64 reference is no larger than cuBLAS's. qkv and gate_up have no entry (the MXFP8 GEMV takes those rows).
TABLE: Mapping[Tuple[int, int, int], Tuple[str, Tuple[int, ...]]] = {
    (6144, 4096, 7): ("s", (64, 8, 2, 12)),  # o_proj (32 heads of 128 per rank)
    (6144, 4096, 14): ("s", (64, 16, 2, 11)),  # o_proj
    (6144, 6144, 7): ("s", (64, 8, 2, 12)),  # down (6144 intermediate per rank)
    (6144, 6144, 14): ("s", (64, 16, 2, 11)),  # down
}

#: The drafter layer projections, by name, in the order every report lists them.
PROJECTIONS = ("qkv", "o_proj", "gate_up", "down")

_ENGAGED: Set[Tuple[str, int, int, int]] = set()


def run_lowm(
    kind: str,
    tactic: Tuple[int, ...],
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    use_pdl: bool,
) -> torch.Tensor:
    """``out`` [M, N] = ``x`` [M, K] @ ``weight`` [N, K]^T on FlashInfer's low-M kernel ``kind`` at ``tactic``."""
    if kind == "s":
        from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
            SplitKTactic,
            run_splitk_dense,
        )

        return run_splitk_dense(x, weight.t(), None, out, use_pdl, SplitKTactic(*tactic))
    if kind == "d":
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense

        return run_direct_dense(x, weight.t(), out, use_pdl, DirectTactic(*tactic))
    raise ValueError(f"unknown low-M kernel kind {kind!r}")


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


def _takes(x, m_served: Mapping[int, tuple]) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and int(x.shape[0]) in m_served
        and x.data_ptr() % 32 == 0
    )


def _oneshot_rows(lin) -> int:
    """The largest row count a row-parallel ``lin``'s all-reduce takes on the one-shot path (0: none)."""
    ar = getattr(lin, "all_reduce", None)
    if getattr(ar, "oneshot", None) is None:
        return 0
    return int(getattr(ar, "max_rows", 0) or 0)


def route_lowm_linears(
    draft_model,
    rows: Iterable[int],
    table: Optional[Mapping[Tuple[int, int, int], tuple]] = None,
    handoff: bool = True,
    rank: Optional[int] = None,
) -> int:
    """Route each drafter projection whose (N, K) the table holds at some of ``rows`` that MXDRAFT does not execute
    to the low-M kernel for those row counts; compiles and runs each routed call once now (outside capture). Returns
    how many projections were routed. ``handoff=False``: a projection MXDRAFT routed is left alone."""
    table = dict(TABLE) if table is None else table
    if not table:
        return 0
    rows = sorted({int(m) for m in rows if int(m) > 0})
    use_pdl = get_env_enable_pdl()
    n_routed = 0
    for _, name, lin in named_projections(draft_model):
        weight = getattr(lin, "weight", None)
        qm = getattr(lin, "quant_method", None)
        mx = getattr(lin, "_m31_mxdraft", None)
        if (
            weight is None
            or qm is None
            or getattr(lin, "bias", None) is not None
            or (mx is not None and not handoff)
            or getattr(lin, "_m31_t6", None) is not None
            or weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or not weight.is_contiguous()
            or weight.data_ptr() % 32
        ):
            continue
        n, k = (int(s) for s in weight.shape)
        mx_rows = set(executed_rows(lin)) if mx is not None else set()
        served = {m: table[(n, k, m)] for m in rows if (n, k, m) in table and m not in mx_rows}
        if getattr(lin, "tp_mode", None) == TensorParallelMode.ROW and getattr(
            lin, "reduce_output", False
        ):
            limit = _oneshot_rows(lin)
            served = {m: v for m, v in served.items() if m <= limit}
        if not served:
            continue
        for m, (kind, tactic) in served.items():
            x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
            out = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)
            run_lowm(kind, tactic, x, weight, out, use_pdl)
        original = qm.apply

        def apply(module, input, bias, *, _o=original, _lin=lin, _served=served, _pdl=use_pdl, _name=name,
                  _rank=rank):  # fmt: skip
            if module is _lin and bias is None and _takes(input, _served):
                w = _lin.weight
                m = int(input.shape[0])
                kind, tactic = _served[m]
                key = (_name, int(w.shape[0]), int(w.shape[1]), m)
                if key not in _ENGAGED:
                    _ENGAGED.add(key)
                    logger.info(
                        f"[M3 sm_107 tp2] fast path engaged: DRAFT_LOWM_GEMM drafter low-M {_name} "
                        f"{key[1]}x{key[2]} at {m} rows "
                        f"({kind} {tactic}) on rank {_rank} (device {input.device.index})"
                    )
                m3_report.work("DRAFT_LOWM_GEMM", f"{_name} {key[1]}x{key[2]} at {m} rows")
                out = torch.empty((m, int(w.shape[0])), dtype=input.dtype, device=input.device)
                return run_lowm(kind, tactic, input, w, out, _pdl)
            return _o(module, input, bias)

        qm.apply = apply
        lin._m31_t6 = served
        n_routed += 1
    return n_routed


def route_of(lin, m: int) -> str:
    """The kernel a decode call of ``m`` rows of ``lin`` runs: "LowM", "MXDRAFT" (its hook executes the row) or "cuBLAS"
    (the Linear's own GEMM). LowM is checked first: it is installed below MXDRAFT and only on rows MXDRAFT does not
    execute."""
    if m in (getattr(lin, "_m31_t6", None) or {}):
        return "LowM"
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
