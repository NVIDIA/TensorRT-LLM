# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL candidates for the gate projection.

One kernel, three precisions. The only knob is how many BF16 weight terms to
carry: one is BF16-grade and shows what precision costs, two is the drop-in, and
three checks that two leaves no accuracy on the table.
"""

from __future__ import annotations

from typing import Callable

import torch

from ._repo_import import import_bare
from .baselines import Candidate

_KERNELS = "tensorrt_llm._torch.cute_dsl_kernels.blackwell"


def _runner():
    return import_bare(f"{_KERNELS}.minimax_m3_gate_gemm_runner")


def _build(terms: int, tactic=None) -> Callable:
    def build(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
        mod = _runner()
        # The weight is frozen, so the split belongs at load time and is hoisted
        # out of the timed region.
        w_split = mod.split_weight(w, terms)
        return lambda: mod.gate_gemm(x, w_split, terms, tactic)

    return build


def _build_tf32(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    """The same kernel driven at TF32, for the precision per microsecond question.

    An FP32 activation reaches the TF32 tensor cores through the TMA descriptor's
    `internal_type`, so this is one pass at ten mantissa bits against the BF16
    path's two passes at sixteen. The cast is inside the timed region because TF32
    cannot avoid it, which is the asymmetry the BF16 path exploits.
    """
    mod = _runner()
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device=x.device)

    def run() -> torch.Tensor:
        mod.GateGemmRunner.run(x.to(torch.float32), w, out)
        return out

    return run


def cute_candidates(terms: tuple[int, ...] = (1, 2, 3)) -> list[Candidate]:
    cands = [Candidate(f"cute gate gemm, {t} term{'s' if t > 1 else ''}", _build(t)) for t in terms]
    cands.append(Candidate("cute tf32 (needs the cast)", _build_tf32))
    return cands


def tactics() -> list:
    """Tile and cluster shapes worth trying for an N=256, K=6144 GEMM.

    Two axes matter and both push the same way. A tile covering all of N reads the
    activation once instead of once per column block, and a tall tile means fewer
    CTAs re-reading the 3MB weight out of L2: with a 64-row tile at 16k tokens
    that re-read is 800MB, several times the activation itself. Clustering along M
    multicasts the weight to the CTAs sharing it, attacking the same traffic from
    the other side.
    """
    return [
        (use_2cta, tiler, cluster)
        for use_2cta in (False, True)
        for tiler in (
            (64, 128),
            (128, 128),
            (256, 128),
            (64, 256),
            (128, 256),
            (256, 256),
        )
        for cluster in ((1, 1), (2, 1), (4, 1), (8, 1), (1, 2), (2, 2), (4, 2))
    ]


#: K partitions worth trying. The kernel keeps getting faster all the way to 32,
#: but the partials have to be reduced afterwards, so the useful range is set by
#: where that reduction starts costing more than the mainloop saves.
SPLIT_K = (1, 2, 4, 8, 16, 32)


def tune(x: torch.Tensor, w: torch.Tensor, terms: int, time_fn) -> list[tuple[float, object]]:
    """Time every tactic at one token count. Returns sorted (micros, tactic).

    The tile shape and the K partition count are swept together because they
    interact: a tile covering all of N lets the epilogue fold the weight terms,
    but it also concentrates the weight into fewer CTAs, which is what splitting K
    is there to undo.

    Folding and non-folding tile shapes are both included. Folding is the better
    deal wherever it is available, but below roughly a thousand tokens the narrow
    tile wins even while paying an extra pass over the output, and excluding
    either one hides a crossover.
    """
    mod = _runner()
    w_split = mod.split_weight(w, terms)
    results = []
    for tactic in tactics():
        for split_k in SPLIT_K:
            if not mod.split_k_is_supported(x.shape[1], split_k, x.dtype):
                continue
            try:
                fn = lambda t=tactic, s=split_k: mod.gate_gemm(  # noqa: E731
                    x, w_split, terms, t, split_k=s
                )
                fn()
                torch.cuda.synchronize()
                results.append((time_fn(fn), (*tactic, split_k)))
            except Exception:  # noqa: BLE001 - most tactics are invalid for a given shape
                continue
    return sorted(results)
