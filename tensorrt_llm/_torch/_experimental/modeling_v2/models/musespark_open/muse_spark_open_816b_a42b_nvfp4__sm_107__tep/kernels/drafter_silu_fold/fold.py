# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter MLP's ``silu(gate) * up`` in the ``gate_up`` GEMM's epilogue (``tep8_drafter_wide_gemms__SILU_FOLD``).

Every drafter layer's ``gate_up_proj -> silu_and_mul -> down_proj`` is three launches; on the 8..64-row blocks
(1..8 requests at K = 7) the activation's launch (1.2 us) and its dependency boundary cost about as much again
(vr_foldw_bench on VR: the GEMM + ``drafter_silu_and_mul`` chain 15.0 us vs the fold 12.5 us at 32 rows). With this
fast path the gate_up GEMM's epilogue applies the activation (``ops/gemm/silu_fold_gemm``, on this target's
split-K kernel: every 8..64-row tactic of the table, 64-wide tiles and deep stages included).

Install (``drafter.py``, after ``drafter_pdl_chain``, whose ``EntryTriggerGatedMLP`` forward consults the fold, and
after ``drafter_fixed_gemms``, whose ``gate_up`` tactics it folds): each layer's ``gate_up_proj`` weight rows are
interleaved IN PLACE per the tactics' kernel-M tile (``interleave_gate_up``: 32 gate rows, then the matching 32 up
rows, at mma_m 64), then every MLP forward runs, by rows:

* the table lists a ``gate_up`` tactic at these rows (``fold_tactic`` keeps it, stages lowered to fit the staging
  buffer): ``silu_fold_gemm``, then ``down_proj``;
* any other row count (65..128 rows: cuBLAS on the interleaved weight): ``gate_up_proj``, then
  ``drafter_silu_and_mul(.., half=32)`` on the interleaved columns, then ``down_proj``.

Bits: a row permutation of the weight changes no column's K-ordered sum; the fold's epilogue is the activation
kernel's statements on the BF16-rounded GEMM values, and the interleaved activation kernel the same statements on
the same values: the MLP's output is bit for bit today's (the tests gate it with ``torch.equal``). A call of a row
count the table lists that the fold does not take logs ERROR naming the knob (the second path runs; same values).
``SILU_FOLD`` off (the default) installs nothing.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from tensorrt_llm.logger import logger

#: ``silu_fold`` (``configure``; ``tep8_drafter_wide_gemms__SILU_FOLD``).
SILU_FOLD = False
#: The MLP attribute the PDL chain's ``EntryTriggerGatedMLP.forward`` reads.
ATTRIBUTE = "_silu_fold"
#: The fold's rows: the table's 8..64-row ``gate_up`` entries.
MIN_ROWS, MAX_ROWS = 8, 64


def configure(*, silu_fold: bool = False) -> None:
    """Set ``SILU_FOLD``; ``lever_settings.apply`` sets it before any install."""
    global SILU_FOLD
    SILU_FOLD = bool(silu_fold)


class SiluFold:
    """One drafter MLP's fold: the interleave tile, the folded tactic per row count, the logged refusals."""

    def __init__(self, mma_m: int, tactics: dict, version: int = 0, fingerprint=None) -> None:
        self.mma_m = mma_m
        #: A copy of the interleaved weight's first tile's up rows' first columns (``FINGERPRINT``): an install
        #: hook after a reload (``weight.data`` rewritten in the plain order, which ``_version`` does not see)
        #: finds them changed and interleaves again.
        self.fingerprint = fingerprint
        #: The gate_up weight's ``_version`` right after the interleave: a later in-place write (a weight reload)
        #: would hand the fold plain rows, which ``folded_forward`` refuses.
        self.version = version
        self.half = mma_m // 2
        #: rows -> the ``fold_tactic`` of the table's gate_up entry at those rows.
        self.tactics = dict(tactics)
        self.calls = {"fold": 0, "plain": 0}
        self._refused = set()

    def tactic(self, rows: int):
        return self.tactics.get(rows)

    def refused(self, rows: int, why: str) -> None:
        if rows not in self._refused:
            self._refused.add(rows)
            logger.error(
                f"[musespark_open] tep8_drafter_wide_gemms__SILU_FOLD: the {rows}-row drafter MLP did not take the "
                f"fold ({why}): gate_up + the interleaved silu_and_mul run instead (the same values)"
            )


def _gate_up_entries(linear: nn.Module, k: int, n: int) -> dict:
    method = getattr(linear, "quant_method", None)
    table = getattr(method, "tactics", None) or {}
    return {
        m: t
        for (m, kk, nn_), t in table.items()
        if kk == k and nn_ == n and MIN_ROWS <= m <= MAX_ROWS
    }


def unsupported_reason(draft_model: Optional[nn.Module]) -> Optional[str]:
    """Why the fold cannot be installed on ``draft_model``, or None."""
    from ...ops.gemm.silu_fold_gemm import fold_tactic
    from ..drafter_pdl_chain.chain import EntryTriggerGatedMLP

    if draft_model is None:
        return "no drafter"
    layers = getattr(getattr(draft_model, "model", None), "layers", None)
    if not layers:
        return "the drafter has no decoder layers"
    for index, layer in enumerate(layers):
        mlp = layer.mlp
        where = f"layer {index} mlp"
        if type(mlp) is not EntryTriggerGatedMLP:
            return f"{where} is {type(mlp).__name__}, not the PDL chain's MLP (select drafter_pdl_chain)"
        if "_dr3_sf" in mlp.__dict__:
            return f"{where} already runs drafter_folds' silu fold"
        linear = mlp.gate_up_proj
        weight = linear.weight
        if linear.bias is not None or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            return f"{where} gate_up_proj is not a bias-free contiguous BF16 projection"
        if type(getattr(linear, "quant_method", None)).__name__ != "FixedTacticLinearMethod":
            return f"{where} gate_up_proj is not on drafter_fixed_gemms' tactics"
        n, k = weight.shape
        entries = _gate_up_entries(linear, k, n)
        if not entries:
            return f"{where} gate_up_proj has no 8..64-row tactic"
        tiles = {t[0] for t in entries.values()}
        if len(tiles) != 1:
            return f"{where} gate_up_proj's 8..64-row tactics mix kernel-M tiles {sorted(tiles)} (one interleave)"
        mma_m = tiles.pop()
        if n % (2 * mma_m):
            return f"{where} gate_up_proj's {n} rows are not a multiple of 2 x {mma_m}"
        for rows, tactic in sorted(entries.items()):
            if fold_tactic(tactic) is None:
                return f"{where}: the {rows}-row tactic {tuple(tactic)} has no fold (split_k > 1 or no stage fits)"
    return None


#: Columns of the fingerprint (``SiluFold.fingerprint``).
FINGERPRINT = 16


def _fingerprint(weight: torch.Tensor, mma_m: int) -> torch.Tensor:
    """Row ``mma_m / 2`` (the first tile's first up row once interleaved; a gate row in the plain order)."""
    return weight.detach()[mma_m // 2, :FINGERPRINT].clone()


def _still_interleaved(mlp: nn.Module, fold: "SiluFold") -> bool:
    weight = mlp.gate_up_proj.weight
    return fold.fingerprint is None or torch.equal(
        _fingerprint(weight, fold.mma_m), fold.fingerprint
    )


def install(draft_model: Optional[nn.Module]) -> int:
    """Interleave every drafter ``gate_up_proj`` weight in place and attach the fold; return how many changed.
    Nothing at ``SILU_FOLD`` off. Idempotent.

    Raises:
        ValueError: naming ``tep8_drafter_wide_gemms__SILU_FOLD`` when ``unsupported_reason`` refuses.
    """
    if not SILU_FOLD:
        return 0
    layers = list(getattr(getattr(draft_model, "model", None), "layers", None) or ())
    for layer in layers:
        fold = layer.mlp.__dict__.get(ATTRIBUTE)
        if isinstance(fold, SiluFold) and not _still_interleaved(layer.mlp, fold):
            # A load hook after a reload: the weight is in the plain order again.
            logger.info(
                "[musespark_open] drafter_silu_fold: a drafter gate_up weight was reloaded; interleaving it again "
                "(tep8_drafter_wide_gemms__SILU_FOLD)"
            )
            del layer.mlp.__dict__[ATTRIBUTE]
    if layers and all(isinstance(layer.mlp.__dict__.get(ATTRIBUTE), SiluFold) for layer in layers):
        return 0
    reason = unsupported_reason(draft_model)
    if reason is not None:
        raise ValueError(
            f"[musespark_open] tep8_drafter_wide_gemms__SILU_FOLD cannot run here: {reason}"
        )
    from ...ops.gemm.silu_fold_gemm import fold_tactic, interleave_gate_up

    changed = 0
    for layer in layers:
        mlp = layer.mlp
        if isinstance(mlp.__dict__.get(ATTRIBUTE), SiluFold):
            continue
        weight = mlp.gate_up_proj.weight
        n, k = weight.shape
        entries = _gate_up_entries(mlp.gate_up_proj, k, n)
        mma_m = next(iter(entries.values()))[0]
        with torch.no_grad():
            weight.data.copy_(interleave_gate_up(weight.data, mma_m))
        mlp.__dict__[ATTRIBUTE] = SiluFold(
            mma_m,
            {rows: fold_tactic(t) for rows, t in entries.items()},
            weight._version,
            _fingerprint(weight, mma_m),
        )
        changed += 1
    if changed:
        fold = layers[0].mlp.__dict__[ATTRIBUTE]
        logger.info(
            f"[musespark_open] drafter_silu_fold: silu(gate) * up in the gate_up GEMM's epilogue on {changed} drafter "
            f"layers, blocks of {','.join(str(r) for r in sorted(fold.tactics))} rows (stages "
            f"{','.join(str(fold.tactics[r].ab_stages) for r in sorted(fold.tactics))}); gate_up weights interleaved "
            f"per {fold.mma_m} rows, other rows on the interleaved silu_and_mul (tep8_drafter_wide_gemms__SILU_FOLD)"
        )
    return changed


def folded_forward(mlp: nn.Module, fold: SiluFold, x, final_all_reduce_params, lora_params):
    """``GatedMLP.forward`` of a folded MLP (interleaved ``gate_up_proj`` weight)."""
    from ...ops.gemm.silu_fold_gemm import silu_fold_gemm, takes
    from ...ops.spec.drafter_silu_and_mul import drafter_silu_and_mul

    if lora_params:
        raise NotImplementedError(
            "[musespark_open] tep8_drafter_wide_gemms__SILU_FOLD: LoRA on an interleaved gate_up"
        )
    weight = mlp.gate_up_proj.weight
    if weight._version != fold.version:
        raise RuntimeError(
            "[musespark_open] tep8_drafter_wide_gemms__SILU_FOLD: the drafter's gate_up weight was written after the "
            "fold interleaved it (a weight reload?); its rows are no longer in the fold's order"
        )
    flat = x.reshape(-1, x.shape[-1]) if isinstance(x, torch.Tensor) and x.dim() >= 2 else None
    rows = flat.shape[0] if flat is not None else 0
    tactic = fold.tactic(rows)
    if tactic is not None and takes(flat, weight, tactic):
        fold.calls["fold"] += 1
        h2 = silu_fold_gemm(flat, weight, tactic, mlp.gate_up_proj.quant_method.enable_pdl)
    else:
        if tactic is not None:
            fold.refused(rows, "silu_fold_gemm.takes refused the operands")
        fold.calls["plain"] += 1
        h = mlp.gate_up_proj(x)
        h2 = drafter_silu_and_mul(h.reshape(-1, h.shape[-1]), False, fold.half)
    h2 = h2.view(*x.shape[:-1], h2.shape[-1])
    return mlp.down_proj(h2, all_reduce_params=final_all_reduce_params, layer_idx=mlp.layer_idx)
