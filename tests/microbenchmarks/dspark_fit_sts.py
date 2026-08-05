# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit the DSpark confidence head's per-position STS temperatures.

WHAT IT FITS, AND WHY THAT OBJECTIVE
------------------------------------
The planner never reads a per-position probability. It reads the *cumulative
product*::

    survival[r][j] = prod_{i <= j} sigmoid(logit[r][i] / T[i])

and sums the largest of those to form tau, the numerator of the budget argmax.
So the thing that has to be calibrated is the survival, not the per-position
confidence -- a temperature vector that makes each position individually
well-calibrated can still leave the depth-5 product badly biased, because a
per-position error of x compounds to x^5.

Hence the objective here is the expected calibration error **of the running
product** at each position, against the label "positions 0..j were ALL
accepted". Positions are fitted left to right and frozen, so every position is
optimised in the cumulative context it will actually be used in.

This mirrors SGLang's ``benchmark/dspark_sts_fit.py`` (same 41-point log grid,
same left-to-right freeze, same cumprod-ECE objective), so a table fitted by
either tool is meaningful to the other. Deliberately so: the vectors are the one
artifact the two stacks can share without a converter.

INPUT
-----
Shards written by ``DSparkStsRecorder`` (set ``TLLM_DSPARK_STS_COLLECT_PATH``
on a normal DSpark run), each a torch save of::

    {"logits": [N, K] float32, "prefix_mask": [N, K] float32}

USAGE
-----
    python3 tests/microbenchmarks/dspark_fit_sts.py \\
        --data "/path/to/sts_collect.*.pt" --out sts.json

The emitted JSON carries both ``sts_temperatures`` and ``temperatures`` so it
loads in this repo and in SGLang unchanged.
"""

import argparse
import glob
import json
import math
import sys
from typing import List, Tuple

import torch

_EPS_PROB = 1e-8


def default_temperature_grid() -> torch.Tensor:
    """41 points, log-spaced over [0.1, 10]. Same grid SGLang searches."""
    return torch.logspace(math.log10(0.1), math.log10(10.0), steps=41)


def expected_calibration_error(*, probs: torch.Tensor, targets: torch.Tensor,
                               num_bins: int) -> float:
    """Bin by predicted probability; average |mean predicted - mean actual|.

    Weighted by bin population, so a bin holding three samples cannot dominate
    one holding thirty thousand.
    """
    probs = probs.reshape(-1).to(torch.float64).clamp(_EPS_PROB, 1.0 - _EPS_PROB)
    targets = targets.reshape(-1).to(torch.float64)
    total = probs.numel()
    if total == 0:
        return float("nan")
    bin_index = (probs * num_bins).long().clamp_(0, num_bins - 1)
    count = torch.zeros(num_bins, dtype=torch.float64)
    pred_sum = torch.zeros(num_bins, dtype=torch.float64)
    target_sum = torch.zeros(num_bins, dtype=torch.float64)
    count.scatter_add_(0, bin_index, torch.ones_like(probs))
    pred_sum.scatter_add_(0, bin_index, probs)
    target_sum.scatter_add_(0, bin_index, targets)
    denom = count.clamp_min(1.0)
    bin_error = (pred_sum / denom - target_sum / denom).abs()
    return float((bin_error * count).sum().item() / total)


def fit_sts_temperatures(*, logits: torch.Tensor, prefix_mask: torch.Tensor,
                         grid: torch.Tensor, num_bins: int = 15) -> dict:
    """Greedy left-to-right search minimising the cumprod ECE per position.

    Left to right and frozen, not joint: position j's temperature is chosen
    against the survival the already-fitted positions 0..j-1 produce. A joint
    search over K temperatures would be exponential in the grid, and the greedy
    order is the causal one -- position j's contribution is only ever multiplied
    into a product that already contains its predecessors.
    """
    logits = logits.to(torch.float64)
    prefix_mask = prefix_mask.to(torch.float64)
    num_samples, block_size = logits.shape
    if num_samples == 0:
        raise ValueError("need at least one sample")
    grid_values = grid.to(torch.float64).tolist()

    temperatures: List[float] = []
    ece_before: List[float] = []
    ece_after: List[float] = []

    # Two running products: what an uncalibrated head would have said (T == 1),
    # and what the fit says so far. Reporting both is the only way to see
    # whether the fit bought anything.
    survival_at_one = torch.ones(num_samples, dtype=torch.float64)
    survival_fitted = torch.ones(num_samples, dtype=torch.float64)

    for position in range(block_size):
        position_logits = logits[:, position]
        position_target = prefix_mask[:, position]

        survival_at_one = survival_at_one * torch.sigmoid(position_logits)
        ece_before.append(
            expected_calibration_error(probs=survival_at_one,
                                       targets=position_target,
                                       num_bins=num_bins))

        best_t = grid_values[0]
        best_survival = survival_fitted * torch.sigmoid(position_logits / best_t)
        best_ece = expected_calibration_error(probs=best_survival,
                                              targets=position_target,
                                              num_bins=num_bins)
        for temperature in grid_values[1:]:
            candidate = survival_fitted * torch.sigmoid(position_logits / temperature)
            ece = expected_calibration_error(probs=candidate,
                                             targets=position_target,
                                             num_bins=num_bins)
            if ece < best_ece:
                best_ece, best_t, best_survival = ece, temperature, candidate

        temperatures.append(float(best_t))
        ece_after.append(float(best_ece))
        survival_fitted = best_survival

    return {
        "temperatures": temperatures,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


def load_shards(pattern: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load shards, refusing any whose pairing provenance is wrong or absent.

    Shards written by the current recorder carry ``meta.pairing ==
    "draft_seq_ring"``: each row's logits were selected from a stamped host
    snapshot of exactly the draft pass that produced the block the label
    verifies, keyed by the worker's own row allocator. Earlier recorders
    paired the label against a sampler-time read of the live buffer -- stale
    by one draft pass for most rows under the overlap scheduler (on job
    2562577, 26383 of 32083 rows differed, correlation 0.14) -- and keyed
    rows by ``py_seq_slot``, a different allocator that drifts after the
    first request completes. Fitting such a shard does not fail loudly; it
    converges to a temperature vector that calibrates noise, which is what
    the 0.23 residual ECE of the first deployed table was. There is
    deliberately no flag to accept them: re-collect.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no shards matched {pattern!r}")
    logits, masks, block_size = [], [], None
    for path in paths:
        blob = torch.load(path, map_location="cpu")
        meta = blob.get("meta") or {}
        pairing = meta.get("pairing")
        if pairing != "draft_seq_ring":
            raise SystemExit(
                f"{path}: pairing provenance is {pairing!r}, expected "
                f"'draft_seq_ring'. This shard predates the stamped snapshot "
                f"ring, so its labels are joined to the wrong draft pass "
                f"and/or the wrong buffer row, and a fit on it calibrates "
                f"noise. Re-collect with the current recorder.")
        lg, pm = blob["logits"], blob["prefix_mask"]
        if lg.shape != pm.shape:
            raise SystemExit(f"{path}: logits {tuple(lg.shape)} != mask {tuple(pm.shape)}")
        if block_size is None:
            block_size = lg.shape[1]
        elif lg.shape[1] != block_size:
            # Silently concatenating these would fit a vector for a block size
            # that never existed.
            raise SystemExit(
                f"{path}: block size {lg.shape[1]} != {block_size} from earlier shards")
        logits.append(lg)
        masks.append(pm)
    return torch.cat(logits, 0), torch.cat(masks, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="glob for shards from DSparkStsRecorder")
    ap.add_argument("--out", required=True, help="where to write the JSON")
    ap.add_argument("--num-bins", type=int, default=15)
    ap.add_argument("--min-samples", type=int, default=1000,
                    help="refuse to fit on fewer; a temperature fitted on a "
                         "handful of blocks is noise with a filename")
    ap.add_argument("--eval-temps", default=None,
                    help="Evaluate an EXISTING temperature table on these "
                         "shards instead of fitting: reports the per-position "
                         "cumprod ECE the loaded vector achieves on fresh "
                         "data. This is the producer+consumer joint check -- "
                         "collect on a serving run with the table loaded and "
                         "the window pinned to the block, then compare this "
                         "number against the table's stored ece_after. A "
                         "large gap means the table does not describe the "
                         "live head (drift, wrong file, wrong column), and "
                         "nothing else in the pipeline can see that.")
    args = ap.parse_args(argv)

    logits, prefix_mask = load_shards(args.data)

    if args.eval_temps:
        with open(args.eval_temps, encoding="utf-8") as fh:
            table = json.load(fh)
        temps = table.get("sts_temperatures") or table["temperatures"]
        lg = logits.to(torch.float64)
        pm = prefix_mask.to(torch.float64)
        survival = torch.ones(lg.shape[0], dtype=torch.float64)
        print(f"evaluating {args.eval_temps} on {lg.shape[0]} fresh rows:")
        stored = table.get("ece_after") or []
        for j in range(min(lg.shape[1], len(temps))):
            survival = survival * torch.sigmoid(lg[:, j] / float(temps[j]))
            ece = expected_calibration_error(probs=survival,
                                             targets=pm[:, j],
                                             num_bins=args.num_bins)
            ref = f"  (fit-time {stored[j]:.4f})" if j < len(stored) else ""
            print(f"  pos {j}: live ECE {ece:.4f}{ref}")
        return 0
    n, block_size = logits.shape
    print(f"loaded {n} samples, block_size={block_size}")
    if n < args.min_samples:
        raise SystemExit(
            f"only {n} samples (< --min-samples {args.min_samples}). Collect "
            f"more before fitting: the deepest positions are the sparsest and "
            f"the ones the planner is most sensitive to.")

    # An all-ones column means every block was fully accepted at that depth, so
    # the ECE is defined but carries no information to fit against -- say so
    # rather than emitting a confident-looking temperature.
    per_position_rate = prefix_mask.mean(dim=0)
    print("prefix-accept rate by position:",
          [round(float(v), 4) for v in per_position_rate])
    degenerate = [j for j, v in enumerate(per_position_rate) if v in (0.0, 1.0)]
    if degenerate:
        print(f"WARNING: positions {degenerate} are degenerate (rate 0 or 1); "
              f"their temperature is unconstrained by this data")

    result = fit_sts_temperatures(logits=logits, prefix_mask=prefix_mask,
                                  grid=default_temperature_grid(),
                                  num_bins=args.num_bins)

    payload = {
        "sts_temperatures": result["temperatures"],
        "temperatures": result["temperatures"],
        "ece_before": result["ece_before"],
        "ece_after": result["ece_after"],
        "dataset": args.data,
        "num_samples": int(n),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nwrote {args.out}")
    print(f"  temperatures : {[round(t, 4) for t in result['temperatures']]}")
    print(f"  ECE before   : {[round(e, 5) for e in result['ece_before']]}")
    print(f"  ECE after    : {[round(e, 5) for e in result['ece_after']]}")
    improved = sum(1 for a, b in zip(result["ece_after"], result["ece_before"]) if a < b)
    print(f"  improved at {improved}/{block_size} positions")
    if improved == 0:
        print("  NOTE: no position improved -- the head was already calibrated "
              "on this data, or the data does not cover the relevant range.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
