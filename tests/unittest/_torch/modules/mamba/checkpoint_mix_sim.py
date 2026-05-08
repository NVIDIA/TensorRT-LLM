#!/usr/bin/env python3
"""
Compute the steady-state checkpoint (write) fraction for a Mamba
replay-style cache, given an acceptance-length (AL) distribution.

Model
-----
* T tokens per step (DL draft + 1 target).
* WINDOW = max_window: cache T-axis size.
* PNAT = "previous number of accepted tokens" already cached at start of
  step.  After accepting AL tokens this step:
    - nowrite step (PNAT + T <= WINDOW): new tokens append at
      [PNAT, PNAT+T) of the active buffer; PNAT_new = PNAT + AL.
    - write step (PNAT + T > WINDOW): new tokens go to the staging
      buffer at [0, T); cache_buf_idx flips; PNAT_new = AL.

The write decision in the kernel is exactly `pnat + T > WINDOW`.

Two methods
-----------
1. **Markov chain stationary** (exact): build the (WINDOW+1)x(WINDOW+1)
   transition matrix from the AL distribution, solve for pi.
       write_frac = sum_{p > WINDOW - T} pi(p)

2. **Depletion sim** (approximate, matches the user's pen-and-paper
   intuition): start with all mass at PNAT=0, propagate forward, count
   mass that hits a write state at each step, remove that mass without
   reinjecting.  Compute E[N] = E[step at first checkpoint | start
   PNAT=0].  Steady-state write_frac = 1 / (E[N] - 1).
   The "-1" is because the first step from PNAT=0 is always nowrite
   (free): subsequent cycles start from PNAT ~ AL_dist (post-write
   distribution), one step shorter than the from-PNAT=0 cycle.

Both methods should agree (verified on small examples).

CSV format for the AL distribution
----------------------------------
Two columns per row: AL (int 1..T), count or probability (float).
First row treated as a header if the first cell isn't numeric.
The histogram is auto-normalized to a probability distribution.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_al_distribution(path: Path, T: int, column: int = 1) -> np.ndarray:
    """Return a length-(T+1) probability vector indexed by AL (0..T).

    Reads column 0 as AL, column ``column`` as count/probability.
    Non-numeric rows (header, trailing summary rows like "total"/"mean")
    are silently skipped.
    """
    rows = list(csv.reader(open(path)))
    if not rows:
        sys.exit(f"empty CSV: {path}")
    al_to_count: dict[int, float] = {}
    for r in rows:
        if not r or not r[0].strip():
            continue
        try:
            al = int(float(r[0]))
            c = float(r[column])
        except (ValueError, IndexError):
            continue  # header / summary row
        al_to_count[al] = al_to_count.get(al, 0.0) + c
    if not al_to_count:
        sys.exit(f"no numeric rows parsed from {path} (column index {column})")
    al_max = max(al_to_count)
    if al_max > T:
        sys.exit(
            f"CSV has AL={al_max} but --T={T} (a step processes only T tokens; "
            f"AL > T is impossible).  Mismatch likely indicates wrong --T or wrong CSV."
        )
    if min(al_to_count) < 0:
        sys.exit(f"CSV has negative AL values")
    if al_to_count.get(0, 0) > 0:
        print(
            f"WARN: CSV has AL=0 mass ({al_to_count[0]:.4f} unnormalized).  "
            "AL=0 means no progress on a step; usually impossible in spec "
            "decoding (target token always accepted).  Treating as a real "
            "value (will produce a chain that doesn't converge if AL=0 has "
            "non-trivial mass).",
            file=sys.stderr,
        )
    dist = np.zeros(T + 1, dtype=np.float64)
    for al, c in al_to_count.items():
        dist[al] = c
    s = dist.sum()
    if s == 0:
        sys.exit(f"AL distribution sums to zero in {path}")
    return dist / s


def markov_stationary(al_dist: np.ndarray, T: int, window: int) -> np.ndarray:
    """
    Build the (window+1) x (window+1) transition matrix and return the
    stationary distribution pi.
    """
    n = window + 1
    P = np.zeros((n, n))
    for p in range(n):
        is_write = (p + T > window)
        for al in range(1, T + 1):
            prob = al_dist[al]
            if prob == 0:
                continue
            p_new = al if is_write else p + al
            assert 0 <= p_new <= window, (
                f"unreachable transition: p={p} al={al} write={is_write} "
                f"-> p_new={p_new} outside [0, {window}]"
            )
            P[p, p_new] += prob
    # Solve pi P = pi via the left eigenvector for eigenvalue 1.
    # Equivalently: P^T pi = pi.
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    if abs(eigvals[idx] - 1.0) > 1e-6:
        sys.exit(
            f"no stationary eigenvalue near 1 (closest is {eigvals[idx]}).  "
            "AL dist may not produce an ergodic chain."
        )
    pi = np.real(eigvecs[:, idx])
    # Iterative refinement: start near the eigenvector, then iterate
    # P^k to clean up any imaginary leakage from the eig solve.
    pi = np.maximum(pi, 0.0)
    if pi.sum() == 0:
        # Fallback: power iteration from uniform.
        pi = np.ones(n) / n
    for _ in range(2000):
        new_pi = pi @ P
        if np.allclose(new_pi, pi, atol=1e-12, rtol=0):
            pi = new_pi
            break
        pi = new_pi
    pi = pi / pi.sum()
    return pi


def sample_steady_state_pnat(
    al_dist: np.ndarray,
    T: int,
    window: int,
    batch: int,
    K: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Draw K independent (batch,)-shaped per-slot PNAT vectors from the
    Markov chain's stationary distribution, given an AL distribution.

    Returns int64 array of shape (K, batch) with values in [0, window].
    PNAT=0 has weight ~0 in steady state so it is effectively never
    sampled (it's purely a boot-up state).
    """
    pi = markov_stationary(al_dist, T, window)
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()
    states = np.arange(window + 1)
    rng = np.random.default_rng(seed)
    return rng.choice(states, size=(K, batch), p=pi).astype(np.int64)


def depletion_sim(
    al_dist: np.ndarray, T: int, window: int, max_steps: int = 256
) -> np.ndarray:
    """
    Start with mass 1 at PNAT=0.  At each step, mass at PNAT > WINDOW-T
    is removed (= it checkpoints this step).  Remaining mass advances
    by AL.  Returns array `removed[step-1]` = mass that checkpointed at
    step.
    """
    n = window + 1
    mass = np.zeros(n)
    mass[0] = 1.0
    write_thresh = window - T  # PNAT > this => write
    removed = []
    for _ in range(max_steps):
        write_mass = mass[write_thresh + 1:].sum()
        removed.append(float(write_mass))
        # Strip the mass that checkpointed (it's been "removed").
        mass = mass.copy()
        mass[write_thresh + 1:] = 0.0
        # Propagate the rest by AL.
        new_mass = np.zeros(n)
        for p in range(write_thresh + 1):
            if mass[p] == 0:
                continue
            for al in range(1, T + 1):
                prob = al_dist[al]
                if prob == 0:
                    continue
                p_new = p + al
                assert p_new <= window, "unreachable"
                new_mass[p_new] += mass[p] * prob
        mass = new_mass
        if mass.sum() < 1e-15:
            break
    return np.array(removed)


def _format_pi(pi: np.ndarray, T: int, window: int) -> str:
    write_thresh = window - T
    lines = []
    for p, prob in enumerate(pi):
        if prob < 1e-9:
            continue
        marker = "  <- write state" if p > write_thresh else ""
        lines.append(f"  pi(PNAT={p:2d}) = {prob:.4f}{marker}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("al_csv", type=Path, help="CSV with AL,count|prob columns")
    ap.add_argument(
        "--T", type=int, default=6,
        help="Tokens per step (DL+1).  Default 6 = production Nemotron config.",
    )
    ap.add_argument(
        "--window", type=int, default=16,
        help="Cache T-axis size (max_window).  Default 16 = production.",
    )
    ap.add_argument(
        "--method", choices=["markov", "depletion", "both"], default="both",
        help="Which method(s) to compute.",
    )
    ap.add_argument(
        "--column", type=int, default=1,
        help="Column index (0-based) of the count/prob in the CSV.  "
        "Use this to pick a specific variant when the CSV has multiple "
        "count columns (e.g. replay_count, true_baseline_count, ...).",
    )
    ap.add_argument(
        "--quiet", action="store_true",
        help="Print only the final write fraction (script-friendly).",
    )
    args = ap.parse_args()

    if args.T >= args.window:
        ap.error(
            f"--T ({args.T}) must be < --window ({args.window}); otherwise "
            "every step from PNAT>0 is a write step."
        )

    al = load_al_distribution(args.al_csv, args.T, column=args.column)

    if not args.quiet:
        print(f"AL distribution (normalized):")
        for i, p in enumerate(al):
            if p > 0:
                print(f"  AL={i}: {p:.4f}")
        e_al = float(sum(i * p for i, p in enumerate(al)))
        print(f"  E[AL] = {e_al:.3f} (per-step throughput)")
        print()
        print(f"T = {args.T}, window = {args.window}")
        print(f"write threshold (exclusive): PNAT > {args.window - args.T}")
        print(f"  => write states: PNAT in {{{args.window - args.T + 1}, ..., {args.window}}}")
        print()

    write_frac_mc = None
    write_frac_dep = None

    if args.method in ("markov", "both"):
        pi = markov_stationary(al, args.T, args.window)
        write_frac_mc = float(pi[args.window - args.T + 1:].sum())
        if not args.quiet:
            print("--- Markov chain stationary (exact) ---")
            print(_format_pi(pi, args.T, args.window))
            print(f"  write fraction         = {write_frac_mc:.4f}")
            if write_frac_mc > 0:
                print(f"  avg steps per checkpoint = {1/write_frac_mc:.2f}")
            print()

    if args.method in ("depletion", "both"):
        removed = depletion_sim(al, args.T, args.window)
        total = float(removed.sum())
        if total < 0.99:
            print(
                f"WARN: depletion sim only depleted {total:.4f} of mass "
                f"in {len(removed)} steps; may be missing tail.",
                file=sys.stderr,
            )
        e_step = float(sum((i + 1) * r for i, r in enumerate(removed)) / max(total, 1e-12))
        # Steady-state cycle length = E[N from PNAT=0] - 1 (the first
        # nowrite step from PNAT=0 is the "free" one; subsequent cycles
        # start from PNAT ~ AL_dist, one step shorter).
        cycle = e_step - 1.0
        write_frac_dep = (1.0 / cycle) if cycle > 0 else float("inf")
        if not args.quiet:
            print("--- Depletion sim (start at PNAT=0) ---")
            for i, r in enumerate(removed):
                if r > 1e-6:
                    print(f"  step {i+1:2d}: checkpoint mass = {r:.4f}")
            print(f"  total mass depleted   = {total:.6f}")
            print(f"  E[step at first checkpoint] = {e_step:.3f}")
            print(f"  steady-state cycle (=E[N]-1) = {cycle:.3f}")
            print(f"  write fraction (1/cycle)    = {write_frac_dep:.4f}")
            print()

    if args.quiet:
        wf = write_frac_mc if write_frac_mc is not None else write_frac_dep
        print(f"{wf:.6f}")
        return

    if write_frac_mc is not None and write_frac_dep is not None:
        diff = abs(write_frac_mc - write_frac_dep)
        rel = diff / max(write_frac_mc, 1e-12)
        print(
            f"--- cross-check ---\n"
            f"  markov:    {write_frac_mc:.6f}\n"
            f"  depletion: {write_frac_dep:.6f}\n"
            f"  abs diff:  {diff:.2e}  (rel: {rel:.2%})"
        )
        if rel > 0.01:
            print(
                "  WARN: methods disagree by >1%.  Possible: AL=0 mass or "
                "non-ergodic chain — check the AL distribution.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
