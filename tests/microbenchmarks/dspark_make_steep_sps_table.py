# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit an SPS cost table steep enough that the planner chooses to trim.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR
----------------------------------------
It exercises the ragged path end to end *through the planner*: with a steep
``theta(M)`` the confidence scheduler decides to trim, and its budget then
flows through the same confidence-ordered top-k that ships, which hands
different windows to requests with different survival. That is the only way to
see per-request verify lengths on a checkpoint whose acceptance is too high for
trimming to pay.

**No throughput or latency number taken with this table means anything.** The
planner is optimising against a fictional cost curve, so it will trim where a
real deployment should not. Use it to answer "does the ragged path correctly
support different verify lengths per request", never "is ragged faster". The
output carries ``"SYNTHETIC": true`` so a run cannot quietly present it as a
measurement -- ``run_dspark_throughput.sh`` refuses to treat such a table as
evidence.

WHY A TABLE RATHER THAN AN OVERRIDE
-----------------------------------
This replaces ``TLLM_DSPARK_FORCE_VERIFY_LENS``, which assigned windows by
batch position. Position is orthogonal to confidence, so that knob reproduced
the ragged *shape* while bypassing the policy entirely -- it could not tell a
correct assignment from an arbitrary one, and it had to sit in production code
ahead of the cost-table gate to be reachable. A table drives the real path.

HOW STEEPNESS IS CHOSEN
-----------------------
The scheduler maximises expected accepted tokens per second, roughly

    (bs + sum of the top-K survival probabilities) / T(bs, K)

With per-token acceptance around 0.9 each extra verify token adds about 0.9 to
the numerator, so trimming only wins once ``theta`` grows faster than that. The
default ``--exponent 2.0`` makes theta quadratic in the token count, which is
comfortably steeper than the measured curve (near-linear) and forces trimming
across the whole batch-size ladder.

    # derive from a real table, keeping its grid
    python3 tests/microbenchmarks/dspark_make_steep_sps_table.py \
        --from sps_real_final.json --out steep.json

    # or synthesise a grid outright
    python3 tests/microbenchmarks/dspark_make_steep_sps_table.py --out steep.json
"""

import argparse
import json
import sys


def _default_grid(max_batch_size: int, width: int):
    """Token totals a ragged batch can land on: bs * (verify_len + 1)."""
    batch_sizes = [b for b in (1, 4, 8, 16, 32, 64, 128) if b <= max_batch_size]
    tokens = sorted({b * w for b in batch_sizes for w in range(2, width + 1)})
    return batch_sizes, tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--from", dest="source", default=None,
                        help="real table to take the grid from; its timings "
                             "are replaced, only token_counts/batch_sizes are "
                             "reused so the shape matches the deployment")
    parser.add_argument("--exponent", type=float, default=2.0,
                        help="theta(M) ~ M**exponent; >1 makes extra verify "
                             "tokens superlinearly expensive")
    parser.add_argument("--peak-ms", type=float, default=400.0,
                        help="theta at the largest token count")
    parser.add_argument("--fixed-overhead-ms", type=float, default=5.0,
                        help="the non-trimmable floor; small on purpose, so "
                             "the trimmable term dominates the decision")
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--max-verify-len", type=int, default=5)
    args = parser.parse_args()

    if args.exponent <= 1.0:
        parser.error(
            f"--exponent {args.exponent} is not steeper than linear, so the "
            f"planner will keep verifying the full block and the ragged path "
            f"will stay dark -- which is the exact failure this tool exists "
            f"to avoid")

    if args.source:
        with open(args.source, encoding="utf-8") as handle:
            src = json.load(handle)
        tokens = [int(t) for t in src["token_counts"]]
        batch_sizes = [int(b) for b in src["batch_sizes"]]
    else:
        batch_sizes, tokens = _default_grid(args.max_batch_size,
                                            args.max_verify_len + 1)

    tokens = sorted({t for t in tokens if t > 0})
    peak = float(max(tokens))
    theta = [args.peak_ms * (t / peak)**args.exponent for t in tokens]

    table = {
        "token_counts": tokens,
        "step_time_ms": theta,
        "fixed_overhead_ms": args.fixed_overhead_ms,
        "batch_sizes": batch_sizes,
        "batch_overhead_ms": [0.0] * len(batch_sizes),
        "SYNTHETIC": True,
        "_meta": {
            "purpose": "force the planner to trim so the ragged path runs; "
                       "NOT a measurement of anything",
            "exponent": args.exponent,
            "derived_from": args.source,
        },
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2)
        handle.write("\n")

    print(f"wrote {args.out}: {len(tokens)} token buckets "
          f"{tokens[0]}..{tokens[-1]}, theta {theta[0]:.3f}..{theta[-1]:.1f} ms "
          f"(exponent {args.exponent}), SYNTHETIC=true")
    print("Use only to check that per-request verify lengths work. Any "
          "throughput taken with this table is meaningless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
