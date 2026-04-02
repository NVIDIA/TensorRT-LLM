#!/usr/bin/env python3
"""Compare real vs sim benchmark report.json files side-by-side.

Usage:
    python3 compare_reports.py --real /tmp/real_report.json \
                               --sim /tmp/sim_report.json
"""
import argparse
import json
import sys


def pct_diff(real_val, sim_val):
    if real_val == 0:
        return "N/A"
    return f"{abs(sim_val - real_val) / real_val * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Compare real vs sim benchmark reports")
    parser.add_argument("--real", required=True, help="Real report.json")
    parser.add_argument("--sim", required=True, help="Sim report.json")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Max acceptable %% difference (default 30)")
    args = parser.parse_args()

    with open(args.real) as f:
        real = json.load(f)
    with open(args.sim) as f:
        sim = json.load(f)

    # Normalize: real report may use different key names
    # trtllm-bench report uses nested structure; sim uses flat dict
    # Handle both formats
    def get_val(report, key, fallback_key=None):
        if key in report:
            return report[key]
        if fallback_key and fallback_key in report:
            return report[fallback_key]
        return None

    metrics = [
        ("completed", "completed", "num_requests"),
        ("total_output", "total_output", "total_output_tokens"),
        ("mean_ttft_ms", "mean_ttft_ms", None),
        ("mean_tpot_ms", "mean_tpot_ms", None),
        ("output_throughput", "output_throughput", None),
        ("mean_e2e_latency_ms", "mean_e2e_latency_ms", None),
    ]

    print("=" * 65)
    print(f"{'Metric':<25} {'Real':>12} {'Sim':>12} {'Diff':>10}")
    print("-" * 65)

    failures = []
    for name, sim_key, real_alt in metrics:
        real_val = get_val(real, name, real_alt)
        sim_val = get_val(sim, sim_key)
        if real_val is None or sim_val is None:
            print(f"{name:<25} {'N/A':>12} {'N/A':>12} {'skip':>10}")
            continue

        diff = pct_diff(float(real_val), float(sim_val))
        print(f"{name:<25} {float(real_val):>12.2f} {float(sim_val):>12.2f} {diff:>10}")

        # Check threshold for numeric metrics (skip count metrics)
        if name not in ("completed", "total_output") and diff != "N/A":
            pct = float(diff.rstrip("%"))
            if pct > args.threshold:
                failures.append((name, pct))

    print("=" * 65)

    if failures:
        print(f"\nWARNING: {len(failures)} metrics exceed "
              f"{args.threshold}% threshold:")
        for name, pct in failures:
            print(f"  {name}: {pct:.1f}%")
        sys.exit(1)
    else:
        print(f"\nAll metrics within {args.threshold}% threshold. PASS")


if __name__ == "__main__":
    main()
