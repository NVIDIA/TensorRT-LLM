#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare two MoE communication benchmark JSON files side-by-side.

Usage:
    # Default: show PrepareCombine + Combine kernels
    python compare_moe_comm.py baseline.json lowprec.json

    # Show all 4 kernels (PrepareDispatch, Dispatch, PrepareCombine, Combine)
    python compare_moe_comm.py baseline.json lowprec.json --all-kernels

    # Pick specific kernels
    python compare_moe_comm.py baseline.json lowprec.json -k Combine PrepareCombine Dispatch

    # Use mean instead of median
    python compare_moe_comm.py baseline.json lowprec.json --stat mean

    # Specific rank
    python compare_moe_comm.py baseline.json lowprec.json --rank rank3
"""

import argparse
import json
from typing import List, Optional, Tuple

# Short aliases -> substrings to match in kernel names
KERNEL_ALIASES = {
    "PrepareDispatch": "PrepareDispatchKernel",
    "Dispatch": "DispatchKernel",
    "PrepareCombine": "PrepareCombineKernel",
    "Combine": "CombineKernel",
}

DEFAULT_KERNELS = ["PrepareCombine", "Combine"]
ALL_KERNELS = ["PrepareDispatch", "Dispatch", "PrepareCombine", "Combine"]


def load_results(path: str) -> Tuple[dict, List[dict]]:
    """Load JSON and return (metadata, results_list)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data.get("benchmark_metadata", {}), data["results"]
    # Fallback: list of result objects
    if isinstance(data, list):
        return {}, data
    raise ValueError(f"Unexpected JSON structure in {path}")


def find_kernel(kernels: List[dict], alias: str) -> Optional[dict]:
    """Find a kernel entry matching the alias substring."""
    substr = KERNEL_ALIASES.get(alias, alias)
    for k in kernels:
        # Match against kernel name, excluding broad matches
        # e.g. "CombineKernel" should not match "PrepareCombineKernel"
        name = k["name"]
        if substr in name:
            # For "DispatchKernel" / "CombineKernel", exclude "Prepare" prefix
            if substr in ("DispatchKernel", "CombineKernel") and "Prepare" in name:
                continue
            return k
    return None


def get_kernel_stat(kernel: Optional[dict], rank: str, stat: str) -> Optional[float]:
    """Extract a statistic from a kernel's per_rank data."""
    if kernel is None:
        return None
    per_rank = kernel.get("per_rank", {})
    rank_data = per_rank.get(rank)
    if rank_data is None:
        return None
    if isinstance(rank_data, (int, float)):
        return float(rank_data)
    if isinstance(rank_data, dict):
        return rank_data.get(stat)
    return None


def main():
    parser = argparse.ArgumentParser(description="Compare two MoE comm benchmark JSON files.")
    parser.add_argument("file_a", help="First JSON file (baseline)")
    parser.add_argument("file_b", help="Second JSON file (comparison)")
    parser.add_argument("--label-a", default=None, help="Label for file A (default: filename)")
    parser.add_argument("--label-b", default=None, help="Label for file B (default: filename)")
    parser.add_argument(
        "-k",
        "--kernels",
        nargs="+",
        default=None,
        choices=list(KERNEL_ALIASES.keys()),
        help="Kernels to show (default: PrepareCombine Combine)",
    )
    parser.add_argument(
        "--all-kernels",
        action="store_true",
        help="Show all 4 kernels (PrepareDispatch, Dispatch, PrepareCombine, Combine)",
    )
    parser.add_argument(
        "--stat",
        default="median",
        choices=["mean", "median", "min", "max"],
        help="Statistic to compare (default: median)",
    )
    parser.add_argument("--rank", default="rank0", help="Rank to show (default: rank0)")
    args = parser.parse_args()

    if args.all_kernels:
        kernels_to_show = ALL_KERNELS
    elif args.kernels:
        kernels_to_show = args.kernels
    else:
        kernels_to_show = DEFAULT_KERNELS

    label_a = args.label_a or args.file_a.rsplit("/", 1)[-1].replace(".json", "")
    label_b = args.label_b or args.file_b.rsplit("/", 1)[-1].replace(".json", "")

    meta_a, results_a = load_results(args.file_a)
    meta_b, results_b = load_results(args.file_b)

    # Index by batch size
    by_batch_a = {r["local_batch_size"]: r for r in results_a}
    by_batch_b = {r["local_batch_size"]: r for r in results_b}
    batches = sorted(set(by_batch_a.keys()) | set(by_batch_b.keys()))

    # Print metadata
    for tag, meta in [("A", meta_a), ("B", meta_b)]:
        lbl = label_a if tag == "A" else label_b
        ep = meta.get("ep_size", "?")
        backend = meta.get("backend", "?")
        print(f"[{tag}] {lbl}  (ep={ep}, backend={backend})")
    print(f"Stat: {args.stat}, Rank: {args.rank}")
    print()

    # Build header
    kernel_col_width = 12
    header_parts = [f"{'batch':>6}"]
    sub_parts = [f"{'':>6}"]
    for kname in kernels_to_show:
        short = kname[:kernel_col_width]
        header_parts.append(f"{short:>{kernel_col_width}}")
        header_parts.append(f"{short:>{kernel_col_width}}")
        header_parts.append(f"{'speedup':>{kernel_col_width}}")
        sub_parts.append(f"{'A (us)':>{kernel_col_width}}")
        sub_parts.append(f"{'B (us)':>{kernel_col_width}}")
        sub_parts.append(f"{'(A/B)':>{kernel_col_width}}")

    # Also show total dispatch and total combine
    for total_name in ["total_dispatch", "total_combine"]:
        header_parts.append(f"{total_name:>{kernel_col_width}}")
        header_parts.append(f"{total_name:>{kernel_col_width}}")
        header_parts.append(f"{'speedup':>{kernel_col_width}}")
        sub_parts.append(f"{'A (us)':>{kernel_col_width}}")
        sub_parts.append(f"{'B (us)':>{kernel_col_width}}")
        sub_parts.append(f"{'(A/B)':>{kernel_col_width}}")

    sep = " | "
    print(sep.join(header_parts))
    print(sep.join(sub_parts))
    print("-" * len(sep.join(header_parts)))

    for bs in batches:
        ra = by_batch_a.get(bs)
        rb = by_batch_b.get(bs)
        if ra is None or rb is None:
            continue

        all_kernels_a = ra.get("dispatch_kernels", []) + ra.get("combine_kernels", [])
        all_kernels_b = rb.get("dispatch_kernels", []) + rb.get("combine_kernels", [])

        row = [f"{bs:>6}"]
        for kname in kernels_to_show:
            ka = find_kernel(all_kernels_a, kname)
            kb = find_kernel(all_kernels_b, kname)
            va = get_kernel_stat(ka, args.rank, args.stat)
            vb = get_kernel_stat(kb, args.rank, args.stat)
            if va is not None and vb is not None and vb > 0:
                speedup = va / vb
                row.append(f"{va:>{kernel_col_width}.1f}")
                row.append(f"{vb:>{kernel_col_width}.1f}")
                row.append(f"{speedup:>{kernel_col_width}.2f}x")
            else:
                row.append(f"{'N/A':>{kernel_col_width}}")
                row.append(f"{'N/A':>{kernel_col_width}}")
                row.append(f"{'N/A':>{kernel_col_width}}")

        # Total dispatch and total combine
        for key in ["dispatch_us", "combine_us"]:
            ta = ra.get(key, {}).get(args.rank)
            tb = rb.get(key, {}).get(args.rank)
            if ta and tb:
                va = ta[args.stat] if isinstance(ta, dict) else ta
                vb = tb[args.stat] if isinstance(tb, dict) else tb
                if vb > 0:
                    row.append(f"{va:>{kernel_col_width}.1f}")
                    row.append(f"{vb:>{kernel_col_width}.1f}")
                    row.append(f"{va / vb:>{kernel_col_width}.2f}x")
                else:
                    row.extend([f"{'N/A':>{kernel_col_width}}"] * 3)
            else:
                row.extend([f"{'N/A':>{kernel_col_width}}"] * 3)

        print(sep.join(row))


if __name__ == "__main__":
    main()
