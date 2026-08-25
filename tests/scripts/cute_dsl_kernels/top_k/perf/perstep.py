#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Per-(layer, step) pairing straight out of the nsys sqlite.
#
# One NVTX range = one (arm, model, isl, N, B, layer); inside it the
# harness runs one kernel per decode step, in step order. So the k-th
# kernel of a range IS step k, and the arms can be paired index by
# index. This is the only way to get a true per-step ratio - the CSV
# summary only carries Avg/Min/Max per range, which cannot be paired.
import glob
import os
import sqlite3

D = os.environ.get("GVR_BENCH_OUT", "./bench_results")


def per_step(path):
    """{(model, isl, B, layer): [step0_us, step1_us, ...]} for one file."""
    out = {}
    con = sqlite3.connect(path)
    rng = con.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'c|%' ORDER BY start"
    ).fetchall()
    ker = con.execute(
        "SELECT start, end - start FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    con.close()
    # ranges do not overlap (one layer at a time), so a single merge walk
    # assigns every kernel to the range containing it
    i = 0
    for rs, re_, text in rng:
        while i < len(ker) and ker[i][0] < rs:
            i += 1
        p = text.lstrip(":").split("|")
        if len(p) != 7:
            while i < len(ker) and ker[i][0] < re_:
                i += 1
            continue
        _, arm, m, isl, _n, b, lay = p
        acc = out.setdefault((m, isl, int(b[1:]), lay), [])
        while i < len(ker) and ker[i][0] < re_:
            acc.append(ker[i][1] / 1000.0)
            i += 1
    return out


def collect(pattern):
    agg = {}
    for f in sorted(glob.glob(os.path.join(D, pattern))):
        for k, v in per_step(f).items():
            agg.setdefault(k, []).extend(v) if k in agg else agg.setdefault(k, v)
    return agg


if __name__ == "__main__":
    arms = {
        "pr": "f22_*_pr.sqlite",
        "st": "f22_*_st.sqlite",
        "va": "f22_*_va.sqlite",
        "vb": "f22_*_vb.sqlite",
        "wf": "f30_*.sqlite",
    }
    data = {a: collect(p) for a, p in arms.items()}
    for a, d in data.items():
        n = sum(len(v) for v in d.values())
        print(f"  {a}: {len(d)} (cell, layer) pairs  {n} steps", flush=True)
    import pickle

    with open("/home/scratch.siyid_coreai/workspace/perstep.pkl", "wb") as f:
        pickle.dump(data, f)
    print("saved perstep.pkl")
