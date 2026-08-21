#!/usr/bin/env python3
# ruff: noqa
# Measurement harness committed verbatim for provenance; bench idioms
# (loop-scoped buffers, del/rebind) trip static analysis.
# f58: B×N 四张表(flash/pro × 算数均值/最小值),逐步配对 vs PR16457,
# 我们的臂按线上路由(plan_emission)选取。
import glob
import os
import pickle
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from perstep import per_step  # noqa: E402

D = os.environ.get("GVR_BENCH_OUT", "./bench_results")
ARMS = {
    "pr": "f58_*_pr.sqlite",
    "st": "f58_*_st.sqlite",
    "va": "f58_*_va.sqlite",
    "vb": "f58_*_vb.sqlite",
    "wf": "f58w_*.sqlite",
}
PKL = "./f58.pkl"
if os.path.exists(PKL) and os.environ.get("REUSE", "1") == "1":
    data = pickle.load(open(PKL, "rb"))
else:
    data = {}
    for a, pat in ARMS.items():
        agg = {}
        for f in sorted(glob.glob(os.path.join(D, pat))):
            for k, v in per_step(f).items():
                agg.setdefault(k, []).extend(v)
        data[a] = agg
        print(f"  {a}: {len(agg)} 个(格,层)  {sum(len(v) for v in agg.values())} 步", flush=True)
    pickle.dump(data, open(PKL, "wb"))

sys.path.insert(
    0,
    str(
        __import__("pathlib").Path(__file__).parents[5]
        / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k"
    ),
)
from gvr_routing import plan_emission  # noqa: E402

KC = {"flash": 512, "pro": 1024, "v32": 2048}
CR = {"flash": 4, "pro": 4, "v32": 1}
TIER2ARM = {"list": "wf", "counts": "va", "rungs": "vb", "none": "st"}
NS = ["8k", "16k", "32k", "64k", "128k", "256k"]
BS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

cell = {}
for m, isl, b in sorted({(m, i, b) for (m, i, b, l) in data["pr"]}):
    n = int(isl[:-1]) * 1024 // CR[m]
    tier = plan_emission(b, n, KC[m], True)
    arm = TIER2ARM[tier]
    ratios = []
    for (mm, ii, bb, l), pr in data["pr"].items():
        if (mm, ii, bb) != (m, isl, b):
            continue
        ou = data[arm].get((mm, ii, bb, l), [])
        for j in range(min(len(pr), len(ou))):
            ratios.append(pr[j] / ou[j])
    if ratios:
        cell[(m, isl, b)] = dict(
            tier=tier, n=len(ratios), mean=sum(ratios) / len(ratios), mn=min(ratios)
        )


def table(m, key):
    print(f"\n### {m} — {'算数均值' if key == 'mean' else '最小值'}(对 PR16457,全层×全步)")
    print("| B \\ N | " + " | ".join(NS) + " |")
    print("|---" * (len(NS) + 1) + "|")
    for b in BS:
        row = [f"**{b}**"]
        for isl in NS:
            c = cell.get((m, isl, b))
            row.append(f"{c[key]:.3f}" if c else "—")
        print("| " + " | ".join(row) + " |")


for m in ("flash", "pro", "v32"):
    for key in ("mean", "mn"):
        table(m, key)

tot = sum(c["n"] for c in cell.values())
print(
    f"\n共 {len(cell)} 格 / {tot} 逐步配对;各格档位:",
    {
        t: sum(1 for c in cell.values() if c["tier"] == t)
        for t in ("list", "counts", "rungs", "none")
    },
)
