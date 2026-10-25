# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage self-test for msa-prewarm (one SM100/SM103 GPU, < 100 MB).

  python3 -m tensorrt_llm._torch.attention.backends.sparse.minimax_m3.m3ax_msa_prewarm.selftest \
      [--out FILE] [--cache-geometry tp2|tp4|adp|none] [--qmax 300]

For each geometry (index heads per rank: TP2 2, TP4 1, attention DP 4) the test builds real fmha_sm100 proxy plans
the way msa_backend._build_step_plans does (whole-row proxy plan; segmented KV-split proxy plan via
msa_indexer.segmented_proxy_plan) for every single-row q in 1..qmax on a short and a long cached prefix and for mixed
multi-row batches (short + long rows in both orders, context + 4- and 8-token generation rows), then runs fmha_sm100
with fmha_sm100.api.get_fmha_variant replaced by a recorder that returns a no-op runner: no kernel runs and nothing
compiles (FMHAVariantManager._compile_only raises if reached). Every recorded key must be in core.enumerate_variants
for that geometry. --cache-geometry: the geometry this container prewarmed; every key recorded for it must also have
its .so in the fmha_sm100 cache (what serving would dlopen). Exit 0 iff nothing is missing. Prints one JSON summary.
"""

import argparse
import json
import math
import sys

GEOMS = {"tp2": 2, "tp4": 1, "adp": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--cache-geometry", default="tp2", choices=list(GEOMS) + ["none"])
    ap.add_argument("--qmax", type=int, default=300)
    a = ap.parse_args()

    import torch

    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer

    from . import core

    fmha, jit, api = core._modules()
    torch.cuda.set_device(0)
    dev = torch.device("cuda", 0)
    page, hd = 128, 128
    fp8 = torch.float8_e4m3fn
    code = jit._dlpack_dtype_code(fp8)
    plan_present = (jit.CACHE_BASE / "plan" / "fmha_sm100_plan.so").exists()

    rec = []

    class _Noop(object):
        def run(self, *args):
            return None

    def recorder(*args, **kwargs):
        rec.append(jit._variant_key_from_runtime(*args)[0])
        return _Noop()

    def no_compile(self, name, params):
        raise RuntimeError("selftest: variant %s would compile" % name)

    orig_get, orig_compile = api.get_fmha_variant, type(jit._variant_manager)._compile_only
    api.get_fmha_variant = recorder
    type(jit._variant_manager)._compile_only = no_compile

    def rows_tensors(qs, prefix):
        q = torch.tensor(qs, dtype=torch.int32)
        p = torch.tensor([prefix] * len(qs), dtype=torch.int32)
        kv = q + p
        pages = [int(math.ceil(int(x) / page)) for x in kv.tolist()]
        kv_idx = torch.arange(sum(pages), dtype=torch.int32, device=dev)
        return q, kv, p, kv_idx, sum(pages)

    results = {}
    errors = []
    try:
        for gname, hidx in GEOMS.items():
            enum = core.enumerate_variants(jit, api, core.families_for(hidx, code, page))
            rec.clear()
            cases = [([q], pre) for pre in (0, 5120) for q in range(1, a.qmax + 1)]
            for short in (1, 3, 6, 12, 24, 48, 100):
                for long_ in (200, 2000):
                    cases += [
                        ([short, long_], 5120),
                        ([long_, short], 5120),
                        ([short, short + 3, long_], 5120),
                    ]
            for dql in (4, 8):
                cases += [([q, dql, dql, dql], 5120) for q in (2, 7, 15, 30, 60, 90, 300)]
            for qs, pre in cases:
                qcpu, kvcpu, pcpu, kv_idx, npages = rows_tensors(qs, pre)
                tot = int(qcpu.sum())
                try:
                    # proxy: plan_for(proxy_rows, num_qo_heads=index heads, num_kv_heads=1, output_maxscore=True)
                    iq = torch.zeros(tot, hidx, hd, dtype=fp8, device=dev)
                    ik = torch.zeros(npages, 1, page, hd, dtype=fp8, device=dev)
                    plan = fmha.fmha_sm100_plan(
                        qcpu,
                        kvcpu,
                        qo_offset=pcpu,
                        page_size=page,
                        num_kv_splits=1,
                        causal=True,
                        num_qo_heads=hidx,
                        num_kv_heads=1,
                        output_maxscore=True,
                    )
                    fmha.fmha_sm100(
                        iq,
                        ik,
                        ik,
                        plan,
                        kv_indices=kv_idx,
                        output_o=False,
                        output_maxscore=True,
                        sm_scale=hd**-0.5,
                    )
                    # segmented KV-split proxy plan (None unless prefix >= 4096 and some row > 64)
                    seg = msa_indexer.segmented_proxy_plan(
                        fmha,
                        qcpu,
                        pcpu,
                        num_index_heads=hidx,
                        page_size=page,
                        kv_indices=kv_idx,
                    )
                    if seg is not None:
                        splan, skv = seg
                        sik = torch.zeros(int(skv.numel()), 1, page, hd, dtype=fp8, device=dev)
                        fmha.fmha_sm100(
                            iq,
                            sik,
                            sik,
                            splan,
                            kv_indices=skv,
                            output_o=False,
                            output_maxscore=True,
                            sm_scale=hd**-0.5,
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        "%s %s pre=%d: %s: %s"
                        % (gname, qs, pre, type(exc).__name__, str(exc)[:200])
                    )
            seen = sorted(set(rec))
            r = {
                "enumerated": sorted(enum),
                "recorded": seen,
                "missing": sorted(set(seen) - set(enum)),
                "unused": sorted(set(enum) - set(seen)),
                "lookups": len(rec),
                "cases": len(cases),
            }
            if a.cache_geometry == gname:
                r["cache_missing"] = sorted(n for n in seen if not core._so(jit, n).exists())
            results[gname] = r
    finally:
        api.get_fmha_variant, type(jit._variant_manager)._compile_only = orig_get, orig_compile

    ok = not errors and all(
        not r["missing"] and not r.get("cache_missing") for r in results.values()
    )
    summary = {
        "ok": ok,
        "plan_so_present_before": plan_present,
        "cache_base": str(jit.CACHE_BASE),
        "errors": errors[:20],
        "n_errors": len(errors),
        "geometries": results,
    }
    txt = json.dumps(summary, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(
        json.dumps(
            {
                "ok": ok,
                "n_errors": len(errors),
                "per_geometry": {
                    g: {
                        "enumerated": len(r["enumerated"]),
                        "recorded": len(r["recorded"]),
                        "missing": r["missing"],
                        "cache_missing": r.get("cache_missing"),
                    }
                    for g, r in results.items()
                },
            }
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
