#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821
# The per-grid buffers are released with an explicit `del` at the end of
# each loop body, which makes the analyzer treat every closure reference
# to them as possibly-unbound. All F821 reports here are that one idiom.
# Layer x step complete grid. One NVTX range per (arm, model, isl, B,
# layer); the COLD reps cycle through ALL usable decode steps (batch
# refilled from the step's row BEFORE the eviction, so the row itself
# stays cold) -> the per-range cold mean IS the all-steps mean for that
# layer. Tables aggregate mean-over-layers; per-layer means are the
# per-range values (saved separately).
#   env: ARM, UNITS ("flash:4k,pro:512k,..."), OUT, BS_LIST
import json
import os
import sys
from pathlib import Path

import torch

CAP = Path(
    os.environ.get(
        "GVR_CAP_ROOT",
        "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/"
        "E2E_exp/indexer_decode_capture/data",
    )
)
LAYERS = {"flash": list(range(2, 43, 2)), "pro": list(range(2, 61, 2)), "v32": list(range(61))}
KCFG = {"flash": (512, 4), "pro": (1024, 4), "v32": (2048, 1)}
ALIGN = 64
FMIN = torch.finfo(torch.float32).min

ARM = os.environ["ARM"]
OUT = os.environ["OUT"]
_LSUB = os.environ.get("LAYER_SUBSET")
if _LSUB:
    _ls = [int(x) for x in _LSUB.split(",")]
    LAYERS = {k: [x for x in v if x in _ls] for k, v in LAYERS.items()}
UNITS = [u.split(":") for u in os.environ["UNITS"].split(",")]
BS_LIST = [
    int(x) for x in os.environ.get("BS_LIST", "1,2,4,8,16,32,64,128,256,512,1024").split(",")
]
_fb = os.environ.get("FORCE_BM")
FORCE_BM = None if _fb is None else int(_fb)
FORCE_CS = int(os.environ.get("FORCE_CS", "0"))
CONLY = os.environ.get("CONLY", "0") == "1"  # codespell:ignore
CONLY_RANK = int(os.environ.get("CONLY_RANK", "0"))
_pw = os.environ.get("FORCE_PWR")
FORCE_PWR = None if _pw is None else (_pw == "1")
_tf = os.environ.get("FORCE_TAILFAST")
FORCE_TAILFAST = None if not _tf else (_tf == "1")
_nf = os.environ.get("FORCE_NOFINE")
FORCE_NOFINE = None if not _nf else (_nf == "1")
_nb = os.environ.get("FORCE_BINS")
FORCE_BINS = int(_nb) if _nb else None
_frt = os.environ.get("FORCE_FINERT")
FORCE_FINERT = None if not _frt else (_frt == "1")
_srt = os.environ.get("FORCE_SCATRT")
FORCE_SCATRT = None if not _srt else (_srt == "1")
_tm = os.environ.get("TIGHT_MULT")
TIGHT_MULT = float(_tm) if _tm else None
_nt = os.environ.get("FORCE_THREADS")
FORCE_THREADS = int(_nt) if _nt else None
_v3 = os.environ.get("FORCE_TAILV3")
FORCE_TAILV3 = None if not _v3 else (_v3 == "1")
_et = os.environ.get("FORCE_EXACTTAIL")
FORCE_EXACTTAIL = None if not _et else (_et == "1")
XSTATE = os.environ.get("XSTATE", "0") == "1"
_ft = os.environ.get("FORCE_TIGHTEN")
FORCE_TIGHTEN = None if _ft is None else bool(int(_ft))
sms = torch.cuda.get_device_properties(0).multi_processor_count
_EVICT = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")

if ARM == "pr":
    sys.path.insert(0, "/home/scratch.siyid_coreai/workspace")
    from prpkg16457.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
elif ARM == "bx":
    # loncheng PR #16877 head (BSX multi-tier): guard -> bsx dispatch,
    # guard-fail -> that branch's own in-tree kernel (its production path)
    sys.path.insert(0, "/home/scratch.siyid_coreai/wt-bsx/tensorrt_llm/_torch/cute_dsl_kernels")
    from blackwell.top_k.gvr_topk_decode import GvrTopKKernel  # noqa: E402
    from blackwell.top_k.gvr_topk_decode_bsx_dispatch import (  # noqa: E402
        bsx_topk,
        is_bsx_supported,
    )
else:
    _root = os.environ.get("HARNESS_ROOT", str(Path(__file__).resolve().parents[5]))
    sys.path.insert(0, _root + "/tests/scripts/cute_dsl_kernels/top_k")
    sys.path.insert(0, _root)
    import run_gvr_topk as rg  # noqa: E402
    from run_gvr_topk import GvrTopKKernel  # noqa: E402

    if ARM != "st":
        from run_gvr_topk import (
            emu_block_max,  # noqa: E402
            emu_cand_bucketed,
            emu_seed_counts,
            pack_seed,
        )


def wf_targets(model, K, N):
    if model == "flash":
        T = (4096, 2048, 1024)
    elif model == "v32":
        T = (16384, 8192, 3072)
    else:
        T = (4096, 3584, 1536) if N <= 40960 else (12288, 5120, 2048)
    return [min(t, N) for t in T]


def main():
    f = open(OUT, "w")
    for model, isl in UNITS:
        K, cr = KCFG[model]
        Ls = LAYERS[model]
        # ---- load per-layer step dicts (logits + captured topk) ----
        lgs, pks = {}, {}
        for L in Ls:
            d = CAP / model / f"ISL_{isl}" / f"layer_{L:02d}"
            lgs[L] = torch.load(d / "decode.logits.in.pt", map_location="cpu", weights_only=False)
            pks[L] = torch.load(d / "decode.topk.out.pt", map_location="cpu", weights_only=False)
        steps_all = sorted(pks[Ls[0]].keys())
        # warmup steps carry -1 sentinel topk (and even wrong-width
        # logits buffers): valid = captured topk present; a measured
        # step also needs the PREVIOUS step valid (its topk is the
        # preIdx seed)
        valid = [s for s in steps_all if int(pks[Ls[0]][s].max()) >= 0]
        vset = set(valid)
        usable = [s for s in valid if (s - 1) in vset]
        assert usable, f"no usable steps for {model}:{isl}"
        NS = {}
        for s in valid:
            NS[s] = max(int(pks[L][s].max()) + 1 for L in Ls)
        N_label = NS[valid[-1]]
        # unit-constant pad width: block_max is validated against the
        # BATCH buffer width (logits.shape[1]), so per-step bm must be
        # built on rows already padded to the unit max
        Npad_u = max((NS[s] + ALIGN - 1) // ALIGN * ALIGN for s in usable)
        # ---- per (layer, step) prep on GPU ----
        # row (padded), preIdx, lines, ref values, arm extras (1-row)
        prep = {}
        for L in Ls:
            for s in usable:
                Ns = NS[s]
                Npad = Npad_u
                row = torch.full((1, Npad), FMIN, dtype=torch.float32, device="cuda")
                src = lgs[L][s][0]
                row[0, :Ns] = src[:Ns].float().cuda()
                pre = pks[L][s - 1].flatten().to(torch.int32).view(1, K)
                pre = pre.cuda().contiguous()
                srt = torch.sort(row[0, :Ns], descending=True).values
                ref = srt[:K].clone()
                ks = wf_targets(model, K, Ns)
                t = torch.empty((1, 3), dtype=torch.float32, device="cuda")
                for j, kc in enumerate(ks):
                    t[0, j] = srt[kc - 1]
                if TIGHT_MULT:
                    # the closed loop derives its tight line from the
                    # PREVIOUS step's k-th value, i.e. it sits just above
                    # rank K - not at the fixed rank wf_targets uses. Model
                    # that here so the tiers are measured on the contract
                    # they actually ship with.
                    t[0, 2] = srt[min(int(TIGHT_MULT * K), Ns) - 1]
                for j in (1, 2):
                    t[0, j] = torch.maximum(t[0, j], t[0, j - 1] + 1e-6)
                if CONLY:  # codespell:ignore
                    # push both tight lines above the row max so every
                    # admitted entry lands in the loosest segment (the
                    # one that already uses the cheap claim window).
                    # CONLY_RANK tightens the collection line so the
                    # claimed count stays under the consumer's line-cut
                    # limit (and less volume is emitted).
                    if CONLY_RANK:
                        t[0, 0] = srt[min(CONLY_RANK * K, Ns) - 1]
                    hi = float(row[0, :Ns].max()) + 1e4
                    t[0, 1] = hi
                    t[0, 2] = hi + 1.0
                e = dict(Ns=Ns, Npad=Npad, row=row, pre=pre, ref=ref, sthr=t)
                if ARM not in ("pr", "st", "bx"):
                    sl1 = torch.tensor([Ns * cr], dtype=torch.int32, device="cuda")
                    scnt = emu_seed_counts(row, sl1, t, compress_ratio=cr)
                    if ARM == "wf":
                        cv, ci, ctl = emu_cand_bucketed(
                            row, sl1, t, 24576, seg_cap=8192, compress_ratio=cr, sentinel_pad=64
                        )
                        e.update(cv=cv, ci=ci, ctl=ctl)
                    # thresholds mirror gvr_routing (real-capture A/B)
                    nb = (ARM == "va" and Ns >= 65536) or (
                        ARM == "vb" and model == "flash" and Ns >= 131072
                    )
                    if FORCE_BM is not None:
                        nb = bool(FORCE_BM)
                    if nb:
                        e["bm"] = emu_block_max(row, sl1, compress_ratio=cr, tail_mode="exact")
                    # col 6 = adaptive-skip pass count when bm attached
                    e["spk"] = pack_seed(t, scnt, block_max=e.get("bm"))
                prep[(L, s)] = e
        Npad_max = max(e["Npad"] for e in prep.values())
        for B in BS_LIST:
            # reusable batch buffers (filled per rep BEFORE eviction)
            lg_b = torch.full((B, Npad_max), FMIN, dtype=torch.float32, device="cuda")
            pre_b = torch.zeros((B, K), dtype=torch.int32, device="cuda")
            sl_b = torch.zeros((B,), dtype=torch.int32, device="cuda")
            thr_b = torch.zeros((B, 3), dtype=torch.float32, device="cuda")
            xs_b = torch.zeros((B, 8), dtype=torch.float32, device="cuda") if XSTATE else None
            spk_b = torch.zeros((B, 8), dtype=torch.float32, device="cuda")
            bufs = {}
            if ARM in ("pr", "st", "bx"):
                bufs["outb"] = torch.empty(B, K, dtype=torch.int32, device="cuda")
            if ARM == "wf":
                # widths from the emu contract (2*seg_cap+cap for the
                # bucketed list), NOT the nominal cap
                e0 = next(iter(prep.values()))
                CW = e0["cv"].shape[1]
                bufs["cv"] = torch.zeros((B, CW), dtype=torch.float32, device="cuda")
                bufs["ci"] = torch.zeros((B, CW), dtype=torch.int32, device="cuda")
                bufs["ctl"] = torch.zeros((B, e0["ctl"].shape[1]), dtype=torch.int32, device="cuda")
            bm_b = None
            if ARM in ("va", "vb"):
                nbp = max((e["bm"].shape[1] for e in prep.values() if "bm" in e), default=0)
                if nbp:
                    bm_b = torch.zeros((B, nbp), dtype=torch.float32, device="cuda")
            state = {}

            def fill(L, s):
                e = prep[(L, s)]
                state.update(e=e)
                lg_b[:, : e["Npad"]].copy_(e["row"].expand(B, e["Npad"]))
                if e["Npad"] < Npad_max:
                    lg_b[:, e["Npad"] :].fill_(FMIN)
                pre_b.copy_(e["pre"].expand(B, K))
                sl_b.fill_(e["Ns"] * cr)
                thr_b.copy_(e["sthr"].expand(B, 3))
                if ARM not in ("pr", "st", "bx"):
                    spk_b.copy_(e["spk"].expand(B, 8))
                if ARM == "wf":
                    bufs["cv"].copy_(e["cv"].expand_as(bufs["cv"]))
                    bufs["ci"].copy_(e["ci"].expand_as(bufs["ci"]))
                    bufs["ctl"].copy_(e["ctl"].expand_as(bufs["ctl"]))
                if bm_b is not None and "bm" in e:
                    # unit-constant Npad -> bm width == bm_b width
                    bm_b.copy_(e["bm"].expand_as(bm_b))
                    state["bmc"] = bm_b
                torch.cuda.synchronize()

            def akw():
                e = state["e"]
                Ns = e["Ns"]
                kw = dict(next_n=1, compress_ratio=cr, num_sms=sms, return_output_values=False)
                if xs_b is not None:
                    kw["xstate"] = xs_b
                if FORCE_TAILFAST is not None:
                    kw["p4_tail_fast"] = FORCE_TAILFAST
                if FORCE_EXACTTAIL is not None:
                    kw["p4_exact_tail"] = FORCE_EXACTTAIL
                if FORCE_TAILV3 is not None:
                    kw["p4_tail_v3"] = FORCE_TAILV3
                if FORCE_NOFINE is not None:
                    kw["p4_no_fine"] = FORCE_NOFINE
                if FORCE_BINS is not None:
                    kw["num_bins"] = FORCE_BINS
                if FORCE_FINERT is not None:
                    kw["p4_fine_rangetest"] = FORCE_FINERT
                if FORCE_SCATRT is not None:
                    kw["p4_scat_rangetest"] = FORCE_SCATRT
                if FORCE_THREADS is not None:
                    kw["num_threads_per_block"] = FORCE_THREADS
                if ARM == "sw":
                    return dict(kw)  # wrapper stock: no seed, no xstate
                if ARM == "wf":
                    a = dict(
                        seed_thr=spk_b,
                        cand_vals=bufs["cv"],
                        cand_idx=bufs["ci"],
                        cand_ctl=bufs["ctl"],
                        cluster_size=1,
                        **kw,
                    )
                    if FORCE_PWR is not None:
                        a["p4_warp_redundant"] = FORCE_PWR
                    return a
                if ARM == "va":
                    a = dict(seed_thr=spk_b, cluster_size=FORCE_CS or 1, **kw)
                    if "bm" in e:
                        a.update(block_max=state["bmc"], skip_min_n=None)
                    if FORCE_PWR is not None:
                        a["p4_warp_redundant"] = FORCE_PWR
                    return a
                # vb
                cs = 1
                if Ns >= 196608:
                    if B * 8 <= sms // 2:
                        cs = 8
                    elif B * 4 <= (sms * 9) // 10:
                        cs = 4
                    elif B * 2 <= (sms * 9) // 10:
                        cs = 2
                a = dict(seed_thr=thr_b, cluster_size=cs, **kw)
                if "bm" in e:
                    a.update(block_max=state["bmc"], skip_min_n=None)
                if FORCE_CS:
                    a["cluster_size"] = FORCE_CS
                if FORCE_PWR is not None:
                    a["p4_warp_redundant"] = FORCE_PWR
                return a

            def call():
                if ARM == "bx":
                    if "bx_ok" not in state:
                        state["bx_ok"] = is_bsx_supported(
                            lg_b, pre_b, sl_b, bufs["outb"], K, 1, cr, None, None
                        )
                    if state["bx_ok"]:
                        bsx_topk(lg_b, pre_b, sl_b, bufs["outb"], K, 1, cr)
                    else:
                        GvrTopKKernel.launch(lg_b, pre_b, sl_b, bufs["outb"], K, compress_ratio=cr)
                    return
                if ARM in ("pr", "st"):
                    ov = {}
                    if ARM == "st" and FORCE_PWR is not None:
                        ov["p4_warp_redundant"] = FORCE_PWR
                    GvrTopKKernel.launch(
                        lg_b, pre_b, sl_b, bufs["outb"], K, compress_ratio=cr, **ov
                    )
                else:
                    rg.gvr_topk_decode(lg_b, pre_b, sl_b, K, **akw())

            def out_idx():
                if ARM in ("pr", "st", "bx"):
                    call()
                    torch.cuda.synchronize()
                    return bufs["outb"]
                _, o = rg.gvr_topk_decode(lg_b, pre_b, sl_b, K, **akw())
                torch.cuda.synchronize()
                return o

            for L in Ls:
                base = f"{ARM}|{model}|{isl}|N{N_label}|B{B}|L{L:02d}"
                rec = dict(
                    model=model, isl=isl, N=N_label, K=K, B=B, arm=ARM, layer=L, steps=len(usable)
                )
                try:
                    # exactness: every step at B==min; spot at larger B
                    chk = usable if B == BS_LIST[0] else usable[:1]
                    ok = True
                    for s in chk:
                        fill(L, s)
                        o = out_idx()
                        e = prep[(L, s)]
                        for i in (0, B - 1) if B > 1 else (0,):
                            gid = o[i, :K].long()
                            # gather with an out-of-range index raises a
                            # device-side assert that poisons the context and
                            # kills the whole worker -> screen first
                            if int(gid.min()) < 0 or int(gid.max()) >= e["Ns"]:
                                ok = False
                                rec["oob"] = rec.get("oob", 0) + 1
                                continue
                            got = lg_b[i, gid].sort(descending=True).values
                            if gid.unique().numel() != K or not torch.equal(got, e["ref"]):
                                ok = False
                        if not ok:
                            break
                    rec["exact"] = ok
                    # warmup
                    for s in usable[:4]:
                        fill(L, s)
                        call()
                    torch.cuda.synchronize()
                    # COLD: one rep per decode step -> range mean = all-
                    # steps mean for this layer
                    xrec = []
                    for s in usable:
                        fill(L, s)
                        if xs_b is not None:
                            xs_b.zero_()
                        _EVICT.uniform_(0, 1)
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_push(f"c|{base}")
                        call()
                        torch.cuda.synchronize()
                        torch.cuda.nvtx.range_pop()
                        if xs_b is not None:
                            xrec.append([round(v, 1) for v in xs_b[0].tolist()])
                    if xs_b is not None:
                        rec["x"] = xrec
                    torch.cuda.synchronize()
                except Exception as ex:  # noqa: BLE001
                    rec["error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
            del lg_b, pre_b, sl_b, thr_b, spk_b, bufs, bm_b, xs_b
            torch.cuda.empty_cache()
        prep.clear()
        lgs.clear()
        pks.clear()
        torch.cuda.empty_cache()
        print(f"[{ARM}] {model} {isl} done", flush=True)
    f.close()
    print(f"{ARM}_STEPS_DONE")


if __name__ == "__main__":
    main()
