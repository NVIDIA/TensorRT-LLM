#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 / Goal-2.2 -- HUMAN FEEDBACK #2 DIRECTIVE 2.4: two-backend
CUTLASS-vs-TRTLLM (trtllm-gen) isolation ON THE VISION PATH.

The human's dispute (feedback #2, HOLE B): the "residual is the accepted
fa4-vs-Triton kernel-family class" claim was an ARGUMENT BY ANALOGY to the text
tower -- never MEASURED on the vision path. This test measures it directly.

Runs the SAME discriminating IMAGE items (SGLang-right / TRT-wrong in the Goal 2.1
5x100 rounds, drawn from ``MMMU_ALIGN_ITEMS``) through TWO DIFFERENT NVFP4 MoE
kernel families on the identical baseline stack -- CUTLASS (production) and TRTLLM
(trtllm-gen) -- with TP=4 + TRTLLM attention + KVCacheManagerV2 + hMLP vision
fusion, cuda_graph=OFF, overlap=OFF, deterministic bs=1 (autotuner off, the
measurement-hygiene switch), and cross-compares them item-for-item:

  * final parsed answer + correctness -- do the two fp4 kernels reach the SAME
    MMMU answer on the same image, and does either match gold?
  * pos0 first generated token off the vision-fused prefill -- argmax + top-K
    logprob cosine (CUTLASS vs TRTLLM), the single cleanest cross-kernel point
    (identical image + prompt, no track needed);
  * first-divergence step in the two deterministic greedy tracks -- both share the
    identical prefix while their tokens agree, so the first step where the argmax
    differs is a well-defined cross-kernel fork; the per-step top-K cosine AND the
    top1-top2 logprob MARGIN at that step say whether the fork is a near-tie
    (accuracy-neutral fp4 noise) or a decisive kernel disagreement.

DECISION RULE (feedback #2 Directive 2.4, mirroring the text-tower
``inkling_moe_backend_isolate_test.py`` logic, now on the vision path):
  * CUTLASS and TRTLLM AGREE per-step (high pos0 cos, late/no divergence, same
    answers) => the vision-path residual is NOT explained by the fp4 MoE kernel
    choice; two different fp4 kernel families produce the same vision decode, so
    the residual is NOT a CUTLASS-specific bug fixable by swapping kernels
    (kernel-INDEPENDENT -- the accepted fp4-kernel class, now MEASURED not
    inherited).
  * CUTLASS and TRTLLM DISAGREE among themselves at magnitudes comparable to the
    TRT-vs-SGLang gap => two fp4 MoE kernels on identical weights already diverge
    at the token level, so per-step parity with SGLang's THIRD (flashinfer) fp4
    kernel is infeasible for ANY faithful implementation -- the divergence is the
    fp4-kernel-family class, and the accuracy gate (Goal 2.1 PASS: mean_delta
    -0.018, within 2pt, adequately powered) is the decider.
Either outcome MEASURES the kernel claim on the vision path. real_gap=False from
Goal 2.1 means there is no large gap to pin here; this test confirms whether the
small residual is kernel-family dependent.

Modes (env INKLING_IMG_ISO_MODE), so the two TP=4 loads run as SEPARATE processes
(robust -- no two-executor-in-one-process teardown risk):
  * ``dump``   : load ONE backend (INKLING_MOE_BACKEND), greedy-decode every item,
                 write tracks+top-K to INKLING_IMG_ISO_DUMP_OUT.
  * ``compare``: CPU-only; read two dumps (INKLING_IMG_ISO_DUMP_A / _DUMP_B),
                 cross-compare, write the isolation artifact to INKLING_IMG_ISO_OUT.
  * ``both``   : (default) both backends in one process + inline compare.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_image_moe_isolate.py
Env: INKLING_CHECKPOINT, MMMU_ALIGN_ITEMS (Config:offset csv), MMMU_ALIGN_CACHE,
     INKLING_IMG_ISO_OUT, INKLING_IMG_ISO_NGEN (default 256),
     INKLING_IMG_ISO_TOPK (default 20), INKLING_TP (default 4),
     INKLING_IMG_ISO_NEARTIE_MARGIN (default 0.5 nats),
     INKLING_IMG_ISO_MODE, INKLING_MOE_BACKEND, INKLING_IMG_ISO_DUMP_OUT,
     INKLING_IMG_ISO_DUMP_A, INKLING_IMG_ISO_DUMP_B.
"""

import gc
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full",
)
OUT = os.environ.get(
    "INKLING_IMG_ISO_OUT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/"
    "img_moe_isolate.json",
)
NGEN = int(os.environ.get("INKLING_IMG_ISO_NGEN", "256"))
TOPK = int(os.environ.get("INKLING_IMG_ISO_TOPK", "20"))
TP = int(os.environ.get("INKLING_TP", "4"))
BACKENDS = [
    b.strip().upper()
    for b in os.environ.get("INKLING_IMG_ISO_BACKENDS", "CUTLASS,TRTLLM").split(",")
    if b.strip()
]
NEARTIE_MARGIN = float(os.environ.get("INKLING_IMG_ISO_NEARTIE_MARGIN", "0.5"))
MODE = os.environ.get("INKLING_IMG_ISO_MODE", "both").strip().lower()


def _cos_maxabs(da, db):
    """cos + max_abs over the shared-token support of two {token_id: logprob}."""
    import torch

    ids = [t for t in da if t in db]
    if len(ids) < 2:
        return float("nan"), float("nan")
    a = torch.tensor([da[i] for i in ids])
    b = torch.tensor([db[i] for i in ids])
    return (
        float(torch.nn.functional.cosine_similarity(a[None], b[None]).item()),
        float((a - b).abs().max()),
    )


def _top_margin(lpd):
    """top1-top2 logprob gap (nats) of one per-step {token_id: logprob} dict."""
    if not lpd or len(lpd) < 2:
        return float("nan")
    vals = sorted(lpd.values(), reverse=True)
    return float(vals[0] - vals[1])


def _build_recs():
    import inkling_image_prompts as P

    n = len([t for t in (os.environ.get("MMMU_ALIGN_ITEMS") or "").split(",") if t.strip()]) or int(
        os.environ.get("INKLING_MMMU_N", "5")
    )
    recs = P.build_prompts(n, with_num_patches=True)
    # PER-ITEM image-use assertion (feedback #2 D1): every item genuinely uses its
    # image (nonzero hMLP patch rows + exactly one placeholder) or the run fails.
    for r in recs:
        n_ph = sum(1 for t in r["input_ids"] if t == P.IMAGE_TOKEN_ID)
        assert int(r.get("num_patches", 0)) > 0 and n_ph == 1, (
            f"item {r['id']} image-blind (npatch={r.get('num_patches')} n_ph={n_ph})"
        )
    return recs


def _run_backend(backend, recs):
    """Free deterministic greedy decode of every image item under one MoE backend.

    Returns {id: {"track":[ids], "topk":[{id:lp},...], "parsed":X, "score":s,
                  "n_gen":n, "gold":g, "finish":f, "num_patches":p}}."""
    import io

    import inkling_mmmu_harness as H
    import inkling_mmmu_real_align_test as R
    import torch
    from PIL import Image
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    items_by_id = {it["id"]: it for it in R.load_fixed_items()}
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)

    print(f"\n[img-iso] ===== loading backend={backend} (TP={TP}) =====", flush=True)
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=backend),
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=0.7, dtype="auto", enable_block_reuse=False
        ),
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_autotuner=False,  # deterministic: no autotuner tactic jitter
        max_seq_len=8192,
        max_batch_size=1,
        max_num_tokens=8192,
    )
    print(
        f"[img-iso] backend={backend} loaded; cuda_graph=eager(no-graph) "
        f"overlap=off deterministic bs=1 autotuner=off",
        flush=True,
    )

    def _img(r):
        return Image.open(io.BytesIO(r["image_bytes"]))

    res = {}
    try:
        for r in recs:
            out = llm.generate(
                [
                    TokensPrompt(
                        prompt_token_ids=list(r["input_ids"]), multi_modal_data={"image": [_img(r)]}
                    )
                ],
                SamplingParams(max_tokens=NGEN, temperature=0.0, logprobs=TOPK),
            )
            gen = out[0].outputs[0]
            track = [int(x) for x in gen.token_ids]
            topk = []
            for step in gen.logprobs or []:
                if isinstance(step, dict):
                    topk.append({int(k): float(getattr(v, "logprob", v)) for k, v in step.items()})
                else:
                    topk.append({})
            it = items_by_id[r["id"]]
            gold = it.get("answer")
            options = it.get("options")
            qtype = it.get("question_type") or ("multiple-choice" if options else "open")
            index2ans, all_choices = H.build_mc_mapping(options) if options else (None, None)
            text = getattr(gen, "text", "") or tok.decode(track)
            score, parsed = H.score_sample(text, gold, qtype, all_choices, index2ans)
            res[r["id"]] = {
                "track": track,
                "topk": topk,
                "parsed": parsed,
                "score": float(score),
                "n_gen": len(track),
                "gold": gold,
                "finish": getattr(gen, "finish_reason", None),
                "num_patches": int(r.get("num_patches", 0)),
            }
            print(
                f"  [{backend}][{r['id']:<28}] gold={gold} parsed={parsed} "
                f"score={score} n_gen={len(track)} npatch={r.get('num_patches')} "
                f"finish={getattr(gen, 'finish_reason', None)}",
                flush=True,
            )
    finally:
        llm.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return res


def _norm_topk(topk):
    """JSON round-trips dict keys to str; normalize per-step top-K keys back to int."""
    out = []
    for step in topk:
        out.append({int(k): float(v) for k, v in step.items()})
    return out


def _compare(a, b, ra, rb):
    """Cross-compare two per-item backend dicts; return (rows, aggregate result)."""
    rows = []
    ids = [i for i in ra if i in rb]
    for iid in ids:
        ea, eb = ra[iid], rb[iid]
        ta, tb = _norm_topk(ea["topk"]), _norm_topk(eb["topk"])
        pos0_a = ta[0] if ta else {}
        pos0_b = tb[0] if tb else {}
        pos0_argmax_a = ea["track"][0] if ea["track"] else None
        pos0_argmax_b = eb["track"][0] if eb["track"] else None
        pos0_cos, pos0_maxabs = _cos_maxabs(pos0_a, pos0_b)
        L = min(len(ea["track"]), len(eb["track"]))
        first_div = -1
        for t in range(L):
            if ea["track"][t] != eb["track"][t]:
                first_div = t
                break
        lock = first_div if first_div >= 0 else L
        step_cos = []
        for t in range(lock):
            if t < len(ta) and t < len(tb):
                c, _ = _cos_maxabs(ta[t], tb[t])
                if c == c:
                    step_cos.append(c)
        min_lock_cos = min(step_cos) if step_cos else float("nan")
        if first_div >= 0:
            marg_a = _top_margin(ta[first_div]) if first_div < len(ta) else float("nan")
            marg_b = _top_margin(tb[first_div]) if first_div < len(tb) else float("nan")
            fork_cos = float("nan")
            if first_div < len(ta) and first_div < len(tb):
                fork_cos, _ = _cos_maxabs(ta[first_div], tb[first_div])
            near_tie = (marg_a == marg_a and marg_a < NEARTIE_MARGIN) or (
                marg_b == marg_b and marg_b < NEARTIE_MARGIN
            )
        else:
            marg_a = marg_b = fork_cos = float("nan")
            near_tie = None
        rows.append(
            {
                "id": iid,
                "gold": ea.get("gold"),
                "num_patches": ea.get("num_patches"),
                f"{a}_parsed": ea["parsed"],
                f"{a}_score": ea["score"],
                f"{a}_n_gen": ea["n_gen"],
                f"{a}_finish": ea.get("finish"),
                f"{b}_parsed": eb["parsed"],
                f"{b}_score": eb["score"],
                f"{b}_n_gen": eb["n_gen"],
                f"{b}_finish": eb.get("finish"),
                "answer_agree": (ea["parsed"] == eb["parsed"]),
                "pos0_argmax_agree": (pos0_argmax_a == pos0_argmax_b),
                "pos0_cos": pos0_cos,
                "pos0_maxabs": pos0_maxabs,
                "first_divergence_step": first_div,
                "lockstep_len": lock,
                "min_lockstep_cos": min_lock_cos,
                f"fork_margin_{a}": marg_a,
                f"fork_margin_{b}": marg_b,
                "fork_cos": fork_cos,
                "fork_near_tie": near_tie,
            }
        )
        print(
            f"\n  [cross {iid}] answer_agree={ea['parsed'] == eb['parsed']} "
            f"({a}={ea['parsed']}/{ea['score']:.0f} "
            f"{b}={eb['parsed']}/{eb['score']:.0f} gold={ea.get('gold')})\n"
            f"        pos0: argmax_agree={pos0_argmax_a == pos0_argmax_b} "
            f"cos={pos0_cos:.6f} maxabs={pos0_maxabs:.4f}\n"
            f"        first_divergence_step={first_div} (lockstep_len={lock} "
            f"min_lockstep_cos={min_lock_cos:.6f})\n"
            f"        fork: margin_{a}={marg_a:.4f} margin_{b}={marg_b:.4f} "
            f"cos={fork_cos:.6f} near_tie={near_tie}",
            flush=True,
        )

    n_items = len(rows)
    n_answer_agree = sum(int(x["answer_agree"]) for x in rows)
    n_pos0_argmax_agree = sum(int(x["pos0_argmax_agree"]) for x in rows)
    pos0_coss = [x["pos0_cos"] for x in rows if x["pos0_cos"] == x["pos0_cos"]]
    mean_pos0_cos = sum(pos0_coss) / len(pos0_coss) if pos0_coss else float("nan")
    min_pos0_cos = min(pos0_coss) if pos0_coss else float("nan")
    diverged = [x for x in rows if x["first_divergence_step"] >= 0]
    n_diverged = len(diverged)
    n_neartie = sum(int(bool(x["fork_near_tie"])) for x in diverged)
    all_forks_neartie = (n_diverged == 0) or (n_neartie == n_diverged)
    high_agreement = (mean_pos0_cos >= 0.99) if mean_pos0_cos == mean_pos0_cos else False
    verdict = "KERNEL_INDEPENDENT" if (high_agreement and all_forks_neartie) else "KERNEL_DEPENDENT"
    result = {
        "test": "inkling_image_moe_isolate",
        "goal": "Stage2-Goal2.2 / human feedback #2 Directive 2.4",
        "backends": [a, b],
        "tp": TP,
        "ngen": NGEN,
        "topk": TOPK,
        "cuda_graph": False,
        "overlap": False,
        "deterministic_bs1": True,
        "neartie_margin_nats": NEARTIE_MARGIN,
        "n_items": n_items,
        "n_answer_agree": n_answer_agree,
        "n_pos0_argmax_agree": n_pos0_argmax_agree,
        "mean_pos0_cos": mean_pos0_cos,
        "min_pos0_cos": min_pos0_cos,
        "n_diverged": n_diverged,
        "n_diverged_near_tie": n_neartie,
        "all_forks_near_tie": all_forks_neartie,
        "verdict": verdict,
        "rows": rows,
    }
    return rows, result


def _emit(result, a, b):
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[img-iso] {a} vs {b} on {result['n_items']} discriminating image items:", flush=True)
    print(
        f"  answer_agree={result['n_answer_agree']}/{result['n_items']} "
        f"pos0_argmax_agree={result['n_pos0_argmax_agree']}/{result['n_items']} "
        f"mean_pos0_cos={result['mean_pos0_cos']:.6f} "
        f"min_pos0_cos={result['min_pos0_cos']:.6f}",
        flush=True,
    )
    print(
        f"  diverged={result['n_diverged']}/{result['n_items']} of which "
        f"near_tie={result['n_diverged_near_tie']} "
        f"(all_forks_near_tie={result['all_forks_near_tie']})",
        flush=True,
    )
    explain = (
        "two fp4 kernels agree except near-tie forks => residual NOT "
        "CUTLASS-specific (kernel-family class, measured on vision)"
        if result["verdict"] == "KERNEL_INDEPENDENT"
        else "two fp4 kernels decisively diverge => kernel-DEPENDENT; first_divergence localizes it"
    )
    print(
        f"INKLING_IMG_MOE_ISOLATE verdict={result['verdict']} backends={a},{b} "
        f"answer_agree={result['n_answer_agree']}/{result['n_items']} "
        f"pos0_argmax_agree={result['n_pos0_argmax_agree']}/{result['n_items']} "
        f"mean_pos0_cos={result['mean_pos0_cos']:.6f} "
        f"min_pos0_cos={result['min_pos0_cos']:.6f} "
        f"diverged={result['n_diverged']}/{result['n_items']} "
        f"near_tie={result['n_diverged_near_tie']}/{result['n_diverged']} "
        f"out={OUT} ({explain})",
        flush=True,
    )


def main() -> int:
    if MODE == "compare":
        # CPU-only: read the two per-backend dumps and cross-compare.
        da = json.load(open(os.environ["INKLING_IMG_ISO_DUMP_A"]))
        db = json.load(open(os.environ["INKLING_IMG_ISO_DUMP_B"]))
        a, b = da["backend"], db["backend"]
        _, result = _compare(a, b, da["items"], db["items"])
        _emit(result, a, b)
        return 0

    import torch

    assert torch.cuda.is_available(), "image two-backend isolation needs CUDA GPUs"
    recs = _build_recs()
    print(
        f"[img-iso] mode={MODE} items={[r['id'] for r in recs]} "
        f"ngen={NGEN} topk={TOPK} neartie_margin={NEARTIE_MARGIN}",
        flush=True,
    )

    if MODE == "dump":
        backend = os.environ["INKLING_MOE_BACKEND"].strip().upper()
        res = _run_backend(backend, recs)
        dump_out = os.environ["INKLING_IMG_ISO_DUMP_OUT"]
        os.makedirs(os.path.dirname(dump_out) or ".", exist_ok=True)
        with open(dump_out, "w") as f:
            json.dump({"backend": backend, "tp": TP, "ngen": NGEN, "topk": TOPK, "items": res}, f)
        print(f"INKLING_IMG_ISO_DUMP_OK backend={backend} n={len(res)} out={dump_out}", flush=True)
        return 0

    # mode == "both": both backends in one process + inline compare.
    assert len(BACKENDS) == 2, f"need exactly 2 backends, got {BACKENDS}"
    a, b = BACKENDS
    ra = _run_backend(a, recs)
    rb = _run_backend(b, recs)
    _, result = _compare(a, b, ra, rb)
    _emit(result, a, b)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
