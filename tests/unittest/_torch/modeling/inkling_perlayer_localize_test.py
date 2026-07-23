#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goal 1.10 / feedback #4 -- deterministic teacher-forced per-kernel localization,
TRT side (Stage A worst-step + Stage B per-module h_attn/moe_out dump).

Runs under the ACCEPTED Goal 1.9 deterministic config (max_batch_size=1,
enable_autotuner=False, TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1, TP=4, cuda_graph=off,
overlap=off, same NVFP4 checkpoint) so per-step / per-layer diffs are signal, not
run-to-run noise. The SGLang **triton** attention backend is the debugging
reference (results/sglang_gsm8k_disc_ref_TRITON.json); SGLang-fa4 stays the
official gate and is untouched.

Two GPU modes (env INKLING_LOCALIZE_MODE), because the per-layer dump env must be
set STATICALLY at srun launch (post-launch os.environ writes do NOT reach the TP
workers), so teacher-forcing (dump OFF) and the module dump (dump ON) are separate
srun steps sharing one bootstrapped container:

  * worst  -- Stage A. For each SGLang-correct candidate: free-run TRT to confirm
              TRT is WRONG (discriminating), then teacher-force TRT with SGLang's
              greedy tokens and record the worst decode step (max final-logit
              divergence). Writes worst_steps.json (>=3 discriminating required).
  * dump   -- Stage B (TRT half). For each discriminating sample, prefill the
              SGLang-teacher-forced prefix up to its worst token and dump per-layer
              h_attn / moe_out (answer position, all 66 layers) via the
              INKLING_DUMP_ALLLAYERS + INKLING_DUMP_MODULES model hook.

  * selftest -- CPU-only. Exercises the pure selection / worst-step / collision
                logic on synthetic data (no CUDA), so the harness logic is
                Reviewer-verifiable without a GPU allocation.

Run (GPU): trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_perlayer_localize_test.py
Run (CPU): INKLING_LOCALIZE_MODE=selftest python tests/unittest/_torch/modeling/inkling_perlayer_localize_test.py
"""
import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")
DISC_REF = os.environ.get(
    "INKLING_DISC_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-bringup/results/"
    "sglang_gsm8k_disc_ref_TRITON.json")
OUTDIR = os.environ.get(
    "INKLING_LOCALIZE_OUTDIR",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-bringup/results/localize_trt")
MODE = os.environ.get("INKLING_LOCALIZE_MODE", "worst")
TP = int(os.environ.get("INKLING_TP", "4"))
# Teacher-force horizon per sample (capped): covers the bulk of SGLang's committed
# answer while bounding restart cost. Raise to reach a late stop-decision divergence.
MAXSTEP = int(os.environ.get("INKLING_LOCALIZE_MAXSTEP", "256"))
MINSTEP = int(os.environ.get("INKLING_LOCALIZE_MINSTEP", "32"))
FREERUN_CAP = int(os.environ.get("INKLING_LOCALIZE_FREERUN_CAP", "640"))
NEED_DISC = int(os.environ.get("INKLING_LOCALIZE_NEED_DISC", "3"))
MOE_BACKEND = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
# Fixed seq budget so the KV-estimation warmup prefill (~max_num_tokens) stays far
# outside the Stage-B module-dump token window [min,max prefix_len] (~350-900),
# exactly like the proven inkling_perlayer_trt window. All prefixes fit: free-run
# n_input(<=560)+cap(640) and teacher-force n_input+MAXSTEP are both < 2048.
MAXSEQ = int(os.environ.get("INKLING_LOCALIZE_MAXSEQ", "2048"))

GOLD_RE = re.compile(r"####\s*([\-0-9\.,]+)")
STOP_TOKENS = {200006, 200010}


def _extract_answer(text):
    m = GOLD_RE.search(text or "")
    return m.group(1).replace(",", "").strip() if m else None


# ---------------------------------------------------------------------------
# Pure logic (CPU-testable): selection + worst-step + prefix-length uniqueness.
# ---------------------------------------------------------------------------
def worst_step_of(steps):
    """The step dict with the lowest finite cosine (max TRT-vs-SGLang logit
    divergence). ``steps`` is a list of {t, cos, max_abs, sg, trt}. Returns None
    if no step carries a finite cosine."""
    worst = None
    for s in steps:
        c = s.get("cos")
        if c is None or (isinstance(c, float) and math.isnan(c)):
            continue
        if worst is None or c < worst["cos"]:
            worst = s
    return worst


def is_discriminating(rec):
    """SGLang correct AND TRT wrong on this prompt."""
    return bool(rec.get("sg_correct")) and bool(rec.get("trt_wrong"))


def enforce_unique_prefix_lens(selected):
    """Stage B dumps one file per prefix token-count (trt_mod.n<len>.rank0), so two
    discriminating samples must not share n_input+worst_t. On a collision, bump the
    later sample to its next-most-divergent step with a distinct prefix_len (its
    steps are pre-sorted by ascending cosine). Mutates + returns ``selected``;
    records the adjustment in each row. Raises if a sample cannot be made unique."""
    used = {}
    for rec in selected:
        plen = rec["prefix_len"]
        if plen not in used:
            used[plen] = rec["idx"]
            continue
        alt = None
        for s in rec.get("ranked_steps", []):
            cand = rec["n_input"] + s["t"]
            if cand not in used:
                alt = s
                break
        if alt is None:
            raise ValueError(
                f"idx {rec['idx']} prefix_len {plen} collides and has no distinct "
                f"alternative divergent step (collides with idx {used[plen]})")
        rec["worst_t"] = alt["t"]
        rec["worst_cos"] = alt.get("cos")
        rec["worst_max_abs"] = alt.get("max_abs")
        rec["prefix_len"] = rec["n_input"] + alt["t"]
        rec["prefix_len_adjusted"] = True
        used[rec["prefix_len"]] = rec["idx"]
    return selected


def _selftest():
    """CPU-only proof of the pure logic (no CUDA / no model)."""
    ok = 0
    fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  [selftest] PASS {name}")
        else:
            fail += 1
            print(f"  [selftest] FAIL {name}")

    steps = [{"t": 0, "cos": 0.999, "max_abs": 0.1, "sg": 5, "trt": 5},
             {"t": 1, "cos": 0.80, "max_abs": 4.0, "sg": 7, "trt": 9},
             {"t": 2, "cos": 0.95, "max_abs": 1.0, "sg": 3, "trt": 3},
             {"t": 3, "cos": None, "max_abs": None, "sg": 4, "trt": -1}]
    w = worst_step_of(steps)
    chk("worst_step picks lowest finite cos (t=1)", w is not None and w["t"] == 1)
    chk("worst_step ignores None cos",
        worst_step_of([{"t": 0, "cos": None, "sg": 1, "trt": 1}]) is None)

    chk("discriminating = sg_correct AND trt_wrong",
        is_discriminating({"sg_correct": True, "trt_wrong": True})
        and not is_discriminating({"sg_correct": True, "trt_wrong": False})
        and not is_discriminating({"sg_correct": False, "trt_wrong": True}))

    # collision: two samples land on prefix_len 100; the 2nd is bumped to its next
    # divergent step (ranked ascending by cos) that yields a distinct prefix_len.
    sel = [
        {"idx": 10, "n_input": 90, "worst_t": 10, "prefix_len": 100,
         "ranked_steps": [{"t": 10, "cos": 0.7}]},
        {"idx": 20, "n_input": 80, "worst_t": 20, "prefix_len": 100,
         "ranked_steps": [{"t": 20, "cos": 0.72}, {"t": 25, "cos": 0.75}]},
    ]
    sel = enforce_unique_prefix_lens(sel)
    lens = sorted(r["prefix_len"] for r in sel)
    chk("collision bumped to unique prefix_lens", len(set(lens)) == 2)
    chk("bumped row flagged + repointed to t=25",
        sel[1]["prefix_len_adjusted"] and sel[1]["worst_t"] == 25
        and sel[1]["prefix_len"] == 105)

    raised = False
    try:
        enforce_unique_prefix_lens(
            [{"idx": 1, "n_input": 10, "worst_t": 5, "prefix_len": 15,
              "ranked_steps": [{"t": 5, "cos": 0.7}]},
             {"idx": 2, "n_input": 10, "worst_t": 5, "prefix_len": 15,
              "ranked_steps": [{"t": 5, "cos": 0.7}]}])
    except ValueError:
        raised = True
    chk("no distinct alternative -> raises", raised)

    chk("_extract_answer parses #### and strips commas",
        _extract_answer("blah\n#### 1,234") == "1234"
        and _extract_answer("no answer here") is None)

    print(f"\nINKLING_LOCALIZE_SELFTEST {'OK' if fail == 0 else 'FAIL'} "
          f"pass={ok} fail={fail}")
    print(f"=== INKLING_LOCALIZE_SELFTEST_DONE rc={0 if fail == 0 else 1} ===")
    return 0 if fail == 0 else 1


# ---------------------------------------------------------------------------
# GPU paths.
# ---------------------------------------------------------------------------
def _load_ref():
    with open(DISC_REF) as f:
        doc = json.load(f)
    prompts = doc["prompts"] if isinstance(doc, dict) else doc
    return doc, prompts


def _build_llm():
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig
    # Accepted Goal 1.9 deterministic config: bs=1 + autotuner off. TP all-reduce
    # autotune + cublas workspace pinning come from the sbatch env.
    llm = LLM(
        CKPT, tensor_parallel_size=TP, trust_remote_code=True,
        attn_backend="TRTLLM", moe_config=MoeConfig(backend=MOE_BACKEND),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75,
                                      dtype="auto", enable_block_reuse=False),
        cuda_graph_config=None, disable_overlap_scheduler=True,
        enable_autotuner=False, max_seq_len=MAXSEQ,
        max_batch_size=1, max_num_tokens=MAXSEQ)
    return llm


def _provenance():
    """Recoverable job/config/source provenance for the localization artifacts."""
    sha = os.environ.get("INKLING_DET_REPO_SHA", "")
    if not sha:
        import subprocess
        try:
            sha = subprocess.run(["git", "-C", _HERE, "rev-parse", "HEAD"],
                                 capture_output=True, text=True,
                                 timeout=30).stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001
            sha = "unknown"
    return {
        "job_id": (os.environ.get("INKLING_DET_JOB_ID")
                   or os.environ.get("SLURM_JOB_ID", "")),
        "repo_sha": sha,
        "checkpoint": CKPT,
        "disc_ref": DISC_REF,
        "tp": TP, "moe_backend": MOE_BACKEND,
        "cuda_graph": False, "overlap": False,
        "max_batch_size": 1, "enable_autotuner": False,
        "allreduce_autotune_disabled":
        os.environ.get("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "maxstep": MAXSTEP, "freerun_cap": FREERUN_CAP,
    }


def run_worst():
    """Stage A: confirm discriminating + find each sample's worst decode step."""
    import torch  # noqa: F401
    from tensorrt_llm import SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401  registers auto-model
    from tensorrt_llm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    import inkling_generation_parity_test as gp

    assert torch.cuda.is_available(), "Stage A needs CUDA GPUs"
    os.makedirs(OUTDIR, exist_ok=True)
    _doc, prompts = _load_ref()
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    # candidates: SGLang correct + enough committed tokens to localize.
    cands = [r for r in prompts if r.get("sg_correct")
             and len(r.get("greedy_token_ids", [])) >= MINSTEP
             and len(r.get("pos_top", [])) >= MINSTEP]
    max_in = max((r["n_input"] for r in prompts), default=1024)
    print(f"[localize.worst] tp={TP} moe={MOE_BACKEND} maxstep={MAXSTEP} "
          f"maxseq={MAXSEQ} max_in={max_in} n_candidates={len(cands)} "
          f"(of {len(prompts)}) ref={DISC_REF}", flush=True)
    llm = _build_llm()
    gp.TOPK, gp.TIE_MARGIN = 20, 0.75

    results = []
    try:
        for r in cands:
            idx = r["idx"]
            input_ids = list(r["input_ids"])
            sg_ids = list(r["greedy_token_ids"])
            n_step = min(len(sg_ids), len(r["pos_top"]), MAXSTEP)
            # (1) free-run TRT to establish TRT-wrong (discriminating half).
            fr = llm.generate([TokensPrompt(prompt_token_ids=input_ids)],
                              SamplingParams(max_tokens=FREERUN_CAP,
                                             temperature=0.0))[0].outputs[0]
            fr_ids = list(fr.token_ids)
            fr_text = tok.decode([int(i) for i in fr_ids])
            trt_ans = _extract_answer(fr_text)
            stopped = any(int(t) in STOP_TOKENS for t in fr_ids) \
                or (getattr(fr, "finish_reason", None) == "stop")
            trt_wrong = (not stopped) or (trt_ans != r.get("gold"))
            # (2) teacher-force to find the worst decode step.
            gp.NSTEP = n_step
            per_step, n_calls = gp.teacher_force(
                llm, SamplingParams, TokensPrompt, input_ids, sg_ids, r["pos_top"])
            steps = [{"t": s["t"], "sg": int(s["sg"]), "trt": int(s["trt"]),
                      "match": bool(s["match"]),
                      "cos": (None if math.isnan(s["cos"]) else round(s["cos"], 6)),
                      "max_abs": (None if math.isnan(s["max_abs"])
                                  else round(s["max_abs"], 6))}
                     for s in per_step]
            w = worst_step_of(steps)
            ranked = sorted([s for s in steps if s["cos"] is not None],
                            key=lambda s: s["cos"])
            n_mis = sum(1 for s in steps if not s["match"])
            rec = {
                "idx": idx, "prompt": r.get("prompt", f"gsm8k#{idx}"),
                "gold": r.get("gold"), "sg_answer": r.get("sg_answer"),
                "sg_correct": bool(r.get("sg_correct")),
                "trt_answer": trt_ans, "trt_stopped": bool(stopped),
                "trt_finish_reason": getattr(fr, "finish_reason", None),
                "trt_n_freerun": len(fr_ids), "trt_wrong": bool(trt_wrong),
                "discriminating": bool(r.get("sg_correct")) and bool(trt_wrong),
                "n_input": len(input_ids), "n_step": n_step,
                "tf_mismatches": n_mis, "tf_calls": n_calls,
                "worst_t": (w["t"] if w else None),
                "worst_cos": (w["cos"] if w else None),
                "worst_max_abs": (w["max_abs"] if w else None),
                "prefix_len": (len(input_ids) + w["t"]) if w else None,
                "ranked_steps": ranked[:8],
            }
            results.append(rec)
            print(f"[localize.worst] idx={idx:>4} gold={r.get('gold')} "
                  f"sg_ok={rec['sg_correct']} trt_ans={trt_ans} "
                  f"trt_wrong={rec['trt_wrong']} disc={rec['discriminating']} "
                  f"n_in={rec['n_input']} n_step={n_step} tf_mis={n_mis} "
                  f"worst_t={rec['worst_t']} cos={rec['worst_cos']} "
                  f"max_abs={rec['worst_max_abs']}", flush=True)
    finally:
        llm.shutdown()

    disc = [r for r in results if r["discriminating"] and r["worst_t"] is not None]
    disc = enforce_unique_prefix_lens(disc)
    out = {"schema": "localize_worst_v1", "provenance": _provenance(),
           "n_candidates": len(results), "n_discriminating": len(disc),
           "need_discriminating": NEED_DISC,
           "discriminating": disc, "all": results}
    outp = os.path.join(OUTDIR, "worst_steps.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)
    ok = len(disc) >= NEED_DISC
    print(f"\nINKLING_LOCALIZE_WORST {'OK' if ok else 'FAIL'} "
          f"n_discriminating={len(disc)} need={NEED_DISC} "
          f"idxs={[r['idx'] for r in disc]} "
          f"prefix_lens={[r['prefix_len'] for r in disc]} -> {outp}", flush=True)
    print(f"=== INKLING_LOCALIZE_WORST_DONE rc={0 if ok else 3} ===", flush=True)
    return 0 if ok else 3


def run_dump():
    """Stage B (TRT half): dump per-layer h_attn/moe_out at each worst-token prefix.

    INKLING_DUMP_PREFILL / _ALLLAYERS / _MODULES / _MINTOK / _MAXTOK are set in the
    sbatch env (they must reach every TP worker); this driver only sends each
    discriminating sample's SGLang-teacher-forced prefix as a single prefill."""
    import torch  # noqa: F401
    from tensorrt_llm import SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401  registers auto-model
    from tensorrt_llm.inputs import TokensPrompt

    assert torch.cuda.is_available(), "Stage B dump needs CUDA GPUs"
    dump_base = os.environ.get("INKLING_DUMP_PREFILL")
    assert dump_base, "INKLING_DUMP_PREFILL must be set in the sbatch env"
    assert os.environ.get("INKLING_DUMP_ALLLAYERS") == "1", \
        "INKLING_DUMP_ALLLAYERS=1 required"
    assert os.environ.get("INKLING_DUMP_MODULES") == "1", \
        "INKLING_DUMP_MODULES=1 required"
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "worst_steps.json")) as f:
        worst = json.load(f)
    disc = worst["discriminating"]
    _doc, prompts = _load_ref()
    by_idx = {r["idx"]: r for r in prompts}
    max_plen = max((r["prefix_len"] for r in disc), default=1024)
    assert max_plen < MAXSEQ, f"max prefix_len {max_plen} >= MAXSEQ {MAXSEQ}"
    llm = _build_llm()
    print(f"[localize.dump] n_disc={len(disc)} dump_base={dump_base} "
          f"window=[{os.environ.get('INKLING_DUMP_MINTOK')},"
          f"{os.environ.get('INKLING_DUMP_MAXTOK')}]", flush=True)

    summary = []
    try:
        for rec in disc:
            idx = rec["idx"]
            r = by_idx[idx]
            worst_t = rec["worst_t"]
            prefix_ids = list(r["input_ids"]) + list(r["greedy_token_ids"][:worst_t])
            plen = len(prefix_ids)
            assert plen == rec["prefix_len"], \
                f"idx {idx} prefix_len drift {plen} != {rec['prefix_len']}"
            llm.generate([TokensPrompt(prompt_token_ids=prefix_ids)],
                         SamplingParams(max_tokens=1, temperature=0.0))
            dumped = f"{dump_base}.n{plen}.rank0"
            ok = os.path.exists(dumped)
            summary.append({"idx": idx, "worst_t": worst_t, "prefix_len": plen,
                            "dump_file": dumped if ok else None, "dumped": ok})
            print(f"[localize.dump] idx={idx:>4} worst_t={worst_t} plen={plen} "
                  f"dump={'OK' if ok else 'MISSING'} {dumped}", flush=True)
    finally:
        llm.shutdown()

    outp = os.path.join(OUTDIR, "trt_module_dump_summary.json")
    with open(outp, "w") as f:
        json.dump({"schema": "localize_dump_v1", "provenance": _provenance(),
                   "per": summary}, f, indent=1)
    n_ok = sum(1 for s in summary if s["dumped"])
    print(f"\nINKLING_LOCALIZE_DUMP n_disc={len(summary)} n_dumped={n_ok} "
          f"-> {outp}", flush=True)
    print(f"=== INKLING_LOCALIZE_DUMP_DONE rc={0 if n_ok >= NEED_DISC else 3} ===",
          flush=True)
    return 0 if n_ok >= NEED_DISC else 3


def main():
    if MODE == "selftest":
        return _selftest()
    if MODE == "worst":
        return run_worst()
    if MODE == "dump":
        return run_dump()
    print(f"unknown INKLING_LOCALIZE_MODE={MODE!r} (worst|dump|selftest)")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("=== INKLING_LOCALIZE_DONE rc=1 ===", flush=True)
        sys.exit(1)
