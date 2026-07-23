#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit7 generation_parity: per-step greedy-token equality vs the SGLang reference.

Redesigned iter68 after the Reviewer showed free-running greedy comparison is the
wrong tool: TensorRT-LLM (trtllm-gen fp4 MoE + TRTLLM attention) and SGLang
(flashinfer fp4 + fa4) run DIFFERENT kernels on the same NVFP4 weights, so two
free-running greedy decodes fork at the first near-tie and every downstream
position is then conditioned on a different prefix -- non-comparable. A single
benign near-tie flip cascades into a wall of spurious "mismatches". This harness
uses the methodology-mandated approach instead:

TWO complementary evaluations, ONE model construction per config:

1. TEACHER-FORCED per-step greedy equality (the STRICT crit7 gate).
   Drive the DECODE path with SGLang's own greedy tokens: greedy-decode, and at
   the FIRST step where TensorRT-LLM's greedy token disagrees with SGLang, record
   it, then FORCE SGLang's token and restart decoding from the corrected prefix.
   Because the prefix is held identical to SGLang the whole sequence, each of the
   >=32 steps is a true per-step comparison (no cascade), and the segment after
   every restart is real autoregressive decode (KVCacheManagerV2 + short-conv
   state carry + attention decode) -- exactly the path GSM8K uses. crit7 requires
   per-step greedy-token EQUALITY, so EVERY step whose TensorRT-LLM greedy token
   != SGLang's token fails the gate, regardless of SGLang's top1-top2 margin. The
   margin is still recorded per mismatch (a mismatch at a tiny margin is labeled a
   near-tie) so a failure can be localized -- a benign NVFP4-vs-NVFP4 tie flip vs a
   confident-margin model defect -- but that label is DIAGNOSTIC ONLY and never
   exempts a mismatch from failing.

2. FREE-RUNNING BATCHED collapse detector (anti-gaming guard).
   Teacher forcing alone could pass while the real nc>1 batched decode-state bug
   ('!!!!' / repeated-token / empty garbage) is still live, because teacher
   forcing keeps re-anchoring to the correct prefix. So we ALSO free-run all
   prompts BATCHED and flag any prompt whose output degenerates (a long run of one
   repeated token, or collapses to very few unique tokens) when the SGLang
   reference for that same prompt does NOT. This is the signal that correlates
   with the served nc=4 GSM8K crater.

crit7 GATE = zero teacher-forced per-step greedy MISMATCHES (near-tie or not) AND
zero batched-collapse prompts. The gate cannot be satisfied by fixing only prefill
numerics: the batched free-running decode must also be garbage-free.

Config matrix (env-selected, one script covers both acceptance rows):
  * INKLING_CUDA_GRAPH=0/1  -> cuda_graph_config None / CudaGraphConfig()
  * INKLING_OVERLAP=0/1     -> disable_overlap_scheduler True / False

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_generation_parity_test.py
Env: INKLING_CHECKPOINT, INKLING_SGLANG_REF (the crit6 capture json).
"""

import json
import math
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")
REF = os.environ.get(
    "INKLING_SGLANG_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-bringup/results/sglang_ref_logit_replay.json")

CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "1" if CUDA_GRAPH else "0") == "1"
# Tensor-parallel size. Default 4 (the deployment config). INKLING_TP=1 runs the
# whole 66-layer model on ONE GPU with NO TP collectives -- the B2-localization
# lever: iter75's TP=1 hand-rolled localizer showed the model decode is graph-clean,
# so if TP=1 production cuda_graph is ALSO clean (tf_mismatch ~= baseline) while TP=4
# is corrupt (157), B2 is a TP-collective-under-cuda-graph bug (all-reduce /
# inkling_ar_scattered_sconv); if TP=1 cuda_graph is ALSO corrupt, B2 is generic to
# the production CUDAGraphRunner path.
TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_GP_STEPS", "32"))     # >=32 required
TOPK = int(os.environ.get("INKLING_GP_TOPK", "20"))
# SGLang top1-top2 margin (nats) below which a teacher-forced greedy disagreement
# is an expected NVFP4-vs-NVFP4 near tie (benign), not a decode defect. crit6 saw
# confident-step margins 0.75-4.25; observed near-tie flips sit at 0.125-0.25.
TIE_MARGIN = float(os.environ.get("INKLING_GP_TIE_MARGIN", "0.75"))
# Free-running batched collapse thresholds (garbage detector, anchored to SGLang).
REPEAT_THRESH = int(os.environ.get("INKLING_GP_REPEAT_THRESH", "8"))
MIN_UNIQUE = int(os.environ.get("INKLING_GP_MIN_UNIQUE", "3"))

# ---- DETERMINISTIC measurement config (human feedback #3 / #4) ----------------
# INKLING_DETERMINISTIC=1 makes batch=1 + autotuner-off the PRIMARY deterministic
# baseline so run-to-run noise stops contaminating every parity number. Two named
# non-determinism sources are removed:
#   (1) the CROSS-ROW atomic reduction in the BATCHED fused-MoE combine (float add
#       is non-associative; atomicAdd order is unfixed at batch>1) -- gone at
#       max_batch_size=1 (a single row has no cross-row reduction);
#   (2) the AUTOTUNER, which times candidate kernel tactics at warmup and can pick a
#       different tactic across runs on timing noise -> different accumulation order
#       -- disabled by enable_autotuner=False (every tunable op takes its fixed
#       fallback tactic, no warmup timing at all).
# The driving sbatch additionally exports TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1 so the
# TP=4 all-reduce also uses a fixed strategy instead of an AUTO-tuned one.
# Default 0 preserves the official crit7 gate config (max_batch_size=8, autotuner
# on) byte-for-byte -- this is a measurement-hygiene switch, NOT a gate relaxation.
DETERMINISTIC = os.environ.get("INKLING_DETERMINISTIC", "0") == "1"
MAX_BS = 1 if DETERMINISTIC else int(os.environ.get("INKLING_GP_MAX_BS", "8"))
ENABLE_AUTOTUNER = not DETERMINISTIC
# Optional fingerprint artifact for the two-run byte-identical determinism proof.
FINGERPRINT_OUT = os.environ.get("INKLING_GP_FINGERPRINT_OUT", "")
# Documented rounding for the stable per-step logit checksum (acceptance crit12:
# "byte-identical ... final logits or stable logit checksums"). TRT's top-K
# log-probs are rounded to LOGIT_ROUND decimals before hashing so bit-noise in
# float repr cannot create a false determinism MISMATCH, while any real change in
# TRT's output distribution (top-K membership or a value beyond ~1e-5) flips the
# checksum. Default 5 decimals; override via INKLING_GP_LOGIT_ROUND.
LOGIT_ROUND = int(os.environ.get("INKLING_GP_LOGIT_ROUND", "5"))


def _repo_provenance():
    """Repo-SHA + dirty-source identity for the deterministic run (acceptance
    crit12: each artifact must carry "job id/config/repo SHA provenance").

    Prefer the values the driving sbatch captured from the mounted TensorRT-LLM
    checkout and exported (INKLING_DET_REPO_SHA / _DIRTY / _DIFF_SHA) -- that is
    the authoritative same-path mount the run actually executed on. Fall back to
    computing them here from this test file's own git repo (via subprocess) so the
    provenance field is never empty even when the test is run directly without the
    sbatch wrapper. Returns (repo_sha, dirty_file_count, diff_sha) all as strings;
    diff_sha is a short sha256 over `git diff HEAD` so two runs of the SAME working
    tree share one dirty-source identity while any uncommitted edit changes it.
    """
    sha = os.environ.get("INKLING_DET_REPO_SHA", "").strip()
    dirty = os.environ.get("INKLING_DET_REPO_DIRTY", "").strip()
    diff_sha = os.environ.get("INKLING_DET_REPO_DIFF_SHA", "").strip()
    if sha and dirty and diff_sha:
        return sha, dirty, diff_sha
    import subprocess
    import hashlib
    repo = os.path.dirname(os.path.abspath(__file__))

    def _git(args):
        try:
            r = subprocess.run(["git", "-C", repo, *args],
                               capture_output=True, text=True, timeout=30)
            return r.stdout
        except Exception:  # noqa: BLE001
            return ""

    if not sha:
        sha = _git(["rev-parse", "HEAD"]).strip() or "unknown"
    if not dirty:
        porcelain = _git(["status", "--porcelain"])
        dirty = str(len([ln for ln in porcelain.splitlines() if ln.strip()]))
    if not diff_sha:
        diff = _git(["diff", "HEAD"])
        diff_sha = hashlib.sha256(diff.encode()).hexdigest()[:16]
    return sha, dirty, diff_sha


def _canon_blob(fr_records, tf_records, counts):
    """Canonical byte blob over the determinism-relevant discrete state AND the
    stable per-step logit checksum.

    Acceptance crit12 requires two deterministic runs to be byte-identical for
    greedy token ids, the tf_mismatch / near-tie / confident counts, AND
    "final logits or stable logit checksums". The blob hashes exactly those:
    every free-run greedy token id sequence, every teacher-forced per-step
    (t, trt, sg, match, lp_ck) tuple -- where lp_ck is the stable logit checksum
    of TRT's top-K log-probs at that step (see _lp_checksum) -- and the summary
    counts. The raw cos/max_abs floats stay OUT of the hash (they are the
    TRT-vs-SGLang gap, reported separately via tf_min_cos); the determinism of
    TRT's own logits is carried by lp_ck, which is rounded to LOGIT_ROUND decimals
    so float-repr noise cannot create a false MISMATCH.
    """
    canon = {
        "freerun": [[rec["prompt"], list(rec["token_ids"])]
                    for rec in fr_records],
        "teacher": [[rec["prompt"],
                     [[s["t"], s["trt"], s["sg"], int(s["match"]),
                       s.get("lp_ck", "")]
                      for s in rec["steps"]]]
                    for rec in tf_records],
        "counts": list(counts),
    }
    return json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()


def _fingerprint_sha(fr_records, tf_records, counts):
    """sha256 of the canonical determinism blob (see _canon_blob)."""
    import hashlib
    return hashlib.sha256(_canon_blob(fr_records, tf_records, counts)).hexdigest()


def _logit_checksum(tf_records):
    """Aggregate stable logit checksum over every teacher-forced step (acceptance
    crit12 "stable logit checksums"). Deterministic function of the per-step lp_ck
    values (TRT top-K log-probs rounded to LOGIT_ROUND decimals, see _lp_checksum),
    keyed by (prompt, step) so ordering is fixed. A separate, directly-comparable
    field from sha256 so cmd_det can assert the logit values -- not just the greedy
    decisions -- are byte-identical run-to-run. Steps with no TRT log-probs
    contribute an empty lp_ck, so a run where TRT emits nothing is still captured.
    """
    import hashlib
    parts = []
    for rec in tf_records:
        for s in rec["steps"]:
            parts.append(f"{rec['prompt']}|{s['t']}|{s.get('lp_ck', '')}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _worst_step(tf_records):
    """Feedback #4 Stage A: the teacher-forced step with MAX TRT-vs-SGLang logit
    divergence (lowest cosine over the shared vocab support). Returns the
    (prompt, t, cos, max_abs, sg, trt) of that step, or None if no step carried a
    finite cosine. This is the (sample_id, token_position) Stage A must output.
    """
    worst = None
    for rec in tf_records:
        for s in rec["steps"]:
            c = s.get("cos")
            if c is None:
                continue
            if worst is None or c < worst["cos"]:
                worst = {"prompt": rec["prompt"], "t": s["t"], "cos": c,
                         "max_abs": s.get("max_abs"), "sg": s["sg"],
                         "trt": s["trt"]}
    return worst


def _sg_margin(sg_top):
    """SGLang top1-top2 log-prob margin (nats); large if only one entry."""
    if len(sg_top) >= 2:
        return float(sg_top[0][1] - sg_top[1][1])
    return float("inf")


def _lp_stats(trt_lp_dict, sg_top):
    """max_abs + cosine of the top-K log-probs over the shared token support."""
    import torch
    sg = {int(tid): float(lp) for tid, lp in sg_top}
    ids = [tid for tid in sg if tid in trt_lp_dict]
    if len(ids) < 2:
        return float("nan"), float("nan"), len(ids)
    a = torch.tensor([trt_lp_dict[i] for i in ids])
    b = torch.tensor([sg[i] for i in ids])
    mx = float((a - b).abs().max())
    cos = float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
    return mx, cos, len(ids)


def _max_consec_repeat(ids):
    """Longest run of an identical consecutive token id."""
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def _lp_dict(lp_entry):
    """Normalize an LLM-API per-step logprob entry to {token_id: logprob}."""
    if not isinstance(lp_entry, dict):
        return {}
    return {int(k): float(getattr(v, "logprob", v)) for k, v in lp_entry.items()}


def _lp_checksum(trt_lp_dict):
    """Stable per-step logit checksum: sha over TRT's top-K (token_id, rounded
    log-prob) pairs (acceptance crit12 "stable logit checksums ... with documented
    rounding"). Values are rounded to LOGIT_ROUND decimals (default 5) and -0.0 is
    normalized to 0.0 so identical float64 outputs hash identically and sub-1e-5
    repr noise cannot create a false determinism MISMATCH, while any real change in
    TRT's output distribution (top-K membership or a value) flips the checksum.
    Returns "" when the step carried no TRT log-probs (e.g. TRT emitted nothing),
    which is itself a determinism-relevant, comparable state.
    """
    if not trt_lp_dict:
        return ""
    import hashlib

    def _q(x):
        v = round(float(x), LOGIT_ROUND)
        return 0.0 if v == 0.0 else v  # normalize -0.0 -> 0.0

    items = sorted((int(tid), _q(lp)) for tid, lp in trt_lp_dict.items())
    blob = json.dumps(items, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def teacher_force(llm, SamplingParams, TokensPrompt, input_ids, sg_ids, sg_top):
    """Restart-on-fork teacher-forced greedy decode against the SGLang tokens.

    Returns (per_step, n_calls). per_step[i] = dict(t, trt, sg, match, margin,
    neartie, cos, max_abs). Decodes real autoregressive steps; on each greedy
    disagreement it records the step, forces SGLang's token, and re-decodes from
    the corrected prefix -- so no fork cascades and every step is comparable.
    """
    forced = list(input_ids)
    t = 0
    per_step = []
    n_calls = 0
    guard = NSTEP + 4
    while t < NSTEP and n_calls < guard:
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=forced)],
            SamplingParams(max_tokens=NSTEP - t, temperature=0.0,
                           logprobs=TOPK))[0]
        n_calls += 1
        gen = out.outputs[0]
        trt_ids = list(gen.token_ids)
        trt_lps = gen.logprobs or []
        if not trt_ids:  # TRT emitted nothing where SGLang continues -> defect
            margin = _sg_margin(sg_top[t])
            per_step.append(dict(t=t, trt=-1, sg=int(sg_ids[t]), match=False,
                                 margin=margin, neartie=(margin < TIE_MARGIN),
                                 cos=float("nan"), max_abs=float("nan"),
                                 lp_ck=""))
            forced = list(input_ids) + list(sg_ids[:t + 1])
            t += 1
            continue
        forked = False
        consumed = 0
        for i, tt in enumerate(trt_ids):
            tt_t = t + i
            if tt_t >= NSTEP:
                break
            sg = int(sg_ids[tt_t])
            margin = _sg_margin(sg_top[tt_t])
            match = (int(tt) == sg)
            trt_lp = _lp_dict(trt_lps[i]) if i < len(trt_lps) else {}
            mx, cos, _ = _lp_stats(trt_lp, sg_top[tt_t])
            per_step.append(dict(t=tt_t, trt=int(tt), sg=sg, match=match,
                                 margin=margin, neartie=(margin < TIE_MARGIN),
                                 cos=cos, max_abs=mx,
                                 lp_ck=_lp_checksum(trt_lp)))
            consumed += 1
            if not match:
                forced = list(input_ids) + list(sg_ids[:tt_t + 1])
                t = tt_t + 1
                forked = True
                break
        if not forked:
            next_t = t + consumed
            if next_t >= NSTEP:
                t = NSTEP
            else:  # TRT stopped early (EOS) before NSTEP while SGLang continues
                margin = _sg_margin(sg_top[next_t])
                per_step.append(dict(t=next_t, trt=-1, sg=int(sg_ids[next_t]),
                                     match=False, margin=margin,
                                     neartie=(margin < TIE_MARGIN),
                                     cos=float("nan"), max_abs=float("nan"),
                                     lp_ck=""))
                forced = list(input_ids) + list(sg_ids[:next_t + 1])
                t = next_t + 1
    return per_step, n_calls


def main() -> int:
    import torch  # noqa: F401
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401  (registers auto-model)
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "crit7 generation_parity needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids") and r.get("pos_top")
           and len(r.get("greedy_token_ids", [])) >= NSTEP
           and len(r.get("pos_top", [])) >= NSTEP]
    assert len(ref) >= 5, f"need >=5 prompts with >={NSTEP} ref tokens, got {len(ref)}"
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    print(f"[gp] tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} n_prompts={len(ref)} "
          f"steps={NSTEP} topk={TOPK} tie_margin={TIE_MARGIN} "
          f"deterministic={DETERMINISTIC} max_batch_size={MAX_BS} "
          f"enable_autotuner={ENABLE_AUTOTUNER} "
          f"allreduce_autotune_disabled={os.environ.get('TLLM_DISABLE_ALLREDUCE_AUTOTUNE', '0')} "
          f"ref={REF}", flush=True)

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                    dtype="auto", enable_block_reuse=False)
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        enable_autotuner=ENABLE_AUTOTUNER,
        max_seq_len=2048,
        max_batch_size=MAX_BS,
        max_num_tokens=2048,
    )
    hard_path = "CudaGraphConfig()" if CUDA_GRAPH else "eager(no-graph)"
    print(f"[gp] moe_backend={moe_backend} cuda_graph_hard_path={hard_path} "
          f"deterministic={DETERMINISTIC} max_batch_size={MAX_BS} "
          f"enable_autotuner={ENABLE_AUTOTUNER}", flush=True)

    def decode(ids):
        try:
            return tok.decode([int(i) for i in ids if int(i) >= 0])
        except Exception:  # noqa: BLE001
            return "<decode-err>"

    try:
        # ---- PHASE 1: FREE-RUNNING BATCHED (collapse / garbage detector) --------
        prompts = [TokensPrompt(prompt_token_ids=list(r["input_ids"])) for r in ref]
        fr_out = llm.generate(
            prompts, SamplingParams(max_tokens=NSTEP, temperature=0.0))
        collapse = []
        fr_matchlens = []
        fr_records = []  # per-prompt free-run greedy token ids (determinism proof)
        for r, out in zip(ref, fr_out):
            trt_ids = list(out.outputs[0].token_ids)
            sg_ids = r["greedy_token_ids"]
            fr_records.append({"prompt": r["prompt"],
                               "token_ids": [int(x) for x in trt_ids]})
            # leading per-step match length (diagnostic only)
            ml = 0
            for a, b in zip(trt_ids, sg_ids[:NSTEP]):
                if int(a) == int(b):
                    ml += 1
                else:
                    break
            fr_matchlens.append(ml)
            trt_rep = _max_consec_repeat([int(x) for x in trt_ids])
            sg_rep = _max_consec_repeat([int(x) for x in sg_ids[:NSTEP]])
            trt_uni = len(set(int(x) for x in trt_ids))
            sg_uni = len(set(int(x) for x in sg_ids[:NSTEP]))
            is_collapse = ((trt_rep >= REPEAT_THRESH and sg_rep < REPEAT_THRESH)
                           or (trt_uni < MIN_UNIQUE and sg_uni >= MIN_UNIQUE))
            if is_collapse:
                collapse.append((r["prompt"], trt_rep, trt_uni,
                                 decode(trt_ids)[:60]))
            print(f"  [freerun] match_len={ml}/{NSTEP} trt_maxrep={trt_rep} "
                  f"trt_uniq={trt_uni} (sg_maxrep={sg_rep} sg_uniq={sg_uni}) "
                  f"{'COLLAPSE' if is_collapse else 'ok'} {r['prompt']!r} -> "
                  f"{decode(trt_ids)[:50]!r}", flush=True)

        # ---- PHASE 2: TEACHER-FORCED per-step greedy equality (STRICT gate) -----
        # crit7 requires per-step greedy-token equality: EVERY teacher-forced step
        # whose TensorRT-LLM greedy token != SGLang's token fails the gate,
        # regardless of SGLang's top1-top2 margin. The near-tie label is recorded
        # per mismatch (diagnostic only) so a failure can be localized to a benign
        # NVFP4-vs-NVFP4 tie flip vs a confident-margin defect -- it never exempts a
        # mismatch from failing.
        tf_bad = []          # ALL per-step greedy-token mismatches (gate-failing)
        tf_neartie = 0       # subset of tf_bad at a tiny SGLang margin (diagnostic)
        tf_total = 0
        tf_min_cos = float("inf")
        tf_records = []      # per-prompt per-step records (determinism proof + Stage A)
        for r in ref:
            per_step, n_calls = teacher_force(
                llm, SamplingParams, TokensPrompt,
                r["input_ids"], r["greedy_token_ids"], r["pos_top"])
            tf_records.append({"prompt": r["prompt"], "steps": [
                {"t": s["t"], "trt": int(s["trt"]), "sg": int(s["sg"]),
                 "match": bool(s["match"]), "neartie": bool(s["neartie"]),
                 "cos": (None if math.isnan(s["cos"]) else round(s["cos"], 6)),
                 "max_abs": (None if math.isnan(s["max_abs"])
                             else round(s["max_abs"], 6)),
                 "lp_ck": s.get("lp_ck", "")}
                for s in per_step]})
            mism = [s for s in per_step if not s["match"]]
            near = [s for s in mism if s["neartie"]]
            tf_neartie += len(near)
            tf_total += len(per_step)
            for s in per_step:
                if not math.isnan(s["cos"]):
                    tf_min_cos = min(tf_min_cos, s["cos"])
            for s in mism:
                tf_bad.append((r["prompt"], s["t"], s["sg"], s["trt"], s["margin"],
                               s["neartie"], decode([s["sg"]]), decode([s["trt"]])))
            near_txt = ",".join(
                f"@{s['t']}({decode([s['sg']])!r}->{decode([s['trt']])!r})"
                for s in near) or "none"
            print(f"  [teacher] mismatches={len(mism)} (neartie={len(near)}) "
                  f"calls={n_calls} steps={len(per_step)} neartie=[{near_txt}] "
                  f"{r['prompt']!r}", flush=True)
            for p, t, sg, trt, m, nt, sgtx, trttx in [
                    x for x in tf_bad if x[0] == r["prompt"]]:
                print(f"    [MISMATCH] step={t} SGLang={sg}({sgtx!r} margin={m:.3f} "
                      f"neartie={nt}) TRT={trt}({trttx!r})", flush=True)
    finally:
        llm.shutdown()

    if tf_min_cos is math.inf:
        tf_min_cos = float("nan")
    n_collapse = len(collapse)
    n_bad = len(tf_bad)          # ALL teacher-forced per-step greedy mismatches
    fr_min_ml = min(fr_matchlens) if fr_matchlens else 0
    for p, rep, uni, txt in collapse:
        print(f"[gp] COLLAPSE prompt {p!r}: max_repeat={rep} unique={uni} "
              f"trt_out={txt!r}", flush=True)
    print(f"\n[gp] TEACHER-FORCED per-step equality: mismatch_steps={n_bad} "
          f"(of which neartie={tf_neartie}) total_steps={tf_total} "
          f"min_cos={tf_min_cos:.5f} | FREE-RUN batched: collapse={n_collapse}/"
          f"{len(ref)} min_match_len={fr_min_ml}/{NSTEP} | cuda_graph={CUDA_GRAPH} "
          f"overlap={OVERLAP} cuda_graph_hard_path={hard_path}", flush=True)
    print(f"TF_BADSTEP count={n_bad} COLLAPSE count={n_collapse}", flush=True)
    # crit7 GATE (STRICT per-step greedy-token equality): pass only when EVERY
    # teacher-forced step reproduces SGLang's greedy token (n_bad==0, no exemption
    # for near-tie margins) AND no free-running batched prompt degenerates into
    # garbage the SGLang reference does not show (collapse==0). tf_neartie and the
    # free-run match lengths are diagnostics that localize a failure; they never
    # turn a mismatch into a pass.
    ok = (n_bad == 0) and (n_collapse == 0)
    print(f"INKLING_GP_{'OK' if ok else 'FAIL'} tp={TP} tf_mismatch_steps={n_bad} "
          f"tf_neartie_flips={tf_neartie} tf_total_steps={tf_total} "
          f"freerun_collapse={n_collapse}/{len(ref)} "
          f"freerun_min_matchlen={fr_min_ml}/{NSTEP} "
          f"min_cos={tf_min_cos:.5f} cuda_graph={CUDA_GRAPH} "
          f"overlap={OVERLAP} cuda_graph_hard_path={hard_path}", flush=True)

    # ---- DETERMINISM FINGERPRINT + Stage-A worst step (feedback #3 / #4) ------
    # sha256 over greedy token ids + teacher per-step decisions + counts. Two
    # deterministic runs of the SAME code must print the SAME sha (byte-identical).
    tf_confident = n_bad - tf_neartie
    counts = [n_bad, tf_neartie, tf_confident, n_collapse, fr_min_ml]
    sha = _fingerprint_sha(fr_records, tf_records, counts)
    logit_ck = _logit_checksum(tf_records)
    worst = _worst_step(tf_records)
    # Provenance (acceptance crit12): job id + repo SHA + dirty-source identity.
    # job_id prefers the sbatch-exported INKLING_DET_JOB_ID because SLURM_JOB_ID is
    # not reliably propagated into the pyxis --container-name login shell (it landed
    # empty in the iter-25/26 artifacts); SLURM_JOB_ID stays as a fallback.
    prov_sha, prov_dirty, prov_diff_sha = _repo_provenance()
    fp = {
        "config": {"tp": TP, "cuda_graph": CUDA_GRAPH, "overlap": OVERLAP,
                   "moe_backend": moe_backend, "max_batch_size": MAX_BS,
                   "enable_autotuner": ENABLE_AUTOTUNER,
                   "deterministic": DETERMINISTIC,
                   "allreduce_autotune_disabled":
                   os.environ.get("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0"),
                   "cublas_workspace_config":
                   os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
                   "logit_round": LOGIT_ROUND,
                   "job_id": (os.environ.get("INKLING_DET_JOB_ID")
                              or os.environ.get("SLURM_JOB_ID", "")),
                   "repo_sha": prov_sha,
                   "repo_dirty_files": prov_dirty,
                   "repo_diff_sha": prov_diff_sha,
                   "tag": os.environ.get("INKLING_DET_TAG", ""),
                   "ref": REF, "steps": NSTEP, "tie_margin": TIE_MARGIN,
                   "n_prompts": len(ref)},
        "summary": {"tf_mismatch_steps": n_bad, "tf_neartie_flips": tf_neartie,
                    "tf_confident": tf_confident, "freerun_collapse": n_collapse,
                    "freerun_min_matchlen": fr_min_ml,
                    "tf_min_cos": (None if math.isnan(tf_min_cos)
                                   else round(tf_min_cos, 6))},
        "worst_step": worst,
        "freerun": fr_records,
        "teacher": tf_records,
        "sha256": sha,
        "logit_checksum": logit_ck,
    }
    if FINGERPRINT_OUT:
        try:
            with open(FINGERPRINT_OUT, "w") as f:
                json.dump(fp, f, sort_keys=True)
            print(f"[gp] fingerprint written: {FINGERPRINT_OUT}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[gp] WARN fingerprint write failed: {e}", flush=True)
    print(f"INKLING_GP_FINGERPRINT sha256={sha} logit_checksum={logit_ck} "
          f"logit_round={LOGIT_ROUND} deterministic={DETERMINISTIC} "
          f"max_batch_size={MAX_BS} enable_autotuner={ENABLE_AUTOTUNER} "
          f"job_id={fp['config']['job_id']} repo_sha={prov_sha} "
          f"repo_dirty_files={prov_dirty} repo_diff_sha={prov_diff_sha} "
          f"tf_mismatch_steps={n_bad} tf_neartie={tf_neartie} "
          f"tf_confident={tf_confident} freerun_collapse={n_collapse}/{len(ref)}",
          flush=True)
    if worst is not None:
        wtxt_sg = decode([worst["sg"]])
        wtxt_trt = decode([worst["trt"]])
        print(f"INKLING_GP_WORSTSTEP prompt={worst['prompt']!r} t={worst['t']} "
              f"cos={worst['cos']} max_abs={worst['max_abs']} "
              f"sg={worst['sg']}({wtxt_sg!r}) trt={worst['trt']}({wtxt_trt!r})",
              flush=True)
    else:
        print("INKLING_GP_WORSTSTEP none (no finite-cosine step)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
