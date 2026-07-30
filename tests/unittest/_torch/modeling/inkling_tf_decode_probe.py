#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HUMAN FEEDBACK #10 -- Stage 2: TEACHER-FORCED DECODE SWEEP vs SGLang.

``reference_tier=real_source``, ``validation_tier=real_runtime``.

Stage 1 (final-prefill logits) and the prefill DIVE both ACQUITTED the prefill
path: the first generated token (200008) is identical on all 3 selected samples,
the completeness witness is clean, and the per-layer control-relative gaps stay
inside the noise floor.  So divergence, if any, starts in DECODE.  This driver is
feedback #10 Stage 2: it feeds SGLang's OWN token sequence into TRT (teacher
forcing, so both stacks walk the identical prefix at every position -- free-run
forking can never be offered as an excuse) and scores EVERY swept position.

Samples (feedback #10 Stage 0, DO NOT re-derive):
  * PRIMARY (TRT wrong / SGLang right): validation_Geography_6 (gold C, TRT D),
    validation_Math_17 (gold A, TRT B).
  * CONTROL (both arms correct): validation_Marketing_7 (gold D).
  NOTE both primaries AND the control run to n_gen=2560 finish=length on BOTH
  stacks (the parsed answer flips inside a long reasoning stream).  The human
  ranked these by EARLIEST decoded-text divergence (prefix 8), so the answer-
  determining fork is early; we sweep a bounded window (INKLING_TF_STEPS, default
  512) that covers it and report the window explicitly.

Per-step metrics (feedback #10 Stage 2.2), teacher-forced on SGLang's tokens:
  * ``top1_agree``  -- TRT's full-vocab argmax == SGLang's greedy token at this
    step (SGLang decoded greedily, so its token IS its argmax).
  * ``cos_raw``     -- cosine of TRT's logits vs SGLang's top-K logprobs over the
    shared support (max-shifted RAW units, identical to the accepted Stage-1
    ``compare_pos`` contract; SGLang serve returns top-K, not the full 200k).
  * ``gap``         -- TRT's OWN logit gap between its top-1 and SGLang's top-1,
    ``logit[trt_top1] - logit[sg_tok]``.  Because true-logprob = logit - lse and
    the lse cancels in the difference, this gap is already in log-prob units
    (same scale SGLang reports), so it is directly comparable to the accepted
    iter51 / Accounting-TF confident-margin threshold (~1.5).

Split (feedback #10 Stage 2.3), computed by the torch-free ``classify_step``:
  * match                         -> agree
  * mismatch AND gap <  CONF      -> NEAR-TIE FLIP  (top-2 within a hair; expected
                                      under fp4; harmless)
  * mismatch AND gap >= CONF      -> CONFIDENT MISMATCH (TRT confidently prefers a
                                      different token; a real defect)
  The gap histogram at several thresholds is reported so the split is transparent
  and not sensitive to the exact CONF.

Target / verdict (feedback #10 Stage 2.4):
  * the DIVE target is the worst CONFIDENT step (lowest ``cos_raw``);
  * if there are ZERO confident mismatches across ALL primaries -> DECODE_CLEAN_
    ACQUIT: the honest acquittal (R.6), reported with the confident-mismatch count
    behind it -- ``diffuse`` is NOT an allowed outcome (R.1).

DETERMINISM PREREQUISITE (feedback #10): bs=1, TP=4, cuda_graph=off, overlap=off,
CUTLASS, ``enable_autotuner=False`` (+ sbatch ``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``).
Before any diff is trusted, a free-run of DET_STEPS positions is generated TWICE
from the raw prompt and the two ``generation_logits`` tensors must be BYTE-IDENTICAL
(``torch.equal`` + sha256) with identical token ids -- the same determinism
argument the accepted Stage-1 probe used for pos0, extended across decode steps.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_tf_decode_probe.py
Env: INKLING_CHECKPOINT, INKLING_MM_DECODE_REF (sglang_mm_decode_fb10.json),
     INKLING_TFDEC_OUT, INKLING_TF_STEPS, INKLING_TF_CONF, INKLING_TF_DET_STEPS.
"""

import base64
import hashlib
import io
import json
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/hf_data/hf_home/hub/"
    "models--thinkingmachines--Inkling-NVFP4/snapshots/95e51a54d9486020a80d49ae4f9103fb2b3f9686",
)
REF = os.environ.get(
    "INKLING_MM_DECODE_REF",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/codes/agent-flow/workspace/"
    "inkling-advanced-bringup/results/sglang_mm_decode_fb10.json",
)
OUT = os.environ.get(
    "INKLING_TFDEC_OUT",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/codes/agent-flow/workspace/"
    "inkling-advanced-bringup/results/tf_decode_sweep_fb10.json",
)
TP = int(os.environ.get("INKLING_TP", "4"))
# Bounded teacher-force window; covers the early-divergence region (the human
# ranked these samples by EARLIEST text divergence). Reported explicitly.
TF_STEPS = int(os.environ.get("INKLING_TF_STEPS", "512"))
# Determinism window: two free-runs of this many positions must be byte-identical.
DET_STEPS = int(os.environ.get("INKLING_TF_DET_STEPS", "48"))
# Confident-margin threshold in log-prob units (matches the accepted iter51 /
# Accounting-TF methodology). The split is also reported at other thresholds.
CONF = float(os.environ.get("INKLING_TF_CONF", "1.5"))
UNPADDED_VOCAB = int(os.environ.get("INKLING_UNPADDED_VOCAB", "200058"))
MOE_BACKEND = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
# Per-call generation cap. Teacher forcing must restart from SGLang's true prefix
# at every fork (to stay teacher-forced), so an UNBOUNDED per-call generation
# (max_tokens = n - t) regenerates the whole remaining suffix on every fork -- a
# fork-storm then costs ~60s/fork (Reviewer iter127 observed this). Capping the
# per-call generation to CHUNK bounds the wasted suffix to <= CHUNK tokens per
# fork while leaving every per-step metric identical. Small enough to be cheap on
# heavy forking, large enough to cover long matched runs in few calls.
CHUNK = int(os.environ.get("INKLING_TF_CHUNK", "16"))
# Guard so a pathological fork-storm can't run unbounded (restart-on-fork bound).
MAX_TF_CALLS = int(os.environ.get("INKLING_TF_MAX_CALLS", "0")) or (TF_STEPS + 64)
# Transparency histogram of TRT's own top1-vs-sgtok gap over mismatched steps.
GAP_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101

PRIMARY_IDS = set(
    (os.environ.get("INKLING_PRIMARY_IDS") or "validation_Geography_6,validation_Math_17").split(",")
)
CONTROL_IDS = set((os.environ.get("INKLING_CONTROL_IDS") or "validation_Marketing_7").split(","))


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _role(rid):
    if rid in PRIMARY_IDS:
        return "PRIMARY"
    if rid in CONTROL_IDS:
        return "CONTROL"
    return "other"


def classify_step(match, gap, conf=CONF):
    """feedback #10 Stage 2.3 per-step split (torch-free, unit-tested).

    ``match``: TRT's argmax == SGLang's token at this step.
    ``gap``  : TRT's own logit[top1] - logit[sg_tok] (log-prob units, >= 0).
    """
    if match:
        return "agree"
    return "confident_mismatch" if float(gap) >= float(conf) else "near_tie_flip"


def _gap_histogram(gaps, thresholds=GAP_THRESHOLDS):
    """Count of mismatched-step gaps at or above each threshold (transparency)."""
    return {f">={t}": int(sum(1 for g in gaps if float(g) >= t)) for t in thresholds}


def summarize_item(rid, role, per_step, conf=CONF, thresholds=GAP_THRESHOLDS):
    """Aggregate one item's per-step records into the Stage-2 counts + DIVE target.

    Torch-free (operates on the plain-dict ``per_step`` records) so it is unit
    tested on the login node. Each per_step dict carries at least: ``t`` (step),
    ``match`` (bool), ``gap`` (float), ``cos_raw`` (float or None), ``kind``.
    The DIVE target is the CONFIDENT step with the LOWEST ``cos_raw`` (feedback
    #10 Stage 2.4); ties break to the largest gap then the earliest step.
    """
    mism = [s for s in per_step if not s["match"]]
    conf_steps = [s for s in mism if s["kind"] == "confident_mismatch"]
    near = [s for s in mism if s["kind"] == "near_tie_flip"]
    first_fork = mism[0] if mism else None

    def _cos_key(s):
        c = s.get("cos_raw")
        c = 2.0 if c is None else float(c)  # missing cos sorts as "not worst"
        return (c, -float(s.get("gap", 0.0)), int(s["t"]))

    worst = min(conf_steps, key=_cos_key) if conf_steps else None
    first_conf = min(conf_steps, key=lambda s: int(s["t"])) if conf_steps else None
    return dict(
        id=rid,
        role=role,
        n_steps=len(per_step),
        n_match=sum(1 for s in per_step if s["match"]),
        n_mismatch=len(mism),
        n_near_tie=len(near),
        n_confident=len(conf_steps),
        matched_before_first_fork=(first_fork["t"] if first_fork else len(per_step)),
        first_fork_step=(first_fork["t"] if first_fork else None),
        first_fork_gap=(round(float(first_fork["gap"]), 4) if first_fork else None),
        first_fork_kind=(first_fork["kind"] if first_fork else None),
        first_confident_step=(int(first_conf["t"]) if first_conf else None),
        worst_confident_step=(dict(worst) if worst else None),
        gap_histogram=_gap_histogram([float(s["gap"]) for s in mism], thresholds),
    )


def decide_decode_verdict(det_all, summaries):
    """feedback #10 Stage 2.4 branch (torch-free, unit-tested).

    * determinism must hold, else NONDETERMINISTIC (untrustworthy diff);
    * DECODE_CLEAN_ACQUIT iff there is >=1 PRIMARY and ZERO confident mismatches
      across ALL primaries -> the honest acquittal (R.6), reported with the count;
    * DECODE_CONFIDENT_MISMATCH iff >=1 primary confident mismatch -> DIVE the
      worst step.
    """
    if not det_all:
        return "NONDETERMINISTIC"
    primaries = [s for s in summaries if s["role"] == "PRIMARY"]
    if not primaries:
        return "NO_PRIMARY"
    total_conf = sum(s["n_confident"] for s in primaries)
    return "DECODE_CLEAN_ACQUIT" if total_conf == 0 else "DECODE_CONFIDENT_MISMATCH"


def pick_dive_target(summaries):
    """The single worst CONFIDENT step across PRIMARIES (lowest cos_raw)."""
    cands = [
        s["worst_confident_step"]
        for s in summaries
        if s["role"] == "PRIMARY" and s["worst_confident_step"] is not None
    ]
    if not cands:
        return None

    def _key(w):
        c = w.get("cos_raw")
        c = 2.0 if c is None else float(c)
        return (c, -float(w.get("gap", 0.0)), int(w["t"]))

    return min(cands, key=_key)


# --------------------------------------------------------------------------- #
# Torch metric helpers (exercised by the GPU run; imported lazily).
# --------------------------------------------------------------------------- #
def _cosine(a, b):
    import torch

    return float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def _rel_rms(trt_vec, sg_vec):
    import torch

    num = float(torch.sqrt(torch.mean((trt_vec - sg_vec) ** 2)))
    den = float(torch.sqrt(torch.mean(sg_vec ** 2))) or 1e-12
    return num / den


def compare_step(logits_pos, sg_tok, sg_top, eff, conf=CONF):
    """Compare TRT's full logit vector at one teacher-forced step against SGLang.

    ``logits_pos`` : TRT full logit vector (len>=eff) at this generated position.
    ``sg_tok``     : SGLang's greedy token at this step (its argmax).
    ``sg_top``     : SGLang [[tid, logprob], ...] top-K at this step (for cos).
    Returns the per-step record (torch-free values) incl. ``gap`` and ``kind``.
    """
    import torch

    trt_argmax = int(logits_pos[:eff].argmax())
    match = bool(trt_argmax == int(sg_tok))
    # gap = logit[trt_top1] - logit[sg_tok]; lse cancels -> already log-prob units.
    gap = float(logits_pos[trt_argmax] - logits_pos[int(sg_tok)])
    # cosine over SGLang's returned top-K support (max-shifted RAW), like Stage 1.
    supp = [(int(tid), float(lp)) for tid, lp in sg_top if 0 <= int(tid) < eff]
    cos_raw = None
    rel_rms_raw = None
    max_abs_raw = None
    if len(supp) >= 2:
        ids = torch.tensor([tid for tid, _ in supp], dtype=torch.long)
        sg_lp = torch.tensor([lp for _, lp in supp], dtype=torch.float32)
        sel = logits_pos.index_select(0, ids)
        trt_raw = sel - logits_pos[:eff].max()
        sg_raw = sg_lp - sg_lp.max()
        cos_raw = round(_cosine(trt_raw, sg_raw), 8)
        rel_rms_raw = round(_rel_rms(trt_raw, sg_raw), 6)
        max_abs_raw = round(float((trt_raw - sg_raw).abs().max()), 6)
    trt_lse = float(torch.logsumexp(logits_pos[:eff], dim=0))
    tv, ti = torch.topk(logits_pos[:eff], k=min(5, eff))
    trt_top5 = [[int(ti[j]), round(float(tv[j] - trt_lse), 4)] for j in range(tv.shape[0])]
    kind = classify_step(match, gap, conf)
    return dict(
        trt_top1=trt_argmax,
        sg_tok=int(sg_tok),
        match=match,
        gap=round(gap, 4),
        trt_logp_sgtok=round(float(logits_pos[int(sg_tok)] - trt_lse), 4),
        cos_raw=cos_raw,
        rel_rms_raw=rel_rms_raw,
        max_abs_raw=max_abs_raw,
        kind=kind,
        trt_top5=trt_top5,
        sglang_top5=[[int(t), round(float(l), 4)] for t, l in supp[:5]],
    )


def _gen(llm, SamplingParams, TokensPrompt, forced, mm, max_tokens):
    """One teacher-forced generate call. Returns (trt_token_ids, generation_logits
    [k,eff] cpu float tensor, eff)."""
    import torch

    out = llm.generate(
        [TokensPrompt(prompt_token_ids=list(forced), multi_modal_data=mm)],
        SamplingParams(max_tokens=max_tokens, temperature=0.0, return_generation_logits=True),
    )[0]
    o = out.outputs[0]
    gl = o.generation_logits
    assert gl is not None, "generation_logits is None (gather_generation_logits not honored)"
    gl = torch.as_tensor(gl).float().cpu()
    if gl.dim() == 1:
        gl = gl.unsqueeze(0)
    eff = min(gl.shape[-1], UNPADDED_VOCAB)
    return [int(t) for t in o.token_ids], gl[:, :eff].contiguous(), eff


def teacher_force(llm, SamplingParams, TokensPrompt, input_ids, sg_ids, sg_top, mm, conf=CONF, label=""):
    """Bounded restart-on-fork teacher-forced sweep along SGLang's own token ids.

    Each call re-prefills SGLang's TRUE prefix ``input_ids + sg_ids[:t]`` (so both
    stacks are always on the identical prefix -- never free-run) and generates at
    most ``CHUNK`` tokens. Positions are compared step by step; between forks TRT's
    greedy tokens == SGLang's so the chunk's ``generation_logits`` are valid
    teacher-forced logits. At the FIRST mismatch inside a chunk we record the
    full-vector metrics and restart from SGLang's token (``t = fork+1``); the
    bounded CHUNK caps the wasted suffix to < CHUNK tokens per fork. Progress is
    printed each call so a long sweep is observable. Uses SGLang's token IDS
    directly (no re-tokenization).
    """
    n = min(TF_STEPS, len(sg_ids))
    t = 0
    per_step = []
    n_calls = 0
    while t < n and n_calls < MAX_TF_CALLS:
        forced = list(input_ids) + [int(x) for x in sg_ids[:t]]
        want = min(CHUNK, n - t)
        trt_ids, gl, eff = _gen(llm, SamplingParams, TokensPrompt, forced, mm, want)
        n_calls += 1
        t_before = t
        forked = False
        if not trt_ids:
            # TRT emitted nothing (e.g. immediate EOS); force SGLang's token on so
            # t always advances (termination guarantee) and record the step.
            per_step.append(dict(t=t, trt_top1=-1, sg_tok=int(sg_ids[t]), match=False,
                                 gap=0.0, cos_raw=None, kind="near_tie_flip", note="trt_empty"))
            t += 1
        else:
            for i in range(len(trt_ids)):
                tt_t = t + i
                if tt_t >= n:
                    break
                rec = compare_step(gl[i], sg_ids[tt_t], sg_top[tt_t], eff, conf)
                rec["t"] = tt_t
                per_step.append(rec)
                if not rec["match"]:
                    t = tt_t + 1
                    forked = True
                    break
            if not forked:
                t = t + len(trt_ids)
        # Safety: guarantee forward progress even in a degenerate no-advance case.
        if t <= t_before:
            t = t_before + 1
        if n_calls == 1 or forked or (n_calls % 8 == 0) or t >= n:
            nc = sum(1 for s in per_step if not s["match"] and s["kind"] == "confident_mismatch")
            print(f"  [tfdec {label}] t={t}/{n} calls={n_calls} steps={len(per_step)} "
                  f"confident_so_far={nc}", flush=True)
    return per_step, n_calls


def _det_check(llm, SamplingParams, TokensPrompt, input_ids, mm):
    """Two free-run generates of DET_STEPS positions must be byte-identical."""
    import torch

    a_ids, a_gl, _ = _gen(llm, SamplingParams, TokensPrompt, input_ids, mm, DET_STEPS)
    b_ids, b_gl, _ = _gen(llm, SamplingParams, TokensPrompt, input_ids, mm, DET_STEPS)
    shaA = hashlib.sha256(a_gl.numpy().tobytes()).hexdigest()
    shaB = hashlib.sha256(b_gl.numpy().tobytes()).hexdigest()
    identical = bool(a_ids == b_ids and a_gl.shape == b_gl.shape and torch.equal(a_gl, b_gl))
    return identical, dict(n=len(a_ids), shaA=shaA[:16], shaB=shaB[:16], ids_equal=bool(a_ids == b_ids))


def main() -> int:
    import torch
    from PIL import Image

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "tf-decode probe needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [
        r
        for r in ref
        if r.get("input_ids") and r.get("greedy_token_ids") and r.get("pos_top") and r.get("image_b64")
    ]
    assert len(ref) >= 1, f"no usable SGLang decode refs in {REF}"
    print(
        f"[tfdec] tp={TP} moe={MOE_BACKEND} n={len(ref)} tf_steps={TF_STEPS} det_steps={DET_STEPS} "
        f"conf={CONF} baseline cuda_graph=off overlap=off enable_autotuner=False bs=1 ref={REF}\n"
        f"[tfdec] allreduce_autotune_disabled={os.environ.get('TLLM_DISABLE_ALLREDUCE_AUTOTUNE','0')} "
        f"ids={[r['id'] for r in ref]}",
        flush=True,
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.6, dtype="auto", enable_block_reuse=False, host_cache_size=0
    )
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=MOE_BACKEND),
        kv_cache_config=kv_cache_config,
        gather_generation_logits=True,
        cuda_graph_config=None,  # baseline: eager, no graph
        disable_overlap_scheduler=True,  # overlap OFF
        enable_autotuner=False,  # deterministic
        max_seq_len=4096,
        max_batch_size=1,
        max_num_tokens=4096,
    )
    print(
        "[tfdec] moe_backend=CUTLASS cuda_graph_hard_path=eager(no-graph) attn=TRTLLM "
        "kv=KVCacheManagerV2 bs=1",
        flush=True,
    )

    rows = []
    summaries = []
    det_all = True
    try:
        for r in ref:
            rid = r["id"]
            role = _role(rid)
            image = Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))
            mm = {"image": [image]}
            input_ids = _trt_ids(r["input_ids"])
            sg_ids = [int(t) for t in r["greedy_token_ids"]]
            sg_top = r["pos_top"]
            n_win = min(TF_STEPS, len(sg_ids))
            print(
                f"\n[tfdec] START {role} {rid} n_input={len(input_ids)} sg_ref_tokens={len(sg_ids)} "
                f"window={n_win} chunk={CHUNK} det_steps={DET_STEPS}",
                flush=True,
            )

            det_ok, det_diag = _det_check(llm, SamplingParams, TokensPrompt, input_ids, mm)
            det_all = det_all and det_ok
            print(f"  [tfdec {rid}] det_ok={det_ok} {det_diag}", flush=True)

            per_step, n_calls = teacher_force(
                llm, SamplingParams, TokensPrompt, input_ids, sg_ids, sg_top, mm, CONF, label=rid
            )
            summ = summarize_item(rid, role, per_step, CONF)
            summ["tf_calls"] = n_calls
            summ["sg_ref_tokens"] = len(sg_ids)
            summ["det_ok"] = det_ok
            summ["det"] = det_diag
            summaries.append(summ)
            rows.append(dict(id=rid, role=role, per_step=per_step))
            w = summ["worst_confident_step"]
            print(
                f"\n  [{role:<7} {rid}] det={det_ok} steps={summ['n_steps']} "
                f"match={summ['n_match']} near_tie={summ['n_near_tie']} confident={summ['n_confident']}\n"
                f"        matched_before_first_fork={summ['matched_before_first_fork']} "
                f"first_fork=(step={summ['first_fork_step']},kind={summ['first_fork_kind']},"
                f"gap={summ['first_fork_gap']})\n"
                f"        first_confident_step={summ['first_confident_step']} "
                f"gap_hist={summ['gap_histogram']}\n"
                f"        worst_confident={'None' if not w else dict(t=w['t'], cos_raw=w['cos_raw'], gap=w['gap'], trt=w['trt_top1'], sg=w['sg_tok'])}",
                flush=True,
            )
    finally:
        llm.shutdown()

    verdict = decide_decode_verdict(det_all, summaries)
    dive_target = pick_dive_target(summaries) if verdict == "DECODE_CONFIDENT_MISMATCH" else None
    primaries = [s for s in summaries if s["role"] == "PRIMARY"]
    total_conf_primaries = sum(s["n_confident"] for s in primaries)
    if verdict == "DECODE_CLEAN_ACQUIT":
        branch = (
            "HONEST DECODE ACQUITTAL (feedback #10 Stage 2.4): zero confident mismatches on "
            "primaries; STOP for human review."
        )
    elif verdict == "DECODE_CONFIDENT_MISMATCH":
        branch = "DIVE PROCEDURE on the worst confident decode step (feedback #10 Stage 2.4)"
    else:
        branch = "re-check determinism"

    doc = dict(
        title="feedback #10 Stage 2 -- teacher-forced decode sweep vs live SGLang",
        config=dict(
            tp=TP,
            moe_backend=MOE_BACKEND,
            cuda_graph=False,
            overlap=False,
            enable_autotuner=False,
            allreduce_autotune_disabled=os.environ.get("TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0"),
            bs=1,
            attn="TRTLLM",
            kv="KVCacheManagerV2",
            tf_steps=TF_STEPS,
            det_steps=DET_STEPS,
            conf=CONF,
            gap_thresholds=GAP_THRESHOLDS,
            method="teacher-forced on SGLang's OWN token ids (restart-on-fork); "
            "full-vector TRT logits per step; cos over SGLang top-K support; "
            "gap = logit[trt_top1]-logit[sg_tok] in log-prob units. Window is a "
            "bounded prefix of the (finish=length, n_gen=2560) reasoning stream, "
            "sized to cover the earliest-divergence region the human ranked on.",
        ),
        reference=REF,
        determinism_all_byte_identical=det_all,
        verdict=verdict,
        next_branch=branch,
        total_confident_mismatches_primaries=total_conf_primaries,
        dive_target=dive_target,
        summaries=summaries,
        records=rows,
    )
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)

    print(
        f"\nINKLING_TFDEC_DET {'PASS' if det_all else 'FAIL'} "
        f"byte_identical={sum(1 for s in summaries if s['det_ok'])}/{len(summaries)}",
        flush=True,
    )
    tstr = "None"
    if dive_target is not None:
        tstr = f"(t={dive_target['t']},cos_raw={dive_target['cos_raw']},gap={dive_target['gap']})"
    print(
        f"INKLING_TFDEC_VERDICT {verdict} confident_primaries={total_conf_primaries} "
        f"dive_target={tstr} next={branch} out={OUT}",
        flush=True,
    )
    # rc: 0 iff determinism held (a decisive acquit/convict is a valid outcome).
    return 0 if det_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("INKLING_TFDEC FAIL: exception producing evidence", flush=True)
        sys.exit(1)
