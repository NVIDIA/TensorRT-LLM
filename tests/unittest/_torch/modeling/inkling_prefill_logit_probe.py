#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HUMAN FEEDBACK #10 -- Stage 1: FINAL-PREFILL-LOGIT comparison vs SGLang.

``reference_tier=real_source``, ``validation_tier=real_runtime``.

The 197-item MMMU measurement returned INCONCLUSIVE (TRT 0.6802 vs SGLang 0.7157,
mean_delta -0.0355). Feedback #10 asks: is the gap a real prefill defect, or does
divergence start in decode?  This driver answers the STAGE-1 branch question by
comparing, for the SELECTED failing/control samples, the FINAL PREFILL LOGITS --
the full logit vector at the last prompt position, i.e. the distribution that
produces the FIRST generated token -- between the TRT-LLM production stack and the
live SGLang reference (SGLang is the standard).

Samples (feedback #10 Stage 0, DO NOT re-derive):
  * PRIMARY (TRT wrong / SGLang right, earliest text divergence):
      validation_Geography_6  (cache Geography:5, gold C, TRT D / SGLang C)
      validation_Math_17      (cache Math:16,     gold A, TRT B / SGLang A)
  * CONTROL (both arms correct, long common prefix):
      validation_Marketing_7  (cache Marketing:6, gold D)
  A real prefill defect must appear on the PRIMARIES and NOT on the CONTROL.

Method (mirrors the accepted ``inkling_image_logit_replay_test`` contract):
  * Feed byte-identical ``input_ids`` + aligned image (rebuilt from the reference
    ``image_b64``) to the FULL TP=4 stack (KVCacheManagerV2 + TRTLLM attention +
    NVFP4 CUTLASS MoE + hMLP vision fusion).
  * ``gather_generation_logits=True`` -> ``out.generation_logits[0]`` is the full
    logit vector at the FINAL PREFILL position (produces the first gen token).
  * SGLang's ``pos_top[0]`` (top-K [token_id, logprob] at the same position,
    captured live from ``sglang serve``) is the standard.  We report top-1 argmax
    agreement, the top-5 with values, and cos / rel_rms / max_abs over SGLang's
    returned top-K support (SGLang serve returns top-K logprobs, not the full
    200k vector, so the vector metrics are over that shared support -- stated
    explicitly; the TRT full-vocab argmax is ALSO reported so a divergence whose
    argmax escapes the support is still caught).
  * COMPLETENESS WITNESS (Reviewer iter124): a top-K-only cos may CONVICT a
    divergence but must not, alone, ACQUIT one that hides outside the top-K. Since
    TRT gives the FULL vocab vector, ``support_complete`` certifies -- using TRT's
    full vector as the witness -- that SGLang's captured support reaches below the
    coverage threshold AND that neither arm has a probable token (logprob >=
    COVERAGE_LOGPROB) the other missed. PREFILL_CLEAN requires this witness, so a
    top-K-only acquittal is honest without SGLang's full 200k vector.

DETERMINISM PREREQUISITE (feedback #10): bs=1, TP=4, cuda_graph=off, overlap=off,
CUTLASS, ``enable_autotuner=False`` (+ sbatch ``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``).
The forward is run TWICE and the two ``generation_logits[0]`` tensors must be
BYTE-IDENTICAL (``torch.equal`` + sha256) before any diff vs SGLang is trusted.

Branch (feedback #10 Stage 1.3):
  * ``PREFILL_DIVERGENT``  -> root cause is in prefill; run the DIVE next.
  * ``PREFILL_CLEAN``      -> cos_raw >= COS_CLEAN AND top-1 agree on PRIMARIES
    AND CONTROL -> divergence is in decode; go to Stage 2 next iteration.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_prefill_logit_probe.py
Env: INKLING_CHECKPOINT, INKLING_MM_REF (sglang_mm_ref_fb10.json), INKLING_PREFILL_OUT.
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
    "INKLING_MM_REF",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/codes/agent-flow/workspace/"
    "inkling-advanced-bringup/results/sglang_mm_ref_fb10.json",
)
OUT = os.environ.get(
    "INKLING_PREFILL_OUT",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/codes/agent-flow/workspace/"
    "inkling-advanced-bringup/results/prefill_logit_compare_fb10.json",
)
TP = int(os.environ.get("INKLING_TP", "4"))
CONT = int(os.environ.get("INKLING_PREFILL_CONT", "8"))  # gen tokens (only pos0 used)
UNPADDED_VOCAB = int(os.environ.get("INKLING_UNPADDED_VOCAB", "200058"))
# feedback #10 Stage 1.3: prefill CLEAN requires cos >= 0.9999 on primaries AND control.
COS_CLEAN = float(os.environ.get("INKLING_PREFILL_COS_CLEAN", "0.9999"))
# Completeness-witness threshold. SGLang serve returns only its top-K logprobs, so a
# cos over that shared support cannot, alone, rule out a divergence hiding OUTSIDE the
# top-K. TRT gives the FULL 200k vector, which lets us certify that no NON-NEGLIGIBLE
# token (true logprob >= COVERAGE_LOGPROB, i.e. prob >= ~3e-7 at -15) hides outside the
# shared support on either arm. Reviewer(iter124): a top-K-only cos may CONVICT a
# divergence but must NOT acquit (PREFILL_CLEAN) unless this witness certifies coverage.
COVERAGE_LOGPROB = float(os.environ.get("INKLING_PREFILL_COVERAGE_LOGPROB", "-15.0"))
MOE_BACKEND = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")

TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101

# feedback #10 Stage 0 role labels (union ids), used only to classify the verdict.
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


def classify_item_clean(
    byte_identical, finite, top1_agree, cos_raw, support_complete, cos_clean=COS_CLEAN
):
    """feedback #10 Stage 1.3 per-item CLEAN test (torch-free, unit-tested).

    An item is prefill-clean only if the two forwards were byte-identical
    (determinism is a hard prerequisite -- a non-deterministic pos0 invalidates
    any diff), the logits are finite, the TRT full-vocab argmax equals SGLang's
    top-1, the cosine over SGLang's support meets the CLEAN threshold, AND the
    completeness witness certifies that no non-negligible token hides outside the
    shared top-K support (Reviewer iter124: a top-K cos alone cannot acquit).
    """
    return bool(
        byte_identical
        and finite
        and top1_agree
        and (cos_raw >= cos_clean)
        and support_complete
    )


def support_complete(sg_support, trt_probable, sg_support_min_logprob, cov_logprob=COVERAGE_LOGPROB):
    """Completeness witness for a top-K-only SGLang reference (torch-free, unit-tested).

    SGLang serve returns only its top-K logprobs, so a cosine over that shared
    support cannot, by itself, rule out a divergence hiding OUTSIDE the top-K. TRT
    gives the full 200k vector, so we certify that no NON-NEGLIGIBLE token (true
    logprob >= ``cov_logprob``) hides outside the shared support on either arm:

      * ``support_deep_enough``: SGLang's captured support must itself reach at or
        below ``cov_logprob`` (its weakest reported token <= cov_logprob). If the
        top-K is too shallow to cover the >= cov_logprob region, we must NOT certify
        CLEAN from it -- widen K instead.
      * no ``sg_leak``: every SGLang-probable token (logprob >= cov_logprob) is in
        TRT's probable set. A leak here means TRT ranks a SGLang-probable token
        improbable -> a real prefill divergence.
      * no ``trt_leak``: every TRT-probable token (logprob >= cov_logprob) is in
        SGLang's reported support. A leak here means TRT ranks a token probable that
        SGLang ranked outside its top-K -> a real prefill divergence.

    ``sg_support``   : ``[[tid, logprob], ...]`` SGLang top-K (true log-probs).
    ``trt_probable`` : ``[[tid, logprob], ...]`` TRT tokens with logprob >= cov_logprob
                       over the FULL vocab (true log-probs, same units as SGLang).
    Returns ``(complete: bool, diag: dict)``.
    """
    sg_ids = {int(t): float(lp) for t, lp in sg_support}
    trt_ids = {int(t): float(lp) for t, lp in trt_probable}
    deep = bool(float(sg_support_min_logprob) <= cov_logprob)
    sg_leak = {t: lp for t, lp in sg_ids.items() if lp >= cov_logprob and t not in trt_ids}
    trt_leak = {t: lp for t, lp in trt_ids.items() if lp >= cov_logprob and t not in sg_ids}
    complete = bool(deep and not sg_leak and not trt_leak)

    def _mx(d):
        return round(max(d.values()), 4) if d else None

    def _top(d):
        return {int(k): round(float(v), 4) for k, v in sorted(d.items(), key=lambda kv: -kv[1])[:10]}

    return complete, dict(
        support_complete=complete,
        support_deep_enough=deep,
        n_sg_probable_leak=len(sg_leak),
        n_trt_probable_leak=len(trt_leak),
        max_sg_leak_logprob=_mx(sg_leak),
        max_trt_leak_logprob=_mx(trt_leak),
        sg_leak_tokens=_top(sg_leak),
        trt_leak_tokens=_top(trt_leak),
        sg_support_min_logprob=round(float(sg_support_min_logprob), 4),
        n_trt_probable=len(trt_ids),
        cov_logprob=cov_logprob,
    )


def decide_verdict(det_all, rows):
    """feedback #10 Stage 1.3 branch decision (torch-free, unit-tested).

    ``rows`` carry ``role`` in {PRIMARY, CONTROL, other} and ``item_clean``.
      * determinism must hold across ALL items, else NONDETERMINISTIC;
      * PREFILL_CLEAN iff every PRIMARY and CONTROL item is clean (and at least
        one of each exists), which routes to Stage 2 (decode sweep);
      * otherwise PREFILL_DIVERGENT, which routes to the DIVE over prefill.
    """
    if not det_all:
        return "NONDETERMINISTIC"
    primaries = [r for r in rows if r["role"] == "PRIMARY"]
    controls = [r for r in rows if r["role"] == "CONTROL"]
    if not primaries or not controls:
        return "PREFILL_DIVERGENT"
    clean_all = all(r["item_clean"] for r in primaries + controls)
    return "PREFILL_CLEAN" if clean_all else "PREFILL_DIVERGENT"


def _cosine(a, b):
    import torch

    return float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def _rel_rms(trt_vec, sg_vec):
    """RMS(trt - sg) / RMS(sg) over the shared support (feedback #10 D.2 metric)."""
    import torch

    num = float(torch.sqrt(torch.mean((trt_vec - sg_vec) ** 2)))
    den = float(torch.sqrt(torch.mean(sg_vec ** 2))) or 1e-12
    return num / den


def compare_pos(logits_pos, sg_top, eff):
    """Compare TRT's full final-prefill logit vector against SGLang's top-K.

    ``logits_pos`` : TRT full logit vector at the final prefill position (len=eff).
    ``sg_top``     : SGLang [[token_id, logprob], ...] top-K at the same position.
    All vector metrics (cos / rel_rms / max_abs) are over SGLang's returned
    support only, because SGLang serve does not return the full 200k vector.
    Two normalizations are reported:
      * RAW: (logit - max_logit) vs SGLang (logprob - max_logprob); these are the
        SAME units, since SGLang logprob = logit - logsumexp so the lse constant
        cancels under the per-arm max-shift.
      * LOGP: true log-probs (logit - logsumexp) vs SGLang logprob.
    """
    import torch

    trt_argmax_full = int(logits_pos.argmax())  # over the FULL vocab (not just support)
    supp = [(int(tid), float(lp)) for tid, lp in sg_top if 0 <= int(tid) < eff]
    ids = torch.tensor([tid for tid, _ in supp], dtype=torch.long)
    sg_lp = torch.tensor([lp for _, lp in supp], dtype=torch.float32)
    trt_lse = torch.logsumexp(logits_pos, dim=0)
    sel = logits_pos.index_select(0, ids)
    trt_lp = sel - trt_lse
    trt_raw = sel - logits_pos.max()
    sg_raw = sg_lp - sg_lp.max()
    sg_argmax = int(ids[int(sg_lp.argmax())])
    # ---- completeness witness (uses TRT's FULL vocab as the coverage witness) ----
    # Every TRT token whose true log-prob clears the coverage threshold; if SGLang's
    # captured top-K is deep enough and neither arm has a probable token the other
    # missed, a top-K-only cos may honestly ACQUIT (Reviewer iter124).
    trt_lp_full = logits_pos - trt_lse
    prob_idx = (trt_lp_full >= COVERAGE_LOGPROB).nonzero(as_tuple=False).flatten()
    trt_probable = [[int(i), float(trt_lp_full[int(i)])] for i in prob_idx.tolist()]
    sg_support_min = float(sg_lp.min()) if sg_lp.numel() else 0.0
    _complete, comp_diag = support_complete(supp, trt_probable, sg_support_min, COVERAGE_LOGPROB)
    # TRT top-5 over full vocab (token, logit, logprob)
    tv, ti = torch.topk(logits_pos, k=min(5, logits_pos.shape[0]))
    trt_top5 = [
        [int(ti[j]), round(float(tv[j]), 4), round(float(tv[j] - trt_lse), 4)]
        for j in range(tv.shape[0])
    ]
    sg_top5 = [[int(t), round(float(l), 4)] for t, l in supp[:5]]
    return dict(
        trt_argmax_full=trt_argmax_full,
        sg_argmax=sg_argmax,
        top1_agree=bool(trt_argmax_full == sg_argmax),
        argmax_in_support=bool(trt_argmax_full in [tid for tid, _ in supp]),
        k=len(ids),
        finite=bool(torch.isfinite(logits_pos).all()),
        cos_raw=round(_cosine(trt_raw, sg_raw), 8),
        cos_lp=round(_cosine(trt_lp, sg_lp), 8),
        rel_rms_raw=round(_rel_rms(trt_raw, sg_raw), 6),
        rel_rms_lp=round(_rel_rms(trt_lp, sg_lp), 6),
        max_abs_raw=round(float((trt_raw - sg_raw).abs().max()), 6),
        max_abs_lp=round(float((trt_lp - sg_lp).abs().max()), 6),
        trt_top5=trt_top5,
        sglang_top5=sg_top5,
        **comp_diag,
    )


def _prefill_logits(llm, SamplingParams, TokensPrompt, ref, tag):
    """One full forward over the 3 prompts (bs=1 each). Returns per-id the pos0
    full logit vector (cpu float tensor) + its sha256, plus the sampled tok0."""
    import torch

    sampling = SamplingParams(max_tokens=CONT, temperature=0.0, return_generation_logits=True)
    out = {}
    for r in ref:
        prompt = TokensPrompt(
            prompt_token_ids=_trt_ids(r["input_ids"]),
            multi_modal_data={"image": [Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))]},
        )
        gen = llm.generate([prompt], sampling)[0].outputs[0]
        gl = gen.generation_logits
        assert gl is not None, f"generation_logits is None for {r['id']} (gather not honored)"
        gl = torch.as_tensor(gl).float().cpu()
        if gl.dim() == 1:
            gl = gl.unsqueeze(0)
        eff = min(gl.shape[-1], UNPADDED_VOCAB)
        pos0 = gl[0, :eff].contiguous()
        sha = hashlib.sha256(pos0.numpy().tobytes()).hexdigest()
        samp0 = int(gen.token_ids[0]) if gen.token_ids else -1
        out[r["id"]] = dict(pos0=pos0, eff=eff, sha=sha, samp0=samp0)
        print(
            f"  [{tag}] {r['id']:<26} pos0_argmax={int(pos0.argmax())} "
            f"samp0={samp0} sha={sha[:12]} eff={eff}",
            flush=True,
        )
    return out


def main() -> int:
    import torch
    global Image
    from PIL import Image

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "prefill logit probe needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids") and r.get("pos_top") and r.get("image_b64")]
    assert len(ref) >= 1, f"no usable SGLang prefill refs in {REF}"
    print(
        f"[prefill] tp={TP} moe={MOE_BACKEND} n_prompts={len(ref)} cont={CONT} "
        f"cos_clean={COS_CLEAN} baseline cuda_graph=off overlap=off "
        f"enable_autotuner=False deterministic bs=1 ref={REF}\n"
        f"[prefill] allreduce_autotune_disabled="
        f"{os.environ.get('TLLM_DISABLE_ALLREDUCE_AUTOTUNE', '0')} "
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
        max_batch_size=1,  # bs=1 deterministic
        max_num_tokens=4096,
    )
    print(
        "[prefill] moe_backend=CUTLASS cuda_graph_hard_path=eager(no-graph) "
        "attn=TRTLLM kv=KVCacheManagerV2 bs=1",
        flush=True,
    )

    # ---- DETERMINISM: two full forwards, require byte-identical pos0 logits ----
    try:
        runA = _prefill_logits(llm, SamplingParams, TokensPrompt, ref, "runA")
        runB = _prefill_logits(llm, SamplingParams, TokensPrompt, ref, "runB")
    finally:
        llm.shutdown()

    ref_by_id = {r["id"]: r for r in ref}

    rows = []
    det_all = True
    for rid in [r["id"] for r in ref]:
        a, b = runA[rid], runB[rid]
        byte_identical = bool(torch.equal(a["pos0"], b["pos0"])) and (a["sha"] == b["sha"])
        det_all = det_all and byte_identical
        sg = ref_by_id[rid]
        sg_top0 = sg["pos_top"][0]
        cmp = compare_pos(a["pos0"], sg_top0, a["eff"])
        role = _role(rid)
        item_clean = classify_item_clean(
            byte_identical, cmp["finite"], cmp["top1_agree"], cmp["cos_raw"],
            cmp["support_complete"], COS_CLEAN
        )
        sg_greedy0 = int(sg["greedy_token_ids"][0]) if sg.get("greedy_token_ids") else cmp["sg_argmax"]
        row = dict(
            id=rid,
            role=role,
            byte_identical_runs=byte_identical,
            shaA=a["sha"][:16],
            shaB=b["sha"][:16],
            trt_samp0=a["samp0"],
            sglang_greedy0=sg_greedy0,
            item_clean=item_clean,
            **cmp,
        )
        rows.append(row)
        print(
            f"\n  [{role:<7} {rid}] det={byte_identical} clean={item_clean}\n"
            f"        top-1: TRT={cmp['trt_argmax_full']} SGLang={cmp['sg_argmax']} "
            f"agree={cmp['top1_agree']} (trt_argmax_in_support={cmp['argmax_in_support']})\n"
            f"        cos_raw={cmp['cos_raw']:.6f} cos_lp={cmp['cos_lp']:.6f} "
            f"rel_rms_raw={cmp['rel_rms_raw']:.4f} max_abs_raw={cmp['max_abs_raw']:.4f} "
            f"k={cmp['k']}\n"
            f"        support_complete={cmp['support_complete']} deep_enough={cmp['support_deep_enough']} "
            f"sg_min_logp={cmp['sg_support_min_logprob']} n_trt_probable={cmp['n_trt_probable']} "
            f"sg_leak={cmp['n_sg_probable_leak']} trt_leak={cmp['n_trt_probable_leak']}\n"
            f"        TRT top5 (tok,logit,logp): {cmp['trt_top5']}\n"
            f"        SGL top5 (tok,logp):       {cmp['sglang_top5']}",
            flush=True,
        )

    primaries = [r for r in rows if r["role"] == "PRIMARY"]
    controls = [r for r in rows if r["role"] == "CONTROL"]
    # Branch verdict (torch-free, unit-tested helper).
    verdict = decide_verdict(det_all, rows)
    branch = (
        "Stage 2 (teacher-forced decode sweep)"
        if verdict == "PREFILL_CLEAN"
        else ("DIVE PROCEDURE over prefill forward" if verdict == "PREFILL_DIVERGENT" else "re-check determinism")
    )

    doc = dict(
        title="feedback #10 Stage 1 -- final-prefill-logit comparison vs live SGLang",
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
            cont=CONT,
            cos_clean=COS_CLEAN,
            cov_logprob=COVERAGE_LOGPROB,
            support="SGLang top-K logprobs (serve does not return full 200k vector); "
            "vector metrics (cos/rel_rms/max_abs) are over that shared support. A "
            "top-K-only cos can CONVICT a divergence but CANNOT acquit; PREFILL_CLEAN "
            "additionally requires the completeness witness (support_complete) that uses "
            "TRT's FULL vocab to certify no token with logprob>=cov_logprob hides outside "
            "the shared support on either arm, and that SGLang's support reaches below "
            "cov_logprob (support_deep_enough).",
        ),
        reference=REF,
        determinism_all_byte_identical=det_all,
        verdict=verdict,
        next_branch=branch,
        primaries=[r["id"] for r in primaries],
        controls=[r["id"] for r in controls],
        records=rows,
    )
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)

    print(
        f"\nINKLING_PREFILL_DET {'PASS' if det_all else 'FAIL'} "
        f"byte_identical={sum(r['byte_identical_runs'] for r in rows)}/{len(rows)}",
        flush=True,
    )
    print(
        f"INKLING_PREFILL_VERDICT {verdict} clean_primaries="
        f"{sum(r['item_clean'] for r in primaries)}/{len(primaries)} "
        f"clean_controls={sum(r['item_clean'] for r in controls)}/{len(controls)} "
        f"next={branch} out={OUT}",
        flush=True,
    )
    # rc: 0 iff determinism held AND we produced a decisive branch (clean or divergent).
    # A NONDETERMINISTIC result is a failure to produce trustworthy evidence.
    return 0 if det_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("INKLING_PREFILL FAIL: exception producing evidence", flush=True)
        sys.exit(1)
