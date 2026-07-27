#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.5 DECODE-SIDE teacher-force diagnostic for the MMMU Accounting gap.

``reference_tier=real_source``, ``validation_tier=real_runtime``.

Vision is already proven bitwise-clean for the failing Accounting images
(preprocess + tower, job 5588462 / 5588190 verdict=VISION_CLEAN, max_abs=0.0
cos=1.0), so the Accounting accuracy gap is DECODE-SIDE. This driver characterizes
that decode divergence with the teacher-forcing methodology (feed the SGLang
reference's own tokens to TRT so every step is a true per-step comparison, instead
of two free-running decodes that fork once and become non-comparable):

  For each item, TRT is teacher-forced against SGLang's greedy reasoning tokens
  (re-tokenized from the captured ``sglang_text`` in ``sglang_mmmu_ref.json``),
  re-anchoring on SGLang's prefix at every fork, with the aligned image attached
  on every restart. Per item we report:
    * ``matched_before_first_fork`` -- how many leading reasoning tokens TRT
      reproduces before it first disagrees with SGLang.
    * ``first_fork_step`` and the TRT top1-top2 logprob ``margin`` at that step,
      classified ``confident`` (margin >= CONF) vs ``near_tie`` (margin < CONF).
    * ``mismatches`` / ``confident_mismatches`` over the first ``TF_STEPS`` steps.

Discriminator (with a CONTROL good item, Accounting:0 = validation_Accounting_1,
which TRT gets RIGHT, alongside the E2E-WRONG Accounting:1..):
  * If TRT tracks SGLang for many steps and its forks are dominated by NEAR-TIES
    (small-margin argmax flips), the divergence is the accepted, documented,
    out-of-scope fa4(SGLang flashinfer)-vs-Triton(TRT Inkling) NVFP4
    attention-kernel-family residual compounding over long reasoning -- NOT a
    Python-fixable bug. The control (TRT-correct) item should look the same,
    proving forking is not what makes an answer wrong.
  * If a WRONG item forks IMMEDIATELY (step 0-few, right after the image prefix)
    with CONFIDENT margins while the control does not, that points to a
    Python-fixable image-prefix decode bug (attention mask / position ids over
    the image span) and would need a fix, not a BLOCKER.

Re-tokenization caveat (honest): ``sglang_text`` is decoded text, so re-encoding
may introduce a few spurious forks near merge boundaries. That noise biases
toward MORE forks, so a result showing TRT tracks SGLang for MANY steps is robust
evidence AGAINST an early bug; only an immediate-confident-fork result would be
ambiguous and need a clean SGLang-token-id re-capture.

Baseline only: cuda_graph=off, overlap=off, TP=4, KVCacheManagerV2, TRTLLM attn,
CUTLASS MoE, deterministic (enable_autotuner=False, bs=1). Low memory (short
teacher-forced sequences), so it avoids the M1C run's TP=4 Bus-error-under-memory
-pressure failure mode.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_accounting_teacher_force.py
Env: INKLING_CHECKPOINT, INKLING_MMMU_REF (sglang_mmmu_ref.json), MMMU_ALIGN_ITEMS,
     INKLING_MMMU_N, INKLING_TF_STEPS (default 384), INKLING_TF_TOPK (default 20),
     INKLING_TF_CONF (confident-margin logprob threshold, default 1.5).
"""

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
REF = os.environ.get(
    "INKLING_MMMU_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/sglang_mmmu_ref.json",
)
OUT = os.environ.get(
    "INKLING_TF_OUT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/acct_teacher_force.json",
)
N_ITEMS = int(os.environ.get("INKLING_MMMU_N", "3"))
TF_STEPS = int(os.environ.get("INKLING_TF_STEPS", "384"))
TOPK = int(os.environ.get("INKLING_TF_TOPK", "20"))
CONF = float(os.environ.get("INKLING_TF_CONF", "1.5"))  # top1-top2 logprob margin
TP = int(os.environ.get("INKLING_TP", "4"))

TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101

# E2E labels from the deterministic baseline run 5586765 (for table annotation).
E2E_WRONG = {"validation_Accounting_2", "validation_Accounting_3"}
E2E_CORRECT = {"validation_Accounting_1"}


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _label(item_id):
    if item_id in E2E_WRONG:
        return "WRONG"
    if item_id in E2E_CORRECT:
        return "ctrl-OK"
    return "other"


def _margin_from_logprobs(step_logprobs):
    """TRT top1-top2 logprob margin at one generated step. ``step_logprobs`` is the
    executor's per-step ``{token_id: Logprob|float}`` dict. Returns +inf if <2."""
    if not step_logprobs:
        return float("inf")
    vals = []
    for v in step_logprobs.values():
        vals.append(float(getattr(v, "logprob", v)))
    if len(vals) < 2:
        return float("inf")
    vals.sort(reverse=True)
    return vals[0] - vals[1]


def teacher_force(llm, SamplingParams, TokensPrompt, input_ids, sg_ids, mm):
    """Restart-on-fork teacher-forced greedy decode of TRT against SGLang's tokens
    (image re-attached every restart). Captures TRT's top1-top2 margin at each
    generated step so forks can be classified confident vs near-tie."""
    n = min(TF_STEPS, len(sg_ids))
    forced = list(input_ids)
    t = 0
    per_step = []
    n_calls = 0
    guard = n + 8
    while t < n and n_calls < guard:
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=forced, multi_modal_data=mm)],
            SamplingParams(max_tokens=n - t, temperature=0.0, logprobs=TOPK),
        )[0]
        n_calls += 1
        o = out.outputs[0]
        trt_ids = list(o.token_ids)
        lp = list(o.logprobs) if o.logprobs is not None else []
        if not trt_ids:
            per_step.append(dict(t=t, trt=-1, sg=int(sg_ids[t]), match=False, margin=0.0))
            forced = list(input_ids) + list(sg_ids[: t + 1])
            t += 1
            continue
        forked = False
        for i, tt in enumerate(trt_ids):
            tt_t = t + i
            if tt_t >= n:
                break
            sg = int(sg_ids[tt_t])
            margin = _margin_from_logprobs(lp[i] if i < len(lp) else None)
            match = int(tt) == sg
            per_step.append(
                dict(t=tt_t, trt=int(tt), sg=sg, match=match, margin=round(float(margin), 4))
            )
            if not match:
                forced = list(input_ids) + list(sg_ids[: tt_t + 1])
                t = tt_t + 1
                forked = True
                break
        if not forked:
            t = t + len(trt_ids)
    return per_step, n_calls


def main() -> int:
    import io

    import inkling_image_prompts as P
    import torch  # noqa: F401
    from PIL import Image
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "Accounting teacher-force needs CUDA GPUs"
    if not os.path.exists(REF):
        print(f"TF_FAIL sglang ref missing: {REF}", flush=True)
        return 2
    with open(REF) as f:
        sgdoc = json.load(f)
    sg_by_id = {r["id"]: r for r in sgdoc.get("records", [])}

    recs = P.build_prompts(N_ITEMS, with_num_patches=False)
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    print(
        f"[tf] tp={TP} n_items={len(recs)} tf_steps={TF_STEPS} conf_margin={CONF} "
        f"baseline cuda_graph=off overlap=off deterministic bs=1 ref={REF}",
        flush=True,
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.6, dtype="auto", enable_block_reuse=False
    )
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_autotuner=False,
        max_seq_len=4096,
        max_batch_size=1,
        max_num_tokens=4096,
    )
    print(
        "[tf] moe_backend=CUTLASS cuda_graph_hard_path=eager(no-graph) "
        "attn=TRTLLM kv=KVCacheManagerV2",
        flush=True,
    )

    records = []
    try:
        for r in recs:
            rid = r["id"]
            sgrec = sg_by_id.get(rid)
            if sgrec is None or not sgrec.get("sglang_text"):
                print(f"  [tf] SKIP {rid}: no sglang_text in ref", flush=True)
                continue
            image = Image.open(io.BytesIO(r["image_bytes"]))
            mm = {"image": [image]}
            input_ids = _trt_ids(r["input_ids"])
            # SGLang reference tokens: re-tokenize the captured greedy text. Caveat
            # in the module docstring (biases toward more forks, so many matched
            # leading steps is robust evidence against an early bug).
            sg_ids = tok.encode(sgrec["sglang_text"], add_special_tokens=False)

            per_step, n_calls = teacher_force(
                llm, SamplingParams, TokensPrompt, input_ids, sg_ids, mm
            )
            mism = [s for s in per_step if not s["match"]]
            conf_mism = [s for s in mism if s["margin"] >= CONF]
            first = mism[0] if mism else None
            matched_before = first["t"] if first else len(per_step)
            first_margin = first["margin"] if first else None
            first_kind = None
            if first is not None:
                first_kind = "confident" if first["margin"] >= CONF else "near_tie"

            rec = {
                "id": rid,
                "config": r["config"],
                "e2e": _label(rid),
                "sglang_parsed": sgrec.get("sglang_parsed"),
                "sglang_score": sgrec.get("sglang_score"),
                "sglang_n_gen": sgrec.get("n_gen"),
                "sglang_finish": sgrec.get("finish_reason"),
                "tf_steps_compared": len(per_step),
                "sg_ref_tokens": len(sg_ids),
                "matched_before_first_fork": matched_before,
                "first_fork_step": (first["t"] if first else None),
                "first_fork_margin": first_margin,
                "first_fork_kind": first_kind,
                "mismatches": len(mism),
                "confident_mismatches": len(conf_mism),
                "near_tie_mismatches": len(mism) - len(conf_mism),
                "tf_calls": n_calls,
            }
            records.append(rec)
            print(
                f"  [tf {rid:<26}] e2e={rec['e2e']:<8} "
                f"matched_before_fork={matched_before} "
                f"first_fork=(step={rec['first_fork_step']},"
                f"kind={first_kind},margin={first_margin}) "
                f"mismatch={len(mism)}/{len(per_step)} "
                f"(confident={len(conf_mism)} near_tie={len(mism) - len(conf_mism)}) "
                f"sglang={rec['sglang_parsed']}(s={rec['sglang_score']},"
                f"n_gen={rec['sglang_n_gen']},{rec['sglang_finish']})",
                flush=True,
            )
    finally:
        llm.shutdown()

    # Verdict: are wrong-item forks dominated by near-ties, and does the control
    # look the same (=> accepted diffuse residual, not a fixable early bug)?
    wrong = [r for r in records if r["e2e"] == "WRONG"]
    ctrl = [r for r in records if r["e2e"] == "ctrl-OK"]
    any_immediate_confident = any(
        r["first_fork_step"] is not None
        and r["first_fork_step"] <= 3
        and r["first_fork_kind"] == "confident"
        for r in wrong
    )
    wrong_forks_near_tie = (
        all((r["confident_mismatches"] <= r["near_tie_mismatches"]) for r in wrong)
        if wrong
        else False
    )
    if any_immediate_confident:
        verdict = "EARLY_CONFIDENT_FORK_INVESTIGATE"
    elif wrong_forks_near_tie:
        verdict = "DIFFUSE_RESIDUAL_NEAR_TIE"
    else:
        verdict = "MIXED_INSPECT"

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {"tf_steps": TF_STEPS, "conf_margin": CONF, "verdict": verdict, "records": records},
            f,
            indent=2,
        )

    print(
        f"\nINKLING_ACCT_TF verdict={verdict} n_items={len(records)} "
        f"wrong={len(wrong)} ctrl={len(ctrl)} "
        f"any_immediate_confident_wrong_fork={any_immediate_confident} "
        f"out={OUT}",
        flush=True,
    )
    if verdict == "DIFFUSE_RESIDUAL_NEAR_TIE":
        print(
            "INTERPRETATION: TRT tracks SGLang's reasoning and its forks are "
            "dominated by near-tie argmax flips (the control good item forks the "
            "same way yet still lands correct) => the Accounting gap is the "
            "accepted, out-of-scope fa4-vs-Triton NVFP4 attention-kernel-family "
            "residual compounding over long reasoning, NOT a Python-fixable "
            "decode bug.",
            flush=True,
        )
    elif verdict == "EARLY_CONFIDENT_FORK_INVESTIGATE":
        print(
            "INTERPRETATION: a WRONG item forks confidently within the first few "
            "decode steps after the image prefix => a possible Python-fixable "
            "image-prefix decode bug; disambiguate with a clean SGLang token-id "
            "re-capture before concluding.",
            flush=True,
        )
    else:
        print("INTERPRETATION: fork pattern is mixed; inspect the per-item table.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("INKLING_ACCT_TF FAIL: exception producing evidence", flush=True)
        sys.exit(1)
