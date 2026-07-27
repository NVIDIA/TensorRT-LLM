#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.5 DECODE-SIDE mechanism probe for the MMMU Accounting gap.

``reference_tier=real_source``, ``validation_tier=real_runtime``.

Vision is already proven bitwise-clean for the failing Accounting images
(preprocess + tower, jobs 5588462/5588190 verdict=VISION_CLEAN, max_abs=0.0
cos=1.0), so the Accounting accuracy gap is DECODE-SIDE. The offline paired
analysis of the deterministic bs=1 run (job 5586765) showed a specific,
reproducible signature the previous "diffuse residual" framing did not explain:
on the items where SGLang is right and TRT is wrong, **SGLang terminates early
(finish=stop, n_gen 609-2247) and lands correct while TRT runs to the 2560 cap
(finish=length) and lands wrong** (e.g. Accounting_10 SG stop@609 correct vs TRT
2560 wrong). Correct items also hit the cap, so termination is not the whole
story -- but the harness scorer (``parse_multi_choice_response``) takes the
**LAST** ``Answer: X`` occurrence, so over-generation can flip a correct answer
to a wrong one if TRT reaches the right answer early and then revises.

This probe captures the MISSING evidence that the 5586765 crash lost (the run
never persisted per-item text) so the mechanism can be NAMED rather than
guessed, exactly as the human-feedback input-scale playbook requires ("do not
stop before you have a named location"):

  PART A -- full-generation capture (robust, no logprob/context-logit API risk).
    Generate TRT's full answer for each discriminating item (deterministic bs=1,
    baseline cuda_graph=off/overlap=off) and record, per item:
      * full decoded text + n_gen + finish (eos 200006 emitted? -> stop vs length)
      * the COMPLETE ``Answer: X`` trajectory (every occurrence, char position,
        letter, correct-vs-gold) -- so we can see whether TRT reaches the CORRECT
        answer early and then REVISES to a wrong one (over-generation causal) or
        never reaches it (genuine reasoning-quality divergence).
      * three parses of the SAME generation under the SAME shared scorer:
          - ``parsed_full``    : scorer on the full text (what the runner scores;
                                 last-Answer over thinking+answer).
          - ``parsed_visible`` : scorer on only the last ``<|end_message|>``
                                 (200010)-delimited visible segment (the
                                 ``<|content_text|>`` answer a reasoning parser
                                 would isolate).
          - ``parsed_first``   : the FIRST ``Answer: X`` letter (what an
                                 answer-commit / stop-on-first-answer would score).
        Comparing these three localizes whether the wrongness lives in the
        over-generated tail (parsed_full wrong but parsed_visible/parsed_first
        right => a termination/scoring-window effect) or in the reasoning itself
        (all three wrong => genuine divergence).

  PART B -- single-pass teacher-force localization (best-effort; wrapped so a
    prompt_logprobs/context-logit limitation on the multimodal path cannot lose
    Part A). ONE forward pass per item over ``input_ids + sglang_reasoning_ids``
    with ``prompt_logprobs`` (NO slow restart-on-fork loop that stalled job
    5588775): read TRT's per-position greedy argmax vs SGLang's next token, and
    report matched fraction, first-fork step/margin, and confident-vs-near-tie
    mismatch counts. If TRT tracks SGLang for many steps with near-tie forks and
    the CONTROL-good item forks the same way, the divergence is the accepted,
    out-of-scope fa4-vs-Triton NVFP4 attention-kernel-family residual; an
    early-confident fork on a WRONG-but-not-control item would instead point to a
    Python-fixable decode bug.

Discriminating set (control-good + E2E-wrong Accounting, E2E labels from 5586765):
  control-good (TRT correct): Accounting:0 (=validation_Accounting_1),
                              Accounting:7 (=validation_Accounting_8)
  E2E-wrong (SGLANG_BETTER)  : Accounting:1,2,5,8,9
                              (=validation_Accounting_2,3,6,9,10)

Baseline only: cuda_graph=off, overlap=off, TP=4, KVCacheManagerV2, TRTLLM attn,
CUTLASS MoE, deterministic (enable_autotuner=False, bs=1). Config matches the
authoritative M1c run (job 5586765) EXACTLY -- max_seq_len=8192,
free_gpu_memory_fraction=0.7 -- so the decode is faithful to the gap being
characterized; running only 7 items keeps it under the ~15-item cumulative
Bus-error-under-memory-pressure threshold, and JSON is written INCREMENTALLY
after every item so a crash keeps prior evidence.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_accounting_decode_probe.py
Env: INKLING_CHECKPOINT, INKLING_MMMU_REF (sglang_mmmu_ref.json), MMMU_ALIGN_ITEMS,
     INKLING_MMMU_N, INKLING_PROBE_OUT, INKLING_MAXTOK (default 2560),
     INKLING_TF_CAP (default 1536), INKLING_TF_TOPK (default 8),
     INKLING_TF_CONF (default 1.5), INKLING_MOE_BACKEND (default CUTLASS).
"""

import json
import os
import re
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
    "INKLING_PROBE_OUT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/acct_decode_probe.json",
)
N_ITEMS = int(os.environ.get("INKLING_MMMU_N", "7"))
MAXTOK = int(os.environ.get("INKLING_MAXTOK", "2560"))
# Runtime footprint knobs (env-overridable so a memory-safe rerun does not need a
# code edit). Defaults match the authoritative M1c run (job 5586765) so the
# decode stays faithful; the sbatch lowers them only for memory-safety.
FREE_GPU_FRAC = float(os.environ.get("INKLING_FREE_GPU_FRAC", "0.7"))
MAX_SEQ_LEN = int(os.environ.get("INKLING_MAX_SEQ_LEN", "8192"))
MAX_NUM_TOKENS = int(os.environ.get("INKLING_MAX_NUM_TOKENS", "8192"))
TF_CAP = int(os.environ.get("INKLING_TF_CAP", "1536"))
TOPK = int(os.environ.get("INKLING_TF_TOPK", "8"))
CONF = float(os.environ.get("INKLING_TF_CONF", "1.5"))  # top1-top2 logprob margin
TP = int(os.environ.get("INKLING_TP", "4"))

# Inkling protocol framing (tokenizer_config added_tokens_decoder).
EOS_ID = 200006  # <|content_model_end_sampling|>  (generation_config eos)
END_MESSAGE_ID = 200010  # <|end_message|>  (ends a thinking / text segment)
TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101

# E2E labels from the deterministic baseline run 5586765 (validation ids).
E2E_CORRECT = {"validation_Accounting_1", "validation_Accounting_8"}
E2E_WRONG = {
    "validation_Accounting_2",
    "validation_Accounting_3",
    "validation_Accounting_6",
    "validation_Accounting_9",
    "validation_Accounting_10",
}

ANSWER_RE = re.compile(r"[Aa]nswer\s*:\s*\*?\*?\s*\(?([A-Z])\)?")

# Image-blindness marker: the crisp, config-robust signal for the iter22
# embed_norm fusion fix. PRE-FIX (job 5589948), on ALL 7 Accounting items TRT's
# reasoning literally said the image was "not visible" and fell back to guessing
# from the options ("image content not visible. Must infer from options: ..."),
# while SGLang -- SAME NVFP4 checkpoint, SAME image -- read the real table values.
# Whether the decoder can READ the image is a property of the fusion (was the
# fused stream wrongly re-normed?), NOT of KV size / max_tokens, so this signal is
# robust to the memory-safety config knobs. POST-FIX expectation: no item is
# blind (n_all_blind == 0), overturning the pre-fix 7/7 blindness.
# Anchored on the IMAGE being unreadable (not a bare negation), so a post-fix
# generation that merely says "the tax rate is not provided in the table" is NOT
# counted blind. Validated offline against the artifacts: 7/7 pre-fix TRT items
# (job 5589948) match with hits "image not visible" / "image content not visible",
# and 0/12 SGLang reference items (which read the image) match -> clean separation.
IMAGE_BLIND_RE = re.compile(
    r"image[^.\n]{0,45}?\b(?:not|isn'?t|is not|n't)\b[^.\n]{0,25}?"
    r"(?:visible|shown|provided|available|clear|legible|readable|display)"
    r"|(?:not able|unable|can'?t|cannot|could ?n'?t|couldn'?t|do ?n'?t|don'?t)"
    r"[^.\n]{0,30}?(?:see|view|read|make out|access)[^.\n]{0,30}?"
    r"(?:image|picture|figure|chart|table|diagram|graph|photo)"
    r"|no image (?:is |was |provided|available|attached|given)"
    r"|(?:image|picture) (?:is |was |)(?:not |un)(?:visible|available|provided|shown)"
    r"|(?:image|picture)[^.\n]{0,25}?not (?:visible|provided|shown|available)"
    r"|content[^.\n]{0,10}not visible",
    re.IGNORECASE,
)


def _image_blind(text):
    """Return (blind, phrase): True if the generation claims it cannot read the
    image (the pre-fix multimodal-corruption signature)."""
    m = IMAGE_BLIND_RE.search(text or "")
    return (bool(m), (m.group(0) if m else None))


def _label(item_id):
    if item_id in E2E_WRONG:
        return "WRONG"
    if item_id in E2E_CORRECT:
        return "ctrl-OK"
    return "other"


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _answer_trajectory(text, gold, all_choices):
    """Every ``Answer: X`` occurrence in order: (char_pos, letter, correct)."""
    traj = []
    for m in ANSWER_RE.finditer(text):
        letter = m.group(1)
        if all_choices and letter not in all_choices:
            continue
        traj.append(
            {
                "pos": m.start(),
                "letter": letter,
                "correct": bool(gold is not None and letter == gold),
            }
        )
    return traj


def _visible_segment_text(tok, gen_ids):
    """Decode only the LAST non-empty ``<|end_message|>`` (200010)-delimited
    segment before EOS -- the ``<|content_text|>`` visible answer a reasoning
    parser isolates from the thinking. Falls back to the full decode if the
    structured segments are absent."""
    # Drop a trailing EOS so the final segment is the visible answer, not empty.
    ids = [int(t) for t in gen_ids]
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    segs, cur = [], []
    for t in ids:
        if t == END_MESSAGE_ID:
            segs.append(cur)
            cur = []
        else:
            cur.append(t)
    segs.append(cur)
    for seg in reversed(segs):
        txt = tok.decode(seg, skip_special_tokens=True).strip()
        if txt:
            return txt
    return tok.decode(ids, skip_special_tokens=True)


def _margin_from_entry(entry):
    """top1-top2 logprob margin from a prompt_logprobs dict {tid: Logprob|float}
    and the rank-1 (argmax) token id."""
    if not entry:
        return float("inf"), None
    vals = []
    for tid, v in entry.items():
        lp = float(getattr(v, "logprob", v))
        rank = getattr(v, "rank", None)
        vals.append((lp, int(tid), rank))
    # argmax = rank==1 if ranks present, else max logprob.
    arg = None
    for lp, tid, rank in vals:
        if rank == 1:
            arg = tid
            break
    if arg is None:
        arg = max(vals)[1]
    lps = sorted((lp for lp, _, _ in vals), reverse=True)
    margin = (lps[0] - lps[1]) if len(lps) >= 2 else float("inf")
    return margin, arg


def teacher_force_single_pass(llm, SamplingParams, TokensPrompt, input_ids, sg_ids, mm):
    """ONE forward pass: prompt = input_ids + sg_ids[:CAP], prompt_logprobs=TOPK.
    Returns (per_step, err). per_step[j] compares TRT's greedy argmax at reasoning
    position j to sg_ids[j+1]."""
    cap = min(TF_CAP, len(sg_ids))
    if cap < 2:
        return [], "sg_ids too short"
    forced = list(input_ids) + list(sg_ids[:cap])
    try:
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=forced, multi_modal_data=mm)],
            SamplingParams(
                max_tokens=1, temperature=0.0, prompt_logprobs=TOPK, return_context_logits=True
            ),
        )[0]
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"
    plp = out.outputs[0].prompt_logprobs
    if not plp:
        return [], "no prompt_logprobs returned"
    # plp is aligned to the EXPANDED prompt: len(plp) == expanded_seq_len, and
    # plp[k] predicts expanded position k+1. The single image placeholder in
    # input_ids expands to num_patches rows at runtime, so DO NOT index from
    # len(input_ids) (that ignores the expansion). sg_ids[:cap] are the LAST
    # `cap` tokens of the expanded prompt regardless of the mid-prompt image
    # expansion, so anchor from the END: sg_ids[j] sits at expanded pos base+j
    # with base = len(plp) - cap, and its next-token prediction is plp[base+j].
    base = len(plp) - cap
    if base < 0:
        return [], f"prompt_logprobs len {len(plp)} < cap {cap}"
    per_step = []
    for j in range(cap - 1):
        k = base + j
        if k >= len(plp):
            break
        margin, arg = _margin_from_entry(plp[k])
        nxt = int(sg_ids[j + 1])
        match = arg is not None and int(arg) == nxt
        per_step.append(
            {
                "t": j,
                "trt": (int(arg) if arg is not None else -1),
                "sg": nxt,
                "match": match,
                "margin": round(float(margin), 4),
            }
        )
    return per_step, None


def _persist(out_path, doc):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, out_path)


def main() -> int:
    import io

    import torch  # noqa: F401

    # Memory-safety (Reviewer iter22 REJECT of job 5591292): the multimodal image
    # tensor is shared to the TP=4 MGMN worker processes via torch.multiprocessing.
    # The default ``file_descriptor`` strategy backs those shared CPU tensors with
    # ``/dev/shm``; on a node whose container ``/dev/shm`` is small the worker
    # mmap SIGBUSes at the first image request ("Bus error (7)" in mgmn_worker,
    # ranks 0-3, job 5591292). The ``file_system`` strategy
    # backs them with regular temp files (node-local /tmp / TMPDIR) instead of
    # ``/dev/shm``, sidestepping the tmpfs exhaustion. Set process-globally on
    # rank 0 BEFORE the executor spawns the workers so the image broadcast uses it.
    import torch.multiprocessing as _mp
    from PIL import Image

    try:
        _mp.set_sharing_strategy("file_system")
        print(f"[probe] torch mp sharing strategy = {_mp.get_sharing_strategy()}", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[probe] WARN could not set file_system sharing strategy: {_e}", flush=True)
    import inkling_image_prompts as P
    import inkling_mmmu_harness as H
    import inkling_mmmu_real_align_test as R
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "Accounting decode probe needs CUDA GPUs"
    sg_by_id = {}
    if os.path.exists(REF):
        with open(REF) as f:
            sg_by_id = {r["id"]: r for r in json.load(f).get("records", [])}
    else:
        print(f"WARN sglang ref missing (Part B disabled): {REF}", flush=True)

    recs = P.build_prompts(N_ITEMS, with_num_patches=False)
    items_by_id = {it["id"]: it for it in R.load_fixed_items()}
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    print(
        f"[probe] tp={TP} n_items={len(recs)} maxtok={MAXTOK} tf_cap={TF_CAP} "
        f"moe={moe_backend} free_gpu={FREE_GPU_FRAC} max_seq_len={MAX_SEQ_LEN} "
        f"max_num_tokens={MAX_NUM_TOKENS} baseline cuda_graph=off overlap=off "
        f"deterministic bs=1 ref={'yes' if sg_by_id else 'no'}",
        flush=True,
    )

    # Match the authoritative M1c run (job 5586765) config EXACTLY so the decode
    # behavior is faithful to the gap being characterized: max_seq_len=8192,
    # free_gpu_memory_fraction=0.7. (A tighter max_seq_len=4096 config was tried
    # in job 5589589 and changed the decode -- both control items ran away with
    # n_ans=0 -- so it conflated a config effect with the residual; do not use
    # it.) 7 items stays under the ~15-item cumulative Bus-error threshold, and
    # the incremental JSON persist protects partial results if it still trips.
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=FREE_GPU_FRAC, dtype="auto", enable_block_reuse=False
    )
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_autotuner=False,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=1,
        max_num_tokens=MAX_NUM_TOKENS,
    )
    print(
        "[probe] moe=CUTLASS cuda_graph_hard_path=eager(no-graph) attn=TRTLLM kv=KVCacheManagerV2",
        flush=True,
    )

    records = []
    doc = {
        "maxtok": MAXTOK,
        "tf_cap": TF_CAP,
        "conf_margin": CONF,
        "moe_backend": moe_backend,
        "records": records,
    }
    try:
        for r in recs:
            rid = r["id"]
            it = items_by_id[rid]
            gold = it.get("answer")
            options = it.get("options")
            qtype = it.get("question_type") or ("multiple-choice" if options else "open")
            if options:
                index2ans, all_choices = H.build_mc_mapping(options)
            else:
                index2ans, all_choices = None, None
            image = Image.open(io.BytesIO(r["image_bytes"]))
            mm = {"image": [image]}
            input_ids = _trt_ids(r["input_ids"])

            # ---- PART A: full generation + Answer trajectory ----------------
            out = llm.generate(
                [TokensPrompt(prompt_token_ids=list(input_ids), multi_modal_data=mm)],
                SamplingParams(max_tokens=MAXTOK, temperature=0.0),
            )[0]
            gen_ids = [int(x) for x in out.outputs[0].token_ids]
            full_text = tok.decode(gen_ids, skip_special_tokens=False)
            has_eos = EOS_ID in gen_ids
            n_gen = len(gen_ids)
            finish = "stop" if has_eos else ("length" if n_gen >= MAXTOK else "stop")
            score_full, parsed_full = H.score_sample(full_text, gold, qtype, all_choices, index2ans)
            vis_text = _visible_segment_text(tok, gen_ids)
            score_vis, parsed_vis = H.score_sample(vis_text, gold, qtype, all_choices, index2ans)
            # Primary fix signal: does the decoder READ the image or claim it is
            # not visible? (pre-fix job 5589948 was blind on all 7 items.)
            blind, blind_phrase = _image_blind(full_text)
            traj = _answer_trajectory(full_text, gold, all_choices)
            first_ans = traj[0] if traj else None
            last_ans = traj[-1] if traj else None
            # over-generation causal := reaches correct answer early but the
            # scored (last) answer is wrong.
            revised_away = bool(
                first_ans and last_ans and first_ans["correct"] and not last_ans["correct"]
            )
            any_correct_in_traj = any(a["correct"] for a in traj)

            sgrec = sg_by_id.get(rid, {})
            rec = {
                "id": rid,
                "config": r["config"],
                "gold": gold,
                "e2e": _label(rid),
                "trt_n_gen": n_gen,
                "trt_finish": finish,
                "trt_has_eos": has_eos,
                "n_end_message": sum(1 for t in gen_ids if t == END_MESSAGE_ID),
                "trt_parsed_full": parsed_full,
                "trt_score_full": score_full,
                "trt_parsed_visible": parsed_vis,
                "trt_score_visible": score_vis,
                "trt_image_blind": blind,
                "trt_blind_phrase": blind_phrase,
                "n_answer_occurrences": len(traj),
                "answer_trajectory": [
                    {"pos": a["pos"], "letter": a["letter"], "correct": a["correct"]} for a in traj
                ],
                "first_answer": (first_ans["letter"] if first_ans else None),
                "first_answer_correct": bool(first_ans and first_ans["correct"]),
                "last_answer": (last_ans["letter"] if last_ans else None),
                "revised_away_from_gold": revised_away,
                "any_correct_answer_in_trajectory": any_correct_in_traj,
                "sglang_parsed": sgrec.get("sglang_parsed"),
                "sglang_score": sgrec.get("sglang_score"),
                "sglang_n_gen": sgrec.get("n_gen"),
                "sglang_finish": sgrec.get("finish_reason"),
                # truncated text for offline inspection (keep JSON bounded).
                "trt_text_head": full_text[:1200],
                "trt_text_tail": full_text[-1200:],
                "trt_visible_text": vis_text[-800:],
            }

            # ---- PART B: single-pass teacher-force (best-effort) ------------
            tf_summary = {"ran": False}
            if sgrec.get("sglang_text"):
                sg_ids = tok.encode(sgrec["sglang_text"], add_special_tokens=False)
                per_step, tf_err = teacher_force_single_pass(
                    llm, SamplingParams, TokensPrompt, input_ids, sg_ids, mm
                )
                if tf_err:
                    tf_summary = {"ran": False, "error": tf_err}
                else:
                    mism = [s for s in per_step if not s["match"]]
                    conf_m = [s for s in mism if s["margin"] >= CONF]
                    first = mism[0] if mism else None
                    tf_summary = {
                        "ran": True,
                        "steps": len(per_step),
                        "matched_fraction": round(1.0 - len(mism) / max(1, len(per_step)), 4),
                        "matched_before_first_fork": (first["t"] if first else len(per_step)),
                        "first_fork_step": (first["t"] if first else None),
                        "first_fork_margin": (first["margin"] if first else None),
                        "first_fork_kind": (
                            None
                            if first is None
                            else ("confident" if first["margin"] >= CONF else "near_tie")
                        ),
                        "mismatches": len(mism),
                        "confident_mismatches": len(conf_m),
                        "near_tie_mismatches": len(mism) - len(conf_m),
                    }
            rec["teacher_force"] = tf_summary
            records.append(rec)
            _persist(OUT, doc)  # incremental: survive a later-item crash

            tf = tf_summary
            print(
                f"  [{rid:<24} {rec['e2e']:<7}] gold={gold} "
                f"TRT_full={parsed_full}(s={score_full:.0f}) "
                f"TRT_vis={parsed_vis}(s={score_vis:.0f}) "
                f"img_blind={blind} "
                f"first_ans={rec['first_answer']}"
                f"({'ok' if rec['first_answer_correct'] else 'x'}) "
                f"n_ans={len(traj)} revised_away={revised_away} "
                f"n_gen={n_gen}/{finish} eos={has_eos} | "
                f"SG={sgrec.get('sglang_parsed')}"
                f"(s={sgrec.get('sglang_score')},n={sgrec.get('n_gen')},"
                f"{sgrec.get('finish_reason')}) | "
                f"TF matched={tf.get('matched_fraction')} "
                f"first_fork=(step={tf.get('first_fork_step')},"
                f"{tf.get('first_fork_kind')}) "
                f"conf/near={tf.get('confident_mismatches')}/"
                f"{tf.get('near_tie_mismatches')}"
                f"{' TFERR=' + tf.get('error', '') if not tf.get('ran') else ''}",
                flush=True,
            )
    finally:
        llm.shutdown()

    # ---- Verdict -----------------------------------------------------------
    wrong = [r for r in records if r["e2e"] == "WRONG"]
    ctrl = [r for r in records if r["e2e"] == "ctrl-OK"]
    # (0) PRIMARY fix signal: image-blindness. Pre-fix (5589948) every item was
    #     blind ("image not visible"); post-fix the decoder must read the image.
    n_all_blind = sum(1 for r in records if r.get("trt_image_blind"))
    n_wrong_blind = sum(1 for r in wrong if r.get("trt_image_blind"))
    n_ctrl_blind = sum(1 for r in ctrl if r.get("trt_image_blind"))
    n_wrong_now_correct = sum(1 for r in wrong if r.get("trt_score_full") == 1.0)
    # (1) over-generation causal: wrong items reach the correct answer early then
    #     revise away (parsed_full wrong) -- i.e. the residual manifests via
    #     failure-to-terminate + last-Answer scoring, not garbage reasoning.
    n_wrong_revised = sum(1 for r in wrong if r["revised_away_from_gold"])
    n_wrong_visible_recovers = sum(
        1 for r in wrong if r["trt_score_full"] == 0.0 and r["trt_score_visible"] == 1.0
    )
    n_wrong_never_correct = sum(1 for r in wrong if not r["any_correct_answer_in_trajectory"])
    # (2) teacher-force: does TRT track SGLang with near-tie forks (residual) or
    #     fork early+confident on wrong-but-not-control (fixable bug)?
    tf_ran = [r for r in records if r["teacher_force"].get("ran")]
    early_conf_wrong = [
        r
        for r in wrong
        if r["teacher_force"].get("ran")
        and r["teacher_force"].get("first_fork_step") is not None
        and r["teacher_force"]["first_fork_step"] <= 3
        and r["teacher_force"].get("first_fork_kind") == "confident"
    ]
    early_conf_ctrl = [
        r
        for r in ctrl
        if r["teacher_force"].get("ran")
        and r["teacher_force"].get("first_fork_step") is not None
        and r["teacher_force"]["first_fork_step"] <= 3
        and r["teacher_force"].get("first_fork_kind") == "confident"
    ]

    if wrong and n_wrong_never_correct == len(wrong):
        verdict = "GENUINE_REASONING_DIVERGENCE"
    elif (
        wrong
        and (n_wrong_revised + n_wrong_visible_recovers) >= 1
        and n_wrong_visible_recovers >= n_wrong_never_correct
    ):
        verdict = "OVERGENERATION_SCORING_WINDOW"
    elif early_conf_wrong and not early_conf_ctrl:
        verdict = "EARLY_CONFIDENT_FORK_INVESTIGATE"
    else:
        verdict = "MIXED_INSPECT"

    doc["verdict"] = verdict
    # POST-FIX image-read verdict (the iter22 embed_norm fusion fix): the fix is
    # confirmed at runtime when the decoder no longer claims the image is invisible.
    image_read_verdict = (
        "IMAGE_FUSION_FIX_CONFIRMED"
        if n_all_blind == 0
        else ("IMAGE_STILL_BLIND" if n_all_blind == len(records) else "IMAGE_PARTIALLY_BLIND")
    )
    doc["image_read_verdict"] = image_read_verdict
    doc["summary"] = {
        "n_wrong": len(wrong),
        "n_ctrl": len(ctrl),
        "n_all_image_blind": n_all_blind,
        "n_wrong_image_blind": n_wrong_blind,
        "n_ctrl_image_blind": n_ctrl_blind,
        "n_wrong_now_correct": n_wrong_now_correct,
        "prefix_baseline_blind": "7/7 (job 5589948)",
        "n_wrong_revised_away": n_wrong_revised,
        "n_wrong_visible_recovers": n_wrong_visible_recovers,
        "n_wrong_never_correct": n_wrong_never_correct,
        "tf_ran": len(tf_ran),
        "early_confident_fork_wrong": len(early_conf_wrong),
        "early_confident_fork_ctrl": len(early_conf_ctrl),
    }
    _persist(OUT, doc)

    print(
        f"\nINKLING_IMAGE_READ {image_read_verdict} "
        f"blind_all={n_all_blind}/{len(records)} "
        f"blind_wrong={n_wrong_blind}/{len(wrong)} "
        f"blind_ctrl={n_ctrl_blind}/{len(ctrl)} "
        f"wrong_now_correct={n_wrong_now_correct}/{len(wrong)} "
        f"(pre-fix baseline job 5589948 was blind_all=7/7)",
        flush=True,
    )
    print(
        f"INKLING_ACCT_PROBE verdict={verdict} n_items={len(records)} "
        f"wrong={len(wrong)} ctrl={len(ctrl)} "
        f"revised_away={n_wrong_revised} visible_recovers={n_wrong_visible_recovers} "
        f"never_correct={n_wrong_never_correct} tf_ran={len(tf_ran)} "
        f"early_conf_wrong={len(early_conf_wrong)} out={OUT}",
        flush=True,
    )
    if verdict == "OVERGENERATION_SCORING_WINDOW":
        print(
            "INTERPRETATION: on the E2E-wrong Accounting items TRT reaches the "
            "CORRECT answer during reasoning but over-generates past SGLang's "
            "stop point and the LAST-Answer scorer then reads a revised wrong "
            "letter; scoring the visible answer segment recovers it. The gap is "
            "a decode-side termination + scoring-window effect, not garbage "
            "reasoning -- a candidate IN-SCOPE mitigation exists.",
            flush=True,
        )
    elif verdict == "GENUINE_REASONING_DIVERGENCE":
        print(
            "INTERPRETATION: the E2E-wrong Accounting items never reach the "
            "correct answer at all under TRT decode; the reasoning itself "
            "diverges. Combined with a near-tie teacher-force fork pattern this "
            "is the accepted, out-of-scope fa4-vs-Triton NVFP4 "
            "attention-kernel-family residual compounding over long reasoning.",
            flush=True,
        )
    elif verdict == "EARLY_CONFIDENT_FORK_INVESTIGATE":
        print(
            "INTERPRETATION: a WRONG-but-not-control item forks confidently "
            "within the first few teacher-forced steps => a possible "
            "Python-fixable decode bug to localize before any BLOCKER.",
            flush=True,
        )
    else:
        print("INTERPRETATION: mixed signals; inspect the per-item table + JSON.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("INKLING_ACCT_PROBE FAIL: exception producing evidence", flush=True)
        sys.exit(1)
