#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""HUMAN FEEDBACK #11 / #131 -- TERMINATION teacher-forced probe (TRT side).

Item-level localization (gap_localization_itemlevel.py) put the whole -0.0355 MMMU
gap in the both-truncate bucket. The ASYMMETRIC sub-case is 4 SGLANG_BETTER items
where SGLang STOPPED cleanly (right answer) but TRT ran to the 2560 cap and got it
wrong. feedback #131 "locate with teacher-forced methods": does TRT have a
stop-SUPPRESSION defect, or did it merely fork earlier onto a longer path?

Method (deterministic, teacher-forced, SGLang is the standard):
  For each item, feed TRT the SGLang answer context VERBATIM --
    prompt = _trt_ids(input_ids) + sglang_greedy_ids[:stop_idx]
  where stop_idx is SGLang's first terminal marker (200010 <|end_message|>, else
  200006 <|content_model_end_sampling|>). Then read TRT's next-token distribution
  at that exact position (return_generation_logits) AND let TRT free-generate a
  bounded window. The question is narrow and causal: GIVEN SGLANG'S OWN answer
  context, does TRT want to terminate?
    * TRT argmax == a terminal token, OR TRT free-gen emits one within a few tokens
      -> TRT_TERMINATES: TRT agrees with SGLang's stop; its free-run runaway is an
         EARLIER near-tie fork, not stop suppression. No defect.
    * terminal token in TRT top-K but not argmax -> TRT_NEAR_TIE_STOP: near-tie, no
      confident suppression.
    * terminal token low-rank (outside top-K) AND free-gen runs the full window
      without terminating -> TRT_SUPPRESSES_STOP: a real termination defect to DIVE.

Determinism prereq (mirrors the accepted prefill/decode probes): bs=1, TP=4,
cuda_graph=off, overlap=off, CUTLASS, enable_autotuner=False,
TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1; the logit forward is run TWICE and the terminal
position's logit vector must be byte-identical before any verdict is trusted.

Run:   trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_tf_terminate_probe.py
Self:  python3 tests/unittest/_torch/modeling/inkling_tf_terminate_probe.py --selftest
Env:   INKLING_CHECKPOINT, INKLING_MM_TERMINATE_REF (sglang_mm_terminate_fb11.json),
       INKLING_TFTERM_OUT, INKLING_TFTERM_FREEGEN, INKLING_TFTERM_TOPK.
"""
import hashlib
import json
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/hf_data/hf_home/hub/"
    "models--thinkingmachines--Inkling-NVFP4/snapshots/"
    "95e51a54d9486020a80d49ae4f9103fb2b3f9686",
)
WS = ("/lustre/fsw/coreai_comparch_trtllm/kleinc/codes/agent-flow/"
      "workspace/inkling-advanced-bringup")
REF = os.environ.get("INKLING_MM_TERMINATE_REF",
                     os.path.join(WS, "results/sglang_mm_terminate_fb11.json"))
OUT = os.environ.get("INKLING_TFTERM_OUT",
                     os.path.join(WS, "results/tf_terminate_probe_fb11.json"))
TP = int(os.environ.get("INKLING_TP", "4"))
UNPADDED_VOCAB = int(os.environ.get("INKLING_UNPADDED_VOCAB", "200058"))
MOE_BACKEND = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
FREEGEN = int(os.environ.get("INKLING_TFTERM_FREEGEN", "128"))
TOPK = int(os.environ.get("INKLING_TFTERM_TOPK", "10"))
# "soon" = TRT free-gen terminates within this many tokens of SGLang's stop point.
SOON = int(os.environ.get("INKLING_TFTERM_SOON", "16"))

# checkpoint terminal control tokens (config.json eos_token_id=200006).
END_MESSAGE = 200010          # <|end_message|>
END_SAMPLING = 200006         # <|content_model_end_sampling|>  (eos)
TERMINALS = (END_MESSAGE, END_SAMPLING)

# TRT in-vocab image placeholder vs SGLang's -101 sentinel (same as the other probes)
TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t)
            for t in input_ids]


def find_stop_idx(sg_ids):
    """Locate SGLang's FINAL termination transition (the unambiguous stop test).

    Inkling greedy output is ``[thinking] 200010 [answer-message] 200010 200006``:
    the FIRST 200010 ends the thinking message, the SECOND ends the answer message
    and is immediately followed by 200006 (eos). The stop-suppression question is
    'after SGLang's COMPLETE answer, does TRT terminate?', so the probe position is
    the TERMINAL 200010 (the token right before the last eos) -- teacher-forcing to
    ``sg_ids[:stop_idx]`` feeds TRT the whole thinking+answer and asks for the next
    token, which SGLang made 200010. Falls back to the last eos position if no
    200010 precedes it. Returns (stop_idx, terminal_token)."""
    eos_positions = [i for i, t in enumerate(sg_ids) if int(t) == END_SAMPLING]
    if eos_positions:
        eos_idx = eos_positions[-1]
        if eos_idx >= 1 and int(sg_ids[eos_idx - 1]) == END_MESSAGE:
            return eos_idx - 1, END_MESSAGE          # terminal <|end_message|>
        return eos_idx, END_SAMPLING                 # eos directly
    em = [i for i, t in enumerate(sg_ids) if int(t) == END_MESSAGE]
    if em:
        return em[-1], END_MESSAGE
    return None, None


def find_endthinking_idx(sg_ids):
    """FIRST <|end_message|> = SGLang's end-of-thinking decision (secondary probe:
    does TRT want to conclude reasoning at the same point, or keep running?)."""
    for i, t in enumerate(sg_ids):
        if int(t) == END_MESSAGE:
            return i
    return None


def classify_termination(trt_argmax, endmsg_rank, free_terminate_offset, topk=TOPK,
                         soon=SOON):
    """Pure verdict logic (CPU-testable).

    trt_argmax          : TRT's own next-token argmax at SGLang's stop position.
    endmsg_rank         : 1-based rank of SGLang's terminal token in TRT's logits
                          (None/<=0 -> outside the reported top-K).
    free_terminate_offset: offset (>=0) at which TRT's free-gen first emits a
                          terminal token, or None if it never terminates in-window.
    """
    if trt_argmax in TERMINALS:
        return "TRT_TERMINATES"
    if free_terminate_offset is not None and free_terminate_offset <= soon:
        return "TRT_TERMINATES"
    if endmsg_rank is not None and 0 < endmsg_rank <= topk:
        return "TRT_NEAR_TIE_STOP"
    if free_terminate_offset is None:
        return "TRT_SUPPRESSES_STOP"
    # terminal token low-rank but TRT did terminate later in-window -> late but not
    # a confident suppression.
    return "TRT_NEAR_TIE_STOP"


def decide_overall(per_item):
    if any(r.get("verdict") == "TRT_SUPPRESSES_STOP" for r in per_item):
        return "TERMINATION_DEFECT_CANDIDATE"
    if per_item and all(r.get("verdict") in ("TRT_TERMINATES", "TRT_NEAR_TIE_STOP")
                        for r in per_item):
        return "TERMINATION_CLEAN"
    return "TERMINATION_INCOMPLETE"


def _rank_and_logprob(logits_pos, tok, torch):
    """1-based rank + logprob of `tok` in a full logit vector."""
    lse = torch.logsumexp(logits_pos, dim=0)
    lp = float(logits_pos[tok] - lse)
    rank = int((logits_pos > logits_pos[tok]).sum().item()) + 1
    return rank, lp


def _img(r):
    import base64
    import io

    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))


def _term_logits(llm, SamplingParams, TokensPrompt, ref, tag):
    """Per item: PRIMARY forward teacher-forces to SGLang's complete-answer context
    (terminal 200010) and free-gens a bounded window; SECONDARY forward reads the
    end-of-thinking (first 200010) logit only. Returns per-id logit vector + sha +
    free-gen + the end-of-thinking rank of <|end_message|>."""
    import torch

    sampling = SamplingParams(max_tokens=FREEGEN, temperature=0.0, top_k=1,
                              return_generation_logits=True)
    sampling1 = SamplingParams(max_tokens=1, temperature=0.0, top_k=1,
                               return_generation_logits=True)
    out = {}
    for r in ref:
        sg_ids = [int(t) for t in r["greedy_token_ids"]]
        stop_idx, term_tok = find_stop_idx(sg_ids)
        t1_idx = find_endthinking_idx(sg_ids)
        if stop_idx is None:
            out[r["id"]] = dict(skip="no terminal marker in SGLang seq",
                                n_sg=len(sg_ids), last=sg_ids[-3:])
            print(f"  [{tag}] {r['id']:<32} SKIP no terminal marker (n_sg={len(sg_ids)})",
                  flush=True)
            continue
        # PRIMARY: complete-answer context -> does TRT emit the terminal token?
        forced = _trt_ids(r["input_ids"]) + sg_ids[:stop_idx]
        gen = llm.generate([TokensPrompt(prompt_token_ids=forced,
                                         multi_modal_data={"image": [_img(r)]})],
                           sampling)[0].outputs[0]
        gl = torch.as_tensor(gen.generation_logits).float().cpu()
        if gl.dim() == 1:
            gl = gl.unsqueeze(0)
        eff = min(gl.shape[-1], UNPADDED_VOCAB)
        pos = gl[0, :eff].contiguous()
        sha = hashlib.sha256(pos.numpy().tobytes()).hexdigest()
        free_ids = [int(t) for t in (gen.token_ids or [])]
        # SECONDARY: end-of-thinking context -> does TRT rank <|end_message|> high?
        t1_argmax = t1_rank = None
        if t1_idx is not None and 0 < t1_idx < stop_idx:
            forced1 = _trt_ids(r["input_ids"]) + sg_ids[:t1_idx]
            gen1 = llm.generate([TokensPrompt(prompt_token_ids=forced1,
                                              multi_modal_data={"image": [_img(r)]})],
                                sampling1)[0].outputs[0]
            gl1 = torch.as_tensor(gen1.generation_logits).float().cpu()
            if gl1.dim() == 1:
                gl1 = gl1.unsqueeze(0)
            p1 = gl1[0, :eff].contiguous()
            t1_argmax = int(p1.argmax())
            t1_rank = int((p1 > p1[END_MESSAGE]).sum().item()) + 1
        out[r["id"]] = dict(pos=pos, eff=eff, sha=sha, stop_idx=stop_idx,
                            term_tok=term_tok, free_ids=free_ids,
                            forced_len=len(forced), t1_idx=t1_idx,
                            t1_argmax=t1_argmax, t1_endmsg_rank=t1_rank)
        print(f"  [{tag}] {r['id']:<32} stop_idx={stop_idx} term={term_tok} "
              f"trt_argmax={int(pos.argmax())} sha={sha[:12]} free0={free_ids[:1]} "
              f"t1_endmsg_rank={t1_rank}", flush=True)
    return out


def main() -> int:
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "termination probe needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids") and r.get("greedy_token_ids")
           and r.get("image_b64")]
    assert len(ref) >= 1, f"no usable SGLang termination refs in {REF}"
    print(f"[tfterm] tp={TP} moe={MOE_BACKEND} n={len(ref)} freegen={FREEGEN} "
          f"topk={TOPK} baseline cuda_graph=off overlap=off enable_autotuner=False "
          f"bs=1 allreduce_autotune_disabled="
          f"{os.environ.get('TLLM_DISABLE_ALLREDUCE_AUTOTUNE','0')} "
          f"ids={[r['id'] for r in ref]}", flush=True)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6, dtype="auto",
                                    enable_block_reuse=False, host_cache_size=0)
    llm = LLM(
        CKPT, tensor_parallel_size=TP, trust_remote_code=True, attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=MOE_BACKEND), kv_cache_config=kv_cache_config,
        gather_generation_logits=True, cuda_graph_config=None,
        disable_overlap_scheduler=True, enable_autotuner=False,
        max_seq_len=6144, max_batch_size=1, max_num_tokens=6144,
    )
    print("[tfterm] moe_backend=CUTLASS cuda_graph_hard_path=eager(no-graph) "
          "attn=TRTLLM kv=KVCacheManagerV2 bs=1", flush=True)

    try:
        runA = _term_logits(llm, SamplingParams, TokensPrompt, ref, "runA")
        runB = _term_logits(llm, SamplingParams, TokensPrompt, ref, "runB")
    finally:
        llm.shutdown()

    per_item = []
    det_all = True
    for r in ref:
        rid = r["id"]
        a, b = runA.get(rid, {}), runB.get(rid, {})
        if a.get("skip"):
            per_item.append(dict(id=rid, verdict="TERMINATION_INCOMPLETE",
                                 note=a["skip"]))
            det_all = False
            continue
        det = (a["sha"] == b["sha"])
        det_all = det_all and det
        pos = a["pos"]
        term_tok = a["term_tok"]
        trt_argmax = int(pos.argmax())
        endmsg_rank, endmsg_lp = _rank_and_logprob(pos, term_tok, torch)
        eos_rank, eos_lp = _rank_and_logprob(pos, END_SAMPLING, torch)
        # TRT top-K at the stop position
        tv, ti = torch.topk(pos, k=min(TOPK, pos.shape[0]))
        lse = torch.logsumexp(pos, dim=0)
        trt_topk = [[int(i), round(float(v - lse), 4)] for v, i in zip(tv, ti)]
        # SGLang's own logprob for its terminal token at this step (argmax -> ~0)
        sg_top = (r.get("pos_top") or [])
        sg_term_lp = None
        if a["stop_idx"] < len(sg_top) and sg_top[a["stop_idx"]]:
            for pair in sg_top[a["stop_idx"]]:
                if int(pair[0]) == term_tok:
                    sg_term_lp = round(float(pair[1]), 4)
                    break
        # free-gen termination offset
        free_ids = a["free_ids"]
        free_off = next((k for k, t in enumerate(free_ids) if t in TERMINALS), None)
        verdict = classify_termination(trt_argmax, endmsg_rank, free_off)
        per_item.append(dict(
            id=rid, config=r.get("config"),
            sg_n_gen=len(r["greedy_token_ids"]), stop_idx=a["stop_idx"],
            terminal_token=term_tok, forced_len=a["forced_len"],
            trt_argmax=trt_argmax, trt_argmax_logprob=round(float(
                pos[trt_argmax] - lse), 4),
            trt_terminal_rank=endmsg_rank, trt_terminal_logprob=round(endmsg_lp, 4),
            trt_eos_rank=eos_rank, trt_eos_logprob=round(eos_lp, 4),
            sglang_terminal_logprob=sg_term_lp,
            trt_topk=trt_topk,
            free_gen_terminate_offset=free_off,
            free_gen_head=free_ids[:12],
            end_thinking_idx=a.get("t1_idx"),
            end_thinking_trt_argmax=a.get("t1_argmax"),
            end_thinking_endmsg_rank=a.get("t1_endmsg_rank"),
            deterministic=det, verdict=verdict,
        ))

    overall = decide_overall(per_item)
    doc = dict(
        title=("feedback #11/#131 -- TRT termination teacher-forced probe on the 4 "
               "asymmetric SGLANG_BETTER items (SGLang stops, TRT ran to cap)"),
        reference=REF,
        config=dict(tp=TP, moe_backend=MOE_BACKEND, cuda_graph=False, overlap=False,
                    enable_autotuner=False, bs=1, attn="TRTLLM", kv="KVCacheManagerV2",
                    freegen=FREEGEN, topk=TOPK, terminals=list(TERMINALS),
                    allreduce_autotune_disabled=os.environ.get(
                        "TLLM_DISABLE_ALLREDUCE_AUTOTUNE", "0")),
        method=("teacher-force TRT to SGLang's pre-terminal answer context, read the "
                "stop-position logit + free-gen; SGLang is the standard"),
        determinism_all_byte_identical=det_all,
        overall_verdict=overall,
        interpretation=_interpret(overall, per_item),
        records=per_item,
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"\nINKLING_TFTERM_DET {'PASS' if det_all else 'FAIL'} "
          f"identical={sum(1 for r in per_item if r.get('deterministic'))}/{len(per_item)}")
    for r in per_item:
        print(f"  {r['id']:<32} verdict={r.get('verdict'):<20} "
              f"trt_argmax={r.get('trt_argmax')} term_rank={r.get('trt_terminal_rank')} "
              f"free_term_off={r.get('free_gen_terminate_offset')}")
    print(f"\nINKLING_TFTERM_VERDICT {overall} det={det_all} out={OUT}")
    return 0


def _interpret(overall, per_item):
    if overall == "TERMINATION_CLEAN":
        return ("On every asymmetric item, given SGLang's OWN answer context TRT "
                "either emits a terminal token or ranks it within top-K / terminates "
                "in-window -- TRT has NO stop-suppression defect. The free-run runaway "
                "on these items is an earlier near-tie fork onto a longer reasoning "
                "path, consistent with the item-level truncation-coin-flip verdict.")
    if overall == "TERMINATION_DEFECT_CANDIDATE":
        return ("At least one item shows TRT confidently suppressing the terminal "
                "token on SGLang's own answer context AND running the full free-gen "
                "window without terminating -- a real termination defect. DIVE it.")
    return "Incomplete -- a terminal marker or determinism check was missing."


def _selftest() -> int:
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS {name}")
        else:
            fail += 1
            print(f"  FAIL {name}")

    # find_stop_idx targets the TERMINAL 200010 before the last eos (Inkling shape
    # [thinking] 200010 [answer] 200010 200006); NOT the end-of-thinking 200010.
    seq = [1, END_MESSAGE, 2, 3, END_MESSAGE, END_SAMPLING]  # T1@1, terminal@4, eos@5
    chk("stop_idx = terminal 200010 (not the first)", find_stop_idx(seq) == (4, END_MESSAGE))
    chk("endthinking_idx = first 200010", find_endthinking_idx(seq) == 1)
    chk("stop_idx falls back to eos when no preceding 200010",
        find_stop_idx([1, 2, 3, END_SAMPLING]) == (3, END_SAMPLING))
    chk("stop_idx none when absent", find_stop_idx([1, 2, 3]) == (None, None))
    chk("endthinking none when absent", find_endthinking_idx([1, 2, 3]) is None)

    # classify_termination
    chk("argmax is terminal -> terminates",
        classify_termination(END_MESSAGE, 9, None) == "TRT_TERMINATES")
    chk("free-gen terminates soon -> terminates",
        classify_termination(555, 40, 3) == "TRT_TERMINATES")
    chk("terminal in top-K, not argmax -> near-tie",
        classify_termination(555, 3, None) == "TRT_NEAR_TIE_STOP")
    chk("terminal low-rank + never terminates -> SUPPRESSES",
        classify_termination(555, 999, None) == "TRT_SUPPRESSES_STOP")
    chk("terminal low-rank but terminates late -> near-tie (not suppress)",
        classify_termination(555, 999, 80) == "TRT_NEAR_TIE_STOP")

    # decide_overall
    chk("overall clean", decide_overall(
        [{"verdict": "TRT_TERMINATES"}, {"verdict": "TRT_NEAR_TIE_STOP"}])
        == "TERMINATION_CLEAN")
    chk("overall defect", decide_overall(
        [{"verdict": "TRT_TERMINATES"}, {"verdict": "TRT_SUPPRESSES_STOP"}])
        == "TERMINATION_DEFECT_CANDIDATE")
    chk("overall incomplete", decide_overall(
        [{"verdict": "TRT_TERMINATES"}, {"verdict": "TERMINATION_INCOMPLETE"}])
        == "TERMINATION_INCOMPLETE")

    print(f"\nTFTERM_SELFTEST {'PASS' if fail == 0 else 'FAIL'} ok={ok} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(main())
