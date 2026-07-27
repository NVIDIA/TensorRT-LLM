#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.4 LOCALIZER for the baseline image ``generation_parity``
teacher-forced divergence (NOT a gate -- a diagnostic that produces a named,
numeric classification of the 20/160 mismatches from job 5583360).

Human feedback #1 / directive #2 asks: do NOT wave the baseline divergence off as
"known flaky collapse"; reproduce it, dump the per-step token/logit trajectory,
compare a TEXT-only run of the shared decoder, read the EXACT numeric divergence
(a ratio, not a vibe), and NAME the op/buffer/scale root cause. This script does
exactly that with ONE model construction, baseline only (cuda_graph=off,
overlap=off), TP=4, KVCacheManagerV2, TRTLLM attention, CUTLASS MoE:

  ARM TEXT  -- teacher-force the accepted text-tower SGLang reference
               (sglang_ref_logit_replay.json, short prompts) through the SAME
               shared decoder. This is the control: does the shared NVFP4 decoder
               collapse to token-0 (`!`) at HIGH-confidence positions under
               teacher forcing WITHOUT any image fusion?
  ARM IMAGE -- teacher-force the SGLang multimodal reference (sglang_mm_ref.json)
               with the aligned image re-attached each restart, and for EVERY
               mismatch dump: SGLang top1-top2 margin, TRT greedy token, TRT
               log-prob of token-0, TRT log-prob (and rank) of SGLang's token,
               and the shared-support cos/max_abs. Also report a per-prompt image
               SIZE proxy so a collapse-vs-context-length correlation is visible
               (the fusion-bug vs long-context-SWA-residual discriminator that a
               short-prompt text control alone cannot settle).

The two mismatch populations seen in 5583360 are labeled explicitly:
  * near-tie / low-margin different-real-token flips (SGLang margin small) --
    the accepted fa4(SGLang)-vs-Triton(TRT) attention-kernel-family residual;
  * token-0 (`!`) collapse at HIGH SGLang margin (>= CONF_MARGIN) -- the damning
    class that is NOT accuracy-neutral if it is image-fusion-induced.

Prints, per arm: n_mismatch, n_neartie, n_tok0, n_confident_tok0, min_cos, and a
per-mismatch table; then a VERDICT line that names whether the confident-tok0
collapse is present in the shared decoder (TEXT) or only with images (IMAGE), and
whether IMAGE collapse tracks image size.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_gp_tok0_localize.py
Env: INKLING_CHECKPOINT, INKLING_MM_REF (image ref), INKLING_TEXT_REF (text ref).
"""

import base64
import io
import json
import math
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full",
)
MM_REF = os.environ.get(
    "INKLING_MM_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/sglang_mm_ref.json",
)
TEXT_REF = os.environ.get(
    "INKLING_TEXT_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-bringup/results/sglang_ref_logit_replay.json",
)

TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_GP_STEPS", "32"))
TOPK = int(os.environ.get("INKLING_GP_TOPK", "20"))
TIE_MARGIN = float(os.environ.get("INKLING_GP_TIE_MARGIN", "0.75"))
# DETERMINISTIC mode (same measurement-hygiene switch as the text GP test): bs=1
# removes the cross-row batched-MoE atomic-reduction non-determinism and
# enable_autotuner=False removes autotuner tactic-selection noise (the driving
# sbatch also exports TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1). This pins the
# REPRODUCIBLE mismatch floor so a swinging count (job 5583360=20 vs 5584088=9)
# is separated into a stable kernel-family residual vs batched non-determinism.
DETERMINISTIC = os.environ.get("INKLING_DETERMINISTIC", "0") == "1"
MAX_BS = 1 if DETERMINISTIC else int(os.environ.get("INKLING_LOC_MAX_BS", "4"))
ENABLE_AUTOTUNER = not DETERMINISTIC
# A mismatch at SGLang margin >= CONF_MARGIN is a CONFIDENT divergence: the
# reference is (near-)certain of a real token, so a TRT disagreement there (esp.
# TRT->token-0) is a defect, not a benign NVFP4-vs-NVFP4 tie flip.
CONF_MARGIN = float(os.environ.get("INKLING_CONF_MARGIN", "2.0"))
TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101
TOK0 = 0  # `!` in the Inkling tokenizer -- the documented collapse token.


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _sg_margin(sg_top):
    if len(sg_top) >= 2:
        return float(sg_top[0][1] - sg_top[1][1])
    return float("inf")


def _lp_dict(lp_entry):
    if not isinstance(lp_entry, dict):
        return {}
    return {int(k): float(getattr(v, "logprob", v)) for k, v in lp_entry.items()}


def _lp_stats(trt_lp_dict, sg_top):
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


def _sg_rank(sg_top, tok):
    for i, (t, _lp) in enumerate(sg_top):
        if int(t) == int(tok):
            return i
    return None


def _img_from(r):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))


def _img_size_proxy(r):
    """(W, H, est_patch_rows) size proxy for the collapse-vs-context correlation.

    est_patch_rows uses the HMLP patch grid (patch_size=40, the documented
    'width // patch + 1' padding on each axis) as a monotone proxy for how many
    vision tokens the image expands into -- exact count is not needed, only the
    ordering across the 5 prompts.
    """
    try:
        im = _img_from(r)
        w, h = im.size
        patch = 40
        est = (w // patch + 1) * (h // patch + 1)
        return w, h, int(est)
    except Exception:  # noqa: BLE001
        return -1, -1, -1


def teacher_force(llm, SamplingParams, TokensPrompt, input_ids, sg_ids, sg_top, mm_data=None):
    """Restart-on-fork teacher-forced greedy decode vs SGLang tokens, requesting
    TRT top-K logprobs so every mismatch carries the numeric divergence. Returns
    per_step list of dicts. If mm_data is given it is re-attached on every restart
    (the real image-chat decode path)."""

    def _prompt(ids):
        if mm_data is not None:
            return TokensPrompt(prompt_token_ids=ids, multi_modal_data=mm_data)
        return TokensPrompt(prompt_token_ids=ids)

    forced = list(input_ids)
    t = 0
    per_step = []
    n_calls = 0
    guard = NSTEP + 4
    while t < NSTEP and n_calls < guard:
        out = llm.generate(
            [_prompt(forced)], SamplingParams(max_tokens=NSTEP - t, temperature=0.0, logprobs=TOPK)
        )[0]
        n_calls += 1
        gen = out.outputs[0]
        trt_ids = list(gen.token_ids)
        trt_lps = gen.logprobs or []
        if not trt_ids:
            per_step.append(_mk_step(t, -1, int(sg_ids[t]), sg_top[t], {}))
            forced = list(input_ids) + list(sg_ids[: t + 1])
            t += 1
            continue
        forked = False
        consumed = 0
        for i, tt in enumerate(trt_ids):
            tt_t = t + i
            if tt_t >= NSTEP:
                break
            sg = int(sg_ids[tt_t])
            trt_lp = _lp_dict(trt_lps[i]) if i < len(trt_lps) else {}
            per_step.append(_mk_step(tt_t, int(tt), sg, sg_top[tt_t], trt_lp))
            consumed += 1
            if int(tt) != sg:
                forced = list(input_ids) + list(sg_ids[: tt_t + 1])
                t = tt_t + 1
                forked = True
                break
        if not forked:
            next_t = t + consumed
            if next_t >= NSTEP:
                t = NSTEP
            else:
                per_step.append(_mk_step(next_t, -1, int(sg_ids[next_t]), sg_top[next_t], {}))
                forced = list(input_ids) + list(sg_ids[: next_t + 1])
                t = next_t + 1
    return per_step, n_calls


def _mk_step(t, trt, sg, sg_top, trt_lp):
    margin = _sg_margin(sg_top)
    mx, cos, _k = _lp_stats(trt_lp, sg_top)
    return dict(
        t=t,
        trt=int(trt),
        sg=int(sg),
        match=(int(trt) == int(sg)),
        margin=margin,
        neartie=(margin < TIE_MARGIN),
        confident=(margin >= CONF_MARGIN),
        trt_lp_tok0=trt_lp.get(TOK0),  # TRT log-prob of `!` (None if <topK)
        trt_lp_sg=trt_lp.get(int(sg)),  # TRT log-prob of SGLang token
        sg_rank_of_trt=_sg_rank(sg_top, trt),  # where TRT's token sits in SG topK
        cos=cos,
        max_abs=mx,
    )


def _summarize(arm, records):
    n_mis = n_near = n_tok0 = n_conf_tok0 = 0
    min_cos = float("inf")
    worst = None
    for rec in records:
        for s in rec["steps"]:
            if not math.isnan(s["cos"]):
                if s["cos"] < min_cos:
                    min_cos = s["cos"]
                if worst is None or s["cos"] < worst["cos"]:
                    worst = dict(prompt=rec["id"], **s)
            if s["match"]:
                continue
            n_mis += 1
            if s["neartie"]:
                n_near += 1
            if s["trt"] == TOK0:
                n_tok0 += 1
                if s["confident"]:
                    n_conf_tok0 += 1
    if min_cos is math.inf:
        min_cos = float("nan")
    return dict(
        arm=arm,
        n_mismatch=n_mis,
        n_neartie=n_near,
        n_tok0=n_tok0,
        n_confident_tok0=n_conf_tok0,
        min_cos=min_cos,
        worst=worst,
    )


def _print_arm(tag, records, summ, with_size):
    print(f"\n===== ARM {tag} =====", flush=True)
    for rec in records:
        mism = [s for s in rec["steps"] if not s["match"]]
        near = [s for s in mism if s["neartie"]]
        tok0 = [s for s in mism if s["trt"] == TOK0]
        ctok0 = [s for s in tok0 if s["confident"]]
        size = ""
        if with_size:
            w, h, est = rec.get("size", (-1, -1, -1))
            size = f" img={w}x{h} est_rows={est} prefill≈{rec.get('prefill', '?')}"
        print(
            f"  [{tag}] {rec['id']:<26} mism={len(mism)} neartie={len(near)} "
            f"tok0={len(tok0)} conf_tok0={len(ctok0)} calls={rec['n_calls']}"
            f"{size}",
            flush=True,
        )
        for s in mism:
            lp0 = "None" if s["trt_lp_tok0"] is None else f"{s['trt_lp_tok0']:.3f}"
            lpsg = "None(<topK)" if s["trt_lp_sg"] is None else f"{s['trt_lp_sg']:.3f}"
            rk = s["sg_rank_of_trt"]
            cos = "nan" if math.isnan(s["cos"]) else f"{s['cos']:.4f}"
            tag2 = (
                "CONF_TOK0"
                if (s["trt"] == TOK0 and s["confident"])
                else ("tok0" if s["trt"] == TOK0 else ("neartie" if s["neartie"] else "conf_diff"))
            )
            print(
                f"      step={s['t']:>2} sg={s['sg']:>6} margin={s['margin']:6.3f}"
                f" TRT={s['trt']:>6} trtLP(!)={lp0:>7} trtLP(sg)={lpsg:>11}"
                f" sgRankOfTRT={rk} cos={cos} [{tag2}]",
                flush=True,
            )
    print(
        f"  --> {tag} SUMMARY: mismatch={summ['n_mismatch']} "
        f"neartie={summ['n_neartie']} tok0={summ['n_tok0']} "
        f"confident_tok0={summ['n_confident_tok0']} "
        f"min_cos={summ['min_cos']:.5f}",
        flush=True,
    )
    if summ["worst"]:
        w = summ["worst"]
        print(
            f"  --> {tag} WORST: {w['prompt']} step={w['t']} cos={w['cos']:.5f} "
            f"max_abs={w['max_abs']:.4f} sg={w['sg']} trt={w['trt']} "
            f"margin={w['margin']:.3f}",
            flush=True,
        )


def main() -> int:
    import torch  # noqa: F401

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "gp tok0 localizer needs CUDA GPUs"

    with open(MM_REF) as f:
        md = json.load(f)
    img_ref = md["prompts"] if isinstance(md, dict) else md
    img_ref = [
        r
        for r in img_ref
        if r.get("input_ids")
        and r.get("image_b64")
        and len(r.get("greedy_token_ids", [])) >= NSTEP
        and len(r.get("pos_top", [])) >= NSTEP
    ]
    with open(TEXT_REF) as f:
        td = json.load(f)
    txt_ref = td["prompts"] if isinstance(td, dict) else td
    txt_ref = [
        r
        for r in txt_ref
        if r.get("input_ids")
        and r.get("pos_top")
        and len(r.get("greedy_token_ids", [])) >= NSTEP
        and len(r.get("pos_top", [])) >= NSTEP
    ]
    # keep the control small+fast; the shared decoder either collapses or it does
    # not, 8 short prompts is plenty of signal.
    txt_ref = txt_ref[:8]
    assert len(img_ref) >= 5, f"need >=5 image prompts, got {len(img_ref)}"
    assert len(txt_ref) >= 5, f"need >=5 text prompts, got {len(txt_ref)}"
    print(
        f"[loc] tp={TP} steps={NSTEP} topk={TOPK} tie_margin={TIE_MARGIN} "
        f"conf_margin={CONF_MARGIN} n_img={len(img_ref)} n_txt={len(txt_ref)} "
        f"deterministic={DETERMINISTIC} max_batch_size={MAX_BS} "
        f"enable_autotuner={ENABLE_AUTOTUNER} "
        f"allreduce_autotune_disabled={os.environ.get('TLLM_DISABLE_ALLREDUCE_AUTOTUNE', '0')} "
        f"tok0_is='!' baseline(cuda_graph=off,overlap=off)",
        flush=True,
    )

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.7, dtype="auto", enable_block_reuse=False
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
        enable_autotuner=ENABLE_AUTOTUNER,
        max_seq_len=4096,
        max_batch_size=MAX_BS,
        max_num_tokens=4096,
    )
    print(
        f"[loc] moe_backend={moe_backend} attn=TRTLLM kv=V2 "
        f"deterministic={DETERMINISTIC} max_batch_size={MAX_BS} "
        f"enable_autotuner={ENABLE_AUTOTUNER} "
        f"cuda_graph_hard_path=eager(no-graph)",
        flush=True,
    )

    txt_records, img_records = [], []
    try:
        # ---- ARM TEXT (shared decoder control, no images) ----
        for r in txt_ref:
            ids = _trt_ids(r["input_ids"])
            per_step, n_calls = teacher_force(
                llm,
                SamplingParams,
                TokensPrompt,
                ids,
                r["greedy_token_ids"],
                r["pos_top"],
                mm_data=None,
            )
            txt_records.append(
                dict(
                    id=str(r.get("prompt", r.get("id", "?")))[:26], steps=per_step, n_calls=n_calls
                )
            )
        # ---- ARM IMAGE (vision fusion re-attached each restart) ----
        for r in img_ref:
            ids = _trt_ids(r["input_ids"])
            mm = {"image": [_img_from(r)]}
            w, h, est = _img_size_proxy(r)
            per_step, n_calls = teacher_force(
                llm,
                SamplingParams,
                TokensPrompt,
                ids,
                r["greedy_token_ids"],
                r["pos_top"],
                mm_data=mm,
            )
            img_records.append(
                dict(
                    id=r["id"], steps=per_step, n_calls=n_calls, size=(w, h, est), prefill=len(ids)
                )
            )
    finally:
        llm.shutdown()

    txt_summ = _summarize("TEXT", txt_records)
    img_summ = _summarize("IMAGE", img_records)
    _print_arm("TEXT", txt_records, txt_summ, with_size=False)
    _print_arm("IMAGE", img_records, img_summ, with_size=True)

    # ---- collapse-vs-image-size correlation (fusion-bug vs long-context-SWA) ----
    print("\n===== IMAGE collapse-vs-size =====", flush=True)
    size_rows = []
    for rec in img_records:
        w, h, est = rec.get("size", (-1, -1, -1))
        n_ct0 = sum(
            1 for s in rec["steps"] if (not s["match"]) and s["trt"] == TOK0 and s["confident"]
        )
        size_rows.append((rec["id"], est, n_ct0))
        print(f"  {rec['id']:<26} est_rows={est:>6} confident_tok0={n_ct0}", flush=True)
    ranked = sorted(size_rows, key=lambda x: x[1])
    small_ct0 = sum(x[2] for x in ranked[: len(ranked) // 2])
    large_ct0 = sum(x[2] for x in ranked[len(ranked) // 2 :])
    print(
        f"  smaller-image half confident_tok0={small_ct0} | "
        f"larger-image half confident_tok0={large_ct0}",
        flush=True,
    )

    # ---- VERDICT ----
    txt_ct0 = txt_summ["n_confident_tok0"]
    img_ct0 = img_summ["n_confident_tok0"]
    print("\n===== VERDICT =====", flush=True)
    if txt_ct0 > 0:
        cls = (
            "SHARED_DECODER_TOK0 (inherited): short-prompt text-only decode "
            "ALSO collapses to `!` at confident positions -> not image-fusion"
        )
    elif img_ct0 > 0 and large_ct0 > small_ct0:
        cls = (
            "CONTEXT_LEN_SWA_RESIDUAL (image-context-amplified, not fusion): "
            "confident tok0 only with images AND tracks image size -> the "
            "long visual prefix drives the Triton-SWA-vs-fa4 residual into the "
            "`!`-collapse regime; same class as the accepted text residual"
        )
    elif img_ct0 > 0:
        cls = (
            "IMAGE_FUSION_SUSPECT: confident tok0 only with images and NOT "
            "size-correlated -> localize fusion/KV/position (per-layer replay)"
        )
    else:
        cls = (
            "NO_CONFIDENT_TOK0: only near-tie/low-margin flips (kernel-family "
            "residual); token-0 mismatches are low-confidence"
        )
    print(
        f"INKLING_GP_TOK0_LOCALIZE text_confident_tok0={txt_ct0} "
        f"img_confident_tok0={img_ct0} text_mismatch={txt_summ['n_mismatch']} "
        f"img_mismatch={img_summ['n_mismatch']} text_min_cos={txt_summ['min_cos']:.5f} "
        f"img_min_cos={img_summ['min_cos']:.5f} small_half_ct0={small_ct0} "
        f"large_half_ct0={large_ct0}",
        flush=True,
    )
    print(f"INKLING_GP_TOK0_CLASS {cls}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
