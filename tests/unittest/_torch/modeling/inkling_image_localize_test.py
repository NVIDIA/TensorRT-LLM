#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.4 baseline localization of the ``validation_Accounting_1``
non-finite / repeated-token collapse (``reference_tier=real_source``,
``validation_tier=real_runtime``).

This is the localization step the acceptance criterion requires *before any fix
is accepted*: reproduce the failing prompt at TP=4 with
``cuda_graph=false, overlap_scheduler=false``, dump the finish reason plus the
per-step token / logit-finiteness trajectory, compare the same textual content
as text-only, and read the divergence against the SGLang reference (which does
NOT collapse on this item). The prior evidence already pins two of the three
possible culprits as clean:

  * the isolated hMLP vision tower output for this exact image bitwise-matches
    SGLang (iter-7 ``inkling_vision_tower_artifact_iter7.json``:
    ``validation_Accounting_1`` ``max_abs=0.0 cos=1.0`` at 444 patches), so the
    fused embeds ENTERING the decoder are numerically identical to the
    like-precision source path; and
  * SGLang generates 40 coherent tokens for this item
    (``results/sglang_mm_ref.json``: ``sg_maxrep=1 sg_uniq=37``), so the
    reference decoder handles the same fused stream without collapse.

What remains unknown -- and what this probe measures -- is WHERE in the TRT
decode trajectory the non-finite logit appears relative to the repetition, which
distinguishes a prefill/fusion numerical bug (step-0 logit already non-finite)
from a decode-time free-run repetition spiral (step-0 finite + argmax matches
SGLang, non-finite only AFTER a repeated-token run begins -- the documented
``task.yaml`` residual, whose real gate is the teacher-forced
``generation_parity``, immune to free-run divergence).

For each target id it runs, in ONE TP=4 baseline model construction:
  1. MULTIMODAL free-run: ``llm.generate`` (image attached) with
     ``return_generation_logits`` -> per-step (token, finite, max|logit|),
     first-non-finite step, collapse-start step, fork-from-SGLang step, finish
     reason.
  2. TEXT-ONLY free-run: the SAME question text rendered as a plain text chat
     (no image, no placeholder) -> does the same textual content go
     non-finite / collapse without the image? (isolates fusion-induced vs a
     pre-existing decode edge on this content).

It prints a machine-greppable ``INKLING_LOCALIZE`` verdict line per target and a
``INKLING_LOCALIZE_DONE`` summary. It is a diagnostic probe (env-gated, default
targets the one bad prompt + one good control); it does not gate on its own,
because the localization criterion is satisfied by the dumped evidence + named
root cause, not by a pass/fail.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_image_localize_test.py
Env: INKLING_CHECKPOINT/INKLING_CKPT, INKLING_MM_REF (sglang_mm_ref.json),
     INKLING_LOCALIZE_IDS (comma list, default
     "validation_Accounting_1,validation_Math_1"),
     MMMU_ALIGN_CACHE (warm cache), SGLANG_PY.
"""

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "0") == "1"
TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_LOC_STEPS", "40"))
# The bad prompt first, then a known-good control (Math_1 passed the e2e smoke).
TARGET_IDS = [
    s
    for s in os.environ.get(
        "INKLING_LOCALIZE_IDS", "validation_Accounting_1,validation_Math_1"
    ).split(",")
    if s
]
REPEAT_RUN = int(os.environ.get("INKLING_LOC_REPEAT_RUN", "6"))
REF = os.environ.get(
    "INKLING_MM_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/sglang_mm_ref.json",
)


def _first_run_start(ids, run=REPEAT_RUN):
    """First index i that begins a consecutive run of >= ``run`` identical ids."""
    cur = 1
    for i in range(1, len(ids)):
        cur = cur + 1 if ids[i] == ids[i - 1] else 1
        if cur >= run:
            return i - run + 1
    return -1


def _first_fork(trt_ids, sg_ids):
    """First step where TRT's greedy token differs from SGLang's."""
    n = min(len(trt_ids), len(sg_ids))
    for i in range(n):
        if int(trt_ids[i]) != int(sg_ids[i]):
            return i
    return -1 if n == 0 else n


def _per_step_stats(out, nstep):
    """Return (token_ids, per_step_finite[list[bool]], per_step_maxabs[list],
    step0_finite, first_nonfinite_step) from an LLM output with
    ``return_generation_logits`` set."""
    import torch

    gen = out.outputs[0]
    ids = [int(t) for t in (gen.token_ids or [])]
    gl = gen.generation_logits
    finite = []
    maxabs = []
    first_nf = -1
    if gl is not None:
        glt = torch.as_tensor(gl).float()
        glt = glt.reshape(glt.shape[0], -1) if glt.dim() > 1 else glt.reshape(1, -1)
        for i in range(glt.shape[0]):
            row = glt[i]
            f = bool(torch.isfinite(row).all())
            finite.append(f)
            fin = row[torch.isfinite(row)]
            maxabs.append(float(fin.abs().max()) if fin.numel() else float("inf"))
            if not f and first_nf < 0:
                first_nf = i
    step0_finite = finite[0] if finite else True
    return ids, finite, maxabs, step0_finite, first_nf


def _text_only_ids(tok, prompt_text, effort):
    """Render the SAME question text as a plain text chat (no image content part,
    no placeholder), matching the accepted text path's reasoning_effort."""
    import inkling_image_prompts as P

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    try:
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, reasoning_effort=effort
        )
    except Exception:  # noqa: BLE001
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return P._normalize_ids(ids)


def main() -> int:
    import inkling_image_prompts as P
    import torch
    from PIL import Image

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401 (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "Goal-1.4 localization needs CUDA GPUs"
    ckpt = P.CKPT

    # SGLang reference (does NOT collapse on any item) for token alignment.
    sg_by_id = {}
    if os.path.exists(REF):
        with open(REF) as f:
            refdoc = json.load(f)
        for r in refdoc["prompts"] if isinstance(refdoc, dict) else refdoc:
            sg_by_id[r["id"]] = [int(t) for t in r.get("greedy_token_ids", [])]

    # Build every requested target from the shared canonical prompt builder so the
    # token stream + image are byte-identical to the e2e / parity / SGLang paths.
    all_recs = {r["id"]: r for r in P.build_prompts(max(5, len(TARGET_IDS)))}
    recs = [all_recs[i] for i in TARGET_IDS if i in all_recs]
    assert recs, f"none of {TARGET_IDS} resolved from build_prompts"
    tok = P._build_tokenizer()
    effort = P.REASONING_EFFORT

    print(
        f"[loc] tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"targets={[r['id'] for r in recs]} steps={NSTEP} ckpt={ckpt}",
        flush=True,
    )

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.7, dtype="auto", enable_block_reuse=False
    )
    llm = LLM(
        ckpt,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=kv_cache_config,
        gather_generation_logits=True,
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        max_seq_len=4096,
        max_batch_size=4,
        max_num_tokens=4096,
    )
    hard_path = "CudaGraphConfig()" if CUDA_GRAPH else "eager(no-graph)"
    print(f"[loc] moe_backend={moe_backend} cuda_graph_hard_path={hard_path}", flush=True)

    sampling = SamplingParams(max_tokens=NSTEP, temperature=0.0, return_generation_logits=True)
    verdicts = []
    try:
        for r in recs:
            rid = r["id"]
            img = Image.open(io.BytesIO(r["image_bytes"]))
            sg_ids = sg_by_id.get(rid, [])

            # --- (1) MULTIMODAL free-run --------------------------------------
            mm_out = llm.generate(
                [
                    TokensPrompt(
                        prompt_token_ids=list(r["input_ids"]), multi_modal_data={"image": [img]}
                    )
                ],
                sampling,
            )[0]
            mm_ids, mm_fin, mm_max, mm_s0, mm_nf = _per_step_stats(mm_out, NSTEP)
            mm_fr = getattr(mm_out.outputs[0], "finish_reason", None)
            mm_collapse = _first_run_start(mm_ids)
            mm_fork = _first_fork(mm_ids, sg_ids) if sg_ids else -1

            print(
                f"\n[loc] ===== {rid} (num_patches={r.get('num_patches')}) "
                f"MULTIMODAL free-run =====",
                flush=True,
            )
            print(
                f"[loc] {rid} MM finish={mm_fr} n_tok={len(mm_ids)} "
                f"step0_finite={mm_s0} first_nonfinite={mm_nf} "
                f"collapse_start={mm_collapse} fork_from_sglang={mm_fork} "
                f"maxrep={_maxrep(mm_ids)} uniq={len(set(mm_ids))}",
                flush=True,
            )
            # Per-step trajectory: token, finite, max|logit| (shows the overflow
            # curve). SGLang token shown alongside for the aligned prefix.
            for i in range(len(mm_ids)):
                sg = sg_ids[i] if i < len(sg_ids) else None
                mark = "" if (sg is None or sg == mm_ids[i]) else "  <-- FORK"
                fmark = "" if mm_fin[i] else "  <== NON-FINITE"
                print(
                    f"    step {i:>2} tok={mm_ids[i]:>7} finite={mm_fin[i]} "
                    f"max|logit|={mm_max[i]:.3e} sglang={sg}{mark}{fmark}",
                    flush=True,
                )

            # --- (2) TEXT-ONLY free-run (same question text, no image) --------
            to_ids = _text_only_ids(tok, r["prompt"], effort)
            to_out = llm.generate([TokensPrompt(prompt_token_ids=list(to_ids))], sampling)[0]
            t_ids, t_fin, t_max, t_s0, t_nf = _per_step_stats(to_out, NSTEP)
            t_fr = getattr(to_out.outputs[0], "finish_reason", None)
            t_collapse = _first_run_start(t_ids)
            print(
                f"[loc] {rid} TEXT-ONLY finish={t_fr} n_tok={len(t_ids)} "
                f"step0_finite={t_s0} first_nonfinite={t_nf} "
                f"collapse_start={t_collapse} maxrep={_maxrep(t_ids)} "
                f"uniq={len(set(t_ids))}",
                flush=True,
            )

            # --- named verdict ------------------------------------------------
            mm_collapsed = mm_collapse >= 0
            mm_nonfinite = mm_nf >= 0
            t_collapsed = t_collapse >= 0
            t_nonfinite = t_nf >= 0
            if not mm_s0:
                verdict = "PREFILL_NONFINITE"  # step-0 logit already non-finite
            elif not mm_nonfinite and not mm_collapsed:
                verdict = "CLEAN"
            elif mm_nonfinite and mm_collapsed and mm_nf >= mm_collapse:
                verdict = "DECODE_FREERUN_SPIRAL"  # nonfinite AFTER repetition
            elif mm_nonfinite and (not mm_collapsed or mm_nf < mm_collapse):
                verdict = "NONFINITE_BEFORE_COLLAPSE"  # numeric event precedes rep
            else:
                verdict = "COLLAPSE_FINITE"  # repetition but logits stayed finite
            fusion_induced = (mm_collapsed or mm_nonfinite) and not (t_collapsed or t_nonfinite)
            print(
                f"INKLING_LOCALIZE {rid} verdict={verdict} "
                f"step0_finite={mm_s0} first_nonfinite={mm_nf} "
                f"collapse_start={mm_collapse} fork_from_sglang={mm_fork} "
                f"mm_collapsed={mm_collapsed} mm_nonfinite={mm_nonfinite} "
                f"text_only_collapsed={t_collapsed} "
                f"text_only_nonfinite={t_nonfinite} "
                f"fusion_induced={fusion_induced}",
                flush=True,
            )
            verdicts.append(
                (rid, verdict, mm_s0, mm_nf, mm_collapse, mm_fork, t_collapsed, fusion_induced)
            )
    finally:
        llm.shutdown()

    print("\n[loc] ===== SUMMARY =====", flush=True)
    for rid, v, s0, nf, cs, fk, tc, fi in verdicts:
        print(
            f"[loc] {rid:<26} verdict={v} step0_finite={s0} "
            f"first_nonfinite={nf} collapse_start={cs} fork={fk} "
            f"text_only_collapsed={tc} fusion_induced={fi}",
            flush=True,
        )
    print(
        f"INKLING_LOCALIZE_DONE targets={len(verdicts)} cuda_graph={CUDA_GRAPH} overlap={OVERLAP}",
        flush=True,
    )
    return 0


def _maxrep(ids):
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def test_image_localize():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Goal-1.4 localization")
    assert main() == 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
