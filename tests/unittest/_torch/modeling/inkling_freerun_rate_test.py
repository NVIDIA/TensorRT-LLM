#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.4 confirming probe: is the baseline ``validation_*`` free-run
collapse the accepted text tower's documented non-deterministic ~2/10 residual
(GATE-0, CUTLASS fused-MoE cross-row non-determinism) rather than a Goal-1.4
image-fusion or multimodal-batching bug? (``reference_tier=real_source``,
``validation_tier=real_runtime``.)

The killer experiment: run the SAME fixed batch of prompts R times through the
SAME model at baseline (``cuda_graph=false, overlap_scheduler=false``) under
deterministic greedy decoding. Greedy decode on identical inputs is
mathematically deterministic, so ANY run-to-run variation in which prompt
collapses proves the divergence is kernel-level non-determinism, not a
deterministic property of a prompt / image / fusion path.

Two arms, one model construction:
  * ARM_IMG:  the 5 fixed MMMU IMAGE prompts, batched, repeated R times.
  * ARM_TEXT: the SAME 5 question texts as plain TEXT prompts (no image, no
    placeholder), batched, repeated R times.

Readouts:
  * per-arm collapse rate over the R*5 generations (expect ~2/10 both arms if it
    is the shared-decoder residual);
  * which prompt ids collapse in each rep -> if the collapsing id MIGRATES across
    reps with identical inputs, the collapse is NON-DETERMINISTIC;
  * ARM_TEXT collapsing at all proves the residual exists with NO image, so the
    multimodal fusion path is not its cause.

This is a diagnostic probe (does not gate on its own). It confirms the named
root cause the ``inkling_image_localize_test`` first surfaced (same
``validation_Accounting_1`` prompt collapsed when batched but was clean + fully
SGLang-matched when run alone; the collapse then appeared on ``validation_Math_1``
instead).

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_freerun_rate_test.py
Env: INKLING_CHECKPOINT/INKLING_CKPT, INKLING_FR_REPS (default 4),
     INKLING_FR_N (default 5), MMMU_ALIGN_CACHE, SGLANG_PY.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "0") == "1"
TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_FR_STEPS", "40"))
REPS = int(os.environ.get("INKLING_FR_REPS", "4"))
N_PROMPTS = int(os.environ.get("INKLING_FR_N", "5"))
REPEAT_THRESH = int(os.environ.get("INKLING_FR_REPEAT_THRESH", "12"))
MIN_UNIQUE = int(os.environ.get("INKLING_FR_MIN_UNIQUE", "3"))


def _maxrep(ids):
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def _stats(out):
    import torch

    gen = out.outputs[0]
    ids = [int(t) for t in (gen.token_ids or [])]
    gl = gen.generation_logits
    finite = True
    if gl is not None:
        glt = torch.as_tensor(gl).float()
        finite = bool(torch.isfinite(glt).all())
    mr = _maxrep(ids)
    uq = len(set(ids))
    collapse = (mr >= REPEAT_THRESH) or (uq < MIN_UNIQUE)
    return dict(n=len(ids), maxrep=mr, uniq=uq, finite=finite, collapse=collapse)


def _text_only_ids(tok, prompt_text, effort):
    import inkling_image_prompts as P

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    try:
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, reasoning_effort=effort
        )
    except Exception:  # noqa: BLE001
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return P._normalize_ids(ids)


def _run_batch(llm, SamplingParams, TokensPrompt, prompts_spec, sampling, arm):
    """One batched generate over prompts_spec (list of (id, TokensPrompt))."""
    outs = llm.generate([ts for _id, ts in prompts_spec], sampling)
    rows = []
    for (rid, _ts), out in zip(prompts_spec, outs):
        s = _stats(out)
        rows.append((rid, s))
    return rows


def main() -> int:
    import io

    import inkling_image_prompts as P
    import torch
    from PIL import Image

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401 (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "Goal-1.4 free-run rate probe needs CUDA"
    ckpt = P.CKPT
    recs = P.build_prompts(N_PROMPTS)
    tok = P._build_tokenizer()
    effort = P.REASONING_EFFORT
    print(
        f"[fr] tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"reps={REPS} n_prompts={len(recs)} steps={NSTEP} ckpt={ckpt}",
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
        max_batch_size=8,
        max_num_tokens=8192,
    )
    hard = "CudaGraphConfig()" if CUDA_GRAPH else "eager(no-graph)"
    print(f"[fr] moe_backend={moe_backend} cuda_graph_hard_path={hard}", flush=True)

    sampling = SamplingParams(max_tokens=NSTEP, temperature=0.0, return_generation_logits=True)

    # ARM_IMG: identical 5-image batch, repeated REPS times.
    img_spec = [
        (
            r["id"],
            TokensPrompt(
                prompt_token_ids=list(r["input_ids"]),
                multi_modal_data={"image": [Image.open(io.BytesIO(r["image_bytes"]))]},
            ),
        )
        for r in recs
    ]
    # ARM_TEXT: SAME question text, no image.
    txt_spec = [
        (r["id"], TokensPrompt(prompt_token_ids=list(_text_only_ids(tok, r["prompt"], effort))))
        for r in recs
    ]

    arms = {"IMG": img_spec, "TEXT": txt_spec}
    results = {"IMG": [], "TEXT": []}
    try:
        for arm, spec in arms.items():
            for rep in range(REPS):
                rows = _run_batch(llm, SamplingParams, TokensPrompt, spec, sampling, arm)
                collapsed_ids = [rid for rid, s in rows if s["collapse"]]
                nonfinite_ids = [rid for rid, s in rows if not s["finite"]]
                results[arm].append((rep, collapsed_ids, nonfinite_ids, rows))
                print(
                    f"[fr] {arm} rep={rep} collapsed={collapsed_ids} nonfinite={nonfinite_ids}",
                    flush=True,
                )
                for rid, s in rows:
                    print(
                        f"    {arm} rep={rep} {rid:<26} maxrep={s['maxrep']} "
                        f"uniq={s['uniq']} finite={s['finite']} "
                        f"collapse={s['collapse']}",
                        flush=True,
                    )
    finally:
        llm.shutdown()

    def summarize(arm):
        reps = results[arm]
        total = REPS * len(recs)
        ncol = sum(len(c) for _r, c, _nf, _rows in reps)
        nnf = sum(len(nf) for _r, _c, nf, _rows in reps)
        # per-id collapse count across reps (migration signal)
        per_id = {}
        for _r, c, _nf, _rows in reps:
            for rid in c:
                per_id[rid] = per_id.get(rid, 0) + 1
        # collapse SETS differ across reps with identical inputs => nondeterministic
        sets = [tuple(sorted(c)) for _r, c, _nf, _rows in reps]
        nondet = len(set(sets)) > 1
        return total, ncol, nnf, per_id, nondet, sets

    it, ic, inf_, iper, ind, isets = summarize("IMG")
    tt, tc, tnf, tper, tnd, tsets = summarize("TEXT")
    print("\n[fr] ===== SUMMARY =====", flush=True)
    print(
        f"[fr] IMG  collapse={ic}/{it} nonfinite_gens={inf_} per_id={iper} "
        f"nondeterministic={ind} sets={isets}",
        flush=True,
    )
    print(
        f"[fr] TEXT collapse={tc}/{tt} nonfinite_gens={tnf} per_id={tper} "
        f"nondeterministic={tnd} sets={tsets}",
        flush=True,
    )
    # Named-cause confirmation: collapse exists in BOTH arms (so not fusion-caused)
    # AND migrates across identical-input reps (so non-deterministic, not a
    # deterministic prompt/fusion bug).
    text_has_collapse = tc > 0
    migrates = ind or tnd
    print(
        f"INKLING_FREERUN_RATE img_collapse={ic}/{it} text_collapse={tc}/{tt} "
        f"img_nondeterministic={ind} text_nondeterministic={tnd} "
        f"text_collapse_without_image={text_has_collapse} "
        f"collapse_migrates_identical_inputs={migrates} "
        f"cuda_graph={CUDA_GRAPH} overlap={OVERLAP}",
        flush=True,
    )
    print("INKLING_FREERUN_RATE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
