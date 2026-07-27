#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.4 vision ``generation_parity`` (TensorRT-LLM vs SGLang).

``reference_tier=real_source``, ``validation_tier=real_runtime``.

Per-step greedy-token equality vs the SGLang multimodal reference for IMAGE
prompts, using the same restart-on-fork TEACHER FORCING methodology as the text
``inkling_generation_parity_test.py`` (two free-running decodes fork at the first
near-tie and become non-comparable; teacher forcing re-anchors to SGLang's prefix
so every one of the >=32 steps is a true per-step comparison). Each teacher-forced
``llm.generate`` call re-attaches the aligned image, so the ``-101`` vision
positions are fused on every restart -- exactly the decode path an image chat uses.

Two evaluations, one model construction per config:
  1. TEACHER-FORCED per-step greedy equality (STRICT gate): every step whose TRT
     greedy token != SGLang's fails (near-tie label is diagnostic only). This is
     the canonical ``generation_parity`` definition and is immune to free-run
     divergence, because every step re-anchors to SGLang's exact prefix.
  2. FREE-RUNNING collapse detector (DIAGNOSTIC, task.yaml-bounded): flag any
     prompt whose TRT output degenerates (repeated-token / few-unique) where
     SGLang does not. ``task.yaml`` documents a residual ~2/10 free-run collapse
     for this NVFP4 text tower that is "TRT-specific, backend-independent,
     accuracy-neutral (not a gate)"; the multimodal path reuses that same
     accepted text decoder, so the SAME residual applies. Free-run collapse is
     therefore bounded by the documented residual RATE, not required to be zero.

Gate = zero teacher-forced mismatches AND free-run collapse within the documented
~2/10 residual (<= ceil(0.2 * n_prompts)). Runs TP=4, KVCacheManagerV2, TRTLLM
attention, CUTLASS MoE + hMLP vision fusion, for both cuda_graph configs (enabled
exercises the CudaGraphConfig() hard path).

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_image_generation_parity_test.py
Env: INKLING_CHECKPOINT, INKLING_MM_REF (sglang_mm_ref.json).
"""

import base64
import io
import json
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full",
)
REF = os.environ.get(
    "INKLING_MM_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/sglang_mm_ref.json",
)

CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "1" if CUDA_GRAPH else "0") == "1"
TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_GP_STEPS", "32"))
TOPK = int(os.environ.get("INKLING_GP_TOPK", "20"))
TIE_MARGIN = float(os.environ.get("INKLING_GP_TIE_MARGIN", "0.75"))
REPEAT_THRESH = int(os.environ.get("INKLING_GP_REPEAT_THRESH", "12"))
MIN_UNIQUE = int(os.environ.get("INKLING_GP_MIN_UNIQUE", "3"))
# TRT keys on the in-vocab image placeholder 200054; a reference JSON may carry
# the SGLang -101 sentinel. Map it so the executor accepts the token ids.
TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _sg_margin(sg_top):
    if len(sg_top) >= 2:
        return float(sg_top[0][1] - sg_top[1][1])
    return float("inf")


def _max_consec_repeat(ids):
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def _img_from(r):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))


def teacher_force(llm, SamplingParams, TokensPrompt, r, mm_data):
    """Restart-on-fork teacher-forced greedy decode against SGLang's tokens, with
    the image attached on every restart. Returns per_step list of dicts."""
    input_ids = _trt_ids(r["input_ids"])
    sg_ids = r["greedy_token_ids"]
    sg_top = r["pos_top"]
    forced = list(input_ids)
    t = 0
    per_step = []
    n_calls = 0
    guard = NSTEP + 4
    while t < NSTEP and n_calls < guard:
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=forced, multi_modal_data=mm_data)],
            SamplingParams(max_tokens=NSTEP - t, temperature=0.0),
        )[0]
        n_calls += 1
        trt_ids = list(out.outputs[0].token_ids)
        if not trt_ids:
            margin = _sg_margin(sg_top[t])
            per_step.append(
                dict(t=t, trt=-1, sg=int(sg_ids[t]), match=False, neartie=(margin < TIE_MARGIN))
            )
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
            margin = _sg_margin(sg_top[tt_t])
            match = int(tt) == sg
            per_step.append(
                dict(t=tt_t, trt=int(tt), sg=sg, match=match, neartie=(margin < TIE_MARGIN))
            )
            consumed += 1
            if not match:
                forced = list(input_ids) + list(sg_ids[: tt_t + 1])
                t = tt_t + 1
                forked = True
                break
        if not forked:
            next_t = t + consumed
            if next_t >= NSTEP:
                t = NSTEP
            else:
                margin = _sg_margin(sg_top[next_t])
                per_step.append(
                    dict(
                        t=next_t,
                        trt=-1,
                        sg=int(sg_ids[next_t]),
                        match=False,
                        neartie=(margin < TIE_MARGIN),
                    )
                )
                forced = list(input_ids) + list(sg_ids[: next_t + 1])
                t = next_t + 1
    return per_step, n_calls


def main() -> int:
    import torch  # noqa: F401

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "image generation_parity needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [
        r
        for r in ref
        if r.get("input_ids")
        and r.get("image_b64")
        and len(r.get("greedy_token_ids", [])) >= NSTEP
        and len(r.get("pos_top", [])) >= NSTEP
    ]
    assert len(ref) >= 5, f"need >=5 image prompts with >={NSTEP} ref tokens, got {len(ref)}"
    print(
        f"[img-gp] tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"n_prompts={len(ref)} steps={NSTEP} ref={REF}",
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
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        max_seq_len=4096,
        max_batch_size=4,
        max_num_tokens=4096,
    )
    hard_path = "CudaGraphConfig()" if CUDA_GRAPH else "eager(no-graph)"
    print(f"[img-gp] moe_backend={moe_backend} cuda_graph_hard_path={hard_path}", flush=True)

    tf_bad = []
    tf_neartie = 0
    tf_total = 0
    collapse = []
    try:
        # PHASE 1: free-running (collapse detector)
        for r in ref:
            mm = {"image": [_img_from(r)]}
            out = llm.generate(
                [TokensPrompt(prompt_token_ids=_trt_ids(r["input_ids"]), multi_modal_data=mm)],
                SamplingParams(max_tokens=NSTEP, temperature=0.0),
            )[0]
            trt_ids = [int(x) for x in out.outputs[0].token_ids]
            sg_ids = [int(x) for x in r["greedy_token_ids"][:NSTEP]]
            trt_rep = _max_consec_repeat(trt_ids)
            sg_rep = _max_consec_repeat(sg_ids)
            trt_uni = len(set(trt_ids))
            is_collapse = (trt_rep >= REPEAT_THRESH and sg_rep < REPEAT_THRESH) or (
                trt_uni < MIN_UNIQUE and len(set(sg_ids)) >= MIN_UNIQUE
            )
            if is_collapse:
                collapse.append(r["id"])
            print(
                f"  [freerun] {r['id']:<26} trt_maxrep={trt_rep} trt_uniq={trt_uni} "
                f"(sg_maxrep={sg_rep}) {'COLLAPSE' if is_collapse else 'ok'}",
                flush=True,
            )

        # PHASE 2: teacher-forced per-step greedy equality (STRICT gate)
        for r in ref:
            mm = {"image": [_img_from(r)]}
            per_step, n_calls = teacher_force(llm, SamplingParams, TokensPrompt, r, mm)
            mism = [s for s in per_step if not s["match"]]
            near = [s for s in mism if s["neartie"]]
            tf_neartie += len(near)
            tf_total += len(per_step)
            for s in mism:
                tf_bad.append((r["id"], s["t"], s["sg"], s["trt"], s["neartie"]))
            print(
                f"  [teacher] {r['id']:<26} mismatches={len(mism)} "
                f"(neartie={len(near)}) calls={n_calls} steps={len(per_step)}",
                flush=True,
            )
            for _id, t, sg, trt, nt in [x for x in tf_bad if x[0] == r["id"]]:
                print(f"    [MISMATCH] step={t} SGLang={sg} TRT={trt} neartie={nt}", flush=True)
    finally:
        llm.shutdown()

    n_collapse = len(collapse)
    n_bad = len(tf_bad)
    # STRICT gate: teacher-forced per-step greedy equality (canonical
    # generation_parity). Free-run collapse is bounded by task.yaml's documented
    # ~2/10 residual (backend-independent, accuracy-neutral, "not a gate"): allow
    # up to ceil(0.2 * n) collapsed prompts, above which it is a real regression.
    collapse_tol = max(1, (len(ref) + 4) // 5)  # ceil(0.2 * n); == 1 for n=5
    ok = (n_bad == 0) and (n_collapse <= collapse_tol)
    print(
        f"\n[img-gp] TEACHER-FORCED mismatch_steps={n_bad} (neartie={tf_neartie}) "
        f"total_steps={tf_total} | FREE-RUN collapse={n_collapse}/{len(ref)} "
        f"(tol={collapse_tol}, task.yaml ~2/10 residual)",
        flush=True,
    )
    print(
        f"INKLING_IMG_GP_{'OK' if ok else 'FAIL'} tp={TP} tf_mismatch_steps={n_bad} "
        f"tf_neartie={tf_neartie} tf_total_steps={tf_total} "
        f"freerun_collapse={n_collapse}/{len(ref)} collapse_tol={collapse_tol} "
        f"cuda_graph={CUDA_GRAPH} "
        f"overlap={OVERLAP} cuda_graph_hard_path={hard_path}",
        flush=True,
    )
    if collapse:
        print(f"[img-gp] collapse prompts (within-residual diagnostic): {collapse}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
