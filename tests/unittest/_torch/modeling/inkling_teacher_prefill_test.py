#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit7 diagnostic: TEACHER-FORCED PREFILL per-step parity vs SGLang.

The crit7 free-running run found TensorRT-LLM's greedy generation diverges from
SGLang per-step (large-margin forks, per-step logprob max_abs growing with
position). Because free-running matched SGLang token-for-token up to each fork,
the fork itself happened with an IDENTICAL context -- so it is a teacher-forced
divergence. This test isolates whether that divergence is MODEL-LEVEL (present in
the prefill path too) or DECODE-specific.

Method (robust, reuses the proven ``logprobs`` sampler path -- no
generation-logits gather, no context-logits API)
-------------------------------------------------------------------------------
For each reference prompt, build the family of teacher-forced prefixes
``[prompt_ids + SGLang_greedy[:t]]`` for t in 0..NSTEP-1 and generate ONE token
from each (deterministic greedy, ``logprobs=K``). The single generated token's
distribution is the model's PREFILL prediction for step t given SGLang's exact
prefix. Compare its argmax to SGLang's greedy token t and its top-K logprobs to
SGLang's reference row.

  * TRT-prefill matches SGLang at every step -> the prefill path is correct and
    the free-running divergence is DECODE-specific (KV-cache / decode-attention).
  * TRT-prefill ALSO diverges at the same steps -> the divergence is MODEL-level
    (both paths; e.g. NVFP4 CUTLASS-vs-flashinfer MoE / Triton-vs-fa4 attention
    numerics accumulating with sequence position), NOT a decode-state bug.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_teacher_prefill_test.py
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
NSTEP = int(os.environ.get("INKLING_TP_STEPS", "32"))
TOPK = int(os.environ.get("INKLING_TP_TOPK", "20"))


def _lp_stats(trt_lp_dict, sg_top):
    import torch
    sg = {int(tid): float(lp) for tid, lp in sg_top}
    ids = [tid for tid in sg if tid in trt_lp_dict]
    if len(ids) < 2:
        return float("nan"), float("nan")
    a = torch.tensor([trt_lp_dict[i] for i in ids])
    b = torch.tensor([sg[i] for i in ids])
    return (float((a - b).abs().max()),
            float(torch.nn.functional.cosine_similarity(a[None], b[None]).item()))


def main() -> int:
    import torch  # noqa: F401

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    import torch as _t
    assert _t.cuda.is_available(), "teacher-prefill needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids")
           and len(r.get("greedy_token_ids", [])) >= NSTEP][:6]
    assert len(ref) >= 5, f"need >=5 prompts, got {len(ref)}"
    print(f"[tp] cuda_graph={CUDA_GRAPH} overlap={OVERLAP} n_prompts={len(ref)} "
          f"steps={NSTEP} topk={TOPK}", flush=True)

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    llm = LLM(
        CKPT, tensor_parallel_size=4, trust_remote_code=True,
        attn_backend="TRTLLM", moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75,
                                      dtype="auto", enable_block_reuse=False),
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        max_seq_len=2048, max_batch_size=64, max_num_tokens=4096)
    print(f"[tp] moe_backend={moe_backend}", flush=True)

    # Build every teacher-forced prefix [prompt + SGLang[:t]] and generate 1 token.
    prompts, index = [], []
    for pi, r in enumerate(ref):
        base = list(r["input_ids"])
        sg = r["greedy_token_ids"]
        for t in range(NSTEP):
            prompts.append(TokensPrompt(prompt_token_ids=base + [int(x)
                                                                 for x in sg[:t]]))
            index.append((pi, t))
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOPK)
    try:
        outputs = llm.generate(prompts, sampling)
    finally:
        llm.shutdown()

    # per-prompt: leading teacher-forced-prefill match length + logit stats
    per = {pi: {"match": [False] * NSTEP, "mx": [float("nan")] * NSTEP,
                "cos": [float("nan")] * NSTEP} for pi in range(len(ref))}
    for (pi, t), out in zip(index, outputs):
        gen = out.outputs[0]
        ids = list(gen.token_ids)
        if not ids:
            continue
        trt_tok = int(ids[0])
        sg_tok = int(ref[pi]["greedy_token_ids"][t])
        per[pi]["match"][t] = (trt_tok == sg_tok)
        lps = gen.logprobs or []
        if lps and isinstance(lps[0], dict):
            lpd = {int(k): float(getattr(v, "logprob", v)) for k, v in lps[0].items()}
            mx, cos = _lp_stats(lpd, ref[pi]["pos_top"][t])
            per[pi]["mx"][t] = mx
            per[pi]["cos"][t] = cos

    n_full = 0
    all_cos = []
    for pi, r in enumerate(ref):
        m = per[pi]["match"]
        # leading run of teacher-forced-prefill matches
        lead = 0
        for t in range(NSTEP):
            if m[t]:
                lead += 1
            else:
                break
        full = all(m)
        n_full += int(full)
        cos = [c for c in per[pi]["cos"] if not math.isnan(c)]
        mx = [x for x in per[pi]["mx"] if not math.isnan(x)]
        all_cos += cos
        n_match = sum(int(x) for x in m)
        first_bad = next((t for t in range(NSTEP) if not m[t]), None)
        fb_margin = None
        if first_bad is not None:
            top = r["pos_top"][first_bad]
            fb_margin = (top[0][1] - top[1][1]) if len(top) > 1 else None
        print(f"  prefix-match lead={lead}/{NSTEP} total={n_match}/{NSTEP} "
              f"first_bad_step={first_bad} "
              f"margin={fb_margin if fb_margin is None else round(fb_margin,3)} "
              f"logp(min_cos={min(cos) if cos else float('nan'):.5f} "
              f"max_abs={max(mx) if mx else float('nan'):.4f}) {r['prompt']!r}",
              flush=True)

    min_cos = min(all_cos) if all_cos else float("nan")
    print(f"\n[tp] TEACHER-FORCED PREFILL: full-match(all {NSTEP} steps)="
          f"{n_full}/{len(ref)} prompts | per-step-logp min_cos={min_cos:.5f} | "
          f"cuda_graph={CUDA_GRAPH} overlap={OVERLAP}", flush=True)
    print(f"INKLING_TP_PREFILL full_match={n_full}/{len(ref)} min_cos={min_cos:.5f} "
          f"cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
          f"(full_match>=5 => prefill matches SGLang => decode-specific; "
          f"else => model-level divergence)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
