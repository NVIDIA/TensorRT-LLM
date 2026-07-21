#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""crit7 root-cause ISOLATION: is the per-step SGLang divergence a fixable bug, or
inherent NVFP4-routed-expert-kernel accumulation?

Localization established this iteration
--------------------------------------
``hf_quant_config.json`` shows the ONLY NVFP4 compute in the Inkling text tower is
the routed experts (``model.llm.layers.{3..65}.mlp.experts``). Every other module
-- ALL attention (``layers.N.attn``), the router/gate, shared experts, all norms,
short-convs, dense layers 0-1, embed and unembed -- is bf16 (excluded from NVFP4).
So the only source of a TRT-vs-SGLang *kernel* difference is the routed-expert
GEMM: TRT (CUTLASS grouped-GEMM) vs SGLang (flashinfer_trtllm). crit8 measured
per-MoE-layer decode cos ~0.997; compounded over the 63 MoE layers ~0.997**63 ~=
0.83, matching the observed full-logit cos 0.80-0.91.

The experiment
--------------
Run the IDENTICAL teacher-forced prefixes ``[prompt + SGLang_greedy[:t]]`` through
TWO different TRT NVFP4 MoE kernels -- ``CUTLASS`` (fused grouped GEMM, the
production path) and ``VANILLA`` (unfused per-expert fp4 GEMM) -- and cross-compare
their first-generated-token argmax and top-K logprobs, plus each backend vs the
SGLang fixture.

Decision rule
-------------
* CUTLASS-vs-VANILLA diverges per-step by a magnitude comparable to
  CUTLASS-vs-SGLang  => two different fp4 MoE kernels on the SAME weights already
  disagree at the token level; per-step greedy-token equality with a THIRD fp4
  kernel (SGLang) is not achievable by any faithful implementation. crit7's
  strict per-step equality is then an over-strict proxy for this cross-fp4-kernel
  setup, and the task's real gate (GSM8K/MMLU within 2 pts) is the decider.
* CUTLASS ~= VANILLA but both diverge from SGLang => the MoE kernel choice is NOT
  the dominant factor; the divergence is a systematic TRT-vs-SGLang difference
  (bf16 formula / quant-recipe) worth localizing further.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_moe_backend_isolate_test.py
Env: INKLING_CHECKPOINT, INKLING_SGLANG_REF, INKLING_MOE_BACKENDS (csv,
     default "CUTLASS,VANILLA"), INKLING_TP_STEPS (default 16),
     INKLING_TP_TOPK (default 20).
"""
import gc
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

BACKENDS = [b.strip() for b in
            os.environ.get("INKLING_MOE_BACKENDS", "CUTLASS,VANILLA").split(",")
            if b.strip()]
NSTEP = int(os.environ.get("INKLING_TP_STEPS", "16"))
TOPK = int(os.environ.get("INKLING_TP_TOPK", "20"))


def _cos_maxabs(da: dict, db: dict):
    """cos + max_abs over the shared-token support of two {token_id: logprob}."""
    import torch
    ids = [t for t in da if t in db]
    if len(ids) < 2:
        return float("nan"), float("nan")
    a = torch.tensor([da[i] for i in ids])
    b = torch.tensor([db[i] for i in ids])
    return (float(torch.nn.functional.cosine_similarity(a[None], b[None]).item()),
            float((a - b).abs().max()))


def _run_backend(backend, ref):
    """Return per-(prompt, step) dict: {'tok': argmax_id, 'lp': {id: logprob}}."""
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    print(f"\n[isolate] ===== loading backend={backend} =====", flush=True)
    llm = LLM(
        CKPT, tensor_parallel_size=4, trust_remote_code=True,
        attn_backend="TRTLLM", moe_config=MoeConfig(backend=backend),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75,
                                      dtype="auto", enable_block_reuse=False),
        cuda_graph_config=None, disable_overlap_scheduler=True,
        max_seq_len=2048, max_batch_size=64, max_num_tokens=4096)
    prompts, index = [], []
    for pi, r in enumerate(ref):
        base = list(r["input_ids"])
        sg = r["greedy_token_ids"]
        for t in range(NSTEP):
            prompts.append(
                TokensPrompt(prompt_token_ids=base + [int(x) for x in sg[:t]]))
            index.append((pi, t))
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOPK)
    try:
        outputs = llm.generate(prompts, sampling)
        res = {}
        for (pi, t), out in zip(index, outputs):
            gen = out.outputs[0]
            ids = list(gen.token_ids)
            tok = int(ids[0]) if ids else None
            lpd = {}
            lps = gen.logprobs or []
            if lps and isinstance(lps[0], dict):
                lpd = {int(k): float(getattr(v, "logprob", v))
                       for k, v in lps[0].items()}
            res[(pi, t)] = {"tok": tok, "lp": lpd}
        return res
    finally:
        llm.shutdown()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _vs_sglang(res, ref):
    """full-match count (all NSTEP argmax == SGLang) + min per-step logprob cos."""
    n_full, cos_all = 0, []
    for pi, r in enumerate(ref):
        ok = True
        for t in range(NSTEP):
            e = res.get((pi, t))
            sg_tok = int(r["greedy_token_ids"][t])
            if not e or e["tok"] != sg_tok:
                ok = False
            sg_top = {int(tid): float(lp) for tid, lp in r["pos_top"][t]}
            if e and e["lp"]:
                c, _ = _cos_maxabs(e["lp"], sg_top)
                if not math.isnan(c):
                    cos_all.append(c)
        n_full += int(ok)
    return n_full, (min(cos_all) if cos_all else float("nan"))


def main() -> int:
    import torch
    assert torch.cuda.is_available(), "needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids")
           and len(r.get("greedy_token_ids", [])) >= NSTEP][:6]
    assert len(ref) >= 5, f"need >=5 prompts, got {len(ref)}"
    print(f"[isolate] backends={BACKENDS} n_prompts={len(ref)} steps={NSTEP} "
          f"topk={TOPK}", flush=True)

    per_backend = {}
    for b in BACKENDS:
        per_backend[b] = _run_backend(b, ref)
        nf, mc = _vs_sglang(per_backend[b], ref)
        print(f"[isolate] backend={b} vs SGLang: full_match={nf}/{len(ref)} "
              f"min_step_cos={mc:.5f}", flush=True)

    # Cross-backend agreement (only meaningful with >=2 backends).
    if len(BACKENDS) >= 2:
        a, b = BACKENDS[0], BACKENDS[1]
        ra, rb = per_backend[a], per_backend[b]
        same, tot, cross_cos, first_forks = 0, 0, [], []
        for pi in range(len(ref)):
            forked = False
            for t in range(NSTEP):
                ea, eb = ra.get((pi, t)), rb.get((pi, t))
                if not ea or not eb:
                    continue
                tot += 1
                agree = ea["tok"] == eb["tok"]
                same += int(agree)
                if ea["lp"] and eb["lp"]:
                    c, _ = _cos_maxabs(ea["lp"], eb["lp"])
                    if not math.isnan(c):
                        cross_cos.append(c)
                if not agree and not forked:
                    first_forks.append((ref[pi]["prompt"], t))
                    forked = True
        agree_rate = same / tot if tot else float("nan")
        cross_min = min(cross_cos) if cross_cos else float("nan")
        print(f"\n[isolate] CROSS-BACKEND {a} vs {b}: "
              f"argmax_agree={same}/{tot} ({agree_rate:.3f}) "
              f"min_step_cos={cross_min:.5f}", flush=True)
        for p, t in first_forks:
            print(f"    first fork step={t:>2d}  {p!r}", flush=True)
        # Verdict: if the two fp4 kernels disagree at the token level, per-step
        # equality with SGLang's third fp4 kernel is not achievable.
        print(f"INKLING_MOE_ISOLATE backends={a},{b} "
              f"cross_argmax_agree={agree_rate:.3f} cross_min_cos={cross_min:.5f} "
              f"(agree<1.0 => two fp4 MoE kernels already diverge per-step on "
              f"identical inputs => per-step SGLang parity is cross-fp4-kernel "
              f"infeasible; accuracy is the decider)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
