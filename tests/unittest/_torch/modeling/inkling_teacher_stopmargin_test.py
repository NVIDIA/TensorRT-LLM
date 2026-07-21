#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SYSTEMATIC-vs-NOISE stop-token localizer on the GSM8K bad prompts.

Context
-------
The fair served GSM8K gap (TRT 0.92 vs SGLang 0.98) is driven by RUNAWAY
generations on hard prompts: TRT stays in the reasoning channel, never emits the
reasoning->content transition (200010 <|end_message|>) / turn-close (200006
<|content_model_end_sampling|>), and rambles to the 2048 cap; SGLang commits the
correct answer at 184-433 tokens. iter85 decisively ruled out a TRT cache/window/
paging/sconv/MoE-decode bug (decode==stateless to cos 0.9999). This test settles
the remaining question: is the stop-token failure SYSTEMATIC (a real, localizable
logit bias TRT can fix) or free-run NOISE (TRT forks onto a longer path early)?

Method (teacher-forcing = the ONLY way to compare on an identical context)
--------------------------------------------------------------------------
Fixture = SGLang's WINNING trajectory per bad prompt (capture_sglang_badprompts.py:
input_ids + greedy_token_ids + per-position top-K logprobs, incl. the stop tokens).
For each prompt, feed TRT every teacher-forced prefix [prompt + SGLang_greedy[:t]]
(t up to SGLang's stop step) and generate ONE greedy token with logprobs. Then:
  * first_fork = first t where TRT's argmax != SGLang's winning token.
  * AT SGLang's stop_step (where SGLang emits 200010/200006 to leave reasoning):
    does TRT's argmax == that stop token? what RANK does TRT give the stop token?
      - TRT argmax IS the stop token  -> TRT WOULD stop on the winning prefix =>
        the runaway is a free-run FORK (noise), not stop-suppression.
      - TRT ranks the stop token low  -> SYSTEMATIC stop-suppression => localizable.

Runs the PRODUCTION runtime (TP=4, TRTLLM attn, trtllm-gen MoE, KVCacheManagerV2,
baseline cg=off/ov=off by default). DIAGNOSTIC, not an acceptance gate.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_teacher_stopmargin_test.py
Env: INKLING_CHECKPOINT, INKLING_SGLANG_REF (bad-prompt capture json),
     INKLING_MOE_BACKEND (default TRTLLM), INKLING_SM_CAP (per-prompt step cap).
"""
import json
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")
REF = os.environ.get("INKLING_SGLANG_REF", "")
STOP_TOKENS = {200006, 200010}
CAP = int(os.environ.get("INKLING_SM_CAP", "640"))  # per-prompt teacher-force cap
                                                     # (>= capture max_new=600 so a
                                                     # real stop step is never capped)
# TRT-LLM SamplingParams caps logprobs at 20; top-20 is ample for the decisive
# call (is the stop token TRT's argmax/rank-0 or not), so clamp to the API limit.
TOPK = min(int(os.environ.get("INKLING_SM_TOPK", "20")), 20)
CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "1" if CUDA_GRAPH else "0") == "1"


def main() -> int:
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "stop-margin needs CUDA GPUs"
    assert REF and os.path.exists(REF), f"INKLING_SGLANG_REF not found: {REF!r}"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids") and r.get("greedy_token_ids")]
    assert ref, "no usable trajectories in the fixture"

    # Per-prompt teacher-force horizon: up to stop_step (+3) so the stop position
    # itself is inside the horizon (ss < horizon). Prompts where SGLang emitted NO
    # stop token within the captured window carry no stop-margin signal, so we skip
    # teacher-forcing them (horizon 0) instead of wasting CAP passes.
    horizon = []
    for r in ref:
        ss = r.get("stop_step")
        n = len(r["greedy_token_ids"])
        horizon.append(0 if ss is None else min(ss + 3, n, CAP))
    max_prompt = max(len(r["input_ids"]) for r in ref)
    max_seq = max_prompt + CAP + 8
    print(f"[sm] n_prompts={len(ref)} horizons={horizon} cap={CAP} topk={TOPK} "
          f"max_seq={max_seq} cuda_graph={CUDA_GRAPH} overlap={OVERLAP}", flush=True)

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "TRTLLM")
    llm = LLM(
        CKPT, tensor_parallel_size=4, trust_remote_code=True,
        attn_backend="TRTLLM", moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75,
                                      dtype="auto", enable_block_reuse=False),
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        max_seq_len=max_seq, max_batch_size=64, max_num_tokens=8192)
    print(f"[sm] moe_backend={moe_backend}", flush=True)

    prompts, index = [], []
    for pi, r in enumerate(ref):
        base = list(r["input_ids"])
        sg = [int(x) for x in r["greedy_token_ids"]]
        for t in range(horizon[pi]):
            prompts.append(TokensPrompt(prompt_token_ids=base + sg[:t]))
            index.append((pi, t))
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOPK)
    try:
        outputs = llm.generate(prompts, sampling)
    finally:
        llm.shutdown()

    # Gather TRT's step-t prediction: argmax token + rank/lp of each stop token.
    pred = {pi: {} for pi in range(len(ref))}
    for (pi, t), out in zip(index, outputs):
        gen = out.outputs[0]
        ids = list(gen.token_ids)
        if not ids:
            continue
        trt_tok = int(ids[0])
        lps = gen.logprobs or []
        lpd = {}
        if lps and isinstance(lps[0], dict):
            lpd = {int(k): float(getattr(v, "logprob", v)) for k, v in lps[0].items()}
        # rank within TRT's returned top-K (0 = argmax); None if outside top-K.
        order = sorted(lpd.items(), key=lambda kv: kv[1], reverse=True)
        rank_of = {tid: r for r, (tid, _) in enumerate(order)}
        pred[pi][t] = {
            "trt_tok": trt_tok,
            "stop_rank": {st: rank_of.get(st) for st in STOP_TOKENS},
            "stop_lp": {st: lpd.get(st) for st in STOP_TOKENS},
        }

    print("\n[sm] per-prompt teacher-forced stop-margin "
          "(stop-suppression reported INDEPENDENTLY of the free-run fork):",
          flush=True)
    verdicts = []
    for pi, r in enumerate(ref):
        sg = [int(x) for x in r["greedy_token_ids"]]
        ss = r.get("stop_step")
        H = horizon[pi]
        # INFORMATIONAL only: first step TRT argmax leaves SGLang's winning path.
        # No longer gates the verdict (the old code returned FORK@ here and NEVER
        # reported the stop-rank at ss, masking real stop-suppression).
        first_fork = next((t for t in range(H)
                           if pred[pi].get(t, {}).get("trt_tok") != sg[t]), None)
        # DECISIVE and fork-independent: every prefix is teacher-forced with SGLang's
        # winning tokens, so the context at ss is byte-identical to SGLang's regardless
        # of whether TRT forked earlier -> TRT's stop-token rank there is a clean read.
        stop_ctx_ok = ss is not None and ss < H  # was ss actually inside the horizon?
        at = pred[pi].get(ss, {}) if stop_ctx_ok else {}
        trt_at_stop = at.get("trt_tok")
        stop_tok = sg[ss] if (ss is not None and ss < len(sg)) else None
        rank_at = at.get("stop_rank", {}).get(stop_tok) if stop_tok else None
        lp_at = at.get("stop_lp", {}).get(stop_tok) if stop_tok else None
        trt_is_stop = trt_at_stop in STOP_TOKENS
        # stop-suppression verdict -- does NOT depend on first_fork
        if ss is None:
            v = "NO_SGLANG_STOP"
        elif not stop_ctx_ok:
            # ss fell outside the teacher-force horizon (capped): the stop context
            # was never evaluated, so it must NOT be scored as suppression.
            v = f"STOP_BEYOND_HORIZON(ss{ss}>=H{H})"
        elif trt_is_stop or (rank_at is not None and rank_at == 0):
            v = "AGREES_STOP"      # TRT ranks stop #1 on the winning prefix => noise/fork
        else:
            v = "SUPPRESSES_STOP"  # TRT demotes the stop token => systematic, localizable
        verdicts.append(v)
        fork_note = ("no-fork" if first_fork is None else
                     f"fork@{first_fork}"
                     f"{'<stop' if (ss is not None and first_fork < ss) else '>=stop'}")
        print(f"  idx={r.get('idx','?'):>4} gold={r.get('gold')} n_gen={len(sg)} "
              f"stop_step={ss} stop_tok={stop_tok} H={H} [{fork_note}] "
              f"TRT@stop_argmax={trt_at_stop} TRT_stop_rank={rank_at} "
              f"TRT_stop_lp={None if lp_at is None else round(lp_at,3)} "
              f"SG_stop_lp={_sg_stop_lp(r, ss)} -> {v}", flush=True)

    n_suppress = sum(v == "SUPPRESSES_STOP" for v in verdicts)
    n_agree = sum(v == "AGREES_STOP" for v in verdicts)
    n_beyond = sum(v.startswith("STOP_BEYOND_HORIZON") for v in verdicts)
    n_nostop = sum(v == "NO_SGLANG_STOP" for v in verdicts)
    print(f"\nINKLING_STOPMARGIN suppress={n_suppress} agree={n_agree} "
          f"beyond_horizon={n_beyond} no_sglang_stop={n_nostop} total={len(ref)} "
          f"verdicts={verdicts} moe={moe_backend} cuda_graph={CUDA_GRAPH} "
          f"overlap={OVERLAP}", flush=True)
    print("INKLING_STOPMARGIN_INTERP suppress>0 => systematic stop-suppression "
          "(localizable/fixable, independent of any free-run fork); all agree/no-stop "
          "=> free-run divergence (noise), residual is fp4 kernel-family not a stop "
          "bug; beyond_horizon => raise INKLING_SM_CAP or recapture, do NOT score",
          flush=True)
    return 0


def _sg_stop_lp(r, ss):
    if ss is None:
        return None
    for row in r.get("stop_tail", []):
        if row.get("pos") == ss and row.get("stop_lp") is not None:
            return round(row["stop_lp"], 3)
    return None


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
