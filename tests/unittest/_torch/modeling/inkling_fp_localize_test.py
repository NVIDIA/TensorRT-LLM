#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B2 CUDA-graph decode localizer -- production full-model TP=4 per-layer replay.

Motivation (iter77). The enabled (cuda_graph=on) served path emits stuck-token
garbage ('!!!!!' -- token 0 repeated) while the baseline (cuda_graph=off) path is
correct (served GSM8K 0.91/0.93). crit6 showed prefill logits are IDENTICAL
cg-off-vs-cg-on (pos0 10/10) while the FIRST decode step already diverges (pos1
7/10). Every isolated / reduced-model / TP=1 localizer is graph-clean, so B2 lives
in the full-model TP=4 PRODUCTION stack (TP collectives and/or the production
CUDAGraphRunner). TP=1/TP=2 cannot hold this ~403GB checkpoint (iter76), so B2 can
only be observed at TP=4.

This driver reproduces B2 IN-PROCESS (no server needed) and drives the model-side
capture-safe per-layer fingerprint (env INKLING_FP -> InklingModel._ink_fp): a
persistent GPU buffer written by a device->device copy_ recorded INTO the decode
graph (so it survives capture/replay, unlike the .cpu() dump_sink). It runs a
single fixed prompt (batch=1, the exact B2 smoke condition) free-running for
INKLING_FP_STEPS tokens; the model dumps, per rank per decode step, the residual
after every decoder layer + the final norm.

Because prefill is identical cg-off-vs-cg-on, DECODE STEP 0 receives the SAME input
token in both configs, so its per-layer fingerprints are directly comparable:
inkling_fp_analyze.py loads the cg0 and cg1 dumps and reports (a) the first layer
where cg0 and cg1 diverge on the same rank (B2's origin layer) and (b) cross-rank
residual consistency within each config (the all-reduced residual MUST be identical
across TP ranks; divergence there pins a TP-collective-under-graph bug).

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_fp_localize_test.py
Env: INKLING_CHECKPOINT, INKLING_SGLANG_REF (crit6 capture for input_ids),
     INKLING_TP(=4), INKLING_CUDA_GRAPH(0/1), INKLING_OVERLAP, INKLING_FP(dump base),
     INKLING_FP_STEPS(default 8), INKLING_MOE_BACKEND(default TRTLLM).
"""

import json
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
TP = int(os.environ.get("INKLING_TP", "4"))
STEPS = int(os.environ.get("INKLING_FP_STEPS", "8"))
MOE_BACKEND = os.environ.get("INKLING_MOE_BACKEND", "TRTLLM")


def _max_consec_repeat(ids):
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def main() -> int:
    import torch  # noqa: F401
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401  (registers auto-model)
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "B2 localizer needs CUDA GPUs"
    assert os.environ.get("INKLING_FP"), "INKLING_FP (dump path base) must be set"

    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids")]
    assert ref, "no ref prompt with input_ids"
    # batch=1: the exact B2 smoke condition (1 served chat request collapsed).
    prompt_ids = list(ref[0]["input_ids"])
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    print(f"[fp] tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} moe={MOE_BACKEND} "
          f"steps={STEPS} prompt_len={len(prompt_ids)} fp={os.environ['INKLING_FP']}",
          flush=True)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                    dtype="auto", enable_block_reuse=False)
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=MOE_BACKEND),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        max_seq_len=2048,
        max_batch_size=8,
        max_num_tokens=2048,
    )
    hard_path = "CudaGraphConfig()" if CUDA_GRAPH else "eager(no-graph)"
    print(f"[fp] cuda_graph_hard_path={hard_path}", flush=True)

    try:
        # Free-running batch=1 greedy decode. The model-side hook dumps the
        # per-layer decode fingerprint per rank per step to
        # ${INKLING_FP}.rank{r}.step{s} as a side effect (INKLING_FP set).
        out = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(max_tokens=STEPS, temperature=0.0))[0]
        trt_ids = [int(x) for x in out.outputs[0].token_ids]
        rep = _max_consec_repeat(trt_ids)
        uni = len(set(trt_ids))
        try:
            txt = tok.decode([i for i in trt_ids if i >= 0])
        except Exception:  # noqa: BLE001
            txt = "<decode-err>"
        collapse = (rep >= 8) or (uni < 3)
        print(f"[fp] FREE-RUN out_ids={trt_ids}", flush=True)
        print(f"[fp] FREE-RUN max_repeat={rep} unique={uni} "
              f"{'COLLAPSE' if collapse else 'ok'} text={txt[:80]!r}", flush=True)
        print(f"INKLING_FP_RUN_DONE cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
              f"collapse={collapse} max_repeat={rep} unique={uni} "
              f"cuda_graph_hard_path={hard_path}", flush=True)
    finally:
        llm.shutdown()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
