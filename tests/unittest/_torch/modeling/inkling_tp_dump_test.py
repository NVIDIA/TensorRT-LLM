#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic STEP A: TP=4 runtime PREFILL activation dump.

The crit8 LLM API smoke dispatches and produces on-vocab tokens, but the generated
text is GARBAGE (job 5462395/5460547: "The capital of France is" -> gibberish) --
a full-model correctness bug that the on-vocab check and the (all TP=1) focused
replays masked. Prime suspect: TP=4 weight sharding / all-reduce, which no focused
test exercises.

This runs ONE fixed prompt through the real production TP=4 runtime (identical LLM
config to the smoke) with INKLING_DUMP_PREFILL set, so InklingModel.forward writes
this batch's prefill activations (input_ids, position_ids, embed_norm, per-layer
hidden 0..7, final_norm) to `${INKLING_DUMP_PREFILL}.rank{r}`. STEP B
(inkling_tp_compare_test.py, TP=1) then replays the SAME input_ids through the
validated reduced-model reference and reports the first divergent layer.

Run (TP=4, under MPI):
    INKLING_DUMP_PREFILL=/abs/path/prefill.pt \
    trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_tp_dump_test.py
"""

import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")

PROMPT = "The capital of France is"


def main() -> int:
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "the TP dump needs CUDA GPUs"
    assert os.environ.get("INKLING_DUMP_PREFILL"), \
        "set INKLING_DUMP_PREFILL=<path> so the model dumps prefill activations"
    print(f"[tp-dump] ckpt={CKPT} dump={os.environ['INKLING_DUMP_PREFILL']}",
          flush=True)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.75,
                                    dtype="auto",
                                    enable_block_reuse=False)
    # MoE parallelization + backend are env-configurable so the localizer can A/B
    # them without code churn (default = today's intermediate-TP CUTLASS path).
    #   INKLING_MOE_BACKEND: CUTLASS (default) | TRTLLM | CUTEDSL ...
    #   INKLING_MOE_EP: 0 (default, intermediate-TP moe_tp=tp) | 4 (expert-parallel
    #     moe_ep=4/moe_tp=1 -> each rank computes WHOLE experts over the full
    #     intermediate, matching TP=1 per-expert, instead of intermediate-slicing).
    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    moe_ep = int(os.environ.get("INKLING_MOE_EP", "0"))
    print(f"[tp-dump] moe_backend={moe_backend} moe_ep={moe_ep}", flush=True)
    llm_kwargs = dict(
        tensor_parallel_size=4,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        max_seq_len=2048,
        max_batch_size=8,
        max_num_tokens=2048,
    )
    if moe_ep > 0:
        llm_kwargs["moe_expert_parallel_size"] = moe_ep
        llm_kwargs["moe_tensor_parallel_size"] = 1
    # Baseline config (no CUDA graph, no overlap): keep the forward eager so the
    # dump reflects the plain runtime path, not capture/replay.
    llm = LLM(CKPT, **llm_kwargs)
    # Generate a short continuation (not just 1 token) so the printed text is a
    # coherence check: after the MoE TP all-reduce fix "The capital of France is"
    # should continue sensibly (e.g. " Paris").
    sampling = SamplingParams(max_tokens=20, temperature=0.0)
    try:
        outputs = llm.generate([PROMPT], sampling)
    finally:
        llm.shutdown()

    o = outputs[0].outputs[0]
    print(f"[tp-dump] prompt={PROMPT!r} gen_token={list(o.token_ids)} "
          f"text={o.text!r}",
          flush=True)
    print("INKLING_TP_DUMP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
