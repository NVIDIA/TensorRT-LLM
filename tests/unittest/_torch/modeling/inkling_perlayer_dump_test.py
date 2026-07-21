#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iter94 per-layer TRT-vs-SGLang localizer -- TRT side.

iter93 proved (direct TRT-vs-SGLang, byte-identical tokens) the MMLU 'B'-bias is
TRT-specific and lives in the BF16 path (delta +1.53 on disc, 44/44 same sign).
This teacher-forces the top-delta discriminating prompts through the FULL TP=4
production stack and dumps the answer-position residual stream after EVERY one of
the 66 decoder layers (INKLING_DUMP_ALLLAYERS via the extended INKLING_DUMP_PREFILL
hook in modeling_inkling.InklingModel.forward). compare_perlayer.py then joins these
with the SGLang forward-hook residuals (capture_sglang_perlayer.py) and computes the
per-layer cosine trajectory to decide SMOOTH accumulation (distributed / architecture-
level) vs a SHARP jump at one layer (a fixable bf16 module divergence from SGLang).

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_perlayer_dump_test.py
Env: INKLING_CHECKPOINT, INKLING_BBIAS_FIXTURE, INKLING_PERLAYER_IDX (csv),
     INKLING_PERLAYER_OUTDIR, INKLING_MOE_BACKEND (default TRTLLM).
"""
import json
import os
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")
FIXTURE = os.environ.get(
    "INKLING_BBIAS_FIXTURE",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-bringup/results/bbias_prompts.json")
IDX = os.environ.get("INKLING_PERLAYER_IDX", "4791,20,4752,86,4801")
OUTDIR = os.environ.get(
    "INKLING_PERLAYER_OUTDIR",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-bringup/results/perlayer_trt")


def resolve_letter_ids(tok):
    ids = {}
    for L in "ABCD":
        chosen = None
        for cand in (" " + L, L):
            for t in tok.encode(cand, add_special_tokens=False):
                if tok.decode([t]).strip() == L:
                    chosen = t
                    break
            if chosen is not None:
                break
        ids[L] = chosen
    return ids


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401  (registers auto-model)
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "per-layer dump needs CUDA GPUs"
    os.makedirs(OUTDIR, exist_ok=True)
    want = [int(x) for x in IDX.split(",") if x.strip()]
    fx = json.load(open(FIXTURE))
    by_idx = {r["idx"]: dict(r, kind="disc") for r in fx["discriminating"]}
    by_idx.update({r["idx"]: dict(r, kind="ctrl") for r in fx["controls"]})
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    lid = resolve_letter_ids(tok)
    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "TRTLLM")
    print(f"[perlayer-trt] idx={want} letter_ids={lid} moe={moe_backend} "
          f"outdir={OUTDIR}", flush=True)

    # INKLING_DUMP_PREFILL / INKLING_DUMP_MINTOK / INKLING_DUMP_MAXTOK /
    # INKLING_DUMP_ALLLAYERS are set in the sbatch env so EVERY TP worker sees them
    # (post-launch os.environ writes in this launcher do NOT reach the model
    # workers). Each prompt's prefill dumps to <base>.n<ctx_tok>.rank<r>; the token
    # window excludes the ~max_num_tokens warmup prefill.
    dump_base = os.environ.get("INKLING_DUMP_PREFILL")
    assert dump_base, "INKLING_DUMP_PREFILL must be set in the sbatch env"
    llm = LLM(
        CKPT, tensor_parallel_size=4, trust_remote_code=True,
        attn_backend="TRTLLM", moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75,
                                      dtype="auto", enable_block_reuse=False),
        gather_generation_logits=True, cuda_graph_config=None,
        disable_overlap_scheduler=True,
        max_seq_len=2560, max_batch_size=1, max_num_tokens=4096)
    print("[perlayer-trt] LLM built; teacher-forcing prompts", flush=True)

    summary = []
    try:
        for idx in want:
            r = by_idx.get(idx)
            if r is None:
                print(f"[perlayer-trt] idx {idx} not in fixture; skip", flush=True)
                continue
            ids = tok.encode(r["prompt"], add_special_tokens=True)
            ntok = len(ids)
            out = llm.generate(
                [TokensPrompt(prompt_token_ids=ids)],
                SamplingParams(max_tokens=1, temperature=0.0,
                               return_generation_logits=True))[0]
            gl = out.outputs[0].generation_logits
            gl0 = torch.as_tensor(gl).float().cpu()
            if gl0.dim() == 2:
                gl0 = gl0[0]
            ll = {L: float(gl0[lid[L]]) for L in "ABCD"}
            gold = r["gold"]
            pred = max(ll, key=ll.get)
            # the model wrote <base>.n<ctx_tok>.rank0; ctx_tok == ntok for a single
            # teacher-forced prompt.
            resid = f"{dump_base}.n{ntok}.rank0"
            dumped = os.path.exists(resid)
            summary.append(dict(idx=idx, subject=r["subject"], kind=r["kind"],
                                gold=gold, pred_abcd=pred, n_ids=ntok,
                                b_margin=round(ll["B"] - ll[gold], 4),
                                logits={L: round(ll[L], 4) for L in "ABCD"},
                                resid_file=resid if dumped else None))
            print(f"[perlayer-trt] idx={idx:>5d} gold={gold} pred={pred} "
                  f"b_margin={ll['B']-ll[gold]:+.3f} ntok={ntok} "
                  f"resid={'OK' if dumped else 'MISSING'}", flush=True)
    finally:
        llm.shutdown()

    outp = os.path.join(OUTDIR, "trt_perlayer_summary.json")
    json.dump(dict(letter_ids=lid, idx=want, moe_backend=moe_backend,
                   per=summary), open(outp, "w"), indent=1)
    n_ok = sum(1 for s in summary if s["resid_file"])
    print(f"\nINKLING_TRT_PERLAYER n_prompts={len(summary)} n_resid_ok={n_ok} "
          f"-> {outp}", flush=True)
    print(f"=== INKLING_TRT_PERLAYER_DONE rc={0 if n_ok else 3} ===", flush=True)
    return 0 if n_ok else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("=== INKLING_TRT_PERLAYER_DONE rc=1 ===", flush=True)
        sys.exit(1)
