#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iter90 MMLU B-token-bias localizer (full TP=4 stack, trtllm-gen, baseline cg=off).

iter89 measured MMLU baseline (82.22 vs SGLang 85.66) and found the gap is a
SYSTEMATIC answer-token bias toward 'B' (66% of SGLang-right/TRT-wrong errors are
'B'), NOT diffuse fp4 noise -- so a LOCALIZABLE defect. crit6 (short prompts) passes
argmax 10/10, but the bias appears only on the ~1-2k-token 5-shot MMLU prompts =>
context-length-sensitive ATTENTION path (Inkling relative-position bias RelLogitsProj
+ SWA via inkling_triton score_mod vs SGLang flashinfer fa4). log-scaling tau is a
no-op below n_floor=128k (ruled out by code).

This test feeds the fixture of discriminating prompts (SGLang-right / TRT-wrong /
TRT-answered-'B', built by regen_bbias_prompts.py) through the FULL TP=4 production
model, reads the answer-position logits (generation_logits[0]) over the four answer
tokens ' A'/' B'/' C'/' D', and reports the B-margin. Run TWICE via the sbatch:
  * INKLING_ABLATE_RELBIAS=0 : baseline -- confirm the B-bias reproduces in-process.
  * INKLING_ABLATE_RELBIAS=1 : relative-position bias zeroed in _build_rel_logits.
If zeroing the relative bias collapses the disc B-bias (argmax flips B->gold, mean
B-margin drops) while the controls (both-right cases) stay correct, TRT's relative-
bias implementation is the injector. If the B-bias persists, it lives in the core
QK/PV attention (fp4 GEMM / SWA), not the relative bias.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_bbias_localize_test.py
Env: INKLING_CHECKPOINT, INKLING_BBIAS_FIXTURE, INKLING_ABLATE_RELBIAS (0/1),
     INKLING_MOE_BACKEND (default TRTLLM), INKLING_BBIAS_OUT (per-config json).
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
ABLATE = os.environ.get("INKLING_ABLATE_RELBIAS", "0") == "1"
# The served /v1/completions path adds special tokens by default; TokensPrompt lets
# us control it exactly. Faithfulness is self-checked by the baseline B-reproduction
# rate below; the ablation comparison is relative (same tokenization both configs).
ADD_SPECIAL = os.environ.get("INKLING_BBIAS_ADD_SPECIAL", "1") == "1"
OUT = os.environ.get("INKLING_BBIAS_OUT", "")


def resolve_letter_ids(tok):
    """Token id emitted for each answer letter after 'Answer:' (prefer ' A' form)."""
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
    assert all(v is not None for v in ids.values()), f"unresolved letter ids: {ids}"
    assert len(set(ids.values())) == 4, f"letter ids not distinct: {ids}"
    return ids


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration  # noqa: F401  (registers auto-model)
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "bbias localizer needs CUDA GPUs"
    with open(FIXTURE) as f:
        fx = json.load(f)
    disc, ctrl = fx["discriminating"], fx["controls"]
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    lid = resolve_letter_ids(tok)
    print(f"[bbias] ablate_relbias={ABLATE} add_special={ADD_SPECIAL} "
          f"n_disc={len(disc)} n_ctrl={len(ctrl)} letter_ids={lid} ckpt={CKPT}",
          flush=True)

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "TRTLLM")
    # iter91 MoE-kernel isolation: the CUTLASS *fused* FC2+finalize combine is
    # non-deterministic only ACROSS rows of a batch (iter63: "at nc=1 there is one
    # row so no cross-row fork (correct)"). Running at max_batch_size=1 makes the
    # CUTLASS-fused path deterministic and correct WITHOUT the broken unfused
    # (disable_finalize_fusion=True) path -- so trtllm-gen(bs=1) vs CUTLASS(bs=1) is
    # a clean apples-to-apples fp4-MoE-kernel comparison on the same disc fixture.
    bs = int(os.environ.get("INKLING_BBIAS_BS", "8"))
    llm = LLM(
        CKPT,
        tensor_parallel_size=4,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.75,
                                      dtype="auto", enable_block_reuse=False),
        gather_generation_logits=True,   # TP=4: gather full-vocab logits to rank 0
        cuda_graph_config=None,          # baseline cg=off (B-bias is present here)
        disable_overlap_scheduler=True,
        max_seq_len=2048,
        max_batch_size=bs,
        max_num_tokens=2048,
    )
    print(f"[bbias] moe_backend={moe_backend} max_batch_size={bs} built; "
          f"running prefill", flush=True)

    recs = [dict(r, kind="disc") for r in disc] + \
           [dict(r, kind="ctrl") for r in ctrl]
    prompts = [TokensPrompt(prompt_token_ids=tok.encode(
        r["prompt"], add_special_tokens=ADD_SPECIAL)) for r in recs]
    sampling = SamplingParams(max_tokens=1, temperature=0.0,
                              return_generation_logits=True)
    try:
        outputs = llm.generate(prompts, sampling)
    finally:
        llm.shutdown()

    per = []
    for r, out in zip(recs, outputs):
        gl = out.outputs[0].generation_logits
        assert gl is not None, "generation_logits None (gather not honored)"
        gl0 = torch.as_tensor(gl).float().cpu()
        if gl0.dim() == 2:
            gl0 = gl0[0]
        ll = {L: float(gl0[lid[L]]) for L in "ABCD"}
        pred = max(ll, key=ll.get)                       # argmax over {A,B,C,D}
        full_arg = int(gl0.argmax())
        full_letter = tok.decode([full_arg]).strip()
        gold = r["gold"]
        per.append(dict(idx=r["idx"], subject=r["subject"], kind=r["kind"],
                        gold=gold, pred_abcd=pred,
                        full_letter=full_letter if full_letter in "ABCD" else "?",
                        correct=(pred == gold),
                        b_margin=ll["B"] - ll[gold],
                        gold_margin=ll[gold] - max(ll[c] for c in "ABCD" if c != gold),
                        logits=ll))

    def agg(kind):
        rows = [p for p in per if p["kind"] == kind]
        n = len(rows)
        return dict(
            n=n,
            argmaxB=sum(p["pred_abcd"] == "B" for p in rows),
            correct=sum(p["correct"] for p in rows),
            mean_bmargin=round(sum(p["b_margin"] for p in rows) / n, 4) if n else 0.0,
            full_argmaxB=sum(p["full_letter"] == "B" for p in rows))

    d, c = agg("disc"), agg("ctrl")
    tag = 1 if ABLATE else 0
    # disc: baseline should reproduce B (argmaxB high, correct low, mean_bmargin>0);
    # ctrl: correct should stay high under either config.
    print(f"\nINKLING_BBIAS ablate_relbias={tag} "
          f"disc_n={d['n']} disc_argmaxB={d['argmaxB']} disc_nowcorrect={d['correct']} "
          f"disc_mean_bmargin={d['mean_bmargin']} disc_fullargmaxB={d['full_argmaxB']} "
          f"ctrl_n={c['n']} ctrl_correct={c['correct']} ctrl_mean_bmargin={c['mean_bmargin']} "
          f"add_special={ADD_SPECIAL} moe={moe_backend} bs={bs}", flush=True)
    # human-readable interpretation hint
    if not ABLATE:
        faith = d['argmaxB'] / d['n'] if d['n'] else 0
        print(f"[bbias] BASELINE reproduction fidelity: disc argmax=B on "
              f"{d['argmaxB']}/{d['n']} ({faith:.0%}); if low (<0.6), tokenization "
              f"likely differs from the served run (flip INKLING_BBIAS_ADD_SPECIAL).",
              flush=True)
    else:
        print(f"[bbias] ABLATED (rel-bias=0): disc argmax=B {d['argmaxB']}/{d['n']}, "
              f"disc now-correct {d['correct']}/{d['n']}, ctrl still-correct "
              f"{c['correct']}/{c['n']}. Compare vs baseline: a large drop in disc "
              f"argmaxB + rise in disc-correct with controls preserved => relative "
              f"bias is the B-bias injector.", flush=True)

    if OUT:
        with open(OUT, "w") as f:
            json.dump(dict(ablate_relbias=ABLATE, add_special=ADD_SPECIAL,
                           moe_backend=moe_backend, bs=bs,
                           letter_ids=lid, disc=d, ctrl=c, per=per), f, indent=1)
        print(f"[bbias] wrote {OUT}", flush=True)
    print(f"=== INKLING_BBIAS_DONE ablate_relbias={tag} rc=0 ===", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
