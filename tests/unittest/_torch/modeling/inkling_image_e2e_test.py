#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.4 end-to-end vision RUNTIME proof (TensorRT-LLM, TP=4).

``reference_tier=real_source``, ``validation_tier=real_runtime``.

This drives the FULL production multimodal path through the public LLM API on the
real NVFP4 checkpoint at TP=4 -- the runtime evidence the isolated iter-8 fusion
scatter test could not give:

    llm.generate(TokensPrompt(prompt_token_ids=<200054 stream>,
                              multi_modal_data={"image": [PIL]}))
      -> InklingInputProcessor.call_with_token_ids (supports_token_id_mm_expansion)
      -> assemble: expand 200054 -> num_patches, attach vision_patches_bthwc
      -> model engine _prepare_multimodal_indices (isin 200054) + MultimodalParams
      -> InklingForConditionalGeneration.forward: hMLP tower -> fuse_input_embeds
         at the placeholder positions -> accepted NVFP4 text decoder
         (KVCacheManagerV2 + TRTLLM attention + CUTLASS MoE) -> greedy decode.

The image placeholder is the IN-VOCAB ``<|unused_200054|>`` (id 200054) the chat
template emits, NOT the SGLang-internal -101 (which the TRT executor rejects as
an out-of-range token id). Fusion overwrites those positions with vision embeds,
so a clean finite non-collapsed generation is itself proof the vision tower ran
(the placeholder id is never embedded -- explicit text/mm indices).

What this asserts (per config):
  * >=5 fixed real MMMU image prompts each generate >=32 tokens (or a clean EOS
    with >0 tokens); every NON-collapsed prompt has FINITE generation logits;
  * free-run collapse is bounded by task.yaml's documented ~2/10 residual
    (<= ceil(0.2 * n) prompts). task.yaml records that residual free-run collapse
    for this NVFP4 text tower is "TRT-specific, backend-independent,
    accuracy-neutral (not a gate)"; the multimodal path reuses that decoder. A
    non-collapsed non-finite generation is still a HARD fail (a real numerical
    bug, not the documented residual);
  * both baseline (cuda_graph=false, overlap=false) and enabled
    (cuda_graph=true via CudaGraphConfig(), overlap=true) configs run, and the
    enabled run records the CUDA-graph hard path.

This is a coarse runtime-wiring smoke for Goal 1.4. The AUTHORITATIVE correctness
proof is the SGLang-reference ``source_logit_replay`` / ``generation_parity``
COMPARISON in the sibling files ``inkling_image_logit_replay_test.py`` /
``inkling_image_generation_parity_test.py`` (they consume the served SGLang
capture and are immune to free-run divergence via single-step / teacher forcing).

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_image_e2e_test.py
Env: INKLING_CHECKPOINT/INKLING_CKPT, INKLING_CUDA_GRAPH, INKLING_OVERLAP,
     MMMU_ALIGN_CACHE (warm cache), SGLANG_PY.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "1" if CUDA_GRAPH else "0") == "1"
TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_MM_STEPS", "40"))  # >=32 required
N_PROMPTS = int(os.environ.get("INKLING_MM_N_PROMPTS", "5"))
REPEAT_THRESH = int(os.environ.get("INKLING_MM_REPEAT_THRESH", "12"))
MIN_UNIQUE = int(os.environ.get("INKLING_MM_MIN_UNIQUE", "3"))


def _max_consec_repeat(ids):
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def main() -> int:
    import inkling_image_prompts as P
    import torch
    from PIL import Image

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "Goal-1.4 e2e needs CUDA GPUs"
    ckpt = P.CKPT
    recs = P.build_prompts(N_PROMPTS)
    assert len(recs) >= min(5, N_PROMPTS), f"need >=5 image prompts, resolved {len(recs)}"
    print(
        f"[mm-e2e] tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"n_prompts={len(recs)} steps={NSTEP} ckpt={ckpt}",
        flush=True,
    )
    for r in recs:
        n101 = sum(1 for t in r["input_ids"] if t == P.IMAGE_TOKEN_ID)
        print(
            f"  [prompt] {r['id']:<26} len_ids={len(r['input_ids'])} "
            f"n_sentinel={n101} num_patches={r['num_patches']}",
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
    print(f"[mm-e2e] moe_backend={moe_backend} cuda_graph_hard_path={hard_path}", flush=True)

    # One TokensPrompt per image prompt: the in-vocab 200054 placeholder stream
    # (P.IMAGE_TOKEN_ID) verbatim + the aligned image. The input processor expands
    # the single placeholder into num_patches rows and attaches the vision patches.
    prompts = [
        TokensPrompt(
            prompt_token_ids=list(r["input_ids"]),
            multi_modal_data={"image": [Image.open(_bio(r["image_bytes"]))]},
        )
        for r in recs
    ]
    sampling = SamplingParams(max_tokens=NSTEP, temperature=0.0, return_generation_logits=True)
    try:
        outputs = llm.generate(prompts, sampling)
    finally:
        llm.shutdown()

    n_ok = 0
    rows = []
    for r, out in zip(recs, outputs):
        gen = out.outputs[0]
        ids = [int(t) for t in (gen.token_ids or [])]
        gl = gen.generation_logits
        finite = True
        if gl is not None:
            glt = torch.as_tensor(gl).float()
            finite = bool(torch.isfinite(glt).all())
        n_tok = len(ids)
        maxrep = _max_consec_repeat(ids)
        uniq = len(set(ids))
        collapse = (maxrep >= REPEAT_THRESH) or (uniq < MIN_UNIQUE)
        # >=32 tokens OR a clean EOS-terminated shorter answer (>0 tokens, no
        # collapse). A zero-length or collapsed generation is a failure.
        length_ok = (n_tok >= 32) or (0 < n_tok and not collapse)
        ok = finite and length_ok and (not collapse)
        n_ok += int(ok)
        rows.append((r["id"], n_tok, maxrep, uniq, finite, collapse, ok))
        print(
            f"  [{'OK ' if ok else 'BAD'}] {r['id']:<26} n_tok={n_tok} "
            f"maxrep={maxrep} uniq={uniq} finite={finite} "
            f"collapse={collapse}",
            flush=True,
        )

    n = len(rows)
    # task.yaml documents a residual ~2/10 free-run collapse for this NVFP4 text
    # tower ("TRT-specific, backend-independent, accuracy-neutral, not a gate");
    # the multimodal path reuses that decoder. Tolerate up to ceil(0.2 * n)
    # free-run collapses, but require EVERY non-collapsed prompt to be finite and
    # non-empty -- a non-collapsed non-finite generation is a real numerical bug,
    # NOT the documented residual, and hard-fails. rows: (id,n_tok,maxrep,uniq,
    # finite,collapse,ok).
    collapse_tol = max(1, (n + 4) // 5)  # ceil(0.2 * n); == 1 for n=5
    n_collapsed = sum(1 for x in rows if x[5])
    noncollapse_bad = [x[0] for x in rows if (not x[5]) and not (x[4] and x[1] > 0)]
    ok_all = (
        (n >= min(5, N_PROMPTS)) and (n_collapsed <= collapse_tol) and (len(noncollapse_bad) == 0)
    )
    print(
        f"\n[mm-e2e] generated_ok={n_ok}/{n} collapsed={n_collapsed}/{n} "
        f"(tol={collapse_tol}, task.yaml ~2/10 residual) cuda_graph={CUDA_GRAPH} "
        f"overlap={OVERLAP} cuda_graph_hard_path={hard_path}",
        flush=True,
    )
    print(
        f"INKLING_MM_E2E_{'OK' if ok_all else 'FAIL'} generated_ok={n_ok}/{n} "
        f"collapsed={n_collapsed}/{n} collapse_tol={collapse_tol} "
        f"tp={TP} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"cuda_graph_hard_path={hard_path} steps={NSTEP}",
        flush=True,
    )
    if n_collapsed:
        print(
            f"[mm-e2e] within-residual free-run collapse (diagnostic): "
            f"{[x[0] for x in rows if x[5]]}",
            flush=True,
        )
    if noncollapse_bad:
        print(f"[mm-e2e] HARD-FAIL non-collapsed bad prompts: {noncollapse_bad}", flush=True)
    return 0 if ok_all else 1


def _bio(b):
    import io

    return io.BytesIO(b)


# --------------------------------------------------------------------------
# pytest wrapper (skips only when no CUDA; the driving sbatch runs the script
# form, which is the real_runtime evidence).
# --------------------------------------------------------------------------
def test_image_e2e_runtime():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Goal-1.4 e2e runtime")
    assert main() == 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
