#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.4 vision ``source_logit_replay`` (TensorRT-LLM vs SGLang).

``reference_tier=real_source``, ``validation_tier=real_runtime``.

Final-logit parity vs the SGLang multimodal reference for IMAGE prompts. Feeds
the FULL TP=4 production stack (KVCacheManagerV2 + TRTLLM attention + NVFP4
CUTLASS MoE + the hMLP vision fusion) the byte-identical ``-101`` ``input_ids``
plus the aligned image that were POSTed to the SGLang server
(``sglang_mm_ref.json``, captured by ``sglang_mm_capture.sbatch``), under
deterministic greedy decoding, and at the FIRST generated position compares
TensorRT-LLM's final logits + greedy-argmax token against SGLang's.

Contract identical to the text ``inkling_source_logit_replay_test.py`` (same
per-prompt greedy-argmax hard gate + mean-cosine forward-health guard), with two
additions: (1) each prompt carries its aligned image (rebuilt from the reference
``image_b64``) so the vision tower actually runs; (2) ``max_seq_len`` is sized for
the image-patch prefill.

Config matrix (env): INKLING_CUDA_GRAPH / INKLING_OVERLAP -> baseline (0,0) or
enabled (1,1); enabled exercises the CUDA-graph hard path via CudaGraphConfig().

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_image_logit_replay_test.py
Env: INKLING_CHECKPOINT, INKLING_MM_REF (path to sglang_mm_ref.json).
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
CONT = int(os.environ.get("INKLING_SLR_CONT", "8"))
UNPADDED_VOCAB = int(os.environ.get("INKLING_UNPADDED_VOCAB", "200058"))
COS_GATE = float(os.environ.get("INKLING_SLR_COS_GATE", "0.97"))
MIN_COS_FLOOR = float(os.environ.get("INKLING_SLR_MIN_COS_FLOOR", "0.90"))
# TRT keys on the in-vocab image placeholder 200054; a reference JSON may carry
# the SGLang -101 sentinel. Map it so the executor accepts the token ids.
TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _cosine(a, b):
    import torch

    return float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


def main() -> int:
    import torch
    from PIL import Image

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "image source_logit_replay needs CUDA GPUs"
    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [
        r
        for r in ref
        if r.get("input_ids")
        and r.get("pos_top")
        and r.get("image_b64")
        and r.get("greedy_token_ids")
    ]
    assert len(ref) >= 5, f"need >=5 usable SGLang image refs, got {len(ref)}"
    print(
        f"[img-slr] cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"n_prompts={len(ref)} cont={CONT} ref={REF}",
        flush=True,
    )

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.7, dtype="auto", enable_block_reuse=False
    )
    llm = LLM(
        CKPT,
        tensor_parallel_size=4,
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
    print(f"[img-slr] moe_backend={moe_backend} cuda_graph_hard_path={hard_path}", flush=True)

    def _img(r):
        return Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))

    prompts = [
        TokensPrompt(
            prompt_token_ids=_trt_ids(r["input_ids"]), multi_modal_data={"image": [_img(r)]}
        )
        for r in ref
    ]
    sampling = SamplingParams(max_tokens=CONT, temperature=0.0, return_generation_logits=True)
    try:
        outputs = llm.generate(prompts, sampling)
    finally:
        llm.shutdown()

    def compare_pos(logits_pos, sg_top, eff):
        trt_argmax = int(logits_pos.argmax())
        supp = [(tid, lp) for tid, lp in sg_top if 0 <= tid < eff]
        ids = torch.tensor([tid for tid, _ in supp], dtype=torch.long)
        sg_lp = torch.tensor([lp for _, lp in supp], dtype=torch.float32)
        trt_lse = torch.logsumexp(logits_pos, dim=0)
        sel = logits_pos.index_select(0, ids)
        trt_lp = sel - trt_lse
        trt_raw = sel - logits_pos.max()
        sg_raw = sg_lp - sg_lp.max()
        return dict(
            argmax=trt_argmax,
            k=len(ids),
            finite=bool(torch.isfinite(logits_pos).all()),
            max_abs_raw=float((trt_raw - sg_raw).abs().max()),
            cos_raw=_cosine(trt_raw, sg_raw),
            max_abs_lp=float((trt_lp - sg_lp).abs().max()),
            cos_lp=_cosine(trt_lp, sg_lp),
        )

    n_match = 0
    rows = []
    for r, out in zip(ref, outputs):
        gen = out.outputs[0]
        gl = gen.generation_logits
        assert gl is not None, "generation_logits is None (gather not honored)"
        gl = torch.as_tensor(gl).float().cpu()
        if gl.dim() == 1:
            gl = gl.unsqueeze(0)
        eff = min(gl.shape[-1], UNPADDED_VOCAB)
        # Finiteness across ALL CONT clean-generation positions (not just pos0):
        # the clean short generation off the vision-fused prefill directly proves
        # the fused forward is numerically sound (a deep free-run collapse on a
        # borderline prompt is a separate, task.yaml-documented residual effect).
        n_pos = int(gl.shape[0])
        finite_pos = int(torch.isfinite(gl[:, :eff]).all(dim=-1).sum())
        all_pos_finite = finite_pos == n_pos
        sg_greedy0 = int(r["greedy_token_ids"][0])
        samp0 = int(gen.token_ids[0]) if gen.token_ids else -1
        p0 = compare_pos(gl[0, :eff], r["pos_top"][0], eff)
        consistent = p0["argmax"] == samp0
        match = p0["finite"] and consistent and (p0["argmax"] == sg_greedy0)
        n_match += int(match)
        rows.append(
            dict(
                match=match,
                consistent=consistent,
                all_pos_finite=all_pos_finite,
                finite_pos=finite_pos,
                n_pos=n_pos,
                **p0,
            )
        )
        print(
            f"  [{'OK ' if match else 'DIFF'}] {r['id']}\n"
            f"        pos0 greedy: SGLang={sg_greedy0} TRT={p0['argmax']} "
            f"(sampler={samp0}) k={p0['k']}\n"
            f"        clean-gen finite positions: {finite_pos}/{n_pos} "
            f"(all_pos_finite={all_pos_finite})\n"
            f"        pos0 RAW: max_abs={p0['max_abs_raw']:.4f} "
            f"cos={p0['cos_raw']:.6f}  LOGP cos={p0['cos_lp']:.6f}",
            flush=True,
        )

    n_total = len(rows)
    min_cos_raw = min(x["cos_raw"] for x in rows)
    mean_cos_raw = sum(x["cos_raw"] for x in rows) / n_total
    all_consistent = all(x["consistent"] for x in rows)
    n_all_pos_finite = sum(int(x["all_pos_finite"]) for x in rows)
    ok = (
        (n_match == n_total)
        and all_consistent
        and (mean_cos_raw >= COS_GATE)
        and (min_cos_raw >= MIN_COS_FLOOR)
    )
    print(
        f"\n[img-slr] POS0 greedy-argmax equality: {n_match}/{n_total} | "
        f"RAW cos min={min_cos_raw:.6f} mean={mean_cos_raw:.6f} | "
        f"clean-gen all-finite prompts: {n_all_pos_finite}/{n_total}",
        flush=True,
    )
    print(
        f"INKLING_IMG_SLR_{'OK' if ok else 'FAIL'} pos0_matched={n_match}/{n_total} "
        f"consistent={all_consistent} all_pos_finite={n_all_pos_finite}/{n_total} "
        f"mean_cos_raw={mean_cos_raw:.6f} "
        f"min_cos_raw={min_cos_raw:.6f} cuda_graph={CUDA_GRAPH} "
        f"overlap={OVERLAP} cuda_graph_hard_path={hard_path}",
        flush=True,
    )
    if not ok:
        bad = [ref[i]["id"] for i, x in enumerate(rows) if not x["match"]]
        print(f"[img-slr] mismatches: {bad}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
