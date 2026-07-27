#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 / Goal-1.5 LEAN, memory-safe end-to-end validation of the multimodal
``embed_norm`` image-fusion fix (iter 22).

``reference_tier=real_source``, ``validation_tier=real_runtime``.

The full decode probe (``inkling_accounting_decode_probe.py``, max_seq_len=8192,
maxtok=2560, 7 items) is heavier than needed to prove the fix and tripped the
known TP=4 Bus-error-under-memory-pressure (job 5591292 rc=135, "Bus error (7)"
in mgmn_worker, before persisting any item). This driver proves
the ONE thing the fix must change with a MINIMAL footprint: post-fix, TRT's
reasoning now READS the real image content (the table/number values SGLang reads)
instead of saying "image not visible" and hallucinating.

Footprint knobs (all reduce the Bus-error risk vs the probe): a small
``INKLING_MAX_SEQ_LEN`` (default 4096), short ``INKLING_MAXTOK`` (default 512),
``max_batch_size=1``, ``free_gpu_memory_fraction`` (default 0.6), and only
``INKLING_MM_N`` items (default 2). Per-item JSON is persisted incrementally so a
crash still leaves the completed items.

Per item it records + prints:
  * ``says_not_visible``: the reasoning contains an "image not visible" style
    admission (the pre-fix signature; present in ALL 7 pre-fix traces, absent in
    every SGLang trace). Post-fix this must be False.
  * ``shared_numbers``: count of distinct $-amounts / multi-digit numbers that
    appear in BOTH TRT's reasoning and SGLang's reference reasoning for the same
    item -- a quantitative "reads the same image data" signal that is robust
    without hard-coding per-item values.
  * ``parsed`` vs ``gold`` vs ``sglang_parsed``.

Verdict IMAGE_READ_FIXED when NO item says "not visible" AND every item shares
>=3 image numbers with SGLang (the tables have many, hallucinations share ~0).

Baseline only: cuda_graph=off, overlap=off, TP=4, KVCacheManagerV2, TRTLLM attn,
CUTLASS MoE, deterministic (enable_autotuner=False, bs=1).

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_image_read_check.py
"""

import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full",
)
REF = os.environ.get(
    "INKLING_MMMU_REF",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/sglang_mmmu_ref.json",
)
OUT = os.environ.get(
    "INKLING_READ_OUT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/"
    "codes/agent-flow/workspace/inkling-advanced-bringup/results/img_read_check.json",
)
N_ITEMS = int(os.environ.get("INKLING_MM_N", "2"))
MAXTOK = int(os.environ.get("INKLING_MAXTOK", "512"))
MAX_SEQ_LEN = int(os.environ.get("INKLING_MAX_SEQ_LEN", "4096"))
FGMF = float(os.environ.get("INKLING_FGMF", "0.6"))
TP = int(os.environ.get("INKLING_TP", "4"))

TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101

# "image not visible" style admissions -- the pre-fix signature (absent from
# every SGLang reference trace).
NOT_VISIBLE_RE = re.compile(
    r"image (?:is )?not (?:visible|shown|provided|available|displayed)"
    r"|not visible|cannot see the image|can't see the image|image not"
    r"|since (?:the )?image|without (?:the )?image|image (?:1 )?which likely"
    r"|image content not",
    re.IGNORECASE,
)
# $-amounts and multi-digit numbers (>=2 digits, allowing thousands commas).
NUM_RE = re.compile(r"\$?\d[\d,]{1,}(?:\.\d+)?")
ANSWER_RE = re.compile(r"[Aa]nswer\s*:\s*\*?\*?\s*\(?([A-Z])\)?")


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _numbers(text):
    out = set()
    for m in NUM_RE.finditer(text or ""):
        tok = m.group(0).lstrip("$").replace(",", "")
        # keep multi-digit only (single digits are noise: option labels, years).
        if len(tok.replace(".", "")) >= 2:
            out.add(tok)
    return out


def _persist(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def main() -> int:
    import inkling_image_prompts as P
    import inkling_mmmu_harness as H
    import inkling_mmmu_real_align_test as R
    import torch
    from PIL import Image
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "image-read check needs CUDA GPUs"
    sg_by_id = {}
    if os.path.exists(REF):
        with open(REF) as f:
            sg_by_id = {r["id"]: r for r in json.load(f).get("records", [])}

    recs = P.build_prompts(N_ITEMS, with_num_patches=False)
    items_by_id = {it["id"]: it for it in R.load_fixed_items()}
    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    print(
        f"[read-check] tp={TP} n_items={len(recs)} maxtok={MAXTOK} "
        f"max_seq_len={MAX_SEQ_LEN} fgmf={FGMF} moe=CUTLASS baseline "
        f"cuda_graph=off overlap=off deterministic bs=1",
        flush=True,
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=FGMF, dtype="auto", enable_block_reuse=False
    )
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend="CUTLASS"),
        kv_cache_config=kv_cache_config,
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        enable_autotuner=False,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=1,
        max_num_tokens=MAX_SEQ_LEN,
    )
    print(
        "[read-check] LLM built; attn=TRTLLM kv=KVCacheManagerV2 "
        "cuda_graph_hard_path=eager(no-graph)",
        flush=True,
    )

    records = []
    doc = {
        "maxtok": MAXTOK,
        "max_seq_len": MAX_SEQ_LEN,
        "fgmf": FGMF,
        "moe_backend": "CUTLASS",
        "records": records,
    }
    try:
        for r in recs:
            rid = r["id"]
            it = items_by_id[rid]
            gold = it.get("answer")
            options = it.get("options")
            qtype = it.get("question_type") or ("multiple-choice" if options else "open")
            if options:
                index2ans, all_choices = H.build_mc_mapping(options)
            else:
                index2ans, all_choices = None, None
            image = Image.open(io.BytesIO(r["image_bytes"]))
            mm = {"image": [image]}
            input_ids = _trt_ids(r["input_ids"])

            out = llm.generate(
                [TokensPrompt(prompt_token_ids=list(input_ids), multi_modal_data=mm)],
                SamplingParams(max_tokens=MAXTOK, temperature=0.0),
            )[0]
            gen_ids = [int(x) for x in out.outputs[0].token_ids]
            full_text = tok.decode(gen_ids, skip_special_tokens=False)

            says_not_visible = bool(NOT_VISIBLE_RE.search(full_text))
            sgrec = sg_by_id.get(rid, {})
            sg_text = sgrec.get("sglang_text", "")
            trt_nums = _numbers(full_text)
            sg_nums = _numbers(sg_text)
            shared = sorted(trt_nums & sg_nums)
            score, parsed = H.score_sample(full_text, gold, qtype, all_choices, index2ans)
            am = ANSWER_RE.search(full_text)

            rec = {
                "id": rid,
                "config": r["config"],
                "gold": gold,
                "trt_parsed": parsed,
                "trt_score": score,
                "trt_has_answer": bool(am),
                "sglang_parsed": sgrec.get("sglang_parsed"),
                "says_not_visible": says_not_visible,
                "n_shared_numbers": len(shared),
                "shared_numbers": shared[:20],
                "n_trt_numbers": len(trt_nums),
                "n_sg_numbers": len(sg_nums),
                "trt_text_head": full_text[:900],
            }
            records.append(rec)
            _persist(OUT, doc)
            print(
                f"  [{rid:<24}] gold={gold} TRT={parsed}(s={score:.0f}) "
                f"SG={sgrec.get('sglang_parsed')} not_visible={says_not_visible} "
                f"shared_nums={len(shared)}/{len(sg_nums)} "
                f"e.g.={shared[:6]}",
                flush=True,
            )
    finally:
        llm.shutdown()

    n = len(records)
    n_not_visible = sum(1 for r in records if r["says_not_visible"])
    n_reads = sum(1 for r in records if r["n_shared_numbers"] >= 3)
    n_correct = sum(1 for r in records if r["trt_score"] == 1.0)
    fixed = n > 0 and n_not_visible == 0 and n_reads == n
    doc["summary"] = {
        "n": n,
        "n_says_not_visible": n_not_visible,
        "n_reads_image": n_reads,
        "n_correct": n_correct,
    }
    doc["verdict"] = "IMAGE_READ_FIXED" if fixed else "IMAGE_STILL_BLIND"
    _persist(OUT, doc)
    print(
        f"\nINKLING_IMG_READ verdict={doc['verdict']} n={n} "
        f"not_visible={n_not_visible} reads_image={n_reads} correct={n_correct} "
        f"out={OUT}",
        flush=True,
    )
    if fixed:
        print(
            "INTERPRETATION: post-fix TRT READS the image (shares the real "
            "table/number values with the SGLang reference and never says "
            "'image not visible') -- the embed_norm image-fusion fix works; the "
            "iter21 'kernel residual' blocker is overturned.",
            flush=True,
        )
    else:
        print(
            "INTERPRETATION: TRT still not reading the image on >=1 item; "
            "inspect trt_text_head + shared_numbers per item.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("INKLING_IMG_READ FAIL: exception producing evidence", flush=True)
        sys.exit(1)
