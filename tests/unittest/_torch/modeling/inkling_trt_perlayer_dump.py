#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HUMAN FEEDBACK #10 -- DIVE (Stage-1 DIVERGENT branch): TRT-side per-layer PREFILL dump.

Drives the per-layer PREFILL activation dump built into modeling_inkling.py
(INKLING_DUMP_PREFILL + INKLING_DUMP_ALLLAYERS=1 + INKLING_DUMP_MODULES=1, gated on
the context-token window INKLING_DUMP_MINTOK..MAXTOK) for the 3 selected fb#10
samples, so the resulting per-layer h_attn / moe_out (answer position) can be
compared against the SGLang forward-hook capture by fb10_dive_layer_sweep.py.

Same DETERMINISTIC config as the accepted Stage-1 probe (byte-identical pos0 proven
3/3 already): TP=4, attn=TRTLLM, CUTLASS, KVCacheManagerV2, cuda_graph=None,
disable_overlap_scheduler=True, enable_autotuner=False, bs=1
(+ sbatch TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1). Each sample is run alone (bs=1) so its
prefill dump lands in a distinct file; the driver then moves that dump (rank 0, the
all-reduced-identical copy the comparison reads) to results/trt_perlayer_fb10/<id>.pt.

The dump file the model writes is ``${INKLING_DUMP_PREFILL}.n<ctx_tok>.rank<r>`` (the
ctx_tok suffix keeps distinct-length prompts in distinct files); we move the newest
.rank0 written during THIS sample's generate, so any earlier warmup-prefill dump in
the same window is ignored.

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_trt_perlayer_dump.py
Env: INKLING_CHECKPOINT, INKLING_MM_REF (sglang_mm_ref_fb10.json), INKLING_DUMP_PREFILL,
INKLING_DUMP_ALLLAYERS=1, INKLING_DUMP_MODULES=1, INKLING_TRT_PERLAYER_OUT (dir).
"""
import base64
import glob
import io
import json
import os
import shutil
import sys

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/hf_data/hf_home/hub/"
    "models--thinkingmachines--Inkling-NVFP4/snapshots/95e51a54d9486020a80d49ae4f9103fb2b3f9686",
)
REF = os.environ.get(
    "INKLING_MM_REF",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/codes/agent-flow/workspace/"
    "inkling-advanced-bringup/results/sglang_mm_ref_fb10.json",
)
OUTDIR = os.environ.get(
    "INKLING_TRT_PERLAYER_OUT",
    "/lustre/fsw/coreai_comparch_trtllm/kleinc/codes/agent-flow/workspace/"
    "inkling-advanced-bringup/results/trt_perlayer_fb10",
)
DUMP_BASE = os.environ.get("INKLING_DUMP_PREFILL")  # must be set (in sbatch env)
TP = int(os.environ.get("INKLING_TP", "4"))
MOE_BACKEND = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")

TRT_IMAGE_TOKEN_ID = 200054
SGLANG_IMAGE_TOKEN_ID = -101


def _trt_ids(input_ids):
    return [TRT_IMAGE_TOKEN_ID if int(t) == SGLANG_IMAGE_TOKEN_ID else int(t) for t in input_ids]


def _list_rank0(base):
    """All rank0 dump files for ``base`` (all-layers .n<ctx>.rank0 + fallback .rank0)."""
    return set(glob.glob(f"{base}.n*.rank0")) | set(glob.glob(f"{base}.rank0"))


def _rm_dumps(base):
    """Delete every rank dump file for ``base`` (all ranks, both namings).

    Called BEFORE each sample's generate so the ONLY dump present afterwards is the
    one this sample just wrote. This is what makes the per-sample selection
    deterministic across the MPI worker boundary: the fixed INKLING_DUMP_PREFILL base
    is shared by all TP workers and by every sample, so an mtime-based 'newest'
    selection (the old _newest_rank0) could return a PRIOR sample's file when the
    filesystem mtime resolution is coarse -- that is the Reviewer iter129 bug
    (Marketing_7__t358 received Math_17__t358's dump.n1118.rank0). Clearing first
    removes the ambiguity entirely.
    """
    for f in set(glob.glob(f"{base}.n*.rank*")) | set(glob.glob(f"{base}.rank*")):
        try:
            os.remove(f)
        except OSError:
            pass


def _ctx_from_name(path):
    b = os.path.basename(path)
    return b.split(".n")[-1].split(".rank")[0] if ".n" in b else "?"


def _md5(path):
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import torch
    from PIL import Image

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "trt per-layer dump needs CUDA GPUs"
    assert DUMP_BASE, "INKLING_DUMP_PREFILL must be set (base path for per-layer dumps)"
    assert os.environ.get("INKLING_DUMP_ALLLAYERS") == "1", "need INKLING_DUMP_ALLLAYERS=1"
    assert os.environ.get("INKLING_DUMP_MODULES") == "1", "need INKLING_DUMP_MODULES=1 (h_attn/moe_out)"
    os.makedirs(OUTDIR, exist_ok=True)

    with open(REF) as f:
        refdoc = json.load(f)
    ref = refdoc["prompts"] if isinstance(refdoc, dict) else refdoc
    ref = [r for r in ref if r.get("input_ids") and r.get("image_b64")]
    assert len(ref) >= 1, f"no usable samples in {REF}"
    print(
        f"[trtdump] tp={TP} moe={MOE_BACKEND} n={len(ref)} base={DUMP_BASE} outdir={OUTDIR}\n"
        f"[trtdump] ALLLAYERS={os.environ.get('INKLING_DUMP_ALLLAYERS')} "
        f"MODULES={os.environ.get('INKLING_DUMP_MODULES')} "
        f"MINTOK={os.environ.get('INKLING_DUMP_MINTOK')} MAXTOK={os.environ.get('INKLING_DUMP_MAXTOK')} "
        f"allreduce_autotune_disabled={os.environ.get('TLLM_DISABLE_ALLREDUCE_AUTOTUNE', '0')} "
        f"ids={[r['id'] for r in ref]}",
        flush=True,
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.6, dtype="auto", enable_block_reuse=False, host_cache_size=0
    )
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=MOE_BACKEND),
        kv_cache_config=kv_cache_config,
        gather_generation_logits=True,
        cuda_graph_config=None,  # baseline eager
        disable_overlap_scheduler=True,
        enable_autotuner=False,  # deterministic
        max_seq_len=4096,
        max_batch_size=1,  # bs=1 -> one prefill per sample
        max_num_tokens=4096,
    )
    print("[trtdump] moe_backend=CUTLASS attn=TRTLLM kv=KVCacheManagerV2 bs=1 cuda_graph=off overlap=off",
          flush=True)

    sampling = SamplingParams(max_tokens=1, temperature=0.0)
    moved = {}
    gen_tokens = {}  # {id: first generated token id} for the DIVE reproduction cross-check
    # Clear any stale dumps from a PRIOR job run before the first sample (the OUTDIR
    # persists across jobs; a leftover file must never be mistaken for a fresh dump).
    _rm_dumps(DUMP_BASE)
    try:
        for r in ref:
            # DETERMINISTIC per-sample selection (Reviewer iter129 fix): clear all
            # dumps, snapshot, generate, then take the file(s) that appeared. With the
            # pre-clear the new set is exactly THIS sample's dump -- no mtime race, no
            # cross-sample copy.
            _rm_dumps(DUMP_BASE)
            before = _list_rank0(DUMP_BASE)
            # feedback #12 fix-1 (equal-input gate, TRT side): the EXACT post-canonicalization
            # token sequence this forward is fed. Stamped into the per-sample .pt below so the
            # analyzer's equal-input gate reads a RUNTIME-grounded TRT id list instead of falling
            # back to the ref (Reviewer iter140: Math_17 used trt_ids_src=ref).
            trt_ids = _trt_ids(r["input_ids"])
            prompt = TokensPrompt(
                prompt_token_ids=trt_ids,
                multi_modal_data={"image": [Image.open(io.BytesIO(base64.b64decode(r["image_b64"])))]},
            )
            out = llm.generate([prompt], sampling)  # triggers prefill -> built-in dump fires
            # First generated token == the answer-position argmax of this prefill. For
            # the decode DIVE this is the teacher-forced decode step, so it MUST equal
            # the Stage-2 confident-mismatch token (expected_trt_top1) -- proving the
            # dumped per-layer activations are the SAME forward that diverged. Zero
            # cost for the prefill ref (which carries no expected_trt_top1).
            gtok = None
            try:
                tids = out[0].outputs[0].token_ids
                gtok = int(tids[0]) if tids else None
            except Exception:  # noqa: BLE001
                gtok = None
            gen_tokens[r["id"]] = gtok
            exp = r.get("expected_trt_top1")
            repro = ("" if exp is None
                     else f" expected_trt_top1={exp} match={gtok == int(exp)}")
            new_files = sorted(_list_rank0(DUMP_BASE) - before)
            if not new_files:
                print(f"  [{r['id']}] WARN no NEW dump file under {DUMP_BASE}.n*.rank0 this "
                      f"generate (check MINTOK/MAXTOK window vs ctx_tok) gen_token={gtok}{repro}",
                      flush=True)
                continue
            # Exactly one in-window prefill per bs=1 generate; if >1 appeared, prefer the
            # largest-ctx (the real prompt over any spurious short forward) and note it.
            if len(new_files) > 1:
                print(f"  [{r['id']}] WARN {len(new_files)} new dumps this generate "
                      f"({[os.path.basename(f) for f in new_files]}); taking largest ctx", flush=True)
            src = max(new_files, key=lambda f: int(_ctx_from_name(f)) if _ctx_from_name(f).isdigit() else -1)
            dst = os.path.join(OUTDIR, f"{r['id']}.pt")
            shutil.copyfile(src, dst)
            ntok = _ctx_from_name(src)
            md5 = _md5(dst)  # md5 of the pure activation dump -> the analyzer's primary/control
            # duplicate-integrity check (iter129) must stay keyed on activations, NOT on the
            # input_ids, so input_ids go to the provenance SIDECAR (below), not into the .pt.
            moved[r["id"]] = dict(src=os.path.basename(src), ctx_tok=ntok, gen_token=gtok,
                                  md5=md5, n_new=len(new_files),
                                  # feedback #12 fix-1: RUNTIME-fed post-canonicalization token
                                  # sequence (the exact list passed to llm.generate() above) so the
                                  # analyzer's equal-input gate is runtime-verified on the TRT side
                                  # (Reviewer iter140: Math_17 used trt_ids_src=ref). Sidecar, not
                                  # in the .pt, to keep the activation md5 integrity check intact.
                                  input_ids=list(trt_ids), n_input_ids=len(trt_ids))
            print(f"  [{r['id']}] dumped ctx_tok={ntok} n_input_ids={len(trt_ids)} gen_token={gtok}"
                  f"{repro} md5={md5[:12]} -> {dst}", flush=True)
    finally:
        llm.shutdown()

    # Persist the generated tokens next to the dumps so the analysis can cross-check
    # reproduction without re-running the model (fb10_decode_dive.py INKLING_TRT_GEN_TOKENS).
    with open(os.path.join(OUTDIR, "gen_tokens.json"), "w") as f:
        json.dump(gen_tokens, f, indent=2)
    # Provenance: per-sample src file, ctx token, md5, generated token -- so the
    # Reviewer can audit which dump each .pt came from without re-running.
    with open(os.path.join(OUTDIR, "dump_provenance.json"), "w") as f:
        json.dump(moved, f, indent=2)

    # INTEGRITY (Reviewer iter129): distinct prompts MUST yield distinct dumps. Two
    # moved .pt files sharing an md5 means the mover copied one sample's dump for
    # another (exactly the bug that invalidated iter129) -- FAIL loudly, do not score.
    by_md5 = {}
    for rid, meta in moved.items():
        by_md5.setdefault(meta["md5"], []).append(rid)
    dupes = {m: ids for m, ids in by_md5.items() if len(ids) > 1}

    # Verify each moved dump actually carries the module split the DIVE needs.
    ok = 0
    for rid, meta in moved.items():
        d = torch.load(os.path.join(OUTDIR, f"{rid}.pt"), map_location="cpu", weights_only=False)
        n_ha = len(d.get("h_attn") or {})
        n_mo = len(d.get("moe_out") or {})
        n_ly = len(d.get("layers") or {})
        good = n_ha >= 1 and n_mo >= 1 and n_ly >= 1
        ok += int(good)
        print(f"  [{rid}] verify layers={n_ly} h_attn={n_ha} moe_out={n_mo} "
              f"ctx_tok={meta['ctx_tok']} md5={meta['md5'][:12]} {'OK' if good else 'BAD'}", flush=True)

    if dupes:
        for m, ids in dupes.items():
            print(f"  DUPLICATE DUMP md5={m[:12]} shared by {ids} -- mover selected the wrong file",
                  flush=True)
        print(f"\nINKLING_TRT_PERLAYER_DUMP FAIL: {len(dupes)} duplicate dump(s); "
              f"distinct prompts must produce distinct dumps outdir={OUTDIR}", flush=True)
        return 2

    all_moved = len(moved) == len(ref)
    print(f"\nINKLING_TRT_PERLAYER_DUMP {'OK' if (ok == len(ref) and all_moved) else 'PARTIAL'} "
          f"dumped={ok}/{len(ref)} unique_md5={len(by_md5)}/{len(moved)} outdir={OUTDIR}", flush=True)
    return 0 if (ok == len(ref) and all_moved) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("INKLING_TRT_PERLAYER_DUMP FAIL: exception", flush=True)
        sys.exit(1)
