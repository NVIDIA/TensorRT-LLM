#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-7 / Goal-7.2 end-to-end VIDEO (multi-frame image) runtime smoke (TP=4).

``reference_tier=real_source``, ``validation_tier=real_runtime``.

Drives the FULL production multimodal path through the public LLM API on the real
NVFP4 checkpoint at TP=4 for the VIDEO-as-multi-frame-images modality -- the video
analogue of ``inkling_image_e2e_test.py`` (one image) and ``inkling_audio_e2e_test.py``
(one audio clip). Inkling has NO separate video tower: a video is decoded to frames,
a subset is sampled (:func:`sample_video_as_images`, SGLang-parity), and each sampled
frame is fed as an ordinary ``<image>`` through the SAME hMLP vision tower::

    llm.generate(TokensPrompt(prompt_token_ids=<... K x 200054 ...>,
                              multi_modal_data={"image": [frame_0, ..., frame_{K-1}]}))
      -> InklingInputProcessor.call_with_token_ids (supports_token_id_mm_expansion)
      -> assemble: expand each of the K <image> (200054) placeholders into that
         frame's num_patches copies, attach the K frames' vision_patches_bthwc
         (fail-loud on any placeholder/count mismatch)
      -> model engine _prepare_multimodal_indices (isin 200054) + MultimodalParams
      -> InklingForConditionalGeneration.forward: hMLP vision tower over ALL frames'
         patches -> fuse_input_embeds at the placeholder positions -> accepted NVFP4
         text decoder (KVCacheManagerV2 + TRTLLM attention + CUTLASS MoE) -> greedy.

The per-frame placeholder is the IN-VOCAB ``<|unused_200054|>`` (id 200054) the chat
template emits for each image content part -- one ``<|content_image|><|unused_200054|>
<|end_message|>`` per frame -- NOT an out-of-vocab sentinel (the TRT executor rejects
out-of-range ids). Fusion overwrites those positions with vision rows, so a clean
finite non-collapsed generation is itself proof the multi-frame vision path ran and
fused.

DETERMINISTIC BASELINE (AC59 is strict, mirroring AC54 audio). AC59 requires the
baseline multi-frame smoke to produce non-empty decoded output with NO NaN/Inf
logits and NO immediate repeated-token collapse. task.yaml records a residual
"~2/10 bs>1 free-run collapse" for this NVFP4 decoder that is a batched-execution /
autotuner non-determinism artifact -- so this baseline smoke runs the SAME
determinism hygiene the accepted text GP floor, the P0-C determinism gate, and the
Stage-6 audio smoke used (``enable_autotuner=False`` + ``max_batch_size=1`` + sbatch
``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``), which removes that residual. The verdict is
STRICT: EVERY prompt must be finite, non-empty, and non-collapsed (zero tolerance).

NO Video-MME, MVBench, or scored benchmark, no parity gate (human feedback #22/#23
TASK 2): this is a functional runtime-wiring smoke mirroring SGLang's own light video
coverage. Baseline cuda_graph=false/overlap=false is the Stage-7 config; the script
honors INKLING_CUDA_GRAPH/INKLING_OVERLAP so Stage 8 (enabled) can reuse it, recording
the CUDA-graph hard path via CudaGraphConfig().

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_video_e2e_test.py
Env: INKLING_CHECKPOINT/INKLING_CKPT, INKLING_DETERMINISTIC (default 1),
     INKLING_CUDA_GRAPH, INKLING_OVERLAP, INKLING_MOE_BACKEND, INKLING_TP,
     INKLING_MM_STEPS, INKLING_MM_N_PROMPTS, INKLING_VIDEO_FRAMES.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CKPT = os.environ.get(
    "INKLING_CKPT",
    os.environ.get(
        "INKLING_CHECKPOINT",
        "/lustre/fsw/coreai_comparch_trtllm/kleinc/hf_data/hf_home/hub/"
        "models--thinkingmachines--Inkling-NVFP4/snapshots/"
        "95e51a54d9486020a80d49ae4f9103fb2b3f9686",
    ),
)
CUDA_GRAPH = os.environ.get("INKLING_CUDA_GRAPH", "0") == "1"
OVERLAP = os.environ.get("INKLING_OVERLAP", "1" if CUDA_GRAPH else "0") == "1"
# DETERMINISTIC baseline (default ON): enable_autotuner=False + max_batch_size=1 +
# (sbatch) TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1 -- removes the autotuner-tactic /
# all-reduce-tactic / cross-row batched-MoE non-determinism task.yaml records as the
# "~2/10 bs>1 free-run collapse" residual, so the strict AC59 baseline smoke is clean.
# The enabled Stage-8 revalidation sets INKLING_DETERMINISTIC=0.
DETERMINISTIC = os.environ.get("INKLING_DETERMINISTIC", "1") == "1"
ENABLE_AUTOTUNER = not DETERMINISTIC
TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_MM_STEPS", "40"))
N_PROMPTS = int(os.environ.get("INKLING_MM_N_PROMPTS", "5"))
MAX_BATCH = 1 if DETERMINISTIC else int(os.environ.get("INKLING_MM_MAX_BATCH", "4"))
# Frames kept per clip (after sampling). Small so the fused multi-frame span stays
# modest at bs=1; >=2 so it genuinely exercises the MULTI-frame path (not one image).
N_FRAMES = int(os.environ.get("INKLING_VIDEO_FRAMES", "4"))
REPEAT_THRESH = int(os.environ.get("INKLING_MM_REPEAT_THRESH", "12"))
MIN_UNIQUE = int(os.environ.get("INKLING_MM_MIN_UNIQUE", "3"))
REASONING_EFFORT = float(os.environ.get("INKLING_MM_REASONING_EFFORT", "0.9"))

# Fixed SHORT-clip specs (raw_frames, avg_fps, tint_seed, concise question). Each
# clip is a distinct deterministic moving pattern; sample_video_as_images samples it
# down to N_FRAMES so the real video sampling path is exercised, not raw frames.
_CLIP_SPECS = [
    (8, 8.0, 11, "Answer in one short sentence: what changes across these frames?"),
    (7, 7.0, 29, "Briefly, does the object move left or right?"),
    (9, 9.0, 47, "In a few words, describe this short clip."),
    (6, 6.0, 63, "Short answer: is there motion in this video?"),
    (8, 8.0, 83, "Give a one-line caption for this clip."),
    (7, 7.0, 97, "In one word, what best describes this video?"),
]

_FRAME_H = int(os.environ.get("INKLING_VIDEO_FRAME_H", "48"))
_FRAME_W = int(os.environ.get("INKLING_VIDEO_FRAME_W", "64"))


def _synth_clip(n_frames: int, seed: int):
    """A short deterministic clip: ``n_frames`` distinct RGB frames (a moving box on
    a per-clip tinted background), returned as PIL RGB images (a valid 'video',
    no codec/file I/O)."""
    import numpy as np
    from PIL import Image

    h, w = _FRAME_H, _FRAME_W
    frames = []
    for f in range(n_frames):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = (37 * seed) % 256
        arr[:, :, 1] = (17 * f + 5 * seed) % 256
        # a moving bright box marks temporal progression across frames
        x0 = int((w - 12) * f / max(1, n_frames - 1))
        arr[h // 3 : h // 3 + 12, x0 : x0 + 12, 2] = 255
        frames.append(Image.fromarray(arr, mode="RGB"))
    return frames


def _normalize_ids(ids):
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) > 0 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def _render_video_ids(tok, question, n_frames, image_token_id):
    """Render one user turn (``n_frames`` image content parts + question) to token
    ids with reasoning_effort injected, exactly like the accepted image/text path.
    The chat template emits one ``<|content_image|><|unused_200054|><|end_message|>``
    per image part, so a K-frame clip renders K image placeholders (one span per
    frame)."""
    content = [{"type": "image"} for _ in range(n_frames)]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    try:
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, reasoning_effort=REASONING_EFFORT
        )
    except Exception:  # noqa: BLE001 -- older template without the kwarg
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = _normalize_ids(ids)
    n_img = sum(1 for t in ids if t == image_token_id)
    if n_img != n_frames:
        raise ValueError(
            f"expected exactly {n_frames} image placeholders ({image_token_id}) in "
            f"the rendered stream (one per frame), found {n_img}; the chat template "
            f"did not emit one image content block per frame"
        )
    return ids


def _max_consec_repeat(ids):
    best = cur = 0
    prev = None
    for x in ids:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def main() -> int:
    import torch
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.models.modeling_inkling import (
        InklingForConditionalGeneration,  # noqa: F401  (registers auto-model)
    )
    from tensorrt_llm._torch.models.modeling_inkling_vision import (
        DEFAULT_IMAGE_TOKEN_ID,
        DecodedVideo,
        sample_video_as_images,
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "Goal-7.2 video e2e needs CUDA GPUs"

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    img_id = DEFAULT_IMAGE_TOKEN_ID
    specs = _CLIP_SPECS[: max(N_PROMPTS, 1)]
    recs = []
    for i, (raw_frames, avg_fps, seed, q) in enumerate(specs):
        clip = _synth_clip(raw_frames, seed)
        video = DecodedVideo(clip, avg_fps=avg_fps)
        # Sample the clip down to N_FRAMES via the real video sampling path.
        frames = sample_video_as_images(
            video, desired_fps=max(1, N_FRAMES), max_frames=N_FRAMES
        )
        n_frames = len(frames)
        ids = _render_video_ids(tok, q, n_frames, img_id)
        recs.append(
            {
                "id": f"video_clip_{i}_{raw_frames}f",
                "input_ids": ids,
                "frames": frames,
                "n_frames": n_frames,
            }
        )
    assert len(recs) >= min(5, N_PROMPTS), f"need >=5 video prompts, resolved {len(recs)}"

    print(
        f"[video-e2e] tp={TP} deterministic={DETERMINISTIC} bs={MAX_BATCH} "
        f"enable_autotuner={ENABLE_AUTOTUNER} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"n_prompts={len(recs)} frames_per_clip={N_FRAMES} steps={NSTEP} ckpt={CKPT}",
        flush=True,
    )
    for r in recs:
        n_ph = sum(1 for t in r["input_ids"] if t == img_id)
        print(
            f"  [prompt] {r['id']:<20} len_ids={len(r['input_ids'])} "
            f"n_image_ph={n_ph} n_frames={r['n_frames']}",
            flush=True,
        )

    moe_backend = os.environ.get("INKLING_MOE_BACKEND", "CUTLASS")
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.7, dtype="auto", enable_block_reuse=False
    )
    llm = LLM(
        CKPT,
        tensor_parallel_size=TP,
        trust_remote_code=True,
        attn_backend="TRTLLM",
        moe_config=MoeConfig(backend=moe_backend),
        kv_cache_config=kv_cache_config,
        gather_generation_logits=True,
        cuda_graph_config=CudaGraphConfig() if CUDA_GRAPH else None,
        disable_overlap_scheduler=not OVERLAP,
        enable_autotuner=ENABLE_AUTOTUNER,
        max_seq_len=4096,
        max_batch_size=MAX_BATCH,
        max_num_tokens=4096,
    )
    hard_path = "CudaGraphConfig()" if CUDA_GRAPH else "eager(no-graph)"
    print(
        f"[video-e2e] moe_backend={moe_backend} deterministic={DETERMINISTIC} "
        f"bs={MAX_BATCH} cuda_graph_hard_path={hard_path}",
        flush=True,
    )

    # One TokensPrompt per clip: the K in-vocab 200054 placeholders verbatim + the K
    # sampled frames. The input processor expands each placeholder into that frame's
    # num_patches rows and attaches the frames' patches; the vision tower + fusion
    # overwrite those positions before the accepted text decoder.
    prompts = [
        TokensPrompt(
            prompt_token_ids=list(r["input_ids"]),
            multi_modal_data={"image": list(r["frames"])},
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
        # AC59 STRICT: every prompt must be finite, non-empty, and non-collapsed.
        ok = finite and (n_tok > 0) and (not collapse)
        n_ok += int(ok)
        rows.append((r["id"], n_tok, maxrep, uniq, finite, collapse, ok))
        print(
            f"  [{'OK ' if ok else 'BAD'}] {r['id']:<20} n_tok={n_tok} "
            f"maxrep={maxrep} uniq={uniq} finite={finite} collapse={collapse}",
            flush=True,
        )

    n = len(rows)
    # STRICT verdict (AC59): the baseline multi-frame smoke itself must avoid NaN/Inf
    # logits and immediate repeated-token collapse on EVERY prompt (zero tolerance).
    # Deterministic bs=1 removes the batched free-run residual, so any non-finite/
    # collapsed/empty prompt is a real failure. rows: (id,n_tok,maxrep,uniq,finite,
    # collapse,ok).
    n_collapsed = sum(1 for x in rows if x[5])
    n_nonfinite = sum(1 for x in rows if not x[4])
    n_empty = sum(1 for x in rows if x[1] <= 0)
    bad = [x[0] for x in rows if not x[6]]
    ok_all = (n >= min(5, N_PROMPTS)) and (len(bad) == 0)
    print(
        f"\n[video-e2e] generated_ok={n_ok}/{n} collapsed={n_collapsed}/{n} "
        f"nonfinite={n_nonfinite}/{n} empty={n_empty}/{n} (STRICT: all three must be 0) "
        f"deterministic={DETERMINISTIC} bs={MAX_BATCH} cuda_graph={CUDA_GRAPH} "
        f"overlap={OVERLAP} cuda_graph_hard_path={hard_path}",
        flush=True,
    )
    print(
        f"INKLING_VIDEO_E2E_{'OK' if ok_all else 'FAIL'} generated_ok={n_ok}/{n} "
        f"collapsed={n_collapsed}/{n} nonfinite={n_nonfinite}/{n} empty={n_empty}/{n} "
        f"deterministic={DETERMINISTIC} bs={MAX_BATCH} tp={TP} frames_per_clip={N_FRAMES} "
        f"cuda_graph={CUDA_GRAPH} overlap={OVERLAP} cuda_graph_hard_path={hard_path} "
        f"steps={NSTEP}",
        flush=True,
    )
    if bad:
        print(f"[video-e2e] HARD-FAIL bad prompts (nonfinite/collapse/empty): {bad}", flush=True)
    return 0 if ok_all else 1


# --------------------------------------------------------------------------
# pytest wrapper (skips only when no CUDA; the driving sbatch runs the script
# form, which is the real_runtime evidence).
# --------------------------------------------------------------------------
def test_video_e2e_runtime():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Goal-7.2 video e2e runtime")
    assert main() == 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
