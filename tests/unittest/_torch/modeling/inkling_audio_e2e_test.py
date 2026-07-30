#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 / Goal-6.2 end-to-end AUDIO runtime smoke (TensorRT-LLM, TP=4).

``reference_tier=real_source``, ``validation_tier=real_runtime``.

Drives the FULL production multimodal path through the public LLM API on the real
NVFP4 checkpoint at TP=4, for the dMel AUDIO modality -- the audio analogue of
``inkling_image_e2e_test.py`` (Goal 1.4). The audio tower + preprocessing + fail-
loud placeholder contract already passed as unit evidence (Goal 6.1, job 2534915,
``AUDIO_TOWER_CUDA_OK``); this is the missing ``real_runtime`` link that proves the
dMel path fuses into the accepted decoder and decodes on TP=4::

    llm.generate(TokensPrompt(prompt_token_ids=<...200053...>,
                              multi_modal_data={"audio": [waveform]}))
      -> InklingInputProcessor.call_with_token_ids (supports_token_id_mm_expansion)
      -> assemble: expand the single 200053 -> num_frames copies, attach the dMel
         ``dmel_bins`` features (fail-loud on any placeholder/count mismatch)
      -> model engine _prepare_multimodal_indices (isin 200053) + MultimodalParams
      -> InklingForConditionalGeneration.forward: dMel audio tower (InklingAudioModel)
         -> fuse_input_embeds at the placeholder positions -> accepted NVFP4 text
         decoder (KVCacheManagerV2 + TRTLLM attention + CUTLASS MoE) -> greedy decode.

The audio placeholder is the IN-VOCAB ``<|unused_200053|>`` (id 200053) the chat
template emits for an audio content part
(``<|content_audio_input|><|unused_200053|><|audio_end|><|end_message|>``), NOT an
out-of-vocab sentinel (the TRT executor rejects out-of-range ids). Fusion
overwrites those positions with dMel-tower rows -- the placeholder id itself is
never embedded (explicit text/mm indices) -- so a clean finite non-collapsed
generation is itself proof the audio tower ran and fused.

DETERMINISTIC BASELINE (AC54 is strict). AC54 requires the baseline smoke to
produce non-empty decoded output with NO NaN/Inf logits and NO immediate
repeated-token collapse. task.yaml records a residual "~2/10 bs>1 free-run
collapse" for this NVFP4 decoder that is a batched-execution / autotuner
non-determinism artifact -- so this baseline smoke runs the SAME determinism
hygiene the accepted text GP floor and the P0-C determinism gate used
(``enable_autotuner=False`` + ``max_batch_size=1`` + sbatch
``TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1``), which removes that residual. The verdict
is STRICT: EVERY prompt must be finite, non-empty, and non-collapsed (zero
tolerance -- unlike the vision image e2e, which tolerates the documented bs>1
residual). A single non-finite or collapsed prompt FAILS the smoke.

NO ASR-WER, no scored benchmark, no parity gate (human feedback #22/#23 TASK 2):
this is a functional runtime-wiring smoke, mirroring SGLang's own light audio
coverage. Baseline cuda_graph=false/overlap=false is the Stage-6 config; the
script honors INKLING_CUDA_GRAPH/INKLING_OVERLAP so Stage 8 (enabled) can reuse it,
recording the CUDA-graph hard path via CudaGraphConfig().

Run: trtllm-llmapi-launch python tests/unittest/_torch/modeling/inkling_audio_e2e_test.py
Env: INKLING_CHECKPOINT/INKLING_CKPT, INKLING_DETERMINISTIC (default 1),
     INKLING_CUDA_GRAPH, INKLING_OVERLAP, INKLING_MOE_BACKEND, INKLING_TP,
     INKLING_MM_STEPS, INKLING_MM_N_PROMPTS.
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
# all-reduce-tactic / cross-row batched-MoE non-determinism that task.yaml records
# as the "~2/10 bs>1 free-run collapse" residual, so the strict AC54 baseline smoke
# is clean. The enabled Stage-8 revalidation sets INKLING_DETERMINISTIC=0.
DETERMINISTIC = os.environ.get("INKLING_DETERMINISTIC", "1") == "1"
ENABLE_AUTOTUNER = not DETERMINISTIC
TP = int(os.environ.get("INKLING_TP", "4"))
NSTEP = int(os.environ.get("INKLING_MM_STEPS", "40"))
N_PROMPTS = int(os.environ.get("INKLING_MM_N_PROMPTS", "5"))
MAX_BATCH = 1 if DETERMINISTIC else int(os.environ.get("INKLING_MM_MAX_BATCH", "4"))
REPEAT_THRESH = int(os.environ.get("INKLING_MM_REPEAT_THRESH", "12"))
MIN_UNIQUE = int(os.environ.get("INKLING_MM_MIN_UNIQUE", "3"))
REASONING_EFFORT = float(os.environ.get("INKLING_MM_REASONING_EFFORT", "0.9"))

# Fixed SHORT-clip specs (seconds, tone_hz_a, tone_hz_b, concise question) --
# deterministic, distinct, all in the 0.40-0.60 s / 8-12 dMel-frame range so the
# fused audio span stays modest, with brief-answer prompts that encourage a clean
# short generation. Distinct waveform + question each so prompts are not identical.
_CLIP_SPECS = [
    (0.50, 440.0, 1320.0, "Answer in one short sentence: what do you hear?"),
    (0.60, 523.25, 1046.5, "Briefly, is this sound high-pitched or low-pitched?"),
    (0.40, 659.25, 1318.5, "In a few words, describe this audio."),
    (0.55, 392.0, 784.0, "Short answer: is this speech, music, or a tone?"),
    (0.45, 587.33, 1174.66, "Give a one-line caption for this clip."),
    (0.50, 349.23, 698.46, "In one word, what best describes this sound?"),
]


def _synth_waveform(seconds: float, hz_a: float, hz_b: float, sr: int = 16000):
    """A short deterministic two-tone clip (a valid 'short clip', no file I/O)."""
    import numpy as np

    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    wav = 0.1 * np.sin(2 * np.pi * hz_a * t) + 0.05 * np.sin(2 * np.pi * hz_b * t)
    return wav.astype(np.float32)


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


def _render_audio_ids(tok, question, audio_token_id):
    """Render one user turn (audio content part + question) to token ids with
    reasoning_effort injected, exactly like the accepted image/text path. The
    chat template emits one ``<|unused_200053|>`` (audio placeholder) framed by
    ``<|content_audio_input|>`` / ``<|audio_end|>`` for a single audio part."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "audio"}, {"type": "text", "text": question}],
        }
    ]
    try:
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, reasoning_effort=REASONING_EFFORT
        )
    except Exception:  # noqa: BLE001 -- older template without the kwarg
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = _normalize_ids(ids)
    n_aud = sum(1 for t in ids if t == audio_token_id)
    if n_aud != 1:
        raise ValueError(
            f"expected exactly one audio placeholder ({audio_token_id}) in the "
            f"rendered stream, found {n_aud}; the chat template did not emit a "
            f"single audio content block"
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
        DEFAULT_AUDIO_TOKEN_ID,
        InklingAudioPreprocessor,
    )
    from tensorrt_llm.inputs import TokensPrompt
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    assert torch.cuda.is_available(), "Goal-6.2 audio e2e needs CUDA GPUs"

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    aud_id = DEFAULT_AUDIO_TOKEN_ID
    pre = InklingAudioPreprocessor()
    specs = _CLIP_SPECS[: max(N_PROMPTS, 1)]
    recs = []
    for i, (secs, ha, hb, q) in enumerate(specs):
        wav = _synth_waveform(secs, ha, hb)
        ids = _render_audio_ids(tok, q, aud_id)
        n_frames = int(pre.preprocess(wav)["num_frames"][0])
        recs.append(
            {
                "id": f"audio_clip_{i}_{secs}s",
                "input_ids": ids,
                "waveform": wav,
                "n_frames": n_frames,
            }
        )
    assert len(recs) >= min(5, N_PROMPTS), f"need >=5 audio prompts, resolved {len(recs)}"

    print(
        f"[audio-e2e] tp={TP} deterministic={DETERMINISTIC} bs={MAX_BATCH} "
        f"enable_autotuner={ENABLE_AUTOTUNER} cuda_graph={CUDA_GRAPH} overlap={OVERLAP} "
        f"n_prompts={len(recs)} steps={NSTEP} ckpt={CKPT}",
        flush=True,
    )
    for r in recs:
        n_ph = sum(1 for t in r["input_ids"] if t == aud_id)
        print(
            f"  [prompt] {r['id']:<22} len_ids={len(r['input_ids'])} "
            f"n_audio_ph={n_ph} n_frames={r['n_frames']}",
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
        f"[audio-e2e] moe_backend={moe_backend} deterministic={DETERMINISTIC} "
        f"bs={MAX_BATCH} cuda_graph_hard_path={hard_path}",
        flush=True,
    )

    # One TokensPrompt per clip: the in-vocab 200053 placeholder stream verbatim +
    # the raw waveform. The input processor expands the single placeholder into
    # num_frames rows and attaches the dMel bins; the audio tower + fusion overwrite
    # those positions before the accepted text decoder.
    prompts = [
        TokensPrompt(
            prompt_token_ids=list(r["input_ids"]),
            multi_modal_data={"audio": [r["waveform"]]},
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
        # AC54 STRICT: every prompt must be finite, non-empty, and non-collapsed.
        ok = finite and (n_tok > 0) and (not collapse)
        n_ok += int(ok)
        rows.append((r["id"], n_tok, maxrep, uniq, finite, collapse, ok))
        print(
            f"  [{'OK ' if ok else 'BAD'}] {r['id']:<22} n_tok={n_tok} "
            f"maxrep={maxrep} uniq={uniq} finite={finite} collapse={collapse}",
            flush=True,
        )

    n = len(rows)
    # STRICT verdict (AC54): the baseline smoke itself must avoid NaN/Inf logits and
    # immediate repeated-token collapse on EVERY prompt (zero tolerance). Deterministic
    # bs=1 removes the batched free-run residual, so any non-finite/collapsed/empty
    # prompt is a real failure. rows: (id,n_tok,maxrep,uniq,finite,collapse,ok).
    n_collapsed = sum(1 for x in rows if x[5])
    n_nonfinite = sum(1 for x in rows if not x[4])
    n_empty = sum(1 for x in rows if x[1] <= 0)
    bad = [x[0] for x in rows if not x[6]]
    ok_all = (n >= min(5, N_PROMPTS)) and (len(bad) == 0)
    print(
        f"\n[audio-e2e] generated_ok={n_ok}/{n} collapsed={n_collapsed}/{n} "
        f"nonfinite={n_nonfinite}/{n} empty={n_empty}/{n} (STRICT: all three must be 0) "
        f"deterministic={DETERMINISTIC} bs={MAX_BATCH} cuda_graph={CUDA_GRAPH} "
        f"overlap={OVERLAP} cuda_graph_hard_path={hard_path}",
        flush=True,
    )
    print(
        f"INKLING_AUDIO_E2E_{'OK' if ok_all else 'FAIL'} generated_ok={n_ok}/{n} "
        f"collapsed={n_collapsed}/{n} nonfinite={n_nonfinite}/{n} empty={n_empty}/{n} "
        f"deterministic={DETERMINISTIC} bs={MAX_BATCH} tp={TP} cuda_graph={CUDA_GRAPH} "
        f"overlap={OVERLAP} cuda_graph_hard_path={hard_path} steps={NSTEP}",
        flush=True,
    )
    if bad:
        print(f"[audio-e2e] HARD-FAIL bad prompts (nonfinite/collapse/empty): {bad}", flush=True)
    return 0 if ok_all else 1


# --------------------------------------------------------------------------
# pytest wrapper (skips only when no CUDA; the driving sbatch runs the script
# form, which is the real_runtime evidence).
# --------------------------------------------------------------------------
def test_audio_e2e_runtime():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Goal-6.2 audio e2e runtime")
    assert main() == 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        sys.exit(1)
