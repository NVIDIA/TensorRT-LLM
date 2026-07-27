# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aligned MMMU harness oracle for Inkling TRT-LLM vs SGLang (Goal 1.1 / M1a).

This module encodes the *aligned MMMU harness contract* so that TensorRT-LLM and
the SGLang reference score MMMU comparably: the **same** prompt rendering, the
**same** ``<image>`` placement, the **same** greedy (temperature=0) decoding
intent, the **same** image preprocessing, and the **same** answer
extraction/scoring. M1a requires proving this alignment *before* any accuracy
number is reported (a score from misaligned harnesses is not evidence).

It is a pure ``numpy``/``PIL``/stdlib oracle -- it does NOT import
``tensorrt_llm``, ``sglang`` or ``torch`` at import time -- so the
harness-alignment canary runs on CPU with no TRT-LLM rebuild.

Faithfulness:
  * ``render_mmmu_prompt`` / ``build_mc_mapping`` / ``parse_multi_choice_response``
    / ``parse_open_response`` / ``eval_open`` are verbatim ports of the SGLang
    reference ``simple_eval_mmmu_vlm.py`` (the requested serving comparand).
  * ``scaled_image_dimensions`` / ``preprocess_patches`` reimplement the SGLang
    ``InklingImageProcessor`` (``multimodal/inkling/image_processing.py``) in
    plain float32 numpy -- numerically identical to its numba kernel. The HF
    transformers ``InklingImageProcessor`` produces the same values (it pads
    ``fill_value=-1.0`` *before* rescale(x1/255)+normalize, which equals SGLang's
    ``PAD_NORM``); see ``mmmu_alignment_notes.md``.

Read-only references (on disk, cited by file):
  SGLang eval  : python/sglang/test/simple_eval_mmmu_vlm.py
  SGLang image : python/sglang/srt/multimodal/inkling/image_processing.py
  SGLang mm proc: python/sglang/srt/multimodal/processors/inkling.py
  HF image proc: transformers/.../models/inkling/image_processing_inkling.py
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Image normalization constants -- byte-exact copies of the SGLang comparand
# (image_processing.py:15-18). NOTE: SGLang's IMAGE_STD differs from
# transformers' OPENAI_CLIP_STD only in the 7th+ significant digit
# (0.2613026 vs 0.26130258, 0.2757771 vs 0.27577711); we track the SGLang
# values because SGLang is the requested serving comparand.
# ---------------------------------------------------------------------------
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.2613026, 0.2757771], dtype=np.float32)
PAD_RAW_VALUE = np.float32(-1.0 / 255.0)
PAD_NORM = (np.full((3,), PAD_RAW_VALUE, dtype=np.float32) - IMAGE_MEAN) / IMAGE_STD

# One text-hidden token is emitted per vision patch (hMLP folds each patch's
# interior to channel depth), so placeholder_count == num_patches. The default
# geometry from the checkpoint vision_config / SGLang processor:
DEFAULT_PATCH_SIZE = 40
DEFAULT_TEMPORAL_PATCH_SIZE = 2  # static image is temporally duplicated (T=2)
DEFAULT_RESCALE_IMAGE_FRAC = 2.0
DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE = 2048

# The multiple-choice instruction block appended after the lettered options.
# Verbatim from simple_eval_mmmu_vlm.py:156-162 (leading "\n", no trailing "\n").
MC_INSTRUCTION = (
    "\nAnswer the following multiple-choice question. "
    "The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) "
    "where LETTER is one of the options. "
    "Think step by step before answering."
)


# ===========================================================================
# Prompt rendering + multiple-choice mapping (SGLang simple_eval_mmmu_vlm.py)
# ===========================================================================
def build_mc_mapping(options: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map option list -> (index2ans, all_choices). Ports ``_build_mc_mapping``
    (simple_eval_mmmu_vlm.py:81-91)."""
    index2ans: Dict[str, str] = {}
    all_choices: List[str] = []
    ch = ord("A")
    for opt in options:
        letter = chr(ch)
        index2ans[letter] = opt
        all_choices.append(letter)
        ch += 1
    return index2ans, all_choices


def render_mmmu_prompt(question: str, options: Optional[Sequence[str]]) -> Tuple[str, str]:
    """Render the MMMU textual prompt exactly as SGLang does
    (simple_eval_mmmu_vlm.py:150-164).

    Returns ``(prompt_text, question_type)`` where ``question_type`` is
    ``"multiple-choice"`` or ``"open"``. The ``<image>`` is NOT inlined in this
    text: SGLang passes it as a separate content block and the server renders one
    placeholder per image (expanded to num_patches). Alignment therefore requires
    the same text here *and* the same one-placeholder-per-image convention.
    """
    prompt_text = f"{question}\n"
    if options:
        letters = [chr(ord("A") + i) for i in range(len(options))]
        for letter, opt in zip(letters, options):
            prompt_text += f"{letter}. {opt}\n"
        prompt_text += MC_INSTRUCTION
        return prompt_text, "multiple-choice"
    prompt_text += "\nAnswer: "
    return prompt_text, "open"


# ===========================================================================
# Answer extraction / scoring -- verbatim ports of SGLang
# (simple_eval_mmmu_vlm.py:337-475). These are the SHARED scorer both stacks
# must use for the numbers to be comparable.
# ===========================================================================
def parse_multi_choice_response(response: str, all_choices: List[str], index2ans: dict) -> str:
    """Verbatim port of ``_parse_multi_choice_response`` (lines 337-381)."""
    # First, look for explicit "Answer: X" pattern (last occurrence)
    answer_matches = re.findall(r"[Aa]nswer\s*:\s*\*?\*?\s*\(?([A-Z])\)?", response)
    if answer_matches:
        candidate = answer_matches[-1]
        if candidate in all_choices:
            return candidate

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    candidates: List[str] = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for idx, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(idx)
    if not candidates:
        return all_choices[0]
    if len(candidates) == 1:
        return candidates[0]
    starts = []
    for can in candidates:
        pos = response.rfind(f"({can})")
        if pos == -1:
            pos = response.rfind(f" {can} ")
        if pos == -1 and index2ans.get(can):
            pos = response.lower().rfind(index2ans[can].lower())
        starts.append(pos)
    return candidates[int(max(range(len(starts)), key=lambda i: starts[i]))]


def _check_is_number(s: str) -> bool:
    """Port of ``_check_is_number`` (lines 384-389)."""
    try:
        float(s.replace(",", ""))
        return True
    except Exception:
        return False


def _normalize_str(s: str):
    """Port of ``_normalize_str`` (lines 392-401)."""
    s = s.strip()
    if _check_is_number(s):
        s = s.replace(",", "")
        try:
            v = round(float(s), 2)
            return [v]
        except Exception:
            return [s.lower()]
    return [s.lower()] if len(s) > 1 else [" " + s, s + " "]


def _extract_numbers(s: str) -> List[str]:
    """Port of ``_extract_numbers`` (lines 404-414)."""
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (
        re.findall(pattern_commas, s)
        + re.findall(pattern_scientific, s)
        + re.findall(pattern_simple, s)
    )


def parse_open_response(response: str) -> List[str]:
    """Verbatim port of ``_parse_open_response`` (lines 417-456)."""

    def get_key_subresponses(resp: str) -> List[str]:
        resp = resp.strip().strip(".").lower()
        subs = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        keys = []
        for i, s in enumerate(subs):
            cands = [*indicators]
            if i == len(subs) - 1:
                cands.append("=")
            shortest = None
            for ind in cands:
                if ind in s:
                    part = s.split(ind)[-1].strip()
                    if not shortest or len(part) < len(shortest):
                        shortest = part
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", ":", "'"]:
                keys.append(shortest)
        return keys or [resp]

    key_resps = get_key_subresponses(response)
    pred_list = key_resps.copy()
    for r in key_resps:
        pred_list.extend(_extract_numbers(r))
    out = []
    for x in pred_list:
        out.extend(_normalize_str(x))
    return list(dict.fromkeys(out))


def eval_open(gold, preds: List[str]) -> bool:
    """Verbatim port of ``_eval_open`` (lines 459-474)."""
    if isinstance(gold, list):
        norm_answers = []
        for ans in gold:
            norm_answers.extend(_normalize_str(ans))
    else:
        norm_answers = _normalize_str(gold)
    for p in preds:
        if isinstance(p, str):
            for na in norm_answers:
                if isinstance(na, str) and na in p:
                    return True
        else:
            if p in norm_answers:
                return True
    return False


def score_sample(
    response_text: str,
    gold,
    question_type: str,
    all_choices: Optional[List[str]],
    index2ans: Optional[dict],
) -> Tuple[float, str]:
    """Shared scoring both stacks apply to a generated response
    (simple_eval_mmmu_vlm.py:233-250). Returns ``(score, extracted_answer)``."""
    if question_type == "multiple-choice" and all_choices and index2ans:
        pred = parse_multi_choice_response(response_text or "", all_choices, index2ans)
        return (1.0 if (gold is not None and pred == gold) else 0.0), pred
    parsed_list = parse_open_response(response_text or "")
    score = 1.0 if (gold is not None and eval_open(gold, parsed_list)) else 0.0
    return score, ", ".join(map(str, parsed_list))


# ===========================================================================
# Image preprocessing -- numerically-identical port of the SGLang
# InklingImageProcessor (multimodal/inkling/image_processing.py).
# ===========================================================================
def scaled_image_dimensions(
    width: int,
    height: int,
    rescale_image_frac: Optional[float] = DEFAULT_RESCALE_IMAGE_FRAC,
    rescale_image_max_upscaled_long_edge: Optional[int] = DEFAULT_RESCALE_MAX_UPSCALED_LONG_EDGE,
) -> Tuple[int, int]:
    """Long-edge scale ``(width, height)`` before patching. Verbatim logic from
    ``_scaled_image_dimensions`` (image_processing.py:46-75). Half-away-from-zero
    rounding (``floor(v*ratio + 0.5)``), cap never shrinks."""
    if rescale_image_frac is None:
        return width, height
    long_edge = max(width, height)
    if long_edge == 0:
        return width, height
    target_long_edge = float(long_edge) * rescale_image_frac
    if rescale_image_max_upscaled_long_edge is not None:
        effective_cap = max(rescale_image_max_upscaled_long_edge, long_edge)
        target_long_edge = min(target_long_edge, float(effective_cap))
    ratio = target_long_edge / float(long_edge)
    if ratio == 1.0:
        return width, height
    import math

    def scale(value: int) -> int:
        return max(1, math.floor(float(value) * ratio + 0.5))

    return scale(width), scale(height)


def patch_grid(height: int, width: int, patch_size: int = DEFAULT_PATCH_SIZE):
    """Return ``(num_patches, nph, npw)`` for a (height, width) image.
    ``nph = ceil(H/P)``, ``npw = W // P + 1`` (the asymmetric +1 width padding;
    image_processing.py:117-118,167-168)."""
    nph = (height + patch_size - 1) // patch_size
    npw = width // patch_size + 1
    return nph * npw, nph, npw


def preprocess_patches(
    arr: np.ndarray, patch_size: int = DEFAULT_PATCH_SIZE
) -> Tuple[np.ndarray, int, int, int]:
    """Turn an ``(H, W, 3)`` uint8 array into normalized float32 patches
    ``(num_patches, patch_size, patch_size, 3)``. Numerically identical to
    ``_fill_patches_numba`` (image_processing.py:106-137): interior pixels are
    ``(uint8/255 - mean)/std`` in float32, out-of-bounds pixels are ``PAD_NORM``.

    Returns ``(patches, num_patches, nph, npw)``. Caller does long-edge scaling
    (``scaled_image_dimensions``) and temporal expansion (``to_bthwc``).
    """
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8 image, got shape {arr.shape}")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    num_patches, nph, npw = patch_grid(h, w, patch_size)

    patches = np.empty((num_patches, patch_size, patch_size, 3), dtype=np.float32)
    inv255 = np.float32(1.0) / np.float32(255.0)
    for k in range(num_patches):
        i = k // npw
        j = k - i * npw
        y0, x0 = i * patch_size, j * patch_size
        y1, x1 = min(y0 + patch_size, h), min(x0 + patch_size, w)
        patches[k, :, :, :] = PAD_NORM  # pad first, overwrite interior below
        if y1 > y0 and x1 > x0:
            region = arr[y0:y1, x0:x1, :].astype(np.float32) * inv255
            region = (region - IMAGE_MEAN) / IMAGE_STD
            patches[k, : y1 - y0, : x1 - x0, :] = region
    return patches, num_patches, nph, npw


def to_bthwc(
    patches: np.ndarray, temporal_patch_size: int = DEFAULT_TEMPORAL_PATCH_SIZE
) -> np.ndarray:
    """Expand ``(P, patch, patch, 3)`` -> ``(P, T, patch, patch, 3)`` by temporal
    duplication (image_processing.py:174-179). Mirrors the ``view(...,1,...)`` +
    ``expand(...,T,...)`` in float32 (bf16 cast is applied at model-load time)."""
    p = patches[:, None, :, :, :]
    return np.broadcast_to(p, (p.shape[0], temporal_patch_size, *p.shape[2:])).copy()


# ===========================================================================
# Harness-alignment canary: dump prompt / media / parsed-answer records for a
# fixed identical-item set so TRT and SGLang can be compared item-for-item.
# ===========================================================================
def canary_record(
    item: dict,
    patch_size: int = DEFAULT_PATCH_SIZE,
    apply_scale: bool = False,
    response_text: Optional[str] = None,
) -> dict:
    """Build one aligned harness record for a fixed MMMU-style item.

    ``item`` keys: ``id``, ``question``, ``options`` (list or None), ``answer``,
    and ``image`` (an ``(H, W, 3)`` uint8 array). If ``response_text`` is given
    (e.g. a fixed synthetic response, or a real generation once available), the
    shared scorer is applied so both stacks' parsed answers can be compared.
    """
    prompt, qtype = render_mmmu_prompt(item["question"], item.get("options"))
    if item.get("options"):
        index2ans, all_choices = build_mc_mapping(item["options"])
    else:
        index2ans, all_choices = None, None

    arr = np.asarray(item["image"])
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if apply_scale:
        sw, sh = scaled_image_dimensions(w, h)
    else:
        sw, sh = w, h
    num_patches, nph, npw = patch_grid(sh, sw, patch_size)

    rec = {
        "id": item["id"],
        "prompt": prompt,
        "question_type": qtype,
        "all_choices": all_choices,
        "index2ans": index2ans,
        "gold": item.get("answer"),
        "raw_hw": (h, w),
        "scaled_hw": (sh, sw),
        "grid": (nph, npw),
        "num_patches": num_patches,
        # hMLP emits one token per patch, so the number of <image> placeholder
        # tokens the renderer must expand to MUST equal num_patches.
        "placeholder_count": num_patches,
        "media_shape": (num_patches, DEFAULT_TEMPORAL_PATCH_SIZE, patch_size, patch_size, 3),
    }
    if response_text is not None:
        score, extracted = score_sample(
            response_text, item.get("answer"), qtype, all_choices, index2ans
        )
        rec["response_text"] = response_text
        rec["parsed_answer"] = extracted
        rec["score"] = score
    return rec
