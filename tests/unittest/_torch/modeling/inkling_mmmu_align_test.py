# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M1a MMMU harness-alignment canary (CPU) for Inkling TRT-LLM vs SGLang.

Proves the *deterministic* pieces of the aligned MMMU harness contract before
any accuracy number is reported (Goal 1.1 / M1a):

  * prompt rendering matches the SGLang MMMU format exactly;
  * the shared answer-extraction/scoring reproduces SGLang's decisions on a
    hand-derived battery (values derived directly from the SGLang regex/logic);
  * image preprocessing produces the correct patch grid (the asymmetric ``+1``
    width padding and the default 2x long-edge upscaling) and byte-exact float32
    normalization / PAD_NORM;
  * placeholder_count == num_patches (one text-hidden token per vision patch).

Guarded live cross-checks import the on-disk SGLang reference and assert
byte-identity; they SKIP (not fail) when SGLang's serving deps (datasets/numba)
are absent, so the core assertions -- which encode the SGLang contract with
citations -- always run on plain numpy.

Runnable two ways:
  * ``pytest inkling_mmmu_align_test.py``
  * ``python inkling_mmmu_align_test.py``  (no pytest needed; CPU/container)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling_mmmu_harness as H  # noqa: E402

SGLANG_ROOT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/codes/sglang/python"
)


# ---------------------------------------------------------------------------
# 1. Prompt rendering matches SGLang simple_eval_mmmu_vlm.py exactly
# ---------------------------------------------------------------------------
def test_prompt_rendering_multiple_choice():
    prompt, qtype = H.render_mmmu_prompt("What is shown?", ["A cat", "A dog"])
    expected = (
        "What is shown?\n"
        "A. A cat\n"
        "B. A dog\n"
        "\nAnswer the following multiple-choice question. "
        "The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) "
        "where LETTER is one of the options. "
        "Think step by step before answering."
    )
    assert qtype == "multiple-choice"
    assert prompt == expected, f"prompt mismatch:\n{prompt!r}\n!=\n{expected!r}"


def test_prompt_rendering_open():
    prompt, qtype = H.render_mmmu_prompt("Compute the derivative.", None)
    assert qtype == "open"
    assert prompt == "Compute the derivative.\n\nAnswer: "


def test_build_mc_mapping():
    index2ans, all_choices = H.build_mc_mapping(["cat", "dog", "bird"])
    assert all_choices == ["A", "B", "C"]
    assert index2ans == {"A": "cat", "B": "dog", "C": "bird"}


# ---------------------------------------------------------------------------
# 2. Shared answer extraction reproduces SGLang decisions (hand-derived from
#    simple_eval_mmmu_vlm.py:337-381).
# ---------------------------------------------------------------------------
def test_parse_multi_choice_battery():
    choices = ["A", "B", "C", "D"]
    i2a = {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}
    cases = [
        # (response, expected_letter, note)
        ("Some reasoning... Answer: C", "C", "explicit Answer: X"),
        ("Answer: A\nmore\nAnswer: D", "D", "last Answer: wins"),
        ("The correct choice is (B).", "B", "bracketed letter after punct strip"),
        ("B", "B", "bare standalone letter"),
        ("answer: b but Answer: D", "D", "regex requires uppercase capture"),
        ("No idea", "A", "no signal -> first choice fallback"),
    ]
    for resp, want, note in cases:
        got = H.parse_multi_choice_response(resp, choices, i2a)
        assert got == want, f"[{note}] {resp!r} -> {got!r}, want {want!r}"

    # option-text match path (needs >5 words and no letter signal)
    i2a2 = {"A": "a cat", "B": "a dog"}
    got = H.parse_multi_choice_response("i really think it is a dog for sure", ["A", "B"], i2a2)
    assert got == "B", f"option-text match -> {got!r}, want 'B'"


def test_parse_open_response_and_eval():
    preds = H.parse_open_response("The final answer is 42.")
    assert 42.0 in preds, f"parsed {preds!r} should contain 42.0"
    assert H.eval_open("42", preds) is True
    assert H.eval_open("43", preds) is False


# ---------------------------------------------------------------------------
# 3. Image preprocessing: patch grid, scaling, normalization, PAD_NORM
# ---------------------------------------------------------------------------
def test_patch_grid_formula():
    # nph = ceil(H/40); npw = W//40 + 1  (asymmetric +1 on width)
    assert H.patch_grid(80, 80, 40) == (2 * 3, 2, 3)  # ceil(80/40)=2, 80//40+1=3
    assert H.patch_grid(100, 80, 40) == (3 * 3, 3, 3)  # ceil(100/40)=3
    assert H.patch_grid(40, 40, 40) == (1 * 2, 1, 2)  # exact multiple still +1 wide
    assert H.patch_grid(1, 1, 40) == (1 * 1, 1, 1)


def test_scaled_image_dimensions():
    # long_edge 300 -> x2 = 600 (< cap 2048); returns (scale(w), scale(h))
    assert H.scaled_image_dimensions(300, 100) == (600, 200)
    # already-large: long_edge 2000 -> target 4000 capped to max(2048,2000)=2048;
    # ratio 1.024; floor(2000*1.024+0.5)=2048, floor(1000*1.024+0.5)=1024
    assert H.scaled_image_dimensions(2000, 1000) == (2048, 1024)
    # frac=None disables scaling
    assert H.scaled_image_dimensions(123, 45, rescale_image_frac=None) == (123, 45)


def test_preprocess_normalization_and_pad():
    # 40x80 solid mid-gray image: 1 row x 3 cols of patches (80//40+1=3), so the
    # 3rd column is entirely out-of-bounds -> PAD_NORM, cols 0..1 are interior.
    val = np.uint8(128)
    arr = np.full((40, 80, 3), val, dtype=np.uint8)
    patches, num_patches, nph, npw = H.preprocess_patches(arr, patch_size=40)
    assert (num_patches, nph, npw) == (3, 1, 3)
    assert patches.shape == (3, 40, 40, 3)
    assert patches.dtype == np.float32

    inv255 = np.float32(1.0) / np.float32(255.0)
    interior_expected = (np.float32(val) * inv255 - H.IMAGE_MEAN) / H.IMAGE_STD
    # patch 0 and 1 are fully interior
    np.testing.assert_array_equal(patches[0], np.broadcast_to(interior_expected, (40, 40, 3)))
    np.testing.assert_array_equal(patches[1], np.broadcast_to(interior_expected, (40, 40, 3)))
    # patch 2 (x_base=80) is entirely out of bounds -> PAD_NORM everywhere
    np.testing.assert_array_equal(patches[2], np.broadcast_to(H.PAD_NORM, (40, 40, 3)))


def test_partial_patch_pad_boundary():
    # 40x50 -> npw = 50//40+1 = 2; patch 1 covers x in [40,80) but image width 50,
    # so columns 0..9 interior, 10..39 PAD_NORM.
    arr = np.full((40, 50, 3), np.uint8(200), dtype=np.uint8)
    patches, num_patches, nph, npw = H.preprocess_patches(arr, patch_size=40)
    assert (num_patches, nph, npw) == (2, 1, 2)
    inv255 = np.float32(1.0) / np.float32(255.0)
    interior = (np.float32(200) * inv255 - H.IMAGE_MEAN) / H.IMAGE_STD
    # in patch 1, first 10 cols interior, rest padded
    np.testing.assert_array_equal(patches[1][:, :10, :], np.broadcast_to(interior, (40, 10, 3)))
    np.testing.assert_array_equal(patches[1][:, 10:, :], np.broadcast_to(H.PAD_NORM, (40, 30, 3)))


def test_to_bthwc_temporal_expand():
    arr = np.full((40, 40, 3), np.uint8(10), dtype=np.uint8)
    patches, num_patches, _, _ = H.preprocess_patches(arr, patch_size=40)
    bthwc = H.to_bthwc(patches)
    assert bthwc.shape == (num_patches, 2, 40, 40, 3)
    np.testing.assert_array_equal(bthwc[:, 0], bthwc[:, 1])  # temporal duplicate


# ---------------------------------------------------------------------------
# 4. placeholder_count == num_patches invariant + canary record
# ---------------------------------------------------------------------------
def test_canary_record_placeholder_invariant():
    item = {
        "id": "canary_mc_0",
        "question": "Which animal is in the image?",
        "options": ["cat", "dog", "bird", "fish"],
        "answer": "B",
        "image": np.full((80, 120, 3), np.uint8(100), dtype=np.uint8),
    }
    rec = H.canary_record(item, response_text="I conclude. Answer: B")
    # 80x120 @40: nph=2, npw=120//40+1=4 -> 8 patches
    assert rec["num_patches"] == 8
    assert rec["placeholder_count"] == rec["num_patches"], (
        "hMLP emits one token per patch; placeholder count must equal num_patches"
    )
    assert rec["media_shape"] == (8, 2, 40, 40, 3)
    assert rec["question_type"] == "multiple-choice"
    assert rec["parsed_answer"] == "B" and rec["score"] == 1.0

    # with default 2x scaling the grid grows deterministically
    rec_scaled = H.canary_record(item, apply_scale=True)
    assert rec_scaled["scaled_hw"] == (160, 240)  # 80x120 long-edge 120 -> x2
    assert rec_scaled["num_patches"] == 4 * 7  # ceil(160/40)=4, 240//40+1=7


# ---------------------------------------------------------------------------
# 5. Guarded live cross-checks vs the on-disk SGLang reference (SKIP if deps
#    absent). These make the alignment non-circular when the container has the
#    SGLang serving deps.
# ---------------------------------------------------------------------------
def _try_import_sglang_parse():
    if SGLANG_ROOT not in sys.path:
        sys.path.insert(0, SGLANG_ROOT)
    try:
        from sglang.test.simple_eval_mmmu_vlm import _parse_multi_choice_response  # noqa: F401

        return _parse_multi_choice_response
    except Exception:
        return None


def _try_import_sglang_image():
    if SGLANG_ROOT not in sys.path:
        sys.path.insert(0, SGLANG_ROOT)
    try:
        from sglang.srt.multimodal.inkling.image_processing import (
            IMAGE_MEAN,
            IMAGE_STD,
            PAD_NORM,
            _encode_image_bytes,
        )

        return {
            "IMAGE_MEAN": IMAGE_MEAN,
            "IMAGE_STD": IMAGE_STD,
            "PAD_NORM": PAD_NORM,
            "_encode_image_bytes": _encode_image_bytes,
        }
    except Exception:
        return None


def test_crosscheck_sglang_parse_multi_choice():
    ref = _try_import_sglang_parse()
    if ref is None:
        print("SKIP live sglang parse cross-check (sglang deps unavailable)")
        return
    choices = ["A", "B", "C", "D"]
    i2a = {"A": "cat", "B": "dog", "C": "bird", "D": "fish"}
    battery = [
        "Some reasoning... Answer: C",
        "Answer: A\nmore\nAnswer: D",
        "The correct choice is (B).",
        "B",
        "answer: b but Answer: D",
        "No idea",
        "i really think it is a dog for sure",
        "(A) and (C) but finally (D)",
        "the answer is clearly cat here",
    ]
    for resp in battery:
        assert H.parse_multi_choice_response(resp, choices, i2a) == ref(resp, choices, i2a), (
            f"parse divergence vs sglang on {resp!r}"
        )
    print("OK live sglang parse cross-check (byte-identical on battery)")


def test_crosscheck_sglang_image_processing():
    ref = _try_import_sglang_image()
    if ref is None:
        print("SKIP live sglang image cross-check (numba/sglang unavailable)")
        return
    # constants must match byte-for-byte
    np.testing.assert_array_equal(H.IMAGE_MEAN, np.asarray(ref["IMAGE_MEAN"]))
    np.testing.assert_array_equal(H.IMAGE_STD, np.asarray(ref["IMAGE_STD"]))
    np.testing.assert_array_equal(H.PAD_NORM, np.asarray(ref["PAD_NORM"]))

    # byte-identical float32 patches on a deterministic synthetic image (compare
    # before bf16 cast; sglang casts to bf16 at the end, we compare the pre-cast
    # normalized float32 values which are what the numba kernel computes).
    import io

    try:
        from PIL import Image
    except Exception:
        print("SKIP live sglang image cross-check (PIL unavailable)")
        return

    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(83, 57, 3), dtype=np.uint8)
    ours, num_patches, _, _ = H.preprocess_patches(arr, patch_size=40)

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    theirs_bf16 = ref["_encode_image_bytes"](
        buf.getvalue(),
        patch_size=40,
        rescale_image_frac=None,  # isolate patching from resize
        rescale_image_max_upscaled_long_edge=None,
    )
    # theirs_bf16: (P, 2, 40, 40, 3) bf16; take T=0 and compare to our float32
    theirs = theirs_bf16[:, 0].float().numpy()
    assert theirs.shape == ours.shape, f"{theirs.shape} != {ours.shape}"
    # sglang cast float32 -> bf16 loses mantissa bits; compare at bf16 tolerance
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=8e-3)
    print(f"OK live sglang image cross-check ({num_patches} patches, bf16-tol)")


# ---------------------------------------------------------------------------
# Plain-script runner (no pytest dependency required in the container)
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    total = len(tests)
    print(f"\n{total - failures}/{total} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
