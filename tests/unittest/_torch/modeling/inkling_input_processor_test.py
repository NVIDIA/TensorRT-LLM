# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling multimodal input processor + placeholder registry test (Goal 1.2).

``reference_tier=real_source``, ``validation_tier=unit``.

Proves the Stage-1 / Goal-1.2 contract for the production TRT-LLM
``InklingInputProcessor`` (``tensorrt_llm._torch.models.modeling_inkling_vision``):

  1. REGISTRATION -- ``InklingForConditionalGeneration`` registers the processor
     and the ``{"image": "<image>"}`` placeholder metadata in TRT-LLM's
     input-processor / placeholder registries.
  2. TEXT PASSTHROUGH -- a text-only request tokenizes and returns
     ``(ids, {})`` with the ids unchanged (the accepted text path is untouched).
  3. PLACEHOLDER EXPANSION -- one ``<image>`` placeholder expands into exactly
     ``num_patches`` tokens (one per vision patch), and the attached
     ``vision_patches_bthwc`` has exactly that many feature rows.
  4. FAIL-LOUD -- a placeholder/media count mismatch (either direction) and a
     token-vs-feature-row mismatch raise ``ValueError`` (media is never dropped
     or padded silently).
  5. REAL-SOURCE PREPROCESSING PARITY -- on real ``MMMU/MMMU`` validation images
     the production ``InklingImagePreprocessor`` matches SGLang's real
     ``InklingImageProcessor._encode_image_bytes`` (same patch grid /
     ``num_patches`` and per-patch tensor within bf16 tolerance), and the
     assemble expansion count equals SGLang's ``num_patches``.

The SGLang reference is loaded from on-disk source via the helpers already
proven in ``inkling_mmmu_real_align_test`` (which handle the numba
``<dynamic>`` disk-cache hazard); the production preprocessing path itself uses
NO numba and NO ``sglang`` import. This test is deliberately non-skipping for
the pure-unit checks; the real-source parity check FAILS (does not skip) if the
SGLang reference or the fixed MMMU items cannot be resolved.

Run:
  * ``python inkling_input_processor_test.py``   (container; real-source parity
    needs a warm ``MMMU_ALIGN_CACHE`` or network + the on-disk sglang checkout)
  * ``pytest -q inkling_input_processor_test.py``
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Production code under test.
from tensorrt_llm._torch.configs.inkling import InklingConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_inkling_vision import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN_ID,
    InklingImagePreprocessor,
    InklingInputProcessor,
    patch_grid,
    scaled_image_dimensions,
)

# Serving preprocessing config (InklingImageProcessor defaults).
PATCH_SIZE = 40
TEMPORAL = 2
RESCALE_FRAC = 2.0
RESCALE_CAP = 2048


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
class _FakeTokenizer:
    """Minimal tokenizer stub exposing the ``.encode`` API that the stock
    ``DefaultInputProcessor`` calls; returns a fixed id stream for any text.

    ``id_stream`` may embed the image sentinel to exercise the image path of
    ``call_with_text_prompt`` without the (deferred) real chat renderer.
    """

    def __init__(self, id_stream: List[int]):
        self._ids = list(id_stream)

    def encode(self, text, add_special_tokens=True, **kwargs):
        return list(self._ids)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(self._ids))


def _sampling_params():
    from tensorrt_llm.sampling_params import SamplingParams

    return SamplingParams()


def _make_processor(image_token_id: Optional[int] = None, tokenizer=None) -> InklingInputProcessor:
    """Build an ``InklingInputProcessor`` from a grounded ``InklingConfig``.

    ``vision_config`` carries the real checkpoint geometry (``patch_size=40``,
    ``temporal_patch_size=2``). ``image_token_id`` defaults to the SGLang
    sentinel via the config.
    """
    kwargs = dict(
        vision_config={
            "vision_encoder_type": "hmlp",
            "decoder_dmodel": 6144,
            "patch_size": PATCH_SIZE,
            "temporal_patch_size": TEMPORAL,
            "n_channels": 3,
            "use_vision_norm": True,
        }
    )
    if image_token_id is not None:
        kwargs["image_token_id"] = image_token_id
    config = InklingConfig(**kwargs)
    return InklingInputProcessor(None, config, tokenizer)


def _synthetic_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _expected_num_patches(h: int, w: int) -> int:
    sw, sh = scaled_image_dimensions(w, h, RESCALE_FRAC, RESCALE_CAP)
    n, _, _ = patch_grid(sh, sw, PATCH_SIZE)
    return n


# ---------------------------------------------------------------------------
# 1. Registration
# ---------------------------------------------------------------------------
def test_registration():
    # Importing the model module runs the @register_input_processor decorator.
    from tensorrt_llm._torch.models.modeling_inkling import InklingForConditionalGeneration
    from tensorrt_llm.inputs.registry import (
        INPUT_PROCESSOR_REGISTRY,
        MULTIMODAL_PLACEHOLDER_REGISTRY,
    )

    assert MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid("inkling_mm_model", "image")
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder("inkling_mm_model", "image") == "<image>"
    reg = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type
    assert reg.get(InklingForConditionalGeneration) is InklingInputProcessor
    assert InklingInputProcessor._registered_model_type == "inkling_mm_model"


# ---------------------------------------------------------------------------
# 1b. Serving / profiler interface contract (regression guard)
#
# ``trtllm-serve`` crashed at startup (jobs 5597129/5597134) with
# ``AttributeError: 'InklingInputProcessor' object has no attribute
# 'get_preferred_media_io_kwargs'`` because the processor inherited only
# ``BaseMultimodalInputProcessor`` while ``OpenAIServer.__init__`` calls that
# method (living on ``BaseMultimodalDummyInputsBuilder``) for any
# ``BaseMultimodalInputProcessor``. This test reproduces the exact server +
# KV-cache-encoder-profiler call contract so the interface can never silently
# regress again -- CPU-only, no GPU/checkpoint, so it runs in the fast unit lane.
# ---------------------------------------------------------------------------
def test_serving_and_profiler_interface_contract():
    from tensorrt_llm.inputs.registry import (
        BaseMultimodalDummyInputsBuilder,
        BaseMultimodalInputProcessor,
    )

    proc = _make_processor()  # instantiation itself proves the two-base class
    # is concrete (abstract ``model_path``/``config`` are implemented).

    # The server guards on this base, then calls get_preferred_media_io_kwargs.
    assert isinstance(proc, BaseMultimodalInputProcessor)
    assert isinstance(proc, BaseMultimodalDummyInputsBuilder)

    # (a) OpenAIServer.__init__ contract (serve/openai_server.py):
    #     model_pref = ip.get_preferred_media_io_kwargs() or {}
    media_pref = proc.get_preferred_media_io_kwargs()
    assert isinstance(media_pref, dict)
    # image-only Inkling needs no special media-IO decode format (unlike Qwen's
    # video np-frames); the empty default is the correct behaviour.
    assert media_pref == {}
    _server_seed = proc.get_preferred_media_io_kwargs() or {}  # must not raise
    assert _server_seed == {}

    # (b) KV-cache encoder profiler contract (_torch/pyexecutor/_util.py):
    #     demand = input_processor.get_mm_max_tokens_per_item(); total<=0 -> skip
    demand = proc.get_mm_max_tokens_per_item()
    assert isinstance(demand, dict)
    # Empty demand -> the profiler's ``total_demand <= 0`` guard returns early
    # and never calls get_dummy_mm_data_for_tokens, so image fusion keeps
    # working exactly as under the in-process LLM API (MMMU runner job 5596216).
    assert sum(demand.values()) == 0

    # (c) get_dummy_mm_data_for_tokens exists (default raises NotImplementedError,
    #     which the profiler catches as "no direct profiling"); assert it is at
    #     least present/callable so the attribute-error class cannot recur.
    assert callable(getattr(proc, "get_dummy_mm_data_for_tokens", None))


# ---------------------------------------------------------------------------
# 2. Text-only passthrough
# ---------------------------------------------------------------------------
def test_text_only_passthrough_assemble():
    proc = _make_processor()
    ids = [5, 6, 7, 8]
    out_ids, mm = proc.assemble(ids, None)
    assert out_ids == ids
    assert mm == {}
    out_ids2, mm2 = proc.assemble(ids, [])
    assert out_ids2 == ids and mm2 == {}


def test_text_only_passthrough_call_with_text_prompt():
    proc = _make_processor(tokenizer=_FakeTokenizer([10, 11, 12, 13]))
    out_ids, extra = proc.call_with_text_prompt({"prompt": "hello world"}, _sampling_params())
    # Delegates to DefaultInputProcessor -> (ids, None), byte-identical to the
    # accepted text path.
    assert out_ids == [10, 11, 12, 13]
    assert extra is None


# ---------------------------------------------------------------------------
# 3. Placeholder expansion: one <image> -> num_patches tokens
# ---------------------------------------------------------------------------
def test_placeholder_expansion_count_equals_num_patches():
    proc = _make_processor()
    img = _synthetic_image(80, 120, seed=1)
    exp_np = _expected_num_patches(80, 120)
    ids = [1, 2, DEFAULT_IMAGE_TOKEN_ID, 3]
    out_ids, mm = proc.assemble(ids, [img])

    n_expanded = sum(1 for t in out_ids if t == DEFAULT_IMAGE_TOKEN_ID)
    assert n_expanded == exp_np, (n_expanded, exp_np)
    # length grows by (num_patches - 1) for the one placeholder
    assert len(out_ids) == len(ids) - 1 + exp_np
    # feature-row count == placeholder-token count
    feat = mm["image"]
    assert int(feat["vision_patches_bthwc"].shape[0]) == exp_np
    assert feat["num_patches"] == [exp_np]
    # offsets mark one contiguous span
    ((start, end),) = feat["offsets"]
    assert start == 2 and end == 2 + exp_np - 1
    assert all(t == DEFAULT_IMAGE_TOKEN_ID for t in out_ids[start : end + 1])
    # patch tensor shape (num_patches, T, P, P, 3)
    assert tuple(feat["vision_patches_bthwc"].shape) == (
        exp_np,
        TEMPORAL,
        PATCH_SIZE,
        PATCH_SIZE,
        3,
    )


def test_two_images_expand_independently():
    proc = _make_processor()
    a = _synthetic_image(80, 120, seed=2)
    b = _synthetic_image(200, 40, seed=3)
    na, nb = _expected_num_patches(80, 120), _expected_num_patches(200, 40)
    ids = [1, DEFAULT_IMAGE_TOKEN_ID, 9, DEFAULT_IMAGE_TOKEN_ID, 2]
    out_ids, mm = proc.assemble(ids, [a, b])
    assert sum(1 for t in out_ids if t == DEFAULT_IMAGE_TOKEN_ID) == na + nb
    feat = mm["image"]
    assert feat["num_patches"] == [na, nb]
    assert int(feat["vision_patches_bthwc"].shape[0]) == na + nb
    o0, o1 = feat["offsets"]
    assert o0 == (1, 1 + na - 1)
    assert o1 == (1 + na + 1, 1 + na + 1 + nb - 1)


def test_call_with_text_prompt_image_path():
    # Fake tokenizer emits a stream already carrying one sentinel placeholder.
    proc = _make_processor(tokenizer=_FakeTokenizer([7, DEFAULT_IMAGE_TOKEN_ID, 8]))
    img = _synthetic_image(80, 120, seed=4)
    exp_np = _expected_num_patches(80, 120)
    out_ids, extra = proc.call_with_text_prompt(
        {"prompt": "<image> describe", "multi_modal_data": {"image": [img]}}, _sampling_params()
    )
    assert sum(1 for t in out_ids if t == DEFAULT_IMAGE_TOKEN_ID) == exp_np
    assert "multimodal_data" in extra and "image" in extra["multimodal_data"]
    feat = extra["multimodal_data"]["image"]
    assert int(feat["vision_patches_bthwc"].shape[0]) == exp_np


# ---------------------------------------------------------------------------
# 3b. Pre-tokenized (-101) + image fast path (Goal 1.4 runtime entry)
# ---------------------------------------------------------------------------
def test_supports_token_id_mm_expansion_flag():
    # The processor opts into the LLM-API tokenized+MM fast path so a
    # TokensPrompt(prompt_token_ids=<-101 stream>, multi_modal_data=...) is NOT
    # detokenized (the checkpoint tokenizer has no <image> -> -101 mapping).
    assert InklingInputProcessor.supports_token_id_mm_expansion is True


def test_call_with_token_ids_expands_and_attaches():
    # This is the exact path the end-to-end TP=4 runtime test hits: pre-tokenized
    # ids already carry one -101 sentinel per image; call_with_token_ids expands
    # it to num_patches and attaches the preprocessed vision features.
    proc = _make_processor()
    img = _synthetic_image(80, 120, seed=11)
    exp_np = _expected_num_patches(80, 120)
    ids = [7, DEFAULT_IMAGE_TOKEN_ID, 8]  # one sentinel, already tokenized
    out_ids, extra = proc.call_with_token_ids(
        {"prompt_token_ids": ids, "multi_modal_data": {"image": [img]}}, _sampling_params()
    )
    assert sum(1 for t in out_ids if t == DEFAULT_IMAGE_TOKEN_ID) == exp_np
    assert len(out_ids) == len(ids) - 1 + exp_np
    assert "multimodal_data" in extra and "image" in extra["multimodal_data"]
    feat = extra["multimodal_data"]["image"]
    assert int(feat["vision_patches_bthwc"].shape[0]) == exp_np


def test_call_with_token_ids_text_only_passthrough():
    # No image -> plain token passthrough, no multimodal payload (keeps the
    # accepted text path byte-identical even through the fast-path hook).
    proc = _make_processor()
    ids = [1, 2, 3, 4]
    out_ids, extra = proc.call_with_token_ids(
        {"prompt_token_ids": ids, "multi_modal_data": {}}, _sampling_params()
    )
    assert out_ids == ids and extra is None


def test_call_with_token_ids_fail_loud_count_mismatch():
    import pytest

    proc = _make_processor()
    img = _synthetic_image(80, 120, seed=12)
    # two sentinels but one image -> fail loud (never drop/pad media silently)
    ids = [DEFAULT_IMAGE_TOKEN_ID, 5, DEFAULT_IMAGE_TOKEN_ID]
    with pytest.raises(ValueError, match="counts must match"):
        proc.call_with_token_ids(
            {"prompt_token_ids": ids, "multi_modal_data": {"image": [img]}}, _sampling_params()
        )


# ---------------------------------------------------------------------------
# 4. Fail-loud on count mismatch
# ---------------------------------------------------------------------------
def test_fail_loud_more_images_than_placeholders():
    import pytest

    proc = _make_processor()
    img = _synthetic_image(80, 120, seed=5)
    ids = [1, DEFAULT_IMAGE_TOKEN_ID, 2]  # 1 placeholder
    with pytest.raises(ValueError, match="counts must match"):
        proc.assemble(ids, [img, img])  # 2 images


def test_fail_loud_more_placeholders_than_images():
    import pytest

    proc = _make_processor()
    img = _synthetic_image(80, 120, seed=6)
    ids = [DEFAULT_IMAGE_TOKEN_ID, 1, DEFAULT_IMAGE_TOKEN_ID]  # 2 placeholders
    with pytest.raises(ValueError, match="counts must match"):
        proc.assemble(ids, [img])  # 1 image


def test_fail_loud_image_without_placeholder():
    import pytest

    proc = _make_processor()
    img = _synthetic_image(80, 120, seed=7)
    with pytest.raises(ValueError, match="counts must match"):
        proc.assemble([1, 2, 3], [img])  # 0 placeholders, 1 image


# ---------------------------------------------------------------------------
# 5. Real-source preprocessing parity vs SGLang InklingImageProcessor
# ---------------------------------------------------------------------------
def _tensor_stats(a: np.ndarray, b: np.ndarray):
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    fa, fb = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb)) or 1.0
    cos = float(np.dot(fa, fb) / denom)
    return float(diff.max()), float(diff.mean()), cos


def run_preprocess_parity() -> dict:
    """Compare production preprocessing to SGLang's real numba kernel on real
    MMMU images. Reuses the proven loaders in ``inkling_mmmu_real_align_test``."""
    import inkling_mmmu_real_align_test as R  # noqa: E402

    ip, _ev = R.load_sglang_refs()  # SGLang image_processing (numba-safe load)
    items = R.load_fixed_items()
    assert items, "no fixed MMMU items resolved"

    pre = InklingImagePreprocessor(patch_size=PATCH_SIZE, temporal_patch_size=TEMPORAL)
    proc = _make_processor()
    records: List[dict] = []
    n_ok = 0
    max_abs = 0.0
    for it in items:
        png = it["image_bytes"]
        sg = (
            ip._encode_image_bytes(
                png,
                patch_size=PATCH_SIZE,
                rescale_image_frac=RESCALE_FRAC,
                rescale_image_max_upscaled_long_edge=RESCALE_CAP,
            )
            .float()
            .numpy()
        )  # (P, 2, 40, 40, 3)
        trt = pre.encode_one(png).float().numpy()
        shape_ok = tuple(sg.shape) == tuple(trt.shape)
        if shape_ok:
            m_abs, m_mean, m_cos = _tensor_stats(trt, sg)
        else:
            m_abs, m_mean, m_cos = float("inf"), float("inf"), 0.0
        sg_np = int(sg.shape[0])
        # assemble expansion must equal SGLang's num_patches for this image
        out_ids, mm = proc.assemble([DEFAULT_IMAGE_TOKEN_ID], [png])
        n_expanded = sum(1 for t in out_ids if t == DEFAULT_IMAGE_TOKEN_ID)
        feat_rows = int(mm["image"]["vision_patches_bthwc"].shape[0])
        ok = (
            shape_ok
            and int(trt.shape[0]) == sg_np
            and m_abs <= 1e-2
            and m_cos >= 0.9999
            and n_expanded == sg_np == feat_rows
        )
        n_ok += int(ok)
        max_abs = max(max_abs, m_abs if np.isfinite(m_abs) else 1e9)
        records.append(
            {
                "id": it["id"],
                "trt_num_patches": int(trt.shape[0]) if shape_ok else None,
                "sglang_num_patches": sg_np,
                "assemble_expanded": n_expanded,
                "feat_rows": feat_rows,
                "max_abs": None if not np.isfinite(m_abs) else round(m_abs, 6),
                "mean_abs": None if not np.isfinite(m_mean) else round(m_mean, 6),
                "cosine": round(m_cos, 8),
                "match": bool(ok),
            }
        )
    return {
        "num_items": len(items),
        "aligned": n_ok,
        "max_abs_over_items": round(max_abs, 6),
        "records": records,
    }


def test_real_source_preprocess_parity():
    s = run_preprocess_parity()
    assert s["aligned"] == s["num_items"], [r for r in s["records"] if not r["match"]]


# ---------------------------------------------------------------------------
# Plain-script runner (mirrors the Goal 1.1 dual-mode style)
# ---------------------------------------------------------------------------
def _main() -> int:
    # Pure-unit checks first (no sglang / no network needed).
    unit = [
        test_registration,
        test_text_only_passthrough_assemble,
        test_text_only_passthrough_call_with_text_prompt,
        test_placeholder_expansion_count_equals_num_patches,
        test_two_images_expand_independently,
        test_call_with_text_prompt_image_path,
        test_fail_loud_more_images_than_placeholders,
        test_fail_loud_more_placeholders_than_images,
        test_fail_loud_image_without_placeholder,
    ]
    print("=== Inkling input-processor unit checks ===")
    for fn in unit:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
            return 1

    print("\n=== real-source preprocessing parity vs SGLang ===")
    try:
        s = run_preprocess_parity()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"REAL-SOURCE PARITY FAILED to produce evidence: {type(e).__name__}: {e}")
        return 2
    for r in s["records"]:
        print(
            f"  {r['id']:<28} "
            f"np(trt={r['trt_num_patches']}==sg={r['sglang_num_patches']}) "
            f"expand={r['assemble_expanded']} rows={r['feat_rows']} "
            f"max_abs={r['max_abs']} cos={r['cosine']} "
            f"{'OK' if r['match'] else 'X'}"
        )
    ok = s["aligned"] == s["num_items"]
    print(f"\nparity {s['aligned']}/{s['num_items']}  max_abs={s['max_abs_over_items']}")
    print("ALL ALIGNED" if ok else "PARITY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
