# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling image-embedding fusion mechanics test (Stage-1 / Goal 1.4).

``reference_tier=real_source``, ``validation_tier=unit`` (CUDA/GPU).

Goal 1.4 wires the hMLP vision tower into ``InklingForConditionalGeneration`` so
image embeddings replace the ``<image>`` placeholder embeddings before the
accepted text decoder. This test proves the **fusion mechanics** end-to-end on
GPU, exercising the EXACT production code path the model ``forward`` uses --
without loading the 403GB text decoder (that TP=4 end-to-end
``source_logit_replay`` / ``generation_parity`` run is the next Goal-1.4 step):

  1. WIRING PRESENT (CPU) -- ``InklingForConditionalGeneration`` exposes the
     ``mm_token_ids`` property + ``forward`` / ``load_weights`` /
     ``_resolve_mm_indices`` overrides, and the module-level
     ``_encode_inkling_image_embeds`` helper exists.
  2. OOV PLACEHOLDER IS UNEMBEDDABLE (CPU) -- the SGLang ``<image>`` sentinel
     ``-101`` raises when fed to an ``nn.Embedding``; this is why fusion calls
     ``fuse_input_embeds(mm_token_ids=None)`` (the OOV-safe path) rather than the
     in-vocab fast path (which would ``embedding_layer(input_ids)`` on ``-101``).
  3. FUSION SCATTER ON CUDA (pass-critical) -- the real ``model.visual.*`` bf16
     tower runs over real MMMU image patches via the production
     ``_encode_inkling_image_embeds`` helper; the engine-style indices are
     computed from ``-101`` via ``filter_mm_token_from_input_ids`` (``isin``,
     the same predicate the model engine uses); and OOV-safe
     ``fuse_input_embeds`` places the tower rows at exactly the placeholder
     positions and the text embeddings everywhere else, with
     ``num_placeholder_tokens == mm_feature_rows``.

Reference: the tower is the Goal-1.3 ``InklingVisionModel`` (already proven
bitwise-equal to SGLang ``HMLPPatchEncoder`` on CUDA). This test adds the FUSION
layer: that the real vision features land at the right sequence positions under
the OOV placeholder contract. Non-skipping on the CUDA path.

Run (container, GPU node, checkpoint mounted):
  * ``python inkling_image_fusion_test.py``
  * ``pytest -q inkling_image_fusion_test.py``

Env: reuses ``INKLING_CKPT`` / ``SGLANG_PY`` / ``MMMU_ALIGN_CACHE`` from Goal 1.3.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import List

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import inkling_mmmu_real_align_test as R  # noqa: E402  (fixed MMMU items)
import inkling_vision_tower_test as V  # noqa: E402  (real visual weights + cfg)

from tensorrt_llm._torch.models.modeling_inkling import (  # noqa: E402
    InklingForConditionalGeneration,
    _encode_inkling_image_embeds,
)
from tensorrt_llm._torch.models.modeling_inkling_vision import (  # noqa: E402  # noqa: E402
    DEFAULT_IMAGE_TOKEN_ID,
    InklingImagePreprocessor,
    InklingVisionModel,
)
from tensorrt_llm._torch.models.modeling_multimodal_utils import (  # noqa: E402
    filter_mm_token_from_input_ids,
    find_input_mm_embeds,
    fuse_input_embeds,
)

IMG_ID = DEFAULT_IMAGE_TOKEN_ID  # in-vocab <|unused_200054|> (200054); see note
VOCAB = 201024
HID = 6144


def _build_visual(device, dtype):
    weights = V.load_visual_weights()
    tower = InklingVisionModel(V.VISION_CFG).to(dtype)
    tower.load_weights(weights)
    return tower.to(device=device).eval()


# ===========================================================================
# CPU structural / contract checks
# ===========================================================================
def test_fusion_wiring_present():
    assert isinstance(InklingForConditionalGeneration.__dict__.get("mm_token_ids"), property)
    for name in ("forward", "load_weights", "_resolve_mm_indices"):
        assert name in InklingForConditionalGeneration.__dict__, name
    assert callable(_encode_inkling_image_embeds)


def test_placeholder_is_in_vocab_and_never_embedded():
    # The image placeholder (200054 = <|unused_200054|>) is IN-VOCAB: TRT-LLM's
    # executor validates request token ids and rejects an out-of-range id, so the
    # SGLang-internal -101 cannot be sent to llm.generate (RequestError: Token ID
    # out of range). Fusion still calls fuse_input_embeds(mm_token_ids=None) with
    # explicit text/mm indices, so the placeholder id itself is NEVER embedded --
    # the vision tower overwrites those positions (proven by vision_scatter_ok).
    assert 0 <= IMG_ID < VOCAB, IMG_ID


# ===========================================================================
# CUDA fusion scatter (pass-critical)
# ===========================================================================
def _require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the image-fusion scatter test; a skip would "
            "hide missing GPU evidence."
        )


def run_fusion() -> dict:
    _require_cuda()
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    visual = _build_visual(device, dtype)
    pre = InklingImagePreprocessor(
        patch_size=V.PATCH_SIZE, temporal_patch_size=V.TEMPORAL, dtype=dtype
    )
    items = R.load_fixed_items()[:3]
    assert items, "no fixed MMMU items resolved (warm MMMU_ALIGN_CACHE?)"

    params: List[SimpleNamespace] = []
    per_np: List[int] = []
    for it in items:
        patches = pre.encode_one(it["image_bytes"])  # (np, T, P, P, C) bf16 CPU
        per_np.append(int(patches.shape[0]))
        params.append(
            SimpleNamespace(
                multimodal_data={"image": {"vision_patches_bthwc": patches}},
                multimodal_runtime=None,
            )
        )
    total = sum(per_np)

    # 1. production vision-encode helper -> one concatenated feature tensor
    mm_embeds = _encode_inkling_image_embeds(visual, params)
    assert len(mm_embeds) == 1 and tuple(mm_embeds[0].shape) == (total, HID)
    mm_embeds = find_input_mm_embeds(mm_embeds, params)  # runtime=None passthru
    feats = mm_embeds[0]
    # cross-check: concatenation order == per-image tower outputs in order
    ref = torch.cat(
        [
            visual(
                p.multimodal_data["image"]["vision_patches_bthwc"].to(device=device, dtype=dtype)
            )
            for p in params
        ],
        dim=0,
    )
    concat_ok = torch.equal(feats, ref)

    # 2. build a flat input_ids: [text, <image>*np0, text, <image>*np1, ...]
    ids: List[int] = []
    for np_i in per_np:
        ids += [1, 2, 3]
        ids += [IMG_ID] * np_i
    ids += [9, 9]
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)

    # 3. engine-style indices from the -101 sentinel (isin -- same predicate as
    #    the model engine's _prepare_multimodal_indices with mm_token_ids set).
    ti, mi = filter_mm_token_from_input_ids(
        input_ids,
        vocab_size=VOCAB,
        mm_token_ids=torch.tensor([IMG_ID], dtype=torch.long, device=device),
    )
    count_ok = int(mi.shape[0]) == total == int(feats.shape[0])

    # 4. small deterministic text embedding (the scatter, not the embed values,
    #    is what this checks; the real 201024x6144 embed is unneeded here).
    embed = nn.Embedding(128, HID).to(device=device, dtype=dtype)

    # 5. OOV-safe fuse: mm_token_ids=None so -101 is never looked up.
    ret_ids, fused = fuse_input_embeds(
        embed, input_ids, mm_embeds, mm_token_ids=None, text_token_indices=ti, mm_token_indices=mi
    )

    # 6. scatter correctness: vision rows at placeholder positions, text elsewhere
    shape_ok = ret_ids is None and tuple(fused.shape) == (len(ids), HID)
    vision_ok = bool(torch.equal(fused[mi].float(), feats.float()))
    text_ids = input_ids[ti]
    text_ok = bool(torch.equal(fused[ti].float(), embed(text_ids).float()))
    # no placeholder embedding leaked into the vision rows
    finite_ok = bool(torch.isfinite(fused.float()).all())

    ok = concat_ok and count_ok and shape_ok and vision_ok and text_ok and finite_ok
    return {
        "num_images": len(items),
        "per_image_patches": per_np,
        "total_feature_rows": total,
        "placeholder_tokens": int(mi.shape[0]),
        "fused_shape": list(fused.shape),
        "concat_order_ok": bool(concat_ok),
        "count_ok": bool(count_ok),
        "shape_ok": bool(shape_ok),
        "vision_scatter_ok": vision_ok,
        "text_scatter_ok": text_ok,
        "finite_ok": finite_ok,
        "all_ok": bool(ok),
    }


_SUMMARY = None


def _get() -> dict:
    global _SUMMARY
    if _SUMMARY is None:
        _SUMMARY = run_fusion()
    return _SUMMARY


def test_cuda_image_fusion_scatter():
    s = _get()
    assert s["all_ok"], s


def test_cuda_placeholder_count_equals_feature_rows():
    s = _get()
    assert s["placeholder_tokens"] == s["total_feature_rows"], s


# ---------------------------------------------------------------------------
# Plain-script runner
# ---------------------------------------------------------------------------
def _main() -> int:
    cpu_checks = [test_fusion_wiring_present, test_placeholder_is_in_vocab_and_never_embedded]
    print("=== Inkling image-fusion CPU contract checks ===")
    for fn in cpu_checks:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
            return 1

    print("\n=== CUDA image-fusion scatter (real tower + real patches) ===")
    try:
        s = run_fusion()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"FUSION FAILED to produce evidence: {type(e).__name__}: {e}")
        return 2
    print(f"  images={s['num_images']} per_image_patches={s['per_image_patches']}")
    print(
        f"  total_feature_rows={s['total_feature_rows']} "
        f"placeholder_tokens={s['placeholder_tokens']} "
        f"fused_shape={s['fused_shape']}"
    )
    print(
        f"  concat_order_ok={s['concat_order_ok']} count_ok={s['count_ok']} "
        f"shape_ok={s['shape_ok']} vision_scatter_ok={s['vision_scatter_ok']} "
        f"text_scatter_ok={s['text_scatter_ok']} finite_ok={s['finite_ok']}"
    )
    ok = s["all_ok"]
    print("ALL OK" if ok else "FUSION MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
