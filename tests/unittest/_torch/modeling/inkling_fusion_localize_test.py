# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling fused-stream numeric localizer (S3-C8; feedback #3 STEP 2, probes V3/V4).

``reference_tier=real_source``, ``validation_tier=unit`` (CUDA/GPU).

V1/V2 (vision-tower internals + HMLP/projector) are already proven bit-identical
to SGLang at every sublayer incl. the largest patch (job 5607495 ->
``TOWER_INTERNALS_CLEAN``). Feedback #3 fixes the probe order V1->V5 and makes
localization UNCONDITIONAL once STEP 0 (P0 determinism, job 5605006) holds. This
is the NEXT probe point:

  * V3 -- ``embed_norm``: the RMSNorm the multimodal path applies to the TEXT
    embeddings before the image rows are scattered in. This is exactly where the
    iter22 bug lived (the pre-fix code re-normed the FUSED stream, pushing the
    raw image rows through an extra RMSNorm). Feedback #3: "Check the norm's
    inputs, its statistics, and exactly which token positions it covers."
  * V4 -- image-embed SCATTER into the fused stream: placement indices, count,
    order, dtype, and whether the image positions carry the RAW vision-tower
    rows (NOT a re-normed copy). "Off-by-one/ordering bugs here are invisible to
    a tower-level compare."

Why this can run on ONE GPU (no 403GB decoder, no SGLang server): the fused
stream entering the decoder is, by construction,
``embed_norm(embed_tokens[text_ids])`` at the text positions and the RAW
vision-tower rows at the image positions (TRT
``InklingForConditionalGeneration.forward`` / ``get_input_embeddings`` +
``fuse_input_embeds(..., inputs_embeds_prenormed=True)``; SGLang
``inkling.InklingModel.get_input_embeddings`` + ``general_mm_embed_routine``).
The vision rows are already proven identical to SGLang (V1/V2), so the only
un-verified numeric object is ``embed_norm`` on the real ``model.llm.embed``
table plus the scatter accounting. We load ONLY ``model.llm.embed.weight`` (the
201024x6144 bf16 table, ~2.4GB) and ``model.llm.embed_norm.weight`` via partial
``safe_open`` reads (same pattern as the tower test's ``load_visual_weights``),
build the fused stream through the PRODUCTION helpers (``fuse_input_embeds`` /
``filter_mm_token_from_input_ids`` -- the exact ops the model engine uses), and
compare against a high-precision reference.

What it reports, per real MMMU item (deliberately spanning a SMALL-patch and a
LARGE-patch image, feedback #3's scale-dependence guard), for V3 and V4:
  * V3 embed_norm parity: cosine + magnitude RATIO of the production TRT
    ``RMSNorm`` output vs an fp64 mathematically-exact RMSNorm (the SGLang
    ``RMSNorm(eps=1e-6)`` formula on the SAME weights collapses to the same
    exact reference), over the text positions. The FIRST region whose ratio
    departs from 1.0 names a numeric defect and its scale.
  * V4 scatter accounting: filled-placeholder count == vision rows == patch
    count; image positions carry the RAW tower rows BITWISE (``fused[mi]``
    equals ``mm_rows`` -- the direct iter22 re-norm check); dtype is bf16; and
    the text positions carry embed_norm(text), not raw embeddings.

No teacher-forcing / token sequence is needed: this is a forward-pass activation
replay on identical inputs, so free-run near-tie forking is structurally absent
(the img_moe_isolate finish=length contamination warning is N/A here). If every
region is 1.0 on the largest-patch item too, V3/V4 are clean and the search
steps to V5 (processor/registry accounting, already asserted per-item in the
MMMU runner's ``image_used`` gate) and then the first decoder layer that
CONSUMES the fused stream.

Env (inherited from the tower/fusion tests):
  * ``INKLING_CKPT`` / ``INKLING_CHECKPOINT`` -- checkpoint dir
  * ``MMMU_ALIGN_CACHE`` -- cached real MMMU items
  * ``SGLANG_PY`` -- sglang ``python/`` root (for the real tower loader)
  * ``INKLING_FUSION_LOCALIZE_ARTIFACT`` -- output JSON path
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import inkling_vision_tower_test as T  # noqa: E402  (proven loaders/config/CKPT)

from tensorrt_llm._torch.models.modeling_inkling_vision import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN_ID,
    InklingImagePreprocessor,
)
from tensorrt_llm._torch.models.modeling_multimodal_utils import (  # noqa: E402
    filter_mm_token_from_input_ids,
    fuse_input_embeds,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm  # noqa: E402

IMG_ID = DEFAULT_IMAGE_TOKEN_ID  # 200054, the in-vocab <image> placeholder
HID = 6144
VOCAB = 201024
EPS = 1e-6  # config.json text_config rms_norm_eps

COS_MIN = 0.9999
RATIO_TOL = 1e-3

ARTIFACT = os.environ.get(
    "INKLING_FUSION_LOCALIZE_ARTIFACT", os.path.join(HERE, "inkling_fusion_localize_artifact.json")
)


# ---------------------------------------------------------------------------
# High-precision reference RMSNorm. SGLang's ``RMSNorm(eps=1e-6)`` and TRT's
# ``RMSNorm`` implement the identical formula ``x * rsqrt(mean(x^2)+eps) *
# weight``; computing it in fp64 gives the exact target both bf16 kernels must
# round to. Comparing TRT's production module against this fp64 target (and
# noting SGLang collapses to the SAME target on the SAME weights) is a
# cast-order-agnostic parity check: a real eps / position / re-norm defect moves
# the ratio off 1.0 regardless of either kernel's rounding order.
# ---------------------------------------------------------------------------
def _rmsnorm_ref_fp64(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x64 = x.detach().to(torch.float64)
    var = x64.pow(2).mean(dim=-1, keepdim=True)
    return (x64 * torch.rsqrt(var + eps)) * weight.detach().to(torch.float64)


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Cosine + magnitude ratio ``||a|| / ||b||`` in fp64 (a=trt, b=reference)."""
    if tuple(a.shape) != tuple(b.shape):
        return {
            "shape_mismatch": True,
            "shape_trt": list(a.shape),
            "shape_ref": list(b.shape),
            "cosine": 0.0,
            "mag_ratio": float("inf"),
        }
    a64 = a.detach().to(torch.float64).flatten()
    b64 = b.detach().to(torch.float64).flatten()
    na = float(torch.linalg.norm(a64))
    nb = float(torch.linalg.norm(b64))
    diff = (a64 - b64).abs()
    denom = (na * nb) or 1.0
    cos = float(torch.dot(a64, b64) / denom)
    ratio = (na / nb) if nb else float("inf")
    return {
        "max_abs": round(float(diff.max()), 8),
        "mean_abs": round(float(diff.mean()), 8),
        "cosine": round(cos, 10),
        "mag_ratio": round(ratio, 10) if np.isfinite(ratio) else None,
        "norm_trt": round(na, 6),
        "norm_ref": round(nb, 6),
    }


def _diverges(c: dict) -> bool:
    if c.get("shape_mismatch"):
        return True
    r = c.get("mag_ratio")
    return c.get("cosine", 0.0) < COS_MIN or r is None or abs(r - 1.0) > RATIO_TOL


# ---------------------------------------------------------------------------
# Real ``model.llm.embed.weight`` + ``model.llm.embed_norm.weight`` (partial
# reads, not the full checkpoint), mirroring the tower test's loader.
# ---------------------------------------------------------------------------
def _load_embed_and_norm() -> Dict[str, torch.Tensor]:
    with open(T._resolve_index(T.CKPT)) as f:
        weight_map = json.load(f)["weight_map"]
    want = ["model.llm.embed.weight", "model.llm.embed_norm.weight"]
    for k in want:
        if k not in weight_map:
            raise RuntimeError(
                f"checkpoint index missing {k!r}; have e.g. {sorted(weight_map)[:4]}"
            )
    by_shard: Dict[str, List[str]] = {}
    for k in want:
        by_shard.setdefault(weight_map[k], []).append(k)
    from safetensors import safe_open

    out: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(T.CKPT, shard), framework="pt", device="cpu") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


def _fixed_records():
    """Return real MMMU items that carry expanded input_ids + <image> spans,
    preferring the canonical prompt builder (byte-identical to the e2e / parity
    paths). Falls back to constructing input_ids from cached items."""
    try:
        import inkling_image_prompts as P  # canonical builder w/ input_ids

        recs = P.build_prompts(8)
        out = []
        for r in recs:
            ids = list(r["input_ids"])
            npimg = sum(1 for t in ids if int(t) == IMG_ID)
            if npimg > 0:
                out.append(
                    {
                        "id": r["id"],
                        "input_ids": ids,
                        "image_bytes": r["image_bytes"],
                        "num_patches_prompt": npimg,
                    }
                )
        if out:
            return out, "inkling_image_prompts.build_prompts"
    except Exception as e:  # noqa: BLE001
        print(
            f"[fusion-loc] canonical builder unavailable ({type(e).__name__}: "
            f"{e}); falling back to cached-item construction",
            flush=True,
        )
    # Fallback: cached items + synthetic text ids around a real <image> span.
    import inkling_mmmu_real_align_test as R

    items = R.load_fixed_items()
    out = []
    rng = np.random.RandomState(0)
    for it in items:
        out.append(
            {
                "id": it["id"],
                "input_ids": None,
                "image_bytes": it["image_bytes"],
                "num_patches_prompt": None,
                "_rng": rng,
            }
        )
    return out, "inkling_mmmu_real_align_test.load_fixed_items(fallback)"


def _build_input_ids(rec, num_patches, rng_seed) -> torch.Tensor:
    """Real prompt input_ids with the single ``<image>`` placeholder EXPANDED to
    one token per patch -- exactly what ``InklingInputProcessor`` does at runtime
    (``P.build_prompts`` returns the pre-expansion prompt with a single
    placeholder, so the raw ids carry 1 placeholder while the tower emits
    ``num_patches`` rows). Falls back to a deterministic real-vocab text
    prefix/suffix around the ``<image>`` span (embed_norm is per-row, so real
    vocab rows are what matter for the numeric check)."""
    ids = rec.get("input_ids")
    if ids is not None:
        ids = [int(t) for t in ids]
        img_pos = [i for i, t in enumerate(ids) if t == IMG_ID]
        if len(img_pos) == num_patches:
            return torch.tensor(ids, dtype=torch.long)  # already expanded
        if len(img_pos) >= 1:
            # Replace ALL placeholder tokens with one expanded block placed at the
            # first placeholder's position among the non-placeholder (text) tokens.
            non = [t for t in ids if t != IMG_ID]
            insert_at = sum(1 for i in range(img_pos[0]) if ids[i] != IMG_ID)
            expanded = non[:insert_at] + [IMG_ID] * int(num_patches) + non[insert_at:]
            return torch.tensor(expanded, dtype=torch.long)
    rng = np.random.RandomState(rng_seed)
    # sample real, non-placeholder vocab ids for a representative text context
    pre = rng.randint(0, 200000, size=24).tolist()
    suf = rng.randint(0, 200000, size=8).tolist()
    ids = pre + [IMG_ID] * int(num_patches) + suf
    return torch.tensor(ids, dtype=torch.long)


def run_localize() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the fused-stream (V3/V4) localizer; a skip "
            "would hide missing GPU evidence."
        )
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Real vision tower (TRT), proven == SGLang at every sublayer (V1/V2).
    trt_tower, _sg = T._build_towers(device, dtype)
    pre = InklingImagePreprocessor(
        patch_size=T.PATCH_SIZE, temporal_patch_size=T.TEMPORAL, dtype=dtype
    )

    # Real embed table + embed_norm weight; build the PRODUCTION RMSNorm module.
    w = _load_embed_and_norm()
    embed = (
        torch.nn.Embedding(VOCAB, HID, _weight=w["model.llm.embed.weight"].to(dtype))
        .to(device=device)
        .eval()
    )
    embed_norm = RMSNorm(hidden_size=HID, eps=EPS, dtype=dtype).to(device)
    with torch.no_grad():
        embed_norm.weight.copy_(w["model.llm.embed_norm.weight"].to(dtype))
    norm_w = embed_norm.weight.detach()

    recs, src = _fixed_records()
    assert recs, "no fixed MMMU items resolved (warm MMMU_ALIGN_CACHE?)"

    # Preprocess every item's image -> tower rows; keep patch count for scale span.
    prepared = []
    for i, rec in enumerate(recs):
        patches = pre.encode_one(rec["image_bytes"])
        np_img = int(patches.shape[0])
        prepared.append((np_img, i, rec, patches))
    prepared.sort(key=lambda t: t[0])
    # Probe the SMALL-patch and LARGE-patch extremes (feedback #3 scale guard).
    picks = [prepared[0]]
    if len(prepared) > 1:
        picks.append(prepared[-1])

    records: List[dict] = []
    worst = {"cosine_min": 1.0, "ratio_dev_max": 0.0}
    for np_img, idx, rec, patches in picks:
        x = patches.to(device=device, dtype=dtype)
        with torch.no_grad():
            mm_rows = trt_tower(x)  # (np_img, HID) bf16 -- raw vision rows
        input_ids = _build_input_ids(rec, np_img, idx).to(device)

        # Production engine indices from the placeholder id (isin -- same
        # predicate the model engine's _prepare_multimodal_indices uses).
        ti, mi = filter_mm_token_from_input_ids(
            input_ids,
            vocab_size=VOCAB,
            mm_token_ids=torch.tensor([IMG_ID], dtype=torch.long, device=device),
        )

        # ---- TRT production fusion path -----------------------------------
        # get_input_embeddings folds embed_norm onto the TEXT ids; fuse_input_embeds
        # scatters the RAW tower rows at the placeholder positions (prenormed=True,
        # so the decoder never re-norms). Replicated through the SAME helpers the
        # model.forward calls (fuse_input_embeds / filter_mm_token_from_input_ids).
        def _get_text_embeds(ids):
            return embed_norm(embed(ids))

        _ids, fused = fuse_input_embeds(
            _get_text_embeds,
            input_ids,
            [mm_rows],
            mm_token_ids=None,
            text_token_indices=ti,
            mm_token_indices=mi,
        )

        # ---- Reference fused stream (fp64-exact embed_norm on the SAME table) --
        text_ids = input_ids[ti]
        text_raw = embed(text_ids)
        text_ref = _rmsnorm_ref_fp64(text_raw, norm_w).to(dtype)

        # ---- V3: embed_norm numeric parity over the TEXT positions ---------
        v3 = _cmp(fused[ti], text_ref)
        # Also the raw-vs-normed sanity: text positions must NOT equal the raw
        # embedding (proves embed_norm actually ran on text).
        text_norm_applied = not bool(torch.equal(fused[ti], text_raw))

        # ---- V4: scatter accounting ---------------------------------------
        n_mi = int(mi.shape[0])
        count_ok = n_mi == np_img == int(mm_rows.shape[0])
        # image positions carry the RAW tower rows, bitwise (iter22 re-norm check)
        image_rows_raw = bool(torch.equal(fused[mi].float(), mm_rows.float()))
        v4_image = _cmp(fused[mi], mm_rows)  # expected exactly 1.0
        dtype_ok = fused.dtype == dtype
        shape_ok = tuple(fused.shape) == (int(input_ids.shape[0]), HID)
        finite_ok = bool(torch.isfinite(fused.float()).all())

        first_div = None
        for tag, c in (("V3_embed_norm_text", v3), ("V4_image_scatter", v4_image)):
            if _diverges(c):
                first_div = tag
                break
        if first_div is None and not (
            count_ok
            and image_rows_raw
            and text_norm_applied
            and dtype_ok
            and shape_ok
            and finite_ok
        ):
            first_div = "V4_scatter_accounting"

        for c in (v3, v4_image):
            worst["cosine_min"] = min(worst["cosine_min"], c.get("cosine", 0.0))
            r = c.get("mag_ratio")
            if r is not None and np.isfinite(r):
                worst["ratio_dev_max"] = max(worst["ratio_dev_max"], abs(r - 1.0))

        records.append(
            {
                "id": rec["id"],
                "num_patches": np_img,
                "n_text_positions": int(ti.shape[0]),
                "n_image_positions": n_mi,
                "first_divergent_region": first_div,
                "V3_embed_norm_text": v3,
                "V4_image_scatter": v4_image,
                "v4_count_ok": count_ok,
                "v4_image_rows_raw": image_rows_raw,
                "v3_text_norm_applied": text_norm_applied,
                "v4_dtype_ok": dtype_ok,
                "v4_shape_ok": shape_ok,
                "v4_finite_ok": finite_ok,
            }
        )

    summary = {
        "probe": "V3/V4 fused-stream embed_norm + scatter (feedback #3 STEP 2)",
        "reference": (
            "fp64-exact RMSNorm(eps=1e-6) on real model.llm.embed_norm "
            "(== SGLang RMSNorm formula on identical weights); raw "
            "vision rows proven == SGLang V1/V2 job 5607495"
        ),
        "item_source": src,
        "checkpoint": T.CKPT,
        "p0_gate": "5605006 INKLING_P0_PASS token_flip 0/12",
        "cos_min_thresh": COS_MIN,
        "ratio_tol": RATIO_TOL,
        "n_items": len(records),
        "patch_min": records[0]["num_patches"] if records else None,
        "patch_max": records[-1]["num_patches"] if records else None,
        "worst": worst,
        "any_divergent": any(r["first_divergent_region"] for r in records),
        "divergences": [
            {
                "id": r["id"],
                "num_patches": r["num_patches"],
                "first_divergent_region": r["first_divergent_region"],
                "detail": r.get(
                    r["first_divergent_region"],
                    {
                        "v4_count_ok": r["v4_count_ok"],
                        "v4_image_rows_raw": r["v4_image_rows_raw"],
                        "v3_text_norm_applied": r["v3_text_norm_applied"],
                    },
                ),
            }
            for r in records
            if r["first_divergent_region"]
        ],
        "records": records,
    }
    os.makedirs(os.path.dirname(ARTIFACT) or ".", exist_ok=True)
    with open(ARTIFACT, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ===========================================================================
# Offline mechanics selftest (CPU, random weights; no checkpoint / no SGLang).
# ===========================================================================
def _selftest() -> int:
    """Validate the fused-stream build + compare math without GPU/checkpoint.

    Uses tiny random embed/embed_norm/mm rows, builds the fused stream through
    the SAME production ``fuse_input_embeds`` helper, and asserts: (a)
    embed_norm(text) matches the fp64 reference (clean), (b) the image positions
    carry the raw mm rows bitwise, (c) a deliberately mis-scaled embed_norm
    weight is DETECTED as a V3 mag_ratio departure, and (d) a deliberately
    re-normed image row (the iter22 bug) is DETECTED as a V4 failure.
    """
    torch.manual_seed(0)
    hid = 16
    vocab = 40
    npimg = 5
    emb = torch.nn.Embedding(vocab, hid)
    nw = torch.rand(hid) + 0.5
    img_id = vocab - 1  # placeholder id

    def norm_ref(x, w):
        return _rmsnorm_ref_fp64(x, w).to(x.dtype)

    ids = torch.tensor([1, 2, 3] + [img_id] * npimg + [4, 5], dtype=torch.long)
    ti = torch.tensor([i for i, t in enumerate(ids.tolist()) if t != img_id])
    mi = torch.tensor([i for i, t in enumerate(ids.tolist()) if t == img_id])
    mm_rows = torch.randn(npimg, hid)

    def get_text(idz):
        return norm_ref(emb(idz), nw)

    _r, fused = fuse_input_embeds(
        get_text, ids, [mm_rows], mm_token_ids=None, text_token_indices=ti, mm_token_indices=mi
    )
    # (a) text positions == fp64 embed_norm reference
    c_ok = _cmp(fused[ti], norm_ref(emb(ids[ti]), nw))
    assert not _diverges(c_ok), c_ok
    # (b) image positions == raw mm rows bitwise
    assert torch.equal(fused[mi], mm_rows), "image rows not raw"

    # (c) mis-scaled embed_norm weight -> V3 divergence detected
    bad = norm_ref(emb(ids[ti]), nw * 1.5)
    c_bad = _cmp(bad, norm_ref(emb(ids[ti]), nw))
    assert (
        _diverges(c_bad) and c_bad["mag_ratio"] is not None and abs(c_bad["mag_ratio"] - 1.5) < 1e-6
    ), c_bad

    # (d) re-normed image row (iter22 bug) -> V4 raw check fails
    renormed = norm_ref(mm_rows, nw)
    assert not torch.equal(renormed, mm_rows), "renorm should change rows"
    c_renorm = _cmp(renormed, mm_rows)
    assert _diverges(c_renorm), c_renorm

    print(
        "FUSION_LOCALIZE_SELFTEST_OK fused stream via production "
        "fuse_input_embeds; embed_norm(text) == fp64 ref; image rows raw; "
        f"mis-scaled norm flagged (ratio={c_bad['mag_ratio']:.4f}); "
        "re-normed image row flagged (iter22 bug detectable)."
    )
    return 0


# ===========================================================================
# pytest: MECHANICS/completeness (a real divergence is a NAMED finding to
# surface for human review, not a test failure).
# ===========================================================================
def test_fusion_localize_selftest_cpu():
    assert _selftest() == 0


def test_fusion_localize_artifact_complete():
    s = run_localize()
    assert s["n_items"] >= 1
    for r in s["records"]:
        assert "V3_embed_norm_text" in r and "V4_image_scatter" in r, r["id"]
        assert "cosine" in r["V3_embed_norm_text"], r["id"]
    # Must include a genuinely large-patch item (probe small AND large).
    assert s["patch_max"] and s["patch_max"] >= 512, s["patch_max"]


def _main() -> int:
    print("=== Inkling fused-stream V3/V4 localizer (embed_norm + scatter) ===")
    try:
        s = run_localize()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"FUSION-LOC FAILED to produce evidence: {type(e).__name__}: {e}")
        return 2
    print(f"reference={s['reference']}")
    print(f"item_source={s['item_source']}  p0_gate={s['p0_gate']}")
    print(f"patch range {s['patch_min']}..{s['patch_max']}  n_items={s['n_items']}")
    for r in s["records"]:
        fd = r["first_divergent_region"] or "-none-"
        print(
            f"\n{r['id']:<28} npatch={r['num_patches']:<5} "
            f"n_text={r['n_text_positions']} n_img={r['n_image_positions']} "
            f"first_div={fd}"
        )
        v3 = r["V3_embed_norm_text"]
        v4 = r["V4_image_scatter"]
        print(
            f"   V3 embed_norm(text)  cos={v3.get('cosine')} "
            f"mag_ratio={v3.get('mag_ratio')} max_abs={v3.get('max_abs')} "
            f"norm_applied={r['v3_text_norm_applied']}"
        )
        print(
            f"   V4 image scatter     cos={v4.get('cosine')} "
            f"mag_ratio={v4.get('mag_ratio')} rows_raw={r['v4_image_rows_raw']} "
            f"count_ok={r['v4_count_ok']} dtype_ok={r['v4_dtype_ok']} "
            f"finite_ok={r['v4_finite_ok']}"
        )
    print(
        f"\nworst: cosine_min={s['worst']['cosine_min']:.8f} "
        f"ratio_dev_max={s['worst']['ratio_dev_max']:.3e}  "
        f"any_divergent={s['any_divergent']}  artifact={ARTIFACT}"
    )
    if s["any_divergent"]:
        print("FUSION_V3V4_DIVERGENCE_NAMED " + json.dumps(s["divergences"]))
    else:
        print(
            "FUSION_V3V4_CLEAN (embed_norm numerically exact + image rows raw "
            "at both small and large patch; next probe: first decoder layer "
            "that consumes the fused stream)"
        )
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(_main())
