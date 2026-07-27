# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling vision-tower INTERNAL per-sublayer localizer (S3-C8; feedback #3 STEP 2, probe V1/V2).

``reference_tier=real_source``, ``validation_tier=unit`` (CUDA/GPU).

STEP 0 (Priority-0 MMMU determinism) is proven: job 5605006 -> ``INKLING_P0_PASS``
(p0a/p0b/p0c, n_runs=3, token_flip 0/12). Feedback #3 makes vision localization
UNCONDITIONAL once STEP 0 holds, and fixes the probe order V1->V5. This is the
FIRST probe point (V1 tower internals + V2 projector/HMLP).

Why "go INSIDE": the whole-tower output is bit-identical to SGLang (Goal 1.3:
max_abs=0.0), so a tower-BOUNDARY compare keeps reporting "clean". Feedback #3 V1
requires walking the tower's INTERNAL sublayers because "a defect that cancels at
the tower boundary on short/simple images can still be real on large ones."

What it does: register forward hooks on every named submodule (the four
``layers.linear_i``, the three ``layers.norm_i``, and ``final_norm``) of BOTH the
production TRT ``InklingVisionModel`` and SGLang's REAL ``HMLPPatchEncoder``, run
the SAME real MMMU images -- deliberately spanning small AND large ``num_patches``
-- through both towers on GPU, and report, per sublayer per image, cosine AND the
magnitude RATIO ``||trt|| / ||sglang||``. The FIRST sublayer whose ratio departs
from 1.0 (or cosine from 1.0) NAMES a tower-internal defect and its scale; if every
sublayer is 1.0 on the LARGEST-patch item too, tower internals (V1/V2) are clean
and the search moves to V3 embed_norm / V4 scatter next (a separate full-model
probe), exactly as feedback #3 orders.

No teacher-forcing / token sequence is needed: this is a forward-pass activation
replay on identical inputs, so free-run near-tie forking is structurally absent.
Deterministic by construction (single bf16 forward, seed pinned, no runtime / no
cuda_graph / no overlap scheduler). Reuses the proven Goal-1.3 loaders
(``_build_towers``, ``InklingImagePreprocessor``, fixed MMMU items) rather than
rebuilding them.

Env (inherited from the tower test):
  * ``INKLING_CKPT`` -- checkpoint dir
  * ``MMMU_ALIGN_CACHE`` -- cached real MMMU items
  * ``SGLANG_PY`` -- sglang ``python/`` root
  * ``INKLING_VISION_LOCALIZE_ARTIFACT`` -- output JSON path
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

import inkling_mmmu_real_align_test as R  # noqa: E402  (cached items)
import inkling_vision_tower_test as T  # noqa: E402  (proven loaders/config)

# The eight checkpoint sublayers, in forward order. ``fold_timespace_to_depth``
# and ``F.gelu`` are functional (not submodules) so they carry no hook; a
# divergence introduced by a fold/gelu still surfaces at the NEXT linear's output.
SUBLAYERS = [
    "layers.linear_0",
    "layers.norm_0",
    "layers.linear_1",
    "layers.norm_1",
    "layers.linear_2",
    "layers.norm_2",
    "layers.linear_3",
    "final_norm",
]

# Divergence thresholds: a clean fp-identical tower sits at cosine==1.0 /
# mag_ratio==1.0; anything past these is a named divergence to surface (not fix).
COS_MIN = 0.9999
RATIO_TOL = 1e-3

ARTIFACT = os.environ.get(
    "INKLING_VISION_LOCALIZE_ARTIFACT", os.path.join(HERE, "inkling_vision_localize_artifact.json")
)


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Cosine + magnitude ratio ``||a|| / ||b||`` in fp64 (a=trt, b=sglang)."""
    if tuple(a.shape) != tuple(b.shape):
        return {
            "shape_mismatch": True,
            "shape_trt": list(a.shape),
            "shape_sglang": list(b.shape),
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
        "norm_sglang": round(nb, 6),
    }


def _diverges(c: dict) -> bool:
    if c.get("shape_mismatch"):
        return True
    r = c.get("mag_ratio")
    return c.get("cosine", 0.0) < COS_MIN or r is None or abs(r - 1.0) > RATIO_TOL


def _register(model: torch.nn.Module, store: Dict[str, torch.Tensor], tag: str) -> List:
    """Hook the 8 named sublayers; fail loud (never silently skip) if absent."""
    handles = []
    mods = dict(model.named_modules())
    for name in SUBLAYERS:
        if name not in mods:
            raise RuntimeError(
                f"{tag} tower missing submodule {name!r}; available head={sorted(mods)[:16]}"
            )

        def mk(nm):
            def hook(_m, _inp, out):
                t = out[0] if isinstance(out, (tuple, list)) else out
                store[nm] = t.detach()

            return hook

        handles.append(mods[name].register_forward_hook(mk(name)))
    return handles


def run_localize() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the vision-tower internal localizer; a skip "
            "would hide missing GPU evidence."
        )
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    trt, sg = T._build_towers(device, dtype)  # same real bf16 weights, both eval
    pre = T.InklingImagePreprocessor(
        patch_size=T.PATCH_SIZE, temporal_patch_size=T.TEMPORAL, dtype=dtype
    )
    items = R.load_fixed_items()
    assert items, "no fixed MMMU items resolved (warm MMMU_ALIGN_CACHE?)"

    records: List[dict] = []
    worst = {"cosine_min": 1.0, "ratio_dev_max": 0.0}
    for it in items:
        patches = pre.encode_one(it["image_bytes"])
        num_patches = int(patches.shape[0])
        x = patches.to(device=device, dtype=dtype)
        strt: Dict[str, torch.Tensor] = {}
        ssg: Dict[str, torch.Tensor] = {}
        ht = _register(trt, strt, "trt")
        hs = _register(sg, ssg, "sglang")
        try:
            with torch.no_grad():
                trt_out = trt(x)
                sg_out = sg(x)
        finally:
            for h in ht + hs:
                h.remove()

        per: Dict[str, dict] = {}
        first_div = None
        for name in SUBLAYERS:
            c = _cmp(strt[name], ssg[name])
            per[name] = c
            worst["cosine_min"] = min(worst["cosine_min"], c.get("cosine", 0.0))
            r = c.get("mag_ratio")
            if r is not None and np.isfinite(r):
                worst["ratio_dev_max"] = max(worst["ratio_dev_max"], abs(r - 1.0))
            if first_div is None and _diverges(c):
                first_div = name
        records.append(
            {
                "id": it["id"],
                "num_patches": num_patches,
                "first_divergent_sublayer": first_div,
                "tower_out": _cmp(trt_out, sg_out),
                "per_sublayer": per,
            }
        )

    records.sort(key=lambda r: r["num_patches"])  # small -> large patch
    summary = {
        "probe": "V1/V2 vision-tower internal per-sublayer (feedback #3 STEP 2)",
        "reference": "sglang HMLPPatchEncoder (real source, bf16, CUDA)",
        "checkpoint": T.CKPT,
        "p0_gate": "5605006 INKLING_P0_PASS token_flip 0/12",
        "cos_min_thresh": COS_MIN,
        "ratio_tol": RATIO_TOL,
        "n_items": len(records),
        "patch_min": records[0]["num_patches"] if records else None,
        "patch_max": records[-1]["num_patches"] if records else None,
        "worst": worst,
        "any_divergent": any(r["first_divergent_sublayer"] for r in records),
        "divergences": [
            {
                "id": r["id"],
                "num_patches": r["num_patches"],
                "first_divergent_sublayer": r["first_divergent_sublayer"],
                "detail": r["per_sublayer"][r["first_divergent_sublayer"]],
            }
            for r in records
            if r["first_divergent_sublayer"]
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
    """Validate hook wiring + compare math without GPU/checkpoint/SGLang.

    Builds two ``InklingVisionModel`` copies with IDENTICAL random weights, hooks
    all 8 sublayers on both, runs a random patch batch, and asserts (a) every
    sublayer is captured on both, (b) identical weights => cosine==1.0 /
    mag_ratio==1.0 everywhere, and (c) a deliberate ``final_norm`` perturbation is
    DETECTED as a mag_ratio departure -- proving the localizer can actually name a
    divergence rather than always reporting clean.
    """
    torch.manual_seed(0)
    a = T.InklingVisionModel(T.VISION_CFG).eval()
    b = T.InklingVisionModel(T.VISION_CFG).eval()
    b.load_state_dict(a.state_dict())  # identical weights
    x = torch.randn(3, T.TEMPORAL, T.PATCH_SIZE, T.PATCH_SIZE, T.N_CHANNELS)

    sa: Dict[str, torch.Tensor] = {}
    sb: Dict[str, torch.Tensor] = {}
    ha = _register(a, sa, "a")
    hb = _register(b, sb, "b")
    with torch.no_grad():
        a(x)
        b(x)
    for h in ha + hb:
        h.remove()
    assert set(sa) == set(SUBLAYERS) == set(sb), (sorted(sa), sorted(sb))
    for name in SUBLAYERS:
        c = _cmp(sa[name], sb[name])
        assert not _diverges(c), (name, c)

    # Perturb b.final_norm so its output scales; the localizer MUST flag it.
    with torch.no_grad():
        b.final_norm.weight.mul_(1.5)
    sa, sb = {}, {}
    ha = _register(a, sa, "a")
    hb = _register(b, sb, "b")
    with torch.no_grad():
        a(x)
        b(x)
    for h in ha + hb:
        h.remove()
    cfn = _cmp(sa["final_norm"], sb["final_norm"])
    assert _diverges(cfn) and cfn["mag_ratio"] is not None and abs(cfn["mag_ratio"] - 1.0) > 0.1, (
        cfn
    )
    print(
        "VISION_LOCALIZE_SELFTEST_OK 8 sublayers hooked on both towers; "
        "identical-weights => cosine/mag_ratio == 1.0; perturbed final_norm "
        f"flagged (mag_ratio={cfn['mag_ratio']:.4f})."
    )
    return 0


# ===========================================================================
# pytest: assert MECHANICS/completeness (not absence of divergence -- a real
# divergence is a NAMED finding to surface for human review, not a test failure).
# ===========================================================================
def test_vision_localize_selftest_cpu():
    assert _selftest() == 0


def test_vision_localize_artifact_complete():
    s = run_localize()
    assert s["n_items"] >= 1
    for r in s["records"]:
        assert set(r["per_sublayer"]) == set(SUBLAYERS), r["id"]
        for name in SUBLAYERS:
            c = r["per_sublayer"][name]
            assert "cosine" in c and ("mag_ratio" in c), (r["id"], name, c)
    # Must include a genuinely large-patch item (feedback #3: probe small AND large).
    assert s["patch_max"] and s["patch_max"] >= 512, s["patch_max"]


def _main() -> int:
    print("=== Inkling vision-tower INTERNAL per-sublayer localizer (V1/V2) ===")
    try:
        s = run_localize()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"LOCALIZE FAILED to produce evidence: {type(e).__name__}: {e}")
        return 2
    print(f"reference={s['reference']}  p0_gate={s['p0_gate']}")
    print(f"patch range {s['patch_min']}..{s['patch_max']}  n_items={s['n_items']}")
    for r in s["records"]:
        fd = r["first_divergent_sublayer"] or "-none-"
        print(f"\n{r['id']:<28} npatch={r['num_patches']:<5} first_div={fd}")
        for name in SUBLAYERS:
            c = r["per_sublayer"][name]
            mr = c.get("mag_ratio")
            flag = "  <== DIVERGE" if _diverges(c) else ""
            print(
                f"   {name:<16} cos={c.get('cosine')} "
                f"mag_ratio={mr} max_abs={c.get('max_abs')}{flag}"
            )
    print(
        f"\nworst: cosine_min={s['worst']['cosine_min']:.8f} "
        f"ratio_dev_max={s['worst']['ratio_dev_max']:.3e}  "
        f"any_divergent={s['any_divergent']}  artifact={ARTIFACT}"
    )
    if s["any_divergent"]:
        print("TOWER_INTERNALS_DIVERGENCE_NAMED " + json.dumps(s["divergences"]))
    else:
        print(
            "TOWER_INTERNALS_CLEAN (V1/V2 exactly 1.0 incl. largest patch; "
            "next probe: V3 embed_norm / V4 scatter in the fused stream)"
        )
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(_main())
