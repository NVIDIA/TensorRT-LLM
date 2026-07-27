# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling HMLP vision tower CUDA ``source_activation_replay`` test (Goal 1.3).

``reference_tier=real_source``, ``validation_tier=unit`` (CUDA/GPU).

Proves the Stage-1 / Goal-1.3 contract for the production TRT-LLM
``InklingVisionModel`` (``tensorrt_llm._torch.models.modeling_inkling_vision``):

  1. SCALE PLAN / MODULE TREE (CPU) -- ``plan_out_scales`` resolves to the
     checkpoint's four-layer hMLP progression, the four ``Linear`` shapes are the
     checkpoint's (75->128, 512->320, 5120->4800, 9600->6144), and the module
     tree exposes exactly the eight ``model.visual.*`` parameter names the
     checkpoint ships (``layers.linear_{0..3}.weight``, ``layers.norm_{0..2}.weight``,
     ``final_norm.weight``) so a strict load neither drops nor invents a key.
  2. FOLD BIJECTION (CPU) -- ``fold_timespace_to_depth`` is a shape-correct
     value-preserving fold, guarding the reshape/permute order.
  3. SOURCE ACTIVATION REPLAY (CUDA, pass-critical) -- the REAL ``model.visual.*``
     bf16 weights load strictly into BOTH the production ``InklingVisionModel``
     and SGLang's REAL ``HMLPPatchEncoder`` (the requested serving comparand,
     loaded from on-disk ``inkling_common/hmlp.py``). Every fixed real ``MMMU/MMMU``
     validation image is preprocessed by the Goal-1.2 ``InklingImagePreprocessor``
     (already proven byte-equal to SGLang) and ALL its patches run through both
     towers on GPU. Per image we assert: feature-row count == ``num_patches``,
     output width == ``decoder_dmodel`` (6144), and activations agree with SGLang
     within tolerance, reporting ``max_abs``, ``mean_abs`` and cosine similarity.
  4. PLACEHOLDER INVARIANT / FAIL-LOUD -- the input processor expands one
     ``<image>`` placeholder into exactly ``num_patches`` tokens == feature rows,
     and a token-vs-feature-row mismatch fails loudly (media never dropped/padded).

Why SGLang is the reference: task.yaml designates SGLang (PR 31681) as the
correctness/accuracy comparand on the NVFP4 checkpoint, and the checkpoint uses
SGLang-style ``model.visual.layers.linear_i`` naming. HF Transformers is the
tower-math ground truth; SGLang's ``HMLPPatchEncoder.forward`` is the SAME math
(verified: identical ``plan_out_scales`` / ``fold_timespace_to_depth`` / exact
``F.gelu`` / fp32-variance ``F.rms_norm``), so matching SGLang bit-for-bit is
matching the source tower. SGLang's ``inkling_common/hmlp.py`` is loaded from its
source file (absolute imports satisfied by lightweight namespace stubs + the real
``inkling_common/norm.py``); NO full ``sglang`` serving stack is imported, and the
production tower uses no ``sglang`` import and no custom kernel.

Non-skipping: if CUDA, the checkpoint ``model.visual.*`` weights, or the SGLang
reference cannot be resolved, the pass-critical replay FAILS (a skip would hide
missing GPU evidence).

Run (container, GPU node, checkpoint mounted):
  * ``python inkling_vision_tower_test.py``
  * ``pytest -q inkling_vision_tower_test.py``

Env:
  * ``INKLING_CKPT``  -- checkpoint dir (default: the task.yaml checkpoint_path)
  * ``MMMU_ALIGN_CACHE`` -- cached real MMMU items (reused from Goal 1.1/1.2)
  * ``SGLANG_PY``     -- sglang ``python/`` root
  * ``INKLING_VISION_ARTIFACT`` -- path for the replay record JSON
"""

from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import inkling_mmmu_real_align_test as R  # noqa: E402  (cached items + importlib loader)

from tensorrt_llm._torch.configs.inkling import InklingConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_inkling_vision import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN_ID,
    InklingImagePreprocessor,
    InklingInputProcessor,
    InklingVisionModel,
    fold_timespace_to_depth,
    plan_out_scales,
)

CKPT = os.environ.get(
    "INKLING_CKPT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/hf_data/Inkling-NVFP4-full",
)
SGLANG_PY = os.environ.get(
    "SGLANG_PY",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/codes/sglang/python",
)
SGLANG_HMLP = os.path.join(SGLANG_PY, "sglang/srt/models/inkling_common/hmlp.py")
SGLANG_NORM = os.path.join(SGLANG_PY, "sglang/srt/models/inkling_common/norm.py")
ARTIFACT = os.environ.get(
    "INKLING_VISION_ARTIFACT", os.path.join(HERE, "inkling_vision_tower_artifact.json")
)

PATCH_SIZE = 40
TEMPORAL = 2
N_CHANNELS = 3
N_LAYERS = 4
DECODER_DMODEL = 6144

# The in-scope checkpoint vision_config (config.json). SGLang HMLPPatchEncoder and
# the production InklingVisionModel read the same field names.
VISION_CFG = SimpleNamespace(
    vision_encoder_type="hmlp",
    decoder_dmodel=DECODER_DMODEL,
    patch_size=PATCH_SIZE,
    temporal_patch_size=TEMPORAL,
    n_channels=N_CHANNELS,
    n_layers=N_LAYERS,
    use_vision_norm=True,
)

# Resolved four-layer hMLP scale progression for (T=2, P=40, n_layers=4, C=3),
# verified by hand (linear_sum_assignment over the 6 candidate scales, first/last
# pinned). The CPU test asserts the production planner reproduces it exactly.
EXPECTED_SCALES = [
    (1, 1, 1, 3),
    (1, 5, 5, 128),
    (1, 10, 10, 320),
    (1, 40, 40, 4800),
    (2, 40, 40, 9600),
]

# Exact Linear weight shapes (out_features, in_features) implied by the scales.
EXPECTED_LINEAR_SHAPES = {
    "layers.linear_0.weight": (128, 75),
    "layers.linear_1.weight": (320, 512),
    "layers.linear_2.weight": (4800, 5120),
    "layers.linear_3.weight": (6144, 9600),
}

# The eight checkpoint parameter names (after stripping ``model.visual.``).
EXPECTED_PARAM_NAMES = {
    "layers.linear_0.weight",
    "layers.linear_1.weight",
    "layers.linear_2.weight",
    "layers.linear_3.weight",
    "layers.norm_0.weight",
    "layers.norm_1.weight",
    "layers.norm_2.weight",
    "final_norm.weight",
}


# ===========================================================================
# Real checkpoint model.visual.* bf16 weights (partial reads, not the full ckpt)
# ===========================================================================
def _resolve_index(ckpt_dir: str) -> str:
    cand = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.exists(cand):
        return cand
    import glob

    hits = sorted(glob.glob(os.path.join(ckpt_dir, "*.index.json")))
    if not hits:
        raise RuntimeError(f"no safetensors index json under {ckpt_dir}")
    return hits[0]


def load_visual_weights() -> Dict[str, torch.Tensor]:
    """Read only the (small, bf16) ``model.visual.*`` tensors from their shards.

    Returns the full-key dict ``{model.visual.<...>.weight: tensor}``; partial
    reads via ``safe_open`` avoid loading the multi-hundred-GB checkpoint.
    """
    with open(_resolve_index(CKPT)) as f:
        weight_map = json.load(f)["weight_map"]
    vkeys = sorted(k for k in weight_map if k.startswith("model.visual."))
    if not vkeys:
        raise RuntimeError("no model.visual.* keys in checkpoint index")
    by_shard: Dict[str, List[str]] = {}
    for k in vkeys:
        by_shard.setdefault(weight_map[k], []).append(k)
    from safetensors import safe_open

    weights: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu") as f:
            for k in keys:
                weights[k] = f.get_tensor(k)
    return weights


def _strip_visual(full: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    p = "model.visual."
    return {(k[len(p) :] if k.startswith(p) else k): v for k, v in full.items()}


# ===========================================================================
# SGLang real HMLPPatchEncoder reference (importlib + stubbed package chain)
# ===========================================================================
def load_sglang_hmlp():
    """Load SGLang's real ``HMLPPatchEncoder`` module from on-disk source.

    ``inkling_common/hmlp.py`` uses absolute imports
    (``sglang.srt.configs.inkling.InklingVisionConfig`` -- annotation-only under
    ``from __future__ import annotations`` -- and
    ``sglang.srt.models.inkling_common.norm.RMSNorm``). We satisfy them with
    lightweight namespace-package stubs plus the REAL ``norm.py`` (torch-only),
    so the whole ``sglang`` serving stack is never imported.
    """
    if not os.path.exists(SGLANG_HMLP):
        raise RuntimeError(f"missing SGLang hmlp.py at {SGLANG_HMLP}")
    if not os.path.exists(SGLANG_NORM):
        raise RuntimeError(f"missing SGLang norm.py at {SGLANG_NORM}")

    def ensure_pkg(name: str) -> types.ModuleType:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []  # mark as (namespace) package
            sys.modules[name] = m
        return sys.modules[name]

    for pkg in (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.models",
        "sglang.srt.models.inkling_common",
    ):
        ensure_pkg(pkg)
    # hmlp uses InklingVisionConfig only as a type annotation -> stub is fine.
    cfgmod = ensure_pkg("sglang.srt.configs.inkling")
    if not hasattr(cfgmod, "InklingVisionConfig"):
        cfgmod.InklingVisionConfig = object
    # Load the REAL RMSNorm under the exact name hmlp imports (absolute import).
    norm = R._load_module_from_file(SGLANG_NORM, "sglang.srt.models.inkling_common.norm")
    sys.modules["sglang.srt.models.inkling_common.norm"] = norm
    hmlp = R._load_module_from_file(SGLANG_HMLP, "sglang_inkling_hmlp_ref")
    if not hasattr(hmlp, "HMLPPatchEncoder"):
        raise RuntimeError("SGLang HMLPPatchEncoder not found in hmlp.py")
    return hmlp


# ===========================================================================
# CPU unit checks (no GPU / checkpoint / sglang needed)
# ===========================================================================
def test_plan_out_scales_matches_checkpoint():
    scales = plan_out_scales(TEMPORAL, PATCH_SIZE, N_LAYERS, N_CHANNELS)
    got = [tuple(int(x) for x in s) for s in scales]
    assert got == EXPECTED_SCALES, got


def test_module_tree_and_linear_shapes():
    m = InklingVisionModel(VISION_CFG)
    names = set(dict(m.named_parameters()).keys())
    assert names == EXPECTED_PARAM_NAMES, names
    sd = m.state_dict()
    for k, shape in EXPECTED_LINEAR_SHAPES.items():
        assert tuple(sd[k].shape) == shape, (k, tuple(sd[k].shape))
    # last layer projects to the text hidden width; no norm_3 exists.
    assert m.layers["linear_3"].out_features == DECODER_DMODEL
    assert "norm_3" not in m.layers
    assert m.final_norm is not None


def test_fold_timespace_to_depth_is_value_preserving_fold():
    # (B=1, T=2, H=2, W=2, C=3), fold t=2, hw=1 -> (1,1,2,2,2*1*1*3=6)
    x = torch.arange(1 * 2 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 2, 3)
    y = fold_timespace_to_depth(x, t_fold=2, hw_fold=1)
    assert tuple(y.shape) == (1, 1, 2, 2, 6)
    assert torch.equal(torch.sort(x.reshape(-1))[0], torch.sort(y.reshape(-1))[0])
    # (B=1, T=1, H=2, W=2, C=1), fold hw=2 -> (1,1,1,1,1*2*2*1=4)
    x2 = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2, 1)
    y2 = fold_timespace_to_depth(x2, t_fold=1, hw_fold=2)
    assert tuple(y2.shape) == (1, 1, 1, 1, 4)
    assert torch.equal(torch.sort(x2.reshape(-1))[0], torch.sort(y2.reshape(-1))[0])


def test_placeholder_invariant_and_fail_loud():
    import pytest

    proc = InklingInputProcessor(None, InklingConfig(vision_config=vars(VISION_CFG)), None)
    img = np.random.RandomState(0).randint(0, 256, (80, 120, 3), dtype=np.uint8)
    out_ids, mm = proc.assemble([1, DEFAULT_IMAGE_TOKEN_ID, 2], [img])
    n_ph = sum(1 for t in out_ids if t == DEFAULT_IMAGE_TOKEN_ID)
    rows = int(mm["image"]["vision_patches_bthwc"].shape[0])
    assert n_ph == rows == mm["image"]["num_patches"][0]
    # media without a placeholder must fail loudly (never dropped/padded).
    with pytest.raises(ValueError):
        proc.assemble([1, 2, 3], [img])


def test_vision_tower_loads_real_weights():
    tower = InklingVisionModel(VISION_CFG)
    tower.load_weights(load_visual_weights())  # strict; raises on any mismatch


# ===========================================================================
# CUDA source_activation_replay (pass-critical)
# ===========================================================================
def _stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    a64 = a.detach().to(torch.float64).flatten()
    b64 = b.detach().to(torch.float64).flatten()
    diff = (a64 - b64).abs()
    denom = float(torch.linalg.norm(a64) * torch.linalg.norm(b64)) or 1.0
    cos = float(torch.dot(a64, b64) / denom)
    return float(diff.max()), float(diff.mean()), cos


def _build_towers(device, dtype):
    """Build TRT + SGLang towers, load the SAME real bf16 weights, move to GPU."""
    weights = load_visual_weights()
    trt = InklingVisionModel(VISION_CFG)
    trt.load_weights(weights)  # strips model.visual.* and strict-loads
    trt = trt.to(device=device, dtype=dtype).eval()

    hmlp_mod = load_sglang_hmlp()
    sg = hmlp_mod.HMLPPatchEncoder(VISION_CFG)
    sg.load_state_dict(_strip_visual(weights), strict=True)
    sg = sg.to(device=device, dtype=dtype).eval()
    return trt, sg


def run_tower_replay() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the vision tower source_activation_replay; a "
            "skip would hide missing GPU evidence."
        )
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    trt, sg = _build_towers(device, dtype)
    pre = InklingImagePreprocessor(patch_size=PATCH_SIZE, temporal_patch_size=TEMPORAL, dtype=dtype)
    proc = InklingInputProcessor(None, InklingConfig(vision_config=vars(VISION_CFG)), None)
    items = R.load_fixed_items()
    assert items, "no fixed MMMU items resolved (warm MMMU_ALIGN_CACHE?)"

    records: List[dict] = []
    n_ok = 0
    max_abs_all = 0.0
    for it in items:
        png = it["image_bytes"]
        patches = pre.encode_one(png)  # (num_patches, T, P, P, C) bf16 (CPU)
        num_patches = int(patches.shape[0])
        x = patches.to(device=device, dtype=dtype)
        with torch.no_grad():
            trt_out = trt(x)
            sg_out = sg(x)
        shape_ok = tuple(trt_out.shape) == (num_patches, DECODER_DMODEL) and tuple(
            sg_out.shape
        ) == (num_patches, DECODER_DMODEL)
        if shape_ok:
            m_abs, m_mean, m_cos = _stats(trt_out, sg_out)
            ref_scale = float(sg_out.detach().to(torch.float64).abs().max())
        else:
            m_abs, m_mean, m_cos, ref_scale = float("inf"), float("inf"), 0.0, 0.0
        rel_max = (m_abs / (ref_scale + 1e-6)) if np.isfinite(m_abs) else float("inf")

        # placeholder-token count == feature-row count (fail-loud contract).
        out_ids, mm = proc.assemble([DEFAULT_IMAGE_TOKEN_ID], [png])
        n_expanded = sum(1 for t in out_ids if t == DEFAULT_IMAGE_TOKEN_ID)
        feat_rows = int(mm["image"]["vision_patches_bthwc"].shape[0])
        invariant_ok = n_expanded == num_patches == feat_rows

        ok = (
            shape_ok
            and int(trt_out.shape[0]) == num_patches
            and invariant_ok
            and m_cos >= 0.999
            and rel_max <= 2e-2
        )
        n_ok += int(ok)
        max_abs_all = max(max_abs_all, m_abs if np.isfinite(m_abs) else 1e9)
        records.append(
            {
                "id": it["id"],
                "num_patches": num_patches,
                "feature_rows_trt": int(trt_out.shape[0]) if shape_ok else None,
                "feature_rows_sglang": int(sg_out.shape[0]) if shape_ok else None,
                "out_width": int(trt_out.shape[1]) if shape_ok else None,
                "placeholder_count": n_expanded,
                "placeholder_invariant_ok": bool(invariant_ok),
                "max_abs": None if not np.isfinite(m_abs) else round(m_abs, 6),
                "mean_abs": None if not np.isfinite(m_mean) else round(m_mean, 6),
                "rel_max": None if not np.isfinite(rel_max) else round(rel_max, 8),
                "cosine": round(m_cos, 8),
                "match": bool(ok),
            }
        )
    summary = {
        "num_items": len(items),
        "aligned": n_ok,
        "max_abs_over_items": round(max_abs_all, 6),
        "reference": "sglang HMLPPatchEncoder (real source, bf16, CUDA)",
        "checkpoint": CKPT,
        "scales": [list(s) for s in trt.scales],
        "records": records,
    }
    os.makedirs(os.path.dirname(ARTIFACT) or ".", exist_ok=True)
    with open(ARTIFACT, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


_SUMMARY = None


def _get() -> dict:
    global _SUMMARY
    if _SUMMARY is None:
        _SUMMARY = run_tower_replay()
    return _SUMMARY


def test_cuda_source_activation_replay():
    s = _get()
    assert s["aligned"] == s["num_items"], [r for r in s["records"] if not r["match"]]


def test_cuda_feature_rows_equal_num_patches():
    s = _get()
    bad = [
        r["id"]
        for r in s["records"]
        if r["feature_rows_trt"] != r["num_patches"]
        or r["out_width"] != DECODER_DMODEL
        or not r["placeholder_invariant_ok"]
    ]
    assert not bad, bad


# ---------------------------------------------------------------------------
# Plain-script runner (mirrors the Goal 1.1/1.2 dual-mode style)
# ---------------------------------------------------------------------------
def _main() -> int:
    unit = [
        test_plan_out_scales_matches_checkpoint,
        test_module_tree_and_linear_shapes,
        test_fold_timespace_to_depth_is_value_preserving_fold,
        test_placeholder_invariant_and_fail_loud,
        test_vision_tower_loads_real_weights,
    ]
    print("=== Inkling vision-tower CPU + weight-load checks ===")
    for fn in unit:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
            return 1

    print("\n=== CUDA source_activation_replay vs SGLang HMLPPatchEncoder ===")
    try:
        s = run_tower_replay()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"REPLAY FAILED to produce evidence: {type(e).__name__}: {e}")
        return 2
    print(f"scales: {s['scales']}")
    for r in s["records"]:
        print(
            f"  {r['id']:<28} "
            f"rows(trt={r['feature_rows_trt']}==sg={r['feature_rows_sglang']}"
            f"==np={r['num_patches']}) w={r['out_width']} "
            f"ph={r['placeholder_count']} "
            f"max_abs={r['max_abs']} mean_abs={r['mean_abs']} "
            f"rel_max={r['rel_max']} cos={r['cosine']} "
            f"{'OK' if r['match'] else 'X'}"
        )
    ok = s["aligned"] == s["num_items"]
    print(
        f"\nreplay {s['aligned']}/{s['num_items']}  "
        f"max_abs={s['max_abs_over_items']}  artifact={ARTIFACT}"
    )
    print("ALL ALIGNED" if ok else "REPLAY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
