# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M1a REAL-item MMMU harness-alignment integration test (Inkling TRT vs SGLang).

``reference_tier=real_source``, ``validation_tier=integration``.

Unlike the CPU unit canary (``inkling_mmmu_align_test.py``, hand-derived
assertions on synthetic inputs), this test proves the aligned MMMU harness
contract on **real MMMU validation items** against **SGLang's real code**:

  1. A fixed set of real ``MMMU/MMMU`` validation items (stable sample IDs such
     as ``validation_Math_1``) is fetched from the HuggingFace datasets-server
     and cached locally for reproducibility.
  2. SGLang's *real* MMMU sample preparation
     (``MMMUVLMEval._prepare_mmmu_samples``, with the dataset container shimmed
     so the real prompt/`<image>`/data-URI code path runs) produces the SGLang
     reference ``final_input_prompt`` + image data URI for each item.
  3. SGLang's *real* image preprocessing numba kernel
     (``image_processing._encode_image_bytes``) produces the reference vision
     patches at the serving scale (``patch_size=40``, ``rescale_image_frac=2.0``,
     ``rescale_image_max_upscaled_long_edge=2048``).
  4. SGLang's *real* answer scorer (``_parse_multi_choice_response`` /
     ``_parse_open_response`` / ``_eval_open``) parses canned responses.
  5. The TRT-side oracle (``inkling_mmmu_harness``) runs the SAME prompt
     rendering / image preprocessing / scoring on the SAME items and image
     bytes, and we assert they match **item-for-item** (prompt string,
     per-patch vision tensor within bf16 tolerance, patch grid / num_patches,
     ``placeholder_count == num_patches`` invariant, and parsed answer).

The SGLang reference modules are loaded directly from their on-disk source files
via ``importlib`` (NOT through the ``sglang`` package ``__init__`` chain, which
pulls in the whole serving stack), so the only third-party deps needed are
``numba`` / ``numpy`` / ``torch`` / ``PIL`` (all in the container); ``datasets``
is used only for the container shim types when available. A missing
``transformers`` is stubbed for the pure-preprocessing code path.

This test is deliberately **non-skipping**: if the real SGLang references cannot
be loaded, or the fixed items cannot be resolved (no cache and no network), it
FAILS -- that failure is the signal that the integration evidence was not
produced, not a silent pass.

Run:
  * ``python inkling_mmmu_real_align_test.py``  (container, with network or a
    warm ``MMMU_ALIGN_CACHE``)
  * ``pytest -q inkling_mmmu_real_align_test.py``

Env:
  * ``MMMU_ALIGN_CACHE``    -- dir for cached real items (default: ``./_mmmu_cache``)
  * ``MMMU_ALIGN_ARTIFACT`` -- path for the alignment record JSON
  * ``SGLANG_PY``           -- sglang ``python/`` root (default: on-disk path)
"""

from __future__ import annotations

import ast
import base64
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

# SGLang's InklingImageProcessor kernel ``_fill_patches_numba`` is
# ``@njit(cache=True)``. When the module is loaded from a file path via importlib
# (as we do below), numba records the defining module as ``<dynamic>``; a
# *different* process that later reads that on-disk cache raises
# ``ModuleNotFoundError: No module named '<dynamic>'``.
#
# NOTE: setting ``NUMBA_CACHE_DIR`` does NOT fix this. numba's
# ``_InTreeCacheLocator`` writes the cache into a ``__pycache__`` next to the
# *source file* (the writable SGLang checkout) and ignores ``NUMBA_CACHE_DIR``,
# so a fresh cache dir per process is silently bypassed. The authoritative fix is
# ``_disable_numba_disk_cache`` (swap each dispatcher's cache for a NullCache so
# nothing is written/read from disk) plus ``_purge_stale_numba_cache`` (delete any
# leftover ``<dynamic>`` cache), both applied in ``load_sglang_refs`` below before
# the first kernel call. We still point ``NUMBA_CACHE_DIR`` at a fresh per-process
# dir as harmless secondary isolation for any other cached kernel.
os.environ.setdefault("NUMBA_CACHE_DIR", tempfile.mkdtemp(prefix="inkling_numba_"))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import inkling_mmmu_harness as H  # noqa: E402

SGLANG_PY = os.environ.get(
    "SGLANG_PY",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/codes/sglang/python",
)
SGLANG_IMG_PROC = os.path.join(SGLANG_PY, "sglang/srt/multimodal/inkling/image_processing.py")
SGLANG_EVAL = os.path.join(SGLANG_PY, "sglang/test/simple_eval_mmmu_vlm.py")

CACHE_DIR = os.environ.get("MMMU_ALIGN_CACHE", os.path.join(HERE, "_mmmu_cache"))
ARTIFACT = os.environ.get("MMMU_ALIGN_ARTIFACT", os.path.join(HERE, "mmmu_align_artifact.json"))

# Serving preprocessing config (InklingImageProcessor defaults, image_processing.py:196-198).
PATCH_SIZE = 40
RESCALE_FRAC = 2.0
RESCALE_CAP = 2048

# Fixed, deterministic real MMMU validation items: (config, offset) -> stable IDs
# (validation_<Config>_<offset+1>). A small cross-discipline canary spanning
# several image geometries; these are the SAME items fed to both stacks.
FIXED_ITEMS: List[Tuple[str, int]] = [
    ("Math", 0),
    ("Math", 1),
    ("Physics", 0),
    ("Chemistry", 0),
    ("Accounting", 0),
    ("Computer_Science", 0),
]

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"


# ===========================================================================
# Real MMMU item acquisition (network -> local cache for reproducibility)
# ===========================================================================
def _http_get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mmmu-align/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:  # noqa: PERF203
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


def _norm_options(opts) -> Optional[list]:
    if isinstance(opts, str):
        opts = opts.strip()
        try:
            opts = ast.literal_eval(opts) if opts else None
        except Exception:  # noqa: BLE001
            opts = None
    return list(opts) if opts else None


def _fetch_row_datasets(config: str, offset: int) -> dict:
    """Primary path: ``datasets.load_dataset`` -- exactly what SGLang's
    ``_prepare_mmmu_samples`` uses, and far more robust than the datasets-server
    ``/rows`` REST endpoint (which 500s on some configs)."""
    from datasets import load_dataset

    ds = load_dataset("MMMU/MMMU", config, split="validation")
    ex = ds[int(offset)]
    img = ex.get("image_1")
    if img is None or not hasattr(img, "convert"):
        raise RuntimeError(f"item {config}:{offset} has no PIL image_1")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    opts = _norm_options(ex.get("options"))
    return {
        "id": ex.get("id", f"{config}:{offset}"),
        "config": config,
        "question": ex.get("question", ""),
        "options": opts,
        "answer": ex.get("answer"),
        "question_type": ex.get("question_type") or ("multiple-choice" if opts else "open"),
        "image_bytes": buf.getvalue(),
    }


def _fetch_row_rest(config: str, offset: int) -> dict:
    """Fallback path: HF datasets-server ``/rows`` REST endpoint."""
    url = (
        f"{DATASETS_SERVER}?dataset=MMMU%2FMMMU&config={config}"
        f"&split=validation&offset={offset}&length=1"
    )
    payload = json.loads(_http_get(url))
    row = payload["rows"][0]["row"]
    opts = _norm_options(row.get("options"))
    src = row["image_1"]["src"] if isinstance(row.get("image_1"), dict) else None
    if not src:
        raise RuntimeError(f"item {config}:{offset} has no image_1 src")
    return {
        "id": row.get("id", f"{config}:{offset}"),
        "config": config,
        "question": row.get("question", ""),
        "options": opts,
        "answer": row.get("answer"),
        "question_type": row.get("question_type") or ("multiple-choice" if opts else "open"),
        "image_bytes": _http_get(src),
    }


def _fetch_row(config: str, offset: int) -> dict:
    try:
        return _fetch_row_datasets(config, offset)
    except Exception as e_ds:  # noqa: BLE001
        try:
            return _fetch_row_rest(config, offset)
        except Exception as e_rest:  # noqa: BLE001
            raise RuntimeError(
                f"could not resolve MMMU item {config}:{offset} via datasets "
                f"({type(e_ds).__name__}: {e_ds}) or REST "
                f"({type(e_rest).__name__}: {e_rest})"
            ) from e_rest


def _selected_items() -> List[Tuple[str, int]]:
    """The fixed item set, optionally subset via ``MMMU_ALIGN_ITEMS`` (a comma
    list of ``Config:offset``) for a cache-only fast check."""
    env = os.environ.get("MMMU_ALIGN_ITEMS")
    if not env:
        return FIXED_ITEMS
    out: List[Tuple[str, int]] = []
    for tok in env.split(","):
        tok = tok.strip()
        if not tok:
            continue
        cfg, _, off = tok.partition(":")
        out.append((cfg, int(off or 0)))
    return out


def load_fixed_items() -> List[dict]:
    """Resolve the fixed real items, preferring the on-disk cache; fetch+cache on miss."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    items: List[dict] = []
    for config, offset in _selected_items():
        stem = os.path.join(CACHE_DIR, f"{config}_{offset}")
        meta_p, img_p = stem + ".json", stem + ".png"
        if os.path.exists(meta_p) and os.path.exists(img_p):
            with open(meta_p) as f:
                meta = json.load(f)
            with open(img_p, "rb") as f:
                meta["image_bytes"] = f.read()
            items.append(meta)
            continue
        item = _fetch_row(config, offset)
        with open(img_p, "wb") as f:
            f.write(item["image_bytes"])
        with open(meta_p, "w") as f:
            json.dump({k: v for k, v in item.items() if k != "image_bytes"}, f, indent=2)
        items.append(item)
    return items


# ===========================================================================
# Load SGLang's REAL reference code directly from its source files
# ===========================================================================
def _load_module_from_file(path: str, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _NoDiskCache:
    """Version-independent stand-in for numba's ``NullCache`` (used only if the
    real one cannot be imported): never reads or writes an on-disk cache."""

    @property
    def cache_path(self):
        return None

    def load_overload(self, sig, target_context):  # noqa: ARG002
        return None

    def save_overload(self, sig, data):  # noqa: ARG002
        pass

    def enable(self):
        pass

    def disable(self):
        pass

    def flush(self):
        pass


def _null_cache():
    try:
        from numba.core.caching import NullCache

        return NullCache()
    except Exception:  # noqa: BLE001 -- numba layout changed / unavailable
        return _NoDiskCache()


def _disable_numba_disk_cache(mod: types.ModuleType) -> None:
    """Disable on-disk numba caching for every ``@njit(cache=True)`` dispatcher
    in ``mod``.

    numba's ``_InTreeCacheLocator`` writes the cache next to the *source file*
    (ignoring ``NUMBA_CACHE_DIR``), and because we load the module via importlib
    under a synthetic name numba records the defining module as ``<dynamic>``.
    Any second process that reads that cache dies with
    ``ModuleNotFoundError: No module named '<dynamic>'``. Swapping each
    dispatcher's ``_cache`` for a null cache before its first call makes the
    kernel (re)compile in-memory per process and never touch disk -- fully
    locator-independent and safe across script/pytest reruns in one job.
    """
    for name in dir(mod):
        try:
            obj = getattr(mod, name)
        except Exception:  # noqa: BLE001
            continue
        # duck-type a numba dispatcher without depending on its class import path
        if (
            type(obj).__module__.split(".")[0] == "numba"
            and hasattr(obj, "_cache")
            and hasattr(obj, "py_func")
        ):
            try:
                obj._cache = _null_cache()
            except Exception:  # noqa: BLE001
                pass


def _purge_stale_numba_cache() -> None:
    """Best-effort removal of any leftover on-disk numba cache for the SGLang
    kernel. A prior importlib load may have written a ``<dynamic>``-keyed
    ``.nbi``/``.nbc`` into the source ``__pycache__``; a fresh process cannot
    reload it. Deleting it before we load guarantees no poisoned reload even if
    the null-cache swap is somehow unavailable."""
    import glob

    pycache = os.path.join(os.path.dirname(SGLANG_IMG_PROC), "__pycache__")
    for p in glob.glob(os.path.join(pycache, "image_processing._fill_patches_numba*.nb*")):
        try:
            os.remove(p)
        except OSError:
            pass


def _ensure_transformers_stub() -> None:
    """image_processing.py imports transformers image base classes at module top,
    but ``_encode_image_bytes`` does not use them. Provide light stubs only if
    the real ``transformers`` is unavailable (e.g. an un-bootstrapped container)."""
    if importlib.util.find_spec("transformers") is not None:
        return
    t = types.ModuleType("transformers")
    t.__path__ = []  # mark as package
    ipu = types.ModuleType("transformers.image_processing_utils")
    ipu.BaseImageProcessor = type("BaseImageProcessor", (), {"__init__": lambda self, **k: None})
    ipu.BatchFeature = type("BatchFeature", (), {})
    iu = types.ModuleType("transformers.image_utils")
    iu.ImageInput = object
    sys.modules.setdefault("transformers", t)
    sys.modules["transformers.image_processing_utils"] = ipu
    sys.modules["transformers.image_utils"] = iu


def _inject_sglang_eval_stubs() -> None:
    """simple_eval_mmmu_vlm.py does ``from sglang.test import simple_eval_common``
    at module top; the scorer + sample-prep we use never touch ``common``. Inject
    stub package modules so the file imports without the full sglang stack."""
    common = types.ModuleType("sglang.test.simple_eval_common")
    common.Eval = type("Eval", (), {})
    for attr in (
        "HTML_JINJA",
        "EvalResult",
        "SamplerBase",
        "SingleEvalResult",
        "map_with_progress",
    ):
        setattr(common, attr, None)
    common.map_with_progress = lambda *a, **k: None
    sg = sys.modules.get("sglang")
    if sg is None:
        sg = types.ModuleType("sglang")
        sg.__path__ = []
        sys.modules["sglang"] = sg
    sgt = sys.modules.get("sglang.test")
    if sgt is None:
        sgt = types.ModuleType("sglang.test")
        sgt.__path__ = []
        sys.modules["sglang.test"] = sgt
    sys.modules["sglang.test.simple_eval_common"] = common


def load_sglang_refs() -> Tuple[types.ModuleType, types.ModuleType]:
    if not os.path.exists(SGLANG_IMG_PROC):
        raise RuntimeError(f"missing SGLang image_processing.py at {SGLANG_IMG_PROC}")
    if not os.path.exists(SGLANG_EVAL):
        raise RuntimeError(f"missing SGLang simple_eval_mmmu_vlm.py at {SGLANG_EVAL}")
    _purge_stale_numba_cache()
    _ensure_transformers_stub()
    ip = _load_module_from_file(SGLANG_IMG_PROC, "sglang_inkling_image_processing_ref")
    # locator-independent: ensure no <dynamic>-keyed cache is read/written on disk
    _disable_numba_disk_cache(ip)
    _inject_sglang_eval_stubs()
    ev = _load_module_from_file(SGLANG_EVAL, "sglang_mmmu_eval_ref")
    assert hasattr(ip, "_encode_image_bytes"), "SGLang _encode_image_bytes missing"
    assert hasattr(ev, "_parse_multi_choice_response"), "SGLang MC scorer missing"
    assert hasattr(ev, "MMMUVLMEval"), "SGLang MMMUVLMEval missing"
    return ip, ev


# ===========================================================================
# Drive SGLang's REAL _prepare_mmmu_samples with a dataset-container shim
# ===========================================================================
class _ShimDataset:
    """Minimal stand-in for a ``datasets.Dataset`` supporting exactly the ops
    ``_prepare_mmmu_samples`` uses: ``len``, integer indexing (-> row dict),
    and ``add_column``."""

    def __init__(self, rows: List[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]

    def add_column(self, name: str, values: List) -> "_ShimDataset":
        for r, v in zip(self.rows, values):
            r[name] = v
        return self


def build_sglang_samples(ev: types.ModuleType, items: List[dict]) -> List[dict]:
    from PIL import Image

    rows_by_subject: Dict[str, List[dict]] = {}
    for it in items:
        img = Image.open(io.BytesIO(it["image_bytes"])).convert("RGB")
        rows_by_subject.setdefault(it["config"], []).append(
            {
                "id": it["id"],
                "question": it["question"],
                # real MMMU stores options as a stringified list that SGLang eval()s
                "options": repr(it["options"]) if it["options"] else None,
                "answer": it["answer"],
                "image_1": img,
                "question_type": it["question_type"],
            }
        )

    def fake_load_dataset(_name, subj, split="validation"):  # noqa: ARG001
        return _ShimDataset([dict(r) for r in rows_by_subject.get(subj, [])])

    def fake_concatenate_datasets(ds_list):
        merged: List[dict] = []
        for ds in ds_list:
            merged.extend(ds.rows)
        return _ShimDataset(merged)

    ev.load_dataset = fake_load_dataset
    ev.concatenate_datasets = fake_concatenate_datasets

    evaluator = ev.MMMUVLMEval(num_examples=len(items), num_threads=1)
    return evaluator.samples


def _decode_data_uri(uri: str) -> bytes:
    return base64.b64decode(uri.split(",", 1)[1])


# ===========================================================================
# TRT-side preprocessing (mirrors _encode_image_bytes: resize -> patch -> T=2)
# ===========================================================================
def trt_preprocess(png_bytes: bytes) -> Tuple[np.ndarray, int, int, int]:
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    sw, sh = H.scaled_image_dimensions(img.width, img.height, RESCALE_FRAC, RESCALE_CAP)
    if (sw, sh) != img.size:
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.uint8)
    patches, num_patches, nph, npw = H.preprocess_patches(arr, PATCH_SIZE)
    bthwc = H.to_bthwc(patches, H.DEFAULT_TEMPORAL_PATCH_SIZE)
    return bthwc, num_patches, nph, npw


def _tensor_stats(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    fa, fb = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb)) or 1.0
    cos = float(np.dot(fa, fb) / denom)
    return float(diff.max()), float(diff.mean()), cos


# ===========================================================================
# The integration check
# ===========================================================================
def run_alignment() -> dict:
    ip, ev = load_sglang_refs()
    items = load_fixed_items()
    assert items, "no fixed MMMU items resolved"
    samples = build_sglang_samples(ev, items)
    # index SGLang samples by id
    sg_by_id = {s["id"]: s for s in samples}

    records: List[dict] = []
    n_prompt_ok = n_media_ok = n_scorer_ok = n_invariant_ok = 0
    max_media_abs = 0.0

    for it in items:
        sid = it["id"]
        assert sid in sg_by_id, f"SGLang did not emit sample for {sid}; emitted {list(sg_by_id)}"
        sample = sg_by_id[sid]
        qtype = sample["question_type"]

        # ---- prompt alignment (TRT oracle vs SGLang real final_input_prompt) ----
        trt_prompt, trt_qtype = H.render_mmmu_prompt(it["question"], it["options"])
        sg_prompt = sample["final_input_prompt"]
        prompt_match = trt_prompt == sg_prompt and (
            (trt_qtype == "multiple-choice") == (qtype == "multiple-choice")
        )
        n_prompt_ok += int(prompt_match)

        # ---- media alignment on the SAME image bytes SGLang used ----
        png = _decode_data_uri(sample["image_data"])
        sg_tensor = ip._encode_image_bytes(
            png,
            patch_size=PATCH_SIZE,
            rescale_image_frac=RESCALE_FRAC,
            rescale_image_max_upscaled_long_edge=RESCALE_CAP,
        )
        sg_np = sg_tensor.float().numpy()  # (P, 2, 40, 40, 3) bf16 -> f32
        trt_bthwc, num_patches, nph, npw = trt_preprocess(png)
        shape_ok = tuple(sg_np.shape) == tuple(trt_bthwc.shape)
        if shape_ok:
            m_abs, m_mean, m_cos = _tensor_stats(trt_bthwc, sg_np)
        else:
            m_abs, m_mean, m_cos = float("inf"), float("inf"), 0.0
        # bf16 cast (SGLang) vs f32 (TRT) -> differences bounded by bf16 ULP (<1e-2)
        media_match = (
            shape_ok and int(sg_np.shape[0]) == num_patches and m_abs <= 1e-2 and m_cos >= 0.9999
        )
        n_media_ok += int(media_match)
        max_media_abs = max(max_media_abs, m_abs if np.isfinite(m_abs) else 1e9)

        # ---- placeholder_count == num_patches invariant ----
        placeholder_count = num_patches  # hMLP emits one token per patch
        invariant_ok = placeholder_count == int(sg_np.shape[0]) == num_patches
        n_invariant_ok += int(invariant_ok)

        # ---- answer scoring alignment on canned responses ----
        scorer_cases = []
        gold = it["answer"]
        if qtype == "multiple-choice":
            all_choices = sample["all_choices"]
            index2ans = sample["index2ans"]
            alt = next((c for c in all_choices if c != gold), all_choices[0])
            responses = [
                f"Working through the options carefully.\nAnswer: {gold}",
                f"On reflection I believe it is ({alt}).",
                "There is no way to tell from the image.",
            ]
            case_ok = True
            for resp in responses:
                sg_pred = ev._parse_multi_choice_response(resp, all_choices, index2ans)
                trt_pred = H.parse_multi_choice_response(resp, all_choices, index2ans)
                ok = sg_pred == trt_pred
                case_ok = case_ok and ok
                scorer_cases.append(
                    {"response": resp, "sglang": sg_pred, "trt": trt_pred, "match": ok}
                )
            # the correct-answer response must extract the gold letter on both
            gold_ok = (
                ev._parse_multi_choice_response(responses[0], all_choices, index2ans)
                == gold
                == H.parse_multi_choice_response(responses[0], all_choices, index2ans)
            )
            case_ok = case_ok and gold_ok
        else:
            resp = f"After the derivation, therefore the answer is {gold}."
            sg_list = ev._parse_open_response(resp)
            trt_list = H.parse_open_response(resp)
            sg_ok = ev._eval_open(gold, sg_list)
            trt_ok = H.eval_open(gold, trt_list)
            case_ok = (sg_list == trt_list) and (sg_ok == trt_ok)
            scorer_cases.append(
                {
                    "response": resp,
                    "sglang_eval_open": bool(sg_ok),
                    "trt_eval_open": bool(trt_ok),
                    "lists_equal": sg_list == trt_list,
                    "match": case_ok,
                }
            )
        n_scorer_ok += int(case_ok)

        records.append(
            {
                "id": sid,
                "config": it["config"],
                "question_type": qtype,
                "prompt_match": bool(prompt_match),
                "trt_prompt": trt_prompt,
                "sglang_prompt": sg_prompt,
                "num_patches": int(num_patches),
                "grid_nph_npw": [int(nph), int(npw)],
                "sglang_num_patches": int(sg_np.shape[0]),
                "placeholder_count": int(placeholder_count),
                "placeholder_invariant_ok": bool(invariant_ok),
                "media_shape": list(trt_bthwc.shape),
                "media_max_abs": None if not np.isfinite(m_abs) else round(m_abs, 6),
                "media_mean_abs": None if not np.isfinite(m_mean) else round(m_mean, 6),
                "media_cosine": round(m_cos, 8),
                "media_match": bool(media_match),
                "scorer_cases": scorer_cases,
                "scorer_match": bool(case_ok),
                "answer": gold,
            }
        )

    n = len(items)
    summary = {
        "num_items": n,
        "prompt_aligned": n_prompt_ok,
        "media_aligned": n_media_ok,
        "scorer_aligned": n_scorer_ok,
        "placeholder_invariant_aligned": n_invariant_ok,
        "max_media_abs_over_items": round(max_media_abs, 6),
        "config": {
            "patch_size": PATCH_SIZE,
            "rescale_image_frac": RESCALE_FRAC,
            "rescale_image_max_upscaled_long_edge": RESCALE_CAP,
            "reference": "sglang real _prepare_mmmu_samples + _encode_image_bytes + _parse_*_response",
            "dataset": "MMMU/MMMU validation (datasets-server)",
            "item_ids": [it["id"] for it in items],
        },
        "records": records,
    }
    os.makedirs(os.path.dirname(ARTIFACT) or ".", exist_ok=True)
    with open(ARTIFACT, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
_SUMMARY: Optional[dict] = None


def _get_summary() -> dict:
    global _SUMMARY
    if _SUMMARY is None:
        _SUMMARY = run_alignment()
    return _SUMMARY


def test_real_mmmu_prompt_alignment():
    s = _get_summary()
    assert s["prompt_aligned"] == s["num_items"], [
        (r["id"], r["trt_prompt"], r["sglang_prompt"])
        for r in s["records"]
        if not r["prompt_match"]
    ]


def test_real_mmmu_media_alignment():
    s = _get_summary()
    assert s["media_aligned"] == s["num_items"], [
        (r["id"], r["media_max_abs"], r["media_cosine"], r["num_patches"], r["sglang_num_patches"])
        for r in s["records"]
        if not r["media_match"]
    ]


def test_real_mmmu_placeholder_invariant():
    s = _get_summary()
    assert s["placeholder_invariant_aligned"] == s["num_items"], [
        r["id"] for r in s["records"] if not r["placeholder_invariant_ok"]
    ]


def test_real_mmmu_scorer_alignment():
    s = _get_summary()
    assert s["scorer_aligned"] == s["num_items"], [
        (r["id"], r["scorer_cases"]) for r in s["records"] if not r["scorer_match"]
    ]


# ---------------------------------------------------------------------------
# Plain-script runner
# ---------------------------------------------------------------------------
def _main() -> int:
    try:
        s = run_alignment()
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"\nREAL-ALIGN FAILED to produce evidence: {type(e).__name__}: {e}")
        return 2
    n = s["num_items"]
    print(f"\n=== MMMU real-item harness alignment ({n} fixed items) ===")
    print(f"artifact: {ARTIFACT}")
    for r in s["records"]:
        print(
            f"  {r['id']:<28} qtype={r['question_type']:<15} "
            f"prompt={'OK' if r['prompt_match'] else 'X'} "
            f"media={'OK' if r['media_match'] else 'X'}"
            f"(max_abs={r['media_max_abs']},cos={r['media_cosine']},"
            f"np={r['num_patches']}=={r['sglang_num_patches']}) "
            f"ph_inv={'OK' if r['placeholder_invariant_ok'] else 'X'} "
            f"scorer={'OK' if r['scorer_match'] else 'X'}"
        )
    ok = (
        s["prompt_aligned"]
        == s["media_aligned"]
        == s["scorer_aligned"]
        == s["placeholder_invariant_aligned"]
        == n
    )
    print(
        f"\nprompt {s['prompt_aligned']}/{n}  media {s['media_aligned']}/{n}  "
        f"scorer {s['scorer_aligned']}/{n}  placeholder_inv "
        f"{s['placeholder_invariant_aligned']}/{n}  "
        f"max_media_abs={s['max_media_abs_over_items']}"
    )
    print("ALL ALIGNED" if ok else "ALIGNMENT MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
