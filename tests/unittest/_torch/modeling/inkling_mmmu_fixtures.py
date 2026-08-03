"""Shared MMMU fixtures for the Inkling tests.

Fetches and caches the fixed MMMU validation items (HF datasets-server, with a
REST fallback) and exposes the importlib loader used to pull SGLang's reference
image-processing / eval modules off disk. No tests live here -- this is the
fixture module `inkling_vision_tower_test` and `inkling_input_processor_test`
import.
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

# Larger fixed pool for the MMMU vision-parity gate (Module 2 needs 50 paired
# samples). Every id below -- ``validation_<Config>_<offset+1>`` -- is present in
# the advanced bring-up's cached SGLang 858-item pool, so a 50-item TRT run pairs
# by id with no capture job. The first six entries ARE ``FIXED_ITEMS`` (the
# alignment canary); the remaining 44 spread one item per remaining MMMU
# discipline, then a second item per discipline, for a cross-domain sample.
PARITY_ITEMS: List[Tuple[str, int]] = [
    ("Math", 0),
    ("Math", 1),
    ("Physics", 0),
    ("Chemistry", 0),
    ("Accounting", 0),
    ("Computer_Science", 0),
    ("Agriculture", 0),
    ("Architecture_and_Engineering", 0),
    ("Art", 0),
    ("Art_Theory", 1),
    ("Basic_Medical_Science", 0),
    ("Biology", 0),
    ("Clinical_Medicine", 0),
    ("Design", 0),
    ("Diagnostics_and_Laboratory_Medicine", 0),
    ("Economics", 0),
    ("Electronics", 0),
    ("Energy_and_Power", 0),
    ("Finance", 0),
    ("Geography", 0),
    ("History", 0),
    ("Literature", 0),
    ("Manage", 0),
    ("Marketing", 0),
    ("Materials", 0),
    ("Mechanical_Engineering", 0),
    ("Music", 1),
    ("Pharmacy", 0),
    ("Psychology", 1),
    ("Public_Health", 0),
    ("Sociology", 0),
    ("Accounting", 1),
    ("Agriculture", 1),
    ("Architecture_and_Engineering", 1),
    ("Art", 1),
    ("Art_Theory", 2),
    ("Basic_Medical_Science", 1),
    ("Biology", 1),
    ("Chemistry", 1),
    ("Clinical_Medicine", 1),
    ("Computer_Science", 1),
    ("Design", 1),
    ("Diagnostics_and_Laboratory_Medicine", 1),
    ("Economics", 1),
    ("Electronics", 1),
    ("Energy_and_Power", 1),
    ("Finance", 1),
    ("Geography", 1),
    ("History", 1),
    ("Literature", 1),
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


def _fetch_and_cache_items(specs: List[Tuple[str, int]]) -> List[dict]:
    """Resolve ``(config, offset)`` MMMU items, preferring the on-disk cache;
    fetch+cache on miss."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    items: List[dict] = []
    for config, offset in specs:
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


def load_fixed_items() -> List[dict]:
    """Resolve the fixed real items (the alignment canary), preferring the on-disk
    cache; fetch+cache on miss."""
    return _fetch_and_cache_items(_selected_items())


def load_parity_items(n: Optional[int] = None) -> List[dict]:
    """Resolve the first ``n`` MMMU vision-parity items (default all
    ``PARITY_ITEMS``), fetching+caching each exactly like :func:`load_fixed_items`.

    ``inkling_mmmu_run.py`` uses this to reach the Module-2 ``n_paired == 50``
    bar: every id pairs by construction against the cached SGLang pool. ``n`` is
    clamped to the pool size."""
    specs = PARITY_ITEMS if n is None else PARITY_ITEMS[: max(0, int(n))]
    return _fetch_and_cache_items(specs)


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
