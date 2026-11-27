# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tactic pins shipped with the MiniMax-M3.1-Flash sm_107 targets.

Each target names its topology (``tp2``, ``tp4``, ``tp8``, ...). Before the warmup tuning pass, ``install`` loads the
topology's files from ``m31_tuning/sm107/<topology>/`` in this package:

* ``autotuner_cache.json``: an ordinary TensorRT-LLM AutoTuner cache file (``AutoTunerProfilingCache.save_cache``
  format) whose entries carry their tactic's name (``TunableRunner.tactic_name``, which the AutoTuner records beside
  every entry it tunes). The AutoTuner loads it (``load_bundle``) and checks each entry against its runner on the first
  lookup: a tactic of that name is served, otherwise the entry is dropped with an ``[M31_PIN] unresolved`` line and
  tuned again. Every shape the warmup tunes that the file does not cover logs ``[M31_PIN] retune``. Both are counted per
  rank and reported at exit. An entry without a name (an index a rebuild may reorder) is not loaded; it is reported
  as unresolved and tuned again.
* ``flashinfer_rank_<n>.json``: FlashInfer's own per-rank autotune cache (it keys its configs by index, so it cannot be
  keyed by name). It is served, load only, when its guard (the FlashInfer and CuTe DSL versions it was tuned with)
  matches the installed ones; otherwise one ``[M31_PIN] unresolved flashinfer guard`` line and FlashInfer keeps its
  own selection. The served copy lives under ``TRTLLM_CACHE_DIR`` (``flashinfer_autotune/bundled/``) when it is set.

``TLLM_AUTOTUNER_CACHE_PATH`` (an explicit AutoTuner cache) takes precedence over the bundle; the path
``tensorrt_llm`` derives from ``TRTLLM_CACHE_DIR`` is not explicit, so the bundle is installed and the AutoTuner then
loads and saves that cache file over it. ``TRTLLM_M31_PIN_DIR`` replaces the bundled directory.
``TRTLLM_M31_DISABLE_BUNDLED_PINS=1`` (the family switch ``BUNDLED_PINS``) loads nothing.

``python -m tensorrt_llm._torch.modeling_v2.models.minimax_m3.m31_pins`` makes a bundle from a tuning run's standard
files (its AutoTuner cache file, which records the tactic names, and FlashInfer's per-rank files).
"""

import argparse
import atexit
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .m31_switches import m31_disabled

AUTOTUNER_FILE = "autotuner_cache.json"
DISABLE_ID = "BUNDLED_PINS"
DISABLE_ENV = f"TRTLLM_M31_DISABLE_{DISABLE_ID}"
DIR_ENV = "TRTLLM_M31_PIN_DIR"
AUTOTUNER_CACHE_ENV = "TLLM_AUTOTUNER_CACHE_PATH"
UNIFIED_CACHE_ENV = "TRTLLM_CACHE_DIR"
BUNDLE_ROOT = Path(__file__).resolve().parent / "m31_tuning" / "sm107"


def _disabled() -> bool:
    return m31_disabled(DISABLE_ID)


def _explicit_autotuner_cache() -> Optional[str]:
    """The AutoTuner cache path set for this process, or None when there is none or it is the unified cache's
    default (``tensorrt_llm`` sets it at import from ``TRTLLM_CACHE_DIR``)."""
    path = os.environ.get(AUTOTUNER_CACHE_ENV)
    root = os.environ.get(UNIFIED_CACHE_ENV)
    if path and root:
        from tensorrt_llm._bootstrap import _unified_cache_defaults

        if path == _unified_cache_defaults(root)[AUTOTUNER_CACHE_ENV]:
            return None
    return path or None


def bundle_dir(topology: str) -> Path:
    """The bundle directory of ``topology`` (it may not exist)."""
    root = Path(os.environ[DIR_ENV]) if os.environ.get(DIR_ENV) else BUNDLE_ROOT
    return root / topology


def flashinfer_file(directory: Path, rank: int) -> Path:
    return directory / f"flashinfer_rank_{rank}.json"


def _info(message: str) -> None:
    from tensorrt_llm.logger import logger

    logger.info(message)


class PinReporter:
    """Counts and logs the bundled entries no tactic matches and the shapes the bundle does not cover."""

    def __init__(self, topology: str, rank: int) -> None:
        self.topology = topology
        self.rank = rank
        self.unresolved_count = 0
        self.retune_count = 0
        self._final_logged = False

    def unresolved(self, cache_key: Tuple, name: Optional[str]) -> None:
        self.unresolved_count += 1
        _info(
            f"[M31_PIN] unresolved rank={self.rank} op={cache_key[0]} runner={cache_key[1]} "
            f"shapes={cache_key[3]} tactic={name!r}"
        )

    def unresolved_guard(self, what: str, want: Dict[str, Any], have: Dict[str, Any]) -> None:
        self.unresolved_count += 1
        _info(
            f"[M31_PIN] unresolved {what} guard rank={self.rank}: bundle {want}, installed {have}"
        )

    def retune(self, custom_op: str, shapes: Any) -> None:
        self.retune_count += 1
        _info(f"[M31_PIN] retune rank={self.rank} op={custom_op} shapes={shapes}")

    def final(self) -> None:
        if self._final_logged:
            return
        self._final_logged = True
        _info(
            f"[M31_PIN] final rank={self.rank} topology={self.topology}: unresolved {self.unresolved_count} "
            f"retune {self.retune_count}"
        )


def _flashinfer_versions() -> Dict[str, Optional[str]]:
    import importlib.metadata as md

    versions: Dict[str, Optional[str]] = {}
    for dist in ("flashinfer-python", "nvidia-cutlass-dsl"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    return versions


def _served_flashinfer_dir(rank: int) -> Path:
    """Where this rank's bundled FlashInfer file is served from: under ``TRTLLM_CACHE_DIR`` when it is set (the
    unified cache root), else a temporary directory removed at exit."""
    root = os.environ.get(UNIFIED_CACHE_ENV)
    if root:
        served = (
            Path(os.path.expanduser(root)) / "flashinfer_autotune" / "bundled" / f"pid{os.getpid()}"
        )
        served.mkdir(parents=True, exist_ok=True)
    else:
        served = Path(tempfile.mkdtemp(prefix=f"m31_pins_rank{rank}_"))
    # Read only while the process runs.
    atexit.register(shutil.rmtree, str(served), True)
    return served


def install(topology: str, rank: int, tuner: Optional[Any] = None) -> Optional[PinReporter]:
    """Load this rank's bundled pins into the AutoTuner (and FlashInfer's autotune cache). Returns the reporter, or
    None when nothing was loaded (switched off, an explicit AutoTuner cache, or no bundle)."""
    if _disabled():
        _info(f"[M31_PIN] bundled pins off ({DISABLE_ENV}=1) rank={rank}")
        return None
    if _explicit_autotuner_cache():
        _info(f"[M31_PIN] bundled pins not loaded rank={rank}: {AUTOTUNER_CACHE_ENV} is set")
        return None
    directory = bundle_dir(topology)
    path = directory / AUTOTUNER_FILE
    fi_path = flashinfer_file(directory, rank)
    if not path.is_file() and not fi_path.is_file():
        _info(f"[M31_PIN] no bundle rank={rank} topology={topology} ({directory})")
        return None

    if tuner is None:
        from tensorrt_llm._torch.autotuner import AutoTuner

        tuner = AutoTuner.get()
    reporter = PinReporter(topology, rank)
    loaded = 0
    if path.is_file():
        cache = tuner.profiling_cache
        before = len(cache.bundled_names)
        for key in cache.load_bundle(path, rank):
            reporter.unresolved(key, None)
        loaded = len(cache.bundled_names) - before
    tuner.bundle_reporter = reporter

    fi_state = "none"
    if fi_path.is_file():
        flashinfer = json.loads(fi_path.read_text())
        have = _flashinfer_versions()
        if flashinfer.get("guard") == have:
            from tensorrt_llm._torch import flashinfer_autotune_cache

            served = _served_flashinfer_dir(rank) / f"rank{rank}.json"
            served.write_text(json.dumps(flashinfer["cache"]))
            flashinfer_autotune_cache.use_bundled_file(str(served))
            fi_state = f"{len(flashinfer['cache']) - ('_metadata' in flashinfer['cache'])} shapes"
        else:
            reporter.unresolved_guard("flashinfer", flashinfer.get("guard"), have)
            fi_state = "guard mismatch"
    atexit.register(reporter.final)
    _info(
        f"[M31_PIN] loaded rank={rank} topology={topology}: {loaded} AutoTuner entries, FlashInfer {fi_state}"
    )
    return reporter


# ---------------------------------------------------------------------------------------------------------------
# Converter: a tuning run's standard files -> a bundle
# ---------------------------------------------------------------------------------------------------------------


def _named_entries(section: Dict[str, Any]) -> Dict[str, Any]:
    """The entries of an AutoTuner cache file section that carry a tactic name (an index alone does not survive a
    rebuild)."""
    return {key: value for key, value in section.items() if value.get("tactic_name")}


def convert(
    cache_file: Optional[Path], fi_dir: Optional[Path], out_dir: Path, ranks: int
) -> List[Path]:
    """Write a bundle into ``out_dir`` from a tuning run's AutoTuner cache file (its named entries of every rank
    0..ranks-1) and FlashInfer rank files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if cache_file is not None:
        data = json.loads(Path(cache_file).read_text())
        bundle: Dict[str, Any] = {"metadata": data.get("metadata", {})}
        for section in ["shared"] + [f"rank_{rank}" for rank in range(ranks)]:
            bundle[section] = _named_entries(data.get(section, {}))
        path = out_dir / AUTOTUNER_FILE
        path.write_text(json.dumps(bundle, indent=1, sort_keys=True) + "\n")
        written.append(path)
    if fi_dir is not None:
        guard = _flashinfer_versions()
        for rank in range(ranks):
            fi_file = Path(fi_dir) / f"rank{rank}.json"
            if not fi_file.is_file():
                raise FileNotFoundError(f"{fi_file}: every rank needs its FlashInfer file")
            path = flashinfer_file(out_dir, rank)
            path.write_text(
                json.dumps({"guard": guard, "cache": json.loads(fi_file.read_text())}, indent=1)
                + "\n"
            )
            written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--autotuner-cache", type=Path, help="TLLM_AUTOTUNER_CACHE_PATH file of the tuning run"
    )
    parser.add_argument(
        "--flashinfer-dir", type=Path, help="TLLM_FLASHINFER_AUTOTUNE_CACHE_DIR of the tuning run"
    )
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="m31_tuning/sm107/<topology>")
    args = parser.parse_args()
    for path in convert(args.autotuner_cache, args.flashinfer_dir, args.out, args.ranks):
        print(path)


if __name__ == "__main__":
    main()
