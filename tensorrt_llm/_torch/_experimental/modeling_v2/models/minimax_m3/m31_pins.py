# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tactic pins shipped with the MiniMax-M3.1 sm_107 targets.

Each target names its topology (``tp2``, ``tp4``, ``tp8``, ...). Before the warmup tuning pass, ``install`` loads
``m31_tuning/sm107/<topology>/rank_<n>.json`` (format ``m31_pin_v1``) from this package:

* ``autotuner``: the TensorRT-LLM AutoTuner entries, each keyed by op, runner, runner id and tuning shapes and
  carrying its tactic by NAME (``TunableRunner.tactic_name``). On the first lookup of an entry the AutoTuner checks
  that name against the runner: a tactic of that name is served, otherwise the entry is dropped with an
  ``[M31_PIN] unresolved`` line and tuned again. Every shape the warmup tunes that the bundle does not cover logs
  ``[M31_PIN] retune``. Both are counted per rank and reported at exit. An entry of a runner without tactic names
  (``tactic_name`` null) is index-keyed: it is served only when the bundle's native guard (the md5s of the native
  libraries it was tuned with) matches the installed ones; otherwise every such entry is dropped (one
  ``[M31_PIN] unresolved native guard`` line) and its shapes are tuned again.
* ``flashinfer``: FlashInfer's own per-rank autotune cache (it keys its configs by index, so it cannot be keyed by
  name here). It is served, load only, when its guard (the FlashInfer and CuTe DSL versions it was tuned with)
  matches the installed ones; otherwise one ``[M31_PIN] unresolved flashinfer guard`` line and FlashInfer keeps its
  own selection.

``TLLM_AUTOTUNER_CACHE_PATH`` (an explicit AutoTuner cache) takes precedence over the bundle; the path
``tensorrt_llm`` derives from ``TRTLLM_CACHE_DIR`` is not explicit, so the bundle is installed and the AutoTuner then
loads and saves that cache file over it. ``TRTLLM_M31_PIN_DIR`` replaces the bundled directory.
``TRTLLM_M31_DISABLE_BUNDLED_PINS=1`` loads nothing.

``python -m tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_pins`` converts a tuning run's
standard files (an AutoTuner cache file and FlashInfer's per-rank files) into the bundle format; it needs the native
runners, so it runs on the target GPU with the wheel installed.
"""

import argparse
import ast
import atexit
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FORMAT = "m31_pin_v1"
DISABLE_ENV = "TRTLLM_M31_DISABLE_BUNDLED_PINS"
DIR_ENV = "TRTLLM_M31_PIN_DIR"
AUTOTUNER_CACHE_ENV = "TLLM_AUTOTUNER_CACHE_PATH"
UNIFIED_CACHE_ENV = "TRTLLM_CACHE_DIR"
BUNDLE_ROOT = Path(__file__).resolve().parent / "m31_tuning" / "sm107"


def _disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


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


def pin_file(topology: str, rank: int) -> Path:
    """This rank's bundle file of ``topology`` (it may not exist)."""
    root = Path(os.environ[DIR_ENV]) if os.environ.get(DIR_ENV) else BUNDLE_ROOT
    return root / topology / f"rank_{rank}.json"


def _info(message: str) -> None:
    from tensorrt_llm.logger import logger

    logger.info(message)


class PinReporter:
    """Counts and logs the bundled entries no tactic matches and the shapes the bundle does not cover."""

    def __init__(self, topology: str, rank: int):
        self.topology = topology
        self.rank = rank
        self.unresolved_count = 0
        self.retune_count = 0
        self._final_logged = False

    def unresolved(self, cache_key: Tuple, name: str) -> None:
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


def _key(entry: Dict[str, Any]) -> Tuple:
    """The AutoTuner cache key of a bundle entry: (op, runner class, runner id, shapes as nested tuples)."""
    shapes = tuple(tuple(int(d) for d in dims) for dims in entry["shapes"])
    return (entry["op"], entry["runner"], entry["runner_key"], shapes)


def _flashinfer_versions() -> Dict[str, Optional[str]]:
    import importlib.metadata as md

    versions: Dict[str, Optional[str]] = {}
    for dist in ("flashinfer-python", "nvidia-cutlass-dsl"):
        try:
            versions[dist] = md.version(dist)
        except md.PackageNotFoundError:
            versions[dist] = None
    return versions


_NATIVE_LIBS = ("libtensorrt_llm.so", "libth_common.so")
_native_md5s: Dict[str, Optional[str]] = {}


def _native_guard() -> Dict[str, Optional[str]]:
    """The md5s of the native libraries whose tactic lists index-keyed entries refer to (computed once)."""
    if not _native_md5s:
        import hashlib

        import tensorrt_llm

        libs = Path(tensorrt_llm.__file__).resolve().parent / "libs"
        for name in _NATIVE_LIBS:
            path = libs / name
            if path.is_file():
                digest = hashlib.md5()
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 24), b""):
                        digest.update(block)
                _native_md5s[name] = digest.hexdigest()
            else:
                _native_md5s[name] = None
    return dict(_native_md5s)


def install(topology: str, rank: int, tuner: Optional[Any] = None) -> Optional[PinReporter]:
    """Load this rank's bundled pins into the AutoTuner (and FlashInfer's autotune cache). Returns the reporter, or
    None when nothing was loaded (switched off, an explicit AutoTuner cache, or no bundle file)."""
    if _disabled():
        _info(f"[M31_PIN] bundled pins off ({DISABLE_ENV}=1) rank={rank}")
        return None
    if _explicit_autotuner_cache():
        _info(f"[M31_PIN] bundled pins not loaded rank={rank}: {AUTOTUNER_CACHE_ENV} is set")
        return None
    path = pin_file(topology, rank)
    if not path.is_file():
        _info(f"[M31_PIN] no bundle rank={rank} topology={topology} ({path})")
        return None
    data = json.loads(path.read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: format {data.get('format')!r}, expected {FORMAT!r}")

    if tuner is None:
        from tensorrt_llm._torch.autotuner import AutoTuner

        tuner = AutoTuner.get()
    reporter = PinReporter(topology, rank)
    cache = tuner.profiling_cache
    entries = data.get("autotuner", [])
    unnamed = [e for e in entries if e.get("tactic_name") is None]
    if unnamed:
        have = _native_guard()
        if data.get("native_guard") != have:
            reporter.unresolved_guard(
                f"native ({len(unnamed)} index-keyed entries dropped)",
                data.get("native_guard"),
                have,
            )
            entries = [e for e in entries if e.get("tactic_name") is not None]
    for entry in entries:
        key = _key(entry)
        cache.cache[key] = (
            int(entry.get("runner_id", 0)),
            entry["tactic"],
            float(entry.get("min_time", 0.0)),
        )
        if entry.get("tactic_name") is not None:
            cache.bundled_names[key] = entry["tactic_name"]
    tuner.bundle_reporter = reporter

    flashinfer = data.get("flashinfer")
    fi_state = "none"
    if flashinfer:
        have = _flashinfer_versions()
        if flashinfer.get("guard") == have:
            from tensorrt_llm._torch import flashinfer_autotune_cache

            fi_dir = Path(tempfile.mkdtemp(prefix=f"m31_pins_rank{rank}_"))
            fi_path = fi_dir / f"rank{rank}.json"
            fi_path.write_text(json.dumps(flashinfer["cache"]))
            flashinfer_autotune_cache.use_bundled_file(str(fi_path))
            fi_state = f"{len(flashinfer['cache']) - ('_metadata' in flashinfer['cache'])} shapes"
        else:
            reporter.unresolved_guard("flashinfer", flashinfer.get("guard"), have)
            fi_state = "guard mismatch"
    atexit.register(reporter.final)
    _info(
        f"[M31_PIN] loaded rank={rank} topology={topology}: {len(entries)} AutoTuner entries, "
        f"FlashInfer {fi_state} ({path.name})"
    )
    return reporter


# ---------------------------------------------------------------------------------------------------------------
# Converter: a tuning run's standard files -> the bundle format (run on the target GPU with the wheel installed)
# ---------------------------------------------------------------------------------------------------------------


def _mxfp8_name(runner_key: str, tactic: Any) -> str:
    import torch

    if tactic == -1:
        return "fallback"
    dtype_name = runner_key.strip("()").split(",")[0].strip()
    dtype = getattr(torch, dtype_name.replace("torch.", ""))
    return torch.classes.trtllm.MXFP8GemmRunner(dtype).get_config_name(int(tactic))


def _fp4_moe_name(runner_key: str, tactic: Any) -> str:
    import torch

    tile_n, config = (int(v) for v in tactic)
    if tile_n == -1 or config == -1:
        return "fallback"
    act_type = int(runner_key.strip("()").split(",")[-1])
    return torch.classes.trtllm.FP4BlockScaleMoERunner(act_type).get_config_name(tile_n, config)


TACTIC_NAMERS = {"MXFP8GemmRunner": _mxfp8_name, "FP4BlockScaleMoERunner": _fp4_moe_name}


def convert_autotuner(cache_file: Path, rank: int) -> List[Dict[str, Any]]:
    """One rank's entries of an AutoTuner cache file (shared + rank_<n>), each with its tactic's name, or a null
    name for a runner without one (index-keyed; ``install`` serves it only under the bundle's native guard)."""
    import torch  # noqa: F401  (loads the native runners)

    import tensorrt_llm  # noqa: F401

    data = json.loads(Path(cache_file).read_text())
    entries = []
    for section in ("shared", f"rank_{rank}"):
        for key_str, value in data.get(section, {}).items():
            op, runner, runner_key, shapes = ast.literal_eval(key_str)
            tactic = ast.literal_eval(value["tactic"])
            namer = TACTIC_NAMERS.get(runner)
            entries.append(
                {
                    "op": op,
                    "runner": runner,
                    "runner_key": runner_key,
                    "shapes": [list(dims) for dims in shapes],
                    "runner_id": int(value.get("runner_id", 0)),
                    "tactic": tactic,
                    "tactic_name": namer(runner_key, tactic) if namer is not None else None,
                    "min_time": float(value.get("min_time", 0.0)),
                }
            )
    return entries


def convert(
    cache_file: Optional[Path], fi_dir: Optional[Path], out_dir: Path, ranks: int
) -> List[Path]:
    """Write ``out_dir/rank_<n>.json`` for ranks 0..ranks-1 from an AutoTuner cache file and FlashInfer rank files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    guard = _flashinfer_versions()
    written = []
    for rank in range(ranks):
        bundle: Dict[str, Any] = {"format": FORMAT, "autotuner": [], "flashinfer": None}
        if cache_file is not None:
            bundle["autotuner"] = convert_autotuner(cache_file, rank)
            if any(e["tactic_name"] is None for e in bundle["autotuner"]):
                bundle["native_guard"] = _native_guard()
        if fi_dir is not None:
            fi_file = Path(fi_dir) / f"rank{rank}.json"
            if not fi_file.is_file():
                raise FileNotFoundError(f"{fi_file}: every rank needs its FlashInfer file")
            bundle["flashinfer"] = {"guard": guard, "cache": json.loads(fi_file.read_text())}
        path = out_dir / f"rank_{rank}.json"
        path.write_text(json.dumps(bundle, indent=1, sort_keys=True) + "\n")
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
