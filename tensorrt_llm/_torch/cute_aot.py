# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ahead-of-time CuTe DSL kernels shipped as wheel data.

``compile_or_load(key, compile_fn)`` returns the compiled function of a CuTe DSL kernel without compiling it when this
package ships its exported object: ``cute_aot_data/<toolchain>/<digest>.o``. The toolchain names the CuTe DSL and
TVM-FFI versions and the GPU architecture; the digest is that of ``key`` and of the contents of ``sources``. ``key``
must name everything the compile specializes on (kernel, constexpr arguments, argument dtypes / static shapes /
alignments, compile options), since an object is served for any compile under the same key; ``sources`` are the files
whose code the kernel is built from (its module and the modules it traces), so an edited kernel no longer matches its
old object. The key also carries the process's model switches and CuTe DSL settings (``TRTLLM_M31_DISABLE_*``,
``TRTLLM_M31_ENABLE_*``, ``CUTE_DSL_*`` other than directories and logging), which kernel modules may read when they
are traced: objects are harvested under the serving environment, and a process with other settings compiles.
Without a shipped object, ``compile_fn()`` runs and one ``[M31_CUTE_AOT] compile <key>`` line is logged, so a run can
count the kernels that compiled in-process.

``TRTLLM_M31_CUTE_AOT_DIR`` replaces the shipped directory. ``TRTLLM_M31_CUTE_AOT_EXPORT_DIR`` exports every kernel this
process compiles there (``<toolchain>/<digest>.o`` + ``.h``, and ``<digest>.key`` holding the key and the source
digests), the layout to copy into ``cute_aot_data``. ``TRTLLM_M31_CUTE_AOT_DISABLE=1`` always compiles.

The served object is the exported compile of the same kernel source under the same key and toolchain, so its device
code is that of the in-process compile. A kernel compiled with TVM-FFI is exported and loaded as a TVM-FFI function
(``tvm_ffi=True``); its calling convention is then TVM-FFI's.
"""

import functools
import hashlib
import inspect
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from tensorrt_llm.logger import logger

DIR_ENV = "TRTLLM_M31_CUTE_AOT_DIR"
EXPORT_DIR_ENV = "TRTLLM_M31_CUTE_AOT_EXPORT_DIR"
DISABLE_ENV = "TRTLLM_M31_CUTE_AOT_DISABLE"
DATA_ROOT = Path(__file__).resolve().parent / "cute_aot_data"

_lock = threading.RLock()
_loaded: Dict[Tuple[str, str], Any] = {}
_counts = {"loaded": 0, "compiled": 0}


# Shipped objects are served only after a model that ships them arms the loader (``arm``, at target routing, before
# the model builds its kernels); until then every key compiles in-process with no lookup, export or log line.
_armed = False


def arm() -> None:
    """Serve shipped objects (and export, with ``TRTLLM_M31_CUTE_AOT_EXPORT_DIR``) in this process."""
    global _armed
    _armed = True


def _disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def toolchain() -> str:
    """The toolchain an exported object is valid for: the CuTe DSL and TVM-FFI versions and this GPU's architecture."""
    import importlib.metadata as md

    import torch

    def version(dist: str) -> str:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            return "none"

    major, minor = torch.cuda.get_device_capability()
    return f"cutedsl-{version('nvidia-cutlass-dsl')}-tvmffi-{version('apache-tvm-ffi')}-sm{major}{minor}"


def digest(key: str) -> str:
    """The file stem of ``key``'s object (also its symbol prefix, which must be a C identifier)."""
    return "k" + hashlib.sha256(key.encode()).hexdigest()[:32]


_ENV_PREFIXES = ("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_", "CUTE_DSL_")
_ENV_IGNORED_SUFFIXES = (
    "_DIR",
    "_PATH",
    "_LOG_LEVEL",
    "_LOG_TO_CONSOLE",
    "_LOG_TO_FILE",
    "_DISABLE_FILE_CACHING",
)


def env_settings() -> Tuple[Tuple[str, str], ...]:
    """The environment settings a traced kernel module may read (see the module docstring), sorted."""
    return tuple(
        sorted(
            (k, v)
            for k, v in os.environ.items()
            if k.startswith(_ENV_PREFIXES) and not k.endswith(_ENV_IGNORED_SUFFIXES)
        )
    )


def full_key(key: str, sources: Sequence[str] = ()) -> str:
    """``key`` with a digest of the source files' contents and the environment settings appended (the string
    ``digest`` is taken of). Only the file contents count, not where the files are installed."""
    if sources:
        combined = hashlib.sha256(" ".join(sorted(_file_sha(str(src)) for src in sources)).encode())
        key = f"{key} | src {len(sources)} files {combined.hexdigest()[:16]}"
    env = env_settings()
    return f"{key} | env {' '.join(f'{k}={v}' for k, v in env)}" if env else key


_file_shas: Dict[str, str] = {}
_tree_files: Dict[Tuple[str, ...], Tuple[str, ...]] = {}


def py_files(*dirs: str) -> Tuple[str, ...]:
    """Every ``.py`` file under ``dirs`` (sorted): ``sources`` for kernels that trace code from these trees."""
    found = _tree_files.get(dirs)
    if found is None:
        found = _tree_files[dirs] = tuple(
            sorted(str(f) for d in dirs for f in Path(d).rglob("*.py"))
        )
    return found


def _file_sha(path: str) -> str:
    sha = _file_shas.get(path)
    if sha is None:
        sha = _file_shas[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    return sha


def object_path(key: str, tool: Optional[str] = None) -> Path:
    """The shipped object of a full key (``full_key``)."""
    root = Path(os.environ[DIR_ENV]) if os.environ.get(DIR_ENV) else DATA_ROOT
    return root / (tool or toolchain()) / f"{digest(key)}.o"


def _load(path: Path, symbol: str, tvm_ffi: bool) -> Any:
    from cutlass.cute.runtime import load_module

    return getattr(load_module(str(path), enable_tvm_ffi=tvm_ffi), symbol)


def _export(compiled: Any, key: str, tool: str, symbol: str, tvm_ffi: bool) -> None:
    """Export into a private directory, then rename each file into place: every rank of a role may export the same
    (identical) object concurrently, and a reader never sees a partial file."""
    out = Path(os.environ[EXPORT_DIR_ENV]) / tool
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out, prefix=f".{symbol}.") as tmp:
        if tvm_ffi:
            compiled.export_to_c(
                os.path.join(tmp, f"{symbol}.o"),
                function_name=symbol,
                export_only_tvm_ffi_symbols=True,
            )
        else:
            compiled.export_to_c(tmp, symbol, function_prefix=symbol)
        Path(tmp, f"{symbol}.key").write_text(key + "\n")
        for name in sorted(os.listdir(tmp)):
            os.replace(os.path.join(tmp, name), out / name)


def compile_or_load(
    key: str, compile_fn: Callable[[], Any], *, sources: Sequence[str] = (), tvm_ffi: bool = False
) -> Any:
    """The compiled function of ``key`` built from ``sources``: the shipped object when there is one, else
    ``compile_fn()``."""
    if not _armed:
        with _lock:
            cached = _loaded.get(("", key))
            if cached is None:
                cached = _loaded[("", key)] = compile_fn()
            return cached
    tool = toolchain()
    key = full_key(key, sources)
    with _lock:
        cached = _loaded.get((tool, key))
        if cached is not None:
            return cached
        symbol = digest(key)
        path = object_path(key, tool)
        compiled = None
        if not _disabled() and path.is_file():
            try:
                compiled = _load(path, symbol, tvm_ffi)
                _counts["loaded"] += 1
            except Exception as e:  # noqa: BLE001  (a stale or foreign object falls back to compiling)
                logger.warning(f"[M31_CUTE_AOT] stale {path.name} for {key}: {e!r}")
        if compiled is None:
            logger.info(f"[M31_CUTE_AOT] compile {key}")
            compiled = compile_fn()
            _counts["compiled"] += 1
            if os.environ.get(EXPORT_DIR_ENV):
                _export(compiled, key, tool, symbol, tvm_ffi)
        _loaded[(tool, key)] = compiled
        return compiled


def cached_compile(
    name: str, sources: Callable[[], Sequence[str]], *, tvm_ffi: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function that builds and compiles one kernel from hashable arguments (in place of
    ``functools.cache``): each argument set is compiled or loaded once per process through ``compile_or_load``, keyed
    by ``name`` and the arguments' reprs (bound to the signature, defaults applied)."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(fn)

        @functools.cache
        def compiled(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = f"{name} " + " ".join(f"{k}={v!r}" for k, v in bound.arguments.items())
            return compile_or_load(
                key, lambda: fn(*args, **kwargs), sources=sources(), tvm_ffi=tvm_ffi
            )

        return functools.wraps(fn)(compiled)

    return decorate


def counts() -> Dict[str, int]:
    """How many kernels this process loaded from objects and compiled."""
    return dict(_counts)
