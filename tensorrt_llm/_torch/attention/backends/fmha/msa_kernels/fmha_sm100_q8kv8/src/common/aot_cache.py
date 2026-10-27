# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Persistent AOT cache for CuTe DSL compiled kernels.

Saves compiled TVM FFI kernels as .o files on first compile,
loads them on subsequent runs to skip JIT compilation.

Layout, under a placement-only root::

    <root>/v2/<toolchain>/          sha256 of the compiler stack (CuTe DSL, CUDA, quack, TVM-FFI)
      manifest.json
      <name>_<key>.o                the exported kernel, named by its compile key
      <name>_<key>.json             the compile key and the content hash of every module under
                                    cute/ the kernel is generated from

An object is loaded only while those modules are unchanged. They are the import closure, within
cute/, of the module that defines the kernel (``save_aot(..., sources=...)``; without it, every
cute/src module loaded at compile time), plus the module that calls ``save_aot``: the compile call
site, whose tensor layouts and options shape the code as well. Editing one kernel recompiles only
the kernels that import it. A missing, unreadable or mismatched entry is a miss, never an error.

Environment variables:
    MM_SPARSE_ATTN_AOT_CACHE: Override the cache root
        (default: ~/.cache/minfer/mm_sparse_attn)
    MM_SPARSE_ATTN_AOT_DISABLE=1: Disable AOT cache entirely
"""

import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import cutlass
import cutlass.cute as cute

_SCHEMA = 2
_CUTE_ROOT = Path(__file__).resolve().parents[2]  # cute/
_AOT_DISABLE = os.environ.get("MM_SPARSE_ATTN_AOT_DISABLE", "0") == "1"


def _toolchain_manifest():
    manifest = {"schema": _SCHEMA, "cutlass_dsl": cutlass.__version__,
                "cuda": str(cutlass.CUDA_VERSION)}
    for package in ("quack-kernels", "apache-tvm-ffi"):
        try:
            manifest[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest[package] = None
    return manifest


_MANIFEST = _toolchain_manifest()
_TOOLCHAIN = hashlib.sha256(json.dumps(_MANIFEST, sort_keys=True).encode()).hexdigest()[:16]


def _vendored_layout():
    """TensorRT-LLM: (the vendored tree's name, the tensorrt_llm package dir) of this copy. The trees are
    tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/<tree>."""
    here = os.path.abspath(__file__)
    tree = here
    while os.path.basename(os.path.dirname(tree)) != "msa_kernels":
        tree = os.path.dirname(tree)
    return os.path.basename(tree), os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(tree))))))


_TREE, _TRTLLM_ROOT = _vendored_layout()
# Each vendored tree keeps its own cache: their source paths, relative to their roots, would collide in one.
_AOT_CACHE_DIR = os.path.join(
    os.environ.get(
        "MM_SPARSE_ATTN_AOT_CACHE",
        os.path.expanduser("~/.cache/minfer/mm_sparse_attn"),
    ),
    _TREE,
    f"v{_SCHEMA}",
    _TOOLCHAIN,
)
# TensorRT-LLM: objects harvested from a reference run ship as wheel data and are read before the user cache.
_WHEEL_AOT_DIR = os.path.join(_TRTLLM_ROOT, "libs", "msa_cute_aot", _TREE, f"v{_SCHEMA}", _TOOLCHAIN)

_loaded_modules: dict[str, object] = {}
_file_hashes: dict[Path, str | None] = {}


def _file_hash(path: Path):
    if path not in _file_hashes:
        try:
            _file_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            _file_hashes[path] = None
    return _file_hashes[path]


def _key_to_path(key: tuple, root: str = _AOT_CACHE_DIR) -> str:
    h = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    name = str(key[0]).replace("/", "_")
    return os.path.join(root, f"{name}_{h}")


def _source_files(sources) -> list[Path]:
    """Files of ``sources``: modules, module names, classes or functions, or .py paths
    (relative ones are taken from cute/)."""
    files = []
    for source in sources:
        if isinstance(source, (str, os.PathLike)) and str(source).endswith(".py"):
            path = Path(source)
            files.append(path if path.is_absolute() else _CUTE_ROOT / path)
            continue
        if isinstance(source, types.ModuleType):
            module = source
        else:
            name = source if isinstance(source, str) else getattr(source, "__module__", None)
            module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        if path:
            files.append(Path(path))
    return files


def _resolve_import(name: str) -> list[Path]:
    """The cute/ files executed by importing a dotted name: package __init__s, then the module."""
    files = []
    parts = name.split(".")
    for depth in range(1, len(parts) + 1):
        base = _CUTE_ROOT.joinpath(*parts[:depth])
        if (base / "__init__.py").is_file():
            files.append(base / "__init__.py")
        elif base.with_suffix(".py").is_file():
            files.append(base.with_suffix(".py"))
            break
        else:
            break
    return files


def _import_closure(files) -> list[Path]:
    """``files`` and the cute/ modules they import, transitively (from import statements)."""
    seen = set()
    stack = [Path(path).resolve() for path in files]
    this_file = Path(__file__).resolve()
    while stack:
        path = stack.pop()
        if path in seen or path == this_file or not path.is_file():
            continue
        try:
            package = list(path.relative_to(_CUTE_ROOT).parent.parts)
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (ValueError, SyntaxError, OSError):
            continue  # outside cute/ (the toolchain manifest covers it) or unparsable
        seen.add(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package[:len(package) - node.level + 1]
                    base = ".".join(anchor + ([base] if base else []))
                names = [base] + [f"{base}.{alias.name}" if base else alias.name
                                  for alias in node.names]
            else:
                continue
            for name in filter(None, names):
                stack.extend(found.resolve() for found in _resolve_import(name))
    return sorted(seen)


def _loaded_cute_modules() -> list[Path]:
    src = _CUTE_ROOT / "src"
    files = []
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().is_relative_to(src):
            files.append(Path(path).resolve())
    return files


def _valid_object(key: tuple):
    """The shipped (wheel data), else the cached, object of ``key`` whose entry vouches for the current sources."""
    for root in (_WHEEL_AOT_DIR, _AOT_CACHE_DIR):
        found = _valid_object_in(key, root)
        if found is not None:
            return found
    return None


def _valid_object_in(key: tuple, root: str):
    base = _key_to_path(key, root)
    try:
        with open(base + ".json", encoding="utf-8") as f:
            entry = json.load(f)
        valid = (entry["schema"] == _SCHEMA and entry["key"] == repr(key)
                 and all(_file_hash(_CUTE_ROOT / rel) == digest
                         for rel, digest in entry["inputs"].items()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    obj_path = base + ".o"
    return obj_path if valid and os.path.isfile(obj_path) else None


def aot_object_path(key: tuple) -> str:
    """The exported object of ``key`` when it is current, else "" (for callers that load it
    themselves)."""
    if _AOT_DISABLE:
        return ""
    return _valid_object(key) or ""


def try_load_aot(key: tuple):
    if _AOT_DISABLE:
        return None
    obj_path = _valid_object(key)
    if obj_path is None:
        return None
    func_name = str(key[0])
    try:
        if obj_path not in _loaded_modules:
            _loaded_modules[obj_path] = cute.runtime.load_module(
                obj_path, enable_tvm_ffi=True
            )
        return getattr(_loaded_modules[obj_path], func_name)
    except Exception as e:
        print(f"[aot_cache] Failed to load {obj_path}: {e}")
        return None


def _atomic_write(path: str, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def save_aot(key: tuple, compiled, sources=None) -> None:
    """Export ``compiled`` for ``key``. ``sources`` names what generates the kernel (its class,
    module or module name, or .py paths under cute/); their cute/ import closure and the calling
    module are what the entry checks on later loads."""
    if _AOT_DISABLE:
        return
    caller = Path(sys._getframe(1).f_code.co_filename).resolve()
    if not hasattr(compiled, "export_to_c"):
        return
    base = _key_to_path(key)
    obj_path = base + ".o"
    os.makedirs(_AOT_CACHE_DIR, exist_ok=True)
    manifest = os.path.join(_AOT_CACHE_DIR, "manifest.json")
    if not os.path.isfile(manifest):
        _atomic_write(manifest, json.dumps(_MANIFEST, indent=2, sort_keys=True) + "\n")
    tmp_path = obj_path + f".tmp.{os.getpid()}"
    func_name = str(key[0])
    try:
        t0 = time.time()
        files = _source_files(sources) if sources is not None else _loaded_cute_modules()
        closure = _import_closure(files)
        if caller.is_file() and caller.is_relative_to(_CUTE_ROOT) and caller not in closure:
            closure.append(caller)  # the call site itself, not everything it imports
        inputs = {str(path.relative_to(_CUTE_ROOT)): _file_hash(path) for path in closure}
        compiled.export_to_c(tmp_path, function_name=func_name)
        with contextlib.suppress(FileNotFoundError):  # no reader pairs the old entry with it
            os.unlink(base + ".json")
        os.replace(tmp_path, obj_path)
        # The entry goes last: until it names the current sources, the object is not served.
        _atomic_write(base + ".json", json.dumps(
            {"schema": _SCHEMA, "key": repr(key), "inputs": inputs}, sort_keys=True))
        dt = time.time() - t0
        print(f"[aot_cache] Saved {func_name} -> {obj_path} ({dt:.1f}s, {len(inputs)} sources)")
    except Exception as e:
        print(f"[aot_cache] Failed to save {func_name}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
