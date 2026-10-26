# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Persistent AOT cache for CuTe DSL compiled kernels.

Saves compiled TVM FFI kernels as .o files on first compile,
loads them on subsequent runs to skip JIT compilation.

Cached objects are keyed by the caller's kernel key *and* a build fingerprint
covering the target GPU architecture and the kernel sources. Without the
fingerprint a .o built for one architecture (or from an older revision of the
kernel) is silently reused on another, which both miscompiles across
SM100/SM103 and makes local kernel edits appear to have no effect.

Environment variables:
    MM_SPARSE_ATTN_AOT_CACHE: Override cache directory
        (default: ~/.cache/minfer/mm_sparse_attn)
    MM_SPARSE_ATTN_AOT_DISABLE=1: Disable AOT cache entirely
"""

import hashlib
import os
import time

import cutlass.cute as cute

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
# Each vendored tree keeps its own cache; objects harvested from a reference run ship as wheel data
# (tensorrt_llm/libs/msa_cute_aot/<tree>) and are read first.
_AOT_CACHE_DIR = os.path.join(
    os.environ.get("MM_SPARSE_ATTN_AOT_CACHE", os.path.expanduser("~/.cache/minfer/mm_sparse_attn")), _TREE
)
_WHEEL_AOT_DIR = os.path.join(_TRTLLM_ROOT, "libs", "msa_cute_aot", _TREE)
_AOT_DISABLE = os.environ.get("MM_SPARSE_ATTN_AOT_DISABLE", "0") == "1"

_loaded_modules: dict[str, object] = {}
_build_fingerprint_cache: str | None = None

# .../cute/src/common/aot_cache.py -> .../cute
_PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _source_fingerprint() -> str:
    """Hash every .py under the cute package so edits invalidate the cache."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(_PACKAGE_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            h.update(os.path.relpath(path, _PACKAGE_ROOT).encode())
            try:
                with open(path, "rb") as f:
                    h.update(f.read())
            except OSError:
                # Unreadable source: fold the error in rather than silently
                # producing the same digest as if the file did not exist.
                h.update(b"<unreadable>")
    return h.hexdigest()[:16]


def _target_arch() -> str:
    """Compute capability of the device the kernel will be launched on."""
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}"
    except Exception:
        return "smunknown"


def _build_fingerprint() -> str:
    global _build_fingerprint_cache
    if _build_fingerprint_cache is None:
        try:
            import cutlass

            cutlass_version = getattr(cutlass, "__version__", "unknown")
        except Exception:
            cutlass_version = "unknown"
        _build_fingerprint_cache = "-".join(
            (_target_arch(), str(cutlass_version), _source_fingerprint())
        )
    return _build_fingerprint_cache


def _key_to_path(key: tuple, root: str = _AOT_CACHE_DIR) -> str:
    h = hashlib.sha256(
        repr((_build_fingerprint(), key)).encode()
    ).hexdigest()[:16]
    name = str(key[0]).replace("/", "_")
    return os.path.join(root, f"{name}_{h}")


def try_load_aot(key: tuple):
    if _AOT_DISABLE:
        return None
    obj_path = _key_to_path(key, _WHEEL_AOT_DIR) + ".o"
    if not os.path.isfile(obj_path):
        obj_path = _key_to_path(key) + ".o"
    if not os.path.isfile(obj_path):
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


def save_aot(key: tuple, compiled) -> None:
    if _AOT_DISABLE:
        return
    if not hasattr(compiled, "export_to_c"):
        return
    obj_path = _key_to_path(key) + ".o"
    os.makedirs(_AOT_CACHE_DIR, exist_ok=True)
    tmp_path = obj_path + f".tmp.{os.getpid()}"
    func_name = str(key[0])
    try:
        t0 = time.time()
        compiled.export_to_c(tmp_path, function_name=func_name)
        os.replace(tmp_path, obj_path)
        dt = time.time() - t0
        print(f"[aot_cache] Saved {func_name} -> {obj_path} ({dt:.1f}s)")
    except Exception as e:
        print(f"[aot_cache] Failed to save {func_name}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
