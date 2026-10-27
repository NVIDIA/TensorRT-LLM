# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Per-variant lazy JIT compilation for FMHA varlen kernels.

Each FMHA variant (dtype x tile x sparse x page x split_kv x pack_factor) is compiled
independently on first use and cached to ~/.cache/minfer/fmha_sm100/.

Source changes select a new cache automatically; prewarm using the same sources.
"""

import itertools
import fcntl
import hashlib
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

import jinja2

logger = logging.getLogger(__name__)

# Kernel sources and CUTLASS headers are shipped inside the package directory
# so that JIT compilation works from both editable and wheel installs.
_PACKAGE_DIR = Path(__file__).resolve().parent
_FMHA_VARLEN_DIR = _PACKAGE_DIR / "csrc"
# TensorRT-LLM: the vendored tree has no CUTLASS submodule; a run-time compile takes CUTLASS_ROOT.
_CUTLASS_DIR = Path(os.environ["CUTLASS_ROOT"]) if os.environ.get("CUTLASS_ROOT") else _PACKAGE_DIR / "cutlass"
_CUTLASS_INCLUDE = _CUTLASS_DIR / "include"
_CUTLASS_UTIL_INCLUDE = _CUTLASS_DIR / "tools" / "util" / "include"


def _device_is_sm107() -> bool:
    """Whether the current CUDA device is SM107 (compute capability 10.7)."""
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


_proxy_modes_cache = None
# A process may take the SM107 proxy modes on another device (MiniMax-M3.1 tp2ctx on SM103) by calling
# enable_proxy_modes_off_sm107() before its first proxy plan. The mainloop compiles them only into the tile_q 128
# (thread shape (1, 2, 1)) OnlyScore variants, so tile_q 256 plans are unchanged either way.
_proxy_modes_opt_in = False


def enable_proxy_modes_off_sm107() -> bool:
    """Opt this process into the SM107 OnlyScore proxy modes on another device; returns whether they are in effect.

    Must run before the modes resolve (the first proxy plan). Afterwards it changes nothing and returns whether the
    resolved modes are the lifted ones. TLLM_FMHA_PROXY_CORRECTION_CHAIN=1 / TLLM_FMHA_PROXY_S_DOUBLE_BUFFER=0 still
    keep the stock behaviour.
    """
    global _proxy_modes_opt_in
    if _proxy_modes_cache is None:
        _proxy_modes_opt_in = True
        return True
    return _proxy_modes_cache == (False, False)


def _proxy_modes():
    """(keep correction chain, single S buffer) for the OnlyScore K-split kernels (the MSA indexer proxy).

    By default they skip the per-tile softmax -> correction -> MMA handshake (kSkipPerTileCorrection in
    sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp) and double-buffer S in the O TMEM columns (kSDoubleBuf, which
    requires the skip); TLLM_FMHA_PROXY_CORRECTION_CHAIN=1 / TLLM_FMHA_PROXY_S_DOUBLE_BUFFER=0 keep the stock
    behaviour, each in its own cache directory. Both are SM107-only: every other device compiles the stock mainloop
    (the kill knobs' defines and cache directory). Resolved on first use, not at import, so importing this module
    never initializes CUDA (a parent that later forks workers must not hold a CUDA context).
    """
    global _proxy_modes_cache
    if _proxy_modes_cache is None:
        sm107 = _device_is_sm107() or _proxy_modes_opt_in
        chain = os.environ.get("TLLM_FMHA_PROXY_CORRECTION_CHAIN", "0") == "1" or not sm107
        single_s = os.environ.get("TLLM_FMHA_PROXY_S_DOUBLE_BUFFER", "1") == "0" or not sm107
        logger.warning(
            "fmha_sm100: OnlyScore K-split per-tile correction handshake %s (TLLM_FMHA_PROXY_CORRECTION_CHAIN=%s), "
            "S double buffer %s (TLLM_FMHA_PROXY_S_DOUBLE_BUFFER=%s)%s",
            "kept" if chain else "skipped", "1" if chain else "0",
            "off" if (single_s or chain) else "on", "0" if single_s else "1",
            ("; opted in off SM107" if _proxy_modes_opt_in else "") if sm107 else "; stock mainloop, the device is not SM107",
        )
        _proxy_modes_cache = (chain, single_s)
    return _proxy_modes_cache


# TensorRT-LLM: the modules the MiniMax-M3.1 targets call are built ahead of time
# (cpp/tensorrt_llm/kernels/minimaxM3Msa) into tensorrt_llm/libs/msa_fmha_sm100, in this cache's
# layout, for the SM107 OnlyScore proxy modes without diagnostics. They load from there; anything
# else compiles as before and needs nvcc and CUTLASS headers (CUTLASS_ROOT).
# This file is tensorrt_llm/_torch/attention/backends/fmha/msa_kernels/fmha_sm100/jit.py.
_PREBUILT_DIR = Path(os.path.abspath(__file__)).parents[6] / "libs" / "msa_fmha_sm100"


def _prebuilt(relative: str):
    """The prebuilt module ``relative`` when this process may use the prebuilt set, else None."""
    if any(os.environ.get(v) is not None for v in ("GPU_TRACE", "SM_TIMING", "FMHA_GMEM_CHECK")):
        return None
    if _proxy_modes() != (False, False):
        return None
    path = _PREBUILT_DIR / relative
    return path if path.is_file() else None


def _require_toolchain(what):
    """A module outside the prebuilt set compiles at run time, which needs CUTLASS headers."""
    if not _CUTLASS_INCLUDE.is_dir():
        raise RuntimeError(
            f"fmha_sm100: {what} is not in the prebuilt set {_PREBUILT_DIR} and compiling it needs "
            "CUTLASS headers; set CUTLASS_ROOT, or add it to "
            "cpp/tensorrt_llm/kernels/minimaxM3Msa/generate_fmha_variants.py and rebuild")
    logger.warning("msa fmha JIT fallback variant=%s", what)


def _compute_cache_base():
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR")
    base = Path(explicit) if explicit else Path(os.path.expanduser("~/.cache/minfer/fmha_sm100"))
    # Different build configs get separate cache dirs to avoid conflicts
    suffix = ""
    if os.environ.get("GPU_TRACE") is not None:
        suffix += "_gpu_trace"
    if os.environ.get("SM_TIMING") is not None:
        suffix += "_sm_timing"
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        suffix += "_gmem_check"
    proxy_chain, proxy_single_s = _proxy_modes()
    if proxy_chain:
        suffix += "_proxy_correction_chain"
    if proxy_single_s:
        suffix += "_proxy_single_s"
    if suffix and not explicit:
        base = base.parent / (base.name + suffix)

    # Hash templates, bindings and headers together; jit.py also covers changes
    # to dispatch and compiler flags. Use relative names so installs share keys.
    digest = hashlib.sha256()
    sources = sorted(path for path in _FMHA_VARLEN_DIR.rglob("*") if path.is_file())
    sources.append(_PACKAGE_DIR / "jit.py")
    for path in sources:
        content = path.read_bytes()
        digest.update(path.relative_to(_PACKAGE_DIR).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    # An explicit cache root must also isolate instrumented and normal binaries.
    digest.update(suffix.encode())
    cache_base = base / ("sources_sha256_" + digest.hexdigest())
    logger.info(
        "Using FMHA cache %s (source/build fingerprint; changed sources or "
        "diagnostic modes require recompilation)", cache_base,
    )
    return cache_base


class _LazyCacheBase:
    """The cache root, computed on first use: its suffix depends on the device (see _proxy_modes)."""

    _path = None

    def _resolve(self) -> Path:
        if self._path is None:
            self._path = _compute_cache_base()
        return self._path

    def __truediv__(self, other):
        return self._resolve() / other

    def __fspath__(self):
        return str(self._resolve())

    def __str__(self):
        return str(self._resolve())

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


CACHE_BASE = _LazyCacheBase()


def _acquire_file_lock(lock_path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)



_PACK_FACTORS = [1,2,4,6,8,16]
# _PACK_FACTORS = [1, 6]

# DLPack dtype codes (must match tvm_ffi_utils.h encode_dlpack_dtype)
_BFLOAT16_CODE = (4 << 16) | (16 << 8) | 1      # 266241
_FLOAT8_E4M3FN_CODE = (12 << 16) | (8 << 8) | 1  # 788481

_FMHA_SM100_DISPATCH = [
    ("int64_t dtype_code", [
        (_BFLOAT16_CODE,      {"dtype_in": "nv_bfloat16",   "cutlass_dtype_out": "cutlass::bfloat16_t"}),
        (_FLOAT8_E4M3FN_CODE, {"dtype_in": "__nv_fp8_e4m3", "cutlass_dtype_out": "cutlass::bfloat16_t"}),
    ]),
    ("int qo_tile_size", [
        (128, {"tile_q": "_128", "tile_kv": "_256", "thread_shape": "_1, _2, _1"}),
        (256, {"tile_q": "_256", "tile_kv": "_128", "thread_shape": "_2, _1, _1"}),
    ]),
    ("bool single_wg", [
        ("true",  {"single_wg": "true"}),
        ("false", {"single_wg": "false"}),
    ]),
    ("int sparse_mode", [
        (0,    {"sparse_mode": "Sparse"}),
        (1,    {"sparse_mode": "Full"}),
        (2,    {"sparse_mode": "OnlyScore"}),
        (None, {"sparse_mode": "Off"}),
    ]),
    ("int page_size", [
        (-1,  {"page_size": -1}),
        (128, {"page_size": 128}),
        # (256, {"page_size": 256}),
    ]),
    ("bool split_kv", [
        ("false", {"is_split_kv": "false"}),
        ("true",  {"is_split_kv": "true"}),
    ]),
    ("int pack_factor", [(i, {"pack_factor": i}) for i in _PACK_FACTORS]),
]

_FMHA_SM100_IMPOSSIBLE = lambda p: (
    (p.get("tile_q") == "_256" and p.get("single_wg") == "true") or
    (p.get("tile_q") == "_256" and p.get("is_split_kv") == "true") or
    (p.get("page_size") == -1 and p.get("sparse_mode") == "Sparse") or
    (p.get("pack_factor", 1) > 1 and p.get("tile_q") == "_256")
)


def _dlpack_dtype_code(torch_dtype):
    """Encode a torch dtype as a DLPack int64 code."""
    import torch
    _map = {
        torch.float16: (2 << 16) | (16 << 8) | 1,
        torch.bfloat16: (4 << 16) | (16 << 8) | 1,
        torch.float32: (2 << 16) | (32 << 8) | 1,
        torch.float8_e4m3fn: (12 << 16) | (8 << 8) | 1,
        torch.float8_e5m2: (13 << 16) | (8 << 8) | 1,
    }
    return _map[torch_dtype]


def _variant_key_from_runtime(dtype_code, qo_tile_size, single_wg,
                               sparse_mode, page_size, split_kv, pack_factor):
    """Compute a variant key string from runtime parameters."""
    dims = _FMHA_SM100_DISPATCH

    def _match_idx(dim_values, runtime_val):
        # Convert Python bool to string to match dispatch table ("true"/"false")
        if isinstance(runtime_val, bool):
            runtime_val = "true" if runtime_val else "false"
        for idx, (match_val, _) in enumerate(dim_values):
            if match_val is None:
                return idx
            if match_val == runtime_val:
                return idx
        return len(dim_values) - 1

    runtime_vals = [dtype_code, qo_tile_size, single_wg, sparse_mode,
                    page_size, split_kv, pack_factor]
    indices = []
    params = {}
    for (_, dim_values), rv in zip(dims, runtime_vals):
        idx = _match_idx(dim_values, rv)
        indices.append(idx)
        _, tparams = dim_values[idx]
        params.update(tparams)

    if _FMHA_SM100_IMPOSSIBLE(params):
        raise ValueError(f"Impossible FMHA variant combination: {params}")

    func_name = "fmha_sm100_" + "_".join(str(i) for i in indices)
    variant_name = "_".join(str(i) for i in indices)
    params["func_name"] = func_name
    params["variant_name"] = variant_name
    return variant_name, params


def _get_tvm_ffi_include():
    """Find TVM-FFI include directory."""
    try:
        import tvm_ffi
        tvm_dir = Path(tvm_ffi.__path__[0])
        inc = tvm_dir / "include"
        if inc.exists():
            return str(inc)
        inc2 = tvm_dir.parent / "include"
        if inc2.exists():
            return str(inc2)
    except ImportError:
        pass
    raise RuntimeError("Cannot find TVM-FFI include directory; install apache-tvm-ffi")


def _get_cuda_home():
    """Find CUDA toolkit root."""
    if "CUDA_HOME" in os.environ:
        return os.environ["CUDA_HOME"]
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    for p in ["/usr/local/cuda", "/opt/cuda"]:
        if os.path.isdir(p):
            return p
    raise RuntimeError("Cannot find CUDA toolkit. Set CUDA_HOME.")


def _all_variants_so() -> Path:
    return CACHE_BASE / "_all_variants" / "all_variants.so"

def _get_nvcc_flags(cache_dir, fmha=True):
    tvm_include = _get_tvm_ffi_include()
    fmha_include = str(_FMHA_VARLEN_DIR / "include")
    cutlass_include = str(_CUTLASS_INCLUDE)
    cutlass_util_include = str(_CUTLASS_UTIL_INCLUDE)
    nvcc_flags = [
        "-O3", "-std=c++20",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
        # SM100-family image so the rest of the family can load these kernels;
        # sm_100a/sm_103a are arch-conditional and outrank it on those devices.
        "-gencode=arch=compute_100f,code=sm_100f",
        "-static-global-template-stub=false",
        "-DFLASHINFER_ENABLE_BF16",
        "-DFLASHINFER_ENABLE_FP8_E4M3",
        "-DFLASHINFER_ENABLE_FP8_E5M2",
        "-DFLASHINFER_ENABLE_FP8_E8M0",
        "-DFLASHINFER_ENABLE_FP4_E2M1",
        "-DFLASHINFER_ENABLE_F16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xcudafe --diag_suppress=2908",
        f"-I{fmha_include}",
        f"-I{cutlass_include}",
        f"-I{cutlass_util_include}",
        f"-I{tvm_include}",
        f"-I{cache_dir}",
        "-use_fast_math",
        "-DNDEBUG", "-Xptxas", "-O1" if fmha else "-O3",
        "-Xcompiler", "-fPIC",
    ]
    if os.environ.get("GPU_TRACE") is not None:
        nvcc_flags.append("-DGPU_TRACE_ENABLED")
    if os.environ.get("SM_TIMING") is not None:
        nvcc_flags.append("-DSM_TIMING_ENABLED")
    if os.environ.get("FMHA_GMEM_CHECK") is not None:
        nvcc_flags.append("-DFMHA_GMEM_BOUNDS_CHECK")
    proxy_chain, proxy_single_s = _proxy_modes()
    if proxy_chain:
        nvcc_flags.append("-DFMHA_ONLYSCORE_PER_TILE_CORRECTION")
    if proxy_single_s:
        nvcc_flags.append("-DFMHA_ONLYSCORE_SINGLE_S")
    return " ".join(nvcc_flags)

class _VariantWrapper:
    """Wraps a TVM-FFI function to look like a module with .run()."""
    def __init__(self, fn):
        self._fn = fn

    def run(self, *args):
        return self._fn(*args)


class FMHAVariantManager:
    """Manages FMHA variant kernels. Loads from all_variants.so or per-variant .so."""

    def __init__(self):
        self._loaded = {}
        self._lock = threading.Lock()
        self._all_module = None
        self._all_module_checked = False
        self._inst_template = None
        self._run_template = None

    def _load_templates(self):
        if self._inst_template is None:
            with open(_FMHA_VARLEN_DIR / "fmha_sm100_inst.jinja") as f:
                self._inst_template = jinja2.Template(f.read())
            with open(_FMHA_VARLEN_DIR / "fmha_sm100_variant_run.cu.jinja") as f:
                self._run_template = jinja2.Template(f.read())

    def _ensure_all_module(self):
        if not self._all_module_checked:
            self._all_module_checked = True
            all_variants_so = _all_variants_so()
            if all_variants_so.exists():
                import tvm_ffi
                self._all_module = tvm_ffi.load_module(str(all_variants_so))

    def get_variant(self, dtype_code, qo_tile_size, single_wg,
                    sparse_mode, page_size, split_kv, pack_factor):
        variant_name, params = _variant_key_from_runtime(
            dtype_code, qo_tile_size, single_wg,
            sparse_mode, page_size, split_kv, pack_factor)

        cached = self._loaded.get(variant_name)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._loaded.get(variant_name)
            if cached is not None:
                return cached

            fn_name = f"run_{variant_name}"

            prebuilt = _prebuilt(f"{variant_name}/{variant_name}.so")
            if prebuilt is not None:
                import tvm_ffi
                self._loaded[variant_name] = _VariantWrapper(getattr(tvm_ffi.load_module(str(prebuilt)), fn_name))
                return self._loaded[variant_name]

            # Try all_variants.so
            self._ensure_all_module()
            if self._all_module is not None:
                try:
                    fn = getattr(self._all_module, fn_name)
                    self._loaded[variant_name] = _VariantWrapper(fn)
                    return self._loaded[variant_name]
                except AttributeError:
                    pass

            # Try per-variant .so
            cache_dir = CACHE_BASE / variant_name
            so_path = cache_dir / f"{variant_name}.so"
            lock_fd = _acquire_file_lock(cache_dir / ".compile.lock")
            try:
                if not so_path.exists():
                    self._compile_only(variant_name, params)

                import tvm_ffi
                module = tvm_ffi.load_module(str(so_path))
                fn = getattr(module, fn_name)
            finally:
                _release_file_lock(lock_fd)
            self._loaded[variant_name] = _VariantWrapper(fn)
            return self._loaded[variant_name]

    def _compile_only(self, variant_name, params):
        """Compile a variant without loading. Safe to call from subprocesses."""
        cache_dir = CACHE_BASE / variant_name
        so_path = cache_dir / f"{variant_name}.so"

        if so_path.exists():
            return

        _require_toolchain(variant_name)
        logger.info(f"JIT compiling FMHA variant: {variant_name}")
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_templates()

        inst_cu = cache_dir / f"fmha_sm100_inst_{variant_name}.cu"
        run_cu = cache_dir / f"fmha_sm100_run_{variant_name}.cu"

        inst_cu.write_text(self._inst_template.render(**params))
        run_cu.write_text(self._run_template.render(**params))

        for name in ["fmha_sm100_params.h", "tvm_ffi_utils.h", "gmem_bounds_check.h"]:
            src = _FMHA_VARLEN_DIR / name
            dst = cache_dir / name
            if not dst.exists() or dst.read_text() != src.read_text():
                shutil.copy2(src, dst)

        self._write_ninja(cache_dir, variant_name, inst_cu, run_cu)
        self._run_ninja(cache_dir)

    def _write_ninja(self, cache_dir, variant_name, inst_cu, run_cu):
        cuda_home = _get_cuda_home()
        nvcc = os.path.join(cuda_home, "bin", "nvcc")

        nvcc_flags = _get_nvcc_flags(cache_dir)

        so_path = cache_dir / f"{variant_name}.so"
        inst_obj = cache_dir / f"inst_{variant_name}.o"
        run_obj = cache_dir / f"run_{variant_name}.o"

        ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {inst_obj}: nvcc_compile {inst_cu}
build {run_obj}: nvcc_compile {run_cu}
build {so_path}: nvcc_link {inst_obj} {run_obj}
"""
        (cache_dir / "build.ninja").write_text(ninja_content)

    def _run_ninja(self, cache_dir):
        result = subprocess.run(
            ["ninja", "-j1"],
            cwd=str(cache_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"FMHA variant compilation failed:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )


_variant_manager = FMHAVariantManager()


def get_fmha_variant(dtype_code, qo_tile_size, single_wg,
                     sparse_mode, page_size, split_kv, pack_factor):
    """Get a compiled FMHA variant module. Thread-safe, lazy compilation."""
    return _variant_manager.get_variant(
        dtype_code, qo_tile_size, single_wg,
        sparse_mode, page_size, split_kv, pack_factor)


# ============================================================================
# Plan kernel JIT (also needs sm_100a, can't be statically compiled)
# ============================================================================

_plan_module = None
_plan_lock = threading.Lock()


def _do_compile_plan():
    """Generate and compile plan kernel. No TVM loading."""
    cache_dir = CACHE_BASE / "plan"
    so_path = cache_dir / "fmha_sm100_plan.so"

    if so_path.exists():
        return

    _require_toolchain("plan")
    logger.info("JIT compiling FMHA plan module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    plan_cu = _FMHA_VARLEN_DIR / "fmha_sm100_plan.cu"
    shutil.copy2(plan_cu, cache_dir / "fmha_sm100_plan.cu")

    cuda_home = _get_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    obj = cache_dir / "fmha_sm100_plan.o"

    nvcc_flags = _get_nvcc_flags(cache_dir, False)

    ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {obj}: nvcc_compile {cache_dir / "fmha_sm100_plan.cu"}
build {so_path}: nvcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FMHA plan compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _compile_plan_only():
    """Compile plan kernel without loading. Safe for subprocesses."""
    _do_compile_plan()


def _compile_plan_module():
    """Compile and load plan kernel."""
    lock_fd = _acquire_file_lock(CACHE_BASE / "plan.lock")
    try:
        _do_compile_plan()
        import tvm_ffi
        so_path = CACHE_BASE / "plan" / "fmha_sm100_plan.so"
        return tvm_ffi.load_module(str(so_path))
    finally:
        _release_file_lock(lock_fd)


def get_plan_fn():
    """Get the plan function. JIT compiles on first call."""
    global _plan_module
    if _plan_module is not None:
        return _plan_module

    with _plan_lock:
        if _plan_module is not None:
            return _plan_module
        prebuilt = _prebuilt("plan/fmha_sm100_plan.so")
        if prebuilt is not None:
            import tvm_ffi
            _plan_module = tvm_ffi.load_module(str(prebuilt))
            return _plan_module
        _plan_module = _compile_plan_module()
        return _plan_module


# ============================================================================
# Sparse TopK Select kernel JIT
# ============================================================================

_sparse_topk_module = None
_sparse_topk_lock = threading.Lock()


def _do_compile_sparse_topk():
    cache_dir = CACHE_BASE / "sparse_topk"
    so_path = cache_dir / "sparse_topk_select.so"

    if so_path.exists():
        return

    _require_toolchain("sparse_topk")
    logger.info("JIT compiling sparse_topk_select module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_cu = _FMHA_VARLEN_DIR / "sparse_topk_select.cu"
    shutil.copy2(src_cu, cache_dir / "sparse_topk_select.cu")

    for name in ["tvm_ffi_utils.h"]:
        src = _FMHA_VARLEN_DIR / name
        dst = cache_dir / name
        if not dst.exists() or dst.read_text() != src.read_text():
            shutil.copy2(src, dst)

    cuda_home = _get_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    obj = cache_dir / "sparse_topk_select.o"

    nvcc_flags = _get_nvcc_flags(cache_dir, False)

    ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {obj}: nvcc_compile {cache_dir / "sparse_topk_select.cu"}
build {so_path}: nvcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sparse_topk_select compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def get_sparse_topk_module():
    """Get the sparse_topk_select module. JIT compiles on first call."""
    global _sparse_topk_module
    if _sparse_topk_module is not None:
        return _sparse_topk_module

    with _sparse_topk_lock:
        if _sparse_topk_module is not None:
            return _sparse_topk_module
        prebuilt = _prebuilt("sparse_topk/sparse_topk_select.so")
        if prebuilt is not None:
            import tvm_ffi
            _sparse_topk_module = tvm_ffi.load_module(str(prebuilt))
            return _sparse_topk_module
        lock_fd = _acquire_file_lock(CACHE_BASE / "sparse_topk.lock")
        try:
            _do_compile_sparse_topk()
            import tvm_ffi
            so_path = CACHE_BASE / "sparse_topk" / "sparse_topk_select.so"
            _sparse_topk_module = tvm_ffi.load_module(str(so_path))
        finally:
            _release_file_lock(lock_fd)
        return _sparse_topk_module


# ============================================================================
# Split-KV reduction kernel JIT
# ============================================================================

_reduction_module = None
_reduction_lock = threading.Lock()


def _do_compile_reduction():
    cache_dir = CACHE_BASE / "reduction"
    so_path = cache_dir / "fmha_sm100_reduction.so"

    if so_path.exists():
        return

    _require_toolchain("reduction")
    logger.info("JIT compiling fmha_sm100_reduction module")
    cache_dir.mkdir(parents=True, exist_ok=True)

    src_cu = _FMHA_VARLEN_DIR / "fmha_sm100_reduction.cu"
    shutil.copy2(src_cu, cache_dir / "fmha_sm100_reduction.cu")
    for name in ["gmem_bounds_check.h"]:
        src = _FMHA_VARLEN_DIR / name
        dst = cache_dir / name
        if src.exists() and (not dst.exists() or dst.read_text() != src.read_text()):
            shutil.copy2(src, dst)

    cuda_home = _get_cuda_home()
    nvcc = os.path.join(cuda_home, "bin", "nvcc")

    obj = cache_dir / "fmha_sm100_reduction.o"

    nvcc_flags = _get_nvcc_flags(cache_dir, False)

    ninja_content = f"""ninja_required_version = 1.5

nvcc = {nvcc}
nvcc_flags = {nvcc_flags}

rule nvcc_compile
  command = $nvcc $nvcc_flags -c $in -o $out
  description = Compiling $in

rule nvcc_link
  command = $nvcc -shared $in -o $out -lcuda
  description = Linking $out

build {obj}: nvcc_compile {cache_dir / "fmha_sm100_reduction.cu"}
build {so_path}: nvcc_link {obj}
"""
    (cache_dir / "build.ninja").write_text(ninja_content)

    result = subprocess.run(
        ["ninja", "-j1"],
        cwd=str(cache_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fmha_sm100_reduction compilation failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def get_reduction_module():
    """Get the split-KV reduction module. JIT compiles on first call."""
    global _reduction_module
    if _reduction_module is not None:
        return _reduction_module

    with _reduction_lock:
        if _reduction_module is not None:
            return _reduction_module
        prebuilt = _prebuilt("reduction/fmha_sm100_reduction.so")
        if prebuilt is not None:
            import tvm_ffi
            _reduction_module = tvm_ffi.load_module(str(prebuilt))
            return _reduction_module
        lock_fd = _acquire_file_lock(CACHE_BASE / "reduction.lock")
        try:
            _do_compile_reduction()
            import tvm_ffi
            so_path = CACHE_BASE / "reduction" / "fmha_sm100_reduction.so"
            _reduction_module = tvm_ffi.load_module(str(so_path))
        finally:
            _release_file_lock(lock_fd)
        return _reduction_module
