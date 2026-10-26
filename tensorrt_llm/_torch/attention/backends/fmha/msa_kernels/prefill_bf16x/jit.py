# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Op-local JIT for the SM100 Q8KV4 sparse prefill attention kernel (E4M3 or BF16 data path)."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import sysconfig
import time
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import jinja2
from torch.utils import cpp_extension

# TensorRT-LLM: the module is built ahead of time (cpp/tensorrt_llm/kernels/minimaxM3Msa) and loads through the AOT path
# below; the run-time JIT helpers of the upstream package are not vendored, so a spec outside the prebuilt set logs
# "msa fmha JIT fallback variant=prefill_bf16x" and raises.
try:
    from .. import _jit_cache
    from ..decode_q8kv4._build_utils import cuda_home as _cuda_home
    from ..decode_q8kv4._build_utils import cutlass_root as _cutlass_root
    from ..decode_q8kv4._build_utils import cutlass_version as _cutlass_version
    from ..decode_q8kv4._build_utils import require_cuda_version
    from ..decode_q8kv4._build_utils import supports_qmul4
    from ..decode_q8kv4._build_utils import target_arch as _resolve_target_arch
except ImportError:
    _jit_cache = None

    def _no_jit(*args, **kwargs):
        logging.getLogger(__name__).warning("msa fmha JIT fallback variant=prefill_bf16x")
        raise RuntimeError(
            "prefill_bf16x: no prebuilt module for this spec, and TensorRT-LLM ships no run-time compile for it")

    _cuda_home = _cutlass_root = _cutlass_version = require_cuda_version = _no_jit
    supports_qmul4 = _resolve_target_arch = _no_jit

logger = logging.getLogger(__name__)

# Offline builds (build.py) select the target architecture with this variable.
FMHA_SM100_PREFILL_BF16X_ARCH = "FMHA_SM100_PREFILL_BF16X_ARCH"
# Set to 1 to compile the packed-FP16 dequant path even when the toolchain accepts QMUL4: the path
# SM107 takes, so the two can be compared on one GPU.
DISABLE_QMUL4_ENV = "FMHA_SM100_PREFILL_BF16X_DISABLE_QMUL4"
# E4M3X study: set to 1 to build the E4M3 data path with the SM100 K = 32 f8f6f4 MMA atoms on SM107 (the K the
# CuTe-DSL fp8 forward uses) instead of the SM107 K = 64 ones. Own uri suffix "_k32", so the two never share a cache.
E4M3_K32_ENV = "FMHA_SM100_PREFILL_BF16X_E4M3_K32"
_E4M3_K32 = os.environ.get(E4M3_K32_ENV, "0") == "1"
# E4M3X (default ON, m4f): the E4M3 data path built as the CuTe-DSL fp8 forward's exact transcription (uri suffix
# "_e4m3x"; bitwise vs the DSL fp8 forward, VR oracles 837726 / 837778). Set to 0 for the upstream Q8KV4 E4M3 kernel.
E4M3X_ENV = "FMHA_SM100_PREFILL_BF16X_E4M3X"
_E4M3X = os.environ.get(E4M3X_ENV, "1") == "1"
_SUPPORTED_ARCHES = ("100a", "103a", "107a")
_QMUL4_DEQUANT = "qmul4"
_FP16_DEQUANT = "fp16"
# Data paths (compile-time FMHA_SM100_PREFILL_BF16X_DTYPE): E4M3 Q / KV / P (the upstream Q8KV4
# kernel, define 0) or BF16 Q / KV / P (define 1). Each is its own extension.
E4M3_DATA_PATH = "e4m3"
BF16_DATA_PATH = "bf16"
DATA_PATHS = (E4M3_DATA_PATH, BF16_DATA_PATH)
_DTYPE_DEFINE = {E4M3_DATA_PATH: 0, BF16_DATA_PATH: 1}
# Q ring depth per data path: 16 KB vs 32 KB per stage inside the 227 KB SMEM budget (see
# csrc/include/sm100/common/prefill_attention_config.cuh).
_Q_STAGES = {E4M3_DATA_PATH: 8, BF16_DATA_PATH: 4}

_ROOT = Path(__file__).resolve().parent
_CSRC = _ROOT / "csrc"
_API = _CSRC / "api"
_INCLUDE = _CSRC / "include"
_TEMPLATES = _CSRC / "templates"
# Largest block-scale staging shift (an E4M3 exponent offset, detail::kBlockScaleShift); each
# shift is its own extension so caches whose products already fit pay nothing.
MAX_BLOCK_SCALE_SHIFT = 7


@lru_cache(maxsize=1)
def _cuda_version() -> tuple[int, int]:
    return require_cuda_version(
        (13, 4),
        component="BF16X prefill attention",
    )


@lru_cache(maxsize=None)
def _target_arch(device=None) -> str:
    arch = _resolve_target_arch(
        device,
        component="BF16X prefill attention",
        supported_arches=_SUPPORTED_ARCHES,
        env_var=FMHA_SM100_PREFILL_BF16X_ARCH,
    )
    if arch == "107a" and _cutlass_version() < (4, 8):
        raise RuntimeError(
            "BF16X prefill attention on SM107 requires CUTLASS 4.8 or newer; set "
            f"CUTLASS_ROOT (found {_cutlass_version()} at {_cutlass_root()})"
        )
    return arch


def _qmul4_disabled() -> bool:
    value = os.environ.get(DISABLE_QMUL4_ENV, "").strip().lower() or "0"
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{DISABLE_QMUL4_ENV} must be 0 or 1, got {value!r}")
    return value in ("1", "true")


@cache
def _dequant_mode(arch: str) -> str:
    """QMUL4 where the toolchain accepts it for ``arch``, the packed-FP16 path otherwise."""

    probe_root = _cache_root() / "capability_probes"
    if _qmul4_disabled() or not supports_qmul4(arch, probe_root=probe_root):
        return _FP16_DEQUANT
    return _QMUL4_DEQUANT


def _cache_root() -> Path:
    """Share the package JIT cache location; fall back to the torch extension dir."""
    explicit = os.environ.get("MINFER_FMHA_CACHE_DIR") or os.environ.get(
        "TORCH_EXTENSIONS_DIR"
    )
    root = (
        Path(explicit).expanduser()
        if explicit
        else Path.home() / ".cache/minfer/fmha_sm100"
    )
    return root / "prefill_bf16x"


def _write_text_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


@lru_cache(maxsize=1)
def _namespace() -> _jit_cache.Namespace:
    """The cache namespace of the selected toolchain (see ``fmha_sm100._jit_cache``)."""
    return _jit_cache.Namespace(_cache_root(), _cuda_home())


def _import_extension(module_name: str, library: Path):
    """Import a built extension the way ``cpp_extension.load`` does, without running ninja."""
    spec = importlib.util.spec_from_file_location(module_name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class JitSpec:
    """One compile-time configuration: target architecture, block-scale shift, dequant path,
    data path. Build specs through :func:`gen_jit_spec`, which sets ``q_stages`` for the data
    path."""

    target_arch: str
    block_scale_shift: int = 0
    dequant_mode: str = _QMUL4_DEQUANT
    variant_name: str = "prefill_attention_bf16x"
    q_heads_per_kv: int = 16
    head_dim: int = 128
    page_size: int = 128
    q_stages: int = 8
    score_stages: int = 2
    data_path: str = E4M3_DATA_PATH

    @property
    def uri(self) -> str:
        # The E4M3 uri is the upstream one; the BF16 data path gets its own uri and module.
        data_path = "" if self.data_path == E4M3_DATA_PATH else f"_{self.data_path}"
        k32 = "_k32" if (_E4M3_K32 and self.data_path == E4M3_DATA_PATH) else ""
        k32 += "_e4m3x" if (_E4M3X and self.data_path == E4M3_DATA_PATH) else ""
        return (
            f"{self.variant_name}{data_path}{k32}_shift{self.block_scale_shift}_{self.dequant_mode}_"
            f"{self.target_arch}"
        )

    @property
    def module_name(self) -> str:
        return f"_fmha_sm100_{self.uri}"

    def _template_params(self) -> dict:
        return {
            "q_heads_per_kv": self.q_heads_per_kv,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "q_stages": self.q_stages,
            "score_stages": self.score_stages,
        }

    def _load_arguments(self, cache_dir: Path) -> dict:
        shift_define = f"-DFMHA_SM100_PREFILL_BF16X_BLOCK_SCALE_SHIFT={self.block_scale_shift}"
        has_qmul4 = int(self.dequant_mode == _QMUL4_DEQUANT)
        # The data-path define is passed only for BF16 (the headers default it to 0), so the E4M3
        # build command, and with it its cache recipe, is the upstream one.
        dtype_defines = (
            []
            if self.data_path == E4M3_DATA_PATH
            else [f"-DFMHA_SM100_PREFILL_BF16X_DTYPE={_DTYPE_DEFINE[self.data_path]}"]
        )
        return {
            "sources": [
                str(_API / "prefill_attention_api.cpp"),
                str(_API / "prefill_attention_binding.cpp"),
                str(_API / "locality_domains.cu"),
                str(cache_dir / "prefill_attention_inst.cu"),
            ],
            "extra_include_paths": [
                str(path)
                for path in (
                    _API,
                    _INCLUDE,
                    _cutlass_root() / "include",
                    _cutlass_root() / "tools/util/include",
                )
            ],
            "extra_cflags": ["-O3", "-DNDEBUG", "-std=c++20", shift_define, *dtype_defines,
                             *(["-DFMHA_SM100_PREFILL_BF16X_E4M3X=1"] if (_E4M3X and self.data_path == E4M3_DATA_PATH) else [])],
            "extra_cuda_cflags": [
                "-O3",
                "-DNDEBUG",
                "-lineinfo",
                "-std=c++20",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-static-global-template-stub=false",
                "-Xptxas=-O3",
                # An explicit gencode keeps torch from adding its own arch list, which does not
                # know SM107.
                f"-gencode=arch=compute_{self.target_arch},code=sm_{self.target_arch}",
                shift_define,
                f"-DFMHA_SM100_PREFILL_BF16X_HAS_QMUL4={has_qmul4}",
                # Seen by the host and device passes alike (the cute arch macros are device-only),
                # so the kernel instantiation the host registers is the one the cubin holds.
                f"-DFMHA_SM100_PREFILL_BF16X_SM107={int(self.target_arch == '107a')}",
                # The kernel is launched with programmatic stream serialization (PDL); without
                # this define CUTLASS compiles griddepcontrol.wait / launch_dependents out.
                "-DCUTLASS_ENABLE_GDC_FOR_SM100",
                *dtype_defines,
                *(["-DFMHA_SM100_PREFILL_BF16X_E4M3_K32=1"] if (_E4M3_K32 and self.data_path == E4M3_DATA_PATH) else []),
                *(["-DFMHA_SM100_PREFILL_BF16X_E4M3X=1"] if (_E4M3X and self.data_path == E4M3_DATA_PATH) else []),
            ],
            "extra_ldflags": ["-lcuda", f"-Wl,-rpath,{_cuda_home() / 'lib64'}"],
        }

    def recipe(self) -> _jit_cache.Recipe:
        """The extension's cache recipe. The key holds the torch and Python builds the
        extension links against, so their headers are left out of the recorded inputs."""
        import torch

        arguments = self._load_arguments(Path("<build>"))
        return _jit_cache.Recipe(
            _namespace(),
            self.uri,
            {
                "load": arguments,
                "target_arch": self.target_arch,
                "params": self._template_params(),
                "torch": torch.__version__,
                "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                "python": sys.version,
            },
            templates=(_TEMPLATES / "prefill_attention_inst.cu.jinja",),
            covered=(
                os.path.realpath(Path(torch.__file__).parent) + os.sep,
                os.path.realpath(sysconfig.get_paths()["include"]) + os.sep,
            ),
        )

    def build_and_load(self):
        if not 0 <= self.block_scale_shift <= MAX_BLOCK_SCALE_SHIFT:
            raise ValueError(f"block_scale_shift must be in [0, {MAX_BLOCK_SCALE_SHIFT}]")
        _check_data_path(self.data_path, self.block_scale_shift)
        if self.q_stages != _Q_STAGES[self.data_path]:
            raise ValueError(
                f"q_stages must be {_Q_STAGES[self.data_path]} for the {self.data_path} data path"
            )
        recipe = self.recipe()
        library = recipe.lookup()
        if library is not None:
            return _import_extension(self.module_name, library)
        built = {}

        def builder(cache_dir: Path) -> Path:
            built["module"] = self._compile(cache_dir)
            return cache_dir / f"{self.module_name}.so"

        library = recipe.build(builder)
        # cpp_extension.load already imported what this process built; a library another
        # process published while this one waited for the lock is imported here.
        return built.get("module") or _import_extension(self.module_name, library)

    def _compile(self, cache_dir: Path):
        template = jinja2.Template(
            (_TEMPLATES / "prefill_attention_inst.cu.jinja").read_text(encoding="utf-8")
        )
        _write_text_if_changed(
            cache_dir / "prefill_attention_inst.cu", template.render(**self._template_params())
        )
        started_at = time.time()
        previous_cuda_home = cpp_extension.CUDA_HOME
        cpp_extension.CUDA_HOME = str(_cuda_home())
        try:
            extension = cpp_extension.load(
                name=self.module_name,
                **self._load_arguments(cache_dir),
                build_directory=str(cache_dir),
                verbose=False,
                with_cuda=True,
            )
        finally:
            cpp_extension.CUDA_HOME = previous_cuda_home
        logger.info(
            "Compiled fmha_sm100.prefill_bf16x in %.1fs",
            time.time() - started_at,
        )
        return extension


def _check_data_path(data_path: str, block_scale_shift: int) -> None:
    if data_path not in DATA_PATHS:
        raise ValueError(f"data_path must be one of {DATA_PATHS}, got {data_path!r}")
    if data_path == BF16_DATA_PATH and block_scale_shift != 0:
        raise ValueError("the bf16 data path has no block-scale staging (block_scale_shift 0)")


def _make_spec(arch: str, block_scale_shift: int, data_path: str) -> JitSpec:
    _check_data_path(data_path, block_scale_shift)
    return JitSpec(
        target_arch=arch,
        block_scale_shift=block_scale_shift,
        dequant_mode=_dequant_mode(arch),
        q_stages=_Q_STAGES[data_path],
        data_path=data_path,
    )


def gen_jit_spec(
    device=None, block_scale_shift: int = 0, data_path: str = E4M3_DATA_PATH
) -> JitSpec:
    """Return the compile-time configuration for a device, block-scale shift and data path."""

    return _make_spec(_target_arch(device), int(block_scale_shift), data_path)


@lru_cache(maxsize=None)
def _load_extension_for_arch(
    arch: str, block_scale_shift: int = 0, data_path: str = E4M3_DATA_PATH
):
    return _make_spec(arch, block_scale_shift, data_path).build_and_load()


# MiniMax-M3.1 CPP-E4M3X (MSA, PREREG_M31_MSA_CPP_E4M3X_v1): extensions prebuilt offline (build_aot) load from
# TRTLLM_M31_E4M3X_AOT_DIR (default: the package's aot/ dir) before the JIT's toolchain checks, so serving needs
# neither nvcc nor CUTLASS_ROOT. A spec without a prebuilt library takes the JIT. Set TRTLLM_M31_DISABLE_E4M3X_AOT=1
# for the JIT path everywhere.
E4M3X_AOT_DIR_ENV = "TRTLLM_M31_E4M3X_AOT_DIR"
DISABLE_E4M3X_AOT_ENV = "TRTLLM_M31_DISABLE_E4M3X_AOT"


def _aot_dir() -> Path:
    # TensorRT-LLM: tensorrt_llm/libs/msa_fmha_sm100/prefill_bf16x (this file is tensorrt_llm/_torch/attention/backends/
    # fmha/msa_kernels/prefill_bf16x/jit.py).
    default = Path(os.path.abspath(__file__)).parents[6] / "libs" / "msa_fmha_sm100" / "prefill_bf16x"
    return Path(os.environ.get(E4M3X_AOT_DIR_ENV) or default)


def prebuilt_available(device=None, block_scale_shift: int = 0, data_path: str = E4M3_DATA_PATH) -> bool:
    """Whether the prebuilt module for this device / shift / data path is installed (TensorRT-LLM)."""
    prefix, suffix = aot_library_name(_device_arch(device), int(block_scale_shift), data_path)
    root = _aot_dir()
    return root.is_dir() and len(sorted(root.glob(f"{prefix}*{suffix}"))) == 1


def _device_arch(device) -> str:
    """The arch suffix of the device (``107a``) without the toolchain checks of ``_target_arch``."""
    explicit = os.environ.get(FMHA_SM100_PREFILL_BF16X_ARCH)
    if explicit:
        return explicit
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    return f"{major}{minor}a"


def aot_library_name(arch: str, block_scale_shift: int, data_path: str) -> tuple[str, str]:
    """The (module-name prefix, suffix) of a prebuilt library; the dequant mode in between is the build's."""
    marker = "@"
    spec = JitSpec(target_arch=arch, block_scale_shift=block_scale_shift, dequant_mode=marker,
                   q_stages=_Q_STAGES[data_path], data_path=data_path)
    prefix, suffix = spec.module_name.split(marker)
    return prefix, f"{suffix}.so"


@lru_cache(maxsize=None)
def _load_aot_extension(arch: str, block_scale_shift: int, data_path: str):
    prefix, suffix = aot_library_name(arch, block_scale_shift, data_path)
    root = _aot_dir()
    found = sorted(root.glob(f"{prefix}*{suffix}")) if root.is_dir() else []
    if len(found) != 1:
        return None
    import hashlib

    library = found[0]
    digest = hashlib.md5(library.read_bytes()).hexdigest()[:8]
    module = _import_extension(library.name[: -len(".so")], library)
    print(f"E4M3X_AOT loaded {library.name[len('_fmha_sm100_'):-len('.so')]} {digest}", file=sys.stderr, flush=True)
    return module


@cache
def _aot_disabled() -> bool:
    disabled = os.environ.get(DISABLE_E4M3X_AOT_ENV, "0") == "1"
    if disabled:
        print("E4M3X_AOT disabled", file=sys.stderr, flush=True)
    return disabled


def load_extension(device=None, block_scale_shift: int = 0, data_path: str = E4M3_DATA_PATH):
    """Build and load the extension for the tensor architecture, block-scale shift and data path
    (``"e4m3"``: float8_e4m3fn q; ``"bf16"``: bfloat16 q). A prebuilt library is taken first."""

    block_scale_shift = int(block_scale_shift)
    if not _aot_disabled():
        module = _load_aot_extension(_device_arch(device), block_scale_shift, data_path)
        if module is not None:
            return module
    return _load_extension_for_arch(_target_arch(device), block_scale_shift, data_path)


__all__ = [
    "BF16_DATA_PATH",
    "DATA_PATHS",
    "DISABLE_QMUL4_ENV",
    "E4M3_DATA_PATH",
    "DISABLE_E4M3X_AOT_ENV",
    "E4M3X_AOT_DIR_ENV",
    "FMHA_SM100_PREFILL_BF16X_ARCH",
    "MAX_BLOCK_SCALE_SHIFT",
    "JitSpec",
    "gen_jit_spec",
    "load_extension",
]
