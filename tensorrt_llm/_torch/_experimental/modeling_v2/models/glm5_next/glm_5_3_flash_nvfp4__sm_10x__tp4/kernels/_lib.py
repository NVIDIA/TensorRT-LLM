# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builds and loads the kernel module (the CUDA sources under ``csrc/``) at first use.

The sources compile into one pybind11 module (the step program and its launchers) with torch's
extension builder, for the arch-specific target of the device's compute capability (sm_100a on
B200, sm_103a on B300 / GB300: tcgen05 and tensor memory are arch-specific features, so one
binary does not run on both), against the CUTLASS headers TensorRT-LLM ships for its other
JIT-compiled kernels (the ``fmha_sm100`` package), linked with cuBLASLt and NCCL. The build is
cached under ``$TRTLLM_MODELING_V2_CACHE`` (default ``~/.cache/tensorrt_llm/modeling_v2``) in a
directory keyed by the sources, the flags (the architecture among them) and the CUTLASS version,
so an unchanged tree loads in a second and a changed one never loads a stale build. The ranks
of one model share it: the builder locks the directory, one rank compiles and the others wait
for it. Nothing compiles at import: the first read of a launcher or a constant does.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from pathlib import Path

import torch
from torch.utils import cpp_extension

from tensorrt_llm.logger import logger

_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"
_FLASHMLA = _HERE / "third_party" / "flashmla" / "csrc"

_CXX_FLAGS = ["-O3", "-std=c++20"]
_CUDA_FLAGS = [
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-lineinfo",
    # the kernels use the half / bfloat16 operators and conversions torch's builder turns off
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]


def _arch() -> str:
    """The arch-specific target of the current device: tcgen05 and tensor memory exist on sm_100a /
    sm_103a, not on the sm_100 family target."""
    major, minor = torch.cuda.get_device_capability()
    cc = 10 * major + minor
    if cc not in (100, 103):
        raise RuntimeError(f"the GLM-5.3-Flash kernels run on sm_100 / sm_103, not on sm_{cc}")
    return f"{cc}a"


def _cuda_flags() -> list[str]:
    a = _arch()
    return _CUDA_FLAGS + [f"-gencode=arch=compute_{a},code=sm_{a}"]


def _cutlass_root() -> Path:
    spec = importlib.util.find_spec("fmha_sm100")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "the GLM-5.3-Flash kernels compile against the CUTLASS headers of TensorRT-LLM's "
            "fmha_sm100 package, which is not installed"
        )
    root = Path(next(iter(spec.submodule_search_locations))) / "cutlass"
    if not (root / "include" / "cutlass" / "version.h").is_file():
        raise ImportError(f"no CUTLASS headers under {root}")
    return root


def _cutlass_version(root: Path) -> str:
    text = (root / "include" / "cutlass" / "version.h").read_text()
    parts = [re.search(rf"#define CUTLASS_{k} (\d+)", text) for k in ("MAJOR", "MINOR", "PATCH")]
    return ".".join(p.group(1) if p else "?" for p in parts)


def _sources() -> list[Path]:
    return sorted(_CSRC.glob("*.cpp")) + sorted((_CSRC / "kernels").glob("*.cu"))


def _build_key(cutlass_version: str) -> str:
    digest = hashlib.sha256()
    tree = sorted(p for d in (_CSRC, _FLASHMLA) for p in d.rglob("*") if p.is_file())
    for path in tree + [Path(__file__)]:
        digest.update(str(path.relative_to(_HERE.parent)).encode() + b"\0")
        digest.update(path.read_bytes())
    digest.update(" ".join(_CXX_FLAGS + _cuda_flags()).encode())
    digest.update(cutlass_version.encode())
    digest.update(torch.__version__.encode())
    digest.update(str(torch.version.cuda).encode())
    return digest.hexdigest()[:16]


def _load():
    cutlass = _cutlass_root()
    version = _cutlass_version(cutlass)
    key = _build_key(version)
    cache = Path(
        os.environ.get("TRTLLM_MODELING_V2_CACHE", "~/.cache/tensorrt_llm/modeling_v2")
    ).expanduser()
    build_dir = cache / "glm53" / key
    build_dir.mkdir(parents=True, exist_ok=True)
    name = f"glm53_kernels_{key}"
    if not any(build_dir.glob(f"{name}*.so")):
        logger.info(
            f"GLM-5.3-Flash kernels: compiling {len(_sources())} sources for sm_{_arch()} against "
            f"CUTLASS {version} into {build_dir} (first use of this tree; a few minutes)"
        )
    cuda_home = Path(cpp_extension.CUDA_HOME or "/usr/local/cuda")
    return cpp_extension.load(
        name=name,
        sources=[str(p) for p in _sources()],
        extra_include_paths=[
            str(_CSRC),
            str(cutlass / "include"),
            str(cutlass / "tools" / "util" / "include"),
            str(_FLASHMLA),
            str(_FLASHMLA / "kerutils" / "include"),
        ],
        extra_cflags=_CXX_FLAGS,
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=[
            f"-L{cuda_home / 'lib64' / 'stubs'}",
            "-lcublasLt",
            "-lcublas",
            "-lcuda",
            "-lnccl",
        ],
        build_directory=str(build_dir),
        verbose=os.environ.get("TRTLLM_MODELING_V2_JIT_VERBOSE", "0") == "1",
    )


class _Ops:
    """The kernel module's classes, launchers and constants, compiled and loaded at the first read
    (importing the target, as routing does, compiles nothing)."""

    _module = None

    def __getattr__(self, name: str):
        if _Ops._module is None:
            _Ops._module = _load().ops
        return getattr(_Ops._module, name)


ops = _Ops()


def stream() -> int:
    """The current CUDA stream (the one a captured forward records into)."""
    return torch.cuda.current_stream().cuda_stream


def ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()
