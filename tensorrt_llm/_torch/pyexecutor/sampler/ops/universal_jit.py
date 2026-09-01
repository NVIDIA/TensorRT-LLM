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
"""Development-only JIT build of the universal sampling op.

Kernel work iterates far faster than a wheel rebuild, so during development the same
sources the CMake build compiles are also compilable by ``torch.utils.cpp_extension``:
ninja rebuilds only what changed, and the cache makes an unchanged tree a no-op.

Two rules keep this from becoming a second implementation:

* **The sources are the build's own.** ``cpp/tensorrt_llm/kernels/universalSamplingKernels.cu``
  and ``cpp/tensorrt_llm/thop/universalSamplingOp.cpp`` -- not copies. A JIT-only variant
  of the kernel would mean the thing being tested is not the thing that ships.
* **The registration is the build's own.** Both paths run the same
  ``TORCH_LIBRARY_FRAGMENT``, so callers above :mod:`~...ops.universal` cannot tell which
  produced the ops, and moving the sources into CMake changes no caller.

Off unless ``TLLM_UNIVERSAL_SAMPLING_JIT=1``; a release wheel therefore never reaches
this module.
"""

import hashlib
import os
import shlex
from functools import lru_cache
from pathlib import Path

from tensorrt_llm.logger import logger

ENV_FLAG = "TLLM_UNIVERSAL_SAMPLING_JIT"

_SOURCES = (
    "cpp/tensorrt_llm/thop/universalSamplingOp.cpp",
    "cpp/tensorrt_llm/kernels/universalSamplingKernels.cu",
)


def jit_requested() -> bool:
    return os.environ.get(ENV_FLAG, "0") == "1"


def _repo_root() -> Path:
    """The checkout this package was installed from, editable-install only.

    ``.../tensorrt_llm/_torch/pyexecutor/sampler/ops/universal_jit.py`` -> five parents up
    is ``tensorrt_llm/``, six is the checkout. A wheel install has no ``cpp/`` there,
    which is the correct answer for a wheel: it should carry the compiled op instead.
    """
    return Path(__file__).resolve().parents[5]


@lru_cache(maxsize=1)
def load() -> bool:
    """Compile and register the ops. Returns whether they are now available.

    Cached, so repeated calls after a successful build cost nothing and a failed build is
    not retried once per sampling step.
    """
    root = _repo_root()
    sources = [root / s for s in _SOURCES]
    missing = [str(s) for s in sources if not s.is_file()]
    if missing:
        logger.warning(
            f"{ENV_FLAG}=1 but the kernel sources are not in this install: {missing}. "
            "A JIT build needs an editable install of a checkout that has cpp/."
        )
        return False

    from torch.utils.cpp_extension import load as load_extension

    # Extra nvcc flags for development builds -- e.g. -DTLLM_USAMP_STAGE_TIMING, which
    # turns on the kernel's per-stage clock. Folded into the build directory name because
    # ninja keys its cache on the directory, not on the flags: reusing the same directory
    # with different flags silently returns the previously built object.
    extra_cuda = shlex.split(os.environ.get("TLLM_UNIVERSAL_SAMPLING_JIT_CUDA_FLAGS", ""))
    default_dir = Path.home() / ".cache" / "tensorrt_llm" / "universal_jit"
    if extra_cuda:
        digest = hashlib.sha1(" ".join(extra_cuda).encode()).hexdigest()[:10]
        default_dir = default_dir.with_name(f"{default_dir.name}_{digest}")
    build_dir = Path(os.environ.get("TLLM_UNIVERSAL_SAMPLING_JIT_DIR", default_dir))
    build_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"building the universal sampling op with ninja into {build_dir} (first build takes a minute)"
    )
    load_extension(
        name="tensorrt_llm_universal_sampling_jit",
        sources=[str(s) for s in sources],
        # The thop file includes "tensorrt_llm/kernels/..." exactly as it does under
        # CMake, so cpp/ is the include root in both builds.
        extra_include_paths=[str(root / "cpp")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr", *extra_cuda],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=False,
    )
    return True
