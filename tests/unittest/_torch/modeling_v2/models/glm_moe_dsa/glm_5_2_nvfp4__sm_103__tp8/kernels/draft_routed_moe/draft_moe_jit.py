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
"""Load ``trtllm::glm_moe_dsa_draft_routed_moe`` for tests.

A TensorRT-LLM build of this branch registers the op in ``th_common``. On an
older image (development overlay) it is compiled from this checkout's C++
sources with ``torch.utils.cpp_extension``, unmodified and with the same op
schema; results obtained that way are labelled DEV_JIT.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable, Sequence

import torch

_REPO = Path(__file__).resolve().parents[9]
_SOURCES = (
    _REPO / "cpp/tensorrt_llm/thop/glmMoeDsa/draftRoutedMoeOp.cpp",
    _REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/draftRoutedMoe.cu",
)
_HEADERS = (_REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/draftRoutedMoe.h",)
OP_NAME = "glm_moe_dsa_draft_routed_moe"
_NAME = "draft_moe_jit"


def _digest() -> str:
    h = hashlib.sha256()
    for path in (*_SOURCES, *_HEADERS):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def ensure_op(
    build_root: str,
    is_builder: bool = True,
    barrier: Callable[[], None] = lambda: None,
    defines: Sequence[str] = (),
) -> str:
    """Make the op available in this process; returns where it came from.

    ``is_builder`` must be true on exactly one process per shared build root;
    ``barrier`` synchronizes the processes after that one has built it.
    ``defines`` (e.g. ``("GLM_MOE_DSA_DRAFT_ROUTED_MOE_TIMELINE=1",)``) builds a probe variant of
    the op; it needs a process in which the op is not registered yet.
    """
    import tensorrt_llm  # noqa: F401  (loads th_common and the trtllm namespace)

    if hasattr(torch.ops.trtllm, OP_NAME):
        if defines:
            raise RuntimeError(
                f"{OP_NAME} is already registered; a probe build needs a fresh process"
            )
        barrier()
        return "builtin"
    tag = _digest()
    name = _NAME
    if defines:
        tag += "-" + hashlib.sha256(",".join(defines).encode()).hexdigest()[:8]
        name += "_probe"
    build_dir = Path(build_root) / f"draft_routed_moe-{tag}"
    library = build_dir / f"{name}.so"
    if is_builder and not library.exists():
        from torch.utils.cpp_extension import load

        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.3"
        build_dir.mkdir(parents=True, exist_ok=True)
        extra = [f"-D{d}" for d in defines]
        load(
            name=name,
            sources=[str(s) for s in _SOURCES],
            extra_include_paths=[str(_REPO / "cpp/include"), str(_REPO / "cpp")],
            extra_cflags=["-O3", "-std=c++17", *extra],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-Xptxas=-v",
                *extra,
            ],
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=True,
        )
    barrier()
    if not hasattr(torch.ops.trtllm, OP_NAME):
        torch.ops.load_library(str(library))
    return f"DEV_JIT {library}"
