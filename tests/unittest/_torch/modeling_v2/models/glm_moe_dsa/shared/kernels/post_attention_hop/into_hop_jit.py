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
"""Load ``trtllm::glm_moe_dsa_post_attention_hop_from_partials`` for tests.

A TensorRT-LLM build of this branch registers the op in ``th_common``. On an
older image (development overlay) the op is compiled from this checkout's
C++ sources with ``torch.utils.cpp_extension`` instead, unmodified and with
the same op schema (``-DTLLM_GLM_MOE_DSA_ATTENTION_OUTPUT_INTO_HOP_JIT`` only names the library).
"""

import hashlib
import os
from pathlib import Path
from typing import Callable

import torch

_REPO = Path(__file__).resolve().parents[9]
_SOURCES = (
    _REPO / "cpp/tensorrt_llm/thop/glmMoeDsa/postAttentionHopFromPartialsOp.cpp",
    _REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHopFromPartials.cu",
)
_HEADERS = (
    _REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHopFromPartials.h",
    _REPO / "cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHop.h",
)
_NAME = "into_hop_jit"
_OP = "glm_moe_dsa_post_attention_hop_from_partials"


def _digest() -> str:
    h = hashlib.sha256()
    for path in (*_SOURCES, *_HEADERS):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def ensure_op(build_root: str, is_builder: bool, barrier: Callable[[], None]) -> str:
    """Make the op available on every rank; returns where it came from.

    ``is_builder`` must be true on exactly one rank per shared build root;
    ``barrier`` synchronizes the ranks after that rank has built it.
    """
    import tensorrt_llm  # noqa: F401  (loads th_common and the trtllm namespace)

    if hasattr(torch.ops.trtllm, _OP):
        barrier()
        return "builtin"
    build_dir = Path(build_root) / f"attention_output_into_hop-{_digest()}"
    library = build_dir / f"{_NAME}.so"
    if is_builder and not library.exists():
        from torch.utils.cpp_extension import load

        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.3"
        build_dir.mkdir(parents=True, exist_ok=True)
        load(
            name=_NAME,
            sources=[str(s) for s in _SOURCES],
            extra_include_paths=[str(_REPO / "cpp/include"), str(_REPO / "cpp")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-Xptxas=-v"],
            build_directory=str(build_dir),
            is_python_module=False,
            verbose=True,
        )
    barrier()
    if not hasattr(torch.ops.trtllm, _OP):
        torch.ops.load_library(str(library))
    return str(library)
